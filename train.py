import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import cv2
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import kornia.augmentation as K
import sys
import warnings
import argparse

# Import optimized buffer
from buffer import ReplayBuffer

# Suppress warnings (e.g. from pygame/pkg_resources)
warnings.filterwarnings("ignore")

# Hyperparameters (Optimized for RTX 4050 - 6GB VRAM, 16GB RAM)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True" # Help with fragmentation
NUM_EPISODES = 5000                    # Longer training
MAX_STEPS_PER_EPISODE = 1500           # Full lap completion
BATCH_SIZE = 400                       # Safe default for 6GB VRAM
GAMMA = 0.99
TAU = 0.01                             # Soft target updates
ALPHA_INIT = 0.2
LEARNING_RATE = 3e-4                   # Standard SAC learning rate
MEMORY_SIZE = 300000                   # 300k steps ~ 8.5GB RAM (Optimized)
HIDDEN_SIZE = 256                      # Reduced to 256 to save VRAM and compute
INITIAL_EXPLORATION_STEPS = 20000      # Exploration before policy training
EXPLORATION_NOISE = 0.15               # Action noise for exploration
ACTION_REPEAT = 4                      # Frame skipping (4 for faster training)


# Path for checkpoints and logs
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
VIDEO_DIR = "./videos"

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm for SAC stability (not BatchNorm!)."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.gn1 = nn.GroupNorm(min(8, out_channels // 4), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(8, out_channels // 4), out_channels)
        
        # Skip connection with projection if dimensions change
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.GroupNorm(min(8, out_channels // 4), out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.relu(out + residual)


class ActorNetwork(nn.Module):
    """Actor with D2RL dense connections and residual blocks."""
    def __init__(self, state_dim, action_dim, hidden_dim, action_limit):
        super().__init__()
        
        # 3 residual blocks with GroupNorm
        self.res1 = ResidualBlock(4, 64, stride=2)    # 84 -> 42
        self.res2 = ResidualBlock(64, 128, stride=2)  # 42 -> 21
        self.res3 = ResidualBlock(128, 256, stride=2) # 21 -> 11
        
        # Adaptive pooling to 2x2 for consistent feature size (Reduced from 6x6)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Dense features: (64 + 128 + 256) * 4 = 1,792
        dense_size = (64 + 128 + 256) * 4
        
        # MLP with D2RL skip connection
        self.fc1 = nn.Linear(dense_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + dense_size, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.action_limit = action_limit

    def forward(self, state):
        # Feature extraction with dense collection
        f1 = self.res1(state)   # 64 x 42 x 42
        f2 = self.res2(f1)      # 128 x 21 x 21
        f3 = self.res3(f2)      # 256 x 11 x 11
        
        # Pool all features to same size for dense concatenation
        p1 = self.adaptive_pool(f1).flatten(1)  # 64 * 4 = 256
        p2 = self.adaptive_pool(f2).flatten(1)  # 128 * 4 = 512
        p3 = self.adaptive_pool(f3).flatten(1)  # 256 * 4 = 1024
        
        # D2RL dense concatenation
        features = torch.cat([p1, p2, p3], dim=1)  # 1792
        
        # MLP with skip connection
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(torch.cat([x, features], dim=1)))
        
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_limit
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_limit * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class CriticNetwork(nn.Module):
    """Twin Q-Critic with D2RL dense connections and residual blocks."""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        
        # Shared CNN backbone with residual blocks
        self.res1 = ResidualBlock(4, 64, stride=2)
        self.res2 = ResidualBlock(64, 128, stride=2)
        self.res3 = ResidualBlock(128, 256, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Dense features: (64 + 128 + 256) * 4 = 1,792
        dense_size = (64 + 128 + 256) * 4
        
        # Q1 architecture with D2RL skip
        self.linear1_q1 = nn.Linear(dense_size + action_dim, hidden_dim)
        self.linear2_q1 = nn.Linear(hidden_dim + dense_size + action_dim, hidden_dim)
        self.linear3_q1 = nn.Linear(hidden_dim, 1)

        # Q2 architecture with D2RL skip
        self.linear1_q2 = nn.Linear(dense_size + action_dim, hidden_dim)
        self.linear2_q2 = nn.Linear(hidden_dim + dense_size + action_dim, hidden_dim)
        self.linear3_q2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # Feature extraction with dense collection
        f1 = self.res1(state)
        f2 = self.res2(f1)
        f3 = self.res3(f2)
        
        p1 = self.adaptive_pool(f1).flatten(1)
        p2 = self.adaptive_pool(f2).flatten(1)
        p3 = self.adaptive_pool(f3).flatten(1)
        
        features = torch.cat([p1, p2, p3], dim=1)
        xu = torch.cat([features, action], dim=1)
        
        # Q1 forward with skip
        x1 = F.relu(self.linear1_q1(xu))
        x1 = F.relu(self.linear2_q1(torch.cat([x1, xu], dim=1)))
        x1 = self.linear3_q1(x1)

        # Q2 forward with skip
        x2 = F.relu(self.linear1_q2(xu))
        x2 = F.relu(self.linear2_q2(torch.cat([x2, xu], dim=1)))
        x2 = self.linear3_q2(x2)

        return x1, x2


# ReplayMemory removed in favor of buffer.py ReplayBuffer


class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, action_limit, device):
        self.device = device
        self.action_limit = action_limit
        self.alpha_init = ALPHA_INIT
        
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim, action_limit).to(device)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # OPTIMIZATION: Compile models if available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            print("🚀 Compiling models with torch.compile()...")
            self.actor = torch.compile(self.actor)
            self.critic = torch.compile(self.critic)
            # Target critic usually doesn't need compilation as it's not backpropped, but can be
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

        self.target_entropy = -float(action_dim)  # SAC standard: -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LEARNING_RATE)
        
        # Mixed precision scalers
        self.scaler_critic = GradScaler('cuda')
        self.scaler_actor = GradScaler('cuda')
        
        # Data augmentation: Random crop (biggest proven impact for SAC + pixels)
        self.aug = nn.Sequential(
            K.RandomCrop((84, 84), padding=4, padding_mode='replicate'),
        ).to(device)
        
        self.total_steps = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _safe_load_state(self, model, state_dict):
        """Safely load state dict, handling compiled models and prefix issues."""
        # ADDED DEBUG FOR USER ANALYSIS
        print(f"DEBUG: _safe_load_state called for {type(model)}")
        
        # 1. Sanitize state_dict keys (remove _orig_mod prefix if present)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                # print(f"DEBUG: Cleaning key {k} -> {k[10:]}")
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
                
        # Debug check to catch if keys still have prefixes
        keys = list(new_state_dict.keys())
        if keys:
            # print(f"DEBUG: Sample key in new_state_dict: {keys[0]}")
            if keys[0].startswith('_orig_mod'):
                 print("CRITICAL ERROR: Key still has prefix!")
        
        # 2. Load into the original uncompiled model to avoid key mismatches
        if hasattr(model, '_orig_mod'):
            print("DEBUG: Loading into _orig_mod (Compiled Model)")
            model._orig_mod.load_state_dict(new_state_dict)
        else:
            print("DEBUG: Loading directly into model (Uncompiled)")
            model.load_state_dict(new_state_dict)

    def _safe_get_state(self, model):
        """Safely get state dict, handling compiled models."""
        if hasattr(model, '_orig_mod'):
            return model._orig_mod.state_dict()
        return model.state_dict()

    def select_action(self, state, evaluate=False):
        # State is uint8 (0-255) from buffer or env, normalize on fly
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device).div(255.0)
        
        if evaluate:
            mean, _ = self.actor(state)
            return (torch.tanh(mean) * self.action_limit).cpu().data.numpy().flatten()
        else:
            # No grad needed for sampling action
            with torch.no_grad():
                action, _ = self.actor.sample(state)
            return action.cpu().data.numpy().flatten()

    def update_parameters(self, memory, batch_size):
        # Sample a batch from memory (optimized numpy buffer)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)

        # Convert to float and normalize here (Max RAM savings!)
        # Use torch.tensor or .to() for non_blocking support, as_tensor does not support it directly
        state_batch = torch.as_tensor(state_batch, device=self.device).float().div_(255.0)
        next_state_batch = torch.as_tensor(next_state_batch, device=self.device).float().div_(255.0)
        
        action_batch = torch.as_tensor(action_batch, device=self.device)
        reward_batch = torch.as_tensor(reward_batch, device=self.device)
        done_batch = torch.as_tensor(done_batch, device=self.device)
        
        # Apply data augmentation (random crop)
        # Optimization: Batch the augmentation to reduce kernel launches
        combined_states = torch.cat([state_batch, next_state_batch], dim=0)
        combined_states_aug = self.aug(combined_states)
        state_batch, next_state_batch = torch.split(combined_states_aug, batch_size, dim=0)
        
        self.total_steps += 1

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.target_critic(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - done_batch) * GAMMA * min_qf_next_target

        # Critic update with Mixed Precision
        with autocast('cuda'):
            current_q1, current_q2 = self.critic(state_batch, action_batch)
            qf1_loss = F.mse_loss(current_q1, next_q_value)
            qf2_loss = F.mse_loss(current_q2, next_q_value)
            critic_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        self.scaler_critic.scale(critic_loss).backward()
        self.scaler_critic.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.scaler_critic.step(self.critic_optimizer)
        self.scaler_critic.update()

        # Actor update with Mixed Precision
        with autocast('cuda'):
            pi, log_pi = self.actor.sample(state_batch)
            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        self.scaler_actor.scale(actor_loss).backward()
        self.scaler_actor.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.scaler_actor.step(self.actor_optimizer)
        self.scaler_actor.update()

        # Alpha update
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target network
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        # Return tensors to avoid CPU-GPU sync (wait for logging to call .item())
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'alpha': self.alpha
        }, None

    def save_checkpoint(self, filename, episode, best_reward, memory=None):
        checkpoint = {
            'episode': episode,
            'actor_state_dict': self._safe_get_state(self.actor),
            'critic_state_dict': self._safe_get_state(self.critic),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'best_reward': best_reward,
            'total_steps': self.total_steps,
            'rng_state_torch': torch.get_rng_state(),
            'rng_state_numpy': np.random.get_state(),
            'rng_state_random': random.getstate()
        }
        if torch.cuda.is_available():
            checkpoint['rng_state_cuda'] = torch.cuda.get_rng_state()
        
        # Save GradScaler states for proper resume
        checkpoint['scaler_critic_state'] = self.scaler_critic.state_dict()
        checkpoint['scaler_actor_state'] = self.scaler_actor.state_dict()
            
        # We don't save the buffer in the checkpoint dict anymore to keep it light
        # The buffer is saved separately if needed, or we rely on the separate save logic
        torch.save(checkpoint, filename)
        
        if memory is not None:
            # Save buffer to separate file (compressed)
            buffer_path = filename.replace('.pth', '_buffer.npz')
            # Check if this is the 'latest' checkpoint, we only really need to save buffer for latest
            # to avoid filling disk space
            if "latest" in filename or "best" not in filename: 
                 memory.save(buffer_path)

    def load_checkpoint(self, filename, memory=None):
        checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
        
        self._safe_load_state(self.actor, checkpoint['actor_state_dict'])
        self._safe_load_state(self.critic, checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.total_steps = checkpoint['total_steps']
        
        # Load RNG states (handle errors gracefully)
        try:
            if 'rng_state_torch' in checkpoint:
                print("   🎲 Loading RNG states...")
                torch.set_rng_state(checkpoint['rng_state_torch'].cpu().byte()) # Cast to ByteTensor
                np.random.set_state(checkpoint['rng_state_numpy'])
                random.setstate(checkpoint['rng_state_random'])
                if torch.cuda.is_available() and 'rng_state_cuda' in checkpoint:
                     # Cast CUDA rng state if possible, though usually it's ByteTensor already
                    torch.cuda.set_rng_state(checkpoint['rng_state_cuda'].cpu().byte())
        except Exception as e:
            print(f"⚠️  Could not load RNG states: {e}")
        
        # Load GradScaler states
        if 'scaler_critic_state' in checkpoint:
            self.scaler_critic.load_state_dict(checkpoint['scaler_critic_state'])
        if 'scaler_actor_state' in checkpoint:
            self.scaler_actor.load_state_dict(checkpoint['scaler_actor_state'])
        
        # Handle Buffer Loading (Legacy vs New)
        if memory is not None:
            buffer_path = filename.replace('.pth', '_buffer.npz')
            if os.path.exists(buffer_path):
                print(f"   📥 Loading optimized numpy buffer from {buffer_path}...")
                memory.load(buffer_path)
            elif 'replay_buffer' in checkpoint:
                print("   ⚠️  Found legacy list-based buffer in checkpoint. Migrating... (this may take a moment)")
                legacy_buffer = checkpoint['replay_buffer']
                # Migration logic: iterate and push
                # We assume the legacy buffer data format matches what push expects (s, a, r, ns, d)
                # Note: Legacy buffer stored (state, action, reward, next_state, done)
                for item in legacy_buffer:
                    if item is not None:
                        # Unpack
                        s, a, r, ns, d = item
                        memory.push(s, a, r, ns, d)
                print(f"   ✅ Migrated {len(legacy_buffer)} samples to new buffer.")
            else:
                print("   ⚠️  No replay buffer found.")
            
        return checkpoint['episode'], checkpoint['best_reward']
            
        return checkpoint['episode'], checkpoint['best_reward']


def preprocess_state(state):
    # Convert to grayscale and resize to 84x84
    # State shape is (96, 96, 3)
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    # Return uint8 (0-255) to save 4x-8x RAM
    return resized


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_state(state)
    if is_new_episode:
        stacked_frames = deque([frame] * 4, maxlen=4)
    else:
        stacked_frames.append(frame)
    return np.stack(stacked_frames, axis=0), stacked_frames


def set_seed(seed, env=None):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env:
        env.action_space.seed(seed)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train SAC agent on CarRacing-v3')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for training (default: {BATCH_SIZE})')
    parser.add_argument('--skip-buffer', action='store_true',
                        help='Skip loading replay buffer from checkpoint (use for low RAM)')
    parser.add_argument('--update-freq', type=int, default=8,
                        help='Gradient update frequency (default: 8)')
    args = parser.parse_args()
    
    # Use the command-line batch size (or default)
    batch_size = args.batch_size
    update_freq = args.update_freq
    
    print(f"📦 Batch size: {batch_size} | ⚡ Update Freq: {update_freq}")
    if args.skip_buffer:
        print(f"⚠️  Replay buffer loading disabled (--skip-buffer)")

    # Detect GPU and Enable Optimizations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        # CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # New 40-series/Ampere optimization
        try:
            torch.set_float32_matmul_precision('medium')
            print("✅ TensorFloat-32 (Medium) Enabled")
        except AttributeError:
            pass

        torch.cuda.empty_cache()
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available. Training will be slow.")

    print(f"📊 Architecture: ResidualBlocks + D2RL + GroupNorm")
    print(f"🎨 Data Augmentation: RandomCrop(84, padding=4)")
    print(f"🔄 Action Repeat: {ACTION_REPEAT}")
    print(f"🧠 Optimized RAM: storing uint8 frames")

    # Set Seed
    SEED = 42
    print(f"🌱 Setting random seed to {SEED}")
    set_seed(SEED)

    # Optimization: Disable render_mode for training env to save CPU cycles
    env = gym.make("CarRacing-v3", continuous=True, render_mode=None)
    
    action_limit = env.action_space.high[0]
    state_dim = (4, 84, 84)
    action_dim = env.action_space.shape[0]

    agent = SACAgent(state_dim, action_dim, HIDDEN_SIZE, action_limit, device)
    
    # Initialize Optimized Buffer
    memory = ReplayBuffer(MEMORY_SIZE, state_dim, action_dim, device)
    
    writer = SummaryWriter(LOG_DIR)
    
    # Auto-resume capability
    LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "latest.pth")
    start_episode = 0
    best_eval_reward = -float('inf')

    BEST_MODEL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    if os.path.exists(LATEST_CHECKPOINT):
        print("🔄 Found checkpoint. Resuming training...")
        try:
            # Pass None for memory if --skip-buffer is set to avoid loading the massive buffer
            memory_to_load = None if args.skip_buffer else memory
            start_episode, best_eval_reward = agent.load_checkpoint(LATEST_CHECKPOINT, memory_to_load)
            print(f"   Episode: {start_episode}")
            print(f"   Best reward from latest: {best_eval_reward:.1f}")
            if args.skip_buffer:
                print(f"   Replay buffer: SKIPPED (will start fresh)")
            # Ensure we start from the next episode
            start_episode += 1
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            print("🆕 Starting new training...")
    else:
        print("🆕 Starting new training...")

    # CRITICAL: Always check the best_model.pth to prevent accidental overwriting.
    # This handles the case where latest.pth was deleted but best_model.pth still holds the best score.
    if os.path.exists(BEST_MODEL_CHECKPOINT):
        try:
            print("🔍 Checking best_model.pth for record...")
            best_model_checkpoint_data = torch.load(BEST_MODEL_CHECKPOINT, map_location=device, weights_only=False)
            best_reward_in_file = best_model_checkpoint_data.get('best_reward', -float('inf'))
            if best_reward_in_file > best_eval_reward:
                print(f"   🏆 Found existing best record: {best_reward_in_file:.1f} (will only overwrite if beaten)")
                best_eval_reward = best_reward_in_file
            else:
                 print(f"   👍 Current session best ({best_eval_reward:.1f}) is already better or equal.")
        except Exception as e:
            print(f"⚠️  Could not read best_model.pth for reward comparison: {e}")

    # Global environment step counter
    env_steps = 0

    try:
        for episode in range(start_episode, NUM_EPISODES):
            start_time = time.time()
            state, _ = env.reset(seed=SEED + episode)
            state, stacked_frames = stack_frames(None, state, True)
            episode_reward = 0
            
            for step in range(MAX_STEPS_PER_EPISODE):
                env_steps += 1
                if env_steps < INITIAL_EXPLORATION_STEPS and start_episode == 0:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state)
                    # Add exploration noise
                    action = action + np.random.normal(0, EXPLORATION_NOISE, size=action_dim)
                    action = action.clip(env.action_space.low, env.action_space.high)

                # Action repeat (frame skipping) for sample efficiency
                total_reward = 0
                for _ in range(ACTION_REPEAT):
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                    if done:
                        break
                
                next_state, stacked_frames = stack_frames(stacked_frames, next_obs, False)
                
                memory.push(state, action, total_reward, next_state, done)
                state = next_state
                episode_reward += total_reward

                # Updated frequency: Every N environment steps
                if len(memory) > batch_size and step % update_freq == 0:
                    # Reduce update ratio to 0.5 updates per step to save GPU
                    update_info = None
                    for _ in range(max(1, update_freq // 2)): 
                        update_info, _ = agent.update_parameters(memory, batch_size)
                    
                    # Log only once per batch of updates to reduce CPU-GPU sync overhead
                    # Optimization: Moved logging OUTSIDE the loop
                    if update_info:
                        writer.add_scalar('Loss/critic', update_info['critic_loss'].item(), agent.total_steps)
                        writer.add_scalar('Loss/actor', update_info['actor_loss'].item(), agent.total_steps)
                        writer.add_scalar('Alpha', update_info['alpha'].item(), agent.total_steps)

                if done:
                    break

            writer.add_scalar('Reward/train', episode_reward, episode)
            
            # Calculate FPS and Stats
            duration = time.time() - start_time
            fps = step / duration if duration > 0 else 0
            
            # Enhanced Debug Output
            is_training = len(memory) > batch_size
            debug_info = [
                f"Episode {episode}/{NUM_EPISODES}",
                f"Reward: {episode_reward:.1f}",
                f"Steps: {step}",
                f"Buffer: {len(memory)}/{MEMORY_SIZE}",
            ]
            
            if is_training:
                debug_info.append(f"Alpha: {agent.alpha.item():.3f}")
                if update_info:
                    debug_info.append(f"ActL: {update_info['actor_loss'].item():.1f}")
                    debug_info.append(f"CriL: {update_info['critic_loss'].item():.1f}")
            
            debug_info.append(f"⏱️  {duration:.1f}s")
            debug_info.append(f"⚡ {fps:.1f} FPS")
            
            print(" | ".join(debug_info))
            
            # VRAM monitoring every episode
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"   📊 VRAM: {allocated:.2f}GB / {reserved:.2f}GB | RAM Buffer: ~{len(memory) * 28 / 1e6:.1f}MB")

            # Save latest checkpoint after each episode (Buffer saved periodically)
            agent.save_checkpoint(LATEST_CHECKPOINT, episode, best_eval_reward, memory=None)
            
            # Save buffer periodically (every 10 episodes)
            if episode % 10 == 0:
                memory.save(os.path.join(CHECKPOINT_DIR, "latest_buffer.npz"))

            # Evaluation and Best Checkpoint Saving (every 10 episodes)
            if episode % 10 == 0:
                avg_reward = 0
                eval_episodes = 3 # Small eval to save time
                
                # Create evaluation environment with video recording
                eval_env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
                eval_env = gym.wrappers.RecordVideo(
                    eval_env, 
                    video_folder=VIDEO_DIR, 
                    episode_trigger=lambda x: True, # Record all eval episodes
                    name_prefix=f"eval_ep_{episode}"
                )

                for _ in range(eval_episodes):
                    eval_state, _ = eval_env.reset()
                    eval_state, eval_frames = stack_frames(None, eval_state, True)
                    eval_reward = 0
                    while True:
                        eval_action = agent.select_action(eval_state, evaluate=True)
                        eval_next_state, r, term, trunc, _ = eval_env.step(eval_action)
                        eval_next_state, eval_frames = stack_frames(eval_frames, eval_next_state, False)
                        eval_reward += r
                        eval_state = eval_next_state
                        if term or trunc:
                            break
                    avg_reward += eval_reward
                
                eval_env.close() # Close to save video
                
                avg_reward /= eval_episodes
                writer.add_scalar('Reward/eval', avg_reward, episode)
                print(f"⭐ Evaluation: {avg_reward:.1f} (Video saved to {VIDEO_DIR})")
                
                if avg_reward > best_eval_reward:
                    best_eval_reward = avg_reward
                    agent.save_checkpoint(os.path.join(CHECKPOINT_DIR, "best_model.pth"), episode, best_eval_reward, memory)
                    print(f"🏆 New best model saved!")

    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted!")
        print("Saving checkpoint...")
        agent.save_checkpoint(LATEST_CHECKPOINT, episode, best_eval_reward, memory)
        print(f"✅ Saved: {LATEST_CHECKPOINT}")
        sys.exit(0)
    finally:
        env.close()
        writer.close()

if __name__ == "__main__":
    main()
