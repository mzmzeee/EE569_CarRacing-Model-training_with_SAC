# inference_sac.py
import argparse
import os
import torch
import numpy as np
import cv2
import gymnasium as gym
from collections import deque
from gymnasium.wrappers import RecordVideo

# Import SAC components from training script
import sys
sys.path.append('.')  # Ensure we can import from current directory

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# WRAPPERS (Must match training)
# ============================================================================
class CarRacingImageWrapper(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width), dtype=np.uint8
        )
        self.top_crop = 12
        self.bottom_crop = 96

    def observation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cropped = gray[self.top_crop:self.bottom_crop, :]
        resized = cv2.resize(cropped, (self.width, self.height),
                             interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(resized)


class StackFrames(gym.Wrapper):
    def __init__(self, env, stack_size=4):
        super().__init__(env)
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(stack_size, env.observation_space.shape[0],
                   env.observation_space.shape[1]),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        for _ in range(self.stack_size):
            self.frames.append(observation)
        return np.stack(self.frames, axis=0), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return np.stack(self.frames, axis=0), reward, terminated, truncated, info


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=3):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def make_env(render_mode=None, skip_frames=3):
    env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)
    env = FrameSkip(env, skip=skip_frames)
    env = CarRacingImageWrapper(env)
    env = StackFrames(env, stack_size=4)
    return env


# ============================================================================
# ACTOR NETWORK (Must match training)
# ============================================================================
class ActorNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=1536):
        super(ActorNetwork, self).__init__()
        self.log_std_min, self.log_std_max = -20, 2

        self.conv1 = torch.nn.Conv2d(state_dim[0], 96, kernel_size=5, stride=2)
        self.conv2 = torch.nn.Conv2d(96, 192, kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(192, 256, kernel_size=3, stride=1)

        self.bn1 = torch.nn.BatchNorm2d(96)
        self.bn2 = torch.nn.BatchNorm2d(192)
        self.bn3 = torch.nn.BatchNorm2d(256)

        convw = lambda s, k, st: (s - (k - 1) - 1) // st + 1
        w = convw(convw(convw(state_dim[2], 5, 2), 3, 2), 3, 1)
        h = convw(convw(convw(state_dim[1], 5, 2), 3, 2), 3, 1)
        linear_input_size = w * h * 256

        self.fc1 = torch.nn.Linear(linear_input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_residual = torch.nn.Linear(hidden_size, hidden_size)

        self.ln1 = torch.nn.LayerNorm(hidden_size)
        self.ln2 = torch.nn.LayerNorm(hidden_size)

        self.mean_layer = torch.nn.Linear(hidden_size, action_dim)
        self.log_std_layer = torch.nn.Linear(hidden_size, action_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.orthogonal_(module.weight, gain=0.01 if isinstance(module, torch.nn.Linear) else torch.sqrt(torch.tensor(2.0)))
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, state):
        x = torch.nn.functional.relu(self.bn1(self.conv1(state)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        x = torch.nn.functional.relu(self.ln1(self.fc1(x)))
        residual = x
        x = torch.nn.functional.relu(self.ln2(self.fc2(x)))
        x = x + self.fc_residual(residual)

        mean = torch.tanh(self.mean_layer(x)) * 1.5
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_action(self, state, deterministic=True):
        """Get action for inference (deterministic or stochastic)"""
        with torch.no_grad():
            mean, log_std = self.forward(state)
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                z = normal.sample()
                action = torch.tanh(z)
        return action.cpu().numpy()[0]


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================
def load_model(checkpoint_path, state_dim, action_dim, device):
    """Load SAC actor model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")

    model = ActorNetwork(state_dim, action_dim).to(device)
    
    # Try different loading methods
    try:
        # Try loading just the state dict
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'actor_state_dict' in checkpoint:
            # Full checkpoint
            model.load_state_dict(checkpoint['actor_state_dict'])
        else:
            # Actor-only checkpoint
            model.load_state_dict(checkpoint)
            
    except Exception as e:
        print(f"⚠️  Warning: {e}")
        print("Trying alternative loading method...")
        # If there's a size mismatch, try to load with strict=False
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    
    model.eval()
    print(f"✅ Loaded model from: {checkpoint_path}")
    return model


def run_episode(env, model, device, render=True, deterministic=True):
    """Run a single episode and return total reward"""
    state, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0
    
    while not done:
        # Prepare state tensor (add batch dimension, normalize)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
        
        # Get action from policy
        action = model.get_action(state_tensor, deterministic=deterministic)
        
        # Execute action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        state = next_state
        steps += 1
        
        if render and hasattr(env, 'render'):
            env.render()
        
        # Early stopping for testing
        if steps > 1500:  # Max steps per episode
            break
    
    return total_reward


def main():
    parser = argparse.ArgumentParser(
        description='Load and run trained SAC on CarRacing-v3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_sac.py --checkpoint checkpoints/best_model.pth --episodes 3
  python inference_sac.py --checkpoint checkpoints/final_model.pth --no-render --episodes 10
  python inference_sac.py --save-video --video-dir ./test_videos --stochastic
        """
    )
    
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint (default: checkpoints/best_model.pth)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run (default: 5)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering (faster)')
    parser.add_argument('--save-video', action='store_true',
                        help='Save video recordings')
    parser.add_argument('--video-dir', type=str, default='./videos',
                        help='Directory to save videos (default: ./videos)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (default: auto)')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic actions instead of deterministic')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')
    
    args = parser.parse_args()
    
    # Set device
    if args.device != 'auto':
        global device
        device = torch.device(args.device)
    print(f"🖥️  Using device: {device}")
    
    # Create environment
    render_mode = 'rgb_array' if args.save_video else ('human' if not args.no_render else None)
    env = make_env(render_mode=render_mode)
    
    # Wrap with video recorder if needed
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)
        env = RecordVideo(
            env, 
            args.video_dir, 
            name_prefix='sac_inference',
            episode_trigger=lambda x: True  # Record all episodes
        )
        print(f"🎬 Video recording enabled, saving to: {args.video_dir}")
    
    # Get environment specs
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape
    
    if args.verbose:
        print(f"\n📊 Environment Specifications:")
        print(f"  State shape: {state_dim}")
        print(f"  Action shape: {action_dim}")
        print(f"  Action space: {env.action_space}")
    
    # Load model
    print(f"\n📂 Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, state_dim, action_dim, device)
    
    # Run episodes
    print(f"\n🏁 Running {args.episodes} episode(s)...")
    print(f"  Mode: {'Stochastic' if args.stochastic else 'Deterministic'}")
    print()
    
    rewards = []
    
    try:
        for episode in range(args.episodes):
            print(f"Episode {episode + 1}/{args.episodes}...", end=" ")
            episode_reward = run_episode(
                env, model, device, 
                render=(not args.no_render and not args.save_video),
                deterministic=not args.stochastic
            )
            rewards.append(episode_reward)
            print(f"Reward: {episode_reward:.2f}")
    except KeyboardInterrupt:
        print("\n\n⛔ Interrupted by user")
    finally:
        env.close()
    
    # Print summary
    if rewards:
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Episodes completed: {len(rewards)}")
        print(f"Mean Reward:        {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Max Reward:         {np.max(rewards):.2f}")
        print(f"Min Reward:         {np.min(rewards):.2f}")
        print("=" * 60)
        
        # Performance evaluation
        mean_reward = np.mean(rewards)
        if mean_reward > 800:
            print("🎉 Excellent performance!")
        elif mean_reward > 600:
            print("👍 Good performance!")
        elif mean_reward > 400:
            print("⚠️  Average performance")
        else:
            print("❌ Needs improvement")
    
    if args.save_video:
        print(f"\n🎥 Videos saved in: {args.video_dir}")


if __name__ == '__main__':
    main()
