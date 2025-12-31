"""
inference.py - Evaluation and video recording for CarRacing-v3 SAC agent
EE569 Deep Learning Assignment
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import gymnasium as gym
from collections import deque
from gymnasium.wrappers import RecordVideo
import json

# ==========================================
# MODEL ARCHITECTURE (Must match train.py)
# ==========================================

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
        
        # Adaptive pooling to 2x2 for consistent feature size
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

    def get_action(self, state, deterministic=True):
        """Get action for inference."""
        with torch.no_grad():
            mean, _ = self.forward(state)
            action = torch.tanh(mean) * self.action_limit
        return action.cpu().numpy()[0]


# ==========================================
# ENV WRAPPERS & PREPROCESSING
# ==========================================

class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=2):
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


def preprocess_state(state):
    # Convert to grayscale and resize to 84x84
    # State shape is (96, 96, 3)
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    # Return uint8 (0-255) to save 4x-8x RAM
    return resized


class StackFrames(gym.Wrapper):
    def __init__(self, env, stack_size=4):
        super().__init__(env)
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(stack_size, 84, 84),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        processed_obs = preprocess_state(observation)
        for _ in range(self.stack_size):
            self.frames.append(processed_obs)
        return np.stack(self.frames, axis=0), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = preprocess_state(observation)
        self.frames.append(processed_obs)
        return np.stack(self.frames, axis=0), reward, terminated, truncated, info


def make_env(render_mode=None, skip_frames=4):
    env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)
    env = FrameSkip(env, skip=skip_frames)
    env = StackFrames(env, stack_size=4)
    return env


# ==========================================
# UTILS
# ==========================================

def load_model(checkpoint_path, state_dim, action_dim, hidden_dim, action_limit, device):
    """Load Actor model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = ActorNetwork(state_dim, action_dim, hidden_dim, action_limit).to(device)

    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict):
            if 'actor_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['actor_state_dict'])
                print(f"✅ Loaded from full checkpoint (episode {checkpoint.get('episode', 'N/A')})")
                if 'best_reward' in checkpoint:
                    print(f"   Best reward: {checkpoint['best_reward']:.1f}")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print(f"✅ Loaded from state dict wrapper")
            else:
                model.load_state_dict(checkpoint)
                print(f"✅ Loaded from state dict directly")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded from state dict directly")

    except Exception as e:
        print(f"⚠️  Error loading: {e}")
        print("Trying to load with strict=False...")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False), strict=False)
            print(f"✅ Loaded with strict=False (some layers may be missing)")
        except Exception as e2:
            print(f"❌ Failed to load: {e2}")
            raise e2

    model.eval()
    return model


def run_episode(env, model, device, max_steps=1500, render=False, record_video=False):
    """Run one episode."""
    state, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done and steps < max_steps:
        # Normalize state on the fly as in train.py
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0

        action = model.get_action(state_tensor, deterministic=True)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        state = next_state
        steps += 1

        if render and not record_video:
            env.render()

    return total_reward, steps


def main():
    parser = argparse.ArgumentParser(description='Evaluate SAC agent on CarRacing-v3')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint (default: checkpoints/best_model.pth)')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of evaluation episodes')
    parser.add_argument('--save-video', action='store_true', default=True,
                        help='Save video of the evaluation episodes')
    parser.add_argument('--video-dir', type=str, default='./videos',
                        help='Directory to save videos')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'], help='Device to use')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Hidden size of the actor (must match training, default: 256)')
    
    args = parser.parse_args()

    # Device Setup
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("CAR RACING - SAC EVALUATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Evaluation episodes: {args.episodes}")
    print(f"Save video: {args.save_video}")
    print(f"Hidden Dim: {args.hidden_size}")
    print("=" * 60)

    # Temp env for dims
    temp_env = make_env()
    state_dim = (4, 84, 84) # temp_env.observation_space.shape
    action_dim = temp_env.action_space.shape[0]
    action_limit = temp_env.action_space.high[0]
    temp_env.close()

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # Load Model
    print(f"\n📂 Loading model...")
    try:
        model = load_model(args.checkpoint, state_dim, action_dim, args.hidden_size, action_limit, device)
    except Exception as e:
        print("CRITICAL ERROR: Could not load model. Please check checkpoint path and architecture.")
        return

    # Evaluation Loop
    print(f"\n🏁 Running evaluation ({args.episodes} episodes)...")

    rewards = []
    steps_list = []

    os.makedirs(args.video_dir, exist_ok=True)

    for episode in range(args.episodes):
        if args.save_video and episode == 0:
            video_env = make_env(render_mode='rgb_array')
            video_env = RecordVideo(
                video_env,
                args.video_dir,
                name_prefix='best_run_eval',
                episode_trigger=lambda x: True
            )
            
            print(f"Episode {episode + 1}/{args.episodes} (recording video)...", end="")
            reward, steps = run_episode(video_env, model, device, record_video=True)
            video_env.close()
            
            # Rename video for clarity if needed, or just keep best_run_eval
            # RecordVideo names files like name_prefix-episode-0.mp4
        else:
            env = make_env()
            print(f"Episode {episode + 1}/{args.episodes}...", end="")
            reward, steps = run_episode(env, model, device, render=False)
            env.close()

        rewards.append(reward)
        steps_list.append(steps)
        print(f" Reward: {reward:.1f}, Steps: {steps}")

    # Stats
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_steps = np.mean(steps_list)

    # Save Results
    results = {
        'checkpoint': args.checkpoint,
        'num_episodes': args.episodes,
        'rewards': rewards,
        'steps': steps_list,
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'mean_steps': float(mean_steps),
        'device': str(device),
        'passed': bool(mean_reward > 700)
    }

    results_file = os.path.join(args.video_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")
    print(f"Mean steps: {mean_steps:.0f}")
    if mean_reward > 700:
        print("✅ ASSIGNMENT PASSED (>700)")
    else:
        print("⚠️  ASSIGNMENT NOT PASSED (<700)")
    print(f"\n💾 Results saved to: {results_file}")
    if args.save_video:
        print(f"🎥 Check video in: {args.video_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
