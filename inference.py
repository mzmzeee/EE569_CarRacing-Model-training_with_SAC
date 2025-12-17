# inference.py
import argparse
import os
import torch
import numpy as np
from gymnasium.wrappers import RecordVideo

# Import shared components from training script
from dqn_car_racing import (
    device,
    DQN,
    CarRacingActionWrapper,
    CarRacingImageWrapper,
    StackFrames,
    make_env
)


def load_checkpoint(checkpoint_path, input_shape, n_actions, device):
    """Load model from checkpoint file"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")

    model = DQN(input_shape, n_actions).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Loaded checkpoint from episode {checkpoint.get('episode', 'N/A')}")
    if 'reward' in checkpoint:
        print(f"   Best reward: {checkpoint['reward']:.2f}")
    return model


def run_episode(env, model, device, render=True):
    """Run a single episode and return total reward"""
    state, _ = env.reset()
    total_reward = 0
    done = False

    with torch.no_grad():
        while not done:
            # Prepare state tensor (add batch dimension, normalize)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0

            # Get greedy action from policy
            action = model(state_tensor).max(1)[1].item()

            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state

            if render and hasattr(env, 'render'):
                env.render()

    return total_reward


def main():
    parser = argparse.ArgumentParser(
        description='Load and run trained DQN on CarRacing-v3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --checkpoint checkpoints/best_model.pth --episodes 3
  python inference.py --checkpoint checkpoints/final_model.pth --no-render --episodes 10
  python inference.py --save-video --video-dir ./test_videos
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

    args = parser.parse_args()

    # Override device if specified
    global device
    if args.device != 'auto':
        device = torch.device(args.device)
    print(f"🖥️  Using device: {device}")

    # Create environment with appropriate render mode
    render_mode = 'rgb_array' if args.save_video else ('human' if not args.no_render else None)
    env = make_env(render_mode=render_mode)

    # Wrap with video recorder if needed
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)
        env = RecordVideo(env, args.video_dir, name_prefix='inference')
        print(f"🎬 Video recording enabled, saving to: {args.video_dir}")

    # Get environment specs
    n_actions = env.action_space.n
    input_shape = env.observation_space.shape

    # Load model
    print(f"\n📂 Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, input_shape, n_actions, device)

    # Run episodes
    print(f"\n🏁 Running {args.episodes} episode(s)...\n")
    rewards = []

    try:
        for episode in range(args.episodes):
            print(f"Episode {episode + 1}/{args.episodes}...", end=" ")
            episode_reward = run_episode(env, model, device, render=not args.no_render)
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

    if args.save_video:
        print(f"\n🎥 Videos saved in: {args.video_dir}")


if __name__ == '__main__':
    main()