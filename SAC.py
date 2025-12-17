import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import random
from collections import deque
import os
from torch.utils.tensorboard import SummaryWriter

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
NUM_EPISODES = 4000
MAX_STEPS_PER_EPISODE = 1500
BATCH_SIZE = 768
GAMMA = 0.99
TAU = 0.005
ALPHA_INIT = 0.2
LEARNING_RATE = 8e-5
MEMORY_SIZE = 3000000

HIDDEN_SIZE = 1536
AUTO_ENTROPY_TUNING = True
LR_SCHEDULE = True

INITIAL_EXPLORATION_STEPS = 5000
EXPLORATION_NOISE = 0.1

EVAL_FREQUENCY = 100
NUM_EVAL_EPISODES = 3
SAVE_FREQUENCY = 100
VIDEO_DIR = "./videos"
CHECKPOINT_DIR = "./checkpoints"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================================
# ENVIRONMENT WRAPPERS
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
# NEURAL NETWORKS
# ============================================================================
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.log_std_min, self.log_std_max = -20, 2

        self.conv1 = nn.Conv2d(state_dim[0], 96, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(96, 192, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 256, kernel_size=3, stride=1)

        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(192)
        self.bn3 = nn.BatchNorm2d(256)

        convw = lambda s, k, st: (s - (k - 1) - 1) // st + 1
        w = convw(convw(convw(state_dim[2], 5, 2), 3, 2), 3, 1)
        h = convw(convw(convw(state_dim[1], 5, 2), 3, 2), 3, 1)
        linear_input_size = w * h * 256

        self.fc1 = nn.Linear(linear_input_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc_residual = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

        self.ln1 = nn.LayerNorm(HIDDEN_SIZE)
        self.ln2 = nn.LayerNorm(HIDDEN_SIZE)

        self.mean_layer = nn.Linear(HIDDEN_SIZE, action_dim)
        self.log_std_layer = nn.Linear(HIDDEN_SIZE, action_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, gain=0.01 if isinstance(module, nn.Linear) else np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        x = F.relu(self.ln1(self.fc1(x)))
        residual = x
        x = F.relu(self.ln2(self.fc2(x)))
        x = x + self.fc_residual(residual)

        mean = torch.tanh(self.mean_layer(x)) * 1.5
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        z = normal.rsample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()

        self.conv1 = nn.Conv2d(state_dim[0], 96, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(96, 192, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 256, kernel_size=3, stride=1)

        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(192)
        self.bn3 = nn.BatchNorm2d(256)

        convw = lambda s, k, st: (s - (k - 1) - 1) // st + 1
        w = convw(convw(convw(state_dim[2], 5, 2), 3, 2), 3, 1)
        h = convw(convw(convw(state_dim[1], 5, 2), 3, 2), 3, 1)
        cnn_output_size = w * h * 256

        self.fc1_q1 = nn.Linear(cnn_output_size + action_dim, HIDDEN_SIZE)
        self.fc2_q1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3_q1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2)
        self.q1 = nn.Linear(HIDDEN_SIZE // 2, 1)

        self.fc1_q2 = nn.Linear(cnn_output_size + action_dim, HIDDEN_SIZE)
        self.fc2_q2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3_q2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2)
        self.q2 = nn.Linear(HIDDEN_SIZE // 2, 1)

        self.ln1_q1 = nn.LayerNorm(HIDDEN_SIZE)
        self.ln2_q1 = nn.LayerNorm(HIDDEN_SIZE)
        self.ln1_q2 = nn.LayerNorm(HIDDEN_SIZE)
        self.ln2_q2 = nn.LayerNorm(HIDDEN_SIZE)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state, action):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        xu = torch.cat([x, action], dim=1)

        q1 = F.relu(self.ln1_q1(self.fc1_q1(xu)))
        q1 = F.relu(self.ln2_q1(self.fc2_q1(q1)))
        q1 = F.relu(self.fc3_q1(q1))
        q1 = self.q1(q1)

        q2 = F.relu(self.ln1_q2(self.fc1_q2(xu)))
        q2 = F.relu(self.ln2_q2(self.fc2_q2(q2)))
        q2 = F.relu(self.fc3_q2(q2))
        q2 = self.q2(q2)

        return q1, q2


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.size > 0 else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size < batch_size:
            return None

        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(self.size, batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = self.size
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        state, action, reward, next_state, done = zip(*samples)

        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32),
            indices,
            weights
        )

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6

    def __len__(self):
        return self.size


# ============================================================================
# SAC AGENT
# ============================================================================
class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

        if LR_SCHEDULE:
            self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.actor_optimizer, T_max=NUM_EPISODES, eta_min=1e-6
            )
            self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.critic_optimizer, T_max=NUM_EPISODES, eta_min=1e-6
            )

        if AUTO_ENTROPY_TUNING:
            self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().detach()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LEARNING_RATE)
        else:
            self.alpha = torch.tensor(ALPHA_INIT).to(device)

        self.total_steps = 0
        self.exploration_steps = INITIAL_EXPLORATION_STEPS

    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state_tensor)
                if self.total_steps < self.exploration_steps:
                    noise = torch.randn_like(action) * EXPLORATION_NOISE
                    action = torch.clamp(action + noise, -1.0, 1.0)

        return action.cpu().numpy()[0]

    def update_parameters(self, memory, batch_size):
        sample_result = memory.sample(batch_size)
        if sample_result is None:
            return None, None

        states, actions, rewards, next_states, dones, indices, weights = sample_result
        weights = torch.FloatTensor(weights).to(device).unsqueeze(1)

        states = torch.FloatTensor(states).to(device) / 255.0
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device) / 255.0
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * GAMMA * target_q

        current_q1, current_q2 = self.critic(states, actions)
        td_errors = torch.abs(current_q1 - target_q).detach().cpu().numpy().flatten()

        critic_loss = (weights * F.mse_loss(current_q1, target_q, reduction='none')).mean() + \
                      (weights * F.mse_loss(current_q2, target_q, reduction='none')).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        new_actions, log_probs = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_probs - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        if AUTO_ENTROPY_TUNING:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()
            alpha_loss_val = alpha_loss.item()
        else:
            alpha_loss_val = 0.0

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        memory.update_priorities(indices, td_errors)

        if LR_SCHEDULE and self.total_steps % 1000 == 0:
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss_val,
            'alpha': self.alpha.item(),
            'td_error': td_errors.mean()
        }, td_errors.mean()

    def save_actor_for_inference(self, path):
        """Saves only actor weights for inference (lightweight)."""
        torch.save(self.actor.state_dict(), path)

    def save_checkpoint(self, path):
        """Saves full checkpoint for resuming training."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']


# ============================================================================
# TRAINING
# ============================================================================
def main():
    writer = SummaryWriter("runs/car_racing_sac")

    print("=" * 70)
    print("CarRacing-v3 SAC Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 70)

    env = make_env()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    print(f"State shape: {state_dim}")
    print(f"Action shape: {action_dim}")

    agent = SACAgent(state_dim, action_dim)
    memory = PrioritizedReplayBuffer(MEMORY_SIZE)

    best_reward = -float('inf')
    total_steps = 0
    episode_rewards = []

    print("\nInitial exploration...")
    state, _ = env.reset()
    for _ in range(INITIAL_EXPLORATION_STEPS):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.push(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()[0]
        total_steps += 1
    print(f"Exploration complete: {INITIAL_EXPLORATION_STEPS:,} steps")

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.push(state, action, reward, next_state, done)

            if total_steps % 4 == 0:
                update_info, _ = agent.update_parameters(memory, BATCH_SIZE)
                if update_info:
                    writer.add_scalar("Loss/Critic", update_info['critic_loss'], total_steps)
                    writer.add_scalar("Loss/Actor", update_info['actor_loss'], total_steps)
                    writer.add_scalar("Alpha", update_info['alpha'], total_steps)

            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            agent.total_steps = total_steps

            if done:
                break

        writer.add_scalar("Reward/Episode", episode_reward, episode)
        episode_rewards.append(episode_reward)

        avg_reward_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        writer.add_scalar("Reward/Average_100", avg_reward_100, episode)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:4d}/{NUM_EPISODES} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Avg(100): {avg_reward_100:6.1f} | "
                  f"Steps: {episode_steps:4d}")

        if (episode + 1) % EVAL_FREQUENCY == 0:
            if avg_reward_100 > best_reward and avg_reward_100 > 700:
                best_reward = avg_reward_100
                agent.save_actor_for_inference(os.path.join(CHECKPOINT_DIR, "best_model.pth"))
                agent.save_checkpoint(os.path.join(CHECKPOINT_DIR, f"checkpoint_{episode + 1}.pth"))
                print(f"✅ New best! Saved checkpoint (Avg: {avg_reward_100:.1f})")

        if (episode + 1) % SAVE_FREQUENCY == 0:
            agent.save_checkpoint(os.path.join(CHECKPOINT_DIR, f"checkpoint_ep{episode + 1}.pth"))

    agent.save_actor_for_inference(os.path.join(CHECKPOINT_DIR, "final_model.pth"))
    agent.save_checkpoint(os.path.join(CHECKPOINT_DIR, "final_checkpoint.pth"))

    env.close()
    writer.close()

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Total episodes: {NUM_EPISODES}")
    print(f"Total steps: {total_steps:,}")
    print(f"Best average reward (100): {best_reward:.1f}")
    print(f"\nModels saved:")
    print(f"  - For inference: {CHECKPOINT_DIR}/best_model.pth")
    print(f"  - Full checkpoint: {CHECKPOINT_DIR}/final_checkpoint.pth")
    print(f"\nTensorBoard: tensorboard --logdir=runs/car_racing_sac")
    print("=" * 70)


if __name__ == "__main__":
    main()