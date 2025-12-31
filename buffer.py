import numpy as np
import torch

class ReplayBuffer:
    """
    Optimized Replay Buffer using pre-allocated NumPy arrays.
    Provides O(1) random sampling and minimizes memory allocation overhead.
    """
    def __init__(self, capacity, state_shape, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        # Pre-allocate memory
        # State is uint8 to save RAM (4x-8x savings)
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample batch of experiences."""
        ind = np.random.randint(0, self.size, size=batch_size)

        # Retrieve batch updates (fast slicing)
        # Note: We normalize states here in the buffer return to keep main loop clean,
        # OR we can return uint8 and normalize on GPU. 
        # Returning numpy arrays here, conversion happens in train.py for flexibility
        return (
            self.states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind],
            self.dones[ind]
        )

    def __len__(self):
        return self.size

    def save(self, path):
        """Save buffer state efficiently."""
        np.savez_compressed(
            path,
            states=self.states[:self.size],
            next_states=self.next_states[:self.size],
            actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            dones=self.dones[:self.size],
            ptr=self.ptr
        )

    def load(self, path):
        """Load buffer state."""
        try:
            data = np.load(path)
            # Check if sizes match
            loaded_size = len(data['states'])
            if loaded_size > self.capacity:
                print(f"⚠️  Loaded buffer size ({loaded_size}) > capacity ({self.capacity}). Truncating.")
                loaded_size = self.capacity
            
            self.size = loaded_size
            self.ptr = int(data['ptr']) if 'ptr' in data else 0
            
            self.states[:self.size] = data['states'][:self.size]
            self.next_states[:self.size] = data['next_states'][:self.size]
            self.actions[:self.size] = data['actions'][:self.size]
            self.rewards[:self.size] = data['rewards'][:self.size]
            self.dones[:self.size] = data['dones'][:self.size]
            print(f"✅ Loaded native numpy buffer with {self.size} samples")
            return True
        except Exception as e:
            print(f"⚠️  Could not load native numpy buffer: {e}")
            return False
