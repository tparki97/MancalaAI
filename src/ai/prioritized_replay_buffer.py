# src/ai/prioritized_replay_buffer.py

import random
import numpy as np
from collections import namedtuple

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])

class SumTree:
    """
    A binary tree data structure where the parent’s value is the sum of its children.
    This structure is used to efficiently sample based on priority.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity  # Fixed-size list
        self.write = 0
        self.n_entries = 0

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority

        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self):
        return self.tree[0]

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer using SumTree.
    """
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-5  # Small amount to avoid zero priority

    def push(self, state, action, reward, next_state, done):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:]) if self.tree.n_entries > 0 else 1.0
        experience = Experience(state, action, reward, next_state, done)
        self.tree.add(max_priority, experience)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total_priority() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(s)
            if data is None:
                continue  # Skip if no data is present
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        if len(batch) == 0:
            return None  # Prevent unpacking if batch is empty

        # Avoid division by zero
        total_priority = self.tree.total_priority()
        if total_priority == 0:
            sampling_probabilities = np.ones(len(priorities)) / len(priorities)
        else:
            sampling_probabilities = np.array(priorities) / total_priority

        is_weights = np.power(self.tree.capacity * sampling_probabilities, -beta)
        is_weights /= is_weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones),
                idxs,
                is_weights)

    def update_priorities(self, idxs, priorities):
        for idx, priority in zip(idxs, priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries
