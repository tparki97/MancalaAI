# src/ai/neural_network_ai.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from src.ai.prioritized_replay_buffer import PrioritizedReplayBuffer
import random
from torch.utils.tensorboard import SummaryWriter

class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)

        self.value_fc = nn.Linear(512, 256)
        self.value = nn.Linear(256, 1)

        self.advantage_fc = nn.Linear(512, 256)
        self.advantage = nn.Linear(256, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        value = self.relu(self.value_fc(x))
        value = self.value(value)

        advantage = self.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)

        q_values = value + advantage - advantage.mean()
        return q_values

class NeuralNetworkAI:
    def __init__(self, device=None):
        self.device = device or torch.device("cpu")
        self.input_dim = 14
        self.output_dim = 14
        self.q_network = DuelingQNetwork(self.input_dim, self.output_dim).to(self.device)
        self.target_network = DuelingQNetwork(self.input_dim, self.output_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)  # Reduced learning rate
        self.criterion = nn.MSELoss()
        self.gamma = 0.99

        # Epsilon-greedy parameters
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 500000  # Slower decay rate
        self.epsilon = self.epsilon_start
        self.steps_done = 0

        # Prioritized Experience Replay parameters
        alpha = 0.6
        beta_start = 0.4
        beta_frames = 100000
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        # Model save path
        self.model_path = 'models/neural_network_ai.pth'

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000, alpha=alpha)

        # TensorBoard writer
        self.writer = SummaryWriter('runs/mancala_dqn')

        # Load existing model if available
        self.load_model()

        # Set networks to training mode
        self.q_network.train()
        self.target_network.train()

    def update_target(self):
        """
        Performs a full update of the target network parameters.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.q_network.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.target_network.load_state_dict(self.q_network.state_dict())
                print(f"Loaded model from {self.model_path}")
            except RuntimeError as e:
                print(f"Failed to load model due to: {e}")
                print("Starting with a new model.")
        else:
            print(f"No existing model found at {self.model_path}. Starting with a new model.")

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.q_network.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def select_action(self, state, valid_moves):
        """
        Selects an action using an epsilon-greedy policy.
        """
        self.steps_done += 1
        epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.epsilon = epsilon_threshold

        # Log epsilon
        if self.steps_done % 1000 == 0:
            self.writer.add_scalar('Epsilon', self.epsilon, self.steps_done)

        if random.random() < self.epsilon:
            # Exploration: select a random valid action
            action = random.choice(valid_moves)
            return action
        else:
            # Exploitation: select the best action based on Q-values
            with torch.no_grad():
                state_tensor = torch.tensor(state / 48.0, dtype=torch.float32).unsqueeze(0).to(self.device)  # Normalize state
                q_values = self.q_network(state_tensor)
                q_values = q_values.cpu().numpy()[0]
                # Mask invalid moves
                mask = np.full_like(q_values, -np.inf)
                mask[valid_moves] = q_values[valid_moves]
                action = np.argmax(mask)
                return action

    def train_step(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        # Sample a batch with PER
        sample = self.replay_buffer.sample(batch_size, beta=self.beta)
        if sample is None:
            return  # Prevent unpacking if sample is None
        states, actions, rewards, next_states, dones, idxs, is_weights = sample

        # Update beta towards 1.0 over time
        self.beta = min(1.0, self.beta_start + self.steps_done * (1.0 - self.beta_start) / self.beta_frames)

        # Normalize and convert to tensors, move to device
        states = torch.tensor(states / 48.0, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states / 48.0, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        is_weights = torch.tensor(is_weights, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions)

        # Double DQN: action selection is from the Q-network, evaluation from the target network
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            max_next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        # Compute loss with importance-sampling weights
        loss = (self.criterion(current_q, target_q) * is_weights).mean()

        # Log loss
        if self.steps_done % 1000 == 0:
            self.writer.add_scalar('Loss/train', loss.item(), self.steps_done)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.5)  # Reduced threshold

        self.optimizer.step()

        # Update priorities in PER
        td_errors = (current_q - target_q).detach().cpu().numpy()
        new_priorities = np.abs(td_errors) + 1e-6  # Small epsilon to avoid zero
        self.replay_buffer.update_priorities(idxs, new_priorities.flatten())

        return loss.item()
