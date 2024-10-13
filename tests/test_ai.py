# tests/test_ai.py

import unittest
from src.ai.neural_network_ai import NeuralNetworkAI
from src.ai.prioritized_replay_buffer import PrioritizedReplayBuffer
from src.game.mancala import MancalaEnv
import torch
import random
import numpy as np

class TestNeuralNetworkAI(unittest.TestCase):
    def setUp(self):
        self.env = MancalaEnv()
        self.observation, _ = self.env.reset()
        # Initialize a custom replay buffer for testing
        test_replay_buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
        self.ai = NeuralNetworkAI(model_path='models/test_neural_network_ai.pth', replay_buffer=test_replay_buffer)

    def test_select_action_exploration(self):
        # Force epsilon to 1 to test exploration
        self.ai.epsilon = 1.0
        valid_moves = self.env.game.get_valid_moves()
        action = self.ai.select_action(self.observation, valid_moves)
        print(f"Exploration Action Selected: {action}")
        self.assertIn(action, valid_moves)

    def test_select_action_exploitation(self):
        # Force epsilon_end to 0.0 to ensure no exploration
        self.ai.epsilon_end = 0.0
        self.ai.steps_done = self.ai.epsilon_decay * 1000  # Ensure epsilon_threshold is ~0.0

        # Mock random.random to always return a value greater than epsilon_threshold to ensure exploitation
        original_random = random.random
        def mock_random():
            return 1.0  # Always trigger exploitation
        random.random = mock_random

        # Mock Q-network to return higher value for action 0
        original_forward = self.ai.q_network.forward
        def mock_forward(x):
            # Ensure x is correctly normalized
            assert torch.all(x <= 4.8), "State normalization issue"
            output = torch.zeros(x.size(0), 14, device=self.ai.device)
            output[:, 0] = 10.0  # Highest Q-value for action 0
            print("Mocked Q-values:", output)
            return output
        self.ai.q_network.forward = mock_forward

        valid_moves = self.env.game.get_valid_moves()
        action = self.ai.select_action(self.observation, valid_moves)
        print(f"Exploitation Action Selected: {action}")
        self.assertEqual(action, 0)

        # Restore original methods
        self.ai.q_network.forward = original_forward
        random.random = original_random

    def test_replay_buffer_push_and_sample(self):
        # Push experiences until buffer is full
        for _ in range(100):
            state = self.env.game.get_state()
            valid_moves = self.env.game.get_valid_moves()
            if not valid_moves:
                break
            action = random.choice(valid_moves)
            reward = 1.0
            next_state = state.copy()
            done = False
            self.ai.replay_buffer.push(state, action, reward, next_state, done)

        self.assertEqual(len(self.ai.replay_buffer), 100)

        # Sample a batch
        sample = self.ai.replay_buffer.sample(10, beta=0.4)
        self.assertIsNotNone(sample)
        states, actions, rewards, next_states, dones, idxs, is_weights = sample
        print("Sampled States Shape:", states.shape)
        print("Sampled Actions Shape:", actions.shape)
        print("Sampled Rewards Shape:", rewards.shape)
        print("Sampled Next States Shape:", next_states.shape)
        print("Sampled Dones Shape:", dones.shape)
        print("Sampled Indexes Length:", len(idxs))
        print("Sampled IS Weights Shape:", is_weights.shape)
        self.assertEqual(states.shape, (10, 14))
        self.assertEqual(actions.shape, (10,))
        self.assertEqual(rewards.shape, (10,))
        self.assertEqual(next_states.shape, (10, 14))
        self.assertEqual(dones.shape, (10,))
        self.assertEqual(len(idxs), 10)
        self.assertEqual(is_weights.shape, (10,))

    def tearDown(self):
        # Clean up saved test model if exists
        import os
        if os.path.exists('models/test_neural_network_ai.pth'):
            os.remove('models/test_neural_network_ai.pth')

if __name__ == '__main__':
    unittest.main()
