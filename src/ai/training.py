# src/ai/training.py

import torch
import torch.optim as optim
from src.game.mancala import MancalaGame
from src.ai.neural_network import MancalaNet
from src.ai.neural_network_ai import NeuralNetworkAI
import random
import os

class TrainingPipeline:
    def __init__(self, model_path='models/mancala_net.pth', learning_rate=0.001):
        self.model = MancalaNet()
        self.model_path = model_path
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

    def generate_self_play_data(self, num_games=1000):
        data = []
        for _ in range(num_games):
            game = MancalaGame()
            while not game.is_game_over():
                valid_moves = game.get_valid_moves()
                if not valid_moves:
                    break
                move = random.choice(valid_moves)
                data.append((game.board.copy(), move))
                game.make_move(move)
            game.collect_remaining_stones()
        return data

    def train(self, data, epochs=5, batch_size=32):
        inputs = torch.FloatTensor([state for state, move in data])
        targets = torch.LongTensor([move for state, move in data])
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for batch_inputs, batch_targets in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                loss = self.criterion(outputs, batch_targets)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item()}")
        # Save the trained model
        torch.save(self.model.state_dict(), self.model_path)
        print("Training completed and model saved.")

    def start_training(self):
        print("Generating self-play data...")
        data = self.generate_self_play_data()
        print(f"Generated {len(data)} data points.")
        print("Starting training...")
        self.train(data)
