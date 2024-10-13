# src/ai/random_ai.py

import random

class RandomAI:
    def get_best_move(self, game):
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        return random.choice(valid_moves)
