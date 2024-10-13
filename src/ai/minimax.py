# src/ai/minimax.py

import math
import copy

class MinimaxAI:
    def __init__(self, depth=3):
        self.depth = depth

    def get_best_move(self, game):
        """
        Determines the best move using the Minimax algorithm with alpha-beta pruning.
        """
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None

        _, move = self.minimax(game, self.depth, -math.inf, math.inf, maximizingPlayer=(game.current_player == 2))
        return move

    def minimax(self, game, depth, alpha, beta, maximizingPlayer):
        if depth == 0 or game.is_game_over():
            return self.evaluate(game), None

        valid_moves = game.get_valid_moves()

        if not valid_moves:
            return self.evaluate(game), None

        best_move = None

        if maximizingPlayer:
            max_eval = -math.inf
            for move in valid_moves:
                new_game = copy.deepcopy(game)
                new_game.make_move(move)
                eval, _ = self.minimax(new_game, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = math.inf
            for move in valid_moves:
                new_game = copy.deepcopy(game)
                new_game.make_move(move)
                eval, _ = self.minimax(new_game, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def evaluate(self, game):
        """
        Evaluation function to assess the desirability of the game state.
        """
        # Simple evaluation: difference in Mancala stones from the perspective of current player
        if game.current_player == 2:
            return game.board[13] - game.board[6]
        else:
            return game.board[6] - game.board[13]
