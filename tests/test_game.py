# tests/test_game.py

import unittest
from src.game.mancala import MancalaEnv

class TestMancalaGame(unittest.TestCase):
    def setUp(self):
        self.env = MancalaEnv()
        self.observation, _ = self.env.reset()

    def test_initial_state(self):
        expected_board = [4.0] * 6 + [0.0] + [4.0] * 6 + [0.0]
        self.assertListEqual(list(self.observation), expected_board)

    def test_valid_moves_player1(self):
        valid_moves = self.env.game.get_valid_moves()
        expected_moves = list(range(6))  # All pits have 4 stones initially
        self.assertListEqual(valid_moves, expected_moves)

    def test_invalid_move(self):
        # Set up a scenario where both Mancalas have 0 and both pits are empty
        self.env.game.board = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0]
        self.env.game.current_player = 1
        observation, reward, done, info = self.env.step(0)  # Player 1 has no valid moves
        self.assertTrue(done)
        self.assertEqual(reward, -1.0)
        self.assertIn('invalid_move', info)
        # After invalid move, winner should be set based on remaining stones
        # Since both Mancalas have 0, it's a tie
        self.assertEqual(self.env.game.winner, 0)  # Tie, since both Mancalas have 0

    def test_make_move(self):
        # Player 1 makes a valid move from pit 0 (which has 4 stones)
        action = 0
        observation, reward, done, info = self.env.step(action)
        # After moving pit 0, it should have 0 stones, pits1-4 have 5 stones, pit5 remains 4, Mancala remains 0
        expected_board = [0.0, 5.0, 5.0, 5.0, 5.0, 4.0] + [0.0] + [4.0, 4.0, 4.0, 4.0, 4.0, 4.0] + [0.0]
        self.assertListEqual(list(observation), expected_board)
        self.assertFalse(done)

    def test_capture_condition_player1(self):
        # Set up a scenario where Player 1 can capture
        # Player 1 pits: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], Mancala: [0.0]
        # Player 2 pits: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], Mancala: [0.0]
        self.env.game.board = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0]
        self.env.game.current_player = 1
        observation, reward, done, info = self.env.step(0)
        # After moving pit0, the stone should move to pit1
        # Since Player 2's pits are empty, game ends and the stone in pit1 is moved to Mancala
        expected_board = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [1.0] + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0]
        self.assertListEqual(list(observation), expected_board)
        self.assertTrue(done)
        self.assertEqual(self.env.game.winner, 1)  # Player 1 wins

    def test_game_over_tie(self):
        # Set up a scenario where the game should end with a tie
        self.env.game.board = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [10.0] + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [10.0]
        self.env.game.current_player = 1
        observation, reward, done, info = self.env.step(0)  # Player 1 has no valid moves
        self.assertTrue(done)
        self.assertEqual(self.env.game.winner, 0)  # Tie
        expected_board = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [10.0] + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [10.0]
        self.assertListEqual(list(observation), expected_board)

    def test_game_over_player2_wins(self):
        # Set up a scenario where Player 2 wins
        self.env.game.board = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [10.0] + [4.0, 4.0, 4.0, 4.0, 4.0, 4.0] + [10.0]
        self.env.game.current_player = 1
        observation, reward, done, info = self.env.step(0)  # Player 1 has no valid moves
        self.assertTrue(done)
        self.assertEqual(self.env.game.winner, 2)  # Player 2 wins
        expected_board = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [10.0] + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [34.0]
        self.assertListEqual(list(observation), expected_board)

if __name__ == '__main__':
    unittest.main()
