# src/game/mancala.py

import numpy as np
import gym
from gym import spaces

class MancalaGame:
    def __init__(self):
        self.reset()

    def reset(self):
        # Initialize the board: 4 stones in each of the 12 small pits, 0 in the Mancalas
        self.board = [4.0] * 6 + [0.0] + [4.0] * 6 + [0.0]
        self.current_player = 1
        self.done = False
        self.winner = None
        self.captured_opponent_stones = 0

    def get_valid_moves(self):
        start = 0 if self.current_player == 1 else 7
        end = 6 if self.current_player == 1 else 13
        valid_moves = [i for i in range(start, end) if self.board[i] > 0]
        return valid_moves

    def make_move(self, pit_index):
        stones = int(self.board[pit_index])
        self.board[pit_index] = 0
        index = pit_index

        while stones > 0:
            index = (index + 1) % 14
            # Skip opponent's Mancala
            if (self.current_player == 1 and index == 13) or (self.current_player == 2 and index == 6):
                continue
            self.board[index] += 1
            stones -= 1

        # Capture condition
        if self.current_player == 1 and 0 <= index <= 5 and self.board[index] == 1:
            opposite_index = 12 - index
            captured_stones = self.board[opposite_index]
            if captured_stones > 0:
                self.board[6] += captured_stones + 1
                self.board[opposite_index] = 0
                self.board[index] = 0
                self.captured_opponent_stones = captured_stones
            else:
                self.captured_opponent_stones = 0
        elif self.current_player == 2 and 7 <= index <= 12 and self.board[index] == 1:
            opposite_index = 12 - index
            captured_stones = self.board[opposite_index]
            if captured_stones > 0:
                self.board[13] += captured_stones + 1
                self.board[opposite_index] = 0
                self.board[index] = 0
                self.captured_opponent_stones = captured_stones
            else:
                self.captured_opponent_stones = 0
        else:
            self.captured_opponent_stones = 0

        # Check for extra turn
        if (self.current_player == 1 and index == 6) or (self.current_player == 2 and index == 13):
            pass  # Same player's turn
        else:
            self.current_player = 2 if self.current_player == 1 else 1

        # Check for game over
        if sum(self.board[0:6]) == 0 or sum(self.board[7:13]) == 0:
            self.done = True
            self.board[6] += sum(self.board[0:6])
            self.board[13] += sum(self.board[7:13])
            for i in range(0, 6):
                self.board[i] = 0
            for i in range(7, 13):
                self.board[i] = 0
            if self.board[6] > self.board[13]:
                self.winner = 1
            elif self.board[13] > self.board[6]:
                self.winner = 2
            else:
                self.winner = 0  # Tie

    def get_state(self):
        return np.array(self.board, dtype=np.float32)

    def is_game_over(self):
        return self.done

class MancalaEnv(gym.Env):
    """
    Custom Gym environment for Mancala.
    """
    def __init__(self):
        super(MancalaEnv, self).__init__()
        self.game = MancalaGame()
        # Define action and observation space
        self.action_space = spaces.Discrete(14)
        self.observation_space = spaces.Box(low=0.0, high=48.0, shape=(14,), dtype=np.float32)

    def reset(self):
        self.game.reset()
        return self.game.get_state(), {}

    def step(self, action):
        """
        Executes the action and returns the result.
        """
        info = {}
        if self.game.done:
            # Return done state
            return self.game.get_state(), 0.0, True, False, info

        # Validate action
        valid_moves = self.game.get_valid_moves()
        if action not in valid_moves:
            # Invalid move: assign a negative reward and end the episode
            reward = -1.0  # Reduced penalty for invalid move
            done = True
            truncated = False
            info['invalid_move'] = True
            return self.game.get_state(), reward, done, truncated, info

        # Store previous state for reward calculation
        prev_mancala_stones = self.game.board[6] if self.game.current_player == 1 else self.game.board[13]

        self.game.make_move(action)

        # Reward function
        reward = 0.0  # Initialize reward

        # Reward for depositing stones in own Mancala
        current_mancala_stones = self.game.board[6] if self.game.current_player == 2 else self.game.board[13]
        mancala_gain = current_mancala_stones - prev_mancala_stones
        if mancala_gain > 0:
            reward += 0.1  # Fixed small reward

        # Reward for capturing opponent's stones
        if self.game.captured_opponent_stones > 0:
            reward += 0.2  # Fixed small reward

        # Penalty for giving opponent extra turn
        if (self.game.current_player == 2 and action == 6) or (self.game.current_player == 1 and action == 13):
            reward -= 0.1  # Small penalty

        # Check if the game is over
        if self.game.done:
            if self.game.winner == 1:
                reward += 1.0  # Reduced reward for winning
            elif self.game.winner == 2:
                reward -= 1.0  # Reduced penalty for losing
            else:
                reward += 0.0  # Neutral for a tie
            done = True
            truncated = False
        else:
            done = False
            truncated = False

        # Scale and clip reward
        reward = np.clip(reward, -1.0, 1.0)

        return self.game.get_state(), reward, done, truncated, info

    def render(self, mode='human'):
        # Optional: Implement rendering logic if needed
        pass

    def close(self):
        pass
