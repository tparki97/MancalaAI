# main.py

import torch
from src.ai.neural_network_ai import NeuralNetworkAI
from src.ai.minimax import MinimaxAI
from src.game.mancala import MancalaEnv

def main():
    env = MancalaEnv()
    dqn_agent = NeuralNetworkAI()
    minimax_agent = MinimaxAI(depth=3)  # Initialize Minimax AI with desired depth

    # Load trained model
    dqn_agent.load_model()

    # Configure players
    player1 = 'DQN'      # Options: 'DQN', 'Minimax', 'Human'
    player2 = 'Minimax'  # Options: 'DQN', 'Minimax', 'Human'

    done = False
    state = env.reset()

    env.render()

    while not done:
        current_player = env.game.current_player
        if (current_player == 1 and player1 == 'DQN') or (current_player == 2 and player2 == 'DQN'):
            valid_moves = env.game.get_valid_moves()
            if not valid_moves:
                print(f"Player {current_player} has no valid moves.")
                break
            action = dqn_agent.select_action(state, valid_moves)
        elif (current_player == 1 and player1 == 'Minimax') or (current_player == 2 and player2 == 'Minimax'):
            move = minimax_agent.get_best_move(env.game)
            if move is None:
                print(f"Player {current_player} has no valid moves.")
                break
            action = move
        else:
            # Human player input
            valid_moves = env.game.get_valid_moves()
            print(f"Player {current_player}, select a pit from {valid_moves}:")
            try:
                action = int(input())
                if action not in valid_moves:
                    print("Invalid move! Try again.")
                    continue
            except ValueError:
                print("Invalid input! Enter a valid pit number.")
                continue

        next_state, reward, done, info = env.step(action)
        env.render()
        state = next_state

    # Game over
    if env.game.winner == 1:
        print("Player 1 (DQN) wins!")
    elif env.game.winner == 2:
        print("Player 2 (Minimax) wins!")
    else:
        print("It's a tie!")

if __name__ == "__main__":
    main()
