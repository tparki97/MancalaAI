# train_ai.py

import torch
import numpy as np
import random
import time
from gym.vector import SyncVectorEnv

# Import AI modules
from src.ai.random_ai import RandomAI
from src.ai.minimax import MinimaxAI
from src.ai.neural_network_ai import NeuralNetworkAI

# Import the game environment
from src.game.mancala import MancalaEnv

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_env():
    def _init():
        env = MancalaEnv()
        return env
    return _init

def main():
    # Set random seeds for reproducibility
    set_seed()

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Number of parallel environments
    num_envs = 4  # Adjust based on your system's capacity

    # Create vectorized environments
    envs = SyncVectorEnv([make_env() for _ in range(num_envs)])

    # Initialize agents
    dqn_agent = NeuralNetworkAI(device=device)
    # Opponent pool with increasing difficulty
    opponent_pool = [RandomAI(), MinimaxAI(depth=1), MinimaxAI(depth=2)]

    # Training parameters
    num_episodes = 10000
    batch_size = 32  # Reduced batch size
    target_update = 1000
    max_steps_per_episode = 100
    moving_average = 100  # For plotting moving averages

    latest_loss = None  # To track the latest loss for scheduler

    # Initialize states for all environments
    states, infos = envs.reset()

    print("Starting training...")

    # Initialize lists to collect metrics
    rewards_history = []
    losses_history = []
    epsilon_history = []
    q_value_history = []

    for episode in range(1, num_episodes + 1):
        # Adjust opponent based on curriculum learning
        if episode < 5000:
            minimax_agent = RandomAI()
        elif episode < 8000:
            minimax_agent = MinimaxAI(depth=1)
        else:
            minimax_agent = MinimaxAI(depth=2)

        # Initialize lists to collect rewards and done flags for all environments
        total_rewards = np.zeros(num_envs)
        done_flags = np.zeros(num_envs, dtype=bool)

        # Reset environments if episode > 1
        if episode > 1:
            states, infos = envs.reset()

        # Collect Q-values for analysis
        episode_q_values = []

        for t in range(max_steps_per_episode):
            actions = []
            env_indices = []

            for i in range(num_envs):
                if done_flags[i]:
                    continue  # Skip if the environment is done

                current_player = envs.envs[i].game.current_player

                if current_player == 1:
                    # DQN Agent's turn
                    valid_moves = envs.envs[i].game.get_valid_moves()
                    if not valid_moves:
                        # No valid moves, mark as done
                        done_flags[i] = True
                        continue
                    action = dqn_agent.select_action(states[i], valid_moves)

                    # Collect Q-values for the selected action
                    with torch.no_grad():
                        state_tensor = torch.tensor(states[i] / 48.0, dtype=torch.float32).unsqueeze(0).to(device)
                        q_values = dqn_agent.q_network(state_tensor).cpu().numpy()[0]
                        episode_q_values.append(q_values[action])

                else:
                    # Opponent's turn
                    move = minimax_agent.get_best_move(envs.envs[i].game)
                    if move is None:
                        # Opponent has no valid moves, mark as done
                        done_flags[i] = True
                        continue
                    action = move

                actions.append((i, action))
                env_indices.append(i)

            # If no actions were selected (all environments done), break
            if len(actions) == 0:
                break

            # Prepare actions array for all environments
            actions_array = np.zeros(num_envs, dtype=int)
            for idx, action in actions:
                actions_array[idx] = action

            # Step all environments
            next_states, rewards, dones, truncateds, infos = envs.step(actions_array)

            for idx in env_indices:
                if not done_flags[idx]:
                    total_rewards[idx] += rewards[idx]
                    # Store experience if it's the DQN Agent's turn
                    if envs.envs[idx].game.current_player == 2:
                        dqn_agent.replay_buffer.push(states[idx], actions_array[idx], rewards[idx], next_states[idx], dones[idx])

            # Update done flags
            done_flags = done_flags | dones

            # Move to the next state
            states = next_states

            # Perform a training step
            loss_step = dqn_agent.train_step(batch_size)
            if loss_step is not None:
                if 'loss' not in locals():
                    loss = 0.0
                loss += loss_step
                latest_loss = loss_step

            # Update target network periodically
            if dqn_agent.steps_done % target_update == 0:
                dqn_agent.update_target()

            # If all environments are done, break
            if done_flags.all():
                break

        # Aggregate total rewards from all environments
        episode_rewards = total_rewards.tolist()
        rewards_history.extend(episode_rewards)
        if 'loss' in locals():
            avg_loss = loss / (t + 1)  # Average loss per step
            losses_history.append(avg_loss)
            del loss
        else:
            avg_loss = 0.0

        # Collect epsilon and average Q-value
        epsilon_history.append(dqn_agent.epsilon)
        avg_q_value = np.mean(episode_q_values) if episode_q_values else 0.0
        q_value_history.append(avg_q_value)

        # Print progress every 100 episodes
        if episode % 100 == 0:
            if len(rewards_history) >= moving_average:
                avg_reward = np.mean(rewards_history[-moving_average:])
                avg_loss = np.mean(losses_history[-moving_average:]) if losses_history else 0.0
                avg_epsilon = np.mean(epsilon_history[-moving_average:])
                avg_q_value = np.mean(q_value_history[-moving_average:])
            else:
                avg_reward = np.mean(rewards_history)
                avg_loss = np.mean(losses_history) if losses_history else 0.0
                avg_epsilon = np.mean(epsilon_history)
                avg_q_value = np.mean(q_value_history)
            print(f"Episode {episode} - Avg Reward: {avg_reward:.2f} - Avg Loss: {avg_loss:.4f} - Avg Epsilon: {avg_epsilon:.4f} - Avg Q-Value: {avg_q_value:.4f}")

        # Save checkpoints every 1000 episodes
        if episode % 1000 == 0:
            checkpoint_path = f"models/neural_network_ai_episode_{episode}.pth"
            torch.save(dqn_agent.q_network.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Save the trained model
    dqn_agent.save_model()

    # Close TensorBoard writer
    dqn_agent.writer.close()

    # Close environments
    envs.close()

    print("Training completed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model and exiting...")
        dqn_agent.save_model()
        dqn_agent.writer.close()
        sys.exit(0)
