import gymnasium as gym 
import numpy as np

# Initialize the Gym environment
env = gym.make("Taxi-v3")

# Discretize the state space
state_space_size = env.observation_space.n
action_space_size = env.action_space.n

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.99
exploration_prob = 1.0
min_exploration_prob = 0.01
exploration_decay = 0.995
num_episodes = 5000

# Initialize the Q-table with zeros
q_table = np.zeros((state_space_size, action_space_size))

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Exploration vs. exploitation
        if np.random.rand() < exploration_prob:
            action = env.action_space.sample()  # Choose a random action
        else:
            action = np.argmax(q_table[state, :])  # Choose the best action

        # Take the selected action
        next_state, reward, done, _ = env.step(action)

        # Update the Q-table using the Q-learning formula
        q_table[state, action] = (1 - learning_rate) * q_table[state, action] + \
                                 learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]))

        total_reward += reward
        state = next_state

    # Decay exploration probability
    exploration_prob = max(exploration_prob * exploration_decay, min_exploration_prob)

    # Print episode information
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# Evaluate the trained agent
total_rewards = []
num_evaluations = 100

for _ in range(num_evaluations):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(q_table[state, :])
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    total_rewards.append(total_reward)

# Print average performance over evaluations
average_reward = np.mean(total_rewards)
print(f"Average Reward Over {num_evaluations} Evaluations: {average_reward}")

# Close the environment
env.close()
