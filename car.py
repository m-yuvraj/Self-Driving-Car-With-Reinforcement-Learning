import gym
from stable_baselines3 import DDPG
from stable_baselines3.common.envs import BoxActionWrapper
from stable_baselines3.common.vec_env import VecFrameStack

# Create a CarRacing environment
env = gym.make("CarRacing-v0")
env = BoxActionWrapper(env)
env = VecFrameStack(env, n_stack=4)

# Define and train the DDPG agent
model = DDPG("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ddpg_car_racing")

# Evaluate the trained agent
mean_reward, _ = model.evaluate(env, n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward}")

# Close the environment
env.close()
