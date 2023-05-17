import gymnasium as gym
import gym_game2048

env = gym.make("gym_game2048/Game2048-v0", render_mode="human")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
       observation, info = env.reset()
env.close()
