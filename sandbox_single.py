import gymnasium as gym
from gym_game2048.wrappers import Normalize2048, RewardConverter, TerminateIllegal
from gymnasium.wrappers import TransformReward, DtypeObservation, ReshapeObservation
from tqdm import tqdm
import numpy as np

goal = 2**13


env = gym.make("gym_game2048/Game2048-v0", render_mode="human")
env = RewardConverter(env, term_rew=-5)
env = TerminateIllegal(env, -5)
env = ReshapeObservation(env, (1, 4, 4))
env = DtypeObservation(env, np.float32)
env = Normalize2048(env)
env = gym.wrappers.RecordEpisodeStatistics(env)

observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample(info["action_mask"])
    # action = 0

    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)
    if terminated or truncated:
        obs, info = env.reset()


env.close()
