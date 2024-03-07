import gymnasium as gym
from gym_game2048.wrappers import Normalize2048, RewardByScore, TerminateIllegalWrapper
from gymnasium.wrappers import TransformReward, DtypeObservation, ReshapeObservation
from tqdm import tqdm
import numpy as np

goal = 2**13



env = gym.make("gym_game2048/Game2048-v0", render_mode="human")
env = RewardByScore(env, log=False, goal_bonus=0)
env = TransformReward(env, lambda r: r / goal)
env = TerminateIllegalWrapper(env, -1)
env = ReshapeObservation(env, (1, 4, 4))
env = DtypeObservation(env, np.float32)
env = Normalize2048(env)

env = gym.wrappers.RecordEpisodeStatistics(env)

observation, info = env.reset()
for _ in range(1000):
    # action = env.action_space.sample()
    action = 0

    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)
    if terminated or truncated:
        env.reset()


env.close()
