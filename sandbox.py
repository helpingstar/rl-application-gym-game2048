import gymnasium as gym
from gym_game2048.wrappers import Normalize2048, RewardByScore
from gymnasium.wrappers import TransformReward, DtypeObservation, ReshapeObservation
from tqdm import tqdm
import numpy as np

goal = 2**13


def make_env(idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make("gym_game2048/Game2048-v0", render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make("gym_game2048/Game2048-v0", None)
        env = RewardByScore(env, log=False, goal_bonus=0)
        env = TransformReward(env, lambda r: r / goal)
        env = ReshapeObservation(env, (1, 4, 4))
        env = DtypeObservation(env, np.float32)
        env = Normalize2048(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


envs = gym.vector.SyncVectorEnv(
    [make_env(i, False, "test") for i in range(4)],
)

# Reset the environment to generate the first observation
observation, info = envs.reset()
for _ in range(1000):
    # this is where you would insert your policy
    action = envs.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = envs.step(action)
    print(observation)


envs.close()
