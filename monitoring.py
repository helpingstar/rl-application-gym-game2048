import gymnasium as gym
from gym_game2048.wrappers import Normalize2048
from gymnasium.wrappers import TransformReward, DtypeObservation, ReshapeObservation
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(device))
        return -p_log_p.sum(-1)


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, cnn_channel, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(cnn_channel * 2 * 2, linear_size),
            nn.ReLU(),
        )

        self.actor = nn.Linear(linear_size, 4)
        self.critic = nn.Linear(linear_size, 1)

    def get_action(self, x, invalid_action_mask=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = CategoricalMasked(logits=logits, masks=invalid_action_mask)
        action = probs.sample()
        return action


goal = 2**15
weight_path = "weights/cleanrl_game2048_v2_ppo_action_mask_48825.pt"
n_episode = 3
cnn_channel = 128
linear_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = Agent().to(device)
agent.load_state_dict(torch.load(weight_path))

env = gym.make("gym_game2048/Game2048-v0", goal=goal, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env,
    f"videos/monitoring",
    episode_trigger=lambda x: True,
)
# observation
env = ReshapeObservation(env, (1, 1, 4, 4))
env = DtypeObservation(env, np.float32)
env = Normalize2048(env)


observation, info = env.reset()
while n_episode > 0:
    invalid_action_mask = torch.Tensor(info["action_mask"]).to(device)
    action = agent.get_action(torch.Tensor(observation).to(device), invalid_action_mask=invalid_action_mask)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        n_episode -= 1
        if n_episode > 0:
            obs, info = env.reset()


env.close()
