import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from game2048_config import Args


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=None):
        if masks is None:
            self.masks = None
            super().__init__(probs=probs, logits=logits, validate_args=validate_args)
            return

        if logits is None:
            raise ValueError("logits are required when masks are provided")

        self.masks = masks.to(dtype=torch.bool, device=logits.device)
        masked_logits = torch.where(self.masks, logits, torch.full_like(logits, -1e8))
        super().__init__(probs=probs, logits=masked_logits, validate_args=validate_args)

    def entropy(self):
        if self.masks is None:
            return super().entropy()

        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.zeros_like(p_log_p))
        return -p_log_p.sum(-1)


class Agent(nn.Module):
    def __init__(self, action_dim: int, args: Args):
        super().__init__()
        if args.network == "cnn":
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(1, args.cnn_channel, 3, padding=1)),
                nn.ReLU(),
                layer_init(nn.Conv2d(args.cnn_channel, args.cnn_channel, 3)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(args.cnn_channel * 2 * 2, args.linear_size)),
                nn.ReLU(),
                layer_init(nn.Linear(args.linear_size, args.linear_size)),
                nn.ReLU(),
            )
        elif args.network == "linear":
            self.network = nn.Sequential(
                nn.Flatten(),
                layer_init(nn.Linear(16, args.linear_size)),
                nn.ReLU(),
                layer_init(nn.Linear(args.linear_size, args.linear_size)),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"unsupported network type: {args.network}")

        self.actor = layer_init(nn.Linear(args.linear_size, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(args.linear_size, 1), std=1)

    def forward(self, x: torch.Tensor):
        hidden = self.network(x)
        policy_logits = self.actor(hidden)
        value = self.critic(hidden)
        return policy_logits, value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(x)
        return value

    def get_action_and_value(self, x: torch.Tensor, action=None, invalid_action_mask=None):
        policy_logits, value = self.forward(x)
        probs = CategoricalMasked(logits=policy_logits, masks=invalid_action_mask)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value
