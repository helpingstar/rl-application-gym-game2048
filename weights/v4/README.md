## network


In fact, `layer_init` is not needed if you are loading weights.

```python
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
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
        self.actor = layer_init(nn.Linear(args.linear_size, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(args.linear_size, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None, invalid_action_mask=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        # probs = Categorical(logits=logits)
        probs = CategoricalMasked(logits=logits, masks=invalid_action_mask)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
```

## `v4`

`v4` added a cnn layer with padding 1 to `v3`.

## `v4_01.pt`

```cmd
python game2048_v4_ppo_action_mask.py --wandb_project_name "game2048_v4" --total_timesteps 5000000000
```