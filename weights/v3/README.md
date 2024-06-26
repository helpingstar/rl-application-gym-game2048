## network

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
                layer_init(nn.Conv2d(1, args.cnn_channel, 3)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(args.cnn_channel * 2 * 2, args.linear_size)),
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

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
```

## v3_1

```cmd
python game2048_v2_ppo_action_mask.py --cnn_channel 256 --num_steps 512 --linear_size 1024 --wandb_project_name "game2048_v3" --total_timesteps 5000000000
```

## v3_2

```cmd
python game2048_v2_ppo_action_mask.py --cnn_channel 256 --num_steps 512 --linear_size 1024 --wandb_project_name "game2048_v3" --total_timesteps 5000000000 --load_model "weights/v3/v3_1.pt" --learning_rate 0.0001
```