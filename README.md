This is the code to solve [**gym-game2048**](https://github.com/helpingstar/gym-game2048), a reinforcement learning environment based on the 2048 game.

Detailed experimental results can be found in the wandb project link below.
* https://wandb.ai/iamhelpingstar/game2048_v2?nw=nwuseriamhelpingstar

After training, you can modify the `weight_path` in the `monitoring.py` file to record the agent's gameplay.

The training configuration is managed in `game2048_config.py`, the network definition is in `game2048_network.py`, and the PPO training entrypoint is `game2048_train.py`.

You can start training with:

```bash
python game2048_train.py
```

The code referenced
* https://github.com/vwxyzjn/cleanrl
* https://github.com/vwxyzjn/invalid-action-masking


![score](/figure/score.png)

![episodic_length](/figure/episodic_length.png)

![max_number](/figure/max_number.png)

![episodic_return](/figure/episodic_return.png)
