This is the code to solve [**gym-game2048**](https://github.com/helpingstar/gym-game2048), a reinforcement learning environment based on the 2048 game.

The code referenced [**cleanrl**](https://github.com/vwxyzjn/cleanrl)'s code.

![score](/figure/score.png)

![episodic_length](/figure/episodic_length.png)

![max_number](/figure/max_number.png)

![episodic_return](/figure/episodic_return.png)

# Reward

* Illegal Action : -5 (with termination)
* Combine Block : (the number of blocks combined) x 2 x 0.0025
  * ex) combine 2 and 2 : 2 x 2 0.0025 = 0.01


# HyperParameter

## PPO (with CNN)

| Name                | ppo_cnn    |
| ------------------- | ---------- |
| anneal_lr           | TRUE       |
| batch_size          | 2048       |
| clip_coef           | 0.2        |
| clip_vloss          | TRUE       |
| cnn_channel         | 128        |
| cuda                | TRUE       |
| ent_coef            | 0.01       |
| gae_lambda          | 0.95       |
| gamma               | 0.99       |
| goal                | 2048       |
| learning_rate       | 0.00025    |
| linear_size         | 512        |
| max_grad_norm       | 0.5        |
| minibatch_size      | 512        |
| norm_adv            | TRUE       |
| num_envs            | 16         |
| num_minibatches     | 4          |
| num_steps           | 128        |
| seed                | 1          |
| torch_deterministic | TRUE       |
| total_timesteps     | 1000000000 |
| track               | TRUE       |
| update_epochs       | 4          |
| vf_coef             | 0.5        |
| illegal_terminate   | \-5        |
| log_charts_interval | 1000       |
| log_losses_interval | 10         |
