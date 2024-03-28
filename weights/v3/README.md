## v3_1

```cmd
python game2048_v2_ppo_action_mask.py --cnn_channel 256 --num_steps 512 --linear_size 1024 --wandb_project_name "game2048_v3" --total_timesteps 5000000000
```

## v3_2

```cmd
python game2048_v2_ppo_action_mask.py --cnn_channel 256 --num_steps 512 --linear_size 1024 --wandb_project_name "game2048_v3" --total_timesteps 5000000000 --load_model "weights/v3/v3_1.pt" --learning_rate 0.0001
```