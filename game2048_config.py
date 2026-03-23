from dataclasses import dataclass
from typing import Optional


DEFAULT_EXP_NAME = "game2048_ppo_action_mask"


@dataclass
class Args:
    exp_name: str = DEFAULT_EXP_NAME
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "game2048_v2"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "gym_game2048/Game2048-v0"
    """the id of the environment"""
    total_timesteps: int = 400000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    goal: int = 2**15
    """Goal of gmae"""

    # track interval
    log_charts_interval: int = 1000
    """Record interval for chart"""
    log_losses_interval: int = 20
    """Record interval for losses"""
    record_interval: int = 100
    """Record interval for RecordVideo"""

    # network setting
    linear_size: int = 512
    """size of FCN"""
    cnn_channel: int = 128
    """size of CNN channel"""

    load_model: str = ""
    """whether to load model `runs/{run_name}` folder"""

    network: str = "cnn"
    """kind of network"""
    term_rew: float = -5.0
    """negative reward on termination"""
    div_pos_rew: int = 2**11
    """reward divided by this value"""
    action_mask: bool = True
    """whether to mask action"""


def finalize_args(args: Args) -> Args:
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    return args
