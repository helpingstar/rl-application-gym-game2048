import argparse
import os
import time
from distutils.util import strtobool
from tqdm import tqdm
import gym_game2048
from gym_game2048.wrappers import RewardByScore, TerminateIllegalWrapper
from gymnasium.wrappers import TransformReward
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="gym_game2048/Game2048-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000000,
        help="total timesteps of the experiments")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")

    parser.add_argument("--goal", type=int, default=2048,
        help="goal")
    parser.add_argument("--memo", type=str, default="Linear 128",
        help="memo")

    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id, goal=args.goal)
        #### Add Custom Wrappers ###
        env = TerminateIllegalWrapper(env, -20)
        env = RewardByScore(env)
        env = TransformReward(env, lambda r: r * 0.01)
        #############################
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i) for i in range(args.num_envs)]
    )
    envs.reset(seed=args.seed)

    global_step = 0

    for update in tqdm(range(1, args.total_timesteps + 1)):
        global_step += 1 * args.num_envs

        actions = envs.action_space.sample()
        next_obs, reward, done, _, infos = envs.step(actions)

        # Only print when at least 1 env is done
        if "final_info" not in infos:
            continue

        for info in infos["final_info"]:
            # Skip the envs that are not done
            if info is None:
                continue
            # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            writer.add_scalar("charts/score", info["score"], global_step)
            writer.add_scalar("charts/max_number", info["max"], global_step)

    envs.close()
    writer.close()
