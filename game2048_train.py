import json
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import gym_game2048
from game2048_config import Args, finalize_args
from game2048_network import Agent
from gym_game2048.wrappers import Normalize2048, RewardConverter
from gymnasium.wrappers import DtypeObservation, ReshapeObservation


def make_env(args: Args, idx: int, run_name: str):
    def thunk():
        if args.capture_video and idx == 0:
            env = gym.make(args.env_id, goal=args.goal, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda x: x % args.record_interval == 0,
            )
        else:
            env = gym.make(args.env_id, goal=args.goal)

        env = RewardConverter(env, div_pos_rew=args.div_pos_rew, term_rew=args.term_rew)
        env = ReshapeObservation(env, (1, 4, 4))
        env = DtypeObservation(env, np.float32)
        env = Normalize2048(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def save_checkpoints(
    agent: Agent,
    args: Args,
    run_dir: Path,
    iteration: int,
    action_dim: int,
    input_shape: tuple[int, ...],
) -> None:
    train_checkpoint_dir = run_dir / "checkpoints" / "train"
    infer_checkpoint_dir = run_dir / "checkpoints" / "infer"
    train_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    infer_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_model_path = train_checkpoint_dir / f"train_{iteration}.pt"
    infer_model_path = infer_checkpoint_dir / f"infer_{iteration}.pt"
    infer_metadata_path = infer_checkpoint_dir / f"infer_{iteration}.json"

    state_dict = agent.state_dict()
    torch.save(state_dict, train_model_path)
    torch.save(state_dict, infer_model_path)

    infer_metadata = {
        "network": args.network,
        "cnn_channel": int(args.cnn_channel),
        "linear_size": int(args.linear_size),
        "action_dim": int(action_dim),
        "input_shape": [int(dim) for dim in input_shape],
        "output_names": ["policy_logits", "value"],
    }
    infer_metadata_path.write_text(json.dumps(infer_metadata, indent=2), encoding="utf-8")

    print(f"train model saved to {train_model_path}")
    print(f"infer model saved to {infer_model_path}")
    print(f"infer metadata saved to {infer_metadata_path}")


def train(args: Args) -> None:
    charts_count = 0
    losses_count = 0
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = Path("runs") / run_name

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    writer = SummaryWriter(str(run_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.AsyncVectorEnv(
        [make_env(args, i, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs.single_action_space.n, args).to(device)
    if args.load_model:
        agent.load_state_dict(torch.load(args.load_model, map_location=device))
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    invalid_action_masks = None
    if args.action_mask:
        invalid_action_masks = torch.zeros(
            (args.num_steps, args.num_envs, envs.single_action_space.n),
            device=device,
        )

    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, device=device)
    save_interval = max(1, args.num_iterations // 100)
    infer_input_shape = (1,) + envs.single_observation_space.shape

    for iteration in tqdm(range(1, args.num_iterations + 1)):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            action_mask = None
            if args.action_mask:
                action_mask = torch.as_tensor(infos["action_mask"], device=device, dtype=torch.bool)
                invalid_action_masks[step] = action_mask

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, invalid_action_mask=action_mask)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(next_done, device=device, dtype=torch.float32)

            if "episode" in infos:
                for i, done in enumerate(infos["episode"]["_r"]):
                    if not done:
                        continue
                    if charts_count % args.log_charts_interval == 0:
                        writer.add_scalar("charts/episodic_return", infos["episode"]["r"][i], global_step)
                        writer.add_scalar("charts/episodic_length", infos["episode"]["l"][i], global_step)
                        writer.add_scalar("charts/score", infos["score"][i], global_step)
                        writer.add_scalar("charts/max_number", infos["max"][i], global_step)
                        charts_count = 1
                    else:
                        charts_count += 1

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_invalid_action_masks = None
        if args.action_mask:
            b_invalid_action_masks = invalid_action_masks.reshape((-1, envs.single_action_space.n))

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                minibatch_mask = None if b_invalid_action_masks is None else b_invalid_action_masks[mb_inds]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions.long()[mb_inds],
                    minibatch_mask,
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if losses_count % args.log_losses_interval == 0:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            losses_count = 1
        else:
            losses_count += 1

        if iteration % save_interval == 0:
            save_checkpoints(
                agent=agent,
                args=args,
                run_dir=run_dir,
                iteration=iteration,
                action_dim=envs.single_action_space.n,
                input_shape=infer_input_shape,
            )

    envs.close()
    writer.close()


def main() -> None:
    args = finalize_args(tyro.cli(Args))
    train(args)


if __name__ == "__main__":
    main()
