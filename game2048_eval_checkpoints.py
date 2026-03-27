import csv
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import gymnasium as gym
import numpy as np
import torch
import tyro
from tqdm import tqdm

import gym_game2048
from game2048_config import Args
from game2048_network import Agent
from gym_game2048.wrappers import Normalize2048
from gymnasium.wrappers import DtypeObservation, ReshapeObservation


@dataclass
class EvalArgs:
    checkpoint_glob: str = "runs/gym_game2048/*/checkpoints/infer/*.pt"
    """Glob pattern used to snapshot checkpoint files at startup."""

    output_csv: str = "runs/gym_game2048/checkpoint_eval_scores.csv"
    """CSV file that stores checkpoint paths and average scores."""

    num_episodes: int = 100
    """Number of episodes to evaluate for each checkpoint."""

    num_envs: int = 32
    """Number of parallel environments used during evaluation."""

    goal: int = 2**17
    """Evaluation goal passed to the 2048 environment."""

    cuda: bool = True
    """Use CUDA when available."""

    seed_base: int = 1
    """Base seed used to derive a different seed per checkpoint."""


def make_eval_env(goal: int):
    def thunk():
        env = gym.make("gym_game2048/Game2048-v0", goal=goal)
        env = ReshapeObservation(env, (1, 4, 4))
        env = DtypeObservation(env, np.float32)
        env = Normalize2048(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def snapshot_checkpoints(checkpoint_glob: str) -> List[Path]:
    checkpoint_paths = sorted(Path(path) for path in glob.glob(checkpoint_glob))
    return [path for path in checkpoint_paths if path.is_file()]


def load_agent(checkpoint_path: Path, device: torch.device, action_dim: int) -> Agent:
    metadata_path = checkpoint_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    agent_args = Args(
        network=metadata["network"],
        cnn_channel=int(metadata["cnn_channel"]),
        linear_size=int(metadata["linear_size"]),
    )
    agent = Agent(action_dim, agent_args).to(device)
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()
    return agent


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint_path: Path,
    checkpoint_index: int,
    args: EvalArgs,
    envs: gym.vector.AsyncVectorEnv,
    device: torch.device,
) -> float:
    agent = load_agent(checkpoint_path, device=device, action_dim=envs.single_action_space.n)
    checkpoint_seed = args.seed_base + checkpoint_index

    next_obs, infos = envs.reset(seed=checkpoint_seed)
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    episode_scores: List[float] = []

    while len(episode_scores) < args.num_episodes:
        action_mask = torch.as_tensor(infos["action_mask"], device=device, dtype=torch.bool)
        policy_logits, _ = agent(next_obs)

        # Evaluation uses greedy actions while respecting invalid action masks.
        masked_logits = torch.where(
            action_mask,
            policy_logits,
            torch.full_like(policy_logits, torch.finfo(policy_logits.dtype).min),
        )
        actions = masked_logits.argmax(dim=-1)

        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)

        if "episode" not in infos:
            continue

        done_mask = infos["episode"]["_r"]
        for env_index, done in enumerate(done_mask):
            if not done:
                continue
            episode_scores.append(float(infos["score"][env_index]))
            if len(episode_scores) == args.num_episodes:
                break

    return float(np.mean(episode_scores))


def write_results(output_csv: Path, rows: List[dict]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["pt_path", "avg_score"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = tyro.cli(EvalArgs)
    checkpoint_paths = snapshot_checkpoints(args.checkpoint_glob)
    if not checkpoint_paths:
        raise FileNotFoundError(f"no checkpoints matched: {args.checkpoint_glob}")

    effective_num_envs = max(1, min(args.num_envs, args.num_episodes))
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.AsyncVectorEnv([make_eval_env(args.goal) for _ in range(effective_num_envs)])

    rows: List[dict] = []
    progress = tqdm(checkpoint_paths, desc="Evaluating checkpoints")
    try:
        for checkpoint_index, checkpoint_path in enumerate(progress):
            avg_score = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                checkpoint_index=checkpoint_index,
                args=args,
                envs=envs,
                device=device,
            )
            progress.set_postfix_str(f"avg_score={avg_score:.2f}")
            rows.append(
                {
                    "pt_path": checkpoint_path.as_posix(),
                    "avg_score": avg_score,
                }
            )
    finally:
        envs.close()

    write_results(Path(args.output_csv), rows)


if __name__ == "__main__":
    main()
