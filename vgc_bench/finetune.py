"""
Fine-tuning module for VGC-Bench.

Implements reinforcement learning training for Pokemon VGC agents using PPO.
Supports multiple training paradigms including self-play, fictitious play,
double oracle, and exploiter training, optionally initialized with behavior
cloning.
"""

import argparse
from pathlib import Path
from torch.optim import Adam, AdamW

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from vgc_bench.src.callback import Callback
from vgc_bench.src.env import ShowdownEnv
from vgc_bench.src.policy import MaskedActorCriticPolicy
from vgc_bench.src.utils import LearningStyle, set_global_seed, load_policy_from_zip


def finetune(
    reg_source: str,
    reg_target: str,
    run_id: int,
    num_teams: int | None,
    num_envs: int,
    num_eval_workers: int,
    log_level: int,
    port: int,
    device: str,
    learning_style: LearningStyle,
    behavior_clone: bool,
    allow_mirror_match: bool,
    choose_on_teampreview: bool,
    new_heads: bool,
    l2: bool,
    team1: str | None,
    team2: str | None,
    source_results_suffix: str,
    target_results_suffix: str,
    total_steps: int,
    columns: int = 0,
    evaluate: bool = True,
):
    """
    Fine-tune a Pokemon VGC policy using reinforcement learning.

    Creates the training environment, initializes PPO with the appropriate
    policy architecture, and runs training with periodic evaluation and
    checkpointing.

    Args:
        reg_source: VGC regulation letter (e.g. 'g', 'h', 'i').
        reg_target: VGC regulation letter (e.g. 'g', 'h', 'i').
        run_id: Training run identifier for saving/loading.
        num_teams: Number of teams to train with.
        num_envs: Number of parallel environments.
        num_eval_workers: Number of workers for evaluation battles.
        log_level: Logging verbosity for Showdown clients.
        port: Port for the Pokemon Showdown server.
        device: CUDA device for training.
        learning_style: Training paradigm (self-play, fictitious play, etc.).
        behavior_clone: Whether to initialize from a BC-pretrained policy.
        allow_mirror_match: Whether to allow same-team matchups.
        choose_on_teampreview: Whether policy makes teampreview decisions.
        new_heads: Whether to create new PPO heads or keep existing ones.
        l2: Whether to use L2 regularization with the AdamW optimizer.
        team1: Optional team string for matchup solving (requires team2).
        team2: Optional team string for matchup solving (requires team1).
        source_results_suffix: Suffix appended to results<run_id> for input paths.
        target_results_suffix: Suffix appended to results<run_id> for output paths.
        total_steps: Total training timesteps. Defaults to 1000 * save_interval.
        columns: How many columns the input policy has (0 for non-progressive).
        evaluate: Whether to run evaluations and save checkpoints.
    """
    save_interval = 98_304
    source_suffix = f"_{source_results_suffix}" if source_results_suffix else ""
    target_suffix = f"_{target_results_suffix}" if target_results_suffix else ""
    input_dir = Path(f"results{source_suffix}")
    output_dir = Path(f"results{target_suffix}")
    output_dir.mkdir(exist_ok=True)
    team_paths = None
    if team1 and team2:
        team1_path = input_dir / "team1.txt"
        team2_path = input_dir / "team2.txt"
        team1_path.write_text(team1[1:])
        team2_path.write_text(team2[1:])
        team_paths = [team1_path, team2_path]
    env = (
        ShowdownEnv.create_env(
            reg_target,
            run_id,
            num_teams,
            num_envs,
            log_level,
            port,
            learning_style,
            allow_mirror_match,
            choose_on_teampreview,
            team_paths,
        )
        if learning_style == LearningStyle.PURE_SELF_PLAY
        else SubprocVecEnv(
            [
                lambda: ShowdownEnv.create_env(
                    reg_target,
                    run_id,
                    num_teams,
                    num_envs,
                    log_level,
                    port,
                    learning_style,
                    allow_mirror_match,
                    choose_on_teampreview,
                    team_paths,
                )
                for _ in range(num_envs)
            ]
        )
    )
    method_tags = [
        "bc" if behavior_clone else None,
        learning_style.abbrev,
        "xm" if not allow_mirror_match else None,
        "xt" if not choose_on_teampreview else None,
    ]
    method = "_".join([p for p in method_tags if p is not None])
    method_dir = output_dir / f"saves_{method}"
    source_dir = input_dir / f"saves_{method}"
    policy_kwargs = {
        "d_model": 256,
        "choose_on_teampreview": choose_on_teampreview,
        "progressive": columns > 0,
        "optimizer_class": AdamW if l2 else Adam
    }
    if policy_kwargs["progressive"]:
        method_dir = method_dir / f"{columns+1}_columns"
        source_dir = source_dir / f"{columns}_columns"
        policy_kwargs["n_columns"] = columns # init with correct number of columns
    method_dir = method_dir / f"reg_{reg_source}_to_{reg_target}"
    source_dir = source_dir / f"reg_{reg_source}"
    if num_teams is not None:
        method_dir = method_dir / f"{num_teams}_teams"
        source_dir = source_dir / f"{num_teams}_teams"
    save_dir = method_dir / f"seed{run_id}"
    source_dir = source_dir / f"seed{run_id}"
    ppo = PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=lambda p: 1e-5 * 0.1 ** (1 - p),
        n_steps=(
            3072 // (2 * num_envs)
            if learning_style == LearningStyle.PURE_SELF_PLAY
            else 3072 // num_envs
        ),
        batch_size=512,
        gamma=1,
        # ent_coef is set in callback.py based on training progress
        tensorboard_log=str(output_dir / f"logs_{method}"),
        policy_kwargs=policy_kwargs,
        device=device,
    )
    assert source_dir.exists() and any(source_dir.iterdir()), f"source dir {source_dir} not found"
    saved_policy_timesteps = [
        int(file.stem) for file in source_dir.iterdir() if int(file.stem) >= 0
    ]
    num_saved_timesteps = max(saved_policy_timesteps)
    load_policy_from_zip(ppo.policy, str(source_dir / f"{num_saved_timesteps}.zip"), ppo.device)
    ppo.num_timesteps = num_saved_timesteps
    print(f"starting from {str(source_dir / f'{num_saved_timesteps}.zip')}")
    if policy_kwargs["progressive"]: # add column to progressive extractor
        ppo.policy.pi_features_extractor.add_column() # type: ignore
        ppo.policy.vf_features_extractor.add_column() # type: ignore
        ppo.policy.to(device)
    if new_heads: # reset action_net and value_net
        def reset_module(module):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        ppo.policy.action_net.apply(reset_module)
        ppo.policy.value_net.apply(reset_module)
    ppo.policy.optimizer = ppo.policy.optimizer_class( # reset optimizer, skip frozen modules
        filter(lambda p: p.requires_grad, ppo.policy.parameters()),
        **{"lr": 1e-5, "eps": 1e-5, "weight_decay": 1e-6 if l2 else 0, "betas": (0.99, 0.99) if l2 else (0.9, 0.999)}
    )
    ppo.learn(
        total_steps - num_saved_timesteps,
        callback=Callback(
            run_id,
            num_teams,
            reg_target,
            save_dir,
            num_eval_workers,
            log_level,
            port,
            learning_style,
            behavior_clone,
            allow_mirror_match,
            choose_on_teampreview,
            save_interval,
            team_paths,
            target_results_suffix,
            total_steps,
            evaluate,
        ),
        tb_log_name=str(save_dir.relative_to(output_dir / f"saves_{method}")),
        reset_num_timesteps=False,
    )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train a policy using population-based reinforcement learning. Must choose"
            " EXACTLY ONE of exploiter, self_play, fictitious_play, or double_oracle."
        )
    )
    parser.add_argument(
        "--exploiter",
        action="store_true",
        help=(
            "train against fixed policy, requires fixed policy file in save folder as"
            " -1.zip prior to training"
        ),
    )
    parser.add_argument(
        "--self_play",
        action="store_true",
        help="p1 and p2 are both controlled by same learning policy",
    )
    parser.add_argument(
        "--fictitious_play",
        action="store_true",
        help="p1 controlled by learning policy, p2 controlled by a past saved policy",
    )
    parser.add_argument(
        "--double_oracle",
        action="store_true",
        help=(
            "p1 controlled by learning policy, p2 controlled by past saved policy with"
            " selection weighted based on computed Nash equilibrium"
        ),
    )
    parser.add_argument(
        "--behavior_clone",
        action="store_true",
        help=(
            "use bc model as initial policy; if save folder has no checkpoint,"
            " downloads default BC checkpoint from Hugging Face"
        ),
    )
    parser.add_argument(
        "--no_mirror_match",
        action="store_true",
        help="disables same-team matchups during training, requires num_teams > 1",
    )
    parser.add_argument(
        "--no_teampreview",
        action="store_true",
        help=(
            "training agents will effectively start games after teampreview, with"
            " teampreview decision selected randomly"
        ),
    )
    parser.add_argument(
        "--new_heads",
        action="store_true",
        help="Create new PPO action- and value-network heads",
    )
    parser.add_argument(
        "--l2",
        action="store_true",
        help="Use L2 regularization with the AdamW optimizer",
    )
    parser.add_argument(
        "--reg_source",
        type=str,
        help="VGC regulation to start from (e.g. G).",
    )
    parser.add_argument(
        "--reg_target",
        type=str,
        help="VGC regulation to train on (e.g. G).",
    )
    parser.add_argument(
        "--run_id", type=int, default=1, help="run ID for the training session"
    )
    parser.add_argument(
        "--team1", type=str, default="", help="team 1 string for matchup solving"
    )
    parser.add_argument(
        "--team2", type=str, default="", help="team 2 string for matchup solving"
    )
    parser.add_argument(
        "--source_results_suffix",
        type=str,
        default="",
        help="suffix appended to results<run_id> for input paths",
    )
    parser.add_argument(
        "--target_results_suffix",
        type=str,
        default="",
        help="suffix appended to results<run_id> for output paths",
    )
    parser.add_argument(
        "--num_teams",
        type=int,
        default=None,
        help="number of teams to train with (default: all available teams)",
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="number of parallel envs to run"
    )
    parser.add_argument(
        "--num_eval_workers", type=int, default=1, help="number of eval workers to run"
    )
    parser.add_argument(
        "--log_level", type=int, default=40, help="log level for showdown clients"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="port to run showdown server on"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="device to use for training"
    )
    parser.add_argument(
        "--total_steps", type=int, required=True, help="total training timesteps"
    )
    parser.add_argument(
        "--columns", type=int, default=0, help="number of columns in the input policy"
    )
    args = parser.parse_args()
    set_global_seed(args.run_id)
    reg_source = args.reg_source.lower()
    reg_target = args.reg_target.lower()
    assert (
        int(args.exploiter)
        + int(args.self_play)
        + int(args.fictitious_play)
        + int(args.double_oracle)
        == 1
    )
    if args.exploiter:
        style = LearningStyle.EXPLOITER
    elif args.self_play:
        style = LearningStyle.PURE_SELF_PLAY
    elif args.fictitious_play:
        style = LearningStyle.FICTITIOUS_PLAY
    elif args.double_oracle:
        style = LearningStyle.DOUBLE_ORACLE
    else:
        raise TypeError()
    assert (args.team1 == "") == (args.team2 == ""), (
        "must provide both or neither of --team1 and --team2"
    )
    if args.team1 != "":
        assert args.results_suffix != "", (
            "--results_suffix is required when using --team1 and --team2"
        )
    finetune(
        reg_source,
        reg_target,
        args.run_id,
        args.num_teams,
        args.num_envs,
        args.num_eval_workers,
        args.log_level,
        args.port,
        args.device,
        style,
        args.behavior_clone,
        not args.no_mirror_match,
        not args.no_teampreview,
        args.new_heads,
        args.l2,
        args.team1 or None,
        args.team2 or None,
        args.source_results_suffix,
        args.target_results_suffix,
        args.total_steps,
        args.columns,
    )
