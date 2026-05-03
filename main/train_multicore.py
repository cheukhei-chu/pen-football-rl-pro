import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import sys
import traceback
from collections import defaultdict, deque

import numpy as np
import torch
import torch.optim as optim
import yaml

from policy import (
    DummyPolicy,
    atulPolicy,
    empty_action_storage,
    make_policy,
    policy_from_checkpoint_path,
    sample_action_from_logits,
)
from train import (
    StateArchive,
    compute_gae,
    compute_td_errors,
    evaluate_matchup,
    linear_schedule,
    load_or_create_policy,
    load_policy_cached,
    parse_policy_spec,
    pfsp_weight,
    ppo_update,
    reset_policy_state,
    sample_weighted,
    save_policy_checkpoint,
    set_optimizer_lr,
    hard_weight,
)


def load_opponent_from_identifier(identifier, cache):
    if identifier not in cache:
        if identifier == "builtin:dummy":
            policy = DummyPolicy()
        elif identifier == "builtin:atul":
            policy = atulPolicy()
        else:
            policy, _ = policy_from_checkpoint_path(identifier)
        policy.eval()
        cache[identifier] = policy
    return cache[identifier]


def sample_training_reset_from_snapshot(
    archive_entries,
    archive_reset_prob,
    archive_alpha,
):
    if not archive_entries:
        return None
    if random.random() >= archive_reset_prob:
        return None

    weights = [entry["priority"] ** archive_alpha for entry in archive_entries]
    sampled = random.choices(archive_entries, weights=weights, k=1)[0]
    return {
        "state": copy.deepcopy(sampled["state"]),
        "reset_score": True,
        "reset_time_steps": True,
    }


def worker_rollout(
    env,
    policy_red,
    policy_blue,
    rollout_len,
    gamma,
    lam,
    archive_entries,
    archive_reset_prob,
    archive_alpha,
    capture_states,
):
    obs_rows = []
    act_list = empty_action_storage(policy_red)
    logp_list = []
    rew_list = []
    done_list = []
    val_list = []
    episode_scores = []
    sim_state_list = []
    episode_step_list = []
    archived_reset_episodes = 0
    total_episodes = 0

    m_rew_list = []
    m_val_list = []
    m_logp_list = []
    raw_goal_list = []
    is_feudal = hasattr(policy_red, "_update_manager")

    steps = 0
    obs = None

    while steps < rollout_len:
        drill = None
        policy_red.set_setting(drill)

        if is_feudal:
            policy_red.reset_state()

        reset_options = sample_training_reset_from_snapshot(
            archive_entries,
            archive_reset_prob,
            archive_alpha,
        )
        obs, _ = env.reset(options=reset_options)
        total_episodes += 1
        archived_reset_episodes += int(reset_options is not None)

        done = False
        episode_step = 0

        while not done and steps < rollout_len:
            obs_tensor = torch.tensor(obs["player_red"], dtype=torch.float32).unsqueeze(0)

            if capture_states:
                sim_state_list.append(env.game.get_sim_state())
                episode_step_list.append(episode_step)

            with torch.no_grad():
                if is_feudal:
                    policy_red._update_manager(obs_tensor)
                    curr_raw_goal = policy_red.last_raw_goal.clone().detach()
                    curr_m_logp = policy_red.last_manager_log_prob.clone().detach()
                    _, _, m_val_t = policy_red.evaluate_manager(obs_tensor, curr_raw_goal)
                    m_val = m_val_t.item()

                logits = policy_red.forward(obs_tensor)
                value = logits["value"].item()

                action, action_record, logp = sample_action_from_logits(logits)

            next_obs, rewards, terminated, truncated, info = env.step(
                {
                    "player_red": action,
                    "player_blue": policy_blue.sample_action(obs["player_blue"]),
                }
            )

            done = terminated["__all__"] or truncated["__all__"]

            r_ext = rewards["player_red"]
            r_worker = r_ext

            if is_feudal:
                r_int = policy_red.compute_intrinsic_reward(next_obs["player_red"])
                r_worker += r_int
                m_rew_list.append(r_ext)
                m_val_list.append(m_val)
                raw_goal_list.append(curr_raw_goal.cpu().numpy())
                m_logp_list.append(curr_m_logp.cpu().numpy())

            obs_rows.append(obs["player_red"])
            for key, value in action_record.items():
                act_list[key].append(value)
            logp_list.append(logp)
            rew_list.append(r_worker)
            done_list.append(done)
            val_list.append(value)

            steps += 1
            obs = next_obs
            episode_step += 1

            if done:
                result = info.get("result")
                if result == "red":
                    episode_scores.append(1.0)
                elif result == "blue":
                    episode_scores.append(0.0)
                elif result == "draw":
                    episode_scores.append(0.5)

            if steps >= rollout_len:
                break

    with torch.no_grad():
        w_last_val = policy_red.forward(
            torch.tensor(obs["player_red"], dtype=torch.float32).unsqueeze(0)
        )["value"].item()

    adv = compute_gae(rew_list, val_list, done_list, w_last_val, gamma, lam)
    ret = adv + torch.tensor(val_list, dtype=torch.float32)

    result = {
        "obs": np.asarray(obs_rows, dtype=np.float32),
        "acts": act_list,
        "logp": logp_list,
        "rew": rew_list,
        "done": done_list,
        "old_values": val_list,
        "adv": adv.cpu().numpy(),
        "ret": ret.cpu().numpy(),
        "episode_scores": episode_scores,
        "archived_reset_episodes": archived_reset_episodes,
        "total_episodes": total_episodes,
    }

    if capture_states:
        td_errors = compute_td_errors(rew_list, val_list, done_list, w_last_val, gamma)
        result["sim_states"] = sim_state_list
        result["episode_steps"] = episode_step_list
        result["td_errors"] = td_errors

    if is_feudal:
        with torch.no_grad():
            dummy_g = torch.zeros(1, policy_red.goal_dim)
            _, _, m_last_val_t = policy_red.evaluate_manager(
                torch.tensor(obs["player_red"], dtype=torch.float32).unsqueeze(0),
                dummy_g,
            )
            m_last_val = m_last_val_t.item()
        m_adv = compute_gae(m_rew_list, m_val_list, done_list, m_last_val, gamma, lam)
        m_ret = m_adv + torch.tensor(m_val_list, dtype=torch.float32)
        result["manager_data"] = {
            "raw_goals": raw_goal_list,
            "m_logp": m_logp_list,
            "m_adv": m_adv.cpu().numpy(),
            "m_ret": m_ret.cpu().numpy(),
            "m_old_values": m_val_list,
        }

    return result


def rollout_worker_main(conn, policy_class_name, policy_kwargs):
    from multiagent import FootballMultiAgentEnv

    env = FootballMultiAgentEnv()
    policy_red = make_policy(policy_class_name, **policy_kwargs)
    opponent_cache = {}

    try:
        while True:
            message = conn.recv()
            command = message.get("command")

            if command == "close":
                break

            if command != "collect":
                raise ValueError(f"Unknown worker command: {command}")

            task_seed = message.get("seed")
            if task_seed is not None:
                random.seed(task_seed)
                np.random.seed(task_seed)
                torch.manual_seed(task_seed)

            state_dict = {
                key: torch.from_numpy(value)
                for key, value in message["policy_state_dict"].items()
            }
            policy_red.load_state_dict(state_dict)
            policy_red.eval()

            opponent_id = message["opponent_id"]
            policy_blue = load_opponent_from_identifier(opponent_id, opponent_cache)

            result = worker_rollout(
                env=env,
                policy_red=policy_red,
                policy_blue=policy_blue,
                rollout_len=message["rollout_len"],
                gamma=message["gamma"],
                lam=message["lam"],
                archive_entries=message.get("archive_entries"),
                archive_reset_prob=message["archive_reset_prob"],
                archive_alpha=message["archive_alpha"],
                capture_states=message["capture_states"],
            )
            conn.send({"ok": True, "result": result})
    except Exception:
        conn.send({"ok": False, "error": traceback.format_exc()})
    finally:
        env.close()
        conn.close()


class ParallelRolloutCollector:
    def __init__(self, num_workers, policy_class_name, policy_kwargs, start_method="spawn"):
        if num_workers < 1:
            raise ValueError("--num-workers must be at least 1.")

        self.ctx = mp.get_context(start_method)
        self.parents = []
        self.processes = []

        for _ in range(num_workers):
            parent_conn, child_conn = self.ctx.Pipe()
            process = self.ctx.Process(
                target=rollout_worker_main,
                args=(child_conn, policy_class_name, policy_kwargs),
                daemon=True,
            )
            process.start()
            child_conn.close()
            self.parents.append(parent_conn)
            self.processes.append(process)

    def collect(
        self,
        policy_state_dict,
        opponent_id,
        rollout_len,
        gamma,
        lam,
        archive_entries,
        archive_reset_prob,
        archive_alpha,
        capture_states,
    ):
        chunk_lengths = split_rollout_len(rollout_len, len(self.parents))
        cpu_state_dict = {
            key: value.detach().cpu().numpy()
            for key, value in policy_state_dict.items()
        }

        active = []
        for idx, (conn, chunk_len) in enumerate(zip(self.parents, chunk_lengths)):
            if chunk_len <= 0:
                continue
            conn.send(
                {
                    "command": "collect",
                    "policy_state_dict": cpu_state_dict,
                    "opponent_id": opponent_id,
                    "rollout_len": chunk_len,
                    "gamma": gamma,
                    "lam": lam,
                    "archive_entries": archive_entries,
                    "archive_reset_prob": archive_reset_prob,
                    "archive_alpha": archive_alpha,
                    "capture_states": capture_states,
                    "seed": random.randrange(0, 2**31 - 1),
                }
            )
            active.append(conn)

        results = []
        for conn in active:
            response = conn.recv()
            if not response.get("ok"):
                raise RuntimeError(f"Worker rollout failed:\n{response.get('error', 'unknown error')}")
            results.append(response["result"])
        return merge_worker_results(results)

    def close(self):
        for conn in self.parents:
            try:
                conn.send({"command": "close"})
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()


def split_rollout_len(total, num_parts):
    base = total // num_parts
    remainder = total % num_parts
    return [base + (1 if idx < remainder else 0) for idx in range(num_parts)]


def merge_worker_results(results):
    if not results:
        raise ValueError("No rollout results to merge.")

    obs_chunks = [torch.tensor(result["obs"], dtype=torch.float32) for result in results if len(result["obs"]) > 0]
    action_keys = list(results[0]["acts"].keys())
    merged = {
        "obs": torch.cat(obs_chunks, dim=0),
        "acts": {key: [] for key in action_keys},
        "logp": [],
        "rew": [],
        "done": [],
        "old_values": [],
        "adv": torch.cat([torch.tensor(result["adv"], dtype=torch.float32) for result in results], dim=0),
        "ret": torch.cat([torch.tensor(result["ret"], dtype=torch.float32) for result in results], dim=0),
        "episode_scores": [],
        "archived_reset_episodes": 0,
        "total_episodes": 0,
    }

    has_manager_data = any("manager_data" in result for result in results)
    manager_data = None
    if has_manager_data:
        manager_data = {
            "raw_goals": [],
            "m_logp": [],
            "m_adv": [],
            "m_ret": [],
            "m_old_values": [],
        }

    for result in results:
        for key in action_keys:
            merged["acts"][key].extend(result["acts"][key])
        merged["logp"].extend(result["logp"])
        merged["rew"].extend(result["rew"])
        merged["done"].extend(result["done"])
        merged["old_values"].extend(result["old_values"])
        merged["episode_scores"].extend(result["episode_scores"])
        merged["archived_reset_episodes"] += result["archived_reset_episodes"]
        merged["total_episodes"] += result["total_episodes"]

        if "sim_states" in result:
            merged.setdefault("sim_states", []).extend(result["sim_states"])
            merged.setdefault("episode_steps", []).extend(result["episode_steps"])
            merged.setdefault("td_errors", []).extend(result["td_errors"])

        if manager_data is not None and "manager_data" in result:
            manager_data["raw_goals"].extend(
                torch.tensor(goal, dtype=torch.float32)
                for goal in result["manager_data"]["raw_goals"]
            )
            manager_data["m_logp"].extend(
                torch.tensor(logp, dtype=torch.float32)
                for logp in result["manager_data"]["m_logp"]
            )
            manager_data["m_adv"].append(torch.tensor(result["manager_data"]["m_adv"], dtype=torch.float32))
            manager_data["m_ret"].append(torch.tensor(result["manager_data"]["m_ret"], dtype=torch.float32))
            manager_data["m_old_values"].extend(result["manager_data"]["m_old_values"])

    if manager_data is not None:
        manager_data["m_adv"] = torch.cat(manager_data["m_adv"], dim=0)
        manager_data["m_ret"] = torch.cat(manager_data["m_ret"], dim=0)
        merged["manager_data"] = manager_data

    return merged


def build_archive_snapshot(state_archive, archive_warmup):
    if state_archive is None or len(state_archive) < archive_warmup:
        return None
    return copy.deepcopy(state_archive.entries)


def build_parser():
    parser = argparse.ArgumentParser(description="Train Pen Football agents with multicore rollout collection.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stable_parser = subparsers.add_parser("league-stable", help="Run the stabilized league PPO trainer with multicore rollouts.")
    stable_parser.add_argument("--config", help="Optional YAML config file for this training run.")
    stable_parser.add_argument("--name", required=True, help="Checkpoint folder name to create under ../checkpoints.")
    policy_group = stable_parser.add_mutually_exclusive_group(required=True)
    policy_group.add_argument("--policy-checkpoint", help="Warm-start from an existing checkpoint.")
    policy_group.add_argument("--policy-class", help="Create a fresh policy by class name, e.g. ActorCriticMLPPolicy.")
    stable_parser.add_argument("--policy-kwargs", default="{}", help="JSON object of kwargs for --policy-class.")
    stable_parser.add_argument("--total-steps", type=int, default=30_000_000)
    stable_parser.add_argument("--rollout-len", type=int, default=2048)
    stable_parser.add_argument("--lr", type=float, default=3e-4)
    stable_parser.add_argument("--lr-end", type=float, default=3e-5)
    stable_parser.add_argument("--epochs", type=int, default=10)
    stable_parser.add_argument("--batch-size", type=int, default=256)
    stable_parser.add_argument("--pool-size", type=int, default=200)
    stable_parser.add_argument("--print-every", type=int, default=10_000)
    stable_parser.add_argument("--save-every", type=int, default=1_000_000)
    stable_parser.add_argument("--promotion-games", type=int, default=40)
    stable_parser.add_argument("--benchmark-games", type=int, default=20)
    stable_parser.add_argument("--historical-eval-opponents", type=int, default=4)
    stable_parser.add_argument("--historical-eval-games", type=int, default=10)
    stable_parser.add_argument("--promotion-threshold", type=float, default=0.55)
    stable_parser.add_argument("--benchmark-threshold", type=float, default=0.50)
    stable_parser.add_argument("--historical-threshold", type=float, default=0.50)
    stable_parser.add_argument("--target-kl", type=float, default=0.02)
    stable_parser.add_argument("--max-grad-norm", type=float, default=0.5)
    stable_parser.add_argument("--vf-clip-ratio", type=float, default=0.2)
    stable_parser.add_argument("--ent-coef-start", type=float, default=0.01)
    stable_parser.add_argument("--ent-coef-end", type=float, default=0.001)
    stable_parser.add_argument("--fixed-benchmarks", nargs="*", default=["dummy", "atul"])
    stable_parser.add_argument("--seed-pool", nargs="*", default=[], help="Optional list of checkpoint paths to seed the historical pool.")
    stable_parser.add_argument("--fixed-opponent-prob", type=float, default=0.20)
    stable_parser.add_argument("--champion-prob", type=float, default=0.15)
    stable_parser.add_argument("--recent-prob", type=float, default=0.15)
    stable_parser.add_argument("--easy-opponent-prob", type=float, default=0.10)
    stable_parser.add_argument("--pfsp-prob", type=float, default=0.20)
    stable_parser.add_argument("--hard-prob", type=float, default=0.20)
    stable_parser.add_argument("--easy-threshold", type=float, default=0.70)
    stable_parser.add_argument("--hard-threshold", type=float, default=0.30)
    stable_parser.add_argument(
        "--historical-eval-stratified",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use stratified easy/balanced/hard opponent buckets for promotion-time historical evaluation.",
    )
    stable_parser.add_argument("--save-rejected-checkpoints", action="store_true")
    stable_parser.add_argument("--archive-reset-prob", type=float, default=0.30)
    stable_parser.add_argument("--archive-capacity", type=int, default=5000)
    stable_parser.add_argument("--archive-alpha", type=float, default=0.7)
    stable_parser.add_argument("--archive-min-priority", type=float, default=0.05)
    stable_parser.add_argument("--archive-warmup", type=int, default=500)
    stable_parser.add_argument("--archive-min-step", type=int, default=15)
    stable_parser.add_argument("--archive-stride", type=int, default=4)
    stable_parser.add_argument("--elo-sync-db", default=None, help="Optional eval DB to update automatically on promotion.")
    stable_parser.add_argument("--elo-target-games", type=int, default=0, help="If > 0, also run missing eval games up to this per pair.")
    stable_parser.add_argument("--elo-deterministic", action="store_true", help="Use deterministic actions for automatic eval sync.")
    stable_parser.add_argument("--elo-max-steps", type=int, default=600)
    stable_parser.add_argument("--elo-report-dir", default=None, help="Optional eval report dir to refresh after promotion sync.")
    stable_parser.add_argument("--elo-device", default="cpu")
    stable_parser.add_argument("--elo-seed", type=int, default=None)
    stable_parser.add_argument("--elo-num-workers", type=int, default=1, help="Number of parallel workers to use for training-time Elo sync.")
    stable_parser.add_argument("--num-workers", type=int, default=2, help="Number of rollout worker processes to use.")

    return parser


def extract_config_path(argv):
    for idx, token in enumerate(argv):
        if token == "--config":
            if idx + 1 >= len(argv):
                raise ValueError("--config requires a path.")
            return argv[idx + 1]
        if token.startswith("--config="):
            return token.split("=", 1)[1]
    return None


def find_command_token(argv):
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == "--config":
            idx += 2
            continue
        if token.startswith("--config="):
            idx += 1
            continue
        if not token.startswith("-"):
            return token
        idx += 1
    return None


def load_yaml_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {config_path}")
    return config


def get_subparser_for_command(parser, command_name):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            if command_name in action.choices:
                return action.choices[command_name]
    return None


def option_explicitly_set(argv, action):
    for opt in action.option_strings:
        if opt in argv:
            return True
        if opt.startswith("--") and any(token.startswith(opt + "=") for token in argv):
            return True
    return False


def maybe_json_encode_policy_kwargs(config):
    if "policy_kwargs" in config and isinstance(config["policy_kwargs"], dict):
        config = dict(config)
        config["policy_kwargs"] = json.dumps(config["policy_kwargs"])
    return config


def apply_config_defaults(parser, argv):
    config_path = extract_config_path(argv)
    if config_path is None:
        return argv

    config = maybe_json_encode_policy_kwargs(load_yaml_config(config_path))
    argv_command = find_command_token(argv)
    subparser = get_subparser_for_command(parser, argv_command) if argv_command is not None else None
    command_name = argv_command if subparser is not None else config.get("command")
    if command_name is None:
        raise ValueError("No training command provided. Pass a command such as 'league-stable' or set 'command' in the YAML config.")

    if subparser is None:
        argv = [command_name] + argv

    subparser = get_subparser_for_command(parser, command_name)
    if subparser is None:
        raise ValueError(f"Unknown command in config: {command_name}")

    for action in subparser._actions:
        if not action.dest or action.dest == "help":
            continue
        if action.dest not in config:
            continue
        if option_explicitly_set(argv, action):
            continue
        parser.set_defaults(**{action.dest: config[action.dest]})
        subparser.set_defaults(**{action.dest: config[action.dest]})
        action.required = False

    for group in subparser._mutually_exclusive_groups:
        for action in group._group_actions:
            if action.dest in config or option_explicitly_set(argv, action):
                group.required = False
                break

    return argv


def sync_promoted_checkpoint_to_eval_multicore(
    checkpoint_path,
    db_path,
    target_games=0,
    deterministic=False,
    max_steps=600,
    report_dir=None,
    device="cpu",
    run_tag=None,
    seed=None,
    num_workers=1,
):
    from eval_ratings import (
        add_builtin_model,
        add_checkpoint_model,
        connect_db,
        export_reports,
        load_models,
        run_scheduled_matches,
        schedule_missing_pairs,
    )

    conn = connect_db(db_path)
    for builtin in ("dummy", "atul"):
        add_builtin_model(conn, builtin)
    add_checkpoint_model(conn, checkpoint_path)

    models = load_models(conn)
    schedule = schedule_missing_pairs(
        conn,
        models,
        target_games=target_games,
        deterministic=deterministic,
        max_steps=max_steps,
    )

    if schedule:
        run_scheduled_matches(
            conn,
            schedule,
            deterministic=deterministic,
            max_steps=max_steps,
            device_name=device,
            run_tag=run_tag,
            seed=seed,
            num_workers=num_workers,
        )

    if report_dir:
        export_reports(conn, report_dir, deterministic, max_steps)
    conn.close()

    return len(schedule)


def train_league_ppo_real_multicore(
    name,
    policy,
    total_steps=3_000_000,
    rollout_len=2048,
    lr=3e-4,
    gamma=0.99,
    lam=0.95,
    epochs=10,
    batch_size=256,
    pool_size=10000,
    print_every=10_000,
    save_every=50_000,
    eval_win_window=20,
    opponent_pool=None,
    fixed_benchmarks=("dummy", "atul"),
    fixed_opponent_prob=0.20,
    champion_prob=0.15,
    recent_prob=0.15,
    easy_opponent_prob=0.10,
    pfsp_prob=0.20,
    hard_prob=0.20,
    easy_threshold=0.70,
    hard_threshold=0.30,
    recent_window=12,
    promotion_games=40,
    benchmark_games=20,
    historical_eval_opponents=4,
    historical_eval_games=10,
    historical_eval_stratified=True,
    promotion_threshold=0.55,
    benchmark_threshold=0.50,
    historical_threshold=0.50,
    target_kl=0.02,
    max_grad_norm=0.5,
    vf_clip_ratio=0.2,
    ent_coef_start=0.01,
    ent_coef_end=0.001,
    lr_end=3e-5,
    max_round_steps=600,
    save_rejected_checkpoints=False,
    archive_reset_prob=0.30,
    archive_capacity=5000,
    archive_alpha=0.7,
    archive_min_priority=0.05,
    archive_warmup=500,
    archive_min_step=15,
    archive_stride=4,
    elo_sync_db=None,
    elo_target_games=0,
    elo_deterministic=False,
    elo_max_steps=600,
    elo_report_dir=None,
    elo_device="cpu",
    elo_seed=None,
    elo_num_workers=1,
    num_workers=2,
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    checkpoint_dir = os.path.join(parent_dir, "checkpoints", name)
    os.makedirs(checkpoint_dir, exist_ok=False)

    policy_red, policy_kwargs = load_or_create_policy(policy)
    optimizer = optim.Adam(policy_red.parameters(), lr=lr)
    win_history = defaultdict(lambda: deque(maxlen=eval_win_window))

    fixed_opponents = {}
    if "dummy" in fixed_benchmarks:
        fixed_opponents["dummy"] = DummyPolicy()
        fixed_opponents["dummy"].eval()
    if "atul" in fixed_benchmarks:
        fixed_opponents["atul"] = atulPolicy()
        fixed_opponents["atul"].eval()

    policy_cache = {}
    historical_pool = []
    if opponent_pool is None:
        opponent_pool = []
    for path in opponent_pool:
        if os.path.exists(path):
            historical_pool.append(path)
        else:
            print(f"[WARN] Skipping missing opponent checkpoint: {path}")

    champion_path = policy if isinstance(policy, str) and os.path.exists(policy) else None
    if champion_path is not None and champion_path not in historical_pool:
        historical_pool.append(champion_path)

    recent_promotions = deque(maxlen=recent_window)
    if champion_path is not None:
        recent_promotions.append(champion_path)

    state_archive = None
    if archive_reset_prob > 0.0 and archive_capacity > 0:
        state_archive = StateArchive(
            capacity=archive_capacity,
            alpha=archive_alpha,
            min_priority=archive_min_priority,
        )

    def mean_score_against(identifier):
        history = win_history.get(identifier)
        if history is None or len(history) == 0:
            return 0.5
        return float(sum(history)) / len(history)

    def record_episode_scores(identifier, episode_scores):
        if identifier is None:
            return
        for score in episode_scores:
            win_history[identifier].append(float(score))

    def trim_historical_pool():
        while len(historical_pool) > pool_size:
            removable = None
            for candidate_path in historical_pool:
                if candidate_path != champion_path:
                    removable = candidate_path
                    break
            if removable is None:
                break
            historical_pool.remove(removable)
            if removable in recent_promotions:
                recent_promotions.remove(removable)
            policy_cache.pop(removable, None)
            win_history.pop(removable, None)

    def split_historical_candidates(candidates):
        easy_candidates = []
        balanced_candidates = []
        hard_candidates = []

        for candidate_path in candidates:
            score = mean_score_against(candidate_path)
            if score >= easy_threshold:
                easy_candidates.append(candidate_path)
            elif score <= hard_threshold:
                hard_candidates.append(candidate_path)
            else:
                balanced_candidates.append(candidate_path)

        return easy_candidates, balanced_candidates, hard_candidates

    def sample_stratified_candidates(candidates, sample_size):
        if sample_size <= 0 or not candidates:
            return []

        easy_candidates, balanced_candidates, hard_candidates = split_historical_candidates(candidates)
        bucket_specs = [
            (hard_candidates, max(1, sample_size // 4)),
            (balanced_candidates, max(1, sample_size // 2)),
            (easy_candidates, max(1, sample_size // 4)),
        ]

        selected = []
        selected_set = set()
        for bucket, target_count in bucket_specs:
            available = [path for path in bucket if path not in selected_set]
            if not available:
                continue
            chosen = random.sample(available, min(target_count, len(available)))
            selected.extend(chosen)
            selected_set.update(chosen)

        if len(selected) < sample_size:
            remaining = [path for path in candidates if path not in selected_set]
            if remaining:
                selected.extend(random.sample(remaining, min(sample_size - len(selected), len(remaining))))

        if len(selected) > sample_size:
            selected = selected[:sample_size]
        return selected

    def select_opponent_identifier():
        categories = []
        historical_candidates = [path for path in historical_pool if path != champion_path]
        recent_candidates = [path for path in recent_promotions if path != champion_path]
        easy_candidates, balanced_candidates, hard_candidates = split_historical_candidates(historical_candidates)

        if fixed_opponents:
            categories.append(("fixed", fixed_opponent_prob))
        if champion_path is not None:
            categories.append(("champion", champion_prob))
        if recent_candidates:
            categories.append(("recent", recent_prob))
        if easy_candidates:
            categories.append(("easy", easy_opponent_prob))
        if balanced_candidates:
            categories.append(("pfsp", pfsp_prob))
        if hard_candidates:
            categories.append(("hard", hard_prob))
        elif historical_candidates:
            categories.append(("hard", hard_prob))

        if not categories:
            return "builtin:dummy" if "dummy" in fixed_opponents else None

        names = [name for name, _ in categories]
        probs = [prob for _, prob in categories]
        category = sample_weighted(names, probs)

        if category == "fixed":
            builtin_name = random.choice(list(fixed_opponents.keys()))
            return f"builtin:{builtin_name}"
        if category == "champion":
            return champion_path
        if category == "recent":
            return random.choice(recent_candidates)
        if category == "easy":
            weights = [max(1e-3, mean_score_against(path)) for path in easy_candidates]
            return sample_weighted(easy_candidates, weights)
        if category == "pfsp":
            candidates = balanced_candidates if balanced_candidates else historical_candidates
            weights = [pfsp_weight(mean_score_against(path)) for path in candidates]
            return sample_weighted(candidates, weights)
        if category == "hard":
            candidates = hard_candidates if hard_candidates else historical_candidates
            weights = [hard_weight(mean_score_against(path)) for path in candidates]
            return sample_weighted(candidates, weights)
        return None

    def evaluate_for_promotion(step):
        candidate_policy = make_policy(policy_red.__class__.__name__, **policy_kwargs)
        candidate_policy.load_state_dict(copy.deepcopy(policy_red.state_dict()))
        candidate_policy.eval()

        summary = {
            "step": step,
            "overall_score": 0.5,
            "champion_score": None,
            "benchmark_scores": {},
            "historical_score": None,
            "promoted": False,
        }

        suite_scores = []

        if champion_path is not None:
            champion_policy = load_policy_cached(champion_path, fixed_opponents, policy_cache)
            champion_result = evaluate_matchup(
                candidate_policy,
                champion_policy,
                games=promotion_games,
                max_steps=max_round_steps,
            )
            summary["champion_score"] = champion_result["mean_score"]
            suite_scores.append(champion_result["mean_score"])

        for benchmark_name in fixed_opponents:
            result = evaluate_matchup(
                candidate_policy,
                fixed_opponents[benchmark_name],
                games=benchmark_games,
                max_steps=max_round_steps,
            )
            summary["benchmark_scores"][benchmark_name] = result["mean_score"]
            suite_scores.append(result["mean_score"])

        historical_candidates = [path for path in historical_pool if path != champion_path]
        if historical_candidates:
            sample_size = min(historical_eval_opponents, len(historical_candidates))
            if historical_eval_stratified:
                sampled_paths = sample_stratified_candidates(historical_candidates, sample_size)
            else:
                sampled_paths = random.sample(historical_candidates, sample_size)
            historical_scores = []
            for opponent_path in sampled_paths:
                opponent_policy = load_policy_cached(opponent_path, fixed_opponents, policy_cache)
                result = evaluate_matchup(
                    candidate_policy,
                    opponent_policy,
                    games=historical_eval_games,
                    max_steps=max_round_steps,
                )
                historical_scores.append(result["mean_score"])
            summary["historical_score"] = float(np.mean(historical_scores))
            suite_scores.append(summary["historical_score"])

        if suite_scores:
            summary["overall_score"] = float(np.mean(suite_scores))

        passes = True
        if summary["champion_score"] is not None:
            passes &= summary["champion_score"] >= promotion_threshold
        for benchmark_score in summary["benchmark_scores"].values():
            passes &= benchmark_score >= benchmark_threshold
        if summary["historical_score"] is not None:
            passes &= summary["historical_score"] >= historical_threshold
        passes &= summary["overall_score"] >= benchmark_threshold
        summary["promoted"] = passes
        return summary

    collector = ParallelRolloutCollector(
        num_workers=num_workers,
        policy_class_name=policy_red.__class__.__name__,
        policy_kwargs=policy_kwargs,
    )

    steps = 0
    rewards_save = []
    score_save = []
    stats_save = []

    try:
        while steps < total_steps:
            progress = steps / max(total_steps, 1)
            curr_lr = linear_schedule(lr, lr_end, progress)
            curr_ent_coef = linear_schedule(ent_coef_start, ent_coef_end, progress)
            set_optimizer_lr(optimizer, curr_lr)

            opponent_id = select_opponent_identifier()
            if opponent_id is None:
                opponent_id = "builtin:dummy"

            archive_entries = build_archive_snapshot(state_archive, archive_warmup)
            roll = collector.collect(
                policy_state_dict=policy_red.state_dict(),
                opponent_id=opponent_id,
                rollout_len=rollout_len,
                gamma=gamma,
                lam=lam,
                archive_entries=archive_entries,
                archive_reset_prob=archive_reset_prob,
                archive_alpha=archive_alpha,
                capture_states=state_archive is not None,
            )

            record_episode_scores(
                None if opponent_id.startswith("builtin:") else opponent_id,
                roll.get("episode_scores", []),
            )

            if state_archive is not None and "sim_states" in roll:
                for sim_state, td_error, episode_step in zip(
                    roll["sim_states"],
                    roll["td_errors"],
                    roll["episode_steps"],
                ):
                    if episode_step < archive_min_step:
                        continue
                    if archive_stride > 1 and (episode_step % archive_stride != 0):
                        continue
                    state_archive.add(sim_state, td_error)

            steps += rollout_len

            train_metrics = ppo_update(
                policy_red,
                optimizer,
                roll["obs"],
                roll["acts"],
                roll["logp"],
                roll["adv"],
                roll["ret"],
                old_values=roll["old_values"],
                manager_data=roll.get("manager_data"),
                epochs=epochs,
                batch_size=batch_size,
                clip_ratio=0.2,
                ent_coef=curr_ent_coef,
                vf_clip_ratio=vf_clip_ratio,
                target_kl=target_kl,
                max_grad_norm=max_grad_norm,
            )

            rewards_save.append(sum(roll["rew"]) / len(roll["rew"]))
            if roll.get("episode_scores"):
                score_save.append(float(np.mean(roll["episode_scores"])))
            stats_save.append(train_metrics)

            if steps % print_every < rollout_len:
                mean_metrics = {}
                if stats_save:
                    metric_keys = stats_save[0].keys()
                    for key in metric_keys:
                        mean_metrics[key] = float(np.mean([stats[key] for stats in stats_save if key in stats]))

                print(
                    f"[{steps - (steps % print_every)}] "
                    f"mean reward = {sum(rewards_save)/len(rewards_save):.3f} | "
                    f"mean episode score = {np.mean(score_save) if score_save else 0.5:.3f} | "
                    f"archive size = {len(state_archive) if state_archive is not None else 0} | "
                    f"archived reset rate = "
                    f"{(roll.get('archived_reset_episodes', 0) / max(roll.get('total_episodes', 1), 1)):.2f} | "
                    f"lr = {curr_lr:.6f} | ent = {curr_ent_coef:.4f} | "
                    f"approx_kl = {mean_metrics.get('approx_kl', 0.0):.4f} | "
                    f"clipfrac = {mean_metrics.get('clip_fraction', 0.0):.3f}"
                )
                rewards_save = []
                score_save = []
                stats_save = []

            if steps % save_every < rollout_len:
                checkpoint_step = steps - (steps % save_every)
                summary = evaluate_for_promotion(checkpoint_step)

                if summary["promoted"]:
                    save_path = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_step}.pth")
                    save_policy_checkpoint(save_path, policy_red, policy_kwargs)
                    champion_path = save_path
                    policy_cache[save_path] = load_policy_cached(save_path, fixed_opponents, policy_cache)
                    historical_pool.append(save_path)
                    recent_promotions.append(save_path)
                    trim_historical_pool()

                    eval_schedule_len = None
                    if elo_sync_db is not None:
                        eval_schedule_len = sync_promoted_checkpoint_to_eval_multicore(
                            save_path,
                            db_path=elo_sync_db,
                            target_games=elo_target_games,
                            deterministic=elo_deterministic,
                            max_steps=elo_max_steps,
                            report_dir=elo_report_dir,
                            device=elo_device,
                            run_tag=f"promotion_{checkpoint_step}",
                            seed=elo_seed,
                            num_workers=elo_num_workers,
                        )

                    print(
                        f"Promoted checkpoint at step {checkpoint_step} | "
                        f"overall = {summary['overall_score']:.3f} | "
                        f"champion = {summary['champion_score'] if summary['champion_score'] is not None else float('nan'):.3f} | "
                        f"benchmarks = {summary['benchmark_scores']} | "
                        f"historical = {summary['historical_score'] if summary['historical_score'] is not None else float('nan'):.3f}"
                        + (
                            f" | elo scheduled matches = {eval_schedule_len}"
                            if eval_schedule_len is not None else ""
                        )
                    )
                else:
                    if save_rejected_checkpoints:
                        save_path = os.path.join(checkpoint_dir, f"candidate_{checkpoint_step}.pth")
                        save_policy_checkpoint(save_path, policy_red, policy_kwargs)
                    print(
                        f"Rejected checkpoint at step {checkpoint_step} | "
                        f"overall = {summary['overall_score']:.3f} | "
                        f"champion = {summary['champion_score'] if summary['champion_score'] is not None else float('nan'):.3f} | "
                        f"benchmarks = {summary['benchmark_scores']} | "
                        f"historical = {summary['historical_score'] if summary['historical_score'] is not None else float('nan'):.3f}"
                    )
    finally:
        collector.close()


if __name__ == "__main__":
    parser = build_parser()
    argv = apply_config_defaults(parser, sys.argv[1:])
    args = parser.parse_args(argv)

    if args.command == "league-stable":
        policy_spec = parse_policy_spec(args.policy_checkpoint, args.policy_class, args.policy_kwargs)
        train_league_ppo_real_multicore(
            name=args.name,
            policy=policy_spec,
            total_steps=args.total_steps,
            rollout_len=args.rollout_len,
            lr=args.lr,
            lr_end=args.lr_end,
            epochs=args.epochs,
            batch_size=args.batch_size,
            pool_size=args.pool_size,
            print_every=args.print_every,
            save_every=args.save_every,
            promotion_games=args.promotion_games,
            benchmark_games=args.benchmark_games,
            historical_eval_opponents=args.historical_eval_opponents,
            historical_eval_games=args.historical_eval_games,
            promotion_threshold=args.promotion_threshold,
            benchmark_threshold=args.benchmark_threshold,
            historical_threshold=args.historical_threshold,
            target_kl=args.target_kl,
            max_grad_norm=args.max_grad_norm,
            vf_clip_ratio=args.vf_clip_ratio,
            ent_coef_start=args.ent_coef_start,
            ent_coef_end=args.ent_coef_end,
            fixed_benchmarks=tuple(args.fixed_benchmarks),
            opponent_pool=args.seed_pool,
            fixed_opponent_prob=args.fixed_opponent_prob,
            champion_prob=args.champion_prob,
            recent_prob=args.recent_prob,
            easy_opponent_prob=args.easy_opponent_prob,
            pfsp_prob=args.pfsp_prob,
            hard_prob=args.hard_prob,
            easy_threshold=args.easy_threshold,
            hard_threshold=args.hard_threshold,
            save_rejected_checkpoints=args.save_rejected_checkpoints,
            historical_eval_stratified=args.historical_eval_stratified,
            archive_reset_prob=args.archive_reset_prob,
            archive_capacity=args.archive_capacity,
            archive_alpha=args.archive_alpha,
            archive_min_priority=args.archive_min_priority,
            archive_warmup=args.archive_warmup,
            archive_min_step=args.archive_min_step,
            archive_stride=args.archive_stride,
            elo_sync_db=args.elo_sync_db,
            elo_target_games=args.elo_target_games,
            elo_deterministic=args.elo_deterministic,
            elo_max_steps=args.elo_max_steps,
            elo_report_dir=args.elo_report_dir,
            elo_device=args.elo_device,
            elo_seed=args.elo_seed,
            elo_num_workers=args.elo_num_workers,
            num_workers=args.num_workers,
        )
