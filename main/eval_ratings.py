import argparse
import csv
import multiprocessing as mp
import os
import random
import sqlite3
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing.connection import wait as mp_wait
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from multiagent import FootballMultiAgentEnv
from policy import (
    DummyPolicy,
    atulPolicy,
    deterministic_action_from_logits,
    make_policy,
    policy_from_checkpoint_path,
)


DB_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_key TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_path TEXT,
    policy_class TEXT,
    added_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    red_model_id INTEGER NOT NULL,
    blue_model_id INTEGER NOT NULL,
    red_score REAL NOT NULL,
    result TEXT NOT NULL,
    steps INTEGER NOT NULL,
    deterministic INTEGER NOT NULL,
    max_steps INTEGER NOT NULL,
    seed INTEGER,
    run_tag TEXT,
    played_at REAL NOT NULL,
    FOREIGN KEY(red_model_id) REFERENCES models(id),
    FOREIGN KEY(blue_model_id) REFERENCES models(id)
);

CREATE INDEX IF NOT EXISTS idx_matches_red_blue
ON matches(red_model_id, blue_model_id, deterministic, max_steps);

CREATE INDEX IF NOT EXISTS idx_matches_blue_red
ON matches(blue_model_id, red_model_id, deterministic, max_steps);
"""


Q = np.log(10.0) / 400.0
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELATIVE_PATH_MARKERS = (
    "checkpoints/",
    "results/",
    "configs/",
    "docs/",
    "samples/",
    "main/",
)


@dataclass(frozen=True)
class ModelRecord:
    id: int
    model_key: str
    label: str
    source_type: str
    source_path: Optional[str]
    policy_class: Optional[str]


def connect_db(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(DB_SCHEMA)
    normalize_registered_checkpoint_paths(conn)
    return conn


def normalize_path_separators(path: str) -> str:
    return path.replace("\\", "/")


def infer_repo_relative_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None

    normalized = normalize_path_separators(path)
    if os.path.isabs(path):
        try:
            rel_path = os.path.relpath(path, REPO_ROOT)
        except ValueError:
            rel_path = None
        if rel_path is not None and not rel_path.startswith(".."):
            return normalize_path_separators(rel_path)

    for marker in RELATIVE_PATH_MARKERS:
        idx = normalized.find(marker)
        if idx != -1:
            return normalized[idx:]

    if not os.path.isabs(path):
        return normalized
    return None


def canonical_checkpoint_ref(path: str) -> str:
    rel_path = infer_repo_relative_path(path)
    if rel_path is not None:
        return rel_path
    return normalize_path_separators(os.path.abspath(path))


def resolve_repo_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None

    rel_path = infer_repo_relative_path(path)
    if rel_path is not None:
        candidate = os.path.join(REPO_ROOT, rel_path)
        return os.path.abspath(candidate)

    return os.path.abspath(path)


def normalize_registered_checkpoint_paths(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT id, model_key, source_path
        FROM models
        WHERE source_type = 'checkpoint'
        """
    ).fetchall()

    for row in rows:
        new_key = canonical_checkpoint_ref(row["model_key"] or row["source_path"])
        new_source = canonical_checkpoint_ref(row["source_path"] or row["model_key"])
        if new_key == row["model_key"] and new_source == row["source_path"]:
            continue

        conflict = conn.execute(
            "SELECT id FROM models WHERE model_key = ? AND id != ?",
            (new_key, row["id"]),
        ).fetchone()
        if conflict is not None:
            conn.execute(
                """
                UPDATE models
                SET source_path = ?
                WHERE id = ?
                """,
                (new_source, row["id"]),
            )
            continue

        conn.execute(
            """
            UPDATE models
            SET model_key = ?, source_path = ?
            WHERE id = ?
            """,
            (new_key, new_source, row["id"]),
        )

    conn.commit()


def extract_epoch_from_filename(fname: str) -> Optional[int]:
    base = os.path.basename(fname)
    if "_" not in base:
        return None
    last = base.split("_")[-1]
    token = last.split(".")[0]
    if token.isdigit():
        return int(token)
    return None


def checkpoint_label(path: str) -> str:
    folder = os.path.basename(os.path.dirname(path.rstrip("/\\")))
    fname = os.path.basename(path)
    return f"{folder}_{fname}"


def discover_checkpoints(paths: Sequence[str]) -> List[str]:
    discovered = []
    for path in paths:
        if os.path.isdir(path):
            folder_items = []
            for fname in os.listdir(path):
                if fname.endswith(".pth"):
                    folder_items.append(os.path.join(path, fname))
            folder_items.sort(key=lambda p: (extract_epoch_from_filename(os.path.basename(p)) is None,
                                             extract_epoch_from_filename(os.path.basename(p)) or 0,
                                             os.path.basename(p)))
            discovered.extend(folder_items)
        elif os.path.isfile(path) and path.endswith(".pth"):
            discovered.append(path)
    return discovered


def add_builtin_model(conn: sqlite3.Connection, name: str) -> None:
    if name not in {"dummy", "atul"}:
        raise ValueError(f"Unsupported builtin model: {name}")
    model_key = f"builtin:{name}"
    label = model_key
    policy_class = "DummyPolicy" if name == "dummy" else "atulPolicy"
    conn.execute(
        """
        INSERT INTO models(model_key, label, source_type, source_path, policy_class, added_at)
        VALUES(?, ?, 'builtin', NULL, ?, ?)
        ON CONFLICT(model_key) DO UPDATE SET
            label=excluded.label,
            policy_class=excluded.policy_class
        """,
        (model_key, label, policy_class, time.time()),
    )
    conn.commit()


def add_checkpoint_model(conn: sqlite3.Connection, path: str, label: Optional[str] = None) -> None:
    resolved_path = resolve_repo_path(path)
    if resolved_path is None or not os.path.exists(resolved_path):
        raise FileNotFoundError(path)

    policy, checkpoint = policy_from_checkpoint_path(resolved_path)
    policy_class = checkpoint.get("policy_class") or checkpoint.get("policy_name") or policy.__class__.__name__
    model_label = label or checkpoint_label(resolved_path)
    model_key = canonical_checkpoint_ref(resolved_path)

    conn.execute(
        """
        INSERT INTO models(model_key, label, source_type, source_path, policy_class, added_at)
        VALUES(?, ?, 'checkpoint', ?, ?, ?)
        ON CONFLICT(model_key) DO UPDATE SET
            label=excluded.label,
            source_path=excluded.source_path,
            policy_class=excluded.policy_class
        """,
        (model_key, model_label, model_key, policy_class, time.time()),
    )
    conn.commit()


def load_models(conn: sqlite3.Connection) -> List[ModelRecord]:
    rows = conn.execute(
        """
        SELECT id, model_key, label, source_type, source_path, policy_class
        FROM models
        ORDER BY label
        """
    ).fetchall()
    return [ModelRecord(**dict(row)) for row in rows]


def load_model_map(conn: sqlite3.Connection) -> Dict[int, ModelRecord]:
    return {model.id: model for model in load_models(conn)}


def validate_model_sources(models: Sequence[ModelRecord]) -> None:
    missing = []
    for model in models:
        if model.source_type != "checkpoint":
            continue
        resolved_path = resolve_repo_path(model.source_path)
        if not resolved_path or not os.path.exists(resolved_path):
            missing.append((model.label, model.source_path))

    if missing:
        lines = ["Some registered checkpoint models no longer exist on disk:"]
        for label, path in missing[:20]:
            lines.append(f"- {label}: {path}")
        if len(missing) > 20:
            lines.append(f"... and {len(missing) - 20} more")
        lines.append("Re-register the pool without those entries or use a fresh eval DB.")
        raise FileNotFoundError("\n".join(lines))


def reset_policy_state(policy) -> None:
    if hasattr(policy, "reset_state"):
        policy.reset_state()


def tensorize_obs(obs, device: torch.device) -> torch.Tensor:
    return torch.tensor(np.asarray(obs), dtype=torch.float32, device=device).unsqueeze(0)


def deterministic_action(policy, obs, device: torch.device) -> Dict[str, int]:
    obs_t = tensorize_obs(obs, device)
    with torch.no_grad():
        if hasattr(policy, "_update_manager"):
            policy._update_manager(obs_t)
        logits = policy.forward(obs_t)
        return deterministic_action_from_logits(logits)


def policy_action(policy, obs, deterministic: bool, device: torch.device) -> Dict[str, int]:
    if deterministic:
        return deterministic_action(policy, obs, device)
    return policy.sample_action(obs)


def instantiate_model(record: ModelRecord, device: torch.device):
    if record.source_type == "builtin":
        policy = DummyPolicy() if record.model_key == "builtin:dummy" else atulPolicy()
        policy.to(device)
        policy.eval()
        return policy

    resolved_path = resolve_repo_path(record.source_path)
    policy, _ = policy_from_checkpoint_path(resolved_path)
    policy.to(device)
    policy.eval()
    return policy


def play_match(
    env: FootballMultiAgentEnv,
    red_policy,
    blue_policy,
    deterministic: bool,
    max_steps: int,
    device: torch.device,
) -> Tuple[float, str, int]:
    reset_policy_state(red_policy)
    reset_policy_state(blue_policy)

    env.set_setting(None)
    obs, _ = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        a_red = policy_action(red_policy, obs["player_red"], deterministic, device)
        a_blue = policy_action(blue_policy, obs["player_blue"], deterministic, device)

        obs, _, terminated, truncated, info = env.step({
            "player_red": a_red,
            "player_blue": a_blue,
        })
        done = terminated["__all__"] or truncated["__all__"]
        steps += 1

        if done:
            result = info.get("result", "draw")
            if result == "red":
                return 1.0, result, steps
            if result == "blue":
                return 0.0, result, steps
            return 0.5, result, steps

    return 0.5, "draw", steps


def get_direction_count(
    conn: sqlite3.Connection,
    red_model_id: int,
    blue_model_id: int,
    deterministic: bool,
    max_steps: int,
) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS c
        FROM matches
        WHERE red_model_id = ? AND blue_model_id = ? AND deterministic = ? AND max_steps = ?
        """,
        (red_model_id, blue_model_id, int(deterministic), max_steps),
    ).fetchone()
    return int(row["c"])


def get_pair_total(
    conn: sqlite3.Connection,
    model_a_id: int,
    model_b_id: int,
    deterministic: bool,
    max_steps: int,
) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS c
        FROM matches
        WHERE deterministic = ? AND max_steps = ?
          AND (
            (red_model_id = ? AND blue_model_id = ?)
            OR
            (red_model_id = ? AND blue_model_id = ?)
          )
        """,
        (int(deterministic), max_steps, model_a_id, model_b_id, model_b_id, model_a_id),
    ).fetchone()
    return int(row["c"])


def schedule_missing_pairs(
    conn: sqlite3.Connection,
    models: Sequence[ModelRecord],
    target_games: int,
    deterministic: bool,
    max_steps: int,
) -> List[Tuple[int, int]]:
    schedule = []
    sorted_models = sorted(models, key=lambda model: model.label)

    for idx, model_a in enumerate(sorted_models):
        for model_b in sorted_models[idx + 1:]:
            total = get_pair_total(conn, model_a.id, model_b.id, deterministic, max_steps)
            missing = max(0, target_games - total)
            if missing == 0:
                continue

            red_ab = get_direction_count(conn, model_a.id, model_b.id, deterministic, max_steps)
            red_ba = get_direction_count(conn, model_b.id, model_a.id, deterministic, max_steps)

            for _ in range(missing):
                if red_ab <= red_ba:
                    schedule.append((model_a.id, model_b.id))
                    red_ab += 1
                else:
                    schedule.append((model_b.id, model_a.id))
                    red_ba += 1

    return schedule


def store_match(
    conn: sqlite3.Connection,
    red_model_id: int,
    blue_model_id: int,
    red_score: float,
    result: str,
    steps: int,
    deterministic: bool,
    max_steps: int,
    seed: Optional[int],
    run_tag: Optional[str],
) -> None:
    conn.execute(
        """
        INSERT INTO matches(
            red_model_id, blue_model_id, red_score, result, steps,
            deterministic, max_steps, seed, run_tag, played_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            red_model_id,
            blue_model_id,
            red_score,
            result,
            steps,
            int(deterministic),
            max_steps,
            seed,
            run_tag,
            time.time(),
        ),
    )
    conn.commit()


def run_scheduled_matches(
    conn: sqlite3.Connection,
    schedule: Sequence[Tuple[int, int]],
    deterministic: bool,
    max_steps: int,
    device_name: str,
    run_tag: Optional[str],
    seed: Optional[int],
    num_workers: int = 1,
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if num_workers > 1:
        run_scheduled_matches_parallel(
            conn,
            schedule,
            deterministic=deterministic,
            max_steps=max_steps,
            device_name=device_name,
            run_tag=run_tag,
            seed=seed,
            num_workers=num_workers,
        )
        return

    model_map = load_model_map(conn)
    validate_model_sources(model_map.values())
    device = torch.device(device_name)
    policy_cache: Dict[int, object] = {}
    env = FootballMultiAgentEnv()

    for match_index, (red_id, blue_id) in enumerate(schedule, start=1):
        if red_id not in policy_cache:
            policy_cache[red_id] = instantiate_model(model_map[red_id], device)
        if blue_id not in policy_cache:
            policy_cache[blue_id] = instantiate_model(model_map[blue_id], device)

        red_score, result, steps = play_match(
            env,
            policy_cache[red_id],
            policy_cache[blue_id],
            deterministic=deterministic,
            max_steps=max_steps,
            device=device,
        )
        store_match(
            conn,
            red_id,
            blue_id,
            red_score,
            result,
            steps,
            deterministic,
            max_steps,
            None if seed is None else seed + match_index - 1,
            run_tag,
        )

        if match_index % 25 == 0 or match_index == len(schedule):
            print(f"[EVAL] Completed {match_index}/{len(schedule)} scheduled matches")

    env.close()


def serialize_model_map(model_map: Dict[int, ModelRecord]) -> Dict[int, Dict[str, object]]:
    return {
        model_id: {
            "id": record.id,
            "model_key": record.model_key,
            "label": record.label,
            "source_type": record.source_type,
            "source_path": record.source_path,
            "policy_class": record.policy_class,
        }
        for model_id, record in model_map.items()
    }


def instantiate_model_from_payload(payload: Dict[str, object], device: torch.device):
    record = ModelRecord(**payload)
    return instantiate_model(record, device)


def eval_worker_main(
    conn,
    serialized_model_map: Dict[int, Dict[str, object]],
    deterministic: bool,
    max_steps: int,
    device_name: str,
):
    if deterministic:
        torch.set_grad_enabled(False)

    device = torch.device(device_name)
    env = FootballMultiAgentEnv()
    policy_cache: Dict[int, object] = {}

    try:
        while True:
            message = conn.recv()
            command = message.get("command")

            if command == "close":
                break

            if command != "play":
                raise ValueError(f"Unknown worker command: {command}")

            if message.get("seed") is not None:
                task_seed = int(message["seed"])
                random.seed(task_seed)
                np.random.seed(task_seed)
                torch.manual_seed(task_seed)

            red_id = int(message["red_id"])
            blue_id = int(message["blue_id"])

            if red_id not in policy_cache:
                policy_cache[red_id] = instantiate_model_from_payload(serialized_model_map[red_id], device)
            if blue_id not in policy_cache:
                policy_cache[blue_id] = instantiate_model_from_payload(serialized_model_map[blue_id], device)

            red_score, result, steps = play_match(
                env,
                policy_cache[red_id],
                policy_cache[blue_id],
                deterministic=deterministic,
                max_steps=max_steps,
                device=device,
            )
            conn.send(
                {
                    "ok": True,
                    "red_id": red_id,
                    "blue_id": blue_id,
                    "red_score": red_score,
                    "result": result,
                    "steps": steps,
                    "seed": message.get("seed"),
                }
            )
    except Exception:
        conn.send({"ok": False, "error": traceback.format_exc()})
    finally:
        env.close()
        conn.close()


def run_scheduled_matches_parallel(
    conn: sqlite3.Connection,
    schedule: Sequence[Tuple[int, int]],
    deterministic: bool,
    max_steps: int,
    device_name: str,
    run_tag: Optional[str],
    seed: Optional[int],
    num_workers: int,
) -> None:
    model_map = load_model_map(conn)
    validate_model_sources(model_map.values())
    serialized_model_map = serialize_model_map(model_map)
    ctx = mp.get_context("spawn")
    parent_conns = []
    processes = []

    try:
        for _ in range(num_workers):
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=eval_worker_main,
                args=(child_conn, serialized_model_map, deterministic, max_steps, device_name),
                daemon=True,
            )
            process.start()
            child_conn.close()
            parent_conns.append(parent_conn)
            processes.append(process)

        next_match_index = 0
        completed = 0
        in_flight: Dict[object, int] = {}

        while next_match_index < len(schedule) or in_flight:
            while next_match_index < len(schedule) and len(in_flight) < len(parent_conns):
                conn_worker = next(conn_it for conn_it in parent_conns if conn_it not in in_flight)
                red_id, blue_id = schedule[next_match_index]
                match_seed = None if seed is None else seed + next_match_index
                conn_worker.send(
                    {
                        "command": "play",
                        "red_id": red_id,
                        "blue_id": blue_id,
                        "seed": match_seed,
                    }
                )
                in_flight[conn_worker] = next_match_index
                next_match_index += 1

            ready = mp_wait(list(in_flight.keys()))
            for conn_worker in ready:
                response = conn_worker.recv()
                if not response.get("ok"):
                    raise RuntimeError(f"Eval worker failed:\n{response.get('error', 'unknown error')}")

                store_match(
                    conn,
                    int(response["red_id"]),
                    int(response["blue_id"]),
                    float(response["red_score"]),
                    str(response["result"]),
                    int(response["steps"]),
                    deterministic,
                    max_steps,
                    response.get("seed"),
                    run_tag,
                )
                completed += 1
                del in_flight[conn_worker]

                if completed % 25 == 0 or completed == len(schedule):
                    print(f"[EVAL] Completed {completed}/{len(schedule)} scheduled matches")
    finally:
        for conn_worker in parent_conns:
            try:
                conn_worker.send({"command": "close"})
            except (BrokenPipeError, EOFError):
                pass
        for process in processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()


def aggregate_matches(
    conn: sqlite3.Connection,
    deterministic: bool,
    max_steps: int,
) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT red_model_id, blue_model_id, red_score, steps
        FROM matches
        WHERE deterministic = ? AND max_steps = ?
        ORDER BY id
        """,
        (int(deterministic), max_steps),
    ).fetchall()


def wilson_interval(successes: float, trials: int, z: float = 1.96) -> Tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1.0 + z ** 2 / trials
    center = (p + z ** 2 / (2 * trials)) / denom
    margin = z * np.sqrt((p * (1.0 - p) + z ** 2 / (4 * trials)) / trials) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def compute_sequential_elo(
    models: Sequence[ModelRecord],
    matches: Sequence[sqlite3.Row],
    k_factor: float = 16.0,
) -> Dict[int, float]:
    ratings = {model.id: 1500.0 for model in models}

    for row in matches:
        red_id = row["red_model_id"]
        blue_id = row["blue_model_id"]
        red_score = float(row["red_score"])
        r_red = ratings[red_id]
        r_blue = ratings[blue_id]
        expected_red = 1.0 / (1.0 + 10 ** ((r_blue - r_red) / 400.0))
        expected_blue = 1.0 - expected_red
        ratings[red_id] = r_red + k_factor * (red_score - expected_red)
        ratings[blue_id] = r_blue + k_factor * ((1.0 - red_score) - expected_blue)

    return ratings


def glicko_g(rd: float) -> float:
    return 1.0 / np.sqrt(1.0 + 3.0 * (Q ** 2) * (rd ** 2) / (np.pi ** 2))


def glicko_expected(rating: float, opp_rating: float, opp_rd: float) -> float:
    g = glicko_g(opp_rd)
    return 1.0 / (1.0 + 10 ** (-g * (rating - opp_rating) / 400.0))


def compute_static_glicko(
    models: Sequence[ModelRecord],
    matches: Sequence[sqlite3.Row],
    iterations: int = 20,
) -> Dict[int, Tuple[float, float]]:
    ratings = {model.id: 1500.0 for model in models}
    rds = {model.id: 350.0 for model in models}

    by_player: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for row in matches:
        red_id = row["red_model_id"]
        blue_id = row["blue_model_id"]
        red_score = float(row["red_score"])
        by_player[red_id].append((blue_id, red_score))
        by_player[blue_id].append((red_id, 1.0 - red_score))

    for _ in range(iterations):
        new_ratings = ratings.copy()
        new_rds = rds.copy()
        for model in models:
            games = by_player.get(model.id, [])
            if not games:
                continue

            rating = ratings[model.id]
            rd = rds[model.id]
            denom_term = 0.0
            numer_term = 0.0

            for opp_id, score in games:
                opp_rating = ratings[opp_id]
                opp_rd = rds[opp_id]
                g_val = glicko_g(opp_rd)
                expected = glicko_expected(rating, opp_rating, opp_rd)
                denom_term += (g_val ** 2) * expected * (1.0 - expected)
                numer_term += g_val * (score - expected)

            if denom_term <= 0.0:
                continue

            d2 = 1.0 / ((Q ** 2) * denom_term)
            inv_rd2 = 1.0 / (rd ** 2)
            new_rd = np.sqrt(1.0 / (inv_rd2 + 1.0 / d2))
            new_rating = rating + Q * (new_rd ** 2) * numer_term
            new_ratings[model.id] = float(new_rating)
            new_rds[model.id] = float(new_rd)

        ratings = new_ratings
        rds = new_rds

    return {model.id: (ratings[model.id], rds[model.id]) for model in models}


def logistic(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)


def bt_objective_grad_hessian(
    ratings: np.ndarray,
    match_tuples: Sequence[Tuple[int, int, float]],
    n_models: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    nll = 0.0
    grad = np.zeros(n_models, dtype=np.float64)
    hessian = np.zeros((n_models, n_models), dtype=np.float64)

    for red_idx, blue_idx, red_score in match_tuples:
        eta = Q * (ratings[red_idx] - ratings[blue_idx])
        p_red = logistic(eta)
        p_red = min(max(p_red, 1e-12), 1.0 - 1e-12)

        nll -= red_score * np.log(p_red) + (1.0 - red_score) * np.log(1.0 - p_red)

        grad_term = Q * (p_red - red_score)
        grad[red_idx] += grad_term
        grad[blue_idx] -= grad_term

        weight = (Q ** 2) * p_red * (1.0 - p_red)
        hessian[red_idx, red_idx] += weight
        hessian[blue_idx, blue_idx] += weight
        hessian[red_idx, blue_idx] -= weight
        hessian[blue_idx, red_idx] -= weight

    return nll, grad, hessian


def compute_bradley_terry(
    models: Sequence[ModelRecord],
    matches: Sequence[sqlite3.Row],
    max_iter: int = 50,
    tol: float = 1e-6,
    ridge: float = 1e-6,
) -> Dict[int, Tuple[float, float]]:
    if not models:
        return {}

    id_to_index = {model.id: idx for idx, model in enumerate(models)}
    match_tuples = [
        (id_to_index[row["red_model_id"]], id_to_index[row["blue_model_id"]], float(row["red_score"]))
        for row in matches
    ]

    n_models = len(models)
    ratings = np.zeros(n_models, dtype=np.float64)

    if not match_tuples:
        return {model.id: (1500.0, 350.0) for model in models}

    prev_nll = None
    for _ in range(max_iter):
        nll, grad, hessian = bt_objective_grad_hessian(ratings, match_tuples, n_models)
        solve_hessian = hessian + ridge * np.eye(n_models, dtype=np.float64)

        try:
            delta = np.linalg.solve(solve_hessian, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(solve_hessian, grad, rcond=None)[0]

        step = 1.0
        candidate = ratings - step * delta
        candidate -= np.mean(candidate)
        cand_nll, _, _ = bt_objective_grad_hessian(candidate, match_tuples, n_models)
        while cand_nll > nll and step > 1e-4:
            step *= 0.5
            candidate = ratings - step * delta
            candidate -= np.mean(candidate)
            cand_nll, _, _ = bt_objective_grad_hessian(candidate, match_tuples, n_models)

        ratings = candidate

        if np.max(np.abs(step * delta)) < tol:
            break
        if prev_nll is not None and abs(prev_nll - cand_nll) < tol:
            break
        prev_nll = cand_nll

    _, _, hessian = bt_objective_grad_hessian(ratings, match_tuples, n_models)
    covariance = np.linalg.pinv(hessian)
    std_err = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    results = {}
    for model in models:
        idx = id_to_index[model.id]
        results[model.id] = (1500.0 + ratings[idx], float(std_err[idx]))
    return results


def summarize_pairwise(
    models: Sequence[ModelRecord],
    matches: Sequence[sqlite3.Row],
) -> Tuple[List[Dict[str, object]], Dict[Tuple[int, int], Tuple[float, int]]]:
    model_ids = {model.id for model in models}
    pair_records: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    directed_scores: Dict[Tuple[int, int], List[float]] = defaultdict(list)

    for row in matches:
        red_id = row["red_model_id"]
        blue_id = row["blue_model_id"]
        red_score = float(row["red_score"])
        directed_scores[(red_id, blue_id)].append(red_score)
        directed_scores[(blue_id, red_id)].append(1.0 - red_score)

        pair = tuple(sorted((red_id, blue_id)))
        score_for_lower = red_score if pair[0] == red_id else 1.0 - red_score
        pair_records[pair].append(score_for_lower)

    summary_rows = []
    matrix_entries: Dict[Tuple[int, int], Tuple[float, int]] = {}

    for (id_a, id_b), scores in pair_records.items():
        mean_score = float(np.mean(scores))
        trials = len(scores)
        ci_low, ci_high = wilson_interval(sum(scores), trials)
        summary_rows.append({
            "model_a_id": id_a,
            "model_b_id": id_b,
            "games": trials,
            "mean_score_a": mean_score,
            "ci_low_a": ci_low,
            "ci_high_a": ci_high,
        })

    for id_a in model_ids:
        for id_b in model_ids:
            if id_a == id_b:
                matrix_entries[(id_a, id_b)] = (0.5, 0)
                continue
            scores = directed_scores.get((id_a, id_b), [])
            matrix_entries[(id_a, id_b)] = (
                float(np.mean(scores)) if scores else 0.5,
                len(scores),
            )

    return summary_rows, matrix_entries


def is_anchor_model(model: ModelRecord) -> bool:
    if model.model_key == "builtin:atul":
        return True
    if model.source_path is None:
        return False

    normalized = normalize_path_separators(model.source_path)
    return (
        normalized.startswith("checkpoints/elo_tournament_baseline/")
        or normalized.startswith("checkpoints/elo_tournament_test/")
        or "/checkpoints/elo_tournament_baseline/" in normalized
        or "/checkpoints/elo_tournament_test/" in normalized
    )


def compute_anchor_shift(
    models: Sequence[ModelRecord],
    rating_lookup: Dict[int, float],
    target_mean: float = 1500.0,
) -> Tuple[float, List[ModelRecord]]:
    anchors = [model for model in models if is_anchor_model(model)]
    if not anchors:
        return 0.0, []

    anchor_mean = float(np.mean([rating_lookup[model.id] for model in anchors]))
    return target_mean - anchor_mean, anchors


def export_reports(
    conn: sqlite3.Connection,
    out_dir: str,
    deterministic: bool,
    max_steps: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    models = load_models(conn)
    model_map = {model.id: model for model in models}
    matches = aggregate_matches(conn, deterministic, max_steps)

    raw_elo = compute_sequential_elo(models, matches)
    raw_glicko = compute_static_glicko(models, matches)
    raw_bt = compute_bradley_terry(models, matches)
    pairwise_rows, matrix_entries = summarize_pairwise(models, matches)
    raw_elo_lookup = raw_elo
    raw_glicko_lookup = {model_id: rating for model_id, (rating, _) in raw_glicko.items()}
    raw_bt_lookup = {model_id: rating for model_id, (rating, _) in raw_bt.items()}
    elo_anchor_shift, anchor_models = compute_anchor_shift(models, raw_elo_lookup, target_mean=1500.0)
    glicko_anchor_shift, _ = compute_anchor_shift(models, raw_glicko_lookup, target_mean=1500.0)
    bt_anchor_shift, _ = compute_anchor_shift(models, raw_bt_lookup, target_mean=1500.0)

    ratings_path = os.path.join(out_dir, "ratings.csv")
    with open(ratings_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_key",
            "label",
            "is_anchor",
            "games",
            "bt_rating",
            "raw_bt_rating",
            "bt_std_err",
            "bt_conservative_rating",
            "raw_bt_conservative_rating",
            "elo",
            "raw_elo",
            "glicko_rating",
            "raw_glicko_rating",
            "glicko_rd",
            "conservative_rating",
            "raw_conservative_rating",
            "bt_anchor_shift",
            "elo_anchor_shift",
            "glicko_anchor_shift",
        ])
        games_by_model = defaultdict(int)
        for row in matches:
            games_by_model[row["red_model_id"]] += 1
            games_by_model[row["blue_model_id"]] += 1
        ordered = sorted(
            models,
            key=lambda model: ((raw_bt[model.id][0] + bt_anchor_shift) - 2.0 * raw_bt[model.id][1]),
            reverse=True,
        )
        for model in ordered:
            raw_bt_rating, bt_std_err = raw_bt[model.id]
            raw_bt_cons = raw_bt_rating - 2.0 * bt_std_err
            anchored_bt = raw_bt_rating + bt_anchor_shift
            anchored_bt_cons = raw_bt_cons + bt_anchor_shift

            raw_g_rating, g_rd = raw_glicko[model.id]
            raw_cons = raw_g_rating - 2.0 * g_rd
            anchored_elo = raw_elo[model.id] + elo_anchor_shift
            anchored_glicko = raw_g_rating + glicko_anchor_shift
            anchored_cons = raw_cons + glicko_anchor_shift
            writer.writerow([
                model.model_key,
                model.label,
                int(is_anchor_model(model)),
                games_by_model[model.id],
                f"{anchored_bt:.2f}",
                f"{raw_bt_rating:.2f}",
                f"{bt_std_err:.4f}",
                f"{anchored_bt_cons:.2f}",
                f"{raw_bt_cons:.2f}",
                f"{anchored_elo:.2f}",
                f"{raw_elo[model.id]:.2f}",
                f"{anchored_glicko:.2f}",
                f"{raw_g_rating:.2f}",
                f"{g_rd:.2f}",
                f"{anchored_cons:.2f}",
                f"{raw_cons:.2f}",
                f"{bt_anchor_shift:.2f}",
                f"{elo_anchor_shift:.2f}",
                f"{glicko_anchor_shift:.2f}",
            ])

    metadata_path = os.path.join(out_dir, "ratings_metadata.txt")
    with open(metadata_path, "w") as f:
        f.write("Ratings are anchored so the mean rating of the anchor set is 1500.00.\n")
        f.write(f"Bradley-Terry anchor shift applied: {bt_anchor_shift:.6f}\n")
        f.write(f"Elo anchor shift applied: {elo_anchor_shift:.6f}\n")
        f.write(f"Glicko anchor shift applied: {glicko_anchor_shift:.6f}\n")
        f.write(f"Anchor count: {len(anchor_models)}\n")
        f.write("Anchor models:\n")
        for model in sorted(anchor_models, key=lambda item: item.label):
            f.write(f"- {model.label}\n")

    pairwise_path = os.path.join(out_dir, "pairwise_summary.csv")
    with open(pairwise_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_a", "model_b", "games", "mean_score_a", "ci_low_a", "ci_high_a"])
        for row in sorted(pairwise_rows, key=lambda item: (model_map[item["model_a_id"]].label, model_map[item["model_b_id"]].label)):
            writer.writerow([
                model_map[row["model_a_id"]].label,
                model_map[row["model_b_id"]].label,
                row["games"],
                f"{row['mean_score_a']:.4f}",
                f"{row['ci_low_a']:.4f}",
                f"{row['ci_high_a']:.4f}",
            ])

    matrix_path = os.path.join(out_dir, "payoff_matrix.csv")
    labels = [model.label for model in sorted(models, key=lambda model: model.label)]
    model_by_label = {model.label: model for model in models}
    with open(matrix_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + labels)
        for row_label in labels:
            row_model = model_by_label[row_label]
            row_values = []
            for col_label in labels:
                col_model = model_by_label[col_label]
                score, games = matrix_entries[(row_model.id, col_model.id)]
                value = f"{score:.4f}" if games > 0 or row_model.id == col_model.id else ""
                row_values.append(value)
            writer.writerow([row_label] + row_values)

    counts_path = os.path.join(out_dir, "payoff_counts.csv")
    with open(counts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + labels)
        for row_label in labels:
            row_model = model_by_label[row_label]
            row_values = []
            for col_label in labels:
                col_model = model_by_label[col_label]
                _, games = matrix_entries[(row_model.id, col_model.id)]
                row_values.append(games)
            writer.writerow([row_label] + row_values)

    print(f"[REPORT] Wrote ratings to {ratings_path}")
    print(f"[REPORT] Wrote rating metadata to {metadata_path}")
    print(f"[REPORT] Wrote pairwise summary to {pairwise_path}")
    print(f"[REPORT] Wrote payoff matrix to {matrix_path}")
    print(f"[REPORT] Wrote payoff counts to {counts_path}")


def cmd_sync(args) -> None:
    conn = connect_db(args.db)

    for builtin in args.builtins:
        add_builtin_model(conn, builtin)

    for path in discover_checkpoints(args.inputs):
        add_checkpoint_model(conn, path)

    models = load_models(conn)
    schedule = schedule_missing_pairs(
        conn,
        models,
        target_games=args.target_games,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
    )
    print(f"[SYNC] Registered {len(models)} models")
    print(f"[SYNC] Found {len(schedule)} missing matches to reach {args.target_games} games per pair")

    if schedule:
        run_scheduled_matches(
            conn,
            schedule,
            deterministic=args.deterministic,
            max_steps=args.max_steps,
            device_name=args.device,
            run_tag=args.run_tag,
            seed=args.seed,
            num_workers=args.num_workers,
        )

    if args.report_dir:
        export_reports(conn, args.report_dir, args.deterministic, args.max_steps)

    conn.close()


def cmd_report(args) -> None:
    conn = connect_db(args.db)
    export_reports(conn, args.out_dir, args.deterministic, args.max_steps)
    conn.close()


def cmd_list_models(args) -> None:
    conn = connect_db(args.db)
    models = load_models(conn)
    print(f"{len(models)} models registered:")
    for model in models:
        print(f"- {model.label} [{model.source_type}] {model.model_key}")
    conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robust Pen Football evaluation pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="Register models, run missing matches, and optionally export reports.")
    sync_parser.add_argument("--db", default="results/eval_ratings.sqlite")
    sync_parser.add_argument("--inputs", nargs="*", default=[], help="Checkpoint files or folders to register.")
    sync_parser.add_argument("--builtins", nargs="*", default=["dummy", "atul"])
    sync_parser.add_argument("--target-games", type=int, default=40, help="Target total games per unordered pair.")
    sync_parser.add_argument("--deterministic", action="store_true", help="Use argmax actions instead of sampled actions.")
    sync_parser.add_argument("--max-steps", type=int, default=600)
    sync_parser.add_argument("--device", default="cpu")
    sync_parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel match workers to use.")
    sync_parser.add_argument("--run-tag", default=None)
    sync_parser.add_argument("--seed", type=int, default=None)
    sync_parser.add_argument("--report-dir", default=None, help="If provided, export reports after syncing.")
    sync_parser.set_defaults(func=cmd_sync)

    report_parser = subparsers.add_parser("report", help="Export rating and payoff reports from an existing DB.")
    report_parser.add_argument("--db", default="results/eval_ratings.sqlite")
    report_parser.add_argument("--out-dir", required=True)
    report_parser.add_argument("--deterministic", action="store_true")
    report_parser.add_argument("--max-steps", type=int, default=600)
    report_parser.set_defaults(func=cmd_report)

    list_parser = subparsers.add_parser("list-models", help="List registered models in the database.")
    list_parser.add_argument("--db", default="results/eval_ratings.sqlite")
    list_parser.set_defaults(func=cmd_list_models)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
