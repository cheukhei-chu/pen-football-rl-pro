import argparse
import copy
import json
import os
import random
from collections import deque, defaultdict

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

from multiagent import FootballMultiAgentEnv
from policy import *


def result_to_score(result):
    if result == "red":
        return 1.0
    if result == "blue":
        return 0.0
    if result == "draw":
        return 0.5
    return None


def load_or_create_policy(policy):
    if isinstance(policy, tuple):
        pname, kwargs = policy
        return make_policy(pname, **kwargs), kwargs

    loaded_policy, checkpoint = policy_from_checkpoint_path(policy)
    return loaded_policy, checkpoint.get("policy_kwargs", {})

###############################################################
# =======================  GAE  ============================= #
###############################################################

def compute_gae(rewards, values, dones, last_val, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_value = last_val if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - float(dones[t])

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    return advantages


###############################################################
# ====================  PPO LOSS  =========================== #
###############################################################

def ppo_loss(
    policy,
    obs,
    actions,
    old_logps,
    advantages,
    returns,
    old_values=None,
    clip_ratio=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    vf_clip_ratio=0.2,
):

    logits = policy.forward(obs)

    logps = []
    entropies = []

    for k in ["left", "right", "jump"]:
        dist = torch.distributions.Categorical(logits=logits[k])
        logps.append(dist.log_prob(actions[k]))
        entropies.append(dist.entropy())

    logp = sum(logps)                # shape (batch,)
    entropy = sum(entropies)         # shape (batch,)

    # Ensure old_logps, advantages, returns are tensors of matching shape
    # ratio shape: (batch,)
    log_ratio = logp - old_logps
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    value_pred = logits["value"].squeeze(-1)
    if old_values is not None and vf_clip_ratio is not None:
        value_pred_clipped = old_values + torch.clamp(
            value_pred - old_values,
            -vf_clip_ratio,
            vf_clip_ratio,
        )
        value_loss_unclipped = (returns - value_pred) ** 2
        value_loss_clipped = (returns - value_pred_clipped) ** 2
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
    else:
        value_loss = 0.5 * ((returns - value_pred) ** 2).mean()

    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()
    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
    clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean().item()
    metrics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.mean().item(),
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }
    return loss, metrics


###############################################################
# ===================  ROLLOUT CODE  ======================== #
###############################################################

def rollout(env, policy_red, policy_blue, select_drill,
            rollout_len=2048, gamma=0.99, lam=0.95):

    obs_list = []
    act_list = {"left": [], "right": [], "jump": []}
    logp_list = []
    rew_list = []
    done_list = []
    val_list = []
    episode_scores = []

    steps = 0
    obs = None

    # NEW: Extra storage for Manager data (if Feudal)
    m_rew_list, m_val_list, m_logp_list, raw_goal_list = [], [], [], []
    is_feudal = hasattr(policy_red, "_update_manager")

    while steps < rollout_len:

        # ------ NEW EPISODE ------
        drill = select_drill()
        env.set_setting(drill)
        policy_red.set_setting(drill)

        if is_feudal:
            policy_red.reset_state()

        obs, _ = env.reset()

        done = False

        while not done and steps < rollout_len:

            # observation for red
            obs_tensor = torch.tensor(obs["player_red"], dtype=torch.float32).unsqueeze(0)

            if is_feudal:
                policy_red._update_manager(obs_tensor)
                # Capture the PPO data generated during this tick
                curr_raw_goal = policy_red.last_raw_goal.clone().detach()
                curr_m_logp   = policy_red.last_manager_log_prob.clone().detach()
                # We need Manager's value estimate for GAE
                _, _, m_val_t = policy_red.evaluate_manager(obs_tensor, curr_raw_goal)
                m_val = m_val_t.item()

            logits = policy_red.forward(obs_tensor)
            value = logits["value"].item()

            # sample red action
            a = {
                k: torch.distributions.Categorical(logits=logits[k]).sample().item()
                for k in ["left", "right", "jump"]
            }

            # compute log probability
            logp = 0.0
            for k in ["left", "right", "jump"]:
                dist = torch.distributions.Categorical(logits=logits[k])
                logp += dist.log_prob(torch.tensor(a[k])).item()

            # environment step
            next_obs, rewards, terminated, truncated, info = env.step({
                "player_red": a,
                "player_blue": policy_blue.sample_action(obs["player_blue"]),
            })

            done = terminated["__all__"] or truncated["__all__"]

            r_ext = rewards["player_red"]
            r_worker = r_ext

            if is_feudal:
                # Worker gets Mixed (Extrinsic + Intrinsic)
                r_int = policy_red.compute_intrinsic_reward(next_obs["player_red"])
                r_worker += r_int

                # Manager gets Extrinsic Only
                m_rew_list.append(r_ext)
                m_val_list.append(m_val)
                raw_goal_list.append(curr_raw_goal)
                m_logp_list.append(curr_m_logp)

            # store transition
            obs_list.append(obs_tensor)            # list of (1, obs_dim) tensors
            for k in a:
                act_list[k].append(a[k])           # list of ints
            logp_list.append(logp)                 # list of floats
            rew_list.append(r_worker)              # list of floats
            done_list.append(done)                 # list of bools
            val_list.append(value)                 # list of floats

            steps += 1
            obs = next_obs

            if done:
                score = result_to_score(info.get("result"))
                if score is not None:
                    episode_scores.append(score)

            # if we've reached rollout_len exactly mid-episode, break cleanly
            if steps >= rollout_len:
                break

        # episode ends here — loop automatically restarts new drill

    # ------ BOOTSTRAP VALUE ------
    with torch.no_grad():
        w_last_val = policy_red.forward(
            torch.tensor(obs["player_red"], dtype=torch.float32).unsqueeze(0)
        )["value"].item()

    # NEW: Manager Bootstrap
    m_last_val = 0.0
    if is_feudal:
        with torch.no_grad():
            dummy_g = torch.zeros(1, policy_red.goal_dim, device=obs_tensor.device)
            _, _, m_last_val_t = policy_red.evaluate_manager(torch.tensor(obs["player_red"], dtype=torch.float32).unsqueeze(0), dummy_g)
            m_last_val = m_last_val_t.item()

    result = {
        "obs": obs_list, "acts": act_list, "logp": logp_list,
        "rew": rew_list, "val": val_list, "done": done_list,
        "last_val": w_last_val, "episode_scores": episode_scores
    }

    # NEW: Return Manager data if it exists
    if is_feudal:
        result.update({
            "m_rew": m_rew_list, "m_val": m_val_list,
            "m_logp": m_logp_list, "raw_goals": raw_goal_list,
            "m_last_val": m_last_val
        })

    return result


###############################################################
# ===================  PPO UPDATE  ========================== #
###############################################################

def ppo_update(
    policy,
    optimizer,
    obs,
    actions,
    old_logps,
    advantages,
    returns,
    old_values=None,
    manager_data=None,
    epochs=10,
    batch_size=64,
    clip_ratio=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    vf_clip_ratio=0.2,
    target_kl=None,
    max_grad_norm=None,
):
    """
    obs: list of (1,obs_dim) tensors OR a stacked tensor (N, obs_dim)
    actions: dict of lists -> will be converted to tensors
    old_logps: list or tensor
    advantages, returns: tensors or lists
    """

    # ---------- Convert/stack observations ----------
    if isinstance(obs, list):
        obs = torch.cat(obs, dim=0)            # (N, obs_dim)
    # else assume obs is already a tensor of shape (N, obs_dim)

    # ---------- Convert actions and other lists to tensors ----------
    actions = {
        k: torch.tensor(v, dtype=torch.long)
        for k, v in actions.items()
    }

    old_logps = torch.tensor(old_logps, dtype=torch.float32)
    old_values = None if old_values is None else torch.tensor(old_values, dtype=torch.float32)
    advantages = advantages.clone().detach()
    returns = returns.clone().detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    if manager_data:
        m_goals = torch.cat(manager_data["raw_goals"], dim=0)
        m_old_logps = torch.cat(manager_data["m_logp"], dim=0)
        m_adv = manager_data["m_adv"].clone().detach()
        m_adv = (m_adv - m_adv.mean()) / (m_adv.std() + 1e-8)
        m_ret = manager_data["m_ret"].clone().detach()
        m_old_values = torch.tensor(manager_data["m_old_values"], dtype=torch.float32)

    N = len(returns)
    idxs = np.arange(N)
    metrics_accum = defaultdict(list)
    early_stop = False

    # ---------- PPO training ----------
    for _ in range(epochs):
        np.random.shuffle(idxs)

        for start in range(0, N, batch_size):
            end = start + batch_size
            batch = idxs[start:end]
            if len(batch) == 0:
                continue

            # convert batch indices to torch LongTensor for safe indexing
            batch_idx = torch.tensor(batch, dtype=torch.long)

            obs_b = obs[batch_idx]
            act_b = {k: v[batch_idx] for k, v in actions.items()}
            old_log_b = old_logps[batch_idx]
            adv_b = advantages[batch_idx]
            ret_b = returns[batch_idx]
            old_val_b = None if old_values is None else old_values[batch_idx]

            loss, loss_metrics = ppo_loss(
                policy,
                obs_b,
                act_b,
                old_log_b,
                adv_b,
                ret_b,
                old_values=old_val_b,
                clip_ratio=clip_ratio,
                vf_coef=vf_coef,
                ent_coef=ent_coef,
                vf_clip_ratio=vf_clip_ratio,
            )
            for key, value in loss_metrics.items():
                metrics_accum[key].append(value)

            if manager_data:
                m_goals_b = m_goals[batch_idx]
                m_old_log_b = m_old_logps[batch_idx]
                m_adv_b = m_adv[batch_idx]
                m_ret_b = m_ret[batch_idx]
                m_old_val_b = m_old_values[batch_idx]

                # Get new stats
                m_logp_new, m_ent_new, m_val_pred = policy.evaluate_manager(obs_b, m_goals_b)
                m_val_pred = m_val_pred.squeeze(-1)

                # Standard PPO Ratio
                m_log_ratio = m_logp_new - m_old_log_b
                m_ratio = torch.exp(m_log_ratio)
                m_surr1 = m_ratio * m_adv_b
                m_surr2 = torch.clamp(m_ratio, 1-clip_ratio, 1+clip_ratio) * m_adv_b

                m_loss_pi = -torch.min(m_surr1, m_surr2).mean()
                if vf_clip_ratio is not None:
                    m_val_pred_clipped = m_old_val_b + torch.clamp(
                        m_val_pred - m_old_val_b,
                        -vf_clip_ratio,
                        vf_clip_ratio,
                    )
                    m_v_loss_unclipped = (m_ret_b - m_val_pred) ** 2
                    m_v_loss_clipped = (m_ret_b - m_val_pred_clipped) ** 2
                    m_loss_v = 0.5 * torch.max(m_v_loss_unclipped, m_v_loss_clipped).mean()
                else:
                    m_loss_v = 0.5 * ((m_ret_b - m_val_pred) ** 2).mean()
                m_approx_kl = ((m_ratio - 1.0) - m_log_ratio).mean().item()

                # ADD LOSSES TOGETHER
                loss += (m_loss_pi + 0.5 * m_loss_v - 0.01 * m_ent_new.mean())
                metrics_accum["manager_policy_loss"].append(m_loss_pi.item())
                metrics_accum["manager_value_loss"].append(m_loss_v.item())
                metrics_accum["manager_entropy"].append(m_ent_new.mean().item())
                metrics_accum["manager_approx_kl"].append(m_approx_kl)

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                metrics_accum["grad_norm"].append(float(grad_norm))
            optimizer.step()

            if target_kl is not None and loss_metrics["approx_kl"] > target_kl:
                early_stop = True
                break
        if early_stop:
            break

    return {
        key: float(np.mean(values))
        for key, values in metrics_accum.items()
        if len(values) > 0
    }


def reset_policy_state(policy):
    if hasattr(policy, "reset_state"):
        policy.reset_state()


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def linear_schedule(start, end, progress):
    progress = min(max(progress, 0.0), 1.0)
    return start + (end - start) * progress


def save_policy_checkpoint(path, policy, policy_kwargs):
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "policy_class": policy.__class__.__name__,
        "policy_kwargs": policy_kwargs,
    }, path)


def load_policy_cached(identifier, fixed_opponents, policy_cache):
    if identifier in fixed_opponents:
        return fixed_opponents[identifier]

    if identifier not in policy_cache:
        loaded_policy, _ = policy_from_checkpoint_path(identifier)
        loaded_policy.eval()
        policy_cache[identifier] = loaded_policy

    return policy_cache[identifier]


def play_evaluation_round(env, red_policy, blue_policy, max_steps=600):
    reset_policy_state(red_policy)
    reset_policy_state(blue_policy)

    env.set_setting(None)
    obs, _ = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        with torch.no_grad():
            a_red = red_policy.sample_action(obs["player_red"])
            a_blue = blue_policy.sample_action(obs["player_blue"])

        obs, _, terminated, truncated, info = env.step({
            "player_red": a_red,
            "player_blue": a_blue,
        })
        done = terminated["__all__"] or truncated["__all__"]
        steps += 1

        if done:
            return info.get("result", "draw")

    return "draw"


def evaluate_matchup(candidate_policy, opponent_policy, games=20, max_steps=600):
    env = FootballMultiAgentEnv()
    scores = []

    for game_idx in range(games):
        if game_idx % 2 == 0:
            result = play_evaluation_round(env, candidate_policy, opponent_policy, max_steps=max_steps)
            score = result_to_score(result)
        else:
            result = play_evaluation_round(env, opponent_policy, candidate_policy, max_steps=max_steps)
            score = 1.0 - result_to_score(result)

        scores.append(score)

    env.close()
    return {
        "mean_score": float(np.mean(scores)) if scores else 0.5,
        "win_rate": float(np.mean([s == 1.0 for s in scores])) if scores else 0.0,
        "draw_rate": float(np.mean([s == 0.5 for s in scores])) if scores else 0.0,
        "loss_rate": float(np.mean([s == 0.0 for s in scores])) if scores else 0.0,
        "games": games,
    }


def sample_weighted(items, weights):
    total = sum(weights)
    if total <= 0:
        return random.choice(items)
    return random.choices(items, weights=weights, k=1)[0]


def pfsp_weight(score, min_weight=1e-3):
    score = min(max(score, 0.0), 1.0)
    return max(min_weight, 4.0 * score * (1.0 - score))


def hard_weight(score, min_weight=1e-3):
    score = min(max(score, 0.0), 1.0)
    return max(min_weight, 1.0 - score)


def parse_policy_spec(policy_checkpoint, policy_class, policy_kwargs_json):
    if policy_checkpoint:
        return policy_checkpoint

    if not policy_class:
        raise ValueError("Either --policy-checkpoint or --policy-class must be provided.")

    try:
        policy_kwargs = json.loads(policy_kwargs_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --policy-kwargs JSON: {exc}") from exc

    if not isinstance(policy_kwargs, dict):
        raise ValueError("--policy-kwargs must decode to a JSON object.")

    return (policy_class, policy_kwargs)


###############################################################
# ===============  PPO DRILL TRAINING LOOP  ================= #
###############################################################
def train_drill_ppo(name, policy, select_drill,
                    total_steps=3_000_000, rollout_len=4096,
                    lr=3e-4, gamma=0.99, lam=0.95,
                    epochs=10, batch_size=256,
                    print_every=10_000, save_every=100_000):

    env = FootballMultiAgentEnv()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    checkpoint_dir = os.path.join(parent_dir, "checkpoints", name)
    os.makedirs(checkpoint_dir, exist_ok=False)

    policy_red, policy_kwargs = load_or_create_policy(policy)

    policy_blue = DummyPolicy()
    optimizer = optim.Adam(policy_red.parameters(), lr=lr)

    steps = 0
    rewards_save = []

    while steps < total_steps:

        # -------- GET ROLLOUT DATA --------
        roll = rollout(
            env, policy_red, policy_blue, select_drill,
            rollout_len=rollout_len, gamma=gamma, lam=lam
        )

        obs      = roll["obs"]       # list of tensors (1, obs_dim)
        actions  = roll["acts"]      # dict of lists
        old_logps= roll["logp"]      # list of floats
        rewards  = roll["rew"]       # list of floats
        dones    = roll["done"]      # list of bools
        values   = roll["val"]       # list of floats
        last_val = roll["last_val"]  # float

        steps += rollout_len

        # -------- COMPUTE ADV + RETURNS --------
        adv = compute_gae(rewards, values, dones, last_val, gamma, lam)   # tensor (N,)
        ret = adv + torch.tensor(values, dtype=torch.float32)             # tensor (N,)

        # NOTE: DO NOT cat obs here; we pass the list into ppo_update which handles stacking
        ppo_update(
            policy_red, optimizer,
            obs, actions, old_logps, adv, ret,
            old_values=values,
            epochs=epochs, batch_size=batch_size
        )

        rewards_save.append(sum(rewards)/len(rewards))
        if steps % print_every < rollout_len:
            print(f"[{steps - steps % print_every}] PPO update completed | mean reward = {sum(rewards_save)/len(rewards_save):.3f}")
            rewards_save = []


        if steps % save_every < rollout_len:
            save_path = os.path.join(checkpoint_dir, f"checkpoint_{steps - steps % save_every}.pth")
            torch.save({
                "policy_state_dict": policy_red.state_dict(),
                "policy_class": policy_red.__class__.__name__,
                "policy_kwargs": policy_kwargs
            }, save_path)
            print(f"Saved checkpoint to {save_path}")

def train_league_ppo(
    name, policy,
    total_steps=3_000_000, rollout_len=4096,
    lr=3e-4, gamma=0.99, lam=0.95,
    epochs=10, batch_size=256,
    pool_size=20,
    self_play_prob=None,
    print_every=10_000, save_every=50_000
):

    env = FootballMultiAgentEnv()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    checkpoint_dir = os.path.join(parent_dir, "checkpoints", name)
    os.makedirs(checkpoint_dir, exist_ok=False)

    policy_red, policy_kwargs = load_or_create_policy(policy)

    optimizer = optim.Adam(policy_red.parameters(), lr=lr)

    opponent_pool = []

    if self_play_prob is None: self_play_prob = 1/(pool_size+1)
    def select_opponent():
        if len(opponent_pool) == 0:
            return DummyPolicy()

        if random.random() < self_play_prob:
            return policy_red

        opp_path = random.choice(opponent_pool)
        ckpt = torch.load(opp_path, map_location="cpu")

        opponent = make_policy(ckpt["policy_class"], **ckpt["policy_kwargs"])
        opponent.load_state_dict(ckpt["policy_state_dict"])
        opponent.eval()
        return opponent

    steps = 0
    rewards_save = []

    while steps < total_steps:

        policy_blue = select_opponent()

        roll = rollout(
            env,
            policy_red,
            policy_blue,
            select_drill=lambda: None,
            rollout_len=rollout_len,
            gamma=gamma,
            lam=lam
        )

        obs      = roll["obs"]
        actions  = roll["acts"]
        old_logps= roll["logp"]
        rewards  = roll["rew"]
        dones    = roll["done"]
        values   = roll["val"]
        last_val = roll["last_val"]

        steps += rollout_len

        adv = compute_gae(rewards, values, dones, last_val, gamma, lam)
        ret = adv + torch.tensor(values, dtype=torch.float32)

        ppo_update(
            policy_red, optimizer,
            obs, actions, old_logps, adv, ret,
            old_values=values,
            epochs=epochs, batch_size=batch_size
        )

        rewards_save.append(sum(rewards)/len(rewards))
        if steps % print_every < rollout_len:
            print(f"[{steps - (steps % print_every)}] PPO update | mean reward = {sum(rewards_save)/len(rewards_save):.3f}")
            rewards_save = []

        if steps % save_every < rollout_len:
            save_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_{steps - (steps % save_every)}.pth"
            )
            torch.save({
                "policy_state_dict": policy_red.state_dict(),
                "policy_class": policy_red.__class__.__name__,
                "policy_kwargs": policy_kwargs
            }, save_path)

            print(f"Saved checkpoint to {save_path}")

            opponent_pool.append(save_path)
            if len(opponent_pool) > pool_size:
                opponent_pool.pop(0)

def train_league_ppo_real(
    name, policy,
    total_steps=3_000_000, rollout_len=4096,
    lr=3e-4, gamma=0.99, lam=0.95,
    epochs=10, batch_size=256,
    pool_size=10000,
    print_every=10_000, save_every=50_000,
    eval_win_window=20,
    opponent_pool=None,
    fixed_benchmarks=("dummy", "atul"),
    fixed_opponent_prob=0.15,
    champion_prob=0.20,
    recent_prob=0.15,
    pfsp_prob=0.30,
    hard_prob=0.20,
    recent_window=12,
    promotion_games=40,
    benchmark_games=20,
    historical_eval_opponents=4,
    historical_eval_games=10,
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
):
    env = FootballMultiAgentEnv()

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

    def select_opponent_identifier():
        categories = []
        historical_candidates = [p for p in historical_pool if p != champion_path]
        recent_candidates = [p for p in recent_promotions if p != champion_path]

        if fixed_opponents:
            categories.append(("fixed", fixed_opponent_prob))
        if champion_path is not None:
            categories.append(("champion", champion_prob))
        if recent_candidates:
            categories.append(("recent", recent_prob))
        if historical_candidates:
            categories.append(("pfsp", pfsp_prob))
            categories.append(("hard", hard_prob))

        if not categories:
            return "dummy" if "dummy" in fixed_opponents else None

        names = [name for name, _ in categories]
        probs = [prob for _, prob in categories]
        category = sample_weighted(names, probs)

        if category == "fixed":
            return random.choice(list(fixed_opponents.keys()))
        if category == "champion":
            return champion_path
        if category == "recent":
            return random.choice(recent_candidates)
        if category == "pfsp":
            weights = [pfsp_weight(mean_score_against(path)) for path in historical_candidates]
            return sample_weighted(historical_candidates, weights)
        if category == "hard":
            weights = [hard_weight(mean_score_against(path)) for path in historical_candidates]
            return sample_weighted(historical_candidates, weights)

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

        historical_candidates = [p for p in historical_pool if p != champion_path]
        if historical_candidates:
            sample_size = min(historical_eval_opponents, len(historical_candidates))
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

    steps = 0
    rewards_save = []
    score_save = []
    stats_save = []

    while steps < total_steps:
        progress = steps / max(total_steps, 1)
        curr_lr = linear_schedule(lr, lr_end, progress)
        curr_ent_coef = linear_schedule(ent_coef_start, ent_coef_end, progress)
        set_optimizer_lr(optimizer, curr_lr)

        opponent_id = select_opponent_identifier()
        if opponent_id is None:
            policy_blue = DummyPolicy()
        else:
            policy_blue = load_policy_cached(opponent_id, fixed_opponents, policy_cache)

        roll = rollout(
            env,
            policy_red,
            policy_blue,
            select_drill=lambda: None,
            rollout_len=rollout_len,
            gamma=gamma, lam=lam
        )

        obs      = roll["obs"]
        actions  = roll["acts"]
        old_logps= roll["logp"]
        rewards  = roll["rew"]
        dones    = roll["done"]
        values   = roll["val"]
        last_val = roll["last_val"]
        episode_scores = roll.get("episode_scores", [])

        record_episode_scores(opponent_id, episode_scores)

        steps += rollout_len

        adv = compute_gae(rewards, values, dones, last_val, gamma, lam)
        ret = adv + torch.tensor(values, dtype=torch.float32)

        manager_data = None
        if "m_rew" in roll:
            # roll["m_rew"] contains (Extrinsic Only)
            m_adv = compute_gae(roll["m_rew"], roll["m_val"], roll["done"], roll["m_last_val"], gamma, lam)
            m_ret = m_adv + torch.tensor(roll["m_val"], dtype=torch.float32)

            # Pack it up
            manager_data = {
                "raw_goals": roll["raw_goals"],
                "m_logp": roll["m_logp"],
                "m_adv": m_adv,
                "m_ret": m_ret,
                "m_old_values": roll["m_val"],
            }

        train_metrics = ppo_update(
            policy_red, optimizer,
            obs, actions, old_logps, adv, ret,
            old_values=values,
            manager_data=manager_data,
            epochs=epochs,
            batch_size=batch_size,
            clip_ratio=0.2,
            ent_coef=curr_ent_coef,
            vf_clip_ratio=vf_clip_ratio,
            target_kl=target_kl,
            max_grad_norm=max_grad_norm,
        )

        rewards_save.append(sum(rewards)/len(rewards))
        if episode_scores:
            score_save.append(float(np.mean(episode_scores)))
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

                print(
                    f"Promoted checkpoint at step {checkpoint_step} | "
                    f"overall = {summary['overall_score']:.3f} | "
                    f"champion = {summary['champion_score'] if summary['champion_score'] is not None else float('nan'):.3f} | "
                    f"benchmarks = {summary['benchmark_scores']} | "
                    f"historical = {summary['historical_score'] if summary['historical_score'] is not None else float('nan'):.3f}"
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


###############################################################
# ========================= MAIN ============================ #
###############################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pen Football agents.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stable_parser = subparsers.add_parser("league-stable", help="Run the stabilized league PPO trainer.")
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
    stable_parser.add_argument("--save-rejected-checkpoints", action="store_true")

    args = parser.parse_args()

    if args.command == "league-stable":
        policy_spec = parse_policy_spec(args.policy_checkpoint, args.policy_class, args.policy_kwargs)
        train_league_ppo_real(
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
            save_rejected_checkpoints=args.save_rejected_checkpoints,
        )
