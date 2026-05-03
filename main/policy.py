import os
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from abc import ABC, abstractmethod

# -------------------------
# Abstract base policy
# -------------------------
class FootballPolicy(nn.Module, ABC):
    """
    Abstract base class for football policies.
    Subclasses must implement forward() and sample_action().
    forward() must return a dict containing either:
        - "left", "right", "jump": logits (un-normalized) for each discrete action, or
        - "action": logits over a joint discrete action space
    and always:
        - "value": a (batch, 1) tensor of state values
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, obs: torch.Tensor):
        """Given an observation tensor (batch, obs_dim), return dict of logits + value."""
        raise NotImplementedError

    @abstractmethod
    def sample_action(self, obs):
        """Given a single observation (numpy array or torch tensor), return sampled action dict."""
        raise NotImplementedError


# -------------------------
# Small utility for sampling
# -------------------------
def _to_tensor(obs):
    """Convert obs (np array or torch tensor) to a batched torch.float32 tensor on CPU."""
    if isinstance(obs, torch.Tensor):
        t = obs
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t.float()
    else:
        return torch.tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)


ACTION_KEYS = ("left", "right", "jump")
JOINT_ACTIONS = (
    {"left": 0, "right": 0, "jump": 0},
    {"left": 1, "right": 0, "jump": 0},
    {"left": 0, "right": 1, "jump": 0},
    {"left": 0, "right": 0, "jump": 1},
    {"left": 1, "right": 0, "jump": 1},
    {"left": 0, "right": 1, "jump": 1},
)
JOINT_ACTION_LOOKUP = {
    tuple(int(action[key]) for key in ACTION_KEYS): idx
    for idx, action in enumerate(JOINT_ACTIONS)
}


def uses_joint_action(policy_or_logits) -> bool:
    if isinstance(policy_or_logits, dict):
        return "action" in policy_or_logits
    return getattr(policy_or_logits, "action_head_type", "factorized") == "joint"


def empty_action_storage(policy) -> dict:
    if uses_joint_action(policy):
        return {"action": []}
    return {key: [] for key in ACTION_KEYS}


def clone_action_dict(action_dict):
    return {key: int(action_dict[key]) for key in ACTION_KEYS}


def joint_action_index_to_dict(index: int) -> dict:
    return clone_action_dict(JOINT_ACTIONS[int(index)])


def joint_action_dict_to_index(action_dict) -> int:
    key = tuple(int(action_dict[action_key]) for action_key in ACTION_KEYS)
    if key not in JOINT_ACTION_LOOKUP:
        raise ValueError(f"Unsupported joint action combination: {action_dict}")
    return JOINT_ACTION_LOOKUP[key]


def sample_action_from_logits(logits: dict):
    if "action" in logits:
        dist = torch.distributions.Categorical(logits=logits["action"])
        sampled = dist.sample()
        action_index = int(sampled[0].item())
        return (
            joint_action_index_to_dict(action_index),
            {"action": action_index},
            float(dist.log_prob(sampled)[0].item()),
        )

    action = {}
    action_record = {}
    logp = 0.0
    for key in ACTION_KEYS:
        dist = torch.distributions.Categorical(logits=logits[key])
        sampled = dist.sample()
        action_value = int(sampled[0].item())
        action[key] = action_value
        action_record[key] = action_value
        logp += float(dist.log_prob(sampled)[0].item())
    return action, action_record, logp


def deterministic_action_from_logits(logits: dict):
    if "action" in logits:
        action_index = int(torch.argmax(logits["action"], dim=-1)[0].item())
        return joint_action_index_to_dict(action_index)

    return {
        key: int(torch.argmax(logits[key], dim=-1)[0].item())
        for key in ACTION_KEYS
    }


def action_log_prob_and_entropy(logits: dict, actions: dict):
    if "action" in logits:
        dist = torch.distributions.Categorical(logits=logits["action"])
        action_tensor = actions["action"]
        return dist.log_prob(action_tensor), dist.entropy()

    logps = []
    entropies = []
    for key in ACTION_KEYS:
        dist = torch.distributions.Categorical(logits=logits[key])
        logps.append(dist.log_prob(actions[key]))
        entropies.append(dist.entropy())
    return sum(logps), sum(entropies)


# -------------------------
# MLPPolicy (with value head)
# -------------------------
class MLPPolicy(FootballPolicy):
    def __init__(self, obs_dim=12, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head_left  = nn.Linear(hidden_dim, 2)
        self.head_right = nn.Linear(hidden_dim, 2)
        self.head_jump  = nn.Linear(hidden_dim, 2)
        # self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        """
        obs: (batch, obs_dim)
        returns dict with logits for each action and "value": (batch,1)
        """
        x = self.net(obs)
        return {
            "left":  self.head_left(x),
            "right": self.head_right(x),
            "jump":  self.head_jump(x),
            # "value": self.value_head(x),
        }

    def sample_action(self, obs):
        """Sample action from a single observation (numpy array or 1D torch tensor)."""
        obs_t = _to_tensor(obs)
        with torch.no_grad():
            logits = self.forward(obs_t)
            # logits values are batched; take index 0
            out = {}
            for k in ("left", "right", "jump"):
                dist = torch.distributions.Categorical(logits=logits[k])
                out[k] = int(dist.sample()[0].item())
        return out


# -------------------------
# DummyPolicy (deterministic zeros, returns logits & value)
# -------------------------
class DummyPolicy(FootballPolicy):
    def __init__(self, obs_dim=12):
        super().__init__()
        # We'll create simple constant logits and zero value
        # so forward() works in rollouts
        self.obs_dim = obs_dim

    def forward(self, obs: torch.Tensor):
        batch = obs.shape[0]
        device = obs.device
        # constant logits: prefer action 0 (logits [1.0, 0.0]) but it's arbitrary
        left = torch.zeros(batch, 2, device=device)
        right = torch.zeros(batch, 2, device=device)
        jump = torch.zeros(batch, 2, device=device)
        # small bias to first action
        left[:, 0] = 1.0
        right[:, 0] = 1.0
        jump[:, 0] = 1.0
        value = torch.zeros(batch, 1, device=device)
        return {"left": left, "right": right, "jump": jump, "value": value}

    def sample_action(self, obs):
        # deterministic no-op
        return {"left": 0, "right": 0, "jump": 0}


# -------------------------
# atulPolicy (simple hand-coded policy)
# -------------------------
class atulPolicy(FootballPolicy):
    def __init__(self, obs_dim=12):
        super().__init__()
        self.obs_dim = obs_dim

    def forward(self, obs: torch.Tensor):
        # Return logits that correspond to the actions produced by sample_action.
        # We'll produce batched logits with mild preference for the sampled action.
        batch = obs.shape[0]
        device = obs.device
        left = torch.zeros(batch, 2, device=device)
        right = torch.zeros(batch, 2, device=device)
        jump = torch.zeros(batch, 2, device=device)
        value = torch.zeros(batch, 1, device=device)

        # evaluate heuristic in batch
        # obs layout (per your doc): [rx, ry, rvx, rvy, bx, by, bvx, bvy, ballx, bally, ballvx, ballvy]
        # but original doc used indices 8-11 for ball; keep same indexing
        rx = obs[:, 0]
        ballx = obs[:, 8]
        ballvy = obs[:, 11].abs()

        # if ball is near and ball is to the right of red, prefer move right
        cond = (ballvy < 0.2) & (ballx > rx)
        # set preference
        right[cond, 1] = 1.0  # prefer right action (index 1)
        # otherwise do nothing (left/right/jump pref index 0)
        return {"left": left, "right": right, "jump": jump, "value": value}

    def sample_action(self, obs):
        # Accept numpy or tensor single observation
        arr = np.asarray(obs)
        # indexes: as in your doc
        ballvy = abs(arr[11])
        ballx = arr[8]
        rx = arr[0]
        if (ballvy < 0.2) and (ballx > rx):
            return {"left": 0, "right": 1, "jump": 0}
        return {"left": 0, "right": 0, "jump": 0}


# -------------------------
# CurriculumMLPPolicy (fixed)
# -------------------------
class CurriculumMLPPolicy(FootballPolicy):
    def __init__(self, embed_dim=3, obs_dim=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.obs_dim = obs_dim

        # plan_net used when no explicit setting is provided
        self.plan_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )

        # index embedding for tasks
        self.index_embed = nn.Embedding(10, embed_dim)  # support up to 10 tasks

        # combine (index_emb + par_scalar) -> task embedding
        self.embed_net = nn.Sequential(
            nn.Linear(embed_dim + 1, 20),
            nn.ReLU(),
            nn.Linear(20, embed_dim),
            nn.ReLU(),
        )

        # action network: uses a subset of obs + task embedding
        # subset length = 8 (indices [0,1,2,3,8,9,10,11])
        self.action_net = nn.Sequential(
            nn.Linear(8 + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.head_left  = nn.Linear(128, 2)
        self.head_right = nn.Linear(128, 2)
        self.head_jump  = nn.Linear(128, 2)

        # value head
        self.value_head = nn.Linear(128, 1)

        # setting holds current drill dict (or None)
        self.setting = None

        # mapping from drill name to small integer index (keep in sync with index_embed size)
        self.drill_to_index = {
            "block": 1,
            "block_nobounce": 2,
            "shoot_left": 3,
            "shoot_right": 4,
            "hit_left_wall": 5,
            "hit_right_wall": 6,
            "volley": 7,
            "prepare": 8,
        }

    def set_setting(self, drill):
        """drill is expected to be a dict, e.g. {'drill': 'shoot_left', 'par': 0.1}"""
        self.setting = drill

    def forward(self, obs: torch.Tensor):
        """
        obs: (batch, obs_dim)
        Returns dict with "left","right","jump" logits and "value" (batch,1)
        """
        batch = obs.shape[0]
        device = obs.device

        if self.setting is None:
            # use plan_net applied to the observation to compute a task embedding
            task_emb = self.plan_net(obs)  # (batch, embed_dim)
        else:
            # build index embedding + par scalar and run through embed_net
            drill_name = self.setting.get("drill")
            idx = self.drill_to_index.get(drill_name, 0)
            idx_tensor = torch.tensor([idx], dtype=torch.long, device=device)  # shape (1,)
            idx_emb = self.index_embed(idx_tensor).float()  # (1, embed_dim)
            # expand to batch
            idx_emb = idx_emb.repeat(batch, 1)  # (batch, embed_dim)

            par_val = float(self.setting.get("par", 0.0))
            par_tensor = torch.tensor([[par_val]], dtype=torch.float32, device=device).repeat(batch, 1)  # (batch,1)

            embed_input = torch.cat([idx_emb, par_tensor], dim=-1)  # (batch, embed_dim+1)
            task_emb = self.embed_net(embed_input)  # (batch, embed_dim)

        # pick the relevant obs features: [0,1,2,3,8,9,10,11]
        obs_subset = obs[:, [0, 1, 2, 3, 8, 9, 10, 11]]  # (batch, 8)

        x = self.action_net(torch.cat([obs_subset, task_emb], dim=-1))  # (batch, 128)

        return {
            "left":  self.head_left(x),
            "right": self.head_right(x),
            "jump":  self.head_jump(x),
            "value": self.value_head(x),
        }

    def sample_action(self, obs):
        """Sample action from a single observation (numpy array or 1D tensor)."""
        obs_t = _to_tensor(obs)  # (1, obs_dim)
        with torch.no_grad():
            logits = self.forward(obs_t)
            out = {}
            for k in ("left", "right", "jump"):
                dist = torch.distributions.Categorical(logits=logits[k])
                out[k] = int(dist.sample()[0].item())
        return out

# -------------------------
# RegularMLPPolicy (simple MLP using same obs subset as Curriculum, with value head)
# -------------------------
class RegularMLPPolicy(FootballPolicy):
    def __init__(self, obs_dim=12):
        """
        A plain MLP policy that uses the same 8-element observation subset as
        CurriculumMLPPolicy (indices [0,1,2,3,8,9,10,11]) but has no planning/embedding.
        Produces logits for left/right/jump and a scalar value.
        """
        super().__init__()
        self.obs_dim = obs_dim

        # action network: takes the 8-element subset and produces a 128-d representation
        self.action_net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.head_left  = nn.Linear(128, 2)
        self.head_right = nn.Linear(128, 2)
        self.head_jump  = nn.Linear(128, 2)

        # value head on top of same representation
        self.value_head = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):
        """
        obs: (batch, obs_dim)
        returns dict with logits for each action and "value": (batch,1)
        """
        # pick the relevant obs features: [0,1,2,3,8,9,10,11]
        obs_subset = obs[:, [0, 1, 2, 3, 8, 9, 10, 11]]  # (batch, 8)
        x = self.action_net(obs_subset)  # (batch, 128)

        return {
            "left":  self.head_left(x),
            "right": self.head_right(x),
            "jump":  self.head_jump(x),
            "value": self.value_head(x),
        }

    def sample_action(self, obs):
        """Sample action from a single observation (numpy array or 1D tensor)."""
        obs_t = _to_tensor(obs)  # (1, obs_dim)
        with torch.no_grad():
            logits = self.forward(obs_t)
            out = {}
            for k in ("left", "right", "jump"):
                dist = torch.distributions.Categorical(logits=logits[k])
                out[k] = int(dist.sample()[0].item())
        return out

    def set_setting(self,setting):
        pass

class CurriculumMLPPolicyScaled(FootballPolicy):
    def __init__(self, latent_dims=[128, 128], embed_dim=3, obs_dim=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.obs_dim = obs_dim

        # plan_net used when no explicit setting is provided
        self.plan_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )

        # index embedding for tasks
        self.index_embed = nn.Embedding(10, embed_dim)  # support up to 10 tasks

        # combine (index_emb + par_scalar) -> task embedding
        self.embed_net = nn.Sequential(
            nn.Linear(embed_dim + 1, 20),
            nn.ReLU(),
            nn.Linear(20, embed_dim),
            nn.ReLU(),
        )

        # action network: uses a subset of obs + task embedding
        # subset length = 8 (indices [0,1,2,3,8,9,10,11])
        dims = [8 + embed_dim] + latent_dims
        self.nets = []
        for i in range(len(latent_dims)):
            self.nets.append(nn.Linear(dims[i], dims[i+1]))
            self.nets.append(nn.ReLU())
        self.action_net = nn.Sequential(*self.nets)
        self.head_left  = nn.Linear(latent_dims[-1], 2)
        self.head_right = nn.Linear(latent_dims[-1], 2)
        self.head_jump  = nn.Linear(latent_dims[-1], 2)

        # value head
        self.value_head = nn.Linear(latent_dims[-1], 1)

        # setting holds current drill dict (or None)
        self.setting = None

        # mapping from drill name to small integer index (keep in sync with index_embed size)
        self.drill_to_index = {
            "block_nobounce": 1,
            "shoot_left": 2,
        }

    def set_setting(self, drill):
        """drill is expected to be a dict, e.g. {'drill': 'shoot_left', 'par': 0.1}"""
        self.setting = drill

    def forward(self, obs: torch.Tensor):
        """
        obs: (batch, obs_dim)
        Returns dict with "left","right","jump" logits and "value" (batch,1)
        """
        batch = obs.shape[0]
        device = obs.device

        if self.setting is None:
            # use plan_net applied to the observation to compute a task embedding
            task_emb = self.plan_net(obs)  # (batch, embed_dim)
        else:
            # build index embedding + par scalar and run through embed_net
            drill_name = self.setting.get("drill")
            idx = self.drill_to_index.get(drill_name, 0)
            idx_tensor = torch.tensor([idx], dtype=torch.long, device=device)  # shape (1,)
            idx_emb = self.index_embed(idx_tensor).float()  # (1, embed_dim)
            # expand to batch
            idx_emb = idx_emb.repeat(batch, 1)  # (batch, embed_dim)

            par_val = float(self.setting.get("par", 0.0))
            par_tensor = torch.tensor([[par_val]], dtype=torch.float32, device=device).repeat(batch, 1)  # (batch,1)

            embed_input = torch.cat([idx_emb, par_tensor], dim=-1)  # (batch, embed_dim+1)
            task_emb = self.embed_net(embed_input)  # (batch, embed_dim)

        # pick the relevant obs features: [0,1,2,3,8,9,10,11]
        obs_subset = obs[:, [0, 1, 2, 3, 8, 9, 10, 11]]  # (batch, 8)

        x = self.action_net(torch.cat([obs_subset, task_emb], dim=-1))  # (batch, 128)

        return {
            "left":  self.head_left(x),
            "right": self.head_right(x),
            "jump":  self.head_jump(x),
            "value": self.value_head(x),
        }

    def sample_action(self, obs):
        """Sample action from a single observation (numpy array or 1D tensor)."""
        obs_t = _to_tensor(obs)  # (1, obs_dim)
        with torch.no_grad():
            logits = self.forward(obs_t)
            out = {}
            for k in ("left", "right", "jump"):
                dist = torch.distributions.Categorical(logits=logits[k])
                out[k] = int(dist.sample()[0].item())
        return out

class ActorCriticMLPPolicy(FootballPolicy):
    def __init__(self, latent_dims=[128, 128], obs_dim=12):
        """
        Produces logits for left/right/jump and a scalar value.
        """
        super().__init__()

        # action network: takes the 8-element subset and produces a 128-d representation
        dims = [12] + latent_dims
        self.nets = []
        for i in range(len(latent_dims)):
            self.nets.append(nn.Linear(dims[i], dims[i+1]))
            self.nets.append(nn.ReLU())
        self.action_net = nn.Sequential(*self.nets)
        self.head_left  = nn.Linear(latent_dims[-1], 2)
        self.head_right = nn.Linear(latent_dims[-1], 2)
        self.head_jump  = nn.Linear(latent_dims[-1], 2)

        # value head on top of same representation
        self.value_head = nn.Linear(latent_dims[-1], 1)

    def forward(self, obs: torch.Tensor):
        """
        obs: (batch, obs_dim)
        returns dict with logits for each action and "value": (batch,1)
        """
        x = self.action_net(obs)  # (batch, 128)

        return {
            "left":  self.head_left(x),
            "right": self.head_right(x),
            "jump":  self.head_jump(x),
            "value": self.value_head(x),
        }

    def sample_action(self, obs):
        """Sample action from a single observation (numpy array or 1D tensor)."""
        obs_t = _to_tensor(obs)  # (1, obs_dim)
        with torch.no_grad():
            logits = self.forward(obs_t)
            out = {}
            for k in ("left", "right", "jump"):
                dist = torch.distributions.Categorical(logits=logits[k])
                out[k] = int(dist.sample()[0].item())
        return out

    def set_setting(self, setting):
        pass


class JointActorCriticMLPPolicy(FootballPolicy):
    def __init__(self, latent_dims=[128, 128], obs_dim=12, action_dim=None):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_head_type = "joint"
        self.action_dim = action_dim or len(JOINT_ACTIONS)

        dims = [obs_dim] + latent_dims
        layers = []
        for idx in range(len(latent_dims)):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.ReLU())
        self.action_net = nn.Sequential(*layers)
        self.action_head = nn.Linear(latent_dims[-1], self.action_dim)
        self.value_head = nn.Linear(latent_dims[-1], 1)

    def forward(self, obs: torch.Tensor):
        x = self.action_net(obs)
        return {
            "action": self.action_head(x),
            "value": self.value_head(x),
        }

    def sample_action(self, obs):
        obs_t = _to_tensor(obs)
        with torch.no_grad():
            logits = self.forward(obs_t)
            action, _, _ = sample_action_from_logits(logits)
        return action

    def set_setting(self, setting):
        pass


class FeudalMarkovPolicy(FootballPolicy):
    def __init__(self, obs_dim=12, goal_dim=16, hidden_dim=128, horizon=10):
        super().__init__()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.horizon = horizon

        # --- Internal State ---
        self.internal_step = 0

        # Queue for smoothing the goal (stores last 'horizon' raw slices)
        self.raw_goal_queue = None

        # The actual goal used by the worker (Average of queue)
        self.current_goal = None

        # Stores (s_t, g_t) pairs for Intrinsic Reward calculation
        self.reward_history = deque(maxlen=horizon)

        # Temporary storage for rollout buffer
        self.last_raw_goal = None
        self.last_manager_log_prob = None

        # --- SHARED PERCEPTION (s_t) ---
        self.percept_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),
            nn.LayerNorm(goal_dim),
            nn.Tanh()
        )

        # --- MANAGER (PPO) ---
        # Input: Latent State 's' (dim: goal_dim)
        # Shared Trunk for Manager Policy and Value
        self.manager_net = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.ReLU()
        )

        # Policy Head (Mean + LogStd)
        self.manager_mu = nn.Linear(hidden_dim, goal_dim)
        self.manager_log_std = nn.Parameter(torch.zeros(1, goal_dim) - 0.5)

        # Value Head (Extrinsic Reward) - Corrected to take hidden_dim
        self.manager_value_head = nn.Linear(hidden_dim, 1)

        # --- WORKER ---
        self.worker_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Worker Heads: Output Interaction Matrices U
        self.worker_head_left_U  = nn.Linear(hidden_dim, 2 * goal_dim)
        self.worker_head_right_U = nn.Linear(hidden_dim, 2 * goal_dim)
        self.worker_head_jump_U  = nn.Linear(hidden_dim, 2 * goal_dim)

        # Worker Biases b
        self.worker_head_left_b  = nn.Linear(hidden_dim, 2)
        self.worker_head_right_b = nn.Linear(hidden_dim, 2)
        self.worker_head_jump_b  = nn.Linear(hidden_dim, 2)

        # Worker Value Head (Intrinsic Reward)
        self.worker_value_head = nn.Linear(hidden_dim, 1)

    def _update_manager(self, obs):
        """
        Stateful Tick:
        1. Computes latent state s_t.
        2. Samples new goal slice from Manager (PPO).
        3. Updates sliding window queue.
        4. Updates history for reward calc.
        """
        # Initialize queue if needed
        if self.raw_goal_queue is None:
             self.raw_goal_queue = torch.zeros(1, self.horizon, self.goal_dim, device=obs.device)

        with torch.no_grad():
            # 1. Perception
            s_t = self.percept_net(obs) # (1, goal_dim)

            # 2. Manager PPO Sample
            x = self.manager_net(s_t)
            mu = self.manager_mu(x)
            std = torch.exp(self.manager_log_std)
            dist = torch.distributions.Normal(mu, std)

            # Sample raw goal slice
            raw_slice = dist.sample()

            # Save data for the rollout buffer
            self.last_raw_goal = raw_slice
            self.last_manager_log_prob = dist.log_prob(raw_slice).sum(dim=-1)

            # Normalize slice for use in Worker/Queue
            norm_slice = torch.nn.functional.normalize(raw_slice, p=2, dim=1)

            # 3. Update Queue (Shift left, append new)
            self.raw_goal_queue = torch.cat(
                [self.raw_goal_queue[:, 1:, :], norm_slice.unsqueeze(1)], dim=1
            )

            # 4. Calculate Effective Goal (Average)
            self.current_goal = self.raw_goal_queue.mean(dim=1, keepdim=False)

            # 5. Store History for Intrinsic Reward
            self.reward_history.append((s_t.detach(), self.current_goal.detach()))

        self.internal_step += 1

    def forward(self, obs: torch.Tensor, goal: torch.Tensor = None):
        """
        Pure Forward Pass (Worker Logic).
        """
        # 1. Resolve Goal
        if goal is None:
            if self.current_goal is None:
                device = obs.device
                self.current_goal = torch.zeros(1, self.goal_dim, device=device)

            if self.current_goal.shape[0] != obs.shape[0]:
                goal = self.current_goal.repeat(obs.shape[0], 1)
            else:
                goal = self.current_goal

        # --- DETACH GOAL for Worker Gradient Block ---
        goal_detached = goal.detach()

        # 2. Worker Network
        x = self.worker_net(obs)
        outputs = {}

        if goal_detached.dim() == 2:
            g_unsq = goal_detached.unsqueeze(-1)
        else:
            g_unsq = goal_detached

        # 3. FeUdal Interaction: Logits = (U @ g) + b
        for name, head_U, head_b in [
            ("left", self.worker_head_left_U, self.worker_head_left_b),
            ("right", self.worker_head_right_U, self.worker_head_right_b),
            ("jump", self.worker_head_jump_U, self.worker_head_jump_b)
        ]:
            b = head_b(x)
            U_flat = head_U(x)
            U = U_flat.view(obs.shape[0], 2, self.goal_dim)

            interaction = torch.bmm(U, g_unsq).squeeze(-1)
            outputs[name] = interaction + b

        outputs["value"] = self.worker_value_head(x)
        return outputs

    def sample_action(self, obs):
        """
        Main entry point for Inference/Rollout.
        Updates internal state, then predicts action.
        """
        obs_t = _to_tensor(obs)
        self._update_manager(obs_t)

        with torch.no_grad():
            logits = self.forward(obs_t)
            out = {}
            for k in ("left", "right", "jump"):
                dist = torch.distributions.Categorical(logits=logits[k])
                out[k] = int(dist.sample()[0].item())
        return out

    def compute_intrinsic_reward(self, next_obs):
        """
        Calculates 1/c * sum( Cosine(s_next - s_past, g_past) )
        """
        # Convert next_obs to tensor
        if isinstance(next_obs, np.ndarray):
            device = self.percept_net[0].weight.device
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            next_obs_t = next_obs

        with torch.no_grad():
            s_next = self.percept_net(next_obs_t)

        if len(self.reward_history) == 0:
            return 0.0

        reward_sum = 0.0

        # Flatten for vector math
        s_next_vec = s_next.detach().view(-1)

        for s_past, g_past in self.reward_history:
            s_past_vec = s_past.view(-1)
            g_past_vec = g_past.view(-1)

            # Directional alignment: s_{t+1} - s_{t-i} vs g_{t-i}
            diff = s_next_vec - s_past_vec

            norm_diff = torch.norm(diff)
            norm_goal = torch.norm(g_past_vec)

            if norm_diff > 1e-6 and norm_goal > 1e-6:
                dot = torch.dot(diff, g_past_vec)
                reward_sum += (dot / (norm_diff * norm_goal)).item()

        # Average and Scale
        return (reward_sum / len(self.reward_history)) * 0.1

    def evaluate_manager(self, obs, raw_goal_slices):
        """
        Helper for PPO Update. Re-calculates log_probs for the Manager.
        """
        s_t = self.percept_net(obs)
        x = self.manager_net(s_t)
        mu = self.manager_mu(x)
        std = torch.exp(self.manager_log_std)

        dist = torch.distributions.Normal(mu, std)

        log_prob = dist.log_prob(raw_goal_slices).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        # Corrected: Value head takes 'x' (hidden_dim)
        value = self.manager_value_head(x)

        return log_prob, entropy, value

    def reset_state(self):
        self.internal_step = 0
        if self.raw_goal_queue is not None:
            self.raw_goal_queue.zero_()
        self.current_goal = None
        self.reward_history.clear()
        self.last_raw_goal = None
        self.last_manager_log_prob = None

    def set_setting(self, setting):
        pass


# -------------------------
# Factory + checkpoint loader
# -------------------------
def make_policy(class_name, **kwargs):
    """Initialize policy class given class name and kwargs."""
    name_to_class = {
        "MLPPolicy": MLPPolicy,
        "CurriculumMLPPolicy": CurriculumMLPPolicy,
        "RegularMLPPolicy":RegularMLPPolicy,
        "DummyPolicy": DummyPolicy,
        "atulPolicy": atulPolicy,
        "CurriculumMLPPolicyScaled": CurriculumMLPPolicyScaled,
        "ActorCriticMLPPolicy": ActorCriticMLPPolicy,
        "JointActorCriticMLPPolicy": JointActorCriticMLPPolicy,
        "FeudalMarkovPolicy": FeudalMarkovPolicy,
    }
    if class_name in name_to_class:
        return name_to_class[class_name](**kwargs)
    raise ValueError(f"Unknown policy: {class_name}")


def policy_from_checkpoint_path(checkpoint_path):
    assert os.path.exists(checkpoint_path), f"Error: Checkpoint file not found at {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # older checkpoints might not include policy_kwargs; handle gracefully
    policy_kwargs = checkpoint.get("policy_kwargs", {})
    policy_class_name = checkpoint.get("policy_class") or checkpoint.get("policy_name") or "MLPPolicy"

    policy = make_policy(policy_class_name, **policy_kwargs)

    state = checkpoint.get("policy_state_dict") or checkpoint.get("state_dict")
    if state:
        policy.load_state_dict(state)

    return policy, checkpoint
