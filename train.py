"""
train_ppo.py  —  PPO for Multi-Agent Navigation (shared policy)
================================================================

Observation per agent (flat vector):
    [curr_x, curr_y, vel_x, vel_y,
     goal_dx, goal_dy,
     closest_dx_1, closest_dy_1,
     closest_dx_2, closest_dy_2,
     ...
     closest_dx_K, closest_dy_K]          (K nearest neighbours, zero-padded)

Reward per agent per step:
    r = w1 * time_penalty
      + w2 * collisions_this_step
      + w3 * closest_agent_distance_term
      + goal_bonus  (sparse, on reaching goal)

All N agents share one policy.  Each episode, every agent rolls out
independently, and all (s,a,r,s',done) tuples are pooled into a single
PPO update — identical to the centralised-training / decentralised-execution
paradigm used in the paper.

Usage
-----
    python train_ppo.py                    # train with defaults
    python train_ppo.py --n_agents 6 --episodes 2000 --save_path policy.pt
    python train_ppo.py --eval --load_path policy.pt
"""

import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

sys.path.insert(0, os.path.dirname(__file__))
from env.multi_agent_env import MultiAgentNavEnv, MODE_RL_FULL, DT

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters  (all overridable via CLI)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    # environment
    n_agents        = 6,
    grid_size       = 10.0,
    collision_radius= 0.30,
    recovery_steps  = 6,
    max_steps       = 300,       # max env steps per episode
    k_neighbors     = 3,         # K nearest neighbours in obs

    # reward weights   r = w1*time + w2*collisions + w3*closest_dist + w4*dist_improvement + goal_bonus
    w1              = -0.01,     # time penalty   (negative → penalise slow)
    w2              = -2.0,      # collision penalty per collision this step
    w3              = -0.05,     # closest-agent penalty (closer → more negative)
    w4              =  5.0,       # distance improvement bonus (closer → more positive)
    goal_bonus      = 10.0,      # sparse reward on reaching goal

    # PPO
    lr              = 3e-4,
    gamma           = 0.99,
    lam             = 0.95,      # GAE lambda
    clip_eps        = 0.2,
    entropy_coef    = 0.05,
    value_coef      = 0.5,
    max_grad_norm   = 0.5,
    ppo_epochs      = 4,         # gradient steps per collected batch
    minibatch_size  = 256,

    # training
    episodes        = 3000,
    rollout_steps   = 2048,      # steps collected before each PPO update
    log_interval    = 50,        # episodes between console prints
    save_interval   = 500,
    save_path       = "outputs/policy_w4_2.pt",
    load_path       = None,
    eval            = False,
    seed            = 42,
    device          = "cuda",
)


# ─────────────────────────────────────────────────────────────────────────────
# Observation helper
# ─────────────────────────────────────────────────────────────────────────────

def obs_dim(k: int) -> int:
    """Dimension of the flat observation vector."""
    # [x, y, vx, vy, goal_dx, goal_dy] + K * [nb_dx, nb_dy]
    return 6 + k * 2


def make_obs_vector(agent_obs: dict, k: int, grid_size: float) -> np.ndarray:
    """
    Convert one agent's dict observation into a normalised flat numpy vector.

    Components
    ----------
    curr_x, curr_y          : position (normalised by grid_size)
    vel_x,  vel_y           : velocity (normalised by MAX_SPEED=2)
    goal_dx, goal_dy        : relative goal vector (normalised by grid_size)
    closest_dx_1, dy_1      : relative position of 1st nearest neighbour
    ...
    closest_dx_K, dy_K      : relative position of K-th nearest neighbour
                              (zero-padded if fewer than K neighbours exist)
    """
    from env.multi_agent_env import MAX_SPEED

    pos  = agent_obs["pos"]
    vel  = agent_obs["vel"]
    goal = agent_obs["goal"]

    ego = np.array([
        pos[0]  / grid_size,
        pos[1]  / grid_size,
        vel[0]  / MAX_SPEED,
        vel[1]  / MAX_SPEED,
        (goal[0] - pos[0]) / grid_size,
        (goal[1] - pos[1]) / grid_size,
    ], dtype=np.float32)

    # sort neighbours by distance, take K closest
    neighbours = sorted(agent_obs["neighbors"], key=lambda n: n["dist"])[:k]
    nb_vecs = []
    for nb in neighbours:
        nb_vecs.extend([
            nb["rel_pos"][0] / grid_size,
            nb["rel_pos"][1] / grid_size,
        ])
    # zero-pad
    nb_vecs += [0.0] * (k * 2 - len(nb_vecs))

    return np.concatenate([ego, np.array(nb_vecs, dtype=np.float32)])


# ─────────────────────────────────────────────────────────────────────────────
# Reward function
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward(agent_obs: dict, collisions_this_step: int,
                   reached_goal_this_step: bool,
                   distance_improvement: float,
                   w1: float, w2: float, w3: float, w4: float,
                   goal_bonus: float) -> float:
    """
    r = w1 * 1                          (time penalty, always -w1 per step)
      + w2 * collisions_this_step       (negative when collisions>0)
      + w3 * 1/closest_dist             (closer neighbour → larger penalty)
      + w4 * distance_improvement       (closer → more positive)
      + goal_bonus  (if just reached goal)

    w3 term: we use 1/d so the penalty is large when very close.
    If no neighbours, w3 term = 0.
    """
    r = w1  # time penalty (w1 is negative, so this subtracts)

    r += w2 * collisions_this_step

    r += w4 * distance_improvement

    # closest agent proximity penalty
    neighbours = agent_obs["neighbors"]
    if neighbours:
        min_dist = min(n["dist"] for n in neighbours)
        min_dist = max(min_dist, 1e-2)   # avoid division by zero
        r += w3 * (1.0 / min_dist)       # w3 negative → penalty for being close

    if reached_goal_this_step:
        r += goal_bonus



    return float(r)


# ─────────────────────────────────────────────────────────────────────────────
# Actor-Critic network
# ─────────────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared-trunk Actor-Critic for continuous action space [accel, steer] ∈ [-1,1].

    Architecture
    ------------
    Shared MLP → split into actor head (mean) and critic head (value).
    Action std is a learnable parameter (not state-dependent), following the
    standard PPO implementation style.
    """

    def __init__(self, obs_size: int, action_size: int = 2,
                 hidden: int = 256):
        super().__init__()

        # shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )

        # actor: outputs mean of Gaussian
        self.actor_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, action_size),
            nn.Tanh(),   # squash mean to (-1, 1)
        )

        # critic: outputs scalar value
        self.critic_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

        # learnable log std (one per action dim, shared across states)
        self.log_std = nn.Parameter(torch.zeros(action_size))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # smaller gain for output layers
        nn.init.orthogonal_(self.actor_head[-2].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        trunk_out = self.trunk(obs)
        value      = self.critic_head(trunk_out).squeeze(-1)
        action_mean = self.actor_head(trunk_out)
        return action_mean, value

    def get_dist(self, obs: torch.Tensor) -> Normal:
        action_mean, _ = self.forward(obs)
        std = self.log_std.exp().expand_as(action_mean)
        return Normal(action_mean, std)

    def act(self, obs: torch.Tensor):
        """Sample action and return (action, log_prob, value)."""
        trunk_out   = self.trunk(obs)
        action_mean = self.actor_head(trunk_out)
        value       = self.critic_head(trunk_out).squeeze(-1)
        std         = self.log_std.exp().expand_as(action_mean)
        dist        = Normal(action_mean, std)
        action      = dist.sample()
        log_prob    = dist.log_prob(action).sum(-1)
        action_clipped = action.clamp(-1.0, 1.0)
        return action_clipped, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """Given stored (obs, action), return (log_prob, value, entropy)."""
        trunk_out   = self.trunk(obs)
        action_mean = self.actor_head(trunk_out)
        value       = self.critic_head(trunk_out).squeeze(-1)
        std         = self.log_std.exp().expand_as(action_mean)
        dist        = Normal(action_mean, std)
        log_prob    = dist.log_prob(action).sum(-1)
        entropy     = dist.entropy().sum(-1)
        return log_prob, value, entropy


# ─────────────────────────────────────────────────────────────────────────────
# Rollout buffer
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores per-agent trajectories separately so GAE can be computed correctly
    along each agent's own time axis, then flattens everything for the PPO
    minibatch update.

    Layout
    ------
    _agent_bufs[i] holds the raw transitions for agent i in episode order.
    After compute_returns_and_advantages() is called the flat arrays
    (obs, actions, log_probs, returns, advantages) are ready for sampling.
    """

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self._agent_bufs = [self._empty_agent_buf() for _ in range(n_agents)]
        # filled after GAE
        self.obs        = None
        self.actions    = None
        self.log_probs  = None
        self.returns    = None
        self.advantages = None

    @staticmethod
    def _empty_agent_buf():
        return dict(obs=[], actions=[], log_probs=[], rewards=[], values=[], dones=[])

    def push(self, agent_id: int, obs, action, log_prob, reward, value, done):
        """Push one transition for a single agent."""
        b = self._agent_bufs[agent_id]
        b["obs"].append(obs)
        b["actions"].append(action)
        b["log_probs"].append(log_prob)
        b["rewards"].append(float(reward))
        b["values"].append(float(value))
        b["dones"].append(float(done))

    def clear(self):
        self.__init__(self.n_agents)

    def total_transitions(self) -> int:
        return sum(len(b["rewards"]) for b in self._agent_bufs)

    def __len__(self):
        return self.total_transitions()

    # ------------------------------------------------------------------
    def compute_returns_and_advantages(self,
                                       last_values: np.ndarray,
                                       last_dones:  np.ndarray,
                                       gamma: float, lam: float):
        """
        GAE-Lambda computed independently per agent along their own time axis.

        Parameters
        ----------
        last_values : shape (n_agents,) — V(s_{T+1}) bootstrap values
        last_dones  : shape (n_agents,) — whether s_{T+1} is terminal
        """
        all_obs, all_act, all_lp, all_ret, all_adv = [], [], [], [], []

        for i in range(self.n_agents):
            b = self._agent_bufs[i]
            if len(b["rewards"]) == 0:
                continue

            rewards = np.array(b["rewards"], dtype=np.float32)
            values  = np.array(b["values"],  dtype=np.float32)
            dones   = np.array(b["dones"],   dtype=np.float32)
            T = len(rewards)

            # bootstrap scalar for this agent
            last_v = float(last_values[i])
            last_d = float(last_dones[i])

            advantages = np.zeros(T, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    next_val  = last_v
                    next_done = last_d
                else:
                    next_val  = values[t + 1]
                    next_done = dones[t + 1]

                delta = rewards[t] + gamma * next_val * (1.0 - next_done) - values[t]
                gae   = delta + gamma * lam * (1.0 - next_done) * gae
                advantages[t] = gae

            returns = advantages + values

            all_obs.append(np.array(b["obs"],      dtype=np.float32))
            all_act.append(np.array(b["actions"],  dtype=np.float32))
            all_lp.append (np.array(b["log_probs"],dtype=np.float32))
            all_ret.append(returns)
            all_adv.append(advantages)

        self.obs        = np.concatenate(all_obs,  axis=0)
        self.actions    = np.concatenate(all_act,  axis=0)
        self.log_probs  = np.concatenate(all_lp,   axis=0)
        self.returns    = np.concatenate(all_ret,  axis=0)
        self.advantages = np.concatenate(all_adv,  axis=0)

    def to_tensors(self, device):
        obs       = torch.FloatTensor(self.obs).to(device)
        actions   = torch.FloatTensor(self.actions).to(device)
        log_probs = torch.FloatTensor(self.log_probs).to(device)
        returns   = torch.FloatTensor(self.returns).to(device)
        advantages= torch.FloatTensor(self.advantages).to(device)
        return obs, actions, log_probs, returns, advantages


# ─────────────────────────────────────────────────────────────────────────────
# PPO update
# ─────────────────────────────────────────────────────────────────────────────

def ppo_update(policy: ActorCritic, optimizer: optim.Optimizer,
               buffer: RolloutBuffer,
               cfg: argparse.Namespace, device: str):

    obs, actions, old_log_probs, returns_t, advantages_t = buffer.to_tensors(device)

    # normalise advantages
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    N = len(returns_t)
    total_loss_val = 0.0
    n_updates = 0

    for _ in range(cfg.ppo_epochs):
        indices = np.random.permutation(N)
        for start in range(0, N, cfg.minibatch_size):
            idx = indices[start: start + cfg.minibatch_size]
            mb_obs      = obs[idx]
            mb_actions  = actions[idx]
            mb_old_lp   = old_log_probs[idx]
            mb_returns  = returns_t[idx]
            mb_adv      = advantages_t[idx]

            new_log_probs, values, entropy = policy.evaluate(mb_obs, mb_actions)

            # PPO clipped surrogate objective
            ratio      = (new_log_probs - mb_old_lp).exp()
            surr1      = ratio * mb_adv
            surr2      = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
            actor_loss = -torch.min(surr1, surr2).mean()

            # value loss (clipped)
            value_loss  = nn.functional.mse_loss(values, mb_returns)

            # entropy bonus
            entropy_loss = -entropy.mean()

            loss = actor_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            total_loss_val += loss.item()
            n_updates += 1

    return total_loss_val / max(n_updates, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: argparse.Namespace):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = cfg.device
    os.makedirs(os.path.dirname(os.path.abspath(cfg.save_path)), exist_ok=True)

    env = MultiAgentNavEnv(
        n_agents        = cfg.n_agents,
        grid_size       = cfg.grid_size,
        collision_radius= cfg.collision_radius,
        recovery_steps  = cfg.recovery_steps,
        mode            = MODE_RL_FULL,
        goal_threshold  = 0.28,
        seed            = cfg.seed,
    )

    odim   = obs_dim(cfg.k_neighbors)
    policy = ActorCritic(obs_size=odim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)

    if cfg.load_path and os.path.exists(cfg.load_path):
        ckpt = torch.load(cfg.load_path, map_location=device)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print(f"[train] Loaded checkpoint from {cfg.load_path}")

    buffer   = RolloutBuffer(cfg.n_agents)
    ep_rewards    = deque(maxlen=100)
    ep_collisions = deque(maxlen=100)
    ep_done_frac  = deque(maxlen=100)
    total_steps   = 0
    update_count  = 0
    t0            = time.time()

    print(f"\n{'='*60}")
    print(f"  PPO Multi-Agent Navigation Training")
    print(f"  Agents: {cfg.n_agents}  |  Grid: {cfg.grid_size}x{cfg.grid_size}")
    print(f"  Obs dim: {odim}  |  K neighbours: {cfg.k_neighbors}")
    print(f"  Reward weights: w1={cfg.w1}  w2={cfg.w2}  w3={cfg.w3}  w4={cfg.w4}  goal_bonus={cfg.goal_bonus}")
    print(f"  Episodes: {cfg.episodes}  |  Device: {device}")
    print(f"{'='*60}\n")

    for episode in range(1, cfg.episodes + 1):
        obs_list = env.reset(
            # randomise seed each episode for diversity
            starts=None, goals=None
        )
        env.rng = np.random.default_rng(cfg.seed + episode)

        # track which agents had a collision this step (for reward computation)
        prev_agent_collisions = {i: 0 for i in range(cfg.n_agents)}
        ep_reward_sum   = 0.0
        ep_collision_sum = 0

        for step in range(cfg.max_steps):
            # ── collect actions from policy for all agents ──────────────────
            obs_vecs = np.stack([
                make_obs_vector(o, cfg.k_neighbors, cfg.grid_size)
                for o in obs_list
            ])  # shape: (N, obs_dim)

            obs_t = torch.FloatTensor(obs_vecs).to(device)
            with torch.no_grad():
                actions_t, log_probs_t, values_t = policy.act(obs_t)

            actions_np   = actions_t.cpu().numpy()   # (N, 2)
            log_probs_np = log_probs_t.cpu().numpy() # (N,)
            values_np    = values_t.cpu().numpy()    # (N,)

            action_dict = {i: actions_np[i] for i in range(cfg.n_agents)}

            # Calculate current distances for all agents
            prev_dists = []
            for obs in obs_list:
                d = np.linalg.norm(obs["goal"] - obs["pos"])
                prev_dists.append(d)

            # ── step environment ────────────────────────────────────────────
            next_obs_list, _, env_done, info = env.step(action_dict)

            # ── compute per-agent rewards ───────────────────────────────────
            for i, (agent_obs, next_obs) in enumerate(zip(obs_list, next_obs_list)):
                

                # Calculate new distance
                curr_dist = np.linalg.norm(next_obs["goal"] - next_obs["pos"])
                
                # Improvement = (Old Distance - New Distance)
                # If improvement is 0.05, the agent moved 5cm closer to the goal.
                dist_improvement = prev_dists[i] - curr_dist

                curr_col   = info["agent_collisions"][i]
                new_cols   = curr_col - prev_agent_collisions[i]
                prev_agent_collisions[i] = curr_col

                just_reached = (next_obs["reached_goal"] and
                                not agent_obs["reached_goal"])

                r = compute_reward(
                    agent_obs          = agent_obs,
                    collisions_this_step = new_cols,
                    reached_goal_this_step = just_reached,
                    distance_improvement = dist_improvement,
                    w1=cfg.w1, w2=cfg.w2, w3=cfg.w3, w4=cfg.w4,
                    goal_bonus=cfg.goal_bonus,
                )

                done_flag = float(next_obs["reached_goal"])

                # push each agent's transition individually (shared policy)
                buffer.push(
                    agent_id = i,
                    obs      = obs_vecs[i],
                    action   = actions_np[i],
                    log_prob = log_probs_np[i],
                    reward   = r,
                    value    = values_np[i],
                    done     = done_flag,
                )

                ep_reward_sum   += r
                ep_collision_sum += new_cols

            total_steps += cfg.n_agents
            obs_list = next_obs_list

            # ── PPO update whenever buffer is large enough ──────────────────
            if len(buffer) >= cfg.rollout_steps:
                # Bootstrap V(s_{T+1}) separately for each agent
                last_obs_vecs = np.stack([
                    make_obs_vector(o, cfg.k_neighbors, cfg.grid_size)
                    for o in obs_list
                ])
                with torch.no_grad():
                    _, last_vals = policy(
                        torch.FloatTensor(last_obs_vecs).to(device)
                    )
                # shape (n_agents,) — one scalar bootstrap value per agent
                last_vals_np = last_vals.cpu().numpy()
                last_dones   = np.array([float(o["reached_goal"]) for o in obs_list])

                # GAE computed per-agent along their own time axis
                buffer.compute_returns_and_advantages(
                    last_vals_np, last_dones, cfg.gamma, cfg.lam
                )
                loss = ppo_update(policy, optimizer, buffer, cfg, device)
                buffer.clear()
                update_count += 1

            if env_done:
                break

        ep_rewards.append(ep_reward_sum)
        ep_collisions.append(ep_collision_sum)
        ep_done_frac.append(info["n_done"] / cfg.n_agents)

        # ── logging ─────────────────────────────────────────────────────────
        if episode % cfg.log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"Ep {episode:5d}/{cfg.episodes}"
                f"  |  steps={total_steps:7d}"
                f"  |  R={np.mean(ep_rewards):+8.2f}"
                f"  |  col={np.mean(ep_collisions):5.1f}"
                f"  |  done={np.mean(ep_done_frac)*100:5.1f}%"
                f"  |  updates={update_count}"
                f"  |  t={elapsed:.0f}s"
            )

        # ── checkpoint ──────────────────────────────────────────────────────
        if episode % cfg.save_interval == 0:
            torch.save({
                "episode":   episode,
                "policy":    policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg":       vars(cfg),
            }, cfg.save_path)
            print(f"  → Checkpoint saved to {cfg.save_path}")

    # final save
    torch.save({
        "episode":   cfg.episodes,
        "policy":    policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg":       vars(cfg),
    }, cfg.save_path)
    print(f"\nTraining complete. Policy saved to {cfg.save_path}")
    return policy


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(cfg: argparse.Namespace, n_eval_episodes: int = 20):
    device = cfg.device
    env = MultiAgentNavEnv(
        n_agents        = cfg.n_agents,
        grid_size       = cfg.grid_size,
        collision_radius= cfg.collision_radius,
        recovery_steps  = cfg.recovery_steps,
        mode            = MODE_RL_FULL,
        goal_threshold  = 0.28,
        seed            = cfg.seed + 9999,
    )
    odim   = obs_dim(cfg.k_neighbors)
    policy = ActorCritic(obs_size=odim).to(device)

    ckpt = torch.load(cfg.load_path, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    print(f"Loaded policy from {cfg.load_path} (trained for {ckpt['episode']} episodes)")

    total_done = []
    total_col  = []
    total_time = []

    for ep in range(n_eval_episodes):
        obs_list = env.reset()
        env.rng  = np.random.default_rng(cfg.seed + 9999 + ep)

        for step in range(cfg.max_steps):
            obs_vecs = np.stack([
                make_obs_vector(o, cfg.k_neighbors, cfg.grid_size)
                for o in obs_list
            ])
            with torch.no_grad():
                action_mean, _ = policy(torch.FloatTensor(obs_vecs).to(device))
            actions_np  = action_mean.cpu().numpy().clip(-1, 1)
            action_dict = {i: actions_np[i] for i in range(cfg.n_agents)}
            obs_list, _, done, info = env.step(action_dict)
            if done:
                break

        total_done.append(info["n_done"] / cfg.n_agents)
        total_col.append(info["total_collisions"])
        gt = [v for v in info["agent_goal_times"].values() if v is not None]
        total_time.append(np.mean(gt) if gt else cfg.max_steps * DT)

    print(f"\n{'='*50}")
    print(f"  Evaluation over {n_eval_episodes} episodes")
    print(f"  Success rate : {np.mean(total_done)*100:.1f}%")
    print(f"  Avg collisions: {np.mean(total_col):.2f}")
    print(f"  Avg goal time : {np.mean(total_time):.2f}s")
    print(f"{'='*50}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PPO training for multi-agent navigation")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        elif v is None:
            p.add_argument(f"--{k}", type=str, default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    return p.parse_args()


if __name__ == "__main__":
    cfg = parse_args()

    if cfg.eval:
        if not cfg.load_path:
            raise ValueError("--load_path required for --eval")
        evaluate(cfg)
    else:
        train(cfg)