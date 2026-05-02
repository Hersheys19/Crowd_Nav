"""
visualize_policy.py  —  Test & visualize a trained PPO navigation policy
=========================================================================

Three outputs in one script:

  1. Quantitative report  – success rate, collisions, goal time, speed,
                            vs. random baseline, printed to terminal and
                            optionally saved to CSV.

  2. Trajectory plot      – static PNG showing every agent's full path,
                            start (●), goal (★), and collision events (✕)
                            for each scenario tested.

  3. Animated GIF         – live playback of one episode per scenario with:
                              • agent circles + ID labels
                              • velocity arrows
                              • trajectory trails
                              • collision flash rings
                              • live stats panel (goal time, collisions,
                                reward per step, value estimate)
                              • reward-over-time chart that updates each frame

Usage
-----
    python visualize_policy.py --load_path outputs/policy.pt

    python visualize_policy.py --load_path outputs/policy.pt \\
        --scenario circle --n_agents 8 --n_episodes 20 \\
        --save_gif outputs/eval.gif \\
        --save_plot outputs/trajectories.png \\
        --save_csv  outputs/results.csv
"""

import argparse
import os
import sys
import csv
import time

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
from env.multi_agent_env import (
    MultiAgentNavEnv, MODE_RL_FULL, DT, MAX_SPEED, MAX_ACCEL, MAX_STEER
)
from train import ActorCritic, make_obs_vector, obs_dim, DEFAULTS, compute_reward

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (up to 15 agents)
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = [
    "#e63946","#457b9d","#2a9d8f","#e9c46a","#f4a261",
    "#264653","#b5838d","#8338ec","#06d6a0","#118ab2",
    "#ffd166","#ef476f","#3a86ff","#ff006e","#8ac926",
]
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
FG      = "#e6edf3"
MUTED   = "#8b949e"
RED     = "#ff7b72"
GREEN   = "#3fb950"


# ─────────────────────────────────────────────────────────────────────────────
# Policy loader
# ─────────────────────────────────────────────────────────────────────────────

def load_policy(path: str, k_neighbors: int, device: str):
    ckpt   = torch.load(path, map_location=device, weights_only=False)
    cfg    = ckpt.get("cfg", {})
    k      = cfg.get("k_neighbors", k_neighbors)
    policy = ActorCritic(obs_size=obs_dim(k)).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    print(f"  Loaded '{path}'  (episode {ckpt.get('episode','?')}, k={k})")
    return policy, k, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Scenario builders
# ─────────────────────────────────────────────────────────────────────────────

def circle_config(n, G, frac=0.38):
    cx = cy = G / 2;  r = G * frac
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    S = [[cx + r*np.cos(a), cy + r*np.sin(a)] for a in angles]
    G_ = [[cx + r*np.cos(a+np.pi), cy + r*np.sin(a+np.pi)] for a in angles]
    return S, G_

def corridor_config(n, G):
    half = n // 2;  m = 0.8
    S, G_ = [], []
    yl = np.linspace(G*0.35, G*0.65, half)
    yr = np.linspace(G*0.35, G*0.65, n-half)
    for y in yl:  S.append([m, y]);      G_.append([G-m, y])
    for y in yr:  S.append([G-m, y]);    G_.append([m, y])
    return S, G_

def random_config(env):
    return None, None   # env.reset() handles it


# ─────────────────────────────────────────────────────────────────────────────
# Single episode rollout  — returns rich frame data for visualisation
# ─────────────────────────────────────────────────────────────────────────────

def rollout(policy, env, k, max_steps, device,
            starts=None, goals=None,
            w1=-0.01, w2=-2.0, w3=-0.05, w4=2.0, goal_bonus=10.0,
            greedy=True):
    """
    Returns
    -------
    frames : list of dicts, one per timestep, containing everything needed
             for animation and plotting.
    info   : final episode info dict from the environment.
    """
    if starts is not None:
        obs_list = env.reset(starts=starts, goals=goals)
    else:
        obs_list = env.reset()

    prev_col  = {i: 0 for i in range(env.n_agents)}
    prev_pos  = {i: env.agents[i].pos.copy() for i in range(env.n_agents)}
    dist_trav = {i: 0.0 for i in range(env.n_agents)}
    frames    = []

    for step in range(max_steps):
        # --- NEW: Store current distances before taking a step ---
        prev_dists = [np.linalg.norm(o["goal"] - o["pos"]) for o in obs_list]
        obs_vecs = np.stack([
            make_obs_vector(o, k, env.grid_size) for o in obs_list
        ])
        obs_t = torch.FloatTensor(obs_vecs).to(device)
        with torch.no_grad():
            action_mean, values_t = policy(obs_t)
            if greedy:
                actions_np = action_mean.cpu().numpy().clip(-1.0, 1.0)
            else:
                from torch.distributions import Normal
                std  = policy.log_std.exp().expand_as(action_mean)
                dist = Normal(action_mean, std)
                actions_np = dist.sample().clamp(-1.0, 1.0).cpu().numpy()
            values_np = values_t.cpu().numpy()   # (N,)

        action_dict = {i: actions_np[i] for i in range(env.n_agents)}
        next_obs_list, _, env_done, info = env.step(action_dict)

        # per-agent reward this step
        step_rewards = []
        for i, (ao, no) in enumerate(zip(obs_list, next_obs_list)):
            # --- NEW: Calculate distance improvement ---
            curr_dist = np.linalg.norm(no["goal"] - no["pos"])
            dist_improvement = prev_dists[i] - curr_dist
            new_col = info["agent_collisions"][i] - prev_col[i]
            prev_col[i] = info["agent_collisions"][i]
            just_reached = no["reached_goal"] and not ao["reached_goal"]

            r = compute_reward(ao, new_col, just_reached, dist_improvement,
                               w1, w2, w3, w4, goal_bonus)
            step_rewards.append(r)
            dist_trav[i] += float(np.linalg.norm(
                env.agents[i].pos - prev_pos[i]))
            prev_pos[i] = env.agents[i].pos.copy()

        frames.append({
            "step":       step,
            "sim_time":   info["sim_time"],
            "positions":  [a.pos.copy()              for a in env.agents],
            "velocities": [a.vel.copy()              for a in env.agents],
            "headings":   [a.heading                 for a in env.agents],
            "reached":    [a.reached_goal            for a in env.agents],
            "recovery":   [a.recovery_steps_left     for a in env.agents],
            "collisions": [a.collisions              for a in env.agents],
            "goal_times": [a.goal_time               for a in env.agents],
            "rewards":    step_rewards,
            "values":     values_np.tolist(),
            "info":       dict(info),
            "actions":    actions_np.tolist(),
        })

        obs_list = next_obs_list
        if env_done:
            break

    final_info = env._build_info()
    # attach distance travelled
    final_info["dist_travelled"] = dist_trav
    return frames, final_info


# ─────────────────────────────────────────────────────────────────────────────
# Quantitative benchmark  (multiple episodes)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(policy, env, k, max_steps, n_episodes, device,
              scenario_name, w1, w2, w3, w4, goal_bonus,
              starts_fn=None, goals_fn=None, greedy=True):
    results = []
    for ep in range(n_episodes):
        env.rng = np.random.default_rng(1000 + ep * 31)
        S = starts_fn() if starts_fn else None
        G = goals_fn()  if goals_fn  else None
        _, info = rollout(policy, env, k, max_steps, device,
                          starts=S, goals=G,
                          w1=w1, w2=w2, w3=w3, w4=w4, goal_bonus=goal_bonus,
                          greedy=greedy)
        results.append(info)
    return results


def random_baseline(env, max_steps, n_episodes, starts_fn=None, goals_fn=None):
    results = []
    for ep in range(n_episodes):
        env.rng = np.random.default_rng(9000 + ep * 31)
        if starts_fn:
            env.reset(starts=starts_fn(), goals=goals_fn())
        else:
            env.reset()
        prev_pos  = {i: env.agents[i].pos.copy() for i in range(env.n_agents)}
        dist_trav = {i: 0.0 for i in range(env.n_agents)}
        for _ in range(max_steps):
            acts = {i: np.random.uniform(-1, 1, 2) for i in range(env.n_agents)}
            _, _, done, info = env.step(acts)
            for i, ag in enumerate(env.agents):
                dist_trav[i] += float(np.linalg.norm(ag.pos - prev_pos[i]))
                prev_pos[i] = ag.pos.copy()
            if done: break
        info = env._build_info()
        info["dist_travelled"] = dist_trav
        results.append(info)
    return results


def summarise(results, n_agents, max_steps):
    sr   = [r["n_done"] / n_agents for r in results]
    col  = [r["total_collisions"]  for r in results]
    gt   = [v for r in results for v in r["agent_goal_times"].values()]
    gt_s = [v if v is not None else max_steps * DT for v in gt]
    dist = [r["dist_travelled"][i] for r in results for i in range(n_agents)]
    spd  = []
    for r in results:
        for i in range(n_agents):
            gt_i = r["agent_goal_times"][i]
            if gt_i:
                spd.append(r["dist_travelled"][i] / max(gt_i, 1e-6))
    return dict(
        success_rate     = np.mean(sr),
        success_std      = np.std(sr),
        all_done_pct     = np.mean([r["n_done"]==n_agents for r in results])*100,
        avg_collisions   = np.mean(col),
        std_collisions   = np.std(col),
        avg_goal_time    = np.mean(gt_s),
        std_goal_time    = np.std(gt_s),
        avg_speed        = np.mean(spd) if spd else 0.0,
        avg_dist         = np.mean(dist),
    )


def print_report(trained, rand, scenario, n_agents, n_eps):
    w = 58
    def row(name, tv, rv, better="high"):
        tag = ""
        try:
            tv_f = float(tv.rstrip("%sm/"))
            rv_f = float(rv.rstrip("%sm/"))
            if better == "high":  tag = "✓" if tv_f >= rv_f else "✗"
            else:                 tag = "✓" if tv_f <= rv_f else "✗"
        except Exception: pass
        print(f"  {name:<22} {tv:>10}   {rv:>10}   {tag}")

    print(f"\n╔{'═'*w}╗")
    print(f"║  Scenario: {scenario:<{w-13}}║")
    print(f"║  N={n_agents}  episodes={n_eps}{'':<{w-20}}║")
    print(f"╠{'═'*w}╣")
    print(f"║  {'Metric':<22} {'PPO':>10}   {'Random':>10}   {'':3}║")
    print(f"║  {'─'*52}║")
    row("Success rate",    f"{trained['success_rate']*100:.1f}%",
                           f"{rand['success_rate']*100:.1f}%",     "high")
    row("All done %",      f"{trained['all_done_pct']:.1f}%",
                           f"{rand['all_done_pct']:.1f}%",         "high")
    row("Avg collisions",  f"{trained['avg_collisions']:.2f}",
                           f"{rand['avg_collisions']:.2f}",        "low")
    row("Avg goal time",   f"{trained['avg_goal_time']:.2f}s",
                           f"{rand['avg_goal_time']:.2f}s",        "low")
    row("Avg speed",       f"{trained['avg_speed']:.3f}m/s",
                           f"{rand['avg_speed']:.3f}m/s",          "high")
    print(f"╚{'═'*w}╝")


# ─────────────────────────────────────────────────────────────────────────────
# Static trajectory plot
# ─────────────────────────────────────────────────────────────────────────────

def make_trajectory_plot(frames_list, env_list, labels, save_path):
    """
    frames_list : list of frame lists, one per scenario
    env_list    : list of envs (for goals/starts)
    labels      : scenario name strings
    """
    n_scen = len(frames_list)
    fig    = plt.figure(figsize=(6 * n_scen, 6.5), facecolor=BG)
    axes   = [fig.add_subplot(1, n_scen, i+1) for i in range(n_scen)]

    for ax, frames, env, label in zip(axes, frames_list, env_list, labels):
        ax.set_facecolor(BG)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        G = env.grid_size
        ax.set_xlim(-0.3, G+0.3); ax.set_ylim(-0.3, G+0.3)
        ax.set_aspect("equal")
        ax.grid(True, color="#21262d", lw=0.5, ls="--", zorder=0)
        ax.set_title(label, color=FG, fontsize=10, fontfamily="monospace", pad=6)
        ax.tick_params(colors=MUTED, labelsize=7)

        n = env.n_agents
        colors = [PALETTE[i % len(PALETTE)] for i in range(n)]

        # build position history per agent
        hist = {i: [] for i in range(n)}
        collision_events = []   # (x, y) positions where collisions occurred
        prev_col = [0] * n
        for f in frames:
            for i, pos in enumerate(f["positions"]):
                hist[i].append(pos.copy())
            for i in range(n):
                if f["collisions"][i] > prev_col[i]:
                    collision_events.append(f["positions"][i].copy())
                    prev_col[i] = f["collisions"][i]

        # draw trajectories
        for i in range(n):
            h = np.array(hist[i])
            if len(h) > 1:
                ax.plot(h[:, 0], h[:, 1], "-", color=colors[i],
                        alpha=0.55, lw=1.3, zorder=2)

        # start circles
        for i in range(n):
            s = np.array(hist[i][0])
            ax.plot(*s, "o", ms=7, color=colors[i], alpha=0.9,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5)
            ax.text(s[0], s[1]+0.25, str(i), ha="center", va="bottom",
                    fontsize=6.5, color=colors[i], fontweight="bold", zorder=6)

        # goal stars
        for i, ag in enumerate(env.agents):
            ax.plot(*ag.goal, "*", ms=12, color=colors[i], alpha=0.7, zorder=4)

        # final positions
        for i in range(n):
            e = np.array(hist[i][-1])
            reached = frames[-1]["reached"][i]
            marker = "^" if reached else "x"
            ax.plot(*e, marker, ms=7, color=colors[i], alpha=0.9,
                    markeredgewidth=1.5, zorder=5)

        # collision markers
        for pos in collision_events:
            ax.plot(*pos, "x", ms=9, color=RED, alpha=0.7,
                    markeredgewidth=2, zorder=7)

        # legend
        handles = [Line2D([0],[0], marker="o", color="w", markerfacecolor=colors[i],
                          ms=6, label=f"A{i}") for i in range(n)]
        handles += [
            Line2D([0],[0], marker="*", color="w", markerfacecolor="white", ms=8, label="Goal"),
            Line2D([0],[0], marker="x", color=RED,  ms=8, lw=0, label="Collision"),
        ]
        ax.legend(handles=handles, loc="lower right", ncol=2,
                  facecolor=PANEL, edgecolor=BORDER, labelcolor=FG, fontsize=6.5)

        # stats text
        info = frames[-1]["info"]
        ax.text(0.02, 0.99,
                f"col={info['total_collisions']}  done={info['n_done']}/{n}"
                f"  t={info['sim_time']:.1f}s",
                transform=ax.transAxes, color=FG, fontsize=7.5, va="top",
                fontfamily="monospace")

    fig.suptitle("PPO Policy — Trajectory Evaluation",
                 color=FG, fontsize=12, fontfamily="monospace", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Trajectory plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Animated GIF
# ─────────────────────────────────────────────────────────────────────────────

def make_gif(frames, env, save_path, fps=12, policy=None):
    """
    Full-featured animation with:
      Left panel  : agent simulation (circles, trails, velocity arrows, flash)
      Right panel : live stats + cumulative reward chart
    """
    n = env.n_agents
    G = env.grid_size
    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]

    # build complete position history
    history = {i: [] for i in range(n)}
    for f in frames:
        for i, pos in enumerate(f["positions"]):
            history[i].append(pos.copy())

    # subsample to ≤60 GIF frames
    stride = max(1, len(frames) // 60)
    sel    = frames[::stride]

    # cumulative reward per agent over time
    cum_rewards = {i: [] for i in range(n)}
    running = [0.0] * n
    reward_times = []
    for f in frames:
        for i in range(n):
            running[i] += f["rewards"][i]
            cum_rewards[i].append(running[i])
        reward_times.append(f["sim_time"])

    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7), facecolor=BG)
    gs  = gridspec.GridSpec(
        2, 2,
        width_ratios=[2.2, 1],
        height_ratios=[2.2, 1],
        hspace=0.08, wspace=0.06,
        left=0.03, right=0.97, top=0.93, bottom=0.06
    )
    ax_sim   = fig.add_subplot(gs[:, 0])   # simulation (full left column)
    ax_stats = fig.add_subplot(gs[0, 1])   # stats panel (upper right)
    ax_rew   = fig.add_subplot(gs[1, 1])   # reward chart (lower right)

    for ax in (ax_sim, ax_stats, ax_rew):
        ax.set_facecolor(BG)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

    # ── simulation axis ──────────────────────────────────────────────────────
    ax_sim.set_xlim(-0.3, G+0.3); ax_sim.set_ylim(-0.3, G+0.3)
    ax_sim.set_aspect("equal")
    ax_sim.grid(True, color="#21262d", lw=0.5, ls="--", zorder=0)
    ax_sim.set_title("PPO Policy Rollout", color=FG, fontsize=11,
                     fontfamily="monospace", pad=7)
    ax_sim.tick_params(colors=MUTED, labelsize=7)

    # goal stars
    for i, ag in enumerate(env.agents):
        ax_sim.plot(*ag.goal, "*", ms=14, color=colors[i], alpha=0.55, zorder=3)

    # start halos
    for i in range(n):
        s = history[i][0]
        ax_sim.add_patch(plt.Circle(s, env.collision_radius * 1.6,
                                    color=colors[i], alpha=0.1, zorder=1))

    # dynamic: circles, labels, trails, velocity arrows, collision flashes
    circs  = [plt.Circle(history[i][0], env.collision_radius,
                          color=colors[i], alpha=0.88, zorder=5) for i in range(n)]
    for c in circs: ax_sim.add_patch(c)

    lbls   = [ax_sim.text(*history[i][0], str(i), ha="center", va="center",
                           fontsize=7, color="white", fontweight="bold", zorder=6)
              for i in range(n)]

    trails = [ax_sim.plot([], [], "-", color=colors[i], alpha=0.22,
                           lw=1.2, zorder=2)[0] for i in range(n)]

    # velocity arrows as quiver
    arrow_x = np.array([history[i][0][0] for i in range(n)])
    arrow_y = np.array([history[i][0][1] for i in range(n)])
    quiv = ax_sim.quiver(arrow_x, arrow_y,
                          np.zeros(n), np.zeros(n),
                          color=[colors[i] for i in range(n)],
                          scale=8, width=0.005, alpha=0.8, zorder=4)

    flashes = [plt.Circle((0, 0), 0, color=RED, alpha=0.0, zorder=7)
               for _ in range(n)]
    for f in flashes: ax_sim.add_patch(f)

    time_txt = ax_sim.text(0.01, 0.99, "", transform=ax_sim.transAxes,
                            color=FG, fontsize=8.5, va="top",
                            fontfamily="monospace", zorder=8)

    legend_handles = [mpatches.Patch(color=colors[i], label=f"A{i}")
                      for i in range(n)]
    ax_sim.legend(handles=legend_handles, loc="lower right", ncol=2,
                  facecolor=PANEL, edgecolor=BORDER, labelcolor=FG, fontsize=7)

    # ── stats axis ───────────────────────────────────────────────────────────
    ax_stats.set_xlim(0, 1); ax_stats.set_ylim(0, 1); ax_stats.axis("off")
    ax_stats.set_title("Live Stats", color=FG, fontsize=10,
                        fontfamily="monospace", pad=5)
    n_rows = n + 6
    stat_txts = [
        ax_stats.text(0.04, 0.97 - k * (0.92 / n_rows), "",
                       color=FG, fontsize=8, fontfamily="monospace", va="top")
        for k in range(n_rows)
    ]

    # ── reward chart axis ────────────────────────────────────────────────────
    ax_rew.set_xlim(0, max(reward_times) + 0.1)
    all_cum = [v for vals in cum_rewards.values() for v in vals]
    r_min, r_max = min(all_cum), max(all_cum)
    pad = max(abs(r_max - r_min) * 0.1, 0.5)
    ax_rew.set_ylim(r_min - pad, r_max + pad)
    ax_rew.set_xlabel("sim time (s)", color=MUTED, fontsize=7)
    ax_rew.set_ylabel("cum. reward",  color=MUTED, fontsize=7)
    ax_rew.tick_params(colors=MUTED, labelsize=6)
    ax_rew.axhline(0, color=BORDER, lw=0.8, ls="--")

    rew_lines = [ax_rew.plot([], [], "-", color=colors[i], lw=1.2, alpha=0.8)[0]
                 for i in range(n)]
    step_line = ax_rew.axvline(0, color="white", lw=0.8, alpha=0.4, ls=":")

    # ── animation function ───────────────────────────────────────────────────
    def _animate(fi):
        frame  = sel[fi]
        orig   = fi * stride   # approximate index into full frame list

        # ── sim panel ──────────────────────────────────────────────────────
        vx_arr = np.zeros(n); vy_arr = np.zeros(n)
        for i in range(n):
            pos     = frame["positions"][i]
            vel     = frame["velocities"][i]
            reached = frame["reached"][i]
            recov   = frame["recovery"][i]

            circs[i].center = pos
            circs[i].set_alpha(0.25 if reached else 0.88)
            lbls[i].set_position(pos)

            # trail (last 80 real frames)
            h_start = max(0, orig - 80)
            h       = np.array(history[i][h_start: orig + 2])
            if len(h) > 1:
                trails[i].set_data(h[:, 0], h[:, 1])

            # velocity for quiver
            speed = np.linalg.norm(vel)
            if speed > 0.05 and not reached:
                vx_arr[i] = vel[0] / speed * 0.6
                vy_arr[i] = vel[1] / speed * 0.6

            # collision flash
            flashes[i].center = pos
            flashes[i].set_radius(env.collision_radius * 2.2 if recov > 0 else 0)
            flashes[i].set_alpha(0.4 if recov > 0 else 0.0)

        # update quiver
        quiv.set_offsets(np.array([[frame["positions"][i][0],
                                     frame["positions"][i][1]]
                                    for i in range(n)]))
        quiv.set_UVC(vx_arr, vy_arr)

        t   = frame["sim_time"]
        tc  = frame["info"]["total_collisions"]
        nd  = frame["info"]["n_done"]
        time_txt.set_text(
            f"t={t:.1f}s  step={frame['step']}  col={tc}  done={nd}/{n}"
        )

        # ── stats panel ────────────────────────────────────────────────────
        stat_txts[0].set_text("── Episode ─────────────")
        stat_txts[1].set_text(f"  Sim time  : {t:.2f} s")
        stat_txts[2].set_text(f"  Collisions: {tc}")
        stat_txts[3].set_text(f"  Completed : {nd}/{n}")
        stat_txts[4].set_text(f"  Recovery  : {env.recovery_steps} steps")
        stat_txts[5].set_text("── Per-Agent ────────────")
        for i in range(n):
            reached = frame["reached"][i]
            recov   = frame["recovery"][i]
            ncol    = frame["collisions"][i]
            gt      = frame["goal_times"][i]
            r       = frame["rewards"][i]
            v       = frame["values"][i]
            sym     = "✓" if reached else ("❄" if recov > 0 else "→")
            gts     = f"{gt:.1f}s" if gt else "---"
            stat_txts[6 + i].set_text(
                f"  A{i} {sym}  col={ncol}  t={gts}"
                f"\n     r={r:+.2f}  V={v:+.2f}"
            )
            stat_txts[6 + i].set_color(
                GREEN if reached else (RED if recov > 0 else FG)
            )

        # ── reward chart ───────────────────────────────────────────────────
        end_idx = min(orig + 1, len(reward_times))
        for i in range(n):
            rew_lines[i].set_data(reward_times[:end_idx],
                                   cum_rewards[i][:end_idx])
        step_line.set_xdata([t, t])

        return circs + lbls + trails + flashes + [quiv, time_txt, step_line] + rew_lines

    # ── render ───────────────────────────────────────────────────────────────
    ani = animation.FuncAnimation(
        fig, _animate, frames=len(sel),
        interval=max(40, 1000 // fps), blit=False, repeat=False
    )
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    ani.save(save_path, writer=animation.PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    print(f"  GIF saved       → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(all_results, scenario_names, n_agents, max_steps, save_path):
    rows = []
    for results, scen in zip(all_results, scenario_names):
        for ep, info in enumerate(results):
            for i in range(n_agents):
                gt = info["agent_goal_times"][i]
                rows.append({
                    "scenario":       scen,
                    "episode":        ep,
                    "agent":          i,
                    "goal_time":      gt if gt else max_steps * DT,
                    "reached":        int(gt is not None),
                    "collisions":     info["agent_collisions"][i],
                    "dist_travelled": info["dist_travelled"][i],
                    "ep_total_col":   info["total_collisions"],
                    "ep_success_rate":info["n_done"] / n_agents,
                })
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved       → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--load_path",        default="outputs/policy_w4_2.pt")
    p.add_argument("--n_agents",  type=int,   default=DEFAULTS["n_agents"])
    p.add_argument("--grid_size", type=float, default=DEFAULTS["grid_size"])
    p.add_argument("--collision_radius", type=float, default=DEFAULTS["collision_radius"])
    p.add_argument("--recovery_steps",   type=int,   default=DEFAULTS["recovery_steps"])
    p.add_argument("--k_neighbors",      type=int,   default=DEFAULTS["k_neighbors"],
                   help="Must match training value")
    p.add_argument("--max_steps",        type=int,   default=DEFAULTS["max_steps"])
    p.add_argument("--n_episodes",       type=int,   default=30,
                   help="Episodes per scenario for the benchmark")
    p.add_argument("--scenario",         default="all",
                   choices=["all","random","circle","corridor"],
                   help="Which scenario(s) to test")
    p.add_argument("--stochastic",       action="store_true",
                   help="Sample from policy distribution (default: greedy)")
    p.add_argument("--no_baseline",      action="store_true")
    # reward weights (should match training)
    p.add_argument("--w1",  type=float, default=DEFAULTS["w1"])
    p.add_argument("--w2",  type=float, default=DEFAULTS["w2"])
    p.add_argument("--w3",  type=float, default=DEFAULTS["w3"])
    p.add_argument("--w4",  type=float, default=DEFAULTS["w4"])
    p.add_argument("--goal_bonus", type=float, default=DEFAULTS["goal_bonus"])
    # outputs
    p.add_argument("--save_gif",  default="outputs/eval_w4.gif",
                   help="Path for the animated GIF (one episode per scenario)")
    p.add_argument("--save_plot", default="outputs/trajectories_w4.png",
                   help="Path for the static trajectory plot")
    p.add_argument("--save_csv",  default=None,
                   help="Optional CSV path for per-episode per-agent results")
    p.add_argument("--fps",   type=int,   default=12)
    p.add_argument("--seed",  type=int,   default=2024)
    p.add_argument("--device",default="cpu")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg    = parse_args()
    greedy = not cfg.stochastic
    device = cfg.device

    print(f"\n{'='*60}")
    print(f"  PPO Policy — Test & Visualize")
    print(f"{'='*60}")
    print(f"  Checkpoint : {cfg.load_path}")
    print(f"  N agents   : {cfg.n_agents}  |  Grid: {cfg.grid_size}×{cfg.grid_size}")
    print(f"  Episodes   : {cfg.n_episodes}  |  Max steps: {cfg.max_steps}")
    print(f"  Mode       : {'greedy' if greedy else 'stochastic'}")

    policy, k, train_cfg = load_policy(cfg.load_path, cfg.k_neighbors, device)

    scenarios = (
        ["random", "circle", "corridor"]
        if cfg.scenario == "all"
        else [cfg.scenario]
    )

    # storage for trajectory plot and CSV
    all_vis_frames  = []   # one episode's frames per scenario
    all_vis_envs    = []
    all_bench_results = []
    scenario_labels = []

    for scen in scenarios:
        print(f"\n{'─'*60}")
        print(f"  ▶  {scen.upper()} scenario")
        print(f"{'─'*60}")

        def make_env_fn():
            return MultiAgentNavEnv(
                n_agents        = cfg.n_agents,
                grid_size       = cfg.grid_size,
                collision_radius= cfg.collision_radius,
                recovery_steps  = cfg.recovery_steps,
                mode            = MODE_RL_FULL,
                goal_threshold  = 0.28,
                seed            = cfg.seed,
            )

        env = make_env_fn()

        # fixed starts/goals for structured scenarios
        if scen == "circle":
            starts_fn = lambda: circle_config(cfg.n_agents, cfg.grid_size)[0]
            goals_fn  = lambda: circle_config(cfg.n_agents, cfg.grid_size)[1]
        elif scen == "corridor":
            starts_fn = lambda: corridor_config(cfg.n_agents, cfg.grid_size)[0]
            goals_fn  = lambda: corridor_config(cfg.n_agents, cfg.grid_size)[1]
        else:
            starts_fn = goals_fn = None

        # ── visualisation episode (one episode, full frames) ────────────────
        env.rng = np.random.default_rng(cfg.seed)
        S = starts_fn() if starts_fn else None
        G = goals_fn()  if goals_fn  else None
        vis_frames, vis_info = rollout(
            policy, env, k, cfg.max_steps, device,
            starts=S, goals=G,
            w1=cfg.w1, w2=cfg.w2, w3=cfg.w3, w4=cfg.w4, goal_bonus=cfg.goal_bonus,
            greedy=greedy,
        )
        all_vis_frames.append(vis_frames)
        all_vis_envs.append(env)

        print(f"  Vis episode: {vis_info['n_done']}/{cfg.n_agents} done  "
              f"| col={vis_info['total_collisions']}  "
              f"| t={vis_info['sim_time']:.1f}s")

        # ── benchmark ────────────────────────────────────────────────────────
        print(f"  Running {cfg.n_episodes} benchmark episodes …", end="", flush=True)
        t0 = time.time()
        bench = benchmark(
            policy, make_env_fn(), k, cfg.max_steps, cfg.n_episodes,
            device, scen,
            w1=cfg.w1, w2=cfg.w2, w3=cfg.w3, w4=cfg.w4, goal_bonus=cfg.goal_bonus,
            starts_fn=starts_fn, goals_fn=goals_fn, greedy=greedy,
        )
        print(f" {time.time()-t0:.1f}s")
        all_bench_results.append(bench)
        scenario_labels.append(scen)

        trained_sum = summarise(bench, cfg.n_agents, cfg.max_steps)

        if not cfg.no_baseline:
            print(f"  Running random baseline …", end="", flush=True)
            rand_env = make_env_fn()
            rand  = random_baseline(rand_env, cfg.max_steps, cfg.n_episodes,
                                    starts_fn=starts_fn, goals_fn=goals_fn)
            rand_sum  = summarise(rand, cfg.n_agents, cfg.max_steps)
            print(f" done")
            print_report(trained_sum, rand_sum, scen, cfg.n_agents, cfg.n_episodes)
        else:
            print_report(trained_sum, trained_sum, scen, cfg.n_agents, cfg.n_episodes)

        # per-agent breakdown
        print(f"\n  Per-agent (benchmark avg, {cfg.n_episodes} eps):")
        print(f"  {'Agent':^5} {'Goal time':^11} {'Collisions':^12} {'Dist':^8}")
        print(f"  {'─'*40}")
        for i in range(cfg.n_agents):
            gts   = [r["agent_goal_times"][i] or cfg.max_steps*DT for r in bench]
            cols  = [r["agent_collisions"][i]  for r in bench]
            dists = [r["dist_travelled"][i]    for r in bench]
            print(f"  {i:^5} {np.mean(gts):^11.3f} {np.mean(cols):^12.2f} "
                  f"{np.mean(dists):^8.2f}")

    # ── trajectory plot ───────────────────────────────────────────────────────
    print(f"\n  Generating trajectory plot …")
    make_trajectory_plot(
        all_vis_frames, all_vis_envs,
        [f"{s} (N={cfg.n_agents})" for s in scenarios],
        cfg.save_plot,
    )

    # ── animated GIFs (one per scenario) ────────────────────────────────────
    print(f"  Generating GIF(s) …")
    base, ext = os.path.splitext(cfg.save_gif)
    for i, (scen, frames, env) in enumerate(
        zip(scenarios, all_vis_frames, all_vis_envs)
    ):
        gif_path = f"{base}_{scen}{ext}" if len(scenarios) > 1 else cfg.save_gif
        make_gif(frames, env, gif_path, fps=cfg.fps, policy=policy)

    # ── CSV ──────────────────────────────────────────────────────────────────
    if cfg.save_csv:
        save_csv(all_bench_results, scenario_labels,
                 cfg.n_agents, cfg.max_steps, cfg.save_csv)

    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()