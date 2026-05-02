"""
demo.py — Run both control modes and save visualisation videos.

Run:
    cd multi_agent_nav
    python demo.py

Outputs: mpc_stop_demo.gif, rl_full_demo.gif (and summary tables)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import torch
from env.multi_agent_env   import MultiAgentNavEnv, MODE_MPC_STOP, MODE_RL_FULL
from controllers.controllers import MPCStopController, RLFullController
from train import ActorCritic
# ──────────────────────────────────────────────────────────────────────────────
# Shared palette
# ──────────────────────────────────────────────────────────────────────────────
PALETTE = [
    "#e63946","#457b9d","#2a9d8f","#e9c46a","#f4a261",
    "#264653","#6d6875","#b5838d","#8338ec","#06d6a0",
]


def run_and_save(mode: str, n_agents: int = 8, grid: float = 10.0,
                 max_steps: int = 600, seed: int = 7,
                 out_path: str = "out.gif", policy: ActorCritic = None):

    env  = MultiAgentNavEnv(
        n_agents       = n_agents,
        grid_size      = grid,
        collision_radius = 0.30,
        recovery_steps = 6,
        mode           = mode,
        goal_threshold = 0.28,
        seed           = seed,
    )

    if mode == MODE_MPC_STOP:
        ctrl = MPCStopController(n_agents=n_agents, stop_threshold=0.85)
    else:
        ctrl = RLFullController(n_agents=n_agents, rl_policy=policy)

    # ── pre-simulate ──────────────────────────────────────────────────────────
    obs = env.reset()
    history   = {i: [env.agents[i].pos.copy()] for i in range(n_agents)}
    frames    = []          # (positions, velocities, info, agent_meta)

    for _ in range(max_steps):
        actions = ctrl.act(obs)
        obs, _, done, info = env.step(actions)
        for i, ag in enumerate(env.agents):
            history[i].append(ag.pos.copy())
        frames.append((
            [a.pos.copy()   for a in env.agents],
            [a.vel.copy()   for a in env.agents],
            dict(info),
            [(a.reached_goal, a.recovery_steps_left, a.collisions, a.goal_time)
             for a in env.agents],
        ))
        if done:
            break

    colors = [PALETTE[i % len(PALETTE)] for i in range(n_agents)]
    G = grid

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 6.5), facecolor="#0d1117")
    gs  = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.06,
                           left=0.04, right=0.97, top=0.93, bottom=0.07)
    ax  = fig.add_subplot(gs[0])
    axs = fig.add_subplot(gs[1])

    for a in (ax, axs):
        a.set_facecolor("#0d1117")
        for sp in a.spines.values():
            sp.set_edgecolor("#30363d")

    ax.set_xlim(-0.3, G+0.3);  ax.set_ylim(-0.3, G+0.3)
    ax.set_aspect("equal")
    ax.grid(True, color="#21262d", lw=0.5, ls="--")
    mode_label = "MPC-path + RL-Stop" if mode == MODE_MPC_STOP else "RL Full (accel+steer)"
    ax.set_title(f"Multi-Agent Nav  |  N={n_agents}  |  {mode_label}",
                 color="#e6edf3", fontsize=10, pad=7, fontfamily="monospace")
    ax.tick_params(colors="#484f58", labelsize=7)

    # goal stars
    for i, ag in enumerate(env.agents):
        ax.plot(*ag.goal, marker="*", ms=13, color=colors[i], alpha=0.65, zorder=3)

    # MPC path ghosts (dashed lines, mode 1 only)
    if mode == MODE_MPC_STOP:
        for i, ag in enumerate(env.agents):
            if ag.path:
                px = [ag.path[0][0]] + [p[0] for p in ag.path]
                py = [ag.path[0][1]] + [p[1] for p in ag.path]
                ax.plot(px, py, "--", color=colors[i], alpha=0.12, lw=0.8, zorder=1)

    # dynamic elements
    circles   = [plt.Circle(env.agents[i].pos, env.agents[i].radius,
                            color=colors[i], alpha=0.85, zorder=5) for i in range(n_agents)]
    for c in circles: ax.add_patch(c)

    labels    = [ax.text(*env.agents[i].pos, str(i), ha="center", va="center",
                         fontsize=6.5, color="white", fontweight="bold", zorder=6)
                 for i in range(n_agents)]

    trails    = [ax.plot([], [], "-", color=colors[i], alpha=0.22, lw=1.0, zorder=2)[0]
                 for i in range(n_agents)]

    flashes   = [plt.Circle((0,0), 0, color="#ff7b72", alpha=0.0, zorder=7)
                 for _ in range(n_agents)]
    for fc in flashes: ax.add_patch(fc)

    vel_lines = [ax.plot([], [], "-", color=colors[i], alpha=0.75, lw=1.8, zorder=4)[0]
                 for i in range(n_agents)]

    ttext     = ax.text(0.01, 0.99, "", transform=ax.transAxes,
                        color="#e6edf3", fontsize=8, va="top", fontfamily="monospace")

    legend_handles = [mpatches.Patch(color=colors[i], label=f"A{i}") for i in range(n_agents)]
    ax.legend(handles=legend_handles, loc="lower right", ncol=2,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7)

    # stats panel
    axs.set_xlim(0,1); axs.set_ylim(0,1); axs.axis("off")
    axs.set_title("Stats", color="#e6edf3", fontsize=10,
                  fontfamily="monospace", pad=7)
    slines = [axs.text(0.04, 0.97-k*0.062, "", color="#e6edf3",
                       fontsize=8, fontfamily="monospace", va="top")
              for k in range(n_agents + 8)]

    def _animate(fi):
        if fi >= len(frames): return
        pos_list, vel_list, info, meta = frames[fi]

        for i in range(n_agents):
            pos = pos_list[i]; vel = vel_list[i]
            reached, recov, ncol, gt = meta[i]

            circles[i].center = pos
            circles[i].set_alpha(0.3 if reached else 0.88)
            labels[i].set_position(pos)

            spd = np.linalg.norm(vel)
            if spd > 0.06 and not reached:
                tip = pos + vel/spd * 0.55
                vel_lines[i].set_data([pos[0], tip[0]], [pos[1], tip[1]])
            else:
                vel_lines[i].set_data([], [])

            hist = history[i][:fi+2]
            if len(hist) > 1:
                h = np.array(hist[-80:])
                trails[i].set_data(h[:,0], h[:,1])

            flashes[i].center = pos
            if recov > 0:
                flashes[i].set_radius(env.collision_radius * 2.0)
                flashes[i].set_alpha(0.35)
            else:
                flashes[i].set_alpha(0.0)

        t  = info["sim_time"]
        tc = info["total_collisions"]
        nd = info["n_done"]
        ttext.set_text(f"t={t:.1f}s | step={fi} | collisions={tc} | done={nd}/{n_agents}")

        slines[0].set_text("── Simulation ────────")
        slines[1].set_text(f"  Time      : {t:.2f} s")
        slines[2].set_text(f"  Collisions: {tc}")
        slines[3].set_text(f"  Completed : {nd}/{n_agents}")
        slines[4].set_text(f"  Recovery  : {env.recovery_steps} steps")
        slines[5].set_text("── Per-Agent ─────────")
        for i in range(n_agents):
            reached, recov, ncol, gt = meta[i]
            sym = "✓" if reached else ("❄" if recov > 0 else "→")
            gts = f"{gt:.1f}s" if gt else "---"
            slines[6+i].set_text(f"  A{i} {sym}  col={ncol}  t={gts}")

    ani = animation.FuncAnimation(fig, _animate, frames=len(frames),
                                  interval=60, blit=False, repeat=False)

    writer = animation.PillowWriter(fps=15)
    ani.save(out_path, writer=writer, dpi=110)
    plt.close(fig)
    print(f"[demo] Saved: {out_path}  ({len(frames)} frames)")

    # ── print summary ─────────────────────────────────────────────────────────
    final_info = env._build_info()
    print(f"\n{'='*52}")
    print(f"  Mode: {mode}  |  N={n_agents}  |  Grid={grid}x{grid}")
    print(f"{'='*52}")
    print(f"  Total collisions : {final_info['total_collisions']}")
    print(f"  Agents completed : {final_info['n_done']}/{n_agents}")
    print(f"  {'Agent':^5}  {'Goal Time':^12}  {'Collisions':^10}")
    print(f"  {'-'*34}")
    for i in range(n_agents):
        gt  = final_info["agent_goal_times"].get(i)
        nc  = final_info["agent_collisions"].get(i, 0)
        gts = f"{gt:.3f}s" if gt else "  DNF  "
        print(f"  {i:^5}  {gts:^12}  {nc:^10}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    # print("Running MODE_MPC_STOP demo …")
    # run_and_save(MODE_MPC_STOP, n_agents=10, seed=42,
    #              out_path="outputs/mpc_stop_demo.gif")

    print("Running MODE_RL_FULL demo …")
    rl_policy = torch.load('outputs/policy.pt', weights_only=True)
    # rl_policy.load_state_dict(torch.load('outputs/policy.pt'))
    run_and_save(MODE_RL_FULL,  n_agents=10, seed=42,
                 out_path="outputs/rl_full_demo.gif", policy=rl_policy)