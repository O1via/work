import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _maybe_reexec_into_workspace_venv() -> None:
    """尽量让直接运行脚本时也使用工作区 .venv。"""
    if os.environ.get("RTMPC_NO_REEXEC") == "1":
        return

    this_file = Path(__file__).resolve()
    workspace_root = this_file.parent.parent
    venv_python = workspace_root / ".venv" / "bin" / "python"
    try:
        current = Path(sys.executable).resolve()
    except Exception:
        return

    if venv_python.exists() and current != venv_python.resolve():
        os.environ["RTMPC_NO_REEXEC"] = "1"
        os.execv(str(venv_python), [str(venv_python), *sys.argv])


_maybe_reexec_into_workspace_venv()

import numpy as np

from rtmpc_demo import (
    DoubleIntegrator,
    LinearIrisHover,
    apply_tracking_profile_iris,
    build_circle_reference,
    compute_infinite_lqr,
    solve_rtmc_qp_paper,
)
from rtmpc_constants import (
    base_initial_state,
    base_input_bounds,
    base_state_bounds,
    disturbance_half_bounds,
    input_cost_matrix,
    sample_process_disturbance,
    state_cost_matrix,
)


def build_reference(
    task: str,
    tracking_shape: str,
    tracking_profile: str,
    x0: np.ndarray,
    sim_steps: int,
    horizon: int,
    dt: float,
    circle_radius: float,
    circle_period_steps: int,
    line_goal_n: float,
    line_goal_e: float,
) -> np.ndarray:
    """构造参考轨迹，语义与 demo 保持一致。"""
    if tracking_profile not in ("paper_baseline", "high_speed_extension"):
        raise ValueError("tracking_profile 应为 'paper_baseline' 或 'high_speed_extension'")
    n = x0.shape[0]
    total_len = int(sim_steps) + int(horizon) + 1

    if task == "point":
        out = np.zeros((total_len, n), dtype=float)
        if n == 8:
            out[:, 4] = float(x0[4])
        return out

    if task != "tracking":
        raise ValueError("task 应为 'point' 或 'tracking'")

    if tracking_shape == "circle":
        period_steps = int(circle_period_steps)
        if period_steps <= 0:
            raise ValueError("circle_period_steps 必须为正")
        if n == 4:
            return build_circle_reference(
                x0=x0,
                total_len=total_len,
                dt=float(dt),
                radius=float(circle_radius),
                period_steps=period_steps,
            )
        if n == 8:
            xy_ref = build_circle_reference(
                x0=x0[:4],
                total_len=total_len,
                dt=float(dt),
                radius=float(circle_radius),
                period_steps=period_steps,
            )
            out = np.zeros((total_len, n), dtype=float)
            out[:, 0] = xy_ref[:, 0]
            out[:, 1] = xy_ref[:, 1]
            out[:, 2] = xy_ref[:, 2]
            out[:, 3] = xy_ref[:, 3]
            out[:, 4] = float(x0[4])
            x_min_base, x_max_base = base_state_bounds("iris_linear")
            out = apply_tracking_profile_iris(
                out,
                dt=float(dt),
                tracking_profile=tracking_profile,
                phi_bounds=(float(x_min_base[6]), float(x_max_base[6])),
                theta_bounds=(float(x_min_base[7]), float(x_max_base[7])),
            )
            return out
        raise ValueError(f"unsupported state dimension for circle tracking: n={n}")

    if tracking_shape == "line":
        p_start = x0[:2].copy()
        p_goal = np.array([float(line_goal_n), float(line_goal_e)], dtype=float)
        t_steps = max(1, int(sim_steps))

        alphas = np.linspace(0.0, 1.0, t_steps + 1)
        pos_move = (1.0 - alphas)[:, None] * p_start[None, :] + alphas[:, None] * p_goal[None, :]
        pos_ref = np.vstack([pos_move, np.tile(p_goal[None, :], (total_len - (t_steps + 1), 1))])
        v_const = (p_goal - p_start) / (t_steps * float(dt))
        vel_ref = np.vstack(
            [
                np.tile(v_const[None, :], (t_steps + 1, 1)),
                np.zeros((total_len - (t_steps + 1), 2), dtype=float),
            ]
        )

        if n == 4:
            return np.hstack([pos_ref, vel_ref])
        if n == 8:
            out = np.zeros((total_len, n), dtype=float)
            out[:, 0:2] = pos_ref
            out[:, 2:4] = vel_ref
            out[:, 4] = float(x0[4])
            x_min_base, x_max_base = base_state_bounds("iris_linear")
            out = apply_tracking_profile_iris(
                out,
                dt=float(dt),
                tracking_profile=tracking_profile,
                phi_bounds=(float(x_min_base[6]), float(x_max_base[6])),
                theta_bounds=(float(x_min_base[7]), float(x_max_base[7])),
            )
            return out
        raise ValueError(f"unsupported state dimension for line tracking: n={n}")

    raise ValueError("tracking_shape 应为 'line' 或 'circle'")


def sample_initial_state(
    base_x0: np.ndarray,
    rng: np.random.Generator,
    pos_jitter: float,
    vel_jitter: float,
    pd_jitter: float,
    vd_jitter: float,
) -> np.ndarray:
    """对初始状态做小范围随机扰动，提升数据覆盖。"""
    x0 = np.asarray(base_x0, dtype=float).copy()
    if x0.size >= 2 and pos_jitter > 0.0:
        x0[:2] += rng.uniform(-pos_jitter, pos_jitter, size=2)
    if x0.size >= 4 and vel_jitter > 0.0:
        x0[2:4] += rng.uniform(-vel_jitter, vel_jitter, size=2)
    if x0.size >= 5 and pd_jitter > 0.0:
        x0[4] += float(rng.uniform(-pd_jitter, pd_jitter))
    if x0.size >= 6 and vd_jitter > 0.0:
        x0[5] += float(rng.uniform(-vd_jitter, vd_jitter))
    return x0


def rollout_collect_episode(
    *,
    sim,
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    Qx: np.ndarray,
    Ru: np.ndarray,
    Px: np.ndarray,
    x0: np.ndarray,
    x_ref_all: np.ndarray,
    sim_steps: int,
    horizon: int,
    disturbance_mode: str,
    force_bound_mg: float,
    rng: np.random.Generator,
    w_half: np.ndarray,
    x_bounds_for_qp: Optional[Tuple[np.ndarray, np.ndarray]],
    u_bounds_for_qp: Optional[Tuple[np.ndarray, np.ndarray]],
    u_min_base: np.ndarray,
    u_max_base: np.ndarray,
    x_min_base: np.ndarray,
    x_max_base: np.ndarray,
    qp_fallback: str,
    tube_eps: float,
) -> Dict[str, np.ndarray]:
    """采集单个 episode 的转移数据。"""
    n, m = A.shape[0], B.shape[1]
    x = np.asarray(x0, dtype=float).copy()
    z_zero = np.full(n, float(tube_eps), dtype=float)

    xs: List[np.ndarray] = [x.copy()]
    us: List[np.ndarray] = []
    ws: List[np.ndarray] = []
    refs: List[np.ndarray] = []

    qps_solved = 0
    qps_failed = 0
    state_violations = 0

    for t in range(int(sim_steps)):
        x_des = x_ref_all[t : t + horizon + 1]
        try:
            _, u_bar = solve_rtmc_qp_paper(
                A=A,
                B=B,
                Qx=Qx,
                Ru=Ru,
                Px=Px,
                x_meas=x,
                x_des=x_des,
                N=horizon,
                z_half=z_zero,
                x_bounds=x_bounds_for_qp,
                u_bounds=u_bounds_for_qp,
                d_affine=None,
            )
            u = np.asarray(u_bar[0], dtype=float).reshape(-1)
            qps_solved += 1
        except Exception:
            qps_failed += 1
            if qp_fallback == "stop":
                break
            if qp_fallback == "lqr":
                e = x - x_des[0]
                u = np.asarray(K @ e, dtype=float).reshape(-1)
            elif qp_fallback == "zero":
                u = np.zeros(m, dtype=float)
            else:
                raise ValueError("qp_fallback 应为 stop|lqr|zero")

        u = np.clip(u, u_min_base, u_max_base)

        if np.any(x < x_min_base) or np.any(x > x_max_base):
            state_violations += 1

        w = sample_process_disturbance(
            rng=rng,
            dynamics="double_integrator" if n == 4 else "iris_linear",
            dt=float(sim.dt),
            mode=disturbance_mode,
            force_bound_mg=float(force_bound_mg),
            state_dim=n,
            w_half=w_half,
        )
        x_next = sim.step(x, u) + w

        refs.append(x_des[0].copy())
        us.append(u.copy())
        ws.append(w.copy())
        xs.append(x_next.copy())
        x = x_next

    xs_arr = np.asarray(xs, dtype=float)
    us_arr = np.asarray(us, dtype=float).reshape(-1, m)
    ws_arr = np.asarray(ws, dtype=float).reshape(-1, n)
    refs_arr = np.asarray(refs, dtype=float).reshape(-1, n)
    return {
        "xs": xs_arr,
        "us": us_arr,
        "ws": ws_arr,
        "refs": refs_arr,
        "qps_solved": np.asarray([qps_solved], dtype=int),
        "qps_failed": np.asarray([qps_failed], dtype=int),
        "state_violations": np.asarray([state_violations], dtype=int),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect transition data for GP residual training")
    parser.add_argument("--dynamics", choices=["double_integrator", "iris_linear"], default="iris_linear")
    parser.add_argument("--task", choices=["point", "tracking"], default="tracking")
    parser.add_argument("--tracking-shape", choices=["line", "circle"], default="circle")
    parser.add_argument(
        "--tracking-profile",
        choices=["paper_baseline", "high_speed_extension"],
        default="paper_baseline",
        help="tracking 参考模式：paper_baseline=phi/theta参考为0；high_speed_extension=由速度差分反解姿态参考。",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--sim-steps", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--disturbance-mode", choices=["state_box", "force_only"], default="force_only")
    parser.add_argument("--force-bound-mg", type=float, default=0.05)
    parser.add_argument("--state-box-scale", type=float, default=1.0)
    parser.add_argument("--qp-state-bounds", choices=["base", "none"], default="base")
    parser.add_argument("--qp-input-bounds", choices=["base", "none"], default="base")
    parser.add_argument("--qp-fallback", choices=["stop", "lqr", "zero"], default="lqr")
    parser.add_argument("--tube-eps", type=float, default=1e-6, help="QP 初值tube半宽的极小正数，避免 z=0 冲突")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pos-jitter", type=float, default=0.2)
    parser.add_argument("--vel-jitter", type=float, default=0.1)
    parser.add_argument("--pd-jitter", type=float, default=0.15)
    parser.add_argument("--vd-jitter", type=float, default=0.05)
    parser.add_argument("--circle-radius", type=float, default=4.0)
    parser.add_argument("--circle-period-steps", type=int, default=63, help="圆轨迹每圈步数")
    parser.add_argument("--line-goal-n", type=float, default=0.0)
    parser.add_argument("--line-goal-e", type=float, default=0.0)
    parser.add_argument(
        "--out",
        type=str,
        default="gp_data/transitions.npz",
        help="Output transition npz path (default: gp_data/transitions.npz)",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("episodes 必须为正")
    if args.sim_steps <= 0:
        raise ValueError("sim_steps 必须为正")
    if args.horizon <= 0:
        raise ValueError("horizon 必须为正")
    if args.state_box_scale <= 0.0:
        raise ValueError("state_box_scale 必须为正")
    if args.tube_eps <= 0.0:
        raise ValueError("tube_eps 必须为正")

    if args.dynamics == "double_integrator":
        sim = DoubleIntegrator(dt=0.1)
    else:
        sim = LinearIrisHover(dt=0.1, mass=1.5)

    A, B = sim.A, sim.B
    n, m = A.shape[0], B.shape[1]

    Qx = state_cost_matrix(args.dynamics)
    Ru = input_cost_matrix(args.dynamics, m)
    Px, K = compute_infinite_lqr(A, B, Qx, Ru)

    x_min_base, x_max_base = base_state_bounds(args.dynamics)
    if args.dynamics == "double_integrator":
        u_min_base, u_max_base = base_input_bounds(args.dynamics, m=m)
    else:
        u_min_base, u_max_base = base_input_bounds(args.dynamics, mass=float(sim.mass))
    x0_base = base_initial_state(args.dynamics)

    w_half = disturbance_half_bounds(
        args.dynamics,
        dt=float(sim.dt),
        mode=args.disturbance_mode,
        force_bound_mg=float(args.force_bound_mg),
    )
    if args.disturbance_mode == "state_box":
        w_half = np.asarray(w_half, dtype=float) * float(args.state_box_scale)

    x_bounds_for_qp = None
    u_bounds_for_qp = None
    if args.qp_state_bounds == "base":
        x_bounds_for_qp = (x_min_base, x_max_base)
    if args.qp_input_bounds == "base":
        u_bounds_for_qp = (u_min_base, u_max_base)

    rng = np.random.default_rng(seed=int(args.seed))

    x_t_all: List[np.ndarray] = []
    u_t_all: List[np.ndarray] = []
    x_tp1_all: List[np.ndarray] = []
    w_t_all: List[np.ndarray] = []
    ref_t_all: List[np.ndarray] = []
    ep_id_all: List[np.ndarray] = []
    step_id_all: List[np.ndarray] = []

    total_qps_solved = 0
    total_qps_failed = 0
    total_state_violations = 0
    valid_episodes = 0

    for ep in range(int(args.episodes)):
        x0 = sample_initial_state(
            x0_base,
            rng=rng,
            pos_jitter=float(args.pos_jitter),
            vel_jitter=float(args.vel_jitter),
            pd_jitter=float(args.pd_jitter),
            vd_jitter=float(args.vd_jitter),
        )
        x_ref_all = build_reference(
            task=args.task,
            tracking_shape=args.tracking_shape,
            tracking_profile=args.tracking_profile,
            x0=x0,
            sim_steps=int(args.sim_steps),
            horizon=int(args.horizon),
            dt=float(sim.dt),
            circle_radius=float(args.circle_radius),
            circle_period_steps=int(args.circle_period_steps),
            line_goal_n=float(args.line_goal_n),
            line_goal_e=float(args.line_goal_e),
        )

        rollout = rollout_collect_episode(
            sim=sim,
            A=A,
            B=B,
            K=K,
            Qx=Qx,
            Ru=Ru,
            Px=Px,
            x0=x0,
            x_ref_all=x_ref_all,
            sim_steps=int(args.sim_steps),
            horizon=int(args.horizon),
            disturbance_mode=args.disturbance_mode,
            force_bound_mg=float(args.force_bound_mg),
            rng=rng,
            w_half=w_half,
            x_bounds_for_qp=x_bounds_for_qp,
            u_bounds_for_qp=u_bounds_for_qp,
            u_min_base=u_min_base,
            u_max_base=u_max_base,
            x_min_base=x_min_base,
            x_max_base=x_max_base,
            qp_fallback=args.qp_fallback,
            tube_eps=float(args.tube_eps),
        )

        xs = rollout["xs"]
        us = rollout["us"]
        ws = rollout["ws"]
        refs = rollout["refs"]
        steps = us.shape[0]

        total_qps_solved += int(rollout["qps_solved"][0])
        total_qps_failed += int(rollout["qps_failed"][0])
        total_state_violations += int(rollout["state_violations"][0])

        if steps <= 0:
            print(f"[ep {ep+1:03d}] collected=0 (solver failed before first step)")
            continue

        valid_episodes += 1
        x_t = xs[:-1]
        x_tp1 = xs[1:]
        x_t_all.append(x_t)
        u_t_all.append(us)
        x_tp1_all.append(x_tp1)
        w_t_all.append(ws)
        ref_t_all.append(refs)
        ep_id_all.append(np.full((steps,), ep, dtype=int))
        step_id_all.append(np.arange(steps, dtype=int))

        print(
            f"[ep {ep+1:03d}] steps={steps} | "
            f"qps_solved={int(rollout['qps_solved'][0])} qps_failed={int(rollout['qps_failed'][0])} | "
            f"state_viol={int(rollout['state_violations'][0])}"
        )

    if not x_t_all:
        raise RuntimeError("未采集到任何有效转移样本，请放宽设置后重试")

    x_t_arr = np.vstack(x_t_all)
    u_t_arr = np.vstack(u_t_all)
    x_tp1_arr = np.vstack(x_tp1_all)
    w_t_arr = np.vstack(w_t_all)
    ref_t_arr = np.vstack(ref_t_all)
    ep_id_arr = np.concatenate(ep_id_all)
    step_id_arr = np.concatenate(step_id_all)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        x_t=x_t_arr,
        u_t=u_t_arr,
        x_tp1=x_tp1_arr,
        w_t=w_t_arr,
        refs=ref_t_arr,
        episode_id=ep_id_arr,
        step_id=step_id_arr,
        dynamics=np.array([args.dynamics]),
        task=np.array([args.task]),
        tracking_shape=np.array([args.tracking_shape]),
        tracking_profile=np.array([args.tracking_profile]),
        disturbance_mode=np.array([args.disturbance_mode]),
        force_bound_mg=np.array([float(args.force_bound_mg)], dtype=float),
        dt=np.array([float(sim.dt)], dtype=float),
        horizon=np.array([int(args.horizon)], dtype=int),
        sim_steps=np.array([int(args.sim_steps)], dtype=int),
        episodes=np.array([int(args.episodes)], dtype=int),
        valid_episodes=np.array([int(valid_episodes)], dtype=int),
        seed=np.array([int(args.seed)], dtype=int),
        qp_state_bounds=np.array([args.qp_state_bounds]),
        qp_input_bounds=np.array([args.qp_input_bounds]),
        qp_fallback=np.array([args.qp_fallback]),
        tube_eps=np.array([float(args.tube_eps)], dtype=float),
    )

    print("")
    print(f"[done] saved dataset: {out_path}")
    print(f"[done] transitions: {x_t_arr.shape[0]} | state_dim={n} | input_dim={m}")
    print(
        f"[done] qps_solved={total_qps_solved}, qps_failed={total_qps_failed}, "
        f"state_violations={total_state_violations}, valid_episodes={valid_episodes}/{args.episodes}"
    )
    print("")
    print("[next] fit GP example:")
    print(
        "python py/gp_residual_model.py "
        f"--dynamics {args.dynamics} --dt {float(sim.dt):.3f} "
        f"--data {out_path} --out gp_model/{args.dynamics}_residual_gp.npz"
    )


if __name__ == "__main__":
    main()
