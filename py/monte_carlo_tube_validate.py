import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from alg1_dagger import make_reference
from gp_residual_model import VelocityResidualGP, residual_shrink_bounds
from rtmpc_demo import (
    DoubleIntegrator,
    LinearIrisHover,
    compute_infinite_lqr,
    compute_rpi_box,
    solve_rtmc_qp_paper,
    tighten_box_bounds_with_auto_scale,
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


def _arr_str(x: np.ndarray) -> str:
    return np.array2string(np.asarray(x, dtype=float), precision=6, floatmode="fixed", suppress_small=False)


def build_context(
    dynamics: str,
    task: str,
    sim_steps: int,
    horizon: int,
    tracking_profile: str,
    disturbance_mode: str,
    force_bound_mg: float,
    force_d_axis_scale: float,
    gp_model_path: Optional[str] = None,
    gp_beta_sigma: float = 2.0,
    gp_shrink_mode: str = "residual",
) -> Dict[str, np.ndarray]:
    if dynamics == "double_integrator":
        sim = DoubleIntegrator(dt=0.1)
    else:
        sim = LinearIrisHover(dt=0.1, mass=1.5)

    A, B = sim.A, sim.B
    n, m = A.shape[0], B.shape[1]

    Qx = state_cost_matrix(dynamics)
    Ru = input_cost_matrix(dynamics, m)
    Px, K = compute_infinite_lqr(A, B, Qx, Ru)

    w_half_base = disturbance_half_bounds(
        dynamics,
        dt=float(sim.dt),
        mode="state_box",
        force_bound_mg=float(force_bound_mg),
        force_d_axis_scale=float(force_d_axis_scale),
    )
    w_half_from_force = disturbance_half_bounds(
        dynamics,
        dt=float(sim.dt),
        mode="force_only",
        force_bound_mg=float(force_bound_mg),
        force_d_axis_scale=float(force_d_axis_scale),
    )
    w_half = disturbance_half_bounds(
        dynamics,
        dt=float(sim.dt),
        mode=disturbance_mode,
        force_bound_mg=float(force_bound_mg),
        force_d_axis_scale=float(force_d_axis_scale),
    )
    w_half_nominal = w_half.copy()
    x_min_base, x_max_base = base_state_bounds(dynamics)
    if dynamics == "double_integrator":
        u_min_base, u_max_base = base_input_bounds(dynamics, m=m)
    else:
        u_min_base, u_max_base = base_input_bounds(dynamics, mass=float(sim.mass))

    gp_model = None
    gp_unc_half = np.zeros_like(w_half)
    gp_comp_half = np.zeros_like(w_half)
    if gp_model_path:
        gp_path = Path(gp_model_path)
        if not gp_path.exists():
            raise FileNotFoundError(f"GP model not found: {gp_path}")
        gp_model = VelocityResidualGP.load(str(gp_path))
        if gp_model.dynamics != dynamics:
            raise ValueError(
                f"GP dynamics mismatch: model={gp_model.dynamics}, current={dynamics}"
            )
        if abs(float(gp_model.dt) - float(sim.dt)) > 1e-9:
            raise ValueError(
                f"GP dt mismatch: model={gp_model.dt}, current={sim.dt}"
            )
        if int(gp_model.state_dim) != int(n):
            raise ValueError(
                f"GP state_dim mismatch: model={gp_model.state_dim}, current={n}"
            )
        gp_unc_half = gp_model.conservative_uncertainty_bound(
            x_min=x_min_base,
            x_max=x_max_base,
            beta_sigma=float(gp_beta_sigma),
        )
        gp_comp_half = gp_model.conservative_mean_bound(
            x_min=x_min_base,
            x_max=x_max_base,
        )
        print(f"[gp] loaded model: {gp_path}")
        print(f"[gp] uncertainty bound (beta={gp_beta_sigma:.2f}): {gp_unc_half}")
        print(f"[gp] compensable mean bound: {gp_comp_half}")

    w_half = residual_shrink_bounds(
        base_w_half=w_half,
        gp_comp_half=gp_comp_half,
        gp_unc_half=gp_unc_half,
        mode=gp_shrink_mode,
    )
    print(f"[tube] base disturbance bound ({disturbance_mode}): {w_half_nominal}")
    print(f"[tube] residual disturbance bound after GP shrink (mode={gp_shrink_mode}): {w_half}")
    # 注意：state_box 与 force_only 是两套独立扰动定义，不做相减混合。
    w_half_force = w_half_from_force
    A_cl = A + B @ K
    z_half = compute_rpi_box(A_cl, w_half)
    u_half = np.abs(K) @ z_half

    x_min_t, x_max_t, gamma_x = tighten_box_bounds_with_auto_scale(x_min_base, x_max_base, z_half, name="state")
    u_min_t, u_max_t, gamma_u = tighten_box_bounds_with_auto_scale(u_min_base, u_max_base, u_half, name="input")

    # 逐维可行性诊断：gamma_dim = width/(2*shrink)，最小值对应瓶颈维度。
    eps = 1e-12
    x_width = x_max_base - x_min_base
    u_width = u_max_base - u_min_base
    gamma_x_per_dim = np.where(z_half > eps, x_width / (2.0 * z_half), np.inf)
    gamma_u_per_dim = np.where(u_half > eps, u_width / (2.0 * u_half), np.inf)
    bottleneck_x_idx = int(np.argmin(gamma_x_per_dim))
    bottleneck_u_idx = int(np.argmin(gamma_u_per_dim))

    # 严格检查 full-shrink 可行性：若出现上界<=下界，直接终止并报告 gamma。
    x_min_full = x_min_base + z_half
    x_max_full = x_max_base - z_half
    u_min_full = u_min_base + u_half
    u_max_full = u_max_base - u_half
    x_infeasible = bool(np.any(x_max_full <= x_min_full))
    u_infeasible = bool(np.any(u_max_full <= u_min_full))
    if x_infeasible or u_infeasible:
        print(f"[diag] z_half={z_half.tolist()}")
        print(f"[diag] u_half={u_half.tolist()}")
        print(f"[diag] u_width={u_width.tolist()}")
        print(f"[diag] gamma_u_per_dim={gamma_u_per_dim.tolist()}")
        raise RuntimeError(
            "full-shrink 不可行：检测到收紧后上界<=下界。"
            f" gamma_x={gamma_x:.6f}, gamma_u={gamma_u:.6f}."
            f" bottleneck_x_dim={bottleneck_x_idx}, bottleneck_x_gamma={gamma_x_per_dim[bottleneck_x_idx]:.6f};"
            f" bottleneck_u_dim={bottleneck_u_idx}, bottleneck_u_gamma={gamma_u_per_dim[bottleneck_u_idx]:.6f}."
            " 请先降低扰动边界或放宽原始约束。"
        )

    print(
        f"[diag] gamma_x={gamma_x:.6f}, gamma_u={gamma_u:.6f}; "
        f"bottleneck_x_dim={bottleneck_x_idx} (gamma={gamma_x_per_dim[bottleneck_x_idx]:.6f}), "
        f"bottleneck_u_dim={bottleneck_u_idx} (gamma={gamma_u_per_dim[bottleneck_u_idx]:.6f})"
    )

    x0 = base_initial_state(dynamics)
    x_ref_all = make_reference(
        task,
        x0=x0,
        N=horizon,
        sim_dt=sim.dt,
        total_steps=sim_steps,
        tracking_profile=tracking_profile,
    )
    if task == "tracking":
        x0 = x_ref_all[0].copy()

    return {
        "A": A,
        "B": B,
        "K": K,
        "Qx": Qx,
        "Ru": Ru,
        "Px": Px,
        "sim": sim,
        "n": np.array([n]),
        "m": np.array([m]),
        "w_half": w_half,
        "w_half_base": w_half_base,
        "w_half_force": w_half_force,
        "gp_unc_half": gp_unc_half,
        "gp_comp_half": gp_comp_half,
        "z_half": z_half,
        "u_half": u_half,
        "x_min_base": x_min_base,
        "x_max_base": x_max_base,
        "u_min_base": u_min_base,
        "u_max_base": u_max_base,
        "x_min_t": x_min_t,
        "x_max_t": x_max_t,
        "u_min_t": u_min_t,
        "u_max_t": u_max_t,
        "x0": x0,
        "x_ref_all": x_ref_all,
        "gamma_x": np.array([gamma_x]),
        "gamma_u": np.array([gamma_u]),
        "gamma_x_per_dim": gamma_x_per_dim,
        "gamma_u_per_dim": gamma_u_per_dim,
        "bottleneck_x_dim": np.array([bottleneck_x_idx], dtype=int),
        "bottleneck_u_dim": np.array([bottleneck_u_idx], dtype=int),
        "dynamics": dynamics,
        "disturbance_mode": disturbance_mode,
        "force_bound_mg": np.array([float(force_bound_mg)]),
        "force_d_axis_scale": np.array([float(force_d_axis_scale)]),
        "dt": np.array([float(sim.dt)]),
        "tracking_profile": np.array([tracking_profile]),
        "gp_model": gp_model,
        "gp_beta_sigma": np.array([float(gp_beta_sigma)]),
        "gp_shrink_mode": gp_shrink_mode,
    }


def sample_initial_state(base_x0: np.ndarray, rng: np.random.Generator, pos_jitter: float, vel_jitter: float) -> np.ndarray:
    x0 = base_x0.copy()
    # 位置抖动
    if x0.size >= 2 and pos_jitter > 0:
        x0[:2] += rng.uniform(-pos_jitter, pos_jitter, size=2)
    if x0.size >= 5 and pos_jitter > 0:
        x0[4] += rng.uniform(-pos_jitter, pos_jitter)

    # 速度抖动
    if x0.size >= 4 and vel_jitter > 0:
        x0[2:4] += rng.uniform(-vel_jitter, vel_jitter, size=2)
    if x0.size >= 6 and vel_jitter > 0:
        x0[5] += rng.uniform(-vel_jitter, vel_jitter)

    return x0


def run_monte_carlo(
    ctx: Dict[str, np.ndarray],
    episodes: int,
    sim_steps: int,
    horizon: int,
    disturbance_scale: float,
    init_pos_jitter: float,
    init_vel_jitter: float,
    seed: int,
    violation_ratio_tol: float,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    A = ctx["A"]
    B = ctx["B"]
    K = ctx["K"]
    n = int(ctx["n"][0])
    z_half = ctx["z_half"]
    w_half = ctx["w_half"] * float(disturbance_scale)
    sim = ctx["sim"]
    disturbance_mode = str(ctx.get("disturbance_mode", "state_box"))
    dt = float(ctx.get("dt", np.array([0.1]))[0])
    force_bound_mg = float(ctx.get("force_bound_mg", np.array([0.05]))[0])
    force_d_axis_scale = float(ctx.get("force_d_axis_scale", np.array([0.15]))[0])
    gp_model = ctx.get("gp_model", None)
    gp_beta_sigma = float(ctx.get("gp_beta_sigma", np.array([2.0]))[0])

    total_steps = episodes * sim_steps
    e_now = np.zeros((total_steps, n), dtype=float)
    ratio_now = np.zeros((total_steps, n), dtype=float)
    violations_now = np.zeros((total_steps,), dtype=np.int32)

    eps = 1e-12
    ptr = 0

    for ep in range(episodes):
        x = sample_initial_state(ctx["x0"], rng, init_pos_jitter, init_vel_jitter)

        for t in range(sim_steps):
            x_des = ctx["x_ref_all"][t : t + horizon + 1]
            d_affine = None
            d_mean = None
            if gp_model is not None:
                d_mean, _ = gp_model.predict_state_disturbance(
                    x=x,
                    beta_sigma=gp_beta_sigma,
                )
                d_affine = np.tile(d_mean.reshape(1, -1), (horizon, 1))

            try:
                Xbar, Ubar = solve_rtmc_qp_paper(
                    A=A,
                    B=B,
                    Qx=ctx["Qx"],
                    Ru=ctx["Ru"],
                    Px=ctx["Px"],
                    x_meas=x,
                    x_des=x_des,
                    N=horizon,
                    z_half=ctx["z_half"],
                    x_bounds=(ctx["x_min_t"], ctx["x_max_t"]),
                    u_bounds=(ctx["u_min_t"], ctx["u_max_t"]),
                    d_affine=d_affine,
                )
            except Exception as exc:
                bx = int(ctx["bottleneck_x_dim"][0])
                bu = int(ctx["bottleneck_u_dim"][0])
                x_min_t = np.asarray(ctx["x_min_t"], dtype=float).reshape(-1)
                x_max_t = np.asarray(ctx["x_max_t"], dtype=float).reshape(-1)
                u_min_t = np.asarray(ctx["u_min_t"], dtype=float).reshape(-1)
                u_max_t = np.asarray(ctx["u_max_t"], dtype=float).reshape(-1)
                gamma_x_per_dim = np.asarray(ctx["gamma_x_per_dim"], dtype=float).reshape(-1)
                gamma_u_per_dim = np.asarray(ctx["gamma_u_per_dim"], dtype=float).reshape(-1)

                x_ref0 = np.asarray(x_des[0], dtype=float).reshape(-1)
                x_margin_low = float(x[bx] - x_min_t[bx])
                x_margin_high = float(x_max_t[bx] - x[bx])
                x_ref_margin_low = float(x_ref0[bx] - x_min_t[bx])
                x_ref_margin_high = float(x_max_t[bx] - x_ref0[bx])
                u_width_b = float(u_max_t[bu] - u_min_t[bu])
                d_mean_now = np.zeros((n,), dtype=float) if d_mean is None else np.asarray(d_mean, dtype=float).reshape(-1)

                raise RuntimeError(
                    "QP infeasible during Monte Carlo rollout.\n"
                    f"  episode={ep}, step={t}, ptr={ptr}\n"
                    f"  solver_error={exc}\n"
                    f"  bottleneck_x_dim={bx}, bottleneck_x_gamma={gamma_x_per_dim[bx]:.6f}\n"
                    f"  bottleneck_u_dim={bu}, bottleneck_u_gamma={gamma_u_per_dim[bu]:.6f}\n"
                    f"  x[bottleneck]={x[bx]:.6f}, x_ref0[bottleneck]={x_ref0[bx]:.6f}, "
                    f"x_min_t={x_min_t[bx]:.6f}, x_max_t={x_max_t[bx]:.6f}\n"
                    f"  x_margin_low={x_margin_low:.6f}, x_margin_high={x_margin_high:.6f}, "
                    f"x_ref_margin_low={x_ref_margin_low:.6f}, x_ref_margin_high={x_ref_margin_high:.6f}\n"
                    f"  u_tight_width[bottleneck_u]={u_width_b:.6f}, "
                    f"u_min_t={u_min_t[bu]:.6f}, u_max_t={u_max_t[bu]:.6f}\n"
                    f"  x_meas={_arr_str(x)}\n"
                    f"  x_ref0={_arr_str(x_ref0)}\n"
                    f"  d_mean_now={_arr_str(d_mean_now)}"
                ) from exc

            xbar0 = Xbar[0]
            ubar0 = Ubar[0]

            u = ubar0 + K @ (x - xbar0)
            u = np.clip(u, ctx["u_min_base"], ctx["u_max_base"])

            force_bound_mg_eff = float(force_bound_mg) * float(disturbance_scale)
            d = sample_process_disturbance(
                rng=rng,
                dynamics=str(ctx.get("dynamics", "iris_linear")),
                dt=dt,
                mode=disturbance_mode,
                force_bound_mg=force_bound_mg_eff,
                force_d_axis_scale=force_d_axis_scale,
                state_dim=n,
                w_half=w_half,
            )
            x_next = sim.step(x, u) + d

            e0 = x - xbar0

            e_now[ptr] = e0

            denom = np.maximum(z_half, eps)
            r0 = np.abs(e0) / denom
            ratio_now[ptr] = r0
            violations_now[ptr] = int(np.any(r0 > 1.0 + float(violation_ratio_tol)))

            x = x_next
            ptr += 1

    return {
        "e_now": e_now,
        "ratio_now": ratio_now,
        "violations_now": violations_now,
        "used_w_half": w_half,
    }


def summarize(
    result: Dict[str, np.ndarray],
    z_half: np.ndarray,
    quantile_level: float,
    violation_ratio_tol: float,
) -> Dict[str, object]:
    ratio_now = result["ratio_now"]
    e_now = np.abs(result["e_now"])

    q = float(quantile_level)
    q_ratio_now = np.quantile(ratio_now, q, axis=0)
    q_e_now = np.quantile(e_now, q, axis=0)

    summary = {
        "violation_rate_now": float(np.mean(result["violations_now"])),
        "max_ratio_now": float(np.max(ratio_now)),
        "q_ratio_now_per_dim": q_ratio_now.tolist(),
        "q_abs_e_now_per_dim": q_e_now.tolist(),
        "z_half_per_dim": z_half.tolist(),
        "q_z_scale_suggestion_per_dim": (q_e_now / np.maximum(z_half, 1e-12)).tolist(),
        # 主判据：在线每步（相对当前步中心 xbar_{0|t}）是否在 tube 内
        "is_reasonable_at_q": bool(np.max(q_ratio_now) <= 1.0 + float(violation_ratio_tol)),
        "violation_ratio_tol": float(violation_ratio_tol),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo 检查当前 tube 半径 z_half 是否合理")
    parser.add_argument("--task", choices=["point", "tracking"], default="tracking")
    parser.add_argument("--dynamics", choices=["double_integrator", "iris_linear"], default="iris_linear")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--sim-steps", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument(
        "--tracking-profile",
        choices=["paper_baseline", "high_speed_extension"],
        default="paper_baseline",
        help="tracking 参考模式：paper_baseline=phi/theta参考为0；high_speed_extension=由速度差分反解姿态参考。",
    )
    parser.add_argument("--disturbance-scale", type=float, default=1.0, help="对当前 w_half 的缩放系数")
    parser.add_argument(
        "--disturbance-mode",
        choices=["state_box", "force_only"],
        default="force_only",
        help="扰动构造方案：state_box=仅用当前8维状态扰动盒；force_only=仅用外力上限映射。",
    )
    parser.add_argument(
        "--force_bound_mg",
        type=float,
        default=0.5,
        help="当 disturbance-mode=force_only 时生效：外力上限系数 c，使 ||f_ext||<=c*m*g。",
    )
    parser.add_argument(
        "--force_d_axis_scale",
        type=float,
        default=0.15,
        help="force_only 模式下 d 轴扰动限幅比例（相对 c*g），建议 0~1。",
    )
    parser.add_argument(
        "--gp_model",
        type=str,
        default="gp_model/iris_linear_residual_gp.npz",
        help="Optional GP residual model (.npz)",
    )
    parser.add_argument(
        "--gp_beta_sigma",
        type=float,
        default=1.0,
        help="GP uncertainty envelope multiplier",
    )
    parser.add_argument(
        "--gp_shrink_mode",
        choices=["none", "residual"],
        default="residual",
        help="GP收缩模式：none=不收缩；residual=base-gp_comp+gp_unc 的残差边界。",
    )
    parser.add_argument("--init-pos-jitter", type=float, default=0.05)
    parser.add_argument("--init-vel-jitter", type=float, default=0.05)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument(
        "--violation_ratio_tol",
        type=float,
        default=1e-6,
        help="越界判定相对容差：r0 > 1 + tol 才记为 violation。",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="montcarlo_runs")
    args = parser.parse_args()

    if args.episodes <= 0 or args.sim_steps <= 0 or args.horizon <= 0:
        raise ValueError("episodes/sim-steps/horizon 必须为正")
    if args.disturbance_scale <= 0.0:
        raise ValueError("disturbance-scale 必须 > 0")
    if args.force_bound_mg < 0.0:
        raise ValueError("force_bound_mg 必须 >= 0")
    if not (0.0 <= args.force_d_axis_scale <= 1.0):
        raise ValueError("force_d_axis_scale 必须在 [0,1] 内")
    if not (0.0 < args.quantile < 1.0):
        raise ValueError("quantile 必须在 (0,1) 内")
    if args.violation_ratio_tol < 0.0:
        raise ValueError("violation_ratio_tol 必须 >= 0")

    ctx = build_context(
        args.dynamics,
        args.task,
        args.sim_steps,
        args.horizon,
        tracking_profile=args.tracking_profile,
        disturbance_mode=args.disturbance_mode,
        force_bound_mg=args.force_bound_mg,
        force_d_axis_scale=args.force_d_axis_scale,
        gp_model_path=args.gp_model,
        gp_beta_sigma=float(args.gp_beta_sigma),
        gp_shrink_mode=args.gp_shrink_mode,
    )

    result = run_monte_carlo(
        ctx=ctx,
        episodes=args.episodes,
        sim_steps=args.sim_steps,
        horizon=args.horizon,
        disturbance_scale=args.disturbance_scale,
        init_pos_jitter=args.init_pos_jitter,
        init_vel_jitter=args.init_vel_jitter,
        seed=args.seed,
        violation_ratio_tol=float(args.violation_ratio_tol),
    )

    summary = summarize(
        result,
        z_half=ctx["z_half"],
        quantile_level=args.quantile,
        violation_ratio_tol=float(args.violation_ratio_tol),
    )
    summary.update(
        {
            "task": args.task,
            "dynamics": args.dynamics,
            "episodes": int(args.episodes),
            "sim_steps": int(args.sim_steps),
            "horizon": int(args.horizon),
            "tracking_profile": args.tracking_profile,
            "disturbance_scale": float(args.disturbance_scale),
            "disturbance_mode": args.disturbance_mode,
            "force_bound_mg": float(args.force_bound_mg),
            "force_d_axis_scale": float(args.force_d_axis_scale),
            "gp_model": args.gp_model,
            "gp_beta_sigma": float(args.gp_beta_sigma),
            "gp_shrink_mode": args.gp_shrink_mode,
            "quantile": float(args.quantile),
            "violation_ratio_tol": float(args.violation_ratio_tol),
            "seed": int(args.seed),
            "gamma_x": float(ctx["gamma_x"][0]),
            "gamma_u": float(ctx["gamma_u"][0]),
            "gamma_x_per_dim": ctx["gamma_x_per_dim"].tolist(),
            "gamma_u_per_dim": ctx["gamma_u_per_dim"].tolist(),
            "bottleneck_x_dim": int(ctx["bottleneck_x_dim"][0]),
            "bottleneck_u_dim": int(ctx["bottleneck_u_dim"][0]),
            "w_half_base": ctx["w_half_base"].tolist(),
            "w_half_force": ctx["w_half_force"].tolist(),
            "gp_comp_half": ctx["gp_comp_half"].tolist(),
            "gp_unc_half": ctx["gp_unc_half"].tolist(),
            "used_w_half": result["used_w_half"].tolist(),
            "u_half_per_dim": ctx["u_half"].tolist(),
            "u_min_base": ctx["u_min_base"].tolist(),
            "u_max_base": ctx["u_max_base"].tolist(),
            "u_min_t": ctx["u_min_t"].tolist(),
            "u_max_t": ctx["u_max_t"].tolist(),
            "u_tight_width_per_dim": (ctx["u_max_t"] - ctx["u_min_t"]).tolist(),
        }
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"mc_tube_{args.dynamics}_{args.task}"
    json_path = out_dir / f"{tag}_summary.json"
    npz_path = out_dir / f"{tag}_samples.npz"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    np.savez(
        npz_path,
        e_now=result["e_now"],
        ratio_now=result["ratio_now"],
        violations_now=result["violations_now"],
        z_half=ctx["z_half"],
        used_w_half=result["used_w_half"],
    )

    print("==== Monte Carlo Tube Radius Check ====")
    print(f"summary: {json_path}")
    print(f"samples: {npz_path}")
    print(f"violation_rate_now:  {summary['violation_rate_now']:.6f}")
    print(f"max_ratio_now:  {summary['max_ratio_now']:.6f}")
    print(f"is_reasonable_at_q={args.quantile:.3f}: {summary['is_reasonable_at_q']}")


if __name__ == "__main__":
    main()
