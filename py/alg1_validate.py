import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _maybe_reexec_into_workspace_venv() -> None:
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
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from alg1_dagger import build_mlp, load_policy_checkpoint, make_policy_input, make_reference, policy_forward
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

# 读取指定checkpoint，或者找到最新的策略文件
def resolve_checkpoint(checkpoint: Optional[str], run_dir: str) -> str:
    if checkpoint:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return str(path)

    run_dir_path = Path(run_dir)
    candidates = sorted(run_dir_path.glob("policy_cycle_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found under: {run_dir_path}")

    def cycle_num(path: Path) -> int:
        m = re.search(r"policy_cycle_(\d+)\.pt$", path.name)
        return int(m.group(1)) if m else -1

    best = max(candidates, key=cycle_num)
    return str(best)

# 将验证环境的系统矩阵、LQR增益、约束等信息构建到一个字典中，供后续rollout使用
def build_validation_context(
    task: str,
    sim_steps: int,
    horizon: int,
    tracking_profile: str,
    circle_radius: float,
    circle_period_steps: int,
    dynamics: str,
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

    w_half_target = disturbance_half_bounds(
        dynamics,
        dt=float(sim.dt),
        mode=disturbance_mode,
        force_bound_mg=float(force_bound_mg),
        force_d_axis_scale=float(force_d_axis_scale),
    )
    gp_model = None
    gp_unc_half = np.zeros_like(w_half_target)
    gp_comp_half = np.zeros_like(w_half_target)

    x_min_base, x_max_base = base_state_bounds(dynamics)
    if dynamics == "double_integrator":
        u_min_base, u_max_base = base_input_bounds(dynamics, m=m)
    else:
        u_min_base, u_max_base = base_input_bounds(dynamics, mass=float(sim.mass))

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

    w_half_target = residual_shrink_bounds(
        base_w_half=w_half_target,
        gp_comp_half=gp_comp_half,
        gp_unc_half=gp_unc_half,
        mode=gp_shrink_mode,
    )
    print(f"[tube] residual disturbance bound after GP shrink (mode={gp_shrink_mode}): {w_half_target}")

    A_cl = A + B @ K
    z_half = compute_rpi_box(A_cl, w_half_target)
    u_half = np.abs(K) @ z_half
    x0_base = base_initial_state(dynamics)

    x_min_t, x_max_t, gamma_x = tighten_box_bounds_with_auto_scale(
        x_min_base, x_max_base, z_half, name="state"
    )
    u_min_t, u_max_t, gamma_u = tighten_box_bounds_with_auto_scale(
        u_min_base, u_max_base, u_half, name="input"
    )

    x_ref_all = make_reference(
        task,
        x0=x0_base,
        N=horizon,
        sim_dt=sim.dt,
        total_steps=sim_steps,
        tracking_profile=tracking_profile,
        circle_radius=float(circle_radius),
        circle_period_steps=int(circle_period_steps),
    )
    if task == "tracking":
        x0_base = x_ref_all[0].copy()

    return {
        "A": A,
        "B": B,
        "K": K,
        "Px": Px,
        "Qx": Qx,
        "Ru": Ru,
        "sim": sim,
        "z_half": z_half,
        "w_half_target": w_half_target,
        "x_min_base": x_min_base,
        "x_max_base": x_max_base,
        "u_min_base": u_min_base,
        "u_max_base": u_max_base,
        "x_min_t": x_min_t,
        "x_max_t": x_max_t,
        "u_min_t": u_min_t,
        "u_max_t": u_max_t,
        "x0_base": x0_base,
        "x_ref_all": x_ref_all,
        "gamma_x": np.array([gamma_x], dtype=float),
        "gamma_u": np.array([gamma_u], dtype=float),
        "n": np.array([n]),
        "m": np.array([m]),
        "tracking_profile": np.array([tracking_profile]),
        "gp_model": gp_model,
        "gp_beta_sigma": np.array([float(gp_beta_sigma)], dtype=float),
        "gp_unc_half": gp_unc_half,
        "gp_comp_half": gp_comp_half,
        "disturbance_mode": disturbance_mode,
        "force_bound_mg": np.array([float(force_bound_mg)], dtype=float),
        "force_d_axis_scale": np.array([float(force_d_axis_scale)], dtype=float),
        "dt": np.array([float(sim.dt)], dtype=float),
    }

# 在初始状态上添加一些随机扰动
def sample_initial_state(base_x0: np.ndarray, rng: np.random.Generator, pos_jitter: float, vel_jitter: float) -> np.ndarray:
    x0 = np.asarray(base_x0, dtype=float).copy()
    if x0.size >= 2 and pos_jitter > 0.0:
        x0[:2] += rng.uniform(-pos_jitter, pos_jitter, size=2)
    if x0.size >= 4 and vel_jitter > 0.0:
        x0[2:4] += rng.uniform(-vel_jitter, vel_jitter, size=2)
    return x0

# 学生策略推理验证
def rollout_policy(
    model: torch.nn.Module,
    task: str,
    sim_steps: int,
    horizon: int,
    device: str,
    x0: np.ndarray,
    disturbances: np.ndarray,
    ctx: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    import time
    sim = ctx["sim"]
    x = np.asarray(x0, dtype=float).copy()
    xs = [x.copy()]
    us = []
    refs = []
    state_violations = 0

    t_start = time.perf_counter()
    for t in range(sim_steps):
        x_des = ctx["x_ref_all"][t : t + horizon + 1]
        inp = make_policy_input(task, x=x, x_des_window=x_des, t=t)
        u = policy_forward(model, inp, device=device)
        u = np.clip(u, ctx["u_min_base"], ctx["u_max_base"])

        refs.append(x_des[0].copy())
        us.append(u.copy())

        if np.any(x < ctx["x_min_base"]) or np.any(x > ctx["x_max_base"]):
            state_violations += 1

        x = sim.step(x, u) + disturbances[t]
        xs.append(x.copy())
    t_end = time.perf_counter()
    total_infer_time = t_end - t_start
    avg_infer_time = total_infer_time / sim_steps if sim_steps > 0 else 0.0

    xs_arr = np.asarray(xs)
    us_arr = np.asarray(us)
    refs_arr = np.asarray(refs)
    return {
        "xs": xs_arr,
        "us": us_arr,
        "refs": refs_arr,
        "state_violations": np.asarray([state_violations], dtype=int),
        "total_infer_time": total_infer_time,
        "avg_infer_time": avg_infer_time,
    }

# RTMPC专家策略验证
def rollout_expert(
    task: str,
    sim_steps: int,
    horizon: int,
    x0: np.ndarray,
    disturbances: np.ndarray,
    ctx: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    sim = ctx["sim"]
    x = np.asarray(x0, dtype=float).copy()
    xs = [x.copy()]
    us = []
    refs = []
    state_violations = 0

    for t in range(sim_steps):
        x_des = ctx["x_ref_all"][t : t + horizon + 1]
        d_affine = None
        gp_model = ctx.get("gp_model", None)
        if gp_model is not None:
            d_mean, _ = gp_model.predict_state_disturbance(
                x=x,
                beta_sigma=float(ctx["gp_beta_sigma"][0]),
            )
            d_affine = np.tile(d_mean.reshape(1, -1), (horizon, 1))
        Xbar, Ubar = solve_rtmc_qp_paper(
            A=ctx["A"],
            B=ctx["B"],
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
        xbar_star = Xbar[0]
        ubar_star = Ubar[0]
        u = ubar_star + ctx["K"] @ (x - xbar_star)
        u = np.clip(u, ctx["u_min_base"], ctx["u_max_base"])

        refs.append(x_des[0].copy())
        us.append(u.copy())

        if np.any(x < ctx["x_min_base"]) or np.any(x > ctx["x_max_base"]):
            state_violations += 1

        x = sim.step(x, u) + disturbances[t]
        xs.append(x.copy())

    xs_arr = np.asarray(xs)
    us_arr = np.asarray(us)
    refs_arr = np.asarray(refs)
    return {
        "xs": xs_arr,
        "us": us_arr,
        "refs": refs_arr,
        "state_violations": np.asarray([state_violations], dtype=int),
    }

# 计算性能指标，包括位置和速度的RMSE、最大误差，控制输入的RMS，以及状态约束违规次数和成功标志
def compute_metrics(rollout: Dict[str, np.ndarray]) -> Dict[str, float]:
    xs = rollout["xs"][:-1]
    us = rollout["us"]
    refs = rollout["refs"]

    pos_err = xs[:, :2] - refs[:, :2]
    vel_err = xs[:, 2:4] - refs[:, 2:4]
    pos_norm = np.linalg.norm(pos_err, axis=1)
    vel_norm = np.linalg.norm(vel_err, axis=1)
    u_norm = np.linalg.norm(us, axis=1)

    return {
        "pos_rmse": float(np.sqrt(np.mean(np.sum(pos_err ** 2, axis=1)))),
        "pos_max": float(np.max(pos_norm)),
        "vel_rmse": float(np.sqrt(np.mean(np.sum(vel_err ** 2, axis=1)))),
        "vel_max": float(np.max(vel_norm)),
        "u_rms": float(np.sqrt(np.mean(np.sum(us ** 2, axis=1)))),
        "state_violations": int(rollout["state_violations"][0]),
        "success": int(rollout["state_violations"][0] == 0),
    }

# 统计多个验证集性能指标
def summarize_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not metrics_list:
        return out
    keys = metrics_list[0].keys()
    for key in keys:
        vals = np.asarray([m[key] for m in metrics_list], dtype=float)
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
        out[f"{key}_min"] = float(np.min(vals))
        out[f"{key}_max"] = float(np.max(vals))
    return out


def maybe_plot_domain(
    out_dir: Path,
    domain: str,
    policy_rollout: Dict[str, np.ndarray],
    expert_rollout: Optional[Dict[str, np.ndarray]],
    interactive_trajectory: bool = False,
) -> Optional[object]:
    if interactive_trajectory:
        try:
            import matplotlib
            backend = str(matplotlib.get_backend()).lower()
            # 常见场景：当前是 non-GUI backend（如 agg），导致 plt.show() 不弹窗。
            if "agg" in backend:
                switched = False
                for cand in ("TkAgg", "Qt5Agg"):
                    try:
                        matplotlib.use(cand, force=True)
                        switched = True
                        break
                    except Exception:
                        continue
                if not switched:
                    print(
                        "[warn] interactive trajectory requested, but no GUI backend is available "
                        "(TkAgg/Qt5Agg unavailable). Running in headless mode and only saving PNG."
                    )
        except Exception as e:
            print(f"[warn] failed to prepare interactive backend: {e}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[warn] matplotlib not available, skipped validation plots")
        return None

    refs = policy_rollout["refs"]
    xs_policy = policy_rollout["xs"]
    interactive_fig = None

    # 轨迹图：double_integrator 用 2D；iris(8维) 默认输出 3D 位置轨迹。
    if refs.shape[1] >= 5:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        # NED -> 绘图用高度 Up=-pd
        ref_x, ref_y, ref_z = refs[:, 0], refs[:, 1], -refs[:, 4]
        pol_x, pol_y, pol_z = xs_policy[:, 0], xs_policy[:, 1], -xs_policy[:, 4]
        ax.plot(ref_x, ref_y, ref_z, "k--", linewidth=1.2, label="reference")
        ax.plot(pol_x, pol_y, pol_z, "r-", linewidth=2.0, label="policy")
        if expert_rollout is not None:
            xs_expert = expert_rollout["xs"]
            ax.plot(xs_expert[:, 0], xs_expert[:, 1], -xs_expert[:, 4], "b-", linewidth=1.6, label="expert")

        # 统一三轴量纲：让 n/e/up 的坐标范围在同一数量级，避免 z 轴过窄导致“高度偏差被视觉放大”。
        all_x = [ref_x, pol_x]
        all_y = [ref_y, pol_y]
        all_z = [ref_z, pol_z]
        if expert_rollout is not None:
            all_x.append(xs_expert[:, 0])
            all_y.append(xs_expert[:, 1])
            all_z.append(-xs_expert[:, 4])

        x_cat = np.concatenate(all_x)
        y_cat = np.concatenate(all_y)
        z_cat = np.concatenate(all_z)

        x_mid = 0.5 * (np.min(x_cat) + np.max(x_cat))
        y_mid = 0.5 * (np.min(y_cat) + np.max(y_cat))
        z_mid = 0.5 * (np.min(z_cat) + np.max(z_cat))
        xyz_span = max(
            float(np.max(x_cat) - np.min(x_cat)),
            float(np.max(y_cat) - np.min(y_cat)),
            float(np.max(z_cat) - np.min(z_cat)),
            1e-6,
        )
        half = 0.5 * xyz_span
        ax.set_xlim(x_mid - half, x_mid + half)
        ax.set_ylim(y_mid - half, y_mid + half)
        ax.set_zlim(z_mid - half, z_mid + half)
        ax.set_box_aspect((1.0, 1.0, 1.0))

        ax.set_xlabel("n")
        ax.set_ylabel("e")
        ax.set_zlabel("up")
        ax.set_title(f"Validation trajectory 3D ({domain})")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"trajectory_{domain}.png", dpi=180)
        if interactive_trajectory:
            interactive_fig = fig
        else:
            plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(refs[:, 0], refs[:, 1], "k--", linewidth=1.2, label="reference")
        ax.plot(xs_policy[:, 0], xs_policy[:, 1], "r-", linewidth=2.0, label="policy")
        if expert_rollout is not None:
            xs_expert = expert_rollout["xs"]
            ax.plot(xs_expert[:, 0], xs_expert[:, 1], "b-", linewidth=1.6, label="expert")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Validation trajectory ({domain})")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / f"trajectory_{domain}.png", dpi=180)
        if interactive_trajectory:
            interactive_fig = fig
        else:
            plt.close(fig)

    # 位置误差：iris 使用 3D 位置误差（n,e,pd），double_integrator 使用 2D。
    pos_idx = [0, 1, 4] if refs.shape[1] >= 5 else [0, 1]
    pos_err_policy = np.linalg.norm(xs_policy[:-1, pos_idx] - refs[:, pos_idx], axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pos_err_policy, label="policy pos error", linewidth=2.0)
    if expert_rollout is not None:
        xs_expert = expert_rollout["xs"]
        pos_err_expert = np.linalg.norm(xs_expert[:-1, pos_idx] - refs[:, pos_idx], axis=1)
        ax.plot(pos_err_expert, label="expert pos error", linewidth=1.8)
    ax.set_xlabel("t")
    ax.set_ylabel("position error norm")
    ax.set_title(f"Tracking error ({domain})")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"error_{domain}.png", dpi=180)
    plt.close(fig)

    if refs.shape[1] >= 6:
        vel_idx = [2, 3, 5]
        vel_names = ["v_n", "v_e", "v_d"]
    else:
        vel_idx = [2, 3]
        vel_names = ["v_x", "v_y"]

    vel_policy = xs_policy[:-1, vel_idx]
    vel_ref = refs[:, vel_idx]
    fig, axes = plt.subplots(len(vel_idx), 1, figsize=(8, 2.2 * len(vel_idx) + 1), sharex=True)
    if len(vel_idx) == 1:
        axes = [axes]
    for i, (vidx, vname) in enumerate(zip(vel_idx, vel_names)):
        axes[i].plot(vel_ref[:, i], "k--", linewidth=1.2, label=f"reference ${vname}$")
        axes[i].plot(vel_policy[:, i], "r-", linewidth=2.0, label=f"policy ${vname}$")
    if expert_rollout is not None:
        vel_expert = expert_rollout["xs"][:-1, vel_idx]
        for i, vname in enumerate(vel_names):
            axes[i].plot(vel_expert[:, i], "b-", linewidth=1.6, label=f"expert ${vname}$")
    for i, vname in enumerate(vel_names):
        axes[i].set_ylabel(f"${vname}$")
    axes[-1].set_xlabel("t")
    axes[0].set_title(f"Velocity components ({domain})")
    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"velocity_{domain}.png", dpi=180)
    plt.close(fig)

    speed_policy = np.linalg.norm(vel_policy, axis=1)
    speed_ref = np.linalg.norm(vel_ref, axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(speed_ref, "k--", linewidth=1.2, label="reference $\\|v\\|_2$")
    ax.plot(speed_policy, "r-", linewidth=2.0, label="policy $\\|v\\|_2$")
    if expert_rollout is not None:
        vel_expert = expert_rollout["xs"][:-1, 2:4]
        speed_expert = np.linalg.norm(vel_expert, axis=1)
        ax.plot(speed_expert, "b-", linewidth=1.6, label="expert $\\|v\\|_2$")
    ax.set_xlabel("t")
    ax.set_ylabel("$\\|v\\|_2$")
    ax.set_title(f"Speed norm ({domain})")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"speed_norm_{domain}.png", dpi=180)
    plt.close(fig)

    us_policy = policy_rollout["us"]
    u_dim = us_policy.shape[1]
    if u_dim == 3:
        u_names = ["dT", "phi_cmd", "theta_cmd"]
    else:
        u_names = [f"u_{i}" for i in range(u_dim)]

    fig, axes = plt.subplots(u_dim, 1, figsize=(8, 2.2 * u_dim + 1), sharex=True)
    if u_dim == 1:
        axes = [axes]
    for i, uname in enumerate(u_names):
        axes[i].plot(us_policy[:, i], "r-", linewidth=2.0, label=f"policy ${uname}$")
    if expert_rollout is not None:
        us_expert = expert_rollout["us"]
        for i, uname in enumerate(u_names):
            axes[i].plot(us_expert[:, i], "b-", linewidth=1.6, label=f"expert ${uname}$")
    for i, uname in enumerate(u_names):
        axes[i].set_ylabel(f"${uname}$")
    axes[-1].set_xlabel("t")
    axes[0].set_title(f"Control inputs ({domain})")
    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"control_{domain}.png", dpi=180)
    plt.close(fig)

    u_norm_policy = np.linalg.norm(us_policy, axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(u_norm_policy, "r-", linewidth=2.0, label="policy $\\|u\\|_2$")
    if expert_rollout is not None:
        us_expert = expert_rollout["us"]
        u_norm_expert = np.linalg.norm(us_expert, axis=1)
        ax.plot(u_norm_expert, "b-", linewidth=1.6, label="expert $\\|u\\|_2$")
    ax.set_xlabel("t")
    ax.set_ylabel("$\\|u\\|_2$")
    ax.set_title(f"Control norm ({domain})")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"control_norm_{domain}.png", dpi=180)
    plt.close(fig)

    return interactive_fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a trained DAgger policy in source/target domains")
    parser.add_argument("--checkpoint", type=str, default=None, help="Policy checkpoint to validate. If omitted, the latest checkpoint under --run-dir is used.")
    parser.add_argument("--run-dir", type=str, default="dagger_runs", help="Training output directory containing policy_cycle_*.pt")
    parser.add_argument("--task", choices=["point", "tracking"], default="tracking")
    parser.add_argument(
        "--dynamics",
        choices=["double_integrator", "iris_linear"],
        default="iris_linear",
        help="Dynamics model used by linear RTMPC expert and rollout simulation.",
    )
    parser.add_argument(
        "--disturbance-mode",
        choices=["state_box", "force_only"],
        default="force_only",
        help="扰动构造方案：state_box=仅用当前状态扰动盒；force_only=仅用外力边界映射。",
    )
    parser.add_argument("--force_bound_mg", type=float, default=0.15, help="外力边界系数 c，使 ||f_ext||<=c*m*g")
    parser.add_argument("--force_d_axis_scale", type=float, default=0.15, help="force_only 模式下 d 轴扰动限幅比例（0~1）")
    parser.add_argument("--domain", choices=["source", "target", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=5, help="Number of validation episodes per domain")
    parser.add_argument("--sim-steps", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument(
        "--tracking-profile",
        choices=["paper_baseline", "high_speed_extension"],
        default="high_speed_extension",
        help="tracking 参考模式：paper_baseline=phi/theta参考为0；high_speed_extension=由速度差分反解姿态参考。",
    )
    parser.add_argument("--circle-radius", type=float, default=4.0, help="tracking 圆轨迹半径（m）")
    parser.add_argument("--circle-period-steps", type=int, default=126, help="tracking 圆轨迹一圈步数（dt=0.1s）")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default="validation_runs", help="Directory to save validation summaries and plots")
    parser.add_argument("--skip-expert", action="store_true", help="If set, do not run RTMPC expert rollouts")
    parser.add_argument("--init-pos-jitter", type=float, default=0.1, help="Uniform initial position perturbation magnitude per axis")
    parser.add_argument("--init-vel-jitter", type=float, default=0.05, help="Uniform initial velocity perturbation magnitude per axis")
    parser.add_argument(
        "--init-pos3d",
        type=float,
        nargs=3,
        default=[4.0, 0.0, -0.8],
        metavar=("PN", "PE", "PD"),
        help="仅覆盖 rollout 初始三维位置 [pn, pe, pd]；不会修改参考轨迹。",
    )
    parser.add_argument("--gp-model", type=str, default="gp_model/iris_linear_residual_gp.npz", help="Optional GP residual model (.npz)")
    parser.add_argument("--gp-beta-sigma", type=float, default=1.0, help="GP uncertainty envelope multiplier")
    parser.add_argument(
        "--gp-shrink-mode",
        choices=["none", "residual"],
        default="residual",
        help="GP收缩模式：none=不收缩；residual=base-gp_comp+gp_unc 的残差边界。",
    )
    parser.add_argument(
        "--interactive-trajectory",
        action="store_true",
        help="Show interactive trajectory figure window (3D view can be rotated with mouse).",
    )
    args = parser.parse_args()
    if not (0.0 <= args.force_d_axis_scale <= 1.0):
        raise ValueError("force_d_axis_scale 必须在 [0,1] 内")
    if args.circle_radius <= 0.0:
        raise ValueError("circle_radius 必须 > 0")
    if args.circle_period_steps <= 0:
        raise ValueError("circle_period_steps 必须 > 0")

    # VS Code 的 Run Code 常不带参数；此时默认开启交互轨迹窗口，便于直接旋转视角。
    if len(sys.argv) == 1 and not args.interactive_trajectory:
        args.interactive_trajectory = True
        print("[info] no CLI args detected; enable --interactive-trajectory by default")

    if args.interactive_trajectory:
        print("[info] interactive trajectory is enabled (a GUI window should pop up if backend/display is available)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_checkpoint(args.checkpoint, args.run_dir)
    print(f"using checkpoint: {ckpt_path}")

    ckpt = load_policy_checkpoint(ckpt_path)
    hidden = tuple(int(h) for h in ckpt.get("hidden", (128, 128)))
    ctx = build_validation_context(
        task=args.task,
        sim_steps=args.sim_steps,
        horizon=args.horizon,
        tracking_profile=args.tracking_profile,
        circle_radius=float(args.circle_radius),
        circle_period_steps=int(args.circle_period_steps),
        dynamics=args.dynamics,
        disturbance_mode=args.disturbance_mode,
        force_bound_mg=args.force_bound_mg,
        force_d_axis_scale=args.force_d_axis_scale,
        gp_model_path=args.gp_model,
        gp_beta_sigma=float(args.gp_beta_sigma),
        gp_shrink_mode=args.gp_shrink_mode,
    )
    input_dim = int(ckpt.get("input_dim"))
    output_dim = int(ckpt.get("output_dim", int(ctx["m"][0])))

    n = int(ctx["n"][0])
    m = int(ctx["m"][0])
    expected_input_dim = (n + (args.horizon + 1) * n) if args.task == "tracking" else (2 * n + 1)
    ckpt_task = ckpt.get("task")
    ckpt_dyn = ckpt.get("dynamics")
    ckpt_horizon = ckpt.get("horizon")
    ckpt_tracking_profile = ckpt.get("tracking_profile")
    ckpt_circle_radius = ckpt.get("circle_radius")
    ckpt_circle_period_steps = ckpt.get("circle_period_steps")
    if ckpt_task is not None and str(ckpt_task) != str(args.task):
        raise ValueError(f"checkpoint 任务不匹配：checkpoint task={ckpt_task}, 当前 task={args.task}")
    if ckpt_dyn is not None and str(ckpt_dyn) != str(args.dynamics):
        raise ValueError(f"checkpoint 动力学不匹配：checkpoint dynamics={ckpt_dyn}, 当前 dynamics={args.dynamics}")
    if ckpt_horizon is not None and int(ckpt_horizon) != int(args.horizon):
        raise ValueError(
            f"checkpoint 预测时域不匹配：checkpoint horizon={ckpt_horizon}, 当前 horizon={args.horizon}"
        )
    if ckpt_tracking_profile is not None and str(ckpt_tracking_profile) != str(args.tracking_profile):
        raise ValueError(
            "checkpoint tracking_profile 不匹配："
            f"checkpoint tracking_profile={ckpt_tracking_profile}, 当前 tracking_profile={args.tracking_profile}"
        )
    if ckpt_circle_radius is not None and float(ckpt_circle_radius) != float(args.circle_radius):
        raise ValueError(
            "checkpoint circle_radius 不匹配："
            f"checkpoint circle_radius={ckpt_circle_radius}, 当前 circle_radius={args.circle_radius}"
        )
    if ckpt_circle_period_steps is not None and int(ckpt_circle_period_steps) != int(args.circle_period_steps):
        raise ValueError(
            "checkpoint circle_period_steps 不匹配："
            f"checkpoint circle_period_steps={ckpt_circle_period_steps}, 当前 circle_period_steps={args.circle_period_steps}"
        )

    if input_dim != expected_input_dim:
        raise ValueError(
            "checkpoint 与当前验证配置不一致："
            f"checkpoint input_dim={input_dim}, expected={expected_input_dim}。"
            "请使用匹配的 --checkpoint，或对齐 --task/--dynamics/--horizon。"
        )
    if output_dim != m:
        raise ValueError(
            "checkpoint 与当前动力学动作维度不一致："
            f"checkpoint output_dim={output_dim}, expected={m}。"
            "请使用匹配动力学训练得到的 checkpoint。"
        )

    model = build_mlp(input_dim, output_dim, hidden).to(args.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    base_x0 = np.asarray(ctx["x0_base"], dtype=float).copy()
    if args.init_pos3d is not None:
        if base_x0.size < 5:
            raise ValueError("当前状态维度不支持 3D 位置覆盖（需要至少 5 维，含 pn/pe/pd）")
        base_x0[0] = float(args.init_pos3d[0])
        base_x0[1] = float(args.init_pos3d[1])
        base_x0[4] = float(args.init_pos3d[2])  # NED: Down
        print(
            "[info] rollout initial position override: "
            f"pn={base_x0[0]:.3f}, pe={base_x0[1]:.3f}, pd={base_x0[4]:.3f}"
        )

    domains = [args.domain] if args.domain != "both" else ["source", "target"]
    overall_summary: Dict[str, Dict[str, float]] = {}
    interactive_figs: List[object] = []

    for domain in domains:
        print(f"\n=== validating domain: {domain} ===")
        domain_dir = out_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        episode_rng = np.random.default_rng(args.seed)
        policy_metrics_all: List[Dict[str, float]] = []
        expert_metrics_all: List[Dict[str, float]] = []
        expert_infeasible_count = 0
        expert_feasible_count = 0
        first_policy_rollout: Optional[Dict[str, np.ndarray]] = None
        first_expert_rollout: Optional[Dict[str, np.ndarray]] = None
        saved_rollouts = []

        policy_infer_times = []
        policy_avg_infer_times = []
        for ep in range(args.episodes):
            # 仅修改 rollout 的初始状态，不修改参考轨迹 ctx["x_ref_all"]。
            x0_seed = np.asarray(base_x0, dtype=float).copy()
            x0_ep = sample_initial_state(
                base_x0=x0_seed,
                rng=episode_rng,
                pos_jitter=args.init_pos_jitter,
                vel_jitter=args.init_vel_jitter,
            )

            if domain == "source":
                disturbances = np.zeros((args.sim_steps, int(ctx["n"][0])), dtype=float)
            else:
                disturbances = np.zeros((args.sim_steps, int(ctx["n"][0])), dtype=float)
                for tt in range(args.sim_steps):
                    disturbances[tt] = sample_process_disturbance(
                        rng=episode_rng,
                        dynamics=args.dynamics,
                        dt=float(ctx["dt"][0]),
                        mode=str(ctx["disturbance_mode"]),
                        force_bound_mg=float(ctx["force_bound_mg"][0]),
                        force_d_axis_scale=float(ctx["force_d_axis_scale"][0]),
                        state_dim=int(ctx["n"][0]),
                        w_half=ctx["w_half_target"],
                    )

            policy_rollout = rollout_policy(
                model=model,
                task=args.task,
                sim_steps=args.sim_steps,
                horizon=args.horizon,
                device=args.device,
                x0=x0_ep,
                disturbances=disturbances,
                ctx=ctx,
            )
            policy_metrics = compute_metrics(policy_rollout)
            policy_metrics_all.append(policy_metrics)
            # 输出每个 episode 的推理耗时
            print(f"[timing][ep {ep+1:02d}] policy rollout: total_infer_time = {policy_rollout['total_infer_time']:.6f} s, avg_infer_time = {policy_rollout['avg_infer_time']*1e3:.3f} ms/step")
            policy_infer_times.append(policy_rollout['total_infer_time'])
            policy_avg_infer_times.append(policy_rollout['avg_infer_time'])

            expert_rollout = None
            expert_metrics = None
            expert_error: Optional[str] = None
            if not args.skip_expert:
                try:
                    expert_rollout = rollout_expert(
                        task=args.task,
                        sim_steps=args.sim_steps,
                        horizon=args.horizon,
                        x0=x0_ep,
                        disturbances=disturbances,
                        ctx=ctx,
                    )
                    expert_metrics = compute_metrics(expert_rollout)
                    expert_metrics_all.append(expert_metrics)
                    expert_feasible_count += 1
                except Exception as exc:
                    expert_infeasible_count += 1
                    expert_error = str(exc)
                    print(f"[warn] episode {ep+1:02d} expert rollout infeasible: {expert_error}")

            if first_policy_rollout is None:
                first_policy_rollout = policy_rollout
                first_expert_rollout = expert_rollout

            saved_rollouts.append(
                {
                    "episode": ep,
                    "x0": x0_ep.tolist(),
                    "policy_metrics": policy_metrics,
                    "expert_metrics": expert_metrics,
                    "expert_infeasible": bool(expert_error is not None),
                    "expert_error": expert_error,
                }
            )

            msg = (
                f"episode {ep+1:02d} | policy pos_rmse={policy_metrics['pos_rmse']:.4f} "
                f"success={policy_metrics['success']}"
            )
            if expert_metrics is not None:
                msg += f" | expert pos_rmse={expert_metrics['pos_rmse']:.4f} success={expert_metrics['success']}"
            elif expert_error is not None:
                msg += " | expert infeasible"
            print(msg)

        # 输出所有 episode 的平均推理时间
        if policy_infer_times:
            mean_total = float(np.mean(policy_infer_times))
            std_total = float(np.std(policy_infer_times))
            mean_avg = float(np.mean(policy_avg_infer_times))
            std_avg = float(np.std(policy_avg_infer_times))
            print(f"[timing][summary] policy rollout: mean_total_infer_time = {mean_total:.6f} ± {std_total:.6f} s, mean_avg_infer_time = {mean_avg*1e3:.3f} ± {std_avg*1e3:.3f} ms/step")

        policy_summary = summarize_metrics(policy_metrics_all)
        expert_summary = summarize_metrics(expert_metrics_all) if expert_metrics_all else {}
        overall_summary[domain] = {
            "checkpoint": ckpt_path,
            "episodes": args.episodes,
            "policy": policy_summary,
            "expert": expert_summary,
            "expert_feasible_episodes": int(expert_feasible_count),
            "expert_infeasible_episodes": int(expert_infeasible_count),
            "expert_infeasible_ratio": (
                float(expert_infeasible_count) / float(args.episodes) if args.episodes > 0 else 0.0
            ),
        }

        print("policy summary:")
        for key, value in policy_summary.items():
            print(f"  {key}: {value:.6f}")
        if expert_summary:
            print("expert summary:")
            for key, value in expert_summary.items():
                print(f"  {key}: {value:.6f}")

        with open(domain_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(overall_summary[domain], f, ensure_ascii=False, indent=2)
        with open(domain_dir / "episodes.json", "w", encoding="utf-8") as f:
            json.dump(saved_rollouts, f, ensure_ascii=False, indent=2)

        if first_policy_rollout is not None:
            np.savez(
                domain_dir / "first_episode.npz",
                xs_policy=first_policy_rollout["xs"],
                us_policy=first_policy_rollout["us"],
                refs=first_policy_rollout["refs"],
                xs_expert=(
                    first_expert_rollout["xs"]
                    if first_expert_rollout is not None
                    else np.empty((0, int(ctx["n"][0])))
                ),
                us_expert=(
                    first_expert_rollout["us"]
                    if first_expert_rollout is not None
                    else np.empty((0, int(ctx["m"][0])))
                ),
            )
            maybe_fig = maybe_plot_domain(
                domain_dir,
                domain,
                first_policy_rollout,
                first_expert_rollout,
                interactive_trajectory=args.interactive_trajectory,
            )
            if maybe_fig is not None:
                interactive_figs.append(maybe_fig)

    with open(out_dir / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    if args.interactive_trajectory and interactive_figs:
        try:
            import matplotlib.pyplot as plt
            print("[info] all trajectory figures prepared; close windows to finish validation")
            plt.show()
            for fig in interactive_figs:
                plt.close(fig)
        except Exception as e:
            print(f"[warn] interactive trajectory view unavailable: {e}")

    print(f"\nvalidation summary saved to: {out_dir}")


if __name__ == "__main__":
    main()
