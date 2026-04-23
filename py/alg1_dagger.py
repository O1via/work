import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List


def _try_plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    out_path: str,
    title: str,
    cycle_boundaries: Optional[List[int]] = None,
) -> bool:
    """尽量保存 loss 曲线图；若 matplotlib 不可用则返回 False。"""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="train", linewidth=2.0)
    if len(val_losses) == len(train_losses):
        ax.plot(epochs, val_losses, label="val", linewidth=2.0)

    if cycle_boundaries:
        for b in cycle_boundaries[:-1]:
            ax.axvline(float(b) + 0.5, color="0.7", linestyle="--", linewidth=1.0)

    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


class _TeeStream:
    """Duplicate writes to multiple text streams (console + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def _maybe_reexec_into_workspace_venv() -> None:
    """尽量让 VS Code 的 Run Code 使用工作区 .venv 解释器。

    Code Runner/Run Code 经常直接调用系统 python 来执行当前文件，
    这样即便工作区 .venv 已安装 torch，也会因为解释器不对而报
    `ModuleNotFoundError: No module named 'torch'`。

    这里在导入第三方库前先检查：
    - 如果当前解释器不是工作区根目录 `.venv/bin/python`
    - 且该解释器存在
    则自动 execv 重启到它。
    """
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from rtmpc_demo import (
    DoubleIntegrator,
    LinearIrisHover,
    build_circle_reference,
    compute_infinite_lqr,
    compute_rpi_box,
    sample_box_points,
    solve_rtmc_qp_paper,
    tighten_box_bounds_with_auto_scale,
)
from gp_residual_model import VelocityResidualGP, merge_bounds
from rtmpc_constants import (
    base_initial_state,
    base_input_bounds,
    base_state_bounds,
    disturbance_half_bounds,
    input_cost_matrix,
    state_cost_matrix,
)

# 构建MLP 学生策略网络，输入单个状态/参考窗口，输出单个动作。
def build_mlp(input_dim: int, output_dim: int, hidden: Tuple[int, ...]) -> nn.Module:
    layers = []
    dims = (input_dim, *hidden, output_dim)
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

# 加载策略网络 checkpoint，统一保存格式为包含 "state_dict" 键的 dict
def load_policy_checkpoint(path: str) -> Dict:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt
    if isinstance(ckpt, dict):
        return {"state_dict": ckpt}
    raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")


# 学生策略向前推理，输入单个状态/参考窗口，输出单个动作。
@torch.no_grad()
def policy_forward(model: nn.Module, policy_in: np.ndarray, device: str) -> np.ndarray:
    xb = torch.as_tensor(policy_in, dtype=torch.float32, device=device).reshape(1, -1)
    ub = model(xb).reshape(-1)
    return ub.detach().cpu().numpy()

# 根据任务把状态和参考拼成策略输入，供模型前向使用和训练数据构造。
def make_policy_input(task: str, x: np.ndarray, x_des_window: np.ndarray, t: int) -> np.ndarray:
    """构造论文符号里的策略输入 x_in。

    - tracking: x_in = {x_t, X^des_t}，这里将 X^des_t (N+1,n) 展平拼到后面。
    - point:    x_in = {x_t, x^des_{0|t}, t}，这里使用 x_des_window[0] 作为目标状态，并附加标量 t。
    """
    x = np.asarray(x).reshape(-1)
    x_des_window = np.asarray(x_des_window)
    if task == "tracking":
        return np.concatenate([x, x_des_window.reshape(-1)])
    if task == "point":
        x_goal = x_des_window[0].reshape(-1)
        return np.concatenate([x, x_goal, np.array([float(t)], dtype=float)])
    raise ValueError("task must be point or tracking")


# 计算当前轮次的 β 值
def beta_value(
    cycle_idx: int,
    cycles: int,
    beta_start: float,
    beta_end: float,
    schedule: str,
) -> float:
    # Paper (Tagliabue & How 2024) experimental setup: "we set the probability of
    # using actions of the expert β to be 1 at the first demonstration and 0 otherwise".
    # With 0-based cycle_idx, that is: β_0 = 1, β_i = 0 for i >= 1.
    if schedule == "paper":
        return 1.0 if cycle_idx == 0 else 0.0
    if cycles <= 1:
        return float(beta_end)
    if schedule == "constant":
        return float(beta_start)
    if schedule == "linear":
        a = cycle_idx / (cycles - 1)
        return float((1.0 - a) * beta_start + a * beta_end)
    if schedule == "exp":
        # exponential interpolation in (0,1] domain; fall back to linear if endpoints invalid
        if beta_start <= 0.0 or beta_end <= 0.0:
            a = cycle_idx / (cycles - 1)
            return float((1.0 - a) * beta_start + a * beta_end)
        a = cycle_idx / (cycles - 1)
        return float(beta_start * ((beta_end / beta_start) ** a))
    raise ValueError("schedule must be one of: paper|constant|linear|exp")


# 训练参数打包成 dataclass
@dataclass
class TrainConfig:
    hidden: Tuple[int, ...]
    batch_size: int
    lr: float
    weight_decay: float
    epochs_per_cycle: int
    val_split: float
    device: str


def train_one_cycle(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    xs_all: torch.Tensor,
    us_all: torch.Tensor,
    cfg: TrainConfig,
) -> Dict[str, List[float]]:
    dataset = TensorDataset(xs_all, us_all)
    val_len = max(1, int(len(dataset) * cfg.val_split)) if len(dataset) > 1 else 0
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len]) if val_len > 0 else (dataset, [])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size) if val_len > 0 else None

    loss_fn = nn.MSELoss()
    history = {"train": [], "val": [], "epoch_time_sec": []}

    def run_epoch(loader, train: bool) -> float:
        total, count = 0.0, 0
        for xb, ub in loader:
            xb, ub = xb.to(cfg.device), ub.to(cfg.device)
            pred = model(xb)
            loss = loss_fn(pred, ub)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * xb.size(0)
            count += xb.size(0)
        return total / max(count, 1)

    for ep in range(1, cfg.epochs_per_cycle + 1):
        t0_ep = time.perf_counter()
        train_loss = run_epoch(train_loader, train=True)
        val_loss = run_epoch(val_loader, train=False) if val_loader else float("nan")
        ep_time_sec = time.perf_counter() - t0_ep
        history["train"].append(float(train_loss))
        history["val"].append(float(val_loss))
        history["epoch_time_sec"].append(float(ep_time_sec))
        print(
            f"  epoch {ep:03d} | train {train_loss:.4e} | val {val_loss:.4e} | "
            f"time {ep_time_sec:.3f}s"
        )

    return history


def make_reference(task: str, x0: np.ndarray, N: int, sim_dt: float, total_steps: int) -> np.ndarray:
    n = x0.shape[0]
    if task == "point":
        return np.zeros((total_steps + N + 1, n))
    if task != "tracking":
        raise ValueError("task must be point or tracking")

    # 为了与 rtmpc_demo 的 tracking 任务保持一致，这里直接复用其圆形参考轨迹生成逻辑。
    total_len = total_steps + N + 1
    if n == 4:
        return build_circle_reference(x0=x0, total_len=total_len, dt=float(sim_dt))

    # 8维线性 iris 模型（PX4/NED）：x=[pn,pe,vn,ve,pd,vd,phi,theta]
    # 参考轨迹在 n-e 平面画圆，d(Down) 保持常值，倾角参考置零。
    if n == 8:
        xy_ref = build_circle_reference(x0=x0[:4], total_len=total_len, dt=float(sim_dt))
        out = np.zeros((total_len, n), dtype=float)
        out[:, 0] = xy_ref[:, 0]  # pn
        out[:, 1] = xy_ref[:, 1]  # pe
        out[:, 2] = xy_ref[:, 2]  # vn
        out[:, 3] = xy_ref[:, 3]  # ve
        out[:, 4] = float(x0[4])  # pd (Down)
        # vd, phi, theta 参考保持 0
        return out

    raise ValueError(f"unsupported state dimension for reference generation: n={n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithm 1-style DAgger/BC loop with tube-guided augmentation")
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
        default="state_box",
        help="扰动构造方案：state_box=仅用当前状态扰动盒；force_only=仅用外力边界映射。",
    )
    parser.add_argument("--force-bound-mg", type=float, default=0.35, help="外力边界系数 c，使 ||f_ext||<=c*m*g")
    parser.add_argument("--cycles", type=int, default=8, help="Number of Algorithm 1 iterations")
    parser.add_argument("--sim-steps", type=int, default=100, help="Receding-horizon steps per cycle")
    parser.add_argument("--horizon", type=int, default=30, help="MPC prediction horizon N")
    parser.add_argument("--augment", choices=["dense", "sparse"], default="dense")

    parser.add_argument("--beta-start", type=float, default=1.0, help="Probability of executing teacher")
    parser.add_argument("--beta-end", type=float, default=0.0)
    parser.add_argument(
        "--beta-schedule",
        choices=["paper", "constant", "linear", "exp"],
        default="paper",
        help="Teacher-mix schedule. 'paper' matches the paper setup: beta=1 for the first cycle, then 0.",
    )

    parser.add_argument("--out-dir", type=str, default="dagger_runs", help="Directory for per-cycle datasets and checkpoints")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--hidden", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per cycle")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--init-checkpoint", type=str, default=None, help="Optional checkpoint to warm-start student")
    parser.add_argument("--gp-model", type=str, default=None, help="Optional GP residual model (.npz)")
    parser.add_argument("--gp-beta-sigma", type=float, default=2.0, help="GP uncertainty envelope multiplier")
    parser.add_argument(
        "--gp-shrink-mode",
        choices=["none", "replace", "min"],
        default="none",
        help="How GP uncertainty bound is merged with disturbance bound for tube sizing.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="train.log",
        help="训练日志文件名（保存在 --out-dir 下）。",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, args.log_file)
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log_fp)
    sys.stderr = _TeeStream(sys.stderr, log_fp)
    print(f"[log] writing training logs to: {log_path}")
    print(f"[log] started at: {datetime.now().isoformat(timespec='seconds')}")
    print(f"[log] command: {' '.join(sys.argv)}")

    if args.dynamics == "double_integrator":
        sim = DoubleIntegrator(dt=0.1)
    else:
        sim = LinearIrisHover(dt=0.1, mass=1.5)

    A, B = sim.A, sim.B
    n, m = A.shape[0], B.shape[1]
    x_min_base, x_max_base = base_state_bounds(args.dynamics)
    if args.dynamics == "double_integrator":
        u_min_base, u_max_base = base_input_bounds(args.dynamics, m=m)
    else:
        u_min_base, u_max_base = base_input_bounds(args.dynamics, mass=float(sim.mass))
    x0 = base_initial_state(args.dynamics)

    Qx = state_cost_matrix(args.dynamics)
    Ru = input_cost_matrix(args.dynamics, m)

    Px, K = compute_infinite_lqr(A, B, Qx, Ru)

    w_half = disturbance_half_bounds(
        args.dynamics,
        dt=float(sim.dt),
        mode=args.disturbance_mode,
        force_bound_mg=float(args.force_bound_mg),
    )
    gp_model = None
    gp_w_half = None
    if args.gp_model:
        gp_path = Path(args.gp_model)
        if not gp_path.exists():
            raise FileNotFoundError(f"GP model not found: {gp_path}")
        gp_model = VelocityResidualGP.load(str(gp_path))
        if gp_model.dynamics != args.dynamics:
            raise ValueError(
                f"GP dynamics mismatch: model={gp_model.dynamics}, current={args.dynamics}"
            )
        if abs(float(gp_model.dt) - float(sim.dt)) > 1e-9:
            raise ValueError(
                f"GP dt mismatch: model={gp_model.dt}, current={sim.dt}"
            )
        if int(gp_model.state_dim) != int(n):
            raise ValueError(
                f"GP state_dim mismatch: model={gp_model.state_dim}, current={n}"
            )
        gp_w_half = gp_model.conservative_uncertainty_bound(
            x_min=x_min_base,
            x_max=x_max_base,
            beta_sigma=float(args.gp_beta_sigma),
        )
        print(f"[gp] loaded model: {gp_path}")
        print(f"[gp] uncertainty bound (beta={args.gp_beta_sigma:.2f}): {gp_w_half}")

    w_half = merge_bounds(
        base_w_half=w_half,
        gp_w_half=gp_w_half if gp_w_half is not None else w_half,
        mode=args.gp_shrink_mode,
    )
    print(f"[tube] disturbance bound after GP merge (mode={args.gp_shrink_mode}): {w_half}")
    A_cl = A + B @ K
    z_half = compute_rpi_box(A_cl, w_half)
    u_half = np.abs(K) @ z_half

    x_min_t, x_max_t, gamma_x = tighten_box_bounds_with_auto_scale(
        x_min_base, x_max_base, z_half, name="state"
    )
    u_min_t, u_max_t, gamma_u = tighten_box_bounds_with_auto_scale(
        u_min_base, u_max_base, u_half, name="input"
    )
    if gamma_x < 1.0 or gamma_u < 1.0:
        print(f"[info] tighten scale factors: gamma_x={gamma_x:.4f}, gamma_u={gamma_u:.4f}")

    N = int(args.horizon)
    sim_steps = int(args.sim_steps)
    cycles = int(args.cycles)

    x_ref_all = make_reference(args.task, x0=x0, N=N, sim_dt=sim.dt, total_steps=sim_steps)

    # 与 rtmpc_demo 的 circle tracking 行为一致：将初始状态对齐到参考的第一个点。
    if args.task == "tracking":
        x0 = x_ref_all[0].copy()

    # 论文：tracking 策略输入包含 (x_t, X^des_t)；point 策略输入包含 (x_t, x^des_{0|t}, t)
    if args.task == "tracking":
        policy_in_dim = n + (N + 1) * n
    else:
        policy_in_dim = 2 * n + 1

    hidden = tuple(int(h) for h in args.hidden)
    model = build_mlp(policy_in_dim, m, hidden).to(args.device)

    if args.init_checkpoint:
        ckpt = load_policy_checkpoint(args.init_checkpoint)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        if "hidden" in ckpt:
            print(f"loaded checkpoint hidden={ckpt['hidden']}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    rng = np.random.default_rng(seed=args.seed)

    xs_hist: List[np.ndarray] = []
    us_hist: List[np.ndarray] = []
    train_losses_all: List[float] = []
    val_losses_all: List[float] = []
    cycle_boundaries: List[int] = []

    train_cfg = TrainConfig(
        hidden=hidden,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs_per_cycle=args.epochs,
        val_split=args.val_split,
        device=args.device,
    )

    for cycle_idx in range(cycles):
        beta = beta_value(
            cycle_idx=cycle_idx,
            cycles=cycles,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            schedule=args.beta_schedule,
        )

        print(f"cycle {cycle_idx+1}/{cycles} | beta={beta:.3f} | augment={args.augment}")

        x = x0.copy()
        aug_xs_cycle = []
        aug_us_cycle = []

        # 训练域 S：默认用名义/无扰动的 step（更贴论文“从受控/名义域收集演示，再用 tube 推断目标域支持”）
        w_half_train = np.zeros_like(w_half)

        for t in range(sim_steps):
            x_des = x_ref_all[t : t + N + 1]
            d_affine = None
            if gp_model is not None:
                d_mean, _ = gp_model.predict_state_disturbance(
                    x=x,
                    beta_sigma=float(args.gp_beta_sigma),
                )
                d_affine = np.tile(d_mean.reshape(1, -1), (N, 1))

            # 论文 Eq.(10)：包含 tube 初值约束 x_t ∈ x̄_{0|t} ⊕ Z（这里用盒 Z 的外包框 z_half）
            Xbar, Ubar = solve_rtmc_qp_paper(
                A=A,
                B=B,
                Qx=Qx,
                Ru=Ru,
                Px=Px,
                x_meas=x,
                x_des=x_des,
                N=N,
                z_half=z_half,
                x_bounds=(x_min_t, x_max_t),
                u_bounds=(u_min_t, u_max_t),
                d_affine=d_affine,
            )

            # 论文 Eq.(11)：u_t^{RTMPC} = ū*_t + K(x_t - x̄*_t)
            xbar_star = Xbar[0]
            ubar_star = Ubar[0]
            u_rtmpc = ubar_star + K @ (x - xbar_star)

            # 论文 Algorithm 1 第 7 行：把实际访问状态 (x_t, X^des_t) 的 teacher 动作加入数据集
            inp_t = make_policy_input(args.task, x=x, x_des_window=x_des, t=t)
            aug_xs_cycle.append(inp_t.reshape(1, -1))
            aug_us_cycle.append(u_rtmpc.reshape(1, -1))

            # 论文 Algorithm 1 第 8-10 行：在 x̄*_t ⊕ Ẑ 内采样，并用 Eq.(13) 生成动作标签
            # 这里用轴对齐盒外包框 z_half 作为 Ẑ（对应论文 Fig.4 的 dense/sparse）。
            xs_plus = sample_box_points(center=xbar_star, half=z_half, mode=args.augment)
            delta = (xs_plus - xbar_star.reshape(1, -1)).T  # (n, Ns)
            us_plus = ubar_star.reshape(-1, 1) + K @ delta

            # 为每个采样点拼接同一个 X^des_t（与论文 Line 10 一致）
            for j in range(xs_plus.shape[0]):
                inp_plus = make_policy_input(args.task, x=xs_plus[j], x_des_window=x_des, t=t)
                aug_xs_cycle.append(inp_plus.reshape(1, -1))
                aug_us_cycle.append(us_plus[:, j].reshape(1, -1))

            # 论文 Algorithm 1 第 11 行：DAgger/BC 混合执行（凸组合）
            u_student = policy_forward(model, inp_t, device=args.device)
            u_exec = float(beta) * u_rtmpc + (1.0 - float(beta)) * u_student
            u_exec = np.clip(u_exec, u_min_base, u_max_base)

            # 论文 Algorithm 1 第 12 行：在训练域 S 推进系统（这里默认无扰动）
            w = rng.uniform(low=-w_half_train, high=w_half_train)
            x = sim.step(x, u_exec) + w

        xs_cycle = np.vstack(aug_xs_cycle)
        us_cycle = np.vstack(aug_us_cycle)

        xs_hist.append(xs_cycle)
        us_hist.append(us_cycle)

        xs_all = torch.as_tensor(np.vstack(xs_hist), dtype=torch.float32)
        us_all = torch.as_tensor(np.vstack(us_hist), dtype=torch.float32)

        cycle_data_path = os.path.join(args.out_dir, f"cycle_{cycle_idx+1:02d}.npz")
        np.savez(cycle_data_path, xs=xs_cycle, us=us_cycle)
        print(f"  saved dataset: {cycle_data_path} | cycle_samples={len(xs_cycle)} | total_samples={len(xs_all)}")

        history = train_one_cycle(model, opt, xs_all, us_all, cfg=train_cfg)

        train_losses_all.extend(history["train"])
        val_losses_all.extend(history["val"])
        cycle_boundaries.append(len(train_losses_all))

        cycle_loss_path = os.path.join(args.out_dir, f"loss_cycle_{cycle_idx+1:02d}.npz")
        np.savez(
            cycle_loss_path,
            train=np.asarray(history["train"], dtype=float),
            val=np.asarray(history["val"], dtype=float),
        )

        cycle_plot_path = os.path.join(args.out_dir, f"loss_cycle_{cycle_idx+1:02d}.png")
        if _try_plot_loss_curves(
            train_losses=history["train"],
            val_losses=history["val"],
            out_path=cycle_plot_path,
            title=f"Cycle {cycle_idx+1} loss",
        ):
            print(f"  saved loss plot: {cycle_plot_path}")
        else:
            print("  [warn] matplotlib not available, skipped per-cycle loss plot")

        ckpt_path = os.path.join(args.out_dir, f"policy_cycle_{cycle_idx+1:02d}.pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hidden": hidden,
                "input_dim": policy_in_dim,
                "output_dim": m,
                "task": args.task,
                "dynamics": args.dynamics,
                "horizon": int(args.horizon),
                "sim_steps": int(args.sim_steps),
                "cycle": cycle_idx + 1,
            },
            ckpt_path,
        )
        print(f"  saved checkpoint: {ckpt_path}")

    all_loss_path = os.path.join(args.out_dir, "loss_all_cycles.npz")
    np.savez(
        all_loss_path,
        train=np.asarray(train_losses_all, dtype=float),
        val=np.asarray(val_losses_all, dtype=float),
        cycle_boundaries=np.asarray(cycle_boundaries, dtype=int),
    )

    all_plot_path = os.path.join(args.out_dir, "loss_all_cycles.png")
    if _try_plot_loss_curves(
        train_losses=train_losses_all,
        val_losses=val_losses_all,
        out_path=all_plot_path,
        title="Training loss across all cycles",
        cycle_boundaries=cycle_boundaries,
    ):
        print(f"saved overall loss plot: {all_plot_path}")
    else:
        print("[warn] matplotlib not available, skipped overall loss plot")


if __name__ == "__main__":
    main()
