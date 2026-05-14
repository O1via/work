"""rtmpc_demo.py
=================
单文件演示：线性 RTMPC 的关键组成放在一个程序中，便于阅读与调试。

包含部分：
- 简单线性模型（DoubleIntegrator）
- 无限时域 LQR（DARE）求解：计算终端代价 Px 与稳态反馈 K
- 基于 OSQP 的有限时域带约束 QP 求解器（严格贴近论文 Eq.(10)）
- 将 QP 结果与辅助控制器结合（Eq.(11)），并在仿真中演示闭环行为

所有注释均为中文，代码尽量简洁且自包含（除非依赖 `osqp`、`numpy`、`scipy`）。
"""
from typing import Optional, Tuple
import itertools
import argparse
import os
import sys
from pathlib import Path

import osqp
import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_discrete_are
from gp_residual_model import VelocityResidualGP, residual_shrink_bounds
from rtmpc_constants import (
    sample_process_disturbance,
    base_initial_state,
    base_input_bounds,
    base_state_bounds,
    disturbance_half_bounds,
    input_cost_matrix,
    state_cost_matrix,
)


def _maybe_reexec_into_workspace_venv() -> None:
    """尽量让“VS Code Run Code”也能使用正确的依赖环境。

    一些 VS Code/Code Runner 配置会用系统 python 直接运行当前文件，导致找不到
    osqp/scipy/matplotlib 等包（它们实际装在工作区根目录的 .venv 里）。

    这里做一个安全的自动切换：
    - 若当前解释器不是工作区 .venv/bin/python 且该文件存在，则 execv 到它
    - 通过环境变量防止递归重启
    """
    if os.environ.get("RTMPC_NO_REEXEC") == "1":
        return

    this_file = Path(__file__).resolve()
    workspace_root = this_file.parent.parent  # .../work
    venv_python = workspace_root / ".venv" / "bin" / "python"
    try:
        current = Path(sys.executable).resolve()
    except Exception:
        return

    if venv_python.exists() and current != venv_python.resolve():
        os.environ["RTMPC_NO_REEXEC"] = "1"
        os.execv(str(venv_python), [str(venv_python), *sys.argv])


def build_circle_reference(
    x0: np.ndarray,
    total_len: int,
    dt: float,
    radius: float = 4.0,
    period_steps: int = 63,
) -> np.ndarray:
    """生成圆形参考轨迹（在 x-y 平面做匀速圆周运动 ）。

    参考状态: [px, py, vx, vy]
    - 圆心默认选择为使得 k=0 时位置等于 x0[:2]
    - 角速度按 period_steps 个离散步完成一圈
    """
    x0 = np.asarray(x0)
    if total_len < 2:
        raise ValueError("total_len 必须 >= 2")
    if period_steps <= 0:
        raise ValueError("period_steps 必须为正")
    if radius <= 0.0:
        raise ValueError("radius 必须为正")

    p0 = x0[:2].astype(float)
    theta0 = 0.0
    center = p0 - radius * np.array([np.cos(theta0), np.sin(theta0)])

    k = np.arange(total_len, dtype=float)
    theta = theta0 + 2.0 * np.pi * k / float(period_steps)
    pos = center.reshape(1, 2) + radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)

    omega = 2.0 * np.pi / (float(period_steps) * float(dt))
    vel = radius * omega * np.stack([-np.sin(theta), np.cos(theta)], axis=1)

    return np.hstack([pos, vel])


def apply_tracking_profile_iris(
    x_ref: np.ndarray,
    dt: float,
    tracking_profile: str,
    g: float = 9.81,
    phi_bounds: Optional[Tuple[float, float]] = None,
    theta_bounds: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """为 8 维 iris 参考轨迹应用两套模式。

    - paper_baseline: 与论文线性设置一致，phi/theta 参考恒为 0。
    - high_speed_extension: 根据速度差分得到期望加速度，并反解 phi/theta 参考。
    """
    ref = np.asarray(x_ref, dtype=float)
    if ref.ndim != 2 or ref.shape[1] < 8:
        raise ValueError("x_ref 形状非法，期望 (T,>=8)")
    if dt <= 0.0:
        raise ValueError("dt 必须为正")

    out = ref.copy()
    if tracking_profile == "paper_baseline":
        out[:, 6] = 0.0
        out[:, 7] = 0.0
        return out
    if tracking_profile != "high_speed_extension":
        raise ValueError("tracking_profile 应为 'paper_baseline' 或 'high_speed_extension'")

    if out.shape[0] >= 2:
        a_n = np.gradient(out[:, 2], float(dt), edge_order=1)
        a_e = np.gradient(out[:, 3], float(dt), edge_order=1)
    else:
        a_n = np.zeros((out.shape[0],), dtype=float)
        a_e = np.zeros((out.shape[0],), dtype=float)

    theta_ref = a_n / float(g)
    phi_ref = a_e / float(g)

    if phi_bounds is not None:
        phi_ref = np.clip(phi_ref, float(phi_bounds[0]), float(phi_bounds[1]))
    if theta_bounds is not None:
        theta_ref = np.clip(theta_ref, float(theta_bounds[0]), float(theta_bounds[1]))

    out[:, 6] = phi_ref
    out[:, 7] = theta_ref
    return out




def compute_rpi_box(A_cl: np.ndarray, w_half: np.ndarray, max_iter: int = 500, tol: float = 1e-9) -> np.ndarray:
    """计算围绕原点的鲁棒正不变集 Z 的轴对齐外包框半径。

    采用区间迭代：h_{k+1} = w_half + |A_cl| h_k。
    若 A_cl 稳定且扰动有界，则该序列收敛到 Z 的最小轴对齐外包框。
    返回值为半径向量 h，使得 Z = {z | |z_i| <= h_i}。
    """
    # 使用级数求和：Z = sum_{i=0}^{∞} A_cl^i W，外包框为 sum |A_cl^i| w_half
    n = A_cl.shape[0]
    h = w_half.copy()
    A_power = np.eye(n)
    for _ in range(1, max_iter + 1):
        A_power = A_power @ A_cl
        term = np.abs(A_power) @ w_half
        h += term
        if np.linalg.norm(term, ord=np.inf) < tol:
            return h
    return h  # 若未收敛，返回当前求和作为保守外包框


def tighten_box_bounds(x_min: np.ndarray, x_max: np.ndarray, shrink: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对盒约束执行 Pontryagin 差：X ⊖ shrink_box。

    给定原始上下界 x_min, x_max（逐元素），以及需收紧的半径 shrink，
    返回收紧后的上下界。如果出现不可行（上界 <= 下界），抛出错误。
    """
    x_min_t = x_min + shrink
    x_max_t = x_max - shrink
    if np.any(x_max_t <= x_min_t):
        raise ValueError("约束收紧后不可行：检查 Z 或原始界限")
    return x_min_t, x_max_t


def tighten_box_bounds_with_auto_scale(
    x_min: np.ndarray,
    x_max: np.ndarray,
    shrink: np.ndarray,
    name: str,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """在必要时自动缩放收紧量，避免 demo 因不可行直接报错。

    返回 (x_min_t, x_max_t, gamma)，其中 gamma∈(0,1]。
    - gamma=1: 原始收紧量可行
    - gamma<1: 已按最大可行比例缩放 shrink
    """
    x_min = np.asarray(x_min, dtype=float)
    x_max = np.asarray(x_max, dtype=float)
    shrink = np.asarray(shrink, dtype=float)

    width = x_max - x_min
    if np.any(width <= 0.0):
        raise ValueError(f"{name} 原始约束无效：存在 x_max <= x_min")

    # 约束可行条件：x_min + gamma*shrink < x_max - gamma*shrink
    # 即 gamma < width/(2*shrink)（仅对 shrink>0 的维度约束）。
    ratios = []
    for w, s in zip(width, shrink):
        if s > 0.0:
            ratios.append((w - eps) / (2.0 * s))
    gamma_max = min(ratios) if ratios else 1.0
    gamma = float(min(1.0, gamma_max))

    if gamma <= 0.0:
        raise ValueError(f"{name} 约束收紧后不可行：原始界限过紧或收紧量过大")

    if gamma < 1.0:
        print(
            f"[warn] {name} tightened set infeasible with full shrink; "
            f"auto-scale shrink by gamma={gamma:.4f} for demo feasibility."
        )

    x_min_t = x_min + gamma * shrink
    x_max_t = x_max - gamma * shrink
    if np.any(x_max_t <= x_min_t):
        raise ValueError(f"{name} 约束收紧后仍不可行：请放宽原始界限或减小扰动")
    return x_min_t, x_max_t, gamma

"""从轴对齐盒 center ± half 内采样点。

mode="dense": 取所有顶点（2^n），对应论文 Fig.4 的 dense 策略。
mode="sparse": 取每个面的中心（2n），对应论文 Fig.4 的 sparse 策略。
返回形状 (Ns, n)。
"""
def sample_box_points(center: np.ndarray, half: np.ndarray, mode: str = "dense") -> np.ndarray:

    center = np.asarray(center)
    half = np.asarray(half)
    n = center.shape[0]
    if mode == "dense":
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        pts = center + signs * half
    elif mode == "sparse":
        pts_list = []
        for i in range(n):
            e = np.zeros(n)
            e[i] = half[i]
            pts_list.append(center + e)
            pts_list.append(center - e)
        pts = np.vstack(pts_list)
    else:
        raise ValueError("mode 应为 dense 或 sparse")
    return pts


def augment_at_center(
    x_center: np.ndarray,
    u_center: np.ndarray,
    K: np.ndarray,
    z_half: np.ndarray,
    mode: str = "dense",
) -> Tuple[np.ndarray, np.ndarray]:
    """对齐论文 Algorithm 1 的采样位置：只围绕当前 tube center 采样。

    采样：x^+ ∈ x̄_t^* ⊕ \hat Z（这里用轴对齐盒 half=z_half）。
    标注：u^+ = ū_t^* + K(x^+ - x̄_t^*)（线性情形对应论文 Eq.(13)）。
    """
    x_center = np.asarray(x_center).reshape(-1)
    u_center = np.asarray(u_center).reshape(-1)
    z_half = np.asarray(z_half).reshape(-1)
    pts = sample_box_points(x_center, z_half, mode=mode)
    n = x_center.size
    m = u_center.size
    delta = (pts - x_center.reshape(1, n)).T  # (n, Ns)
    u_pts = u_center.reshape(m, 1) + K @ delta
    return pts, u_pts.T

"""离散时间双积分模型（演示用）。

状态: x = [px, py, vx, vy]
控制: u = [ax, ay]
动力学: x_{t+1} = A x_t + B u_t
"""
class DoubleIntegrator:
    def __init__(self, dt: float = 0.1):
        self.dt = float(dt)
        dt2 = 0.5 * self.dt * self.dt
        self.A = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
        self.B = np.array([
            [dt2, 0.0],
            [0.0, dt2],
            [self.dt, 0.0],
            [0.0, self.dt],
        ], dtype=float)

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.A @ x + self.B @ u


class LinearIrisHover:
    """PX4 iris 的简化悬停线性模型（离散时间，NED 坐标）。

        状态（8维，PX4/NED 习惯）：
            x = [pn, pe, vn, ve, pd, vd, phi, theta]
            - n/e/d 分别是 North / East / Down
            - pd > 0 表示“向下”位置，通常飞行高度为负 pd

        输入（3维）：
            u = [dT, phi_cmd, theta_cmd]
            - dT 为相对悬停推力增量（N），dT>0 表示推力增大（向上加速）

        与文献 [6]（Kamel et al., 2017）一致的内环近似：
            phi_dot   = (k_phi*phi_cmd   - phi)   / tau_phi
            theta_dot = (k_theta*theta_cmd - theta) / tau_theta
        即包含两个时间常数 tau_phi, tau_theta 与两个增益 k_phi, k_theta。
    """

    def __init__(
        self,
        dt: float = 0.1,
        mass: float = 1.5,
        g: float = 9.81,
        tau_phi: float = 0.23646944886126184,
        tau_theta: float = 0.24323448504520204,
        k_phi: float = 1.032622078891337,
        k_theta: float = 1.0374153949636997,
    ):
        self.dt = float(dt)
        self.mass = float(mass)
        self.g = float(g)
        self.tau_phi = float(tau_phi)
        self.tau_theta = float(tau_theta)
        self.k_phi = float(k_phi)
        self.k_theta = float(k_theta)

        n, m = 8, 3
        Ac = np.zeros((n, n), dtype=float)
        Bc = np.zeros((n, m), dtype=float)

        # p_dot = v (NED)
        Ac[0, 2] = 1.0  # pn_dot = vn
        Ac[1, 3] = 1.0  # pe_dot = ve
        Ac[4, 5] = 1.0  # pd_dot = vd

        # v_dot (hover linearization, psi≈0)
        Ac[2, 7] = self.g      # vn_dot = +g * theta
        Ac[3, 6] = self.g      # ve_dot = +g * phi
        Bc[5, 0] = -1.0 / self.mass  # vd_dot = -dT / m (d轴向下)

        # attitude first-order closed-loop model (ref [6], Eq. (4a)(4b))
        Ac[6, 6] = -1.0 / self.tau_phi
        Ac[7, 7] = -1.0 / self.tau_theta
        Bc[6, 1] = self.k_phi / self.tau_phi
        Bc[7, 2] = self.k_theta / self.tau_theta

        # Euler discretization
        self.A = np.eye(n) + self.dt * Ac
        self.B = self.dt * Bc

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.A @ x + self.B @ u

"""求解离散代数 Riccati 方程（DARE），返回 (P, K_inf)。

优先使用 SciPy 的 `solve_discrete_are`，失败时退回到简单迭代。
返回的 `P` 与 `K_inf` 对应论文中用于终端代价 Px 和辅助增益 K。
"""
def compute_infinite_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
                         tol: float = 1e-9, maxiter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    P = solve_discrete_are(A, B, Q, R)
    S = R + B.T @ P @ B
    K_inf = -np.linalg.solve(S, B.T @ P @ A)
    return P, K_inf


def solve_rtmc_qp_paper(
    A: np.ndarray,
    B: np.ndarray,
    Qx: np.ndarray,
    Ru: np.ndarray,
    Px: np.ndarray,
    x_meas: np.ndarray,
    x_des: np.ndarray,
    N: int,
    z_half: np.ndarray,
    x_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    u_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    d_affine: Optional[np.ndarray] = None,
    eps_abs: float = 1e-6,
    eps_rel: float = 1e-6,
    max_iter: int = 20000,
) -> Tuple[np.ndarray, np.ndarray]:
    """严格贴近论文 Eq.(10) 的线性 RTMPC QP（盒约束版本）。

    与 `solve_rtmc_qp` 的主要区别：
    - 将名义初值 x̄_{0|t} 作为优化变量，而不是参数。
    - 显式加入 tube 约束：x_t ∈ x̄_{0|t} ⊕ Z。
      在盒近似下等价于：x̄_{0|t} ∈ [x_t - z_half, x_t + z_half]。

    变量：
      - U_stack = [ū_{0|t}, ..., ū_{N-1|t}] ∈ R^{mN}
      - X_stack = [x̄_{0|t}, ..., x̄_{N|t}] ∈ R^{n(N+1)}

    约束：
      - 动力学：x̄_{i+1|t} = A x̄_{i|t} + B ū_{i|t} + d_i
        其中 d_i 可由 `d_affine` 传入（默认全 0，等价于原实现）。
      - 收紧约束：x̄_{i|t} ∈ X ⊖ Z，ū_{i|t} ∈ U ⊖ KZ（通过传入 box bounds 实现）
      - tube 初值：x_t ∈ x̄_{0|t} ⊕ Z（通过 z_half 盒实现）

    返回：Xbar (N+1,n), Ubar (N,m)，其中 Xbar[0]=x̄_{0|t}^*。
    """
    x_meas = np.asarray(x_meas).reshape(-1)
    n = A.shape[0]
    m = B.shape[1]
    if x_meas.size != n:
        raise ValueError("x_meas 维度不匹配")

    z_half = np.asarray(z_half).reshape(-1)
    if z_half.size != n:
        raise ValueError("z_half 维度不匹配")

    # 参考轨迹 r_0..r_N
    x_des = np.asarray(x_des)
    if x_des.size == n:
        x_des_stack = np.tile(x_des.reshape(1, -1), (N + 1, 1))
    else:
        x_des_stack = x_des.reshape(N + 1, n)
    r_stack = x_des_stack.reshape(n * (N + 1))

    # Hessian：blockdiag(R_bar, Q_bar)
    R_bar = sp.block_diag([Ru] * N).tocsc()
    Q_blocks = [Qx] * N + [Px]
    Q_bar = sp.block_diag(Q_blocks).tocsc()
    Pz = sp.block_diag([R_bar, Q_bar]).tocsc()

    q_u = np.zeros(m * N)
    q_x = (-Q_bar @ r_stack).reshape(n * (N + 1))
    q = np.concatenate([q_u, q_x])

    # 动力学等式：x_{i+1} - A x_i - B u_i = d_i
    A_x = sp.lil_matrix((n * N, n * (N + 1)))
    for i in range(N):
        A_x[i * n : (i + 1) * n, i * n : (i + 1) * n] = -A
        A_x[i * n : (i + 1) * n, (i + 1) * n : (i + 2) * n] = np.eye(n)
    A_x = A_x.tocsc()
    A_u = sp.block_diag([-B] * N).tocsc()
    A_eq = sp.hstack([A_u, A_x]).tocsc()
    if d_affine is None:
        d_stack = np.zeros(n * N)
    else:
        d_affine = np.asarray(d_affine, dtype=float)
        if d_affine.size == n:
            d_stack = np.tile(d_affine.reshape(1, n), (N, 1)).reshape(n * N)
        else:
            d_affine = d_affine.reshape(N, n)
            d_stack = d_affine.reshape(n * N)
    l_eq = d_stack.copy()
    u_eq = d_stack.copy()

    # 变量 box bounds
    z_dim = m * N + n * (N + 1)
    lb = -np.inf * np.ones(z_dim)
    ub = np.inf * np.ones(z_dim)

    # u bounds: 作用于所有 ū_0..ū_{N-1}
    if u_bounds is not None:
        u_min, u_max = u_bounds
        u_min = np.broadcast_to(np.asarray(u_min).reshape(-1, m), (N, m)).reshape(m * N)
        u_max = np.broadcast_to(np.asarray(u_max).reshape(-1, m), (N, m)).reshape(m * N)
        lb[: m * N] = u_min
        ub[: m * N] = u_max

    # x bounds: 论文 Eq.(10) 约束 i=0..N-1（终端 i=N 不强制）
    x_lb = -np.inf * np.ones(n * (N + 1))
    x_ub = np.inf * np.ones(n * (N + 1))
    if x_bounds is not None:
        x_min, x_max = x_bounds
        x_min_blk = np.broadcast_to(np.asarray(x_min).reshape(-1, n), (N, n)).reshape(n * N)
        x_max_blk = np.broadcast_to(np.asarray(x_max).reshape(-1, n), (N, n)).reshape(n * N)
        x_lb[: n * N] = x_min_blk
        x_ub[: n * N] = x_max_blk

    # tube 初值：x̄0 ∈ [x_t - z_half, x_t + z_half]
    x0_lb = x_meas - z_half
    x0_ub = x_meas + z_half
    x_lb[:n] = np.maximum(x_lb[:n], x0_lb)
    x_ub[:n] = np.minimum(x_ub[:n], x0_ub)
    if np.any(x_ub[:n] <= x_lb[:n]):
        bad_idx = np.where(x_ub[:n] <= x_lb[:n])[0].tolist()
        details = ", ".join(
            f"dim={i}, lb={x_lb[i]:.6f}, ub={x_ub[i]:.6f}, x_meas={x_meas[i]:.6f}, z={z_half[i]:.6f}"
            for i in bad_idx
        )
        raise ValueError(f"tube 初值约束与状态约束冲突: {details}")

    lb[m * N :] = x_lb
    ub[m * N :] = x_ub

    # 合并为 OSQP: [A_eq; I] z ∈ [l,u]
    A_total = sp.vstack([A_eq, sp.eye(z_dim)]).tocsc()
    l_total = np.concatenate([l_eq, lb])
    u_total = np.concatenate([u_eq, ub])

    prob = osqp.OSQP()
    prob.setup(
        P=Pz,
        q=q,
        A=A_total,
        l=l_total,
        u=u_total,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        max_iter=max_iter,
        warm_start=True,
        polish=True,
        adaptive_rho=True,
        verbose=False,
    )
    res = prob.solve()
    if res.info.status_val not in (1, 2):
        # 常见于条件数较差时触发 "maximum iterations reached"。
        # demo 场景下尝试一次更宽松容差和更大迭代上限的回退求解。
        prob_fb = osqp.OSQP()
        prob_fb.setup(
            P=Pz,
            q=q,
            A=A_total,
            l=l_total,
            u=u_total,
            eps_abs=max(eps_abs, 1e-4),
            eps_rel=max(eps_rel, 1e-4),
            max_iter=max(max_iter, 50000),
            warm_start=True,
            polish=True,
            adaptive_rho=True,
            verbose=False,
        )
        res = prob_fb.solve()
        if res.info.status_val not in (1, 2):
            raise RuntimeError(f"OSQP 求解失败，状态: {res.info.status}")

    z = res.x
    U = z[: m * N].reshape(N, m)
    X = z[m * N :].reshape(N + 1, n)
    return X, U


def solve_rtmc_qp_with_gp_stagewise(
    A: np.ndarray,
    B: np.ndarray,
    Qx: np.ndarray,
    Ru: np.ndarray,
    Px: np.ndarray,
    x_meas: np.ndarray,
    x_des: np.ndarray,
    N: int,
    z_half: np.ndarray,
    x_bounds: Tuple[np.ndarray, np.ndarray],
    u_bounds: Tuple[np.ndarray, np.ndarray],
    gp_model: Optional[VelocityResidualGP] = None,
    gp_beta_sigma: float = 2.0,
    stagewise_refine_steps: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """QP + GP 均值注入（逐步更新版本）。

    过程：
    1) 先在不注入 d_affine 的条件下求一次名义 QP（基线预测轨迹）；
    2) 用该预测中心轨迹 Xbar[i] 逐步重算 d_i=d_mean(Xbar[i])；
    3) 带阶段相关 d_affine 再求解 QP（可配置迭代次数，默认 1 次）。
    """
    if gp_model is None:
        Xbar, Ubar = solve_rtmc_qp_paper(
            A=A,
            B=B,
            Qx=Qx,
            Ru=Ru,
            Px=Px,
            x_meas=x_meas,
            x_des=x_des,
            N=N,
            z_half=z_half,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            d_affine=None,
        )
        return Xbar, Ubar, None, None

    # 第 1 步：不注入 d_affine，先求一条“中性”预测轨迹。
    Xbar, Ubar = solve_rtmc_qp_paper(
        A=A,
        B=B,
        Qx=Qx,
        Ru=Ru,
        Px=Px,
        x_meas=x_meas,
        x_des=x_des,
        N=N,
        z_half=z_half,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        d_affine=None,
    )

    # 第 2 步起：基于预测轨迹逐步更新 d_affine，并重复求解。
    d_affine = None
    prev_d_affine = None
    for _ in range(max(1, int(stagewise_refine_steps))):
        d_rows = []
        for i in range(N):
            di, _ = gp_model.predict_state_disturbance(
                x=Xbar[i],
                beta_sigma=float(gp_beta_sigma),
            )
            d_rows.append(np.asarray(di, dtype=float).reshape(-1))
        d_affine_new = np.vstack(d_rows)
        d_affine = d_affine_new
        Xbar, Ubar = solve_rtmc_qp_paper(
            A=A,
            B=B,
            Qx=Qx,
            Ru=Ru,
            Px=Px,
            x_meas=x_meas,
            x_des=x_des,
            N=N,
            z_half=z_half,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            d_affine=d_affine,
        )
        if prev_d_affine is not None and np.allclose(d_affine, prev_d_affine, rtol=1e-3, atol=1e-6):
            break
        prev_d_affine = d_affine.copy()

    d0 = None if d_affine is None else np.asarray(d_affine[0], dtype=float).reshape(-1)
    return Xbar, Ubar, d_affine, d0


def demo(
    task: str = "tracking",
    dynamics: str = "iris_linear",
    disturbance_mode: str = "force_only",
    force_bound_mg: float = 0.5,
    force_d_axis_scale: float = 0.15,
    gp_model_path: Optional[str] = None,
    gp_beta_sigma: float = 2.0,
    gp_shrink_mode: str = "residual",
    tracking_shape: str = "line",
    tracking_profile: str = "paper_baseline",
    sim_steps: int = 60,
    circle_radius: float = 4.0,
    circle_period_steps: int = 126,
    plot_path: Optional[str] = None,
    augment_mode: str = "dense",
    include_onpolicy_teacher: bool = True,
):
    """运行一个端到端示例，展示 Eq.(10) 与 Eq.(11) 在代码中的映射与行为。

        task:
            - "point": 目标点调节
            - "tracking": 轨迹跟踪

        tracking_shape:
            - "line": 位置从起点匀速移动到目标点，然后保持
            - "circle": 圆形参考轨迹（在 x-y 平面做匀速圆周运动）
    """
    if dynamics == "double_integrator":
        sim = DoubleIntegrator(dt=0.1)
    elif dynamics == "iris_linear":
        sim = LinearIrisHover(dt=0.1, mass=1.5)
    else:
        raise ValueError("dynamics 应为 'double_integrator' 或 'iris_linear'")

    A, B = sim.A, sim.B
    n, m = A.shape[0], B.shape[1]

    # 代价设置（示例）
    # 调高位置权重、适度增加速度权重，略降输入权重以提高末端精度
    Qx = state_cost_matrix(dynamics)
    Ru = input_cost_matrix(dynamics, m)

    # 先求 DARE 得到 Px 与 K（论文里的终端代价 Px 与辅助收益 K）
    Px, K = compute_infinite_lqr(A, B, Qx, Ru)
    print("Px:")
    print(Px)
    print("K_inf:")
    print(K)

    # 原始状态/控制约束盒（示例）
    x_min_base, x_max_base = base_state_bounds(dynamics)
    if dynamics == "double_integrator":
        u_min_base, u_max_base = base_input_bounds(dynamics, m=m)
    else:
        u_min_base, u_max_base = base_input_bounds(dynamics, mass=float(sim.mass))
    x0 = base_initial_state(dynamics)

    # 干扰假设与鲁棒管 Z 计算
    # 假设加性扰动 w ∈ [-w_half, w_half]（每维独立，演示用）
    # 共享扰动边界，确保与训练/验证脚本一致。
    w_half = disturbance_half_bounds(
        dynamics,
        dt=float(sim.dt),
        mode=disturbance_mode,
        force_bound_mg=float(force_bound_mg),
        force_d_axis_scale=float(force_d_axis_scale),
    )
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
    print(f"[tube] residual disturbance bound after GP shrink (mode={gp_shrink_mode}): {w_half}")

    A_cl = A + B @ K
    z_half = compute_rpi_box(A_cl, w_half)
    u_half = np.abs(K) @ z_half  # KZ 的轴对齐外包框
    print("tube half-width z_half:")
    print(z_half)

    # 收紧约束：X_tight = X ⊖ Z，U_tight = U ⊖ KZ
    # 若 full-shrink 不可行，demo 自动缩放收紧量避免直接报错。
    x_min_t, x_max_t, gamma_x = tighten_box_bounds_with_auto_scale(
        x_min_base, x_max_base, z_half, name="state"
    )
    u_min_t, u_max_t, gamma_u = tighten_box_bounds_with_auto_scale(
        u_min_base, u_max_base, u_half, name="input"
    )
    if gamma_x < 1.0 or gamma_u < 1.0:
        print(
            f"[info] tighten scale factors: gamma_x={gamma_x:.4f}, gamma_u={gamma_u:.4f}"
        )

    N = 30
    # 可选任务构造：目标点调节 vs 轨迹跟踪
    if task == "point":
        # 从当前状态 x0 跟踪到平衡点（iris 保持当前高度）
        x_ref_all = np.zeros((N + 1, n))
        if n == 8:
            x_ref_all[:, 4] = float(x0[4])
    elif task == "tracking":
        if tracking_profile not in ("paper_baseline", "high_speed_extension"):
            raise ValueError("tracking_profile 应为 'paper_baseline' 或 'high_speed_extension'")
        total_len = int(sim_steps) + N + 1  # 需要可切片的参考长度（t..t+N）
        if tracking_shape == "circle":
            if circle_period_steps <= 0:
                raise ValueError("circle_period_steps 必须为正")
            if n == 4:
                # 圆形匀速运动需要向心加速度 a = r * omega^2。
                # 若收紧后的输入上限不足以提供该加速度，则跟踪会明显偏离（这是物理/约束不可行）。
                omega = 2.0 * np.pi / (float(circle_period_steps) * float(sim.dt))
                a_req = float(circle_radius) * omega * omega
                u_cap = float(np.min(np.abs(u_max_t)))
                if a_req > u_cap:
                    print(
                        "[warn] circle tracking may be infeasible under input bounds: "
                        f"required centripetal accel ~ {a_req:.3f} > tightened u_max ~ {u_cap:.3f}. "
                        "Try increasing --circle-period-steps (slower), decreasing --circle-radius, or relaxing u bounds."
                    )
                x_ref_all = build_circle_reference(
                    x0=x0,
                    total_len=total_len,
                    dt=sim.dt,
                    radius=float(circle_radius),
                    period_steps=int(circle_period_steps),
                )
            else:
                # iris 8维：在 n-e 平面给圆轨迹，pd 维度保持常值。
                xy_ref = build_circle_reference(
                    x0=x0[:4],
                    total_len=total_len,
                    dt=sim.dt,
                    radius=float(circle_radius),
                    period_steps=int(circle_period_steps),
                )
                x_ref_all = np.zeros((total_len, n), dtype=float)
                x_ref_all[:, 0] = xy_ref[:, 0]
                x_ref_all[:, 1] = xy_ref[:, 1]
                x_ref_all[:, 2] = xy_ref[:, 2]
                x_ref_all[:, 3] = xy_ref[:, 3]
                x_ref_all[:, 4] = float(x0[4])
                x_ref_all = apply_tracking_profile_iris(
                    x_ref_all,
                    dt=float(sim.dt),
                    tracking_profile=tracking_profile,
                    g=float(sim.g),
                    phi_bounds=(float(x_min_base[6]), float(x_max_base[6])),
                    theta_bounds=(float(x_min_base[7]), float(x_max_base[7])),
                )

            # 给定与轨迹相同的初始状态。
            x0 = x_ref_all[0].copy()
            print(f"[info] align initial state to circle reference: x0={x0}")
            
        elif tracking_shape == "line":
            # 轨迹跟踪示例：位置从 p_start 匀速移动到 p_goal，速度参考与之匹配；到达后保持。
            p_start = x0[:2].copy()
            p_goal = np.array([0.0, 0.0])
            T_move = max(1, int(sim_steps))  # 用 sim_steps 步完成移动

            alphas = np.linspace(0.0, 1.0, T_move + 1)
            pos_move = (1.0 - alphas)[:, None] * p_start[None, :] + alphas[:, None] * p_goal[None, :]
            pos_ref = np.vstack([pos_move, np.tile(p_goal[None, :], (total_len - (T_move + 1), 1))])
            v_const = (p_goal - p_start) / (T_move * sim.dt)
            vel_ref = np.vstack([
                np.tile(v_const[None, :], (T_move + 1, 1)),
                np.zeros((total_len - (T_move + 1), 2)),
            ])
            if n == 4:
                x_ref_all = np.hstack([pos_ref, vel_ref])
            else:
                x_ref_all = np.zeros((total_len, n), dtype=float)
                x_ref_all[:, 0:2] = pos_ref
                x_ref_all[:, 2:4] = vel_ref
                x_ref_all[:, 4] = float(x0[4])
                x_ref_all = apply_tracking_profile_iris(
                    x_ref_all,
                    dt=float(sim.dt),
                    tracking_profile=tracking_profile,
                    g=float(sim.g),
                    phi_bounds=(float(x_min_base[6]), float(x_max_base[6])),
                    theta_bounds=(float(x_min_base[7]), float(x_max_base[7])),
                )
        else:
            raise ValueError("tracking_shape 应为 'line' 或 'circle'")
    else:
        raise ValueError("task 应为 'point' 或 'tracking'")

    # Receding-horizon RTMPC（对齐论文 Eq.(10)-(11)）：
    # - x: 实际系统（含扰动）
    # - x_bar: 管中心（名义“safe”状态）x̄*t := x̄*0|t，由 QP 规划并用名义动力学推进
    # - u_bar: 管中心名义动作 ū*t := ū*0|t
    # - 实际控制：u_t = ū*t + K(x_t - x̄*t)
    x = x0.copy()
    xs = [x.copy()]
    # 记录“实际的管中心”：每个时刻 t 的 x̄_t 取本次QP解得到的 x̄^*_{0|t} = Xbar[0]。
    # 这与 Eq.(11) 中用于反馈的中心一致，也与 tube membership 检查的中心一致。
    xs_bar = []
    us = []
    rng = np.random.default_rng(seed=0)
    aug_xs = []
    aug_us = []
    # Fig.3 语义检查：真实状态是否始终落在管 x̄ ⊕ Z 内（这里用轴对齐外包框 |e|<=z_half 近似检查）
    tube_violation_count = 0
    tube_max_excess = 0.0
    tube_tol = 1e-9
    
    if augment_mode not in ("dense", "sparse"):
        raise ValueError("augment_mode 应为 'dense' 或 'sparse'")

    for t in range(int(sim_steps)):
        # 轨迹跟踪：按时间推进选择当前窗口参考；目标点调节则退化为恒定/零参考。
        if task == "tracking":
            x_des = x_ref_all[t : t + N + 1]
        else:
            x_des = x_ref_all
        # 论文 Eq.(10)：名义轨迹 (X̄,Ū) 的滚动优化。
        # GP 均值注入采用“逐步更新”：
        # - 先用 x_t 常值平铺求解一次；
        # - 再基于预测轨迹 Xbar[i] 逐点重算 d_affine 并再解一次。
        Xbar, Ubar, _, _ = solve_rtmc_qp_with_gp_stagewise(
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
            gp_model=gp_model,
            gp_beta_sigma=float(gp_beta_sigma),
            stagewise_refine_steps=1,
        )

        # 对齐论文 Algorithm 1：仅围绕“当前时刻”的管中心 x̄_t^*=x̄_{0|t}^* 采样，
        # 而不是对整个预测轨迹每个 i 都采样。
        x_bar_now = Xbar[0]
        u_bar = Ubar[0]
        xs_bar.append(x_bar_now.copy())
        xs_aug_step, us_aug_step = augment_at_center(x_bar_now, u_bar, K, z_half, mode=augment_mode)
        aug_xs.append(xs_aug_step)
        aug_us.append(us_aug_step)

        # Algorithm 1 也会把“实际访问到的输入”(x_t 的 teacher 标签)并入数据集。
        # 在线性情形 teacher 标签即 Eq.(11)：u_t^* = ū_t^* + K(x_t - x̄_t^*)。
        if include_onpolicy_teacher:
            e_now = x - x_bar_now
            u_teacher = u_bar + K @ e_now
            aug_xs.append(x.reshape(1, -1))
            aug_us.append(u_teacher.reshape(1, -1))

        # 论文 Eq.(11)：实际控制 = 名义动作 + K(x - 名义中心)
        # 论文 Eq.(11)：实际控制 = 名义动作 + K(x - 名义中心)
        # （注意：上面为了增广已定义了 x_bar_now, u_bar）
        e = x - x_bar_now
        # tube membership 检查（逐维）：|x_t - x̄_t| <= z_half
        excess = np.abs(e) - z_half
        tube_max_excess = max(tube_max_excess, float(np.max(excess)))
        if np.any(excess > tube_tol):
            tube_violation_count += 1

        u = u_bar + K @ e
        us.append(u.copy())

        # 应用控制并推进系统，并叠加过程扰动（与 mode 语义一致）。
        w = sample_process_disturbance(
            rng=rng,
            dynamics=dynamics,
            dt=float(sim.dt),
            mode=disturbance_mode,
            force_bound_mg=float(force_bound_mg),
            force_d_axis_scale=float(force_d_axis_scale),
            state_dim=n,
            w_half=w_half,
        )
        x = sim.step(x, u) + w
        xs.append(x.copy())

    # 汇总增强数据集
    aug_xs = np.vstack(aug_xs)
    aug_us = np.vstack(aug_us)

    print("trajectory:")
    print(np.array(xs))
    print(
        f"augmented samples: {aug_xs.shape[0]} (mode={augment_mode}, "
        f"include_onpolicy_teacher={include_onpolicy_teacher})"
    )
    print(
        "tube membership check (using box over-approx): "
        f"violations={tube_violation_count}/{int(sim_steps)}, "
        f"max_excess_inf={tube_max_excess:.3e} (tol={tube_tol:.1e})"
    )

    if plot_path is not None:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise ImportError(
                "需要 matplotlib 才能绘图。请先在当前 venv 安装：pip install matplotlib"
            ) from e

        xs_bar_arr = np.array(xs_bar)
        ref_xy = None
        if task == "tracking":
            ref_xy = x_ref_all[: len(xs_bar_arr), :2]

            pos_err = xs_bar_arr[:, :2] - ref_xy
            err_norm = np.linalg.norm(pos_err, axis=1)
            print(f"tube-center tracking position error: mean={err_norm.mean():.4f}, max={err_norm.max():.4f}")

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        if ref_xy is not None:
            ax.plot(ref_xy[:, 0], ref_xy[:, 1], "k--", linewidth=1.0, label="reference")
        ax.plot(xs_bar_arr[:, 0], xs_bar_arr[:, 1], "b-", linewidth=2.0, label="tube center ($\\bar{x}$)")

        # # Tube 边界（包络线）可视化：显示 tube x̄ ⊕ Z 在 (x,y) 平面上的外/内边界。
        # # Z 这里用轴对齐盒外包框 half-width=z_half，因此在单位法向 n 上的支持函数为
        # #   h_Z(n) = dx*|n_x| + dy*|n_y|，其中 dx=z_half[0], dy=z_half[1]。
        # # 令中心轨迹位置为 c(s)，法向为 n(s)，则外/内包络近似为 c(s) ± h_Z(n(s)) n(s)。
        # pos = xs_bar_arr[:, :2]
        # tangent = np.gradient(pos, axis=0)
        # t_norm = np.linalg.norm(tangent, axis=1)
        # t_norm = np.maximum(t_norm, 1e-12)
        # normal = np.stack([-tangent[:, 1] / t_norm, tangent[:, 0] / t_norm], axis=1)
        # dx = float(z_half[0])
        # dy = float(z_half[1])
        # h = dx * np.abs(normal[:, 0]) + dy * np.abs(normal[:, 1])
        # outer = pos + normal * h[:, None]
        # inner = pos - normal * h[:, None]
        # ax.plot(outer[:, 0], outer[:, 1], "b-", linewidth=1.0, alpha=0.25, label="tube boundary")
        # ax.plot(inner[:, 0], inner[:, 1], "b-", linewidth=1.0, alpha=0.25)

        ax.scatter(aug_xs[:, 0], aug_xs[:, 1], s=6, alpha=0.12, c="tab:orange", label="augmented")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{task} / {tracking_shape}: nominal vs augmented")
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        print(f"saved plot to: {plot_path}")


if __name__ == "__main__":
    _maybe_reexec_into_workspace_venv()

    parser = argparse.ArgumentParser(description="Robust Tube MPC demo")
    parser.add_argument("--task", choices=["point", "tracking"], default="tracking")
    parser.add_argument("--dynamics", choices=["double_integrator", "iris_linear"], default="iris_linear")
    parser.add_argument(
        "--disturbance-mode",
        choices=["state_box", "force_only"],
        default="force_only",
        help="扰动构造方案：state_box=仅用当前状态扰动盒；force_only=仅用外力边界映射。",
    )
    parser.add_argument("--force_bound_mg", type=float, default=0.15, help="外力边界系数 c，使 ||f_ext||<=c*m*g")
    parser.add_argument("--force_d_axis_scale", type=float, default=0.15, help="force_only 模式下 d 轴扰动限幅比例（0~1）")
    parser.add_argument("--gp-model", type=str, default="gp_model/iris_linear_residual_gp.npz", help="Optional GP residual model (.npz)")
    parser.add_argument("--gp-beta-sigma", type=float, default=1.0, help="GP uncertainty envelope multiplier")
    parser.add_argument(
        "--gp-shrink-mode",
        choices=["none", "residual"],
        default="residual",
        help="GP收缩模式：none=不收缩；residual=base-gp_comp+gp_unc 的残差边界。",
    )
    parser.add_argument("--tracking-shape", choices=["line", "circle"], default="circle")
    parser.add_argument(
        "--tracking-profile",
        choices=["paper_baseline", "high_speed_extension"],
        default="high_speed_extension",
        help="tracking 参考模式：paper_baseline=phi/theta参考为0；high_speed_extension=由速度差分反解姿态参考。",
    )
    parser.add_argument("--sim-steps", type=int, default=150)
    parser.add_argument("--circle-radius", type=float, default=4.0)
    parser.add_argument("--circle-period-steps", type=int, default=126)
    parser.add_argument(
        "--augment-mode",
        choices=["dense", "sparse"],
        default="dense",
        help="tube-guided 采样策略：dense=2^n 顶点；sparse=2n 面中心",
    )
    parser.add_argument(
        "--no-onpolicy-teacher",
        action="store_true",
        help="若指定，则不把 (x_t, u_t^*) 追加到数据集中（仅保留 tube 采样点）",
    )
    parser.add_argument("--plot", type=str, default=None, help="Optional path to save a PNG plot")
    args = parser.parse_args()
    if not (0.0 <= args.force_d_axis_scale <= 1.0):
        raise ValueError("force_d_axis_scale 必须在 [0,1] 内")

    # VS Code Run Code 通常不带任何命令行参数：给一个可复现实验结果的默认输出。
    if args.plot is None and len(sys.argv) == 1:
        workspace_root = Path(__file__).resolve().parent.parent
        args.plot = str(workspace_root / "circle_tracking_aug.png")

    demo(
        task=args.task,
        dynamics=args.dynamics,
        disturbance_mode=args.disturbance_mode,
        force_bound_mg=args.force_bound_mg,
        force_d_axis_scale=args.force_d_axis_scale,
        gp_model_path=args.gp_model,
        gp_beta_sigma=float(args.gp_beta_sigma),
        gp_shrink_mode=args.gp_shrink_mode,
        tracking_shape=args.tracking_shape,
        tracking_profile=args.tracking_profile,
        sim_steps=args.sim_steps,
        circle_radius=args.circle_radius,
        circle_period_steps=args.circle_period_steps,
        plot_path=args.plot,
        augment_mode=args.augment_mode,
        include_onpolicy_teacher=(not args.no_onpolicy_teacher),
    )
