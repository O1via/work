from typing import Tuple

import numpy as np


IRIS_MASS_DEFAULT = 1.5
IRIS_KF = 5.84e-06
IRIS_W_MAX = 1100.0


def state_cost_matrix(dynamics: str) -> np.ndarray:
    """返回与当前脚本统一的状态权重矩阵 Qx。"""
    if dynamics == "double_integrator":
        return np.diag([50.0, 50.0, 4.0, 4.0])
    if dynamics == "iris_linear":
        # x=[pn,pe,vn,ve,pd,vd,phi,theta] (NED)
        return np.diag([50.0, 50.0, 6.0, 6.0, 50.0, 6.0, 3.0, 3.0])
    raise ValueError("dynamics 应为 'double_integrator' 或 'iris_linear'")


def input_cost_matrix(dynamics: str, m: int) -> np.ndarray:
    """返回输入权重矩阵 Ru。"""
    if m <= 0:
        raise ValueError("m 必须为正")
    if dynamics == "double_integrator":
        return np.eye(m) * 0.01
    if dynamics == "iris_linear":
        # u=[dT, phi_cmd, theta_cmd]，适度放宽姿态通道惩罚以增强跟踪能力。
        if m != 3:
            return np.eye(m) * 0.01
        return np.diag([0.01, 0.5, 0.5])
    raise ValueError("dynamics 应为 'double_integrator' 或 'iris_linear'")


def _base_disturbance_half_bounds(dynamics: str) -> np.ndarray:
    """返回基准扰动盒半宽（状态域经验盒）。"""
    if dynamics == "double_integrator":
        return np.array([0.01, 0.01, 0.005, 0.005], dtype=float)
    if dynamics == "iris_linear":
        # x=[pn,pe,vn,ve,pd,vd,phi,theta]
        return np.array([0.003, 0.003, 0.006, 0.006, 0.008, 0.012, 0.002, 0.002], dtype=float)
    raise ValueError("dynamics 应为 'double_integrator' 或 'iris_linear'")


def force_bound_to_state_w_half(
    dynamics: str,
    dt: float,
    force_bound_mg: float,
    force_d_axis_scale: float = 0.15,
) -> np.ndarray:
    """将外力球约束 ||f_ext||<=c*m*g 映射为离散状态扰动盒半宽。

    说明：论文形式是球约束（L2）。为兼容当前基于盒集的 RTMPC 实现，这里采用
    球内接盒（inscribed box）：每轴加速度半宽取 amax/sqrt(3)。
    这样得到的状态盒满足“盒内任一点都对应不超过球半径”的范数约束。
    """
    if force_bound_mg < 0.0:
        raise ValueError("force_bound_mg 必须 >= 0")
    if force_d_axis_scale < 0.0:
        raise ValueError("force_d_axis_scale 必须 >= 0")

    amax = float(force_bound_mg) * 9.80665
    a_axis = amax / np.sqrt(3.0)
    d_scale = float(min(force_d_axis_scale, 1.0))
    a_axis_d = a_axis * d_scale
    pos_half = 0.5 * float(dt) * float(dt) * a_axis
    vel_half = float(dt) * a_axis
    pos_half_d = 0.5 * float(dt) * float(dt) * a_axis_d
    vel_half_d = float(dt) * a_axis_d

    if dynamics == "double_integrator":
        # x=[px,py,vx,vy]
        return np.array([pos_half, pos_half, vel_half, vel_half], dtype=float)

    if dynamics == "iris_linear":
        # x=[pn,pe,vn,ve,pd,vd,phi,theta]
        # 外力扰动主导平动维，姿态维默认由基准盒覆盖。
        return np.array(
            [pos_half, pos_half, vel_half, vel_half, pos_half_d, vel_half_d, 0.0, 0.0],
            dtype=float,
        )

    raise ValueError("dynamics 应为 'double_integrator' 或 'iris_linear'")


def disturbance_half_bounds(
    dynamics: str,
    dt: float = 0.1,
    mode: str = "state_box",
    force_bound_mg: float = 0.05,
    force_d_axis_scale: float = 0.15,
) -> np.ndarray:
    """返回扰动盒半宽 w_half。

    mode:
    - state_box: 仅使用基准状态域扰动盒（与原实现一致）
    - force_only: 仅使用外力上限映射项
    """
    w_half_base = _base_disturbance_half_bounds(dynamics)
    if mode == "state_box":
        return w_half_base
    if mode == "force_only":
        return force_bound_to_state_w_half(
            dynamics,
            dt=dt,
            force_bound_mg=force_bound_mg,
            force_d_axis_scale=force_d_axis_scale,
        )
    raise ValueError("mode 应为 'state_box' 或 'force_only'")


def sample_force_mapped_acceleration(
    rng: np.random.Generator,
    force_bound_mg: float,
    force_d_axis_scale: float = 0.15,
) -> np.ndarray:
    """采样与 `force_bound_to_state_w_half` 一致的加速度扰动（NED 三轴）。

    这里采用“方案A：设计集合=注入集合”：
    - 设计侧（约束收紧）用的是球内接盒：a_axis = c*g/sqrt(3)
    - 注入侧也直接在同一盒内逐轴均匀采样

    这样可保证 `force_only` 模式下，注入扰动始终落在用于计算 z_half 的
    扰动盒内，避免“注入比设计更大”导致的鲁棒性失配。
    """
    if force_bound_mg < 0.0:
        raise ValueError("force_bound_mg 必须 >= 0")
    if force_d_axis_scale < 0.0:
        raise ValueError("force_d_axis_scale 必须 >= 0")
    amax = float(force_bound_mg) * 9.80665
    a_axis = amax / np.sqrt(3.0)
    d_scale = float(min(force_d_axis_scale, 1.0))
    a_axis_d = a_axis * d_scale
    return np.array(
        [
            rng.uniform(-a_axis, a_axis),
            rng.uniform(-a_axis, a_axis),
            rng.uniform(-a_axis_d, a_axis_d),
        ],
        dtype=float,
    )


def accel_to_state_disturbance(
    dynamics: str,
    dt: float,
    accel_ned: np.ndarray,
    state_dim: int,
) -> np.ndarray:
    """将 NED 加速度映射为离散状态加性扰动 d。"""
    a = np.asarray(accel_ned, dtype=float).reshape(-1)
    if a.size < 3:
        raise ValueError("accel_ned 需要至少 3 维 (n,e,d)")
    d = np.zeros(int(state_dim), dtype=float)
    dt = float(dt)
    dt2 = 0.5 * dt * dt

    if dynamics == "double_integrator":
        if state_dim < 4:
            raise ValueError("double_integrator 需要 state_dim >= 4")
        d[0] = dt2 * a[0]
        d[1] = dt2 * a[1]
        d[2] = dt * a[0]
        d[3] = dt * a[1]
        return d

    if dynamics == "iris_linear":
        if state_dim < 6:
            raise ValueError("iris_linear 需要 state_dim >= 6")
        # x=[pn,pe,vn,ve,pd,vd,phi,theta]
        d[0] = dt2 * a[0]
        d[1] = dt2 * a[1]
        d[2] = dt * a[0]
        d[3] = dt * a[1]
        d[4] = dt2 * a[2]
        d[5] = dt * a[2]
        return d

    raise ValueError("dynamics 应为 'double_integrator' 或 'iris_linear'")


def sample_process_disturbance(
    rng: np.random.Generator,
    dynamics: str,
    dt: float,
    mode: str,
    force_bound_mg: float,
    force_d_axis_scale: float,
    state_dim: int,
    w_half: np.ndarray = None,
) -> np.ndarray:
    """统一采样接口：
    - state_box: 盒均匀采样 d ~ U[-w_half, w_half]
    - force_only: 先采样外力映射加速度，再映射到状态扰动 d
    """
    if mode == "state_box":
        if w_half is None:
            raise ValueError("state_box 模式需要提供 w_half")
        w_half = np.asarray(w_half, dtype=float).reshape(-1)
        if w_half.size != int(state_dim):
            raise ValueError("w_half 维度与 state_dim 不一致")
        return rng.uniform(-w_half, w_half)
    if mode == "force_only":
        a = sample_force_mapped_acceleration(
            rng,
            force_bound_mg=force_bound_mg,
            force_d_axis_scale=force_d_axis_scale,
        )
        return accel_to_state_disturbance(
            dynamics=dynamics,
            dt=dt,
            accel_ned=a,
            state_dim=state_dim,
        )
    raise ValueError("mode 应为 'state_box' 或 'force_only'")


def base_state_bounds(dynamics: str) -> Tuple[np.ndarray, np.ndarray]:
    """返回状态盒约束 (x_min_base, x_max_base)。"""
    if dynamics == "double_integrator":
        return (
            np.array([-5.0, -5.0, -5.0, -5.0], dtype=float),
            np.array([5.0, 5.0, 5.0, 5.0], dtype=float),
        )
    if dynamics == "iris_linear":
        # x=[pn,pe,vn,ve,pd,vd,phi,theta] (NED)
        return (
            np.array([-8.0, -8.0, -2.5, -2.5, -5.0, -1.0, -1.0, -1.0], dtype=float),
            np.array([8.0, 8.0, 2.5, 2.5, -0.2, 1.0, 1.0, 1.0], dtype=float),
        )
    raise ValueError("dynamics 应为 'double_integrator' 或 'iris_linear'")


def gp_query_state_bounds(dynamics: str) -> Tuple[np.ndarray, np.ndarray]:
    """返回 GP 管宽保守估计使用的查询状态域 (x_min_gp, x_max_gp)。

    该边界与 `base_state_bounds`（MPC 硬约束）解耦，便于分别调整：
    - MPC 约束用于 QP 可行域；
    - GP 查询域用于 conservative_mean/uncertainty 的保守扫描范围。
    """
    if dynamics == "double_integrator":
        return base_state_bounds(dynamics)
    if dynamics == "iris_linear":
        # x=[pn,pe,vn,ve,pd,vd,phi,theta] (NED)
        return (
            np.array([-8.0, -8.0, -2.2, -2.2, -5.0, -0.6, -1.0, -1.0], dtype=float),
            np.array([8.0, 8.0, 2.2, 2.2, -0.2, 0.6, 1.0, 1.0], dtype=float),
        )
    raise ValueError("dynamics 应为 'double_integrator' 或 'iris_linear'")


def base_input_bounds(dynamics: str, mass: float = IRIS_MASS_DEFAULT, m: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """返回输入盒约束 (u_min_base, u_max_base)。"""
    if dynamics == "double_integrator":
        if m <= 0:
            raise ValueError("double_integrator 需要提供有效输入维度 m")
        return -3.0 * np.ones(m, dtype=float), 3.0 * np.ones(m, dtype=float)

    if dynamics == "iris_linear":
        t_max = 4.0 * IRIS_KF * (IRIS_W_MAX ** 2)
        t_hover = float(mass) * 9.81
        u_min = np.array([-0.7 * t_hover, -1.0, -1.0], dtype=float)
        u_max = np.array([max(0.1, t_max - t_hover), 1.0, 1.0], dtype=float)
        return u_min, u_max

    raise ValueError("dynamics 应为 'double_integrator' 或 'iris_linear'")


def base_initial_state(dynamics: str) -> np.ndarray:
    """返回默认初始状态 x0。"""
    if dynamics == "double_integrator":
        return np.array([1.0, 0.5, 0.0, 0.0], dtype=float)
    if dynamics == "iris_linear":
        # 默认初始高度约 1m（NED: pd=-1）
        return np.array([0.0, 4.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=float)
    raise ValueError("dynamics 应为 'double_integrator' 或 'iris_linear'")
