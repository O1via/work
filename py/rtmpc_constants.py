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
        return np.diag([40.0, 40.0, 4.0, 4.0, 50.0, 6.0, 3.0, 3.0])
    raise ValueError("dynamics 应为 'double_integrator' 或 'iris_linear'")


def input_cost_matrix(dynamics: str, m: int) -> np.ndarray:
    """返回输入权重矩阵 Ru。"""
    if m <= 0:
        raise ValueError("m 必须为正")
    if dynamics == "double_integrator":
        return np.eye(m) * 0.01
    if dynamics == "iris_linear":
        # u=[dT, phi_cmd, theta_cmd]，提高姿态通道惩罚以降低 |K| 与输入收紧量。
        if m != 3:
            return np.eye(m) * 0.01
        return np.diag([0.01, 0.6, 0.6])
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
    """按论文常见设置采样外力映射加速度（NED 三轴）。

    采样方式：
    - 加速度幅值 a_mag ~ U(0, a_max), a_max = c*g
    - theta ~ U(0, pi), phi ~ U(0, 2pi)
    - a = a_mag * [cos(phi)sin(theta), sin(phi)sin(theta), cos(theta)]
    """
    if force_bound_mg < 0.0:
        raise ValueError("force_bound_mg 必须 >= 0")
    if force_d_axis_scale < 0.0:
        raise ValueError("force_d_axis_scale 必须 >= 0")
    amax = float(force_bound_mg) * 9.80665
    a_mag = rng.uniform(0.0, amax)
    theta = rng.uniform(0.0, np.pi)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    a = np.array(
        [
            a_mag * np.cos(phi) * np.sin(theta),
            a_mag * np.sin(phi) * np.sin(theta),
            a_mag * np.cos(theta),
        ],
        dtype=float,
    )
    # 额外限制 d 轴扰动幅值，保持总加速度范数不超过 amax。
    d_cap = float(min(force_d_axis_scale, 1.0)) * amax
    a[2] = float(np.clip(a[2], -d_cap, d_cap))
    return a


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
            np.array([-8.0, -8.0, -5.0, -5.0, -5.0, -1.2, -1.0, -1.0], dtype=float),
            np.array([8.0, 8.0, 5.0, 5.0, -0.2, 1.2, 1.0, 1.0], dtype=float),
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
        return np.array([4.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=float)
    raise ValueError("dynamics 应为 'double_integrator' 或 'iris_linear'")
