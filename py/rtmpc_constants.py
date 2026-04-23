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
        return np.array([0.003 * 1.146397572585021, 0.003, 0.006, 0.006, 0.008, 0.012, 0.002, 0.002], dtype=float)
    raise ValueError("dynamics 应为 'double_integrator' 或 'iris_linear'")


def force_bound_to_state_w_half(dynamics: str, dt: float, force_bound_mg: float) -> np.ndarray:
    """将外力球约束 ||f_ext||<=c*m*g 映射为离散状态扰动盒半宽。

    说明：论文形式是球约束（L2）。为兼容当前基于盒集的 RTMPC 实现，这里采用
    球内接盒（inscribed box）：每轴加速度半宽取 amax/sqrt(3)。
    这样得到的状态盒满足“盒内任一点都对应不超过球半径”的范数约束。
    """
    if force_bound_mg < 0.0:
        raise ValueError("force_bound_mg 必须 >= 0")

    amax = float(force_bound_mg) * 9.80665
    a_axis = amax / np.sqrt(3.0)
    pos_half = 0.5 * float(dt) * float(dt) * a_axis
    vel_half = float(dt) * a_axis

    if dynamics == "double_integrator":
        # x=[px,py,vx,vy]
        return np.array([pos_half, pos_half, vel_half, vel_half], dtype=float)

    if dynamics == "iris_linear":
        # x=[pn,pe,vn,ve,pd,vd,phi,theta]
        # 外力扰动主导平动维，姿态维默认由基准盒覆盖。
        return np.array([pos_half, pos_half, vel_half, vel_half, pos_half, vel_half, 0.0, 0.0], dtype=float)

    raise ValueError("dynamics 应为 'double_integrator' 或 'iris_linear'")


def disturbance_half_bounds(
    dynamics: str,
    dt: float = 0.1,
    mode: str = "state_box",
    force_bound_mg: float = 0.35,
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
        return force_bound_to_state_w_half(dynamics, dt=dt, force_bound_mg=force_bound_mg)
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
            np.array([-5.0, -5.0, -6.0, -6.0, -5.0, -3.0, -0.6, -0.6], dtype=float),
            np.array([5.0, 5.0, 6.0, 6.0, -0.2, 3.0, 0.6, 0.6], dtype=float),
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
        u_min = np.array([-0.7 * t_hover, -0.7, -0.7], dtype=float)
        u_max = np.array([max(0.1, t_max - t_hover), 0.7, 0.7], dtype=float)
        return u_min, u_max

    raise ValueError("dynamics 应为 'double_integrator' 或 'iris_linear'")


def base_initial_state(dynamics: str) -> np.ndarray:
    """返回默认初始状态 x0。"""
    if dynamics == "double_integrator":
        return np.array([1.0, 0.5, 0.0, 0.0], dtype=float)
    if dynamics == "iris_linear":
        # 默认初始高度约 1m（NED: pd=-1）
        return np.array([1.0, 0.5, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=float)
    raise ValueError("dynamics 应为 'double_integrator' 或 'iris_linear'")
