from typing import Tuple

import numpy as np


MODEL_NAME = "iris_linear"
IRIS_MASS_DEFAULT = 1.5
IRIS_KF = 5.84e-06
IRIS_W_MAX = 1100.0


def _require_iris(dynamics: str) -> None:
    if dynamics != MODEL_NAME:
        raise ValueError(
            f"该工作区已精简为仅支持 {MODEL_NAME} 圆轨迹任务，收到 dynamics={dynamics}"
        )


def state_cost_matrix(dynamics: str) -> np.ndarray:
    _require_iris(dynamics)
    # x=[pn,pe,vn,ve,pd,vd,phi,theta] (NED)
    return np.diag([50.0, 50.0, 6.0, 6.0, 50.0, 6.0, 3.0, 3.0])


def input_cost_matrix(dynamics: str, m: int) -> np.ndarray:
    _require_iris(dynamics)
    if m <= 0:
        raise ValueError("m 必须为正")
    if m != 3:
        return np.eye(m) * 0.01
    # u=[dT, phi_cmd, theta_cmd]
    return np.diag([0.01, 0.5, 0.5])


def _base_disturbance_half_bounds(dynamics: str) -> np.ndarray:
    _require_iris(dynamics)
    # x=[pn,pe,vn,ve,pd,vd,phi,theta]
    return np.array([0.003, 0.003, 0.006, 0.006, 0.008, 0.012, 0.002, 0.002], dtype=float)


def force_bound_to_state_w_half(
    dynamics: str,
    dt: float,
    force_bound_mg: float,
    force_d_axis_scale: float = 0.15,
) -> np.ndarray:
    """将 ||f_ext||<=c*m*g 映射为离散状态扰动盒半宽。"""
    _require_iris(dynamics)
    if force_bound_mg < 0.0:
        raise ValueError("force_bound_mg 必须 >= 0")
    if force_d_axis_scale < 0.0:
        raise ValueError("force_d_axis_scale 必须 >= 0")

    amax = float(force_bound_mg) * 9.80665
    a_axis = amax / np.sqrt(3.0)  # 球内接盒
    d_scale = float(min(force_d_axis_scale, 1.0))
    a_axis_d = a_axis * d_scale

    pos_half = 0.5 * float(dt) * float(dt) * a_axis
    vel_half = float(dt) * a_axis
    pos_half_d = 0.5 * float(dt) * float(dt) * a_axis_d
    vel_half_d = float(dt) * a_axis_d

    # x=[pn,pe,vn,ve,pd,vd,phi,theta]
    return np.array(
        [pos_half, pos_half, vel_half, vel_half, pos_half_d, vel_half_d, 0.0, 0.0],
        dtype=float,
    )


def disturbance_half_bounds(
    dynamics: str,
    dt: float = 0.1,
    mode: str = "state_box",
    force_bound_mg: float = 0.05,
    force_d_axis_scale: float = 0.15,
) -> np.ndarray:
    if mode == "state_box":
        return _base_disturbance_half_bounds(dynamics)
    if mode == "force_only":
        return force_bound_to_state_w_half(
            dynamics=dynamics,
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
    _require_iris(dynamics)
    a = np.asarray(accel_ned, dtype=float).reshape(-1)
    if a.size < 3:
        raise ValueError("accel_ned 需要至少 3 维 (n,e,d)")
    if state_dim < 6:
        raise ValueError("iris_linear 需要 state_dim >= 6")

    d = np.zeros(int(state_dim), dtype=float)
    dt = float(dt)
    dt2 = 0.5 * dt * dt
    # x=[pn,pe,vn,ve,pd,vd,phi,theta]
    d[0] = dt2 * a[0]
    d[1] = dt2 * a[1]
    d[2] = dt * a[0]
    d[3] = dt * a[1]
    d[4] = dt2 * a[2]
    d[5] = dt * a[2]
    return d


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
    _require_iris(dynamics)
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
    _require_iris(dynamics)
    # x=[pn,pe,vn,ve,pd,vd,phi,theta] (NED)
    return (
        np.array([-8.0, -8.0, -2.5, -2.5, -5.0, -1.0, -1.0, -1.0], dtype=float),
        np.array([8.0, 8.0, 2.5, 2.5, -0.2, 1.0, 1.0, 1.0], dtype=float),
    )


def gp_query_state_bounds(dynamics: str) -> Tuple[np.ndarray, np.ndarray]:
    _require_iris(dynamics)
    # 与 MPC 硬约束解耦，只用于 GP 管宽保守扫描域
    return (
        np.array([-8.0, -8.0, -2.2, -2.2, -5.0, -0.6, -1.0, -1.0], dtype=float),
        np.array([8.0, 8.0, 2.2, 2.2, -0.2, 0.6, 1.0, 1.0], dtype=float),
    )


def base_input_bounds(dynamics: str, mass: float = IRIS_MASS_DEFAULT, m: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    _require_iris(dynamics)
    t_max = 4.0 * IRIS_KF * (IRIS_W_MAX ** 2)
    t_hover = float(mass) * 9.81
    u_min = np.array([-0.7 * t_hover, -1.0, -1.0], dtype=float)
    u_max = np.array([max(0.1, t_max - t_hover), 1.0, 1.0], dtype=float)
    return u_min, u_max


def base_initial_state(dynamics: str) -> np.ndarray:
    _require_iris(dynamics)
    # 默认初始高度约 1m（NED: pd=-1）
    return np.array([0.0, 4.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=float)

