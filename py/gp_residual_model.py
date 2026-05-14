import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import optimize
from scipy import linalg


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(1, -1)
    if x.ndim == 2:
        return x
    raise ValueError("input must be 1D or 2D array")


def _safe_std(x: np.ndarray, axis: int = 0, eps: float = 1e-9) -> np.ndarray:
    s = np.std(x, axis=axis)
    s = np.asarray(s, dtype=float)
    s[s < eps] = 1.0
    return s


def _rbf_kernel(Xa: np.ndarray, Xb: np.ndarray, log_sigma_f: float, log_lengthscales: np.ndarray) -> np.ndarray:
    sigma_f2 = float(np.exp(2.0 * log_sigma_f))
    ls = np.exp(np.asarray(log_lengthscales, dtype=float))
    Xa_s = Xa / ls.reshape(1, -1)
    Xb_s = Xb / ls.reshape(1, -1)
    xa2 = np.sum(Xa_s * Xa_s, axis=1).reshape(-1, 1)
    xb2 = np.sum(Xb_s * Xb_s, axis=1).reshape(1, -1)
    sqdist = np.maximum(xa2 + xb2 - 2.0 * (Xa_s @ Xb_s.T), 0.0)
    return sigma_f2 * np.exp(-0.5 * sqdist)


@dataclass
class RbfGp1D:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    log_sigma_f: float
    log_sigma_n: float
    log_lengthscales: np.ndarray
    x_train_n: np.ndarray
    alpha: np.ndarray
    chol_l: np.ndarray
    jitter: float = 1e-8

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        optimize_hyperparams: bool = True,
        max_points: int = 2000,
        random_seed: int = 0,
        jitter: float = 1e-8,
    ) -> "RbfGp1D":
        X = _ensure_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X/y sample size mismatch")
        if X.shape[0] < 4:
            raise ValueError("need at least 4 samples to fit GP")

        if X.shape[0] > max_points:
            rng = np.random.default_rng(random_seed)
            sel = rng.choice(X.shape[0], size=max_points, replace=False)
            X = X[sel]
            y = y[sel]

        x_mean = np.mean(X, axis=0)
        x_std = _safe_std(X, axis=0)
        Xn = (X - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)

        y_mean = float(np.mean(y))
        y_std = float(np.std(y))
        if y_std < 1e-9:
            y_std = 1.0
        yn = (y - y_mean) / y_std

        d = Xn.shape[1]
        init = np.zeros(2 + d, dtype=float)
        init[0] = 0.0
        init[1] = -2.0
        bounds = [(-6.0, 6.0), (-10.0, 2.0)] + [(-4.0, 4.0)] * d

        def _build(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            log_sigma_f = float(theta[0])
            log_sigma_n = float(theta[1])
            log_ls = np.asarray(theta[2:], dtype=float)
            Kn = _rbf_kernel(Xn, Xn, log_sigma_f, log_ls)
            sigma_n2 = float(np.exp(2.0 * log_sigma_n))
            Kn = Kn + (sigma_n2 + jitter) * np.eye(Xn.shape[0], dtype=float)
            L = linalg.cholesky(Kn, lower=True, check_finite=False)
            alpha = linalg.cho_solve((L, True), yn, check_finite=False)
            return L, alpha, Kn

        def _nll(theta: np.ndarray) -> float:
            try:
                L, alpha, _ = _build(theta)
            except Exception:
                return float("inf")
            return float(
                0.5 * yn.dot(alpha)
                + np.sum(np.log(np.diag(L)))
                + 0.5 * Xn.shape[0] * np.log(2.0 * np.pi)
            )

        theta = init.copy()
        if optimize_hyperparams:
            res = optimize.minimize(
                _nll,
                x0=init,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 100},
            )
            if res.success and np.isfinite(res.fun):
                theta = np.asarray(res.x, dtype=float)

        L, alpha, _ = _build(theta)
        return cls(
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            log_sigma_f=float(theta[0]),
            log_sigma_n=float(theta[1]),
            log_lengthscales=np.asarray(theta[2:], dtype=float),
            x_train_n=Xn,
            alpha=np.asarray(alpha, dtype=float),
            chol_l=np.asarray(L, dtype=float),
            jitter=float(jitter),
        )

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = _ensure_2d(X)
        Xn = (X - self.x_mean.reshape(1, -1)) / self.x_std.reshape(1, -1)
        Ks = _rbf_kernel(self.x_train_n, Xn, self.log_sigma_f, self.log_lengthscales)
        mean_n = Ks.T @ self.alpha
        v = linalg.solve_triangular(self.chol_l, Ks, lower=True, check_finite=False)
        sigma_f2 = float(np.exp(2.0 * self.log_sigma_f))
        var_n = np.maximum(sigma_f2 - np.sum(v * v, axis=0), 1e-12)
        mean = mean_n * self.y_std + self.y_mean
        std = np.sqrt(var_n) * self.y_std
        return mean.reshape(-1), std.reshape(-1)

    def export(self, prefix: str, out: Dict[str, np.ndarray]) -> None:
        out[f"{prefix}_x_mean"] = self.x_mean
        out[f"{prefix}_x_std"] = self.x_std
        out[f"{prefix}_y_mean"] = np.array([self.y_mean], dtype=float)
        out[f"{prefix}_y_std"] = np.array([self.y_std], dtype=float)
        out[f"{prefix}_log_sigma_f"] = np.array([self.log_sigma_f], dtype=float)
        out[f"{prefix}_log_sigma_n"] = np.array([self.log_sigma_n], dtype=float)
        out[f"{prefix}_log_lengthscales"] = self.log_lengthscales
        out[f"{prefix}_x_train_n"] = self.x_train_n
        out[f"{prefix}_alpha"] = self.alpha
        out[f"{prefix}_chol_l"] = self.chol_l
        out[f"{prefix}_jitter"] = np.array([self.jitter], dtype=float)

    @classmethod
    def load(cls, prefix: str, src: Dict[str, np.ndarray]) -> "RbfGp1D":
        return cls(
            x_mean=np.asarray(src[f"{prefix}_x_mean"], dtype=float),
            x_std=np.asarray(src[f"{prefix}_x_std"], dtype=float),
            y_mean=float(np.asarray(src[f"{prefix}_y_mean"], dtype=float).reshape(-1)[0]),
            y_std=float(np.asarray(src[f"{prefix}_y_std"], dtype=float).reshape(-1)[0]),
            log_sigma_f=float(np.asarray(src[f"{prefix}_log_sigma_f"], dtype=float).reshape(-1)[0]),
            log_sigma_n=float(np.asarray(src[f"{prefix}_log_sigma_n"], dtype=float).reshape(-1)[0]),
            log_lengthscales=np.asarray(src[f"{prefix}_log_lengthscales"], dtype=float),
            x_train_n=np.asarray(src[f"{prefix}_x_train_n"], dtype=float),
            alpha=np.asarray(src[f"{prefix}_alpha"], dtype=float),
            chol_l=np.asarray(src[f"{prefix}_chol_l"], dtype=float),
            jitter=float(np.asarray(src[f"{prefix}_jitter"], dtype=float).reshape(-1)[0]),
        )


class VelocityResidualGP:
    """GP residual model using inertial-frame velocity features.

    Current training mode (axis-wise, Torrente-style):
    - each output axis uses only its own velocity component as GP input.
      e.g. iris_linear: a_n <- v_n, a_e <- v_e, a_d <- v_d.

    Backward compatibility:
    - previously trained models that used full velocity vector input for each
      axis are still supported at inference time.
    """

    def __init__(self, dynamics: str, dt: float, state_dim: int, axis_models: Sequence[RbfGp1D]):
        self.dynamics = str(dynamics)
        self.dt = float(dt)
        self.state_dim = int(state_dim)
        self.axis_models = list(axis_models)
        self.pos_idx, self.vel_idx = self._index_map(self.dynamics, self.state_dim)
        if len(self.axis_models) != len(self.vel_idx):
            raise ValueError("axis model count mismatch with velocity dimensions")

    @staticmethod
    def _index_map(dynamics: str, state_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        if dynamics == "double_integrator":
            if state_dim < 4:
                raise ValueError("double_integrator state_dim must be >=4")
            return np.array([0, 1], dtype=int), np.array([2, 3], dtype=int)
        if dynamics == "iris_linear":
            if state_dim < 6:
                raise ValueError("iris_linear state_dim must be >=6")
            return np.array([0, 1, 4], dtype=int), np.array([2, 3, 5], dtype=int)
        raise ValueError("unsupported dynamics for VelocityResidualGP")

    @classmethod
    def fit_from_transitions(
        cls,
        dynamics: str,
        dt: float,
        A: np.ndarray,
        B: np.ndarray,
        x_t: np.ndarray,
        u_t: np.ndarray,
        x_tp1: np.ndarray,
        optimize_hyperparams: bool = True,
        max_points_per_axis: int = 2000,
        random_seed: int = 0,
    ) -> "VelocityResidualGP":
        x_t = _ensure_2d(x_t)
        u_t = _ensure_2d(u_t)
        x_tp1 = _ensure_2d(x_tp1)
        if not (x_t.shape[0] == u_t.shape[0] == x_tp1.shape[0]):
            raise ValueError("transition sample count mismatch")
        if x_t.shape[1] != A.shape[0] or u_t.shape[1] != B.shape[1]:
            raise ValueError("transition dimension mismatch with A/B")

        pos_idx, vel_idx = cls._index_map(dynamics, x_t.shape[1])
        x_nom_next = x_t @ A.T + u_t @ B.T
        dx_res = x_tp1 - x_nom_next
        accel_res = dx_res[:, vel_idx] / float(dt)
        models: List[RbfGp1D] = []
        for j in range(vel_idx.size):
            # Axis-wise feature: each GP only sees its own velocity component.
            feat_j = x_t[:, vel_idx[j] : vel_idx[j] + 1]
            models.append(
                RbfGp1D.fit(
                    feat_j,
                    accel_res[:, j],
                    optimize_hyperparams=optimize_hyperparams,
                    max_points=max_points_per_axis,
                    random_seed=random_seed + j,
                )
            )
        return cls(dynamics=dynamics, dt=dt, state_dim=x_t.shape[1], axis_models=models)

    def predict_accel(self, vel_inertial: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vel_inertial = _ensure_2d(vel_inertial)
        if vel_inertial.shape[1] != len(self.vel_idx):
            raise ValueError("velocity feature dimension mismatch")
        means = []
        stds = []
        for j, gp in enumerate(self.axis_models):
            feat_dim = int(gp.x_mean.size)
            if feat_dim == 1:
                # Axis-wise model: a_j <- v_j
                gp_in = vel_inertial[:, j : j + 1]
            elif feat_dim == vel_inertial.shape[1]:
                # Legacy model: a_j <- [v_1, ..., v_k]
                gp_in = vel_inertial
            else:
                raise ValueError(
                    f"GP axis {j} feature dim mismatch: model expects {feat_dim}, "
                    f"but runtime velocity dim is {vel_inertial.shape[1]}"
                )
            mu, sd = gp.predict(gp_in)
            means.append(mu.reshape(-1, 1))
            stds.append(sd.reshape(-1, 1))
        mean_acc = np.hstack(means)
        std_acc = np.hstack(stds)
        return mean_acc, std_acc

    def predict_state_disturbance(
        self,
        x: np.ndarray,
        beta_sigma: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = _ensure_2d(x)
        vel = x[:, self.vel_idx]
        mean_acc, std_acc = self.predict_accel(vel)

        d_mean = np.zeros((x.shape[0], self.state_dim), dtype=float)
        d_half = np.zeros((x.shape[0], self.state_dim), dtype=float)
        dt = float(self.dt)
        dt2 = 0.5 * dt * dt
        b = float(beta_sigma)
        for j in range(len(self.vel_idx)):
            p = int(self.pos_idx[j])
            v = int(self.vel_idx[j])
            d_mean[:, p] = dt2 * mean_acc[:, j]
            d_mean[:, v] = dt * mean_acc[:, j]
            d_half[:, p] = dt2 * b * std_acc[:, j]
            d_half[:, v] = dt * b * std_acc[:, j]

        if d_mean.shape[0] == 1:
            return d_mean.reshape(-1), d_half.reshape(-1)
        return d_mean, d_half

    def conservative_uncertainty_bound(
        self,
        x_min: np.ndarray,
        x_max: np.ndarray,
        beta_sigma: float = 2.0,
        grid_points_per_dim: int = 9,
    ) -> np.ndarray:
        x_min = np.asarray(x_min, dtype=float).reshape(-1)
        x_max = np.asarray(x_max, dtype=float).reshape(-1)
        if x_min.size != self.state_dim or x_max.size != self.state_dim:
            raise ValueError("x_min/x_max dimension mismatch")
        if np.any(x_max <= x_min):
            raise ValueError("invalid state bounds: require x_max > x_min")

        k = len(self.vel_idx)
        if grid_points_per_dim < 2:
            grid_points_per_dim = 2

        # Axis-wise fast path (new training mode): evaluate each axis GP over
        # its own velocity range only. Fallback to legacy Cartesian grid when
        # encountering old full-feature models.
        axiswise = all(int(gp.x_mean.size) == 1 for gp in self.axis_models)
        if axiswise:
            std_max = np.zeros((k,), dtype=float)
            for j, idx in enumerate(self.vel_idx):
                lo = float(x_min[idx])
                hi = float(x_max[idx])
                vals = np.linspace(lo, hi, grid_points_per_dim).reshape(-1, 1)
                _, sd = self.axis_models[j].predict(vals)
                std_max[j] = float(np.max(sd))
        else:
            axis_vals = []
            for idx in self.vel_idx:
                lo = float(x_min[idx])
                hi = float(x_max[idx])
                axis_vals.append(np.linspace(lo, hi, grid_points_per_dim))
            points = np.array(list(itertools.product(*axis_vals)), dtype=float).reshape(-1, k)
            _, std_acc = self.predict_accel(points)
            std_max = np.max(std_acc, axis=0)

        out = np.zeros((self.state_dim,), dtype=float)
        dt = float(self.dt)
        dt2 = 0.5 * dt * dt
        b = float(beta_sigma)
        for j in range(k):
            p = int(self.pos_idx[j])
            v = int(self.vel_idx[j])
            out[p] = dt2 * b * std_max[j]
            out[v] = dt * b * std_max[j]
        return out

    def conservative_mean_bound(
        self,
        x_min: np.ndarray,
        x_max: np.ndarray,
        grid_points_per_dim: int = 9,
    ) -> np.ndarray:
        """保守评估 |d_mean(x)| 的逐维上界（状态扰动空间）。

        用于“总扰动边界 - 可补偿均值边界 + GP不确定性边界”的残差管构造。
        返回向量含义与 `predict_state_disturbance` 中 d_mean 的状态维一致。
        """
        x_min = np.asarray(x_min, dtype=float).reshape(-1)
        x_max = np.asarray(x_max, dtype=float).reshape(-1)
        if x_min.size != self.state_dim or x_max.size != self.state_dim:
            raise ValueError("x_min/x_max dimension mismatch")
        if np.any(x_max <= x_min):
            raise ValueError("invalid state bounds: require x_max > x_min")

        k = len(self.vel_idx)
        if grid_points_per_dim < 2:
            grid_points_per_dim = 2

        axiswise = all(int(gp.x_mean.size) == 1 for gp in self.axis_models)
        if axiswise:
            mean_abs_max = np.zeros((k,), dtype=float)
            for j, idx in enumerate(self.vel_idx):
                lo = float(x_min[idx])
                hi = float(x_max[idx])
                vals = np.linspace(lo, hi, grid_points_per_dim).reshape(-1, 1)
                mu, _ = self.axis_models[j].predict(vals)
                mean_abs_max[j] = float(np.max(np.abs(mu)))
        else:
            axis_vals = []
            for idx in self.vel_idx:
                lo = float(x_min[idx])
                hi = float(x_max[idx])
                axis_vals.append(np.linspace(lo, hi, grid_points_per_dim))
            points = np.array(list(itertools.product(*axis_vals)), dtype=float).reshape(-1, k)
            mean_acc, _ = self.predict_accel(points)
            mean_abs_max = np.max(np.abs(mean_acc), axis=0)

        out = np.zeros((self.state_dim,), dtype=float)
        dt = float(self.dt)
        dt2 = 0.5 * dt * dt
        for j in range(k):
            p = int(self.pos_idx[j])
            v = int(self.vel_idx[j])
            out[p] = dt2 * mean_abs_max[j]
            out[v] = dt * mean_abs_max[j]
        return out

    def save(self, path: str) -> None:
        payload: Dict[str, np.ndarray] = {
            "dynamics": np.array([self.dynamics]),
            "dt": np.array([self.dt], dtype=float),
            "state_dim": np.array([self.state_dim], dtype=int),
            "axis_count": np.array([len(self.axis_models)], dtype=int),
        }
        for i, gp in enumerate(self.axis_models):
            gp.export(prefix=f"axis{i}", out=payload)
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str) -> "VelocityResidualGP":
        raw = np.load(path, allow_pickle=False)
        payload = {k: raw[k] for k in raw.files}
        dynamics = str(payload["dynamics"].reshape(-1)[0])
        dt = float(payload["dt"].reshape(-1)[0])
        state_dim = int(payload["state_dim"].reshape(-1)[0])
        axis_count = int(payload["axis_count"].reshape(-1)[0])
        models = [RbfGp1D.load(prefix=f"axis{i}", src=payload) for i in range(axis_count)]
        return cls(dynamics=dynamics, dt=dt, state_dim=state_dim, axis_models=models)

def residual_shrink_bounds(
    base_w_half: np.ndarray,
    gp_comp_half: np.ndarray,
    gp_unc_half: np.ndarray,
    mode: str,
) -> np.ndarray:
    """将 GP 补偿统一映射为“残差扰动边界”。

    设总扰动边界为 base_w_half，GP 估计补偿项均值界为 gp_comp_half，
    GP 未建模/预测不确定性界为 gp_unc_half。

    - mode='none': 不做 GP 收缩，直接用 base_w_half
    - mode='residual': 用启发式残差界
        w_res = max(base_w_half - gp_comp_half, 0) + gp_unc_half

    该构造体现：
    1) 可补偿的速度相关部分会收缩扰动管；
    2) 不确定性与随机残差仍保留在最终鲁棒管中。
    """
    base_w_half = np.asarray(base_w_half, dtype=float).reshape(-1)
    gp_comp_half = np.asarray(gp_comp_half, dtype=float).reshape(-1)
    gp_unc_half = np.asarray(gp_unc_half, dtype=float).reshape(-1)
    if not (base_w_half.size == gp_comp_half.size == gp_unc_half.size):
        raise ValueError("bound dimensions mismatch")
    if mode == "none":
        return base_w_half
    if mode == "residual":
        return np.maximum(base_w_half - gp_comp_half, 0.0) + gp_unc_half
    raise ValueError("gp shrink mode must be one of: none|residual")


def _load_transition_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    # preferred keys: x_t, u_t, x_tp1
    if {"x_t", "u_t", "x_tp1"}.issubset(set(data.files)):
        return data["x_t"], data["u_t"], data["x_tp1"]
    # fallback: xs/us where xs has length T+1, us has length T
    if {"xs", "us"}.issubset(set(data.files)):
        xs = np.asarray(data["xs"], dtype=float)
        us = np.asarray(data["us"], dtype=float)
        if xs.shape[0] != us.shape[0] + 1:
            raise ValueError(f"invalid xs/us lengths in {npz_path}")
        return xs[:-1], us, xs[1:]
    raise ValueError(f"{npz_path} missing transition keys; expected (x_t,u_t,x_tp1) or (xs,us)")


def _axis_labels(dynamics: str, axis_count: int) -> List[str]:
    if dynamics == "double_integrator":
        base = ["a_x", "a_y"]
    elif dynamics == "iris_linear":
        base = ["a_n", "a_e", "a_d"]
    else:
        base = []
    if len(base) >= axis_count:
        return base[:axis_count]
    return [f"axis_{i}" for i in range(axis_count)]


def _print_training_report(
    model: "VelocityResidualGP",
    A: np.ndarray,
    B: np.ndarray,
    x_t: np.ndarray,
    u_t: np.ndarray,
    x_tp1: np.ndarray,
    tag: str = "train",
) -> None:
    x_nom_next = x_t @ A.T + u_t @ B.T
    dx_res = x_tp1 - x_nom_next
    accel_res = dx_res[:, model.vel_idx] / float(model.dt)
    feat = x_t[:, model.vel_idx]

    labels = _axis_labels(model.dynamics, len(model.axis_models))
    print("[gp][report] --------------------------------------------------")
    print(f"[gp][report] split={tag}")
    print(
        f"[gp][report] samples_total={x_t.shape[0]}, feature_dim={feat.shape[1]}, "
        f"axis_count={len(model.axis_models)}"
    )
    for j, gp in enumerate(model.axis_models):
        y = accel_res[:, j]
        feat_dim = int(gp.x_mean.size)
        if feat_dim == 1:
            gp_in = feat[:, j : j + 1]
        elif feat_dim == feat.shape[1]:
            gp_in = feat
        else:
            raise ValueError(
                f"GP axis {j} feature dim mismatch in report: model expects {feat_dim}, "
                f"but runtime velocity dim is {feat.shape[1]}"
            )
        mu, sd = gp.predict(gp_in)
        err = mu - y
        rmse = float(np.sqrt(np.mean(err * err)))
        mae = float(np.mean(np.abs(err)))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        ss_res = float(np.sum(err ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

        sigma_f = float(np.exp(gp.log_sigma_f))
        sigma_n = float(np.exp(gp.log_sigma_n))
        ls = np.exp(gp.log_lengthscales)
        ls_s = np.array2string(ls, precision=5, separator=", ")
        used = int(gp.x_train_n.shape[0])

        print(f"[gp][axis {j}] label={labels[j]} used_points={used} input_dim={feat_dim}")
        print(
            f"  sigma_f={sigma_f:.6g}, sigma_n={sigma_n:.6g}, sigma_n/sigma_f="
            f"{(sigma_n / max(sigma_f, 1e-12)):.6g}"
        )
        print(f"  lengthscales={ls_s}")
        print(
            f"  target_mean={float(np.mean(y)):.6g}, target_std={float(np.std(y)):.6g}, "
            f"pred_std_mean={float(np.mean(sd)):.6g}"
        )
        print(f"  fit_rmse={rmse:.6g}, fit_mae={mae:.6g}, fit_r2={r2:.6g}")
    print("[gp][report] --------------------------------------------------")


def cli_fit() -> None:
    parser = argparse.ArgumentParser(description="Fit inertial-velocity GP residual model from transition npz files")
    parser.add_argument("--dynamics", choices=["double_integrator", "iris_linear"], default="iris_linear")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--out",
        type=str,
        default="gp_model/iris_linear_residual_gp.npz",
        help="Output .npz file path (default: gp_model/iris_linear_residual_gp.npz)",
    )
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        default=["gp_data/transitions_all_filt2ms_1000.npz"],
        help="Transition npz files (default: gp_data/transitions_all_filt2ms_1000.npz)",
    )
    parser.add_argument("--max-points", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-opt", action="store_true", help="Disable GP hyperparameter optimization")
    parser.add_argument("--no-report", action="store_true", help="Disable training report printout")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Holdout validation ratio in [0,1). 0 disables split (default: 0).",
    )
    parser.add_argument(
        "--val-seed",
        type=int,
        default=123,
        help="Random seed for holdout split when --val-ratio>0 (default: 123).",
    )
    parser.add_argument(
        "--val-temporal-split",
        action="store_true",
        help="Use last val-ratio chunk as validation instead of random split.",
    )
    args = parser.parse_args()

    from rtmpc_demo import DoubleIntegrator, LinearIrisHover

    if args.dynamics == "double_integrator":
        sim = DoubleIntegrator(dt=float(args.dt))
    else:
        sim = LinearIrisHover(dt=float(args.dt), mass=1.5)
    A, B = sim.A, sim.B

    x_list: List[np.ndarray] = []
    u_list: List[np.ndarray] = []
    xn_list: List[np.ndarray] = []
    for p in args.data:
        x_t, u_t, x_tp1 = _load_transition_npz(Path(p))
        x_list.append(_ensure_2d(x_t))
        u_list.append(_ensure_2d(u_t))
        xn_list.append(_ensure_2d(x_tp1))

    x_t_all = np.vstack(x_list)
    u_t_all = np.vstack(u_list)
    x_tp1_all = np.vstack(xn_list)

    total_n = x_t_all.shape[0]
    val_ratio = float(args.val_ratio)
    if val_ratio < 0.0 or val_ratio >= 1.0:
        raise ValueError("--val-ratio must satisfy 0 <= val_ratio < 1")

    if val_ratio > 0.0:
        n_val = max(1, int(round(total_n * val_ratio)))
        n_val = min(n_val, total_n - 4)
        if n_val <= 0:
            raise ValueError("not enough samples for holdout split; reduce --val-ratio")

        if args.val_temporal_split:
            train_idx = np.arange(0, total_n - n_val, dtype=int)
            val_idx = np.arange(total_n - n_val, total_n, dtype=int)
        else:
            rng = np.random.default_rng(int(args.val_seed))
            perm = rng.permutation(total_n)
            val_idx = np.sort(perm[:n_val])
            train_idx = np.sort(perm[n_val:])

        x_t_train, u_t_train, x_tp1_train = x_t_all[train_idx], u_t_all[train_idx], x_tp1_all[train_idx]
        x_t_val, u_t_val, x_tp1_val = x_t_all[val_idx], u_t_all[val_idx], x_tp1_all[val_idx]
    else:
        x_t_train, u_t_train, x_tp1_train = x_t_all, u_t_all, x_tp1_all
        x_t_val = u_t_val = x_tp1_val = None

    model = VelocityResidualGP.fit_from_transitions(
        dynamics=args.dynamics,
        dt=float(args.dt),
        A=A,
        B=B,
        x_t=x_t_train,
        u_t=u_t_train,
        x_tp1=x_tp1_train,
        optimize_hyperparams=not args.no_opt,
        max_points_per_axis=int(args.max_points),
        random_seed=int(args.seed),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"[gp] saved model: {args.out}")
    print(f"[gp] samples: {x_t_all.shape[0]}")
    print(f"[gp] train_samples: {x_t_train.shape[0]}")
    if x_t_val is not None:
        print(f"[gp] val_samples: {x_t_val.shape[0]}")
        print(
            f"[gp] val_split: ratio={val_ratio:.3f}, mode="
            f"{'temporal' if args.val_temporal_split else 'random'}, seed={int(args.val_seed)}"
        )
    print(f"[gp] dynamics={args.dynamics}, dt={args.dt}")
    if not args.no_report:
        _print_training_report(
            model=model,
            A=A,
            B=B,
            x_t=x_t_train,
            u_t=u_t_train,
            x_tp1=x_tp1_train,
            tag="train",
        )
        if x_t_val is not None:
            _print_training_report(
                model=model,
                A=A,
                B=B,
                x_t=x_t_val,
                u_t=u_t_val,
                x_tp1=x_tp1_val,
                tag="val",
            )


if __name__ == "__main__":
    cli_fit()
