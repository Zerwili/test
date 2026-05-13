"""
物理观测量计算
自然单位: ℏ = m = 1
"""

import numpy as np
from config import SimParams


def compute_observables(
    psi: np.ndarray,
    x: np.ndarray,
    k: np.ndarray,
    p: SimParams,
    potential_name: str = "square",
) -> dict:
    rho = np.abs(psi) ** 2
    norm = np.trapezoid(rho, x)

    if potential_name == "double":
        barrier_right = p.b2_b
        barrier_left = p.b1_a
    else:
        barrier_right = p.barrier_b
        barrier_left = p.barrier_a

    T = np.trapezoid(rho[x > barrier_right + 2.0], x[x > barrier_right + 2.0])
    R = np.trapezoid(rho[x < barrier_left - 2.0], x[x < barrier_left - 2.0])

    return {"T": T, "R": R, "norm": norm}


def eckart_analytical_T(E: np.ndarray, V0: float, alpha: float) -> np.ndarray:
    """
    Eckart 势解析透射系数
    T = 1 / (1 + V0^2 / (E*(V0-E)) * sinh^2(pi*k/alpha))
    等价形式:
    T = sinh^2(pi*k/alpha) / [sinh^2(pi*k/alpha) + cos^2(pi/2 * sqrt(1 + 8*V0/alpha^2))]
    """
    k = np.sqrt(2.0 * np.maximum(E, 1e-30))
    arg = np.pi * k / alpha
    s = np.pi / 2.0 * np.sqrt(np.maximum(1.0 + 8.0 * V0 / alpha ** 2, 0.0))

    with np.errstate(over="ignore"):
        sinh_val = np.sinh(arg) ** 2
        cos_val = np.cos(s) ** 2

    T = sinh_val / (sinh_val + cos_val)
    return T


def eckart_analytical_T_simple(E: np.ndarray, V0: float, alpha: float) -> np.ndarray:
    """
    Eckart 势透射系数 — 另一等价公式
    T = 1 / (1 + cosh^2(πs) / sinh^2(πk/α))
    其中 s = sqrt(2mV0/ℏ²α² - 1/4) (ℏ=m=1)
    """
    s_sq = 2.0 * V0 / alpha ** 2 - 0.25
    s = np.sqrt(np.maximum(s_sq, 0.0))
    k = np.sqrt(2.0 * np.maximum(E, 1e-30))

    with np.errstate(over="ignore"):
        numerator = np.cosh(np.pi * s) ** 2
        denominator = np.sinh(np.pi * k / alpha) ** 2

    T = 1.0 / (1.0 + numerator / np.maximum(denominator, 1e-30))
    return T


def compute_barrier_region_prob(
    psi: np.ndarray,
    x: np.ndarray,
    p: SimParams,
    potential_name: str = "square",
) -> float:
    if potential_name == "double":
        a, b = p.b1_a, p.b2_b
    elif potential_name == "eckart":
        a, b = -5.0 / p.alpha, 5.0 / p.alpha
    else:
        a, b = p.barrier_a, p.barrier_b
    rho = np.abs(psi) ** 2
    mask = (x >= a) & (x <= b)
    return np.trapezoid(rho[mask], x[mask])


def compute_phase_time(
    psi: np.ndarray,
    x: np.ndarray,
    k: np.ndarray,
    p: SimParams,
    potential_name: str = "square",
) -> float:
    if potential_name == "double":
        barrier_right = p.b2_b
    else:
        barrier_right = p.barrier_b
    transmitted = psi.copy()
    transmitted[x <= barrier_right + 1.0] = 0.0
    t_k = np.fft.fft(transmitted) * (x[1] - x[0]) / np.sqrt(2.0 * np.pi)
    idx_center = np.argmin(np.abs(k - p.k0))
    k_window = 10
    idx_lo = max(idx_center - k_window, 0)
    idx_hi = min(idx_center + k_window, len(k))
    weight = np.abs(t_k[idx_lo:idx_hi]) ** 2
    if weight.sum() < 1e-30:
        return 0.0
    phase = np.angle(t_k[idx_lo:idx_hi])
    dE_dk = k[idx_lo:idx_hi]
    phase_time = np.sum(weight * (-np.gradient(phase, k[idx_lo:idx_hi]))) / weight.sum()
    return phase_time


def square_analytical_phase_time(E: float, V0: float, L: float) -> float:
    if E <= 0:
        return 0.0
    k = np.sqrt(2.0 * E)
    if E < V0:
        kappa = np.sqrt(2.0 * (V0 - E))
        denom = 4.0 * E * (V0 - E) + V0 ** 2 * np.sinh(kappa * L) ** 2
        if denom < 1e-30:
            return 0.0
        dphi_dE = (V0 * L * np.sinh(2.0 * kappa * L) / (2.0 * kappa)
                   + V0 * np.cosh(kappa * L) ** 2 * (1.0 / (kappa * E) - 1.0 / (kappa * (V0 - E)))) / denom
        return dphi_dE
    else:
        kp = np.sqrt(2.0 * (E - V0))
        denom = 4.0 * E * (E - V0) + V0 ** 2 * np.sin(kp * L) ** 2
        if denom < 1e-30:
            return 0.0
        dphi_dE = (V0 * L * np.sin(2.0 * kp * L) / (2.0 * kp)) / denom
        return dphi_dE


def compute_momentum_spectrum(psi: np.ndarray, x: np.ndarray, k_sorted: np.ndarray) -> np.ndarray:
    psi_k = np.fft.fftshift(np.fft.fft(psi)) * (x[1] - x[0]) / np.sqrt(2.0 * np.pi)
    return np.abs(psi_k) ** 2
