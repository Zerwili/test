"""
Split-operator FFT 时间传播器
自然单位: ℏ = m = 1

核心: 二阶 Trotter-Suzuki 分解
  exp(-iHΔt) ≈ exp(-iVΔt/2) · exp(-iTΔt) · exp(-iVΔt/2)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from config import SimParams
from wavepacket import gaussian_wavepacket
from potentials import get_potential
from observables import compute_observables, compute_barrier_region_prob


@dataclass
class PropagationResult:
    times: np.ndarray
    prob_densities: List[np.ndarray]
    T_values: np.ndarray
    R_values: np.ndarray
    norm_values: np.ndarray
    momentum_spectra: List[np.ndarray]
    params: SimParams = field(repr=False)
    potential_name: str = ""
    dwell_time: float = 0.0
    barrier_prob_t: np.ndarray = field(default_factory=lambda: np.array([]))


def split_operator_step(
    psi: np.ndarray,
    exp_V_half: np.ndarray,
    exp_T_full: np.ndarray,
) -> np.ndarray:
    psi = psi * exp_V_half
    psi_k = np.fft.fft(psi)
    psi_k = psi_k * exp_T_full
    psi = np.fft.ifft(psi_k)
    psi = psi * exp_V_half
    return psi


def propagate(p: SimParams, potential_name: str) -> PropagationResult:
    x = p.x
    k = p.k
    V = get_potential(x, p, potential_name, with_cap=True)

    psi = gaussian_wavepacket(x, p)

    exp_V_half = np.exp(-1j * V * p.dt / 2.0)
    exp_T_full = np.exp(-1j * k ** 2 / 2.0 * p.dt)

    n_saves = p.N_t // p.save_every + 1
    times = np.zeros(n_saves)
    T_values = np.zeros(n_saves)
    R_values = np.zeros(n_saves)
    norm_values = np.zeros(n_saves)
    barrier_prob_t = np.zeros(n_saves)
    prob_densities = []
    momentum_spectra = []

    dwell_time_accum = 0.0
    barrier_prob_prev = compute_barrier_region_prob(psi, x, p, potential_name)

    save_idx = 0

    obs = compute_observables(psi, x, k, p, potential_name)
    times[0] = 0.0
    T_values[0] = obs["T"]
    R_values[0] = obs["R"]
    norm_values[0] = obs["norm"]
    barrier_prob_t[0] = barrier_prob_prev
    prob_densities.append(np.abs(psi) ** 2)
    momentum_spectra.append(np.abs(np.fft.fftshift(np.fft.fft(psi))) ** 2)
    save_idx = 1

    for step in range(1, p.N_t + 1):
        psi = split_operator_step(psi, exp_V_half, exp_T_full)

        barrier_prob_curr = compute_barrier_region_prob(psi, x, p, potential_name)
        dwell_time_accum += 0.5 * (barrier_prob_prev + barrier_prob_curr) * p.dt
        barrier_prob_prev = barrier_prob_curr

        if step % p.save_every == 0:
            obs = compute_observables(psi, x, k, p, potential_name)
            times[save_idx] = step * p.dt
            T_values[save_idx] = obs["T"]
            R_values[save_idx] = obs["R"]
            norm_values[save_idx] = obs["norm"]
            barrier_prob_t[save_idx] = barrier_prob_curr
            prob_densities.append(np.abs(psi) ** 2)
            momentum_spectra.append(np.abs(np.fft.fftshift(np.fft.fft(psi))) ** 2)
            save_idx += 1

    return PropagationResult(
        times=times[:save_idx],
        prob_densities=prob_densities,
        T_values=T_values[:save_idx],
        R_values=R_values[:save_idx],
        norm_values=norm_values[:save_idx],
        momentum_spectra=momentum_spectra,
        dwell_time=dwell_time_accum,
        barrier_prob_t=barrier_prob_t[:save_idx],
        params=p,
        potential_name=potential_name,
    )
