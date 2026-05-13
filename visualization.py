"""
可视化工具函数
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from propagator import PropagationResult
from potentials import get_potential
from config import SimParams


def set_style():
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 100,
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 11,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_potential_and_wavepacket(p: SimParams, potential_name: str, psi=None):
    x = p.x
    V_real = np.real(get_potential(x, p, potential_name, with_cap=False))
    fig, ax = plt.subplots()
    ax.plot(x, V_real, "k-", linewidth=2, label=f"V(x) — {potential_name}")
    ax.axhline(p.E_kinetic, color="r", linestyle="--", label=f"E_kin = {p.E_kinetic:.2f}")
    if psi is not None:
        rho = np.abs(psi) ** 2
        ax.fill_between(x, 0, rho * p.V0 / rho.max() * 0.8, alpha=0.4, label="|ψ|²")
    ax.set_xlabel("x")
    ax.set_ylabel("V(x) / |ψ|²")
    ax.set_title(f"势垒与初始波包 — {potential_name}")
    ax.legend()
    ax.set_xlim(p.x_min + p.cap_width, p.x_max - p.cap_width)
    plt.tight_layout()
    return fig


def animate_evolution(result: PropagationResult, x_range=None, interval=30):
    p = result.params
    x = p.x
    V_real = np.real(get_potential(x, p, result.potential_name, with_cap=False))

    if x_range is None:
        x_range = (p.x_min + p.cap_width + 2, p.x_max - p.cap_width - 2)

    mask = (x >= x_range[0]) & (x <= x_range[1])
    x_plot = x[mask]
    V_plot = V_real[mask]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    rho0 = result.prob_densities[0][mask]
    rho_max = max(rho_max_i.max() for rho_max_i in result.prob_densities) * 1.1

    line, = ax1.plot([], [], "b-", linewidth=1.2, label="|ψ(x,t)|²")
    fill = ax1.fill_between(x_plot, 0, rho0, alpha=0.3, color="blue")
    potential_line, = ax1.plot(x_plot, V_plot / max(V_plot.max(), 1) * rho_max * 0.5, "k-", linewidth=2, label="V(x) (scaled)")

    ax1.axhline(p.E_kinetic / max(V_plot.max(), 1) * rho_max * 0.5, color="r", linestyle="--", alpha=0.6, label=f"E = {p.E_kinetic:.2f}")
    ax1.set_xlim(x_range)
    ax1.set_ylim(0, rho_max)
    ax1.set_xlabel("x")
    ax1.set_ylabel("|ψ|²")
    ax1.legend(loc="upper right")
    title = ax1.set_title("")

    ax2.plot(result.times, result.T_values, "g-", label="T(t)")
    ax2.plot(result.times, result.R_values, "r-", label="R(t)")
    ax2.plot(result.times, result.T_values + result.R_values, "k--", label="T+R", alpha=0.5)
    ax2.set_xlabel("t")
    ax2.set_ylabel("Probability")
    ax2.legend()
    marker_T, = ax2.plot([], [], "go", markersize=8)
    marker_R, = ax2.plot([], [], "ro", markersize=8)

    def init():
        line.set_data([], [])
        title.set_text("")
        marker_T.set_data([], [])
        marker_R.set_data([], [])
        return line, title, marker_T, marker_R

    def update(frame):
        nonlocal fill
        rho = result.prob_densities[frame][mask]
        line.set_data(x_plot, rho)

        fill.remove()
        fill = ax1.fill_between(x_plot, 0, rho, alpha=0.3, color="blue")

        t = result.times[frame]
        title.set_text(f"{result.potential_name} — t = {t:.3f},  T = {result.T_values[frame]:.4f},  R = {result.R_values[frame]:.4f}")

        marker_T.set_data([t], [result.T_values[frame]])
        marker_R.set_data([t], [result.R_values[frame]])

        return line, fill, title, marker_T, marker_R

    anim = animation.FuncAnimation(
        fig, update, frames=len(result.prob_densities),
        init_func=init, interval=interval, blit=False
    )
    plt.tight_layout()
    return fig, anim


def plot_TR_evolution(result: PropagationResult):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(result.times, result.T_values, "g-", linewidth=2, label="T(t)")
    ax1.plot(result.times, result.R_values, "r-", linewidth=2, label="R(t)")
    ax1.plot(result.times, result.T_values + result.R_values, "k--", alpha=0.5, label="T+R")
    ax1.set_xlabel("t")
    ax1.set_ylabel("Probability")
    ax1.set_title(f"透射率/反射率演化 — {result.potential_name}")
    ax1.legend()

    ax2.plot(result.times, result.norm_values, "b-", linewidth=2)
    ax2.set_xlabel("t")
    ax2.set_ylabel("∫|ψ|²dx")
    ax2.set_title("概率守恒检验")
    ax2.axhline(1.0, color="k", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_momentum_spectrum(result: PropagationResult, k_range=(-15, 15)):
    p = result.params
    k_sorted = p.k_sorted

    mask = (k_sorted >= k_range[0]) & (k_sorted <= k_range[1])
    k_plot = k_sorted[mask]

    n_frames = len(result.momentum_spectra)
    indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, idx in enumerate(indices):
        spec = result.momentum_spectra[idx][mask]
        spec_norm = spec / spec.max() if spec.max() > 0 else spec
        ax.plot(k_plot, spec_norm, label=f"t = {result.times[idx]:.2f}")

    ax.axvline(p.k0, color="k", linestyle="--", alpha=0.5, label=f"k₀ = {p.k0}")
    ax.axvline(-p.k0, color="k", linestyle=":", alpha=0.5, label=f"-k₀ = {-p.k0}")
    ax.set_xlabel("k")
    ax.set_ylabel("|ψ̃(k)|² (normalized)")
    ax.set_title(f"动量谱演化 — {result.potential_name}")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_snapshots(result: PropagationResult, n_snapshots=6, x_range=None):
    p = result.params
    x = p.x
    V_real = np.real(get_potential(x, p, result.potential_name, with_cap=False))

    if x_range is None:
        x_range = (p.x_min + p.cap_width + 2, p.x_max - p.cap_width - 2)

    mask = (x >= x_range[0]) & (x <= x_range[1])
    x_plot = x[mask]
    V_plot = V_real[mask]

    n_frames = len(result.prob_densities)
    indices = np.linspace(0, n_frames - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(n_snapshots, 1, figsize=(12, 2.5 * n_snapshots), sharex=True)
    if n_snapshots == 1:
        axes = [axes]

    rho_max = max(result.prob_densities[i][mask].max() for i in indices) * 1.1

    for ax, idx in zip(axes, indices):
        rho = result.prob_densities[idx][mask]
        ax.fill_between(x_plot, 0, rho, alpha=0.5, color="blue")
        ax.plot(x_plot, rho, "b-", linewidth=1)
        ax.plot(x_plot, V_plot / max(V_plot.max(), 1) * rho_max * 0.5, "k-", linewidth=1.5)
        ax.axhline(p.E_kinetic / max(V_plot.max(), 1) * rho_max * 0.5, color="r", linestyle="--", alpha=0.4)
        ax.set_ylabel("|ψ|²")
        ax.set_ylim(0, rho_max)
        ax.set_title(f"t = {result.times[idx]:.3f},  T = {result.T_values[idx]:.4f},  R = {result.R_values[idx]:.4f}")

    axes[-1].set_xlabel("x")
    plt.tight_layout()
    return fig


def plot_eckart_comparison(result: PropagationResult):
    from observables import eckart_analytical_T_simple

    p = result.params
    E_range = np.linspace(0.1, 2.0 * p.V0, 500)
    T_analytical = eckart_analytical_T_simple(E_range, p.V0, p.alpha)

    T_numerical = result.T_values[-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(E_range, T_analytical, "k-", linewidth=2, label="Analytical")
    ax.axvline(p.E_kinetic, color="r", linestyle="--", label=f"E_kin = {p.E_kinetic:.2f}")
    ax.plot(p.E_kinetic, T_numerical, "ro", markersize=10, label=f"Numerical T = {T_numerical:.4f}")
    ax.set_xlabel("E")
    ax.set_ylabel("T(E)")
    ax.set_title("Eckart 势透射系数 — 解析 vs 数值")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_three_barriers_comparison(results: list):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for res in results:
        name = res.potential_name
        axes[0, 0].plot(res.times, res.T_values, linewidth=2, label=name)
        axes[0, 1].plot(res.times, res.R_values, linewidth=2, label=name)
        axes[1, 0].plot(res.times, res.T_values + res.R_values, linewidth=2, label=name)
        axes[1, 1].plot(res.times, res.norm_values, linewidth=2, label=name)

    axes[0, 0].set_title("透射率 T(t)")
    axes[0, 0].set_ylabel("T")
    axes[0, 0].legend()

    axes[0, 1].set_title("反射率 R(t)")
    axes[0, 1].set_ylabel("R")
    axes[0, 1].legend()

    axes[1, 0].set_title("T + R")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel("T + R")
    axes[1, 0].legend()

    axes[1, 1].set_title("概率守恒 ∫|ψ|²dx")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel("Norm")
    axes[1, 1].axhline(1.0, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].legend()

    plt.tight_layout()
    return fig
