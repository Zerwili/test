"""
量子势垒隧穿模拟 — 全局参数配置
自然单位: ℏ = m = 1
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SimParams:
    # --- 空间网格 ---
    x_min: float = -150.0
    x_max: float = 150.0
    N_x: int = 4096

    # --- 时间演化 ---
    dt: float = 0.005
    N_t: int = 4000
    save_every: int = 20

    # --- 高斯波包 ---
    x0: float = -20.0
    sigma: float = 3.0
    k0: float = 5.0

    # --- 势垒公共参数 ---
    V0: float = 10.0

    # --- 方势垒 ---
    barrier_a: float = -1.0
    barrier_b: float = 1.0

    # --- Eckart势 ---
    alpha: float = 1.0
    x_c: float = 0.0

    # --- 双势垒 ---
    b1_a: float = -3.0
    b1_b: float = -1.0
    b2_a: float = 1.0
    b2_b: float = 3.0

    # --- 吸收边界 (CAP) ---
    cap_width: float = 15.0
    cap_strength: float = 0.05
    cap_power: int = 3

    @property
    def dx(self):
        return (self.x_max - self.x_min) / (self.N_x - 1)

    @property
    def x(self):
        return np.linspace(self.x_min, self.x_max, self.N_x)

    @property
    def k(self):
        return 2.0 * np.pi * np.fft.fftfreq(self.N_x, d=self.dx)

    @property
    def k_sorted(self):
        return 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(self.N_x, d=self.dx))

    @property
    def E_kinetic(self):
        return 0.5 * self.k0 ** 2

    @property
    def t_total(self):
        return self.N_t * self.dt

    @property
    def barrier_center(self):
        return 0.5 * (self.barrier_a + self.barrier_b)


def tunneling():
    return SimParams(
        V0=15.0,
        k0=5.0,
        x0=-30.0,
        sigma=3.0,
        dt=0.005,
        N_t=6000,
    )


def above_barrier():
    return SimParams(
        V0=5.0,
        k0=5.0,
        x0=-30.0,
        sigma=3.0,
        dt=0.005,
        N_t=6000,
    )


def resonant():
    return SimParams(
        V0=10.0,
        k0=4.0,
        x0=-30.0,
        sigma=5.0,
        b1_a=-3.0,
        b1_b=-1.0,
        b2_a=1.0,
        b2_b=3.0,
        dt=0.005,
        N_t=8000,
    )
