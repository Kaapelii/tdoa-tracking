from pathlib import Path

import numpy as np


class Config:
    def __init__(self):
        root = Path(__file__).resolve().parents[1]

        self.rx_pos = np.array(
            [
                [0.0, 0.0],
                [2000.0, 0.0],
                [0.0, 2000.0],
                [2000.0, 2000.0],
            ],
            dtype=float,
        )

        self.c = 3.0e8
        self.dt = 1.0
        self.n_steps = 80
        self.qc = 0.8
        self.meas_std = 5.0e-8

        self.x0_true = np.array([350.0, 300.0, 18.0, 11.0], dtype=float)
        self.m0 = np.array([450.0, 380.0, 14.0, 8.0], dtype=float)
        self.P0 = np.diag([150.0**2, 150.0**2, 8.0**2, 8.0**2]).astype(float)

        self.alpha = 0.5
        self.beta = 2.0
        self.kappa = 0.0

        self.single_seed = 20260410
        self.mc_seed0 = 20260410
        self.mc_runs = 100

        self.fig_dir = root / "figures"


def default_config():
    return Config()
