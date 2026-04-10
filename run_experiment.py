import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from simulation.config import default_config
from simulation.estimators import run_ekf, run_erts, run_ukf, run_urts
from simulation.plotting import plot_geometry, plot_rmse, plot_single_run
from simulation.scenario import simulate_trial


labels = {
    "ekf": "EKF",
    "ukf": "UKF",
    "erts": "EKF + extended RTS smoother",
    "urts": "UKF + unscented RTS smoother",
}


def run_all(y, cfg):
    est = {}
    times = {}

    t0 = time.perf_counter()
    ekf = run_ekf(y, cfg)
    times["ekf"] = time.perf_counter() - t0
    est["ekf"] = {"x": ekf["filt_mean"], "pos": ekf["pos"]}

    t0 = time.perf_counter()
    ukf = run_ukf(y, cfg)
    times["ukf"] = time.perf_counter() - t0
    est["ukf"] = {"x": ukf["filt_mean"], "pos": ukf["pos"]}

    t0 = time.perf_counter()
    erts = run_erts(ekf, cfg)
    times["erts"] = time.perf_counter() - t0
    est["erts"] = {"x": erts["smooth_mean"], "pos": erts["pos"]}

    t0 = time.perf_counter()
    urts = run_urts(ukf, cfg)
    times["urts"] = time.perf_counter() - t0
    est["urts"] = {"x": urts["smooth_mean"], "pos": urts["pos"]}

    return est, times


def pos_err(x_true, x_est):
    return np.linalg.norm(x_est[:, :2] - x_true[:, :2], axis=1)


def main():
    cfg = default_config()
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.single_seed)
    trial_single = simulate_trial(cfg, rng)
    est_single, t_single = run_all(trial_single["y"], cfg)

    plot_geometry(trial_single["x_true"], cfg, cfg.fig_dir / "simulation_geometry.png")
    plot_single_run(trial_single["x_true"], est_single, cfg, cfg.fig_dir / "single_run_trajectory.png")

    n = cfg.n_steps
    rmse_sum = {k: np.zeros(n, dtype=float) for k in labels}
    se_sum = {k: 0.0 for k in labels}
    se_last = {k: 0.0 for k in labels}
    t_sum = {k: 0.0 for k in labels}

    for i in range(cfg.mc_runs):
        rng = np.random.default_rng(cfg.mc_seed0 + i)
        trial = simulate_trial(cfg, rng)
        est, run_t = run_all(trial["y"], cfg)

        for key in labels:
            err = pos_err(trial["x_true"], est[key]["x"])
            se = err**2
            rmse_sum[key] += se
            se_sum[key] += np.sum(se)
            se_last[key] += se[-1]
            t_sum[key] += run_t[key]

    rmse_t = {}
    mean_rmse = {}
    last_rmse = {}
    mean_t_ms = {}

    for key in labels:
        rmse_t[key] = np.sqrt(rmse_sum[key] / cfg.mc_runs)
        mean_rmse[key] = np.sqrt(se_sum[key] / (cfg.mc_runs * n))
        last_rmse[key] = np.sqrt(se_last[key] / cfg.mc_runs)
        mean_t_ms[key] = 1000.0 * t_sum[key] / cfg.mc_runs

    plot_rmse(trial_single["t"], rmse_t, cfg.fig_dir / "mc_rmse_vs_time.png")

    print("Simulation finished")
    print("")
    print("single run mean position error")
    for key in labels:
        err = pos_err(trial_single["x_true"], est_single[key]["x"])
        print(labels[key], round(float(np.mean(err)), 2), "m")

    print("")
    print("monte carlo summary")
    for key in labels:
        print(labels[key])
        print("  mean rmse:", round(float(mean_rmse[key]), 2), "m")
        print("  final-step rmse:", round(float(last_rmse[key]), 2), "m")
        print("  mean runtime:", round(float(mean_t_ms[key]), 2), "ms/run")

    print("")
    print("figures saved to", cfg.fig_dir)
    print("single-run seed:", cfg.single_seed)
    print("monte carlo runs:", cfg.mc_runs)
    print("single-run runtimes in ms")
    for key in labels:
        print(" ", labels[key], round(1000.0 * t_single[key], 2))


if __name__ == "__main__":
    main()
