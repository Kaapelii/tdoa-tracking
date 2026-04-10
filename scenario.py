import numpy as np


def get_F(cfg):
    dt = cfg.dt
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def get_Q(cfg):
    dt = cfg.dt
    qc = cfg.qc
    return qc * np.array(
        [
            [dt**3 / 3.0, 0.0, dt**2 / 2.0, 0.0],
            [0.0, dt**3 / 3.0, 0.0, dt**2 / 2.0],
            [dt**2 / 2.0, 0.0, dt, 0.0],
            [0.0, dt**2 / 2.0, 0.0, dt],
        ],
        dtype=float,
    )


def get_R(cfg):
    m = cfg.rx_pos.shape[0] - 1
    return (cfg.meas_std**2) * np.eye(m, dtype=float)


def get_B(cfg):
    dt = cfg.dt
    return np.array(
        [
            [0.5 * dt**2, 0.0],
            [0.0, 0.5 * dt**2],
            [dt, 0.0],
            [0.0, dt],
        ],
        dtype=float,
    )


def truth_acc(k):
    if k < 25:
        return np.array([0.03, 0.08], dtype=float)
    if k < 50:
        return np.array([0.00, 0.10], dtype=float)
    return np.array([-0.10, 0.02], dtype=float)


def tdoa_meas(x, cfg):
    p = x[:2]
    rx = cfg.rx_pos
    r0 = rx[0]
    d0 = np.linalg.norm(p - r0)

    y = []
    for ri in rx[1:]:
        di = np.linalg.norm(p - ri)
        y.append((di - d0) / cfg.c)

    return np.array(y, dtype=float)


def tdoa_jac(x, cfg):
    p = x[:2]
    rx = cfg.rx_pos
    r0 = rx[0]
    d0 = np.linalg.norm(p - r0)

    H = []
    for ri in rx[1:]:
        di = np.linalg.norm(p - ri)
        gx = (p[0] - ri[0]) / di - (p[0] - r0[0]) / d0
        gy = (p[1] - ri[1]) / di - (p[1] - r0[1]) / d0
        H.append([gx / cfg.c, gy / cfg.c, 0.0, 0.0])

    return np.array(H, dtype=float)


def simulate_trial(cfg, rng):
    """Simulate one truth trajectory and its TDOA measurements."""

    n = cfg.n_steps
    nx = len(cfg.x0_true)
    ny = cfg.rx_pos.shape[0] - 1

    F = get_F(cfg)
    B = get_B(cfg)
    Q = get_Q(cfg)
    R = get_R(cfg)

    x_true = np.zeros((n, nx), dtype=float)
    y = np.zeros((n, ny), dtype=float)

    x_true[0] = cfg.x0_true.copy()
    y[0] = tdoa_meas(x_true[0], cfg) + cfg.meas_std * rng.normal(size=ny)

    q0 = np.zeros(nx, dtype=float)
    for k in range(1, n):
        a = truth_acc(k)
        q = rng.multivariate_normal(q0, Q)
        x_true[k] = F @ x_true[k - 1] + B @ a + q
        y[k] = tdoa_meas(x_true[k], cfg) + cfg.meas_std * rng.normal(size=ny)

    t = np.arange(n, dtype=float) * cfg.dt

    return {
        "t": t,
        "x_true": x_true,
        "y": y,
        "F": F,
        "Q": Q,
        "R": R,
    }
