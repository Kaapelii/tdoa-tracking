import numpy as np

from simulation.scenario import get_F, get_Q, get_R, tdoa_jac, tdoa_meas


def sym(A):
    return 0.5 * (A + A.T)


def ukf_weights(n, alpha, beta, kappa):
    lam = alpha**2 * (n + kappa) - n
    scale = n + lam

    wm = np.full(2 * n + 1, 1.0 / (2.0 * scale), dtype=float)
    wc = np.full(2 * n + 1, 1.0 / (2.0 * scale), dtype=float)

    wm[0] = lam / scale
    wc[0] = lam / scale + (1.0 - alpha**2 + beta)

    return lam, wm, wc


def sigma_pts(m, P, alpha, beta, kappa):
    n = len(m)
    lam, wm, wc = ukf_weights(n, alpha, beta, kappa)
    L = np.linalg.cholesky(P)
    pts = np.zeros((2 * n + 1, n), dtype=float)
    pts[0] = m

    s = np.sqrt(n + lam)
    for i in range(n):
        d = s * L[:, i]
        pts[i + 1] = m + d
        pts[n + i + 1] = m - d

    return pts, wm, wc


def run_ekf(y, cfg):
    F = get_F(cfg)
    Q = get_Q(cfg)
    R = get_R(cfg)

    n = y.shape[0]
    nx = len(cfg.m0)

    pred_mean = np.zeros((n, nx), dtype=float)
    pred_cov = np.zeros((n, nx, nx), dtype=float)
    filt_mean = np.zeros((n, nx), dtype=float)
    filt_cov = np.zeros((n, nx, nx), dtype=float)

    m = cfg.m0.copy()
    P = cfg.P0.copy()

    for k in range(n):
        m_pred = F @ m
        P_pred = sym(F @ P @ F.T + Q)

        H = tdoa_jac(m_pred, cfg)
        y_pred = tdoa_meas(m_pred, cfg)
        v = y[k] - y_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        m = m_pred + K @ v
        P = sym(P_pred - K @ S @ K.T)

        pred_mean[k] = m_pred
        pred_cov[k] = P_pred
        filt_mean[k] = m
        filt_cov[k] = P

    return {
        "pred_mean": pred_mean,
        "pred_cov": pred_cov,
        "filt_mean": filt_mean,
        "filt_cov": filt_cov,
        "pos": filt_mean[:, :2],
    }


def run_ukf(y, cfg):
    F = get_F(cfg)
    Q = get_Q(cfg)
    R = get_R(cfg)

    n = y.shape[0]
    nx = len(cfg.m0)
    ny = y.shape[1]

    pred_mean = np.zeros((n, nx), dtype=float)
    pred_cov = np.zeros((n, nx, nx), dtype=float)
    filt_mean = np.zeros((n, nx), dtype=float)
    filt_cov = np.zeros((n, nx, nx), dtype=float)

    m = cfg.m0.copy()
    P = cfg.P0.copy()

    for k in range(n):
        X, wm, wc = sigma_pts(m, P, cfg.alpha, cfg.beta, cfg.kappa)
        Xp = (F @ X.T).T

        m_pred = np.sum(wm[:, None] * Xp, axis=0)
        P_pred = np.zeros((nx, nx), dtype=float)
        for i in range(Xp.shape[0]):
            dx = Xp[i] - m_pred
            P_pred += wc[i] * np.outer(dx, dx)
        P_pred = sym(P_pred + Q)

        X2, wm, wc = sigma_pts(m_pred, P_pred, cfg.alpha, cfg.beta, cfg.kappa)
        Y = np.array([tdoa_meas(xi, cfg) for xi in X2], dtype=float)
        y_pred = np.sum(wm[:, None] * Y, axis=0)

        S = np.zeros((ny, ny), dtype=float)
        C = np.zeros((nx, ny), dtype=float)
        for i in range(X2.shape[0]):
            dx = X2[i] - m_pred
            dy = Y[i] - y_pred
            S += wc[i] * np.outer(dy, dy)
            C += wc[i] * np.outer(dx, dy)
        S = sym(S + R)

        K = C @ np.linalg.inv(S)
        m = m_pred + K @ (y[k] - y_pred)
        P = sym(P_pred - K @ S @ K.T)

        pred_mean[k] = m_pred
        pred_cov[k] = P_pred
        filt_mean[k] = m
        filt_cov[k] = P

    return {
        "pred_mean": pred_mean,
        "pred_cov": pred_cov,
        "filt_mean": filt_mean,
        "filt_cov": filt_cov,
        "pos": filt_mean[:, :2],
    }


def run_erts(res, cfg):
    F = get_F(cfg)

    m_f = res["filt_mean"]
    P_f = res["filt_cov"]
    m_p = res["pred_mean"]
    P_p = res["pred_cov"]

    m_s = m_f.copy()
    P_s = P_f.copy()

    for k in range(len(m_f) - 2, -1, -1):
        G = P_f[k] @ F.T @ np.linalg.inv(P_p[k + 1])
        m_s[k] = m_f[k] + G @ (m_s[k + 1] - m_p[k + 1])
        P_s[k] = sym(P_f[k] + G @ (P_s[k + 1] - P_p[k + 1]) @ G.T)

    return {
        "smooth_mean": m_s,
        "smooth_cov": P_s,
        "pos": m_s[:, :2],
    }


def run_urts(res, cfg):
    F = get_F(cfg)

    m_f = res["filt_mean"]
    P_f = res["filt_cov"]
    m_p = res["pred_mean"]
    P_p = res["pred_cov"]

    m_s = m_f.copy()
    P_s = P_f.copy()

    for k in range(len(m_f) - 2, -1, -1):
        X, _, wc = sigma_pts(m_f[k], P_f[k], cfg.alpha, cfg.beta, cfg.kappa)
        Xp = (F @ X.T).T

        D = np.zeros_like(P_f[k], dtype=float)
        for i in range(X.shape[0]):
            dx = X[i] - m_f[k]
            dp = Xp[i] - m_p[k + 1]
            D += wc[i] * np.outer(dx, dp)

        G = D @ np.linalg.inv(P_p[k + 1])
        m_s[k] = m_f[k] + G @ (m_s[k + 1] - m_p[k + 1])
        P_s[k] = sym(P_f[k] + G @ (P_s[k + 1] - P_p[k + 1]) @ G.T)

    return {
        "smooth_mean": m_s,
        "smooth_cov": P_s,
        "pos": m_s[:, :2],
    }
