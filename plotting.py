import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    }
)


styles = {
    "ekf": {"label": "EKF", "color": "#d55e00", "ls": "--", "marker": "o"},
    "ukf": {"label": "UKF", "color": "#0072b2", "ls": "-.", "marker": "s"},
    "erts": {"label": "EKF + extended RTS", "color": "#cc79a7", "ls": "-", "marker": "^"},
    "urts": {"label": "UKF + unscented RTS", "color": "#009e73", "ls": "-", "marker": "D"},
}


def style_ax(ax):
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_line(ax, x, y, key, lw=2.0):
    st = styles[key]
    ax.plot(
        x,
        y,
        color=st["color"],
        linestyle=st["ls"],
        linewidth=lw,
        marker=st["marker"],
        markevery=6,
        markersize=4.5,
        markeredgecolor="white",
        markeredgewidth=0.7,
        label=st["label"],
    )


def plot_geometry(x_true, cfg, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    rx = cfg.rx_pos

    ax.plot(x_true[:, 0], x_true[:, 1], color="#444444", linewidth=2.1, label="True trajectory")
    ax.scatter(rx[:, 0], rx[:, 1], marker="s", s=70, color="black", label="Receivers")
    ax.scatter(x_true[0, 0], x_true[0, 1], marker="o", s=60, color="#444444")
    ax.scatter(x_true[-1, 0], x_true[-1, 1], marker="x", s=75, color="#444444", linewidths=1.5)

    for i, r in enumerate(rx, start=1):
        ax.text(r[0] + 25.0, r[1] + 25.0, f"$s_{i}$", fontsize=10)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Receiver geometry and simulated emitter trajectory")
    ax.set_aspect("equal", adjustable="box")
    style_ax(ax)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_single_run(x_true, est, cfg, out_path):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.5),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    ax0, ax1 = axes

    ax0.plot(x_true[:, 0], x_true[:, 1], color="#222222", linewidth=2.6, label="True trajectory")
    for key in styles:
        pos = est[key]["pos"]
        draw_line(ax0, pos[:, 0], pos[:, 1], key, lw=1.9)

    rx = cfg.rx_pos
    ax0.scatter(rx[:, 0], rx[:, 1], marker="s", s=65, color="black", label="Receivers", zorder=4)
    for i, r in enumerate(rx, start=1):
        ax0.text(r[0] + 25.0, r[1] + 25.0, f"$s_{i}$", fontsize=10)

    ax0.set_xlabel("x [m]")
    ax0.set_ylabel("y [m]")
    ax0.set_title("Full geometry")
    ax0.set_aspect("equal", adjustable="box")
    style_ax(ax0)

    ax1.plot(x_true[:, 0], x_true[:, 1], color="#222222", linewidth=2.8, label="True trajectory")
    for key in styles:
        pos = est[key]["pos"]
        draw_line(ax1, pos[:, 0], pos[:, 1], key, lw=2.0)

    x_all = [x_true[:, 0]]
    y_all = [x_true[:, 1]]
    for key in styles:
        x_all.append(est[key]["pos"][:, 0])
        y_all.append(est[key]["pos"][:, 1])

    x_min = min(arr.min() for arr in x_all)
    x_max = max(arr.max() for arr in x_all)
    y_min = min(arr.min() for arr in y_all)
    y_max = max(arr.max() for arr in y_all)
    x_pad = 0.08 * (x_max - x_min)
    y_pad = 0.08 * (y_max - y_min)

    ax1.set_xlim(x_min - x_pad, x_max + x_pad)
    ax1.set_ylim(y_min - y_pad, y_max + y_pad)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("Zoom near trajectory")
    style_ax(ax1)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Single-run trajectory comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rmse(t, rmse_t, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.6), sharex=True)
    ax0, ax1 = axes

    for key in ["ekf", "ukf"]:
        draw_line(ax0, t, rmse_t[key], key, lw=2.2)
    ax0.set_ylabel("Filtered RMSE [m]")
    ax0.set_title("Filtered estimates")
    style_ax(ax0)
    ax0.legend(loc="upper right", frameon=False)

    for key in ["erts", "urts"]:
        draw_line(ax1, t, rmse_t[key], key, lw=2.2)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Smoothed RMSE [m]")
    ax1.set_title("Smoothed estimates")
    style_ax(ax1)
    ax1.legend(loc="upper right", frameon=False)

    fig.suptitle("Monte Carlo position RMSE over time", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
