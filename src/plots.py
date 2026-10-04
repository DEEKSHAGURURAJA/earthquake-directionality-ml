"""Figures used in the write-up / presentation."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def binned(x, y, bins):
    idx = np.digitize(x, bins) - 1
    xs, ms, es = [], [], []
    for k in range(len(bins) - 1):
        v = y[idx == k]
        if len(v) > 3:
            xs.append((bins[k] + bins[k + 1]) / 2)
            ms.append(v.mean())
            es.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
    return np.array(xs), np.array(ms), np.array(es)


def data_vs_feature(df, col, xlabel, path, bins=30):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(df[col], df["dtheta"], s=6, c="lightgray", label="Data")
    b = np.linspace(df[col].min(), df[col].max(), bins)
    x, m, e = binned(df[col].values, df["dtheta"].values, b)
    ax.errorbar(x, m, e, fmt="-o", c="k", ms=3, label="Moving average")
    ax.set_xlabel(xlabel); ax.set_ylabel("Δθ (deg)"); ax.set_ylim(0, 90); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def true_vs_pred(name, tr_y, tr_p, te_y, te_p, r2_tr, r2_te, path):
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.3), sharey=True)
    for ax, y, p, t, r in ((axs[0], tr_y, tr_p, "Training", r2_tr), (axs[1], te_y, te_p, "Test", r2_te)):
        ax.scatter(y, p, s=5); ax.set_title(f"{name} – {t}, R²={r:.3f}", fontsize=9)
        ax.set_xlabel("True Δθ"); ax.set_xlim(0, 90); ax.set_ylim(0, 90)
    axs[0].set_ylabel("Predicted Δθ")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def trend_comparison(df, pred, col, xlabel, name, path, bins=25):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    b = np.linspace(df[col].min(), df[col].max(), bins)
    x, m, e = binned(df[col].values, df["dtheta"].values, b)
    xp, mp, ep = binned(df[col].values, pred, b)
    ax.errorbar(x, m, e, fmt="-o", ms=3, label="Data (moving avg)")
    ax.errorbar(xp, mp, ep, fmt="-o", ms=3, label=f"{name} (moving avg)")
    ax.set_xlabel(xlabel); ax.set_ylabel("Δθ (deg)"); ax.set_ylim(0, 90); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def grid_heatmap(res, path):
    best = res.groupby(["layers", "neurons"])["val_r2"].max().unstack()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.imshow(best.values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(best.columns))); ax.set_xticklabels(best.columns)
    ax.set_yticks(range(len(best.index))); ax.set_yticklabels(best.index)
    ax.set_xlabel("Neurons per layer"); ax.set_ylabel("Number of layers")
    fig.colorbar(im, label="Validation R²")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
