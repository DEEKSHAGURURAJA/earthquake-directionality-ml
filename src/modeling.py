"""Splitting, preprocessing, feature selection, benchmark, ridge and neural network."""
import itertools
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from . import config as C


# ---------------------------------------------------------------- split
def split_data(pairs, by_event=False, seed=C.SEED):
    """80/10/10 train/validation/test split. by_event=True keeps all pairs of one
    earthquake in the same subset (stricter, avoids leakage between pairs)."""
    rng = np.random.RandomState(seed)
    n = len(pairs)
    if by_event:
        ev = pairs["event"].unique()
        rng.shuffle(ev)
        a, b = int(C.SPLIT[0] * len(ev)), int((C.SPLIT[0] + C.SPLIT[1]) * len(ev))
        grp = {e: (0 if k < a else 1 if k < b else 2) for k, e in enumerate(ev)}
        lab = pairs["event"].map(grp).values
    else:
        idx = rng.permutation(n)
        a, b = int(C.SPLIT[0] * n), int((C.SPLIT[0] + C.SPLIT[1]) * n)
        lab = np.empty(n, dtype=int)
        lab[idx[:a]], lab[idx[a:b]], lab[idx[b:]] = 0, 1, 2
    return [pairs[lab == k].reset_index(drop=True) for k in range(3)]


# ---------------------------------------------------------------- preprocessing
def make_preprocessor(cols, num_all, cat_all):
    num = [c for c in cols if c in num_all]
    cat = [c for c in cols if c in cat_all]
    parts = []
    if num:
        parts.append(("num", StandardScaler(), num))            # zero mean, unit variance
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))  # one-hot
    return ColumnTransformer(parts)


def make_ridge(cols, alpha, num_all, cat_all):
    return Pipeline([("prep", make_preprocessor(cols, num_all, cat_all)),
                     ("reg", Ridge(alpha=alpha))])


def val_r2(model, tr, va):
    model.fit(tr, tr["dtheta"])
    return r2_score(va["dtheta"], model.predict(va))


# ---------------------------------------------------------------- feature selection
def backward_ranking(cols, alpha, tr, va, num_all, cat_all):
    """Rank features by the drop in validation R2 when each is removed."""
    base = val_r2(make_ridge(cols, alpha, num_all, cat_all), tr, va)
    drops = {}
    for c in cols:
        rest = [x for x in cols if x != c]
        drops[c] = base - val_r2(make_ridge(rest, alpha, num_all, cat_all), tr, va)
    return sorted(drops, key=drops.get, reverse=True), drops


def forward_selection(ranked, alpha, tr, va, num_all, cat_all):
    """Add features in ranked order; stop once adding one lowers validation R2."""
    chosen, best = [], -np.inf
    for c in ranked:
        s = val_r2(make_ridge(chosen + [c], alpha, num_all, cat_all), tr, va)
        if s < best:
            break
        chosen, best = chosen + [c], s
    return chosen, best


def tune_alpha(cols, tr, va, num_all, cat_all, grid=None):
    grid = grid if grid is not None else np.logspace(-3, 3, 13)
    scores = [val_r2(make_ridge(cols, a, num_all, cat_all), tr, va) for a in grid]
    return float(grid[int(np.argmax(scores))]), dict(zip(map(float, grid), scores))


def select_features(all_cols, tr, va, num_all, cat_all, rounds=3, verbose=True):
    """Alternate backward ranking / forward selection with alpha tuning."""
    alpha, cols = 1.0, list(all_cols)
    ranked, drops = all_cols, {}
    for r in range(rounds):
        ranked, drops = backward_ranking(cols, alpha, tr, va, num_all, cat_all)
        cols, score = forward_selection(ranked, alpha, tr, va, num_all, cat_all)
        alpha, _ = tune_alpha(cols, tr, va, num_all, cat_all)
        if verbose:
            print(f"  round {r + 1}: {len(cols)} features, alpha={alpha:g}, val R2={score:.4f}")
    return cols, alpha, ranked, drops


# ---------------------------------------------------------------- models
def fit_benchmark(tr):
    """Benchmark: dtheta = c1 + c2 * distance."""
    m = LinearRegression()
    m.fit(tr[["distance_km"]], tr["dtheta"])
    return m


def fit_mlp_gridsearch(all_cols, tr, va, num_all, cat_all, layers, neurons, alphas, max_iter=300):
    """Grid search over depth, width and L2 strength; uses ALL features (as in the paper).
    Adam optimiser, ReLU activation; selection by validation R2."""
    prep = make_preprocessor(all_cols, num_all, cat_all).fit(tr)
    Xtr, Xva = prep.transform(tr), prep.transform(va)
    best, results = None, []
    for L, N, a in itertools.product(layers, neurons, alphas):
        net = MLPRegressor(hidden_layer_sizes=(N,) * L, activation="relu", solver="adam",
                           alpha=a, max_iter=max_iter, random_state=C.SEED)
        net.fit(Xtr, tr["dtheta"])
        s = r2_score(va["dtheta"], net.predict(Xva))
        results.append({"layers": L, "neurons": N, "alpha": a, "val_r2": s})
        if best is None or s > best[0]:
            best = (s, net, (L, N, a))
    return Pipeline([("prep", prep), ("reg", best[1])]), best[2], pd.DataFrame(results)


def evaluate(model, df, X=None):
    X = df if X is None else X
    p = np.clip(model.predict(X), 0, 90)
    return {"R2": r2_score(df["dtheta"], p), "MAE": mean_absolute_error(df["dtheta"], p)}, p
