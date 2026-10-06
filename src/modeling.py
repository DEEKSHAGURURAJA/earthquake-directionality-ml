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
        a = int(C.SPLIT[0] * len(ev))
        b = int((C.SPLIT[0] + C.SPLIT[1]) * len(ev))
        grp = {
            e: (0 if k < a else 1 if k < b else 2)
            for k, e in enumerate(ev)
        }
        lab = pairs["event"].map(grp).values
    else:
        idx = rng.permutation(n)
        a = int(C.SPLIT[0] * n)
        b = int((C.SPLIT[0] + C.SPLIT[1]) * n)

        lab = np.empty(n, dtype=int)
        lab[idx[:a]] = 0
        lab[idx[a:b]] = 1
        lab[idx[b:]] = 2

    return [
        pairs[lab == k].reset_index(drop=True)
        for k in range(3)
    ]


# ---------------------------------------------------------------- preprocessing
def make_preprocessor(cols, num_all, cat_all):
    num = [c for c in cols if c in num_all]
    cat = [c for c in cols if c in cat_all]

    parts = []

    if num:
        parts.append(
            ("num", StandardScaler(), num)
        )

    if cat:
        parts.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                cat,
            )
        )

    return ColumnTransformer(parts)


def make_ridge(cols, alpha, num_all, cat_all):
    if not cols:
        raise ValueError("Cannot create Ridge model with zero features.")

    return Pipeline(
        [
            (
                "prep",
                make_preprocessor(
                    cols,
                    num_all,
                    cat_all,
                ),
            ),
            (
                "reg",
                Ridge(alpha=alpha),
            ),
        ]
    )


def val_r2(model, tr, va):
    model.fit(
        tr,
        tr["dtheta"],
    )

    return r2_score(
        va["dtheta"],
        model.predict(va),
    )


# ---------------------------------------------------------------- feature selection
def backward_ranking(cols, alpha, tr, va, num_all, cat_all):
    """Rank features by the drop in validation R2 when each is removed."""

    # Safety: if only one feature remains, it cannot be removed.
    if len(cols) <= 1:
        return list(cols), {c: 0.0 for c in cols}

    base = val_r2(
        make_ridge(
            cols,
            alpha,
            num_all,
            cat_all,
        ),
        tr,
        va,
    )

    drops = {}

    for c in cols:
        rest = [x for x in cols if x != c]

        # IMPORTANT: never pass zero features to Ridge.
        if not rest:
            continue

        drops[c] = base - val_r2(
            make_ridge(
                rest,
                alpha,
                num_all,
                cat_all,
            ),
            tr,
            va,
        )

    return (
        sorted(
            drops,
            key=drops.get,
            reverse=True,
        ),
        drops,
    )


def forward_selection(ranked, alpha, tr, va, num_all, cat_all):
    """Add features in ranked order; stop once adding one lowers validation R2."""

    chosen = []
    best = -np.inf

    for c in ranked:
        candidate = chosen + [c]

        s = val_r2(
            make_ridge(
                candidate,
                alpha,
                num_all,
                cat_all,
            ),
            tr,
            va,
        )

        if s < best:
            break

        chosen = candidate
        best = s

    # Safety: forward selection must never return zero features.
    if not chosen and ranked:
        chosen = [ranked[0]]

        best = val_r2(
            make_ridge(
                chosen,
                alpha,
                num_all,
                cat_all,
            ),
            tr,
            va,
        )

    return chosen, best


def tune_alpha(cols, tr, va, num_all, cat_all, grid=None):
    grid = (
        grid
        if grid is not None
        else np.logspace(-3, 3, 13)
    )

    # Safety: never tune Ridge with zero features.
    if not cols:
        raise ValueError(
            "Feature selection produced zero features."
        )

    scores = [
        val_r2(
            make_ridge(
                cols,
                a,
                num_all,
                cat_all,
            ),
            tr,
            va,
        )
        for a in grid
    ]

    return (
        float(grid[int(np.argmax(scores))]),
        dict(
            zip(
                map(float, grid),
                scores,
            )
        ),
    )


def select_features(
    all_cols,
    tr,
    va,
    num_all,
    cat_all,
    rounds=3,
    verbose=True,
):
    """Alternate backward ranking / forward selection with alpha tuning."""

    alpha = 1.0
    cols = list(all_cols)

    ranked = list(all_cols)
    drops = {}

    for r in range(rounds):

        # Never enter a round with zero features.
        if not cols:
            cols = list(all_cols)

        ranked, drops = backward_ranking(
            cols,
            alpha,
            tr,
            va,
            num_all,
            cat_all,
        )

        # If ranking somehow becomes empty, retain the current features.
        if not ranked:
            ranked = list(cols)

        cols, score = forward_selection(
            ranked,
            alpha,
            tr,
            va,
            num_all,
            cat_all,
        )

        # Absolute safety fallback.
        if not cols:
            cols = [ranked[0]]

        alpha, _ = tune_alpha(
            cols,
            tr,
            va,
            num_all,
            cat_all,
        )

        if verbose:
            print(
                f"  round {r + 1}: "
                f"{len(cols)} features, "
                f"alpha={alpha:g}, "
                f"val R2={score:.4f}"
            )

    return (
        cols,
        alpha,
        ranked,
        drops,
    )


# ---------------------------------------------------------------- models
def fit_benchmark(tr):
    """Benchmark: dtheta = c1 + c2 * distance."""

    m = LinearRegression()

    m.fit(
        tr[["distance_km"]],
        tr["dtheta"],
    )

    return m


def fit_mlp_gridsearch(
    all_cols,
    tr,
    va,
    num_all,
    cat_all,
    layers,
    neurons,
    alphas,
    max_iter=300,
):
    """Grid search over depth, width and L2 strength; uses ALL features (as in the paper).
    Adam optimiser, ReLU activation; selection by validation R2."""

    prep = make_preprocessor(
        all_cols,
        num_all,
        cat_all,
    ).fit(tr)

    Xtr = prep.transform(tr)
    Xva = prep.transform(va)

    best = None
    results = []

    for L, N, a in itertools.product(
        layers,
        neurons,
        alphas,
    ):
        net = MLPRegressor(
            hidden_layer_sizes=(N,) * L,
            activation="relu",
            solver="adam",
            alpha=a,
            max_iter=max_iter,
            random_state=C.SEED,
        )

        net.fit(
            Xtr,
            tr["dtheta"],
        )

        s = r2_score(
            va["dtheta"],
            net.predict(Xva),
        )

        results.append(
            {
                "layers": L,
                "neurons": N,
                "alpha": a,
                "val_r2": s,
            }
        )

        if best is None or s > best[0]:
            best = (
                s,
                net,
                (L, N, a),
            )

    return (
        Pipeline(
            [
                ("prep", prep),
                ("reg", best[1]),
            ]
        ),
        best[2],
        pd.DataFrame(results),
    )


def evaluate(model, df, X=None):
    X = df if X is None else X

    p = np.clip(
        model.predict(X),
        0,
        90,
    )

    return (
        {
            "R2": r2_score(
                df["dtheta"],
                p,
            ),
            "MAE": mean_absolute_error(
                df["dtheta"],
                p,
            ),
        },
        p,
    )