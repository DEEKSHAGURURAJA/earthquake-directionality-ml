"""End-to-end pipeline: flatfile + waveforms -> pairs -> features -> models -> results.

Real data:       python run_pipeline.py --flatfile PATH.csv --ts-dir PATH/TO/AT2_FILES
Smoke test only: python run_pipeline.py --synthetic
"""
import argparse, json, os
import joblib
import pandas as pd
from src import config as C
from src import pairs as P
from src import modeling as M
from src import plots as PL
from src import synthetic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flatfile"); ap.add_argument("--ts-dir")
    ap.add_argument("--synthetic", action="store_true", help="use fake data to test the code")
    ap.add_argument("--out", default="results")
    ap.add_argument("--by-event-split", action="store_true", help="split by earthquake instead of by pair")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--neurons", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--nn-alphas", type=float, nargs="+", default=[0.001, 0.1])
    ap.add_argument("--max-iter", type=int, default=300)
    a = ap.parse_args()

    out = a.out + ("_synthetic" if a.synthetic else "")
    os.makedirs(out, exist_ok=True)
    if a.synthetic:
        print("[0] Generating SYNTHETIC data (results are NOT scientific)")
        a.flatfile, a.ts_dir = synthetic.make("data/synthetic")
    assert a.flatfile and a.ts_dir, "give --flatfile and --ts-dir (or --synthetic)"

    print("[1] Loading flatfile and filtering magnitude >= %.1f" % C.MIN_MAGNITUDE)
    rec = P.load_flatfile(a.flatfile)
    print(f"    {len(rec)} records")

    print("[2] Computing RotD100 orientation of every record from waveforms")
    rec = P.compute_orientations(rec, a.ts_dir, cache_csv=os.path.join(out, "orientations.csv"))

    print(f"[3] Pairing sites within {C.MAX_PAIR_DIST_KM} km and computing dtheta")
    pairs = P.build_pairs(rec)
    pairs.to_csv(os.path.join(out, "pairs.csv"), index=False)
    print(f"    {len(pairs)} pair examples from {pairs['event'].nunique()} earthquakes")

    num_all, cat_all = P.feature_lists()
    all_cols = num_all + cat_all
    tr, va, te = M.split_data(pairs, by_event=a.by_event_split)
    print(f"    split: train={len(tr)} val={len(va)} test={len(te)}")

    PL.data_vs_feature(pairs, "distance_km", "Distance between sites (km)", f"{out}/fig_dtheta_vs_distance.png")
    pga = "mean_PGA (g)"
    if pga in pairs:
        PL.data_vs_feature(pairs, pga, "Average PGA (g)", f"{out}/fig_dtheta_vs_pga.png")

    print("[4] Feature selection (backward ranking + forward selection, ridge)")
    cols, alpha, ranked, drops = M.select_features(all_cols, tr, va, num_all, cat_all)
    pd.Series(drops).sort_values(ascending=False).to_csv(f"{out}/feature_importance.csv", header=["r2_drop"])
    print(f"    selected {len(cols)} features: {cols}")

    print("[5] Fitting models")
    bench = M.fit_benchmark(tr)
    ridge = M.make_ridge(cols, alpha, num_all, cat_all).fit(tr, tr["dtheta"])
    nn, nn_params, grid = M.fit_mlp_gridsearch(all_cols, tr, va, num_all, cat_all,
                                                a.layers, a.neurons, a.nn_alphas, a.max_iter)
    grid.to_csv(f"{out}/nn_gridsearch.csv", index=False)
    PL.grid_heatmap(grid, f"{out}/fig_nn_gridsearch.png")
    print(f"    ridge alpha={alpha:g}; NN layers={nn_params[0]} neurons={nn_params[1]} alpha={nn_params[2]}")

    print("[6] Evaluating")
    models = {"Benchmark": (bench, lambda d: d[["distance_km"]]), "Ridge regression": (ridge, None),
              "Neural network": (nn, None)}
    rows = []
    for name, (m, sel) in models.items():
        rtr, ptr = M.evaluate(m, tr, sel(tr) if sel else None)
        rte, pte = M.evaluate(m, te, sel(te) if sel else None)
        rows.append({"Model": name, "R2_train": rtr["R2"], "R2_test": rte["R2"], "MAE_test": rte["MAE"]})
        tag = name.split()[0].lower()
        PL.true_vs_pred(name, tr["dtheta"], ptr, te["dtheta"], pte, rtr["R2"], rte["R2"], f"{out}/fig_{tag}_true_vs_pred.png")
        PL.trend_comparison(te, pte, "distance_km", "Distance between sites (km)", name, f"{out}/fig_{tag}_trend_distance.png")
        if pga in te:
            PL.trend_comparison(te, pte, pga, "Average PGA (g)", name, f"{out}/fig_{tag}_trend_pga.png")
    table = pd.DataFrame(rows)
    table.to_csv(f"{out}/metrics.csv", index=False)
    print(table.round(4).to_string(index=False))

    meta = {"n_records": len(rec), "n_pairs": len(pairs), "n_events": int(pairs["event"].nunique()),
            "n_train": len(tr), "n_val": len(va), "n_test": len(te), "selected_features": cols,
            "ridge_alpha": alpha, "nn_layers": nn_params[0], "nn_neurons": nn_params[1],
            "nn_alpha": nn_params[2], "psa_period_s": C.PSA_PERIOD_S,
            "synthetic": bool(a.synthetic)}
    json.dump(meta, open(f"{out}/run_summary.json", "w"), indent=2)
    joblib.dump({"benchmark": bench, "ridge": ridge, "nn": nn, "cols": cols, "all_cols": all_cols,
                 "flatfile": a.flatfile, "ts_dir": a.ts_dir, "out": out}, f"{out}/model_artifacts.joblib")
    print(f"Done. Outputs in ./{out}/")


if __name__ == "__main__":
    main()
