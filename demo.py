"""Live demo: predict dtheta for two recording sites of one earthquake.

   python demo.py --artifacts results/model_artifacts.joblib --rsn1 12 --rsn2 15
   python demo.py --artifacts results/model_artifacts.joblib --random      # pick a random close pair
If waveforms exist for the chosen records, the true dtheta is shown as well.
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from src import config as C
from src import pairs as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--rsn1", type=int); ap.add_argument("--rsn2", type=int)
    ap.add_argument("--random", action="store_true")
    a = ap.parse_args()

    art = joblib.load(a.artifacts)
    pairs = pd.read_csv(f"{art['out']}/pairs.csv")
    if a.random or not (a.rsn1 and a.rsn2):
        row = pairs.sample(1).iloc[0]
        a.rsn1, a.rsn2 = int(row.rsn_i), int(row.rsn_j)
    rec = pd.read_csv(art["flatfile"], low_memory=False).replace(C.MISSING_SENTINELS, np.nan)
    g = rec[rec[C.RSN].isin([a.rsn1, a.rsn2])].reset_index(drop=True)
    assert len(g) == 2 and g[C.EVENT].nunique() == 1, "records must exist and belong to the same earthquake"
    x = P.pair_frame(g, [0], [1])
    d = x["distance_km"].iloc[0]
    print(f"Sites RSN {a.rsn1} and RSN {a.rsn2}  (earthquake {g[C.EVENT].iloc[0]}, M={g[C.MAG].iloc[0]:.1f})")
    print(f"Distance between sites: {d:.2f} km" + ("   [warning: > 5 km, outside training range]" if d > 5 else ""))
    print(f"Benchmark prediction (distance only): {np.clip(art['benchmark'].predict(x[['distance_km']])[0], 0, 90):6.1f} deg")
    print(f"Ridge regression prediction:          {np.clip(art['ridge'].predict(x)[0], 0, 90):6.1f} deg")
    print(f"Neural network prediction:            {np.clip(art['nn'].predict(x)[0], 0, 90):6.1f} deg")
    true = pairs[(pairs.rsn_i == a.rsn1) & (pairs.rsn_j == a.rsn2)]
    if len(true):
        print(f"True dtheta (from waveforms):         {true['dtheta'].iloc[0]:6.1f} deg")


if __name__ == "__main__":
    main()
