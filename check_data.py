"""Inspect the real flatfile: file-name samples, missing values, and which records need waveforms."""
import os
import numpy as np
import pandas as pd
from src import config as C
from src import pairs as P

CSV = r"data\NGAWest2_flatfile.csv"
if not os.path.exists(CSV):
    CSV = CSV.replace("\\", "/")
df = pd.read_csv(CSV, low_memory=False).replace(C.MISSING_SENTINELS, np.nan)
m = df[df[C.MAG] >= C.MIN_MAGNITUDE].reset_index(drop=True)
print(f"Records in file: {len(df)} | with magnitude >= {C.MIN_MAGNITUDE}: {len(m)} | earthquakes: {m[C.EVENT].nunique()}")

print("\n=== Sample waveform file names (look for a direction like 090 / 180 / 270 at the end) ===")
print(m[[C.RSN, C.H1_FILE, C.H2_FILE]].head(8).to_string(index=False))

need = [C.LAT, C.LON, C.STATION] + C.EVENT_NUM + C.EVENT_CAT + C.SITE_NUM + C.SITE_CAT
miss = m[need].isna().sum().sort_values(ascending=False)
print("\n=== Missing values per needed column (among M >= 5 records) ===")
print(miss[miss > 0].to_string() if (miss > 0).any() else "none")


def candidate_rsns(d):
    keep, npairs = set(), 0
    for _, g in d.groupby(C.EVENT):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue
        i, j = np.triu_indices(n, 1)
        dist = P._haversine_km(g[C.LAT].values[i], g[C.LON].values[i], g[C.LAT].values[j], g[C.LON].values[j])
        ok = (dist <= C.MAX_PAIR_DIST_KM) & (g[C.STATION].values[i] != g[C.STATION].values[j])
        npairs += int(ok.sum())
        keep |= set(g[C.RSN].values[i[ok]]) | set(g[C.RSN].values[j[ok]])
    return keep, npairs


loc_only = m.dropna(subset=[C.LAT, C.LON, C.STATION, C.H1_FILE, C.H2_FILE])
strict = m.dropna(subset=need + [C.H1_FILE, C.H2_FILE])
for name, d in (("location only (upper bound)", loc_only), ("all chosen features present", strict)):
    keep, npairs = candidate_rsns(d)
    print(f"\n[{name}] pairs within {C.MAX_PAIR_DIST_KM} km: {npairs} | records needing waveforms: {len(keep)}")
    fn = "data/needed_rsn_all.txt" if name.startswith("location") else "data/needed_rsn_strict.txt"
    with open(fn, "w") as f:
        f.write(",".join(str(int(x)) for x in sorted(keep)))
    print("  wrote", fn)
