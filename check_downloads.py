"""Report what the downloaded waveforms can give: usable records and site pairs within 5 km.
   python check_downloads.py [timeseries_folder]"""
import os, re, sys
import numpy as np
import pandas as pd
from src import config as C
from src import pairs as P

ts = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "timeseries")
have = set()
for root, _, files in os.walk(ts):
    for f in files:
        m = re.match(r"RSN(\d+)_", f, re.I)
        if m and f.lower().endswith(".at2"):
            have.add(int(m.group(1)))
print(f"Downloaded records (RSNs with .AT2 files): {len(have)}")

df = pd.read_csv(os.path.join("data", "NGAWest2_flatfile.csv"), low_memory=False).replace(C.MISSING_SENTINELS, np.nan)
m = df[(df[C.MAG] >= C.MIN_MAGNITUDE) & df[C.RSN].isin(have)].reset_index(drop=True)
print(f"  of these, magnitude >= {C.MIN_MAGNITUDE}: {len(m)}")
sign = [P.component_sign(P.parse_direction(x), P.parse_direction(y)) for x, y in zip(m[C.H1_FILE], m[C.H2_FILE])]
m = m[[s is not None for s in sign]]
print(f"  with a usable component direction in the file names: {len(m)}")
need = [C.LAT, C.LON, C.STATION] + C.EVENT_NUM + C.EVENT_CAT + C.SITE_NUM + C.SITE_CAT
m = m.dropna(subset=need)
print(f"  with all chosen features present: {len(m)}")

# pairs (same earthquake) at several distance limits
dists, ev_of = [], []
for eq, g in m.groupby(C.EVENT):
    g = g.reset_index(drop=True)
    n = len(g)
    if n < 2:
        continue
    i, j = np.triu_indices(n, 1)
    d = P._haversine_km(g[C.LAT].values[i], g[C.LON].values[i], g[C.LAT].values[j], g[C.LON].values[j])
    ok = g[C.STATION].values[i] != g[C.STATION].values[j]
    dists += list(d[ok]); ev_of += [eq] * int(ok.sum())
dists, ev_of = np.array(dists), np.array(ev_of)
print(f"\nUsable records: {len(m)} from {m[C.EVENT].nunique()} earthquakes "
      f"({int((m.groupby(C.EVENT).size() >= 2).sum())} earthquakes have 2 or more usable records)")
print("\nSite pairs (same earthquake, different stations) by distance limit:")
for lim in (5, 10, 20, 50, 100, 1e9):
    k = int((dists <= lim).sum())
    ne = len(set(ev_of[dists <= lim])) if k else 0
    label = "any distance" if lim > 1e8 else f"within {lim:g} km"
    print(f"  {label:>16}: {k:5d} pairs from {ne} earthquakes")

lst = os.path.join("data", "rsn_to_download.txt")
if os.path.exists(lst):
    want = set(int(x) for x in open(lst).read().split(",") if x)
    print(f"\nOverlap with data/rsn_to_download.txt: {len(have & want)} of {len(want)} planned records downloaded")