"""Pick up to N records (RSNs) that form the most same-earthquake site pairs within 5 km."""
import argparse, os
import numpy as np
import pandas as pd
from src import config as C
from src import pairs as P

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=os.path.join("data", "NGAWest2_flatfile.csv"))
ap.add_argument("--n", type=int, default=180)
ap.add_argument("--max-per-event", type=int, default=20)
ap.add_argument("--out", default=os.path.join("data", "rsn_to_download.txt"))
a = ap.parse_args()

df = pd.read_csv(a.csv, low_memory=False).replace(C.MISSING_SENTINELS, np.nan)
df = df[df[C.MAG] >= C.MIN_MAGNITUDE]
need = [C.LAT, C.LON, C.STATION, C.H1_FILE, C.H2_FILE] + C.EVENT_NUM + C.EVENT_CAT + C.SITE_NUM + C.SITE_CAT
df = df.dropna(subset=need)
ok_dir = [P.component_sign(P.parse_direction(x), P.parse_direction(y)) is not None
          for x, y in zip(df[C.H1_FILE], df[C.H2_FILE])]
df = df[ok_dir].reset_index(drop=True)

events = {}
for eq, g in df.groupby(C.EVENT):
    g = g.reset_index(drop=True)
    n = len(g)
    if n < 2:
        continue
    i, j = np.triu_indices(n, 1)
    d = P._haversine_km(g[C.LAT].values[i], g[C.LON].values[i], g[C.LAT].values[j], g[C.LON].values[j])
    ok = (d <= C.MAX_PAIR_DIST_KM) & (g[C.STATION].values[i] != g[C.STATION].values[j])
    if not ok.any():
        continue
    A = np.zeros((n, n), bool)
    A[i[ok], j[ok]] = True
    A |= A.T
    sel = [int(A.sum(1).argmax())]
    steps = []
    while len(sel) < min(a.max_per_event, n):
        gain = A[:, sel].sum(1).astype(int)
        gain[sel] = -1
        v = int(gain.argmax())
        if gain[v] <= 0:
            break
        steps.append((int(gain[v]), 2 if len(sel) == 1 else 1))
        sel.append(v)
    if steps:
        events[eq] = {"rsn": g[C.RSN].values[sel], "steps": steps, "ptr": 0}

budget, chosen_steps = a.n, {}
while True:
    best, best_ratio = None, 0
    for eq, e in events.items():
        if e["ptr"] < len(e["steps"]):
            gain, cost = e["steps"][e["ptr"]]
            if cost <= budget and gain / cost > best_ratio:
                best, best_ratio = eq, gain / cost
    if best is None:
        break
    e = events[best]
    gain, cost = e["steps"][e["ptr"]]
    e["ptr"] += 1
    budget -= cost
    chosen_steps[best] = chosen_steps.get(best, 0) + 1

rsns, pairs = [], 0
for eq, k in chosen_steps.items():
    e = events[eq]
    rsns += [int(x) for x in e["rsn"][:k + 1]]
    pairs += sum(g for g, _ in e["steps"][:k])
os.makedirs(os.path.dirname(a.out), exist_ok=True)
open(a.out, "w").write(",".join(map(str, sorted(rsns))))
print(f"Selected {len(rsns)} records from {len(chosen_steps)} earthquakes -> about {pairs} pairs within {C.MAX_PAIR_DIST_KM} km")
print(f"RSN list written to {a.out}")
