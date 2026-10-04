"""Orientation extraction for all records and construction of site-pair examples."""
import os
import numpy as np
import pandas as pd
from . import config as C
from .waveform import read_at2, rotd100_orientation


def load_flatfile(path):
    """Load the flatfile, mark sentinel values as missing, keep M >= MIN_MAGNITUDE."""
    df = pd.read_csv(path, low_memory=False)
    df = df.replace(C.MISSING_SENTINELS, np.nan)
    df = df[df[C.MAG] >= C.MIN_MAGNITUDE].reset_index(drop=True)
    return df


def compute_orientations(df, ts_dir, cache_csv=None, verbose=True):
    """Add 'theta' (geographic RotD100 orientation, deg 0-180) to each record."""
    if cache_csv and os.path.exists(cache_csv):
        cached = pd.read_csv(cache_csv)
        return df.merge(cached, on=C.RSN, how="inner")
    index = {}
    for root, _, files in os.walk(ts_dir):
        for f in files:
            index[f.lower()] = os.path.join(root, f)
    thetas, ok = [], []
    for k, row in df.iterrows():
        try:
            p1 = index[os.path.basename(str(row[C.H1_FILE])).lower()]
            p2 = index[os.path.basename(str(row[C.H2_FILE])).lower()]
            a1, dt1 = read_at2(p1)
            a2, dt2 = read_at2(p2)
            ang, _ = rotd100_orientation(a1, a2, dt1, C.PSA_PERIOD_S, C.DAMPING, C.ANGLE_STEP_DEG)
            az = row[C.H1_AZIMUTH] if C.H1_AZIMUTH else 0.0
            thetas.append((ang + az) % 180.0)
            ok.append(True)
        except Exception as e:                      # missing/corrupt file
            thetas.append(np.nan)
            ok.append(False)
        if verbose and (k + 1) % 200 == 0:
            print(f"  orientations: {k + 1}/{len(df)}")
    out = df.copy()
    out["theta"] = thetas
    out = out[ok].reset_index(drop=True)
    if verbose:
        print(f"  usable records: {len(out)}/{len(df)}")
    if cache_csv:
        out[[C.RSN, "theta"]].to_csv(cache_csv, index=False)
    return out


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def angle_difference(t1, t2):
    """Absolute difference between two orientations (period 180 deg) folded to [0, 90]."""
    d = np.abs(t1 - t2) % 180.0
    return np.minimum(d, 180.0 - d)


def pair_frame(g, i, j):
    """Build pair features for record index arrays i, j of dataframe g."""
    gi, gj = g.iloc[i].reset_index(drop=True), g.iloc[j].reset_index(drop=True)
    out = pd.DataFrame({
        "event": gi[C.EVENT], "rsn_i": gi[C.RSN], "rsn_j": gj[C.RSN],
        "distance_km": _haversine_km(gi[C.LAT].values, gi[C.LON].values,
                                     gj[C.LAT].values, gj[C.LON].values),
    })
    for c in C.EVENT_NUM + C.EVENT_CAT:
        out[c] = gi[c]
    for c in C.SITE_NUM:
        out[f"mean_{c}"] = (gi[c].values + gj[c].values) / 2
        out[f"diff_{c}"] = np.abs(gi[c].values - gj[c].values)
    for c in C.SITE_CAT:
        out[f"pair_{c}"] = [("|".join(sorted([str(a), str(b)]))) for a, b in zip(gi[c], gj[c])]
    if "theta" in g.columns:
        out["dtheta"] = angle_difference(gi["theta"].values, gj["theta"].values)
    return out


def feature_lists():
    num = ["distance_km"] + C.EVENT_NUM + [f"{p}_{c}" for c in C.SITE_NUM for p in ("mean", "diff")]
    cat = C.EVENT_CAT + [f"pair_{c}" for c in C.SITE_CAT]
    return num, cat


def build_pairs(df, max_km=None):
    """All pairs of different stations that recorded the same earthquake and lie
    within max_km of each other."""
    max_km = max_km or C.MAX_PAIR_DIST_KM
    need = [C.LAT, C.LON, C.STATION, "theta"] + C.EVENT_NUM + C.EVENT_CAT + C.SITE_NUM + C.SITE_CAT
    df = df.dropna(subset=need).reset_index(drop=True)
    frames = []
    for _, g in df.groupby(C.EVENT):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue
        i, j = np.triu_indices(n, 1)
        d = _haversine_km(g[C.LAT].values[i], g[C.LON].values[i], g[C.LAT].values[j], g[C.LON].values[j])
        keep = (d <= max_km) & (g[C.STATION].values[i] != g[C.STATION].values[j])
        if keep.any():
            frames.append(pair_frame(g, i[keep], j[keep]))
    return pd.concat(frames, ignore_index=True)
