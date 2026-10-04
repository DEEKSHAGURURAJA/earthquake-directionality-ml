"""Orientation extraction for all records and construction of site-pair examples."""
import os
import re
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


def parse_direction(name):
    """3-digit compass direction at the end of a PEER file name, e.g. 'A-HMC180.AT2' -> 180."""
    m = re.search(r"(\d{3})\.at2$", str(name).strip(), re.I)
    return int(m.group(1)) % 360 if m else None


def component_sign(az1, az2):
    """+1 if H2 points 90 deg clockwise of H1, -1 if 90 deg anticlockwise, else None."""
    if az1 is None or az2 is None:
        return None
    d = (az2 - az1) % 360
    return 1 if d == 90 else (-1 if d == 270 else None)


def _index_files(ts_dir):
    """Index waveform files by RSN (PEER download names start with 'RSN<n>_') and by plain name."""
    by_rsn, by_name = {}, {}
    for root, _, files in os.walk(ts_dir):
        for f in files:
            path = os.path.join(root, f)
            by_name[f.lower()] = path
            m = re.match(r"RSN(\d+)_", f, re.I)
            if m:
                by_rsn.setdefault(int(m.group(1)), []).append(path)
    return by_rsn, by_name


def _find_file(index, rsn, flat_name):
    by_rsn, by_name = index
    base = os.path.basename(str(flat_name).replace("\\", "/")).lower()     # 'a-hmc180.at2'
    for path in by_rsn.get(int(rsn), []):
        if os.path.basename(path).lower().endswith(base):
            return path
    return by_name.get(base)


def compute_orientations(df, ts_dir, cache_csv=None, verbose=True):
    """Add 'theta' (geographic RotD100 orientation, deg 0-180) to each record."""
    if cache_csv and os.path.exists(cache_csv):
        cached = pd.read_csv(cache_csv)
        return df.merge(cached, on=C.RSN, how="inner")
    index = _index_files(ts_dir)
    thetas, ok, why = [], [], {"no_file": 0, "bad_direction": 0, "read_error": 0}
    for k, row in df.iterrows():
        theta, good = np.nan, False
        try:
            p1 = _find_file(index, row[C.RSN], row[C.H1_FILE])
            p2 = _find_file(index, row[C.RSN], row[C.H2_FILE])
            if p1 is None or p2 is None:
                why["no_file"] += 1
            else:
                az1, az2 = parse_direction(row[C.H1_FILE]), parse_direction(row[C.H2_FILE])
                sign = component_sign(az1, az2)
                if sign is None:
                    why["bad_direction"] += 1
                else:
                    a1, dt1 = read_at2(p1)
                    a2, dt2 = read_at2(p2)
                    ang, _ = rotd100_orientation(a1, a2, dt1, C.PSA_PERIOD_S, C.DAMPING, C.ANGLE_STEP_DEG)
                    theta, good = (az1 + sign * ang) % 180.0, True
        except Exception:
            why["read_error"] += 1
        thetas.append(theta)
        ok.append(good)
        if verbose and (k + 1) % 500 == 0:
            print(f"  orientations: {k + 1}/{len(df)}")
    out = df.copy()
    out["theta"] = thetas
    out = out[ok].reset_index(drop=True)
    if verbose:
        print(f"  usable records: {len(out)}/{len(df)}  (skipped: {why})")
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
