"""Synthetic NGA-West2-like data. ONLY for checking that the pipeline runs.
Results obtained from this data are meaningless scientifically."""
import os
import numpy as np
import pandas as pd
from scipy import signal
from . import config as C


def _write_at2(path, acc, dt):
    with open(path, "w") as f:
        f.write("PEER NGA STRONG MOTION DATABASE RECORD (SYNTHETIC)\nSYNTHETIC\nACCELERATION TIME SERIES IN UNITS OF G\n")
        f.write(f"NPTS= {len(acc):5d}, DT= {dt:.4f} SEC\n")
        for k in range(0, len(acc), 5):
            f.write("  ".join(f"{v: .7E}" for v in acc[k:k + 5]) + "\n")


def make(out_dir, n_events=20, n_sta=25, seed=0):
    rng = np.random.RandomState(seed)
    os.makedirs(os.path.join(out_dir, "waveforms"), exist_ok=True)
    rows, rsn, dt, n = [], 0, 0.01, 3000
    sos = signal.butter(4, 8, fs=1 / dt, output="sos")
    t = np.arange(n) * dt
    env = (t / 3) * np.exp(1 - t / 3)
    for e in range(n_events):
        mag = rng.uniform(5.0, 7.0)
        ev_angle = rng.uniform(0, 180)
        k = rng.uniform(0.3, 1.0, 2)
        for s in range(n_sta):
            x, y = rng.uniform(0, 8, 2)                   # km, 8x8 km patch
            lat, lon = 34 + y / 111.0, -118 + x / 92.0
            phi = (ev_angle + 40 * np.sin(k[0] * x + k[1] * y) + rng.normal(0, 15)) % 180
            src = signal.sosfilt(sos, rng.randn(n)) * env * rng.uniform(0.02, 0.2)
            a1_az = int(rng.randint(0, 360))
            r = np.deg2rad(phi - a1_az)
            h1 = src * np.cos(r) + 0.05 * src.std() * rng.randn(n)
            h2 = src * np.sin(r) + 0.05 * src.std() * rng.randn(n)
            rsn += 1
            d2 = (a1_az + 90) % 360
            flat1, flat2 = f"EVT{e}\\S{s}{a1_az:03d}.AT2", f"EVT{e}\\S{s}{d2:03d}.AT2"
            f1, f2 = f"RSN{rsn}_EVT{e}_S{s}{a1_az:03d}.AT2", f"RSN{rsn}_EVT{e}_S{s}{d2:03d}.AT2"
            _write_at2(os.path.join(out_dir, "waveforms", f1), h1, dt)
            _write_at2(os.path.join(out_dir, "waveforms", f2), h2, dt)
            dist = rng.uniform(5, 60)
            pga = float(np.abs(src).max())
            rows.append({
                C.RSN: rsn, C.EVENT: e, C.STATION: e * 1000 + s, C.LAT: lat, C.LON: lon,
                C.MAG: mag, C.H1_FILE: flat1, C.H2_FILE: flat2,
                "Hypocenter Depth (km)": rng.uniform(5, 20), "Fault Rupture Width (km)": rng.uniform(5, 20),
                "Fault Rupture Length for Calculation of Ry (km)": rng.uniform(10, 60), "Dip (deg)": rng.uniform(40, 90),
                "Rake Angle (deg)": rng.uniform(-180, 180),
                "Mechanism Based on Rake Angle": rng.choice([0, 1, 2, 3]),
                "EpiD (km)": dist, "HypD (km)": dist + 8, "Joyner-Boore Dist. (km)": dist - 2,
                "ClstD (km)": dist - 1, "Vs30 (m/s) selected for analysis": rng.uniform(200, 800),
                "PGA (g)": pga, "PGV (cm/sec)": pga * 40, "PGD (cm)": pga * 8,
                "Preferred NEHRP Based on Vs30": rng.choice(["B", "C", "D"]),
            })
    ff = os.path.join(out_dir, "flatfile_synthetic.csv")
    pd.DataFrame(rows).to_csv(ff, index=False)
    return ff, os.path.join(out_dir, "waveforms")
