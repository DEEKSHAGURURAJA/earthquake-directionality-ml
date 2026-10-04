"""Read PEER .AT2 files and find the RotD100 orientation of a record."""
import re
import numpy as np
from scipy import signal


def read_at2(path):
    """Return (acceleration array, dt) from a PEER NGA .AT2 file."""
    with open(path, "r", errors="ignore") as f:
        lines = f.read().splitlines()
    hdr = lines[3]
    npts = int(re.search(r"NPTS\s*=\s*(\d+)", hdr, re.I).group(1))
    dt = float(re.search(r"DT\s*=\s*([\d.eE+-]+)", hdr, re.I).group(1))
    acc = np.array(" ".join(lines[4:]).split(), dtype=float)[:npts]
    return acc, dt


def sdof_displacement(acc, dt, period, damping=0.05):
    """Relative displacement of a damped SDOF oscillator driven by ground acceleration.
    The ODE  u'' + 2*xi*w*u' + w^2*u = -a(t)  is discretised exactly (zero-order hold)
    and applied with scipy.signal.lfilter (fast, compiled)."""
    w = 2 * np.pi / period
    A = np.array([[0, 1], [-w ** 2, -2 * damping * w]])
    B = np.array([[0.0], [-1.0]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])
    Ad, Bd, Cd, Dd, _ = signal.cont2discrete((A, B, C, D), dt, method="zoh")
    b, a = signal.ss2tf(Ad, Bd, Cd, Dd)
    return signal.lfilter(b[0], a, acc)


def rotd100_orientation(h1, h2, dt, period=1.0, damping=0.05, step=1):
    """Orientation (deg, 0-180, measured from H1 towards H2) in which the oscillator
    response is largest (RotD100), and the corresponding PSA (g if input in g).

    Because the oscillator is linear, the response in direction theta is
    cos(theta)*u1 + sin(theta)*u2, so only two oscillator runs are needed."""
    n = min(len(h1), len(h2))
    u1 = sdof_displacement(h1[:n], dt, period, damping)
    u2 = sdof_displacement(h2[:n], dt, period, damping)
    angles = np.arange(0, 180, step)
    th = np.deg2rad(angles)
    resp = np.abs(np.cos(th)[:, None] * u1 + np.sin(th)[:, None] * u2).max(axis=1)
    k = int(resp.argmax())
    w = 2 * np.pi / period
    return float(angles[k]), float(w ** 2 * resp[k])
