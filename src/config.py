"""Central configuration: column names of the NGA-West2 flatfile and project settings.

IMPORTANT: Column names below follow the PEER NGA-West2 flatfile as best known.
Open your flatfile CSV, compare the header row with this file, and edit any name
that differs. Everything else in the project reads from here.
"""

# ---- identifiers / geometry ------------------------------------------------
RSN = "Record Sequence Number"
EVENT = "EQID"
STATION = "Station Sequence Number"
LAT = "Station Latitude"
LON = "Station Longitude"
MAG = "Earthquake Magnitude"
H1_FILE = "File Name (Horizontal 1)"
H2_FILE = "File Name (Horizontal 2)"
# The compass direction of each horizontal component is read from the 3-digit number
# at the end of its file name (e.g. 'A-HMC180.AT2' -> 180 degrees from north).

# ---- features --------------------------------------------------------------
# Event-level (same for both sites of a pair)
EVENT_NUM = [
    "Earthquake Magnitude", "Hypocenter Depth (km)", "Fault Rupture Width (km)",
    "Fault Rupture Length for Calculation of Ry (km)", "Dip (deg)", "Rake Angle (deg)",
]
EVENT_CAT = ["Mechanism Based on Rake Angle"]

# Site-level numeric (each becomes  mean_<name>  and  diff_<name>  for the pair)
SITE_NUM = [
    "EpiD (km)", "HypD (km)", "Joyner-Boore Dist. (km)", "ClstD (km)",
    "Vs30 (m/s) selected for analysis", "PGA (g)", "PGV (cm/sec)", "PGD (cm)",
]
# Site-level categorical (pair feature = sorted combination, e.g. "C|D")
SITE_CAT = ["Preferred NEHRP Based on Vs30"]

# ---- project settings ------------------------------------------------------
MIN_MAGNITUDE = 5.0        # drop events with M < 5.0 (as in the reference paper)
MAX_PAIR_DIST_KM = 100     # keep site pairs closer than 5 km
PSA_PERIOD_S = 1.0         # oscillator period used for RotD100 orientation
DAMPING = 0.05
ANGLE_STEP_DEG = 1
MISSING_SENTINELS = [-999, -999.0]   # flatfile "no data" markers
SEED = 42
SPLIT = (0.8, 0.1, 0.1)    # train / validation / test
