"""Build the PDF write-up. Uses results/ if present, otherwise leaves blanks to fill."""
import json
import os
import sys

import pandas as pd
from src import config as C

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


RES = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = "Writeup_Earthquake_Directionality.pdf"

# ---- EDIT THESE TWO TEXTS IN YOUR OWN WORDS -------------------------------

# All three models predicted the angle difference no better than guessing its
# average: test R2 was close to zero and the mean absolute error was about
# 23 degrees, which is close to the 22.5 degrees obtained by always
# predicting 45 degrees. Closer site pairs did show smaller angle differences
# (about 37 degrees below 10 km, against about 47 degrees at 50 to 100 km),
# which agrees with the reference paper, but with only 17 earthquakes and
# 20 pairs within 5 km this was too little for the models to learn from.
# The neural network fitted the training data slightly better than the test
# data. With more records at short distances, as in the paper's 5 km dataset,
# we would expect clearer results.

CONCLUSION_EXTRA = (
    "All three models predicted the angle difference no better than guessing "
    "its average: the test R-squared was close to zero and the mean absolute "
    "error was about 23 degrees, close to the 22.5 degrees obtained by "
    "always predicting 45 degrees. Closer site pairs did show smaller angle "
    "differences (about 37 degrees below 10 km, against about 47 degrees at "
    "50 to 100 km), which agrees with the reference paper, but with only "
    "17 earthquakes and 20 pairs within 5 km there was too little "
    "short-distance data for the models to learn from. The neural network "
    "fitted the training data better than the test data. With the full 5 km "
    "dataset of the paper, we would expect clearer results."
)

CONTRIBUTIONS = "Deeksha G: ____ . Harshitha B: ____ ."

# ---------------------------------------------------------------------------

REPO = "https://github.com/DEEKSHAGURURAJA/earthquake-directionality-ml."

S = getSampleStyleSheet()

body = ParagraphStyle(
    "b",
    parent=S["Normal"],
    fontSize=8.8,
    leading=11,
    alignment=4,
)

h = ParagraphStyle(
    "h",
    parent=S["Heading3"],
    fontSize=10,
    spaceBefore=5,
    spaceAfter=2,
)

title = ParagraphStyle(
    "t",
    parent=S["Title"],
    fontSize=14,
    leading=17,
    spaceAfter=3,
)

ctr = ParagraphStyle(
    "c",
    parent=body,
    alignment=1,
)

have = (
    os.path.exists(f"{RES}/metrics.csv")
    and os.path.exists(f"{RES}/run_summary.json")
)

m = pd.read_csv(f"{RES}/metrics.csv") if have else None

info = (
    json.load(open(f"{RES}/run_summary.json"))
    if have
    else {}
)

B = "____"

DIST = f"{C.MAX_PAIR_DIST_KM:g}"

LIMIT = "" if C.MAX_PAIR_DIST_KM == 5 else (
    "Note: the reference paper limited site pairs to 5 km. "
    "PEER restricts waveform downloads to roughly 200 records per two weeks, "
    f"so our dataset is much smaller than the paper's and we allowed sites "
    f"up to {DIST} km apart; our results are therefore not directly "
    "comparable with the paper's."
)

g = lambda k: f"{info[k]:,}" if k in info else B


story = [
    Paragraph(
        "Predicting Earthquake Ground-Motion Directionality Differences "
        "between Nearby Sites using Machine Learning",
        title,
    ),

    Paragraph(
        "UE24CS352A - Machine Learning - Mini-Project",
        ctr,
    ),

    Paragraph(
        "<b>Deeksha G</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
        "<b>Harshitha B</b>",
        ctr,
    ),

    Paragraph(
        f"Code: {REPO}",
        ctr,
    ),

    Paragraph(
        "1. Problem Statement",
        h,
    ),

    Paragraph(
        "During an earthquake, shaking at a site is strongest along one "
        "horizontal orientation (RotD100). Building codes treat this "
        "orientation as uniformly random, but observations show that nearby "
        "sites have correlated orientations. Our task (assigned problem, "
        "based on Poulos &amp; Silva-Lopez) is to build a model "
        "<i>f</i> that takes the seismological, geological and geographical "
        "attributes of two sites and of the earthquake, and predicts the "
        "absolute difference dtheta (0-90 degrees) between their orientations "
        "of maximum shaking. We compare a distance-only benchmark, ridge "
        "regression and a neural network.",
        body,
    ),

    Paragraph(
        "2. Dataset",
        h,
    ),

    Paragraph(
        f"We use the NGA-West2 database (PEER): a flatfile with 274 "
        f"attributes per record plus the two-component acceleration time "
        f"series of shallow crustal earthquakes worldwide. Records from "
        f"earthquakes with magnitude < 5.0 were removed (waveforms were "
        f"available for {g('n_records')} of the remaining records). "
        f"dtheta is not in the flatfile, so we computed the RotD100 "
        f"orientation of each record from its waveforms and paired all "
        f"stations that recorded the same earthquake and are within "
        f"{DIST} km of each other, giving {g('n_pairs')} examples from "
        f"{g('n_events')} earthquakes. Features include earthquake source "
        f"data (magnitude, mechanism, rupture size, depth), distance "
        f"measures, site properties (Vs30, NEHRP class), ground-motion "
        f"intensities (PGA, PGV, PGD) and the distance between the two "
        f"sites. Numeric features were standardised (zero mean, unit "
        f"variance) and categorical ones one-hot encoded. Split: 80% train, "
        f"10% validation, 10% test ({g('n_train')} / {g('n_val')} / "
        f"{g('n_test')}). {LIMIT}",
        body,
    ),

    Paragraph(
        "3. Approach",
        h,
    ),

    Paragraph(
        "<b>Benchmark:</b> linear regression dtheta = c1 + c2*d using "
        "only the inter-site distance. "
        "<b>Ridge regression:</b> least squares with an L2 penalty on the "
        "coefficients; features were chosen by a backward analysis "
        "(rank each feature by the drop in validation R2 when it is removed) "
        "followed by a forward analysis (add features in that order until "
        "validation R2 stops improving), alternating with tuning of the "
        "regularisation parameter alpha on the validation set. "
        "<b>Neural network:</b> a multilayer perceptron (ReLU, Adam) using "
        "all features, with a grid search over number of layers, neurons "
        "per layer and L2 strength, selected by validation R2. Models are "
        "compared by R2 and mean absolute error (MAE) on the test set.",
        body,
    ),

    Paragraph(
        "4. Implementation Overview",
        h,
    ),

    Paragraph(
        "Python with NumPy, pandas, SciPy and scikit-learn. "
        "<i>waveform.py</i> reads the .AT2 files and finds the RotD100 "
        "orientation by passing both horizontal components through a damped "
        "single-degree-of-freedom oscillator and rotating the response "
        "through 0-179 degrees; <i>pairs.py</i> builds the site pairs and "
        "dtheta; <i>modeling.py</i> contains the split, preprocessing, "
        "feature selection, ridge model and neural network; "
        "<i>run_pipeline.py</i> runs everything and saves metrics, figures "
        "and models; <i>demo.py</i> predicts dtheta for two chosen sites. "
        "Setup and run steps are in the repository README.",
        body,
    ),

    Paragraph(
        "5. Results",
        h,
    ),
]


rows = [
    ["Model", "R2 (train)", "R2 (test)", "MAE (test, deg)"]
]

if have:
    for _, r in m.iterrows():
        rows.append([
            r["Model"],
            f"{r.R2_train:.3f}",
            f"{r.R2_test:.3f}",
            f"{r.MAE_test:.1f}",
        ])
else:
    for n in ("Benchmark", "Ridge regression", "Neural network"):
        rows.append([
            n,
            B,
            B,
            B,
        ])


t = Table(
    rows,
    colWidths=[
        4.5 * cm,
        3 * cm,
        3 * cm,
        3.5 * cm,
    ],
)

t.setStyle(
    TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ])
)

story += [
    t,
    Spacer(1, 4),
]


figs = [
    f"{RES}/fig_ridge_trend_distance.png",
    f"{RES}/fig_neural_true_vs_pred.png",
]

if have and all(os.path.exists(x) for x in figs):
    story.append(
        Table([
            [
                Image(figs[0], 6.3 * cm, 4.4 * cm),
                Image(figs[1], 9.6 * cm, 4.5 * cm),
            ]
        ])
    )


story.append(
    Paragraph(
        "6. Conclusions",
        h,
    )
)


if have:
    best = m.loc[m["R2_test"].idxmax()]
    nn = m[m.Model == "Neural network"].iloc[0]

    story.append(
        Paragraph(
            f"The {best['Model'].lower()} had the highest test score "
            f"(R2 = {best.R2_test:.3f}, MAE = {best.MAE_test:.1f} degrees), "
            f"compared with R2 = {m.R2_test.iloc[0]:.3f} for the "
            f"distance-only benchmark. The neural network reached "
            f"R2 = {nn.R2_train:.3f} on training data and "
            f"{nn.R2_test:.3f} on test data. "
            + CONCLUSION_EXTRA,
            body,
        )
    )

else:
    story.append(
        Paragraph(
            "[Fill after running the pipeline: which model performed best, "
            "whether the models capture the mean trend of dtheta with "
            "distance and PGA, overfitting, remaining uncertainty and "
            "future work.]",
            body,
        )
    )


story += [
    Paragraph(
        "7. Contributions",
        h,
    ),

    Paragraph(
        CONTRIBUTIONS,
        body,
    ),

    Paragraph(
        "<i>Reference: Poulos, A. and Silva-Lopez, R., Detecting and "
        "predicting earthquake ground motion directionality patterns using "
        "machine learning tools (Stanford); Ancheta et al. (2014), "
        "NGA-West2 database, Earthquake Spectra.</i>",
        body,
    ),
]


doc = SimpleDocTemplate(
    OUT,
    pagesize=A4,
    leftMargin=1.6 * cm,
    rightMargin=1.6 * cm,
    topMargin=1.3 * cm,
    bottomMargin=1.2 * cm,
    title="ML Mini-Project Write-up",
    author="Deeksha G, Harshitha B",
)

doc.build(story)

print("wrote", OUT)