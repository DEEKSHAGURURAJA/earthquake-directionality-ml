"""Build the PDF write-up. Uses results/ if present, otherwise leaves blanks to fill."""
import json, os, sys
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = "Writeup_Earthquake_Directionality.pdf"
REPO = "https://github.com/DEEKSHAGURURAJA/earthquake-directionality-ml."   # <- edit

S = getSampleStyleSheet()
body = ParagraphStyle("b", parent=S["Normal"], fontSize=8.8, leading=11, alignment=4)
h = ParagraphStyle("h", parent=S["Heading3"], fontSize=10, spaceBefore=5, spaceAfter=2)
title = ParagraphStyle("t", parent=S["Title"], fontSize=14, leading=17, spaceAfter=3)
ctr = ParagraphStyle("c", parent=body, alignment=1)

have = os.path.exists(f"{RES}/metrics.csv") and os.path.exists(f"{RES}/run_summary.json")
m = pd.read_csv(f"{RES}/metrics.csv") if have else None
info = json.load(open(f"{RES}/run_summary.json")) if have else {}
B = "____"
g = lambda k: f"{info[k]:,}" if k in info else B

story = [
    Paragraph("Predicting Earthquake Ground-Motion Directionality Differences between Nearby Sites using Machine Learning", title),
    Paragraph("UE24CS352A – Machine Learning · Mini-Project", ctr),
    Paragraph("<b>Deeksha G</b> (PES1UG24AM078) &nbsp;&nbsp;|&nbsp;&nbsp; <b>Harshitha B</b> (PES1UG24AM118)", ctr),
    Paragraph(f"Code: {REPO}", ctr),
    Paragraph("1. Problem Statement", h),
    Paragraph("During an earthquake, shaking at a site is strongest along one horizontal orientation (RotD100). "
              "Building codes treat this orientation as uniformly random, but observations show that nearby sites have "
              "correlated orientations. Our task (assigned problem, based on Poulos &amp; Silva-Lopez) is to build a model "
              "<i>f</i> that takes the seismological, geological and geographical attributes of two sites and of the earthquake, "
              "and predicts the absolute difference &Delta;&theta; (0&ndash;90&deg;) between their orientations of maximum shaking. "
              "We compare a distance-only benchmark, ridge regression and a neural network.", body),
    Paragraph("2. Dataset", h),
    Paragraph(f"We use the NGA-West2 database (PEER): a flatfile with 274 attributes per record plus the two-component acceleration "
              f"time series of shallow crustal earthquakes worldwide. Records from earthquakes with magnitude &lt; 5.0 were removed "
              f"({g('n_records')} records kept). &Delta;&theta; is not in the flatfile, so we computed the RotD100 orientation of each record "
              f"from its waveforms and paired all stations that recorded the same earthquake and are within 5&nbsp;km of each other, "
              f"giving {g('n_pairs')} examples from {g('n_events')} earthquakes. Features include earthquake source data (magnitude, "
              f"mechanism, rupture size, depth), distance measures, site properties (V<sub>s30</sub>, NEHRP class), ground-motion "
              f"intensities (PGA, PGV, PGD) and the distance between the two sites. Numeric features were standardised "
              f"(zero mean, unit variance) and categorical ones one-hot encoded. Split: 80% train, 10% validation, 10% test "
              f"({g('n_train')} / {g('n_val')} / {g('n_test')}).", body),
    Paragraph("3. Approach", h),
    Paragraph("<b>Benchmark:</b> linear regression &Delta;&theta; = c<sub>1</sub> + c<sub>2</sub>d using only the inter-site distance. "
              "<b>Ridge regression:</b> least squares with an L2 penalty on the coefficients; features were chosen by a backward analysis "
              "(rank each feature by the drop in validation R<super>2</super> when it is removed) followed by a forward analysis (add features in "
              "that order until validation R<super>2</super> stops improving), alternating with tuning of the regularisation parameter &alpha; on the "
              "validation set. <b>Neural network:</b> a multilayer perceptron (ReLU, Adam) using all features, with a grid search over "
              "number of layers, neurons per layer and L2 strength, selected by validation R<super>2</super>. Models are compared by R<super>2</super> and mean "
              "absolute error (MAE) on the test set.", body),
    Paragraph("4. Implementation Overview", h),
    Paragraph("Python with NumPy, pandas, SciPy and scikit-learn. <i>waveform.py</i> reads the .AT2 files and finds the RotD100 orientation "
              "by passing both horizontal components through a damped single-degree-of-freedom oscillator and rotating the response "
              "through 0&ndash;179&deg;; <i>pairs.py</i> builds the site pairs and &Delta;&theta;; <i>modeling.py</i> contains the split, preprocessing, feature "
              "selection, ridge model and neural network; <i>run_pipeline.py</i> runs everything and saves metrics, figures and models; "
              "<i>demo.py</i> predicts &Delta;&theta; for two chosen sites. Setup and run steps are in the repository README.", body),
    Paragraph("5. Results", h),
]

rows = [["Model", "R² (train)", "R² (test)", "MAE (test, deg)"]]
if have:
    for _, r in m.iterrows():
        rows.append([r["Model"], f"{r.R2_train:.3f}", f"{r.R2_test:.3f}", f"{r.MAE_test:.1f}"])
else:
    for n in ("Benchmark", "Ridge regression", "Neural network"):
        rows.append([n, B, B, B])
t = Table(rows, colWidths=[4.5 * cm, 3 * cm, 3 * cm, 3.5 * cm])
t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                       ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("ALIGN", (1, 0), (-1, -1), "CENTER")]))
story += [t, Spacer(1, 4)]
for f in ("fig_neural_trend_distance.png", "fig_neural_true_vs_pred.png"):
    pass
figs = [f"{RES}/fig_ridge_trend_distance.png", f"{RES}/fig_neural_true_vs_pred.png"]
if have and all(os.path.exists(x) for x in figs):
    story.append(Table([[Image(figs[0], 6.3 * cm, 4.4 * cm), Image(figs[1], 9.6 * cm, 4.5 * cm)]]))

story.append(Paragraph("6. Conclusions", h))
if have:
    best = m.loc[m["R2_test"].idxmax()]
    nn = m[m.Model == "Neural network"].iloc[0]
    story.append(Paragraph(
        f"The {best['Model'].lower()} gave the best test performance (R<super>2</super> = {best.R2_test:.3f}, MAE = {best.MAE_test:.1f}&deg;), "
        f"compared with R<super>2</super> = {m.R2_test.iloc[0]:.3f} for the distance-only benchmark. The neural network reached R<super>2</super> = {nn.R2_train:.3f} "
        f"on training data and {nn.R2_test:.3f} on test data. "
        "[Add 2&ndash;3 sentences in your own words: Do the models capture the mean trend of &Delta;&theta; versus distance and PGA (see figures)? "
        "Is there overfitting? What uncertainty remains? What would you do next, e.g. predict mean and variance, or more than two sites?]", body))
else:
    story.append(Paragraph("[Fill after running the pipeline: which model performed best, whether the models capture the mean trend of "
                           "&Delta;&theta; with distance and PGA, overfitting, remaining uncertainty and future work.]", body))
story += [Paragraph("7. Contributions", h),
          Paragraph("[Describe who did what, e.g. Deeksha: ____ ; Harshitha: ____ .]", body),
          Paragraph("<i>Reference: Poulos, A. and Silva-Lopez, R., Detecting and predicting earthquake ground motion directionality patterns "
                    "using machine learning tools (Stanford); Ancheta et al. (2014), NGA-West2 database, Earthquake Spectra.</i>", body)]

SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.3 * cm, bottomMargin=1.2 * cm,
                  title="ML Mini-Project Write-up", author="Deeksha G, Harshitha B").build(story)
print("wrote", OUT)
