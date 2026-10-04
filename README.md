# Predicting Earthquake Ground-Motion Directionality (Δθ) with Machine Learning

UE24CS352A – Machine Learning · Mini-Project

**Team:** Deeksha G (PES1UG24AM078) · Harshitha B (PES1UG24AM118)

Based on: *Detecting and predicting earthquake ground motion directionality patterns using machine learning tools* (Poulos & Silva-Lopez, Stanford).

## Problem
At a recording site, ground shaking is strongest in one horizontal direction (the RotD100 orientation).
Given two nearby sites and an earthquake, predict the absolute difference Δθ (0–90°) between
their orientations: `Δθ = f(site_i, site_j, earthquake)`.
Three models are compared: a distance-only linear **benchmark**, **ridge regression** (with
backward/forward feature selection) and a **neural network** (MLP).

## Project layout
```
run_pipeline.py     whole pipeline (data -> pairs -> features -> models -> metrics -> plots)
demo.py             live demo: predict Δθ for two chosen sites
make_report.py      builds the PDF write-up from results/ (fills in the real numbers)
src/config.py       flatfile column names + settings  (EDIT to match your flatfile header)
src/waveform.py     .AT2 reader, oscillator response, RotD100 orientation
src/pairs.py        orientations for all records, site pairing (<= 5 km), Δθ
src/modeling.py     split, scaling/one-hot, feature selection, benchmark, ridge, MLP grid search
src/plots.py        figures
src/synthetic.py    fake data, ONLY to test that the code runs
```

## Setup
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data (NGA-West2)
1. Register (free) at https://ngawest2.berkeley.edu and download the **flatfile** (CSV) and the
   **time series** (`.AT2` acceleration files; two horizontal components per record).
2. Open the flatfile header and make sure the names in `src/config.py` match it
   (add/remove feature columns as you like – the paper kept 71 attributes).
3. `H1_AZIMUTH` in `config.py` must point to the column giving the compass azimuth of the
   horizontal-1 component, so orientations from different stations share one reference frame.

## Run
```bash
# 1) quick check that the code works (fake data - results are meaningless)
python run_pipeline.py --synthetic --layers 2 --neurons 30 --max-iter 100

# 2) real run
python run_pipeline.py --flatfile data/NGAWest2_flatfile.csv --ts-dir data/timeseries

# optional: split by earthquake instead of by pair (stricter, no leakage between pairs)
python run_pipeline.py --flatfile ... --ts-dir ... --by-event-split

# paper-sized network search (slow): --layers 5 15 25 --neurons 100 200 400

# 3) demo
python demo.py --artifacts results/model_artifacts.joblib --random
python demo.py --artifacts results/model_artifacts.joblib --rsn1 <RSN> --rsn2 <RSN>

# 4) PDF write-up with real numbers
python make_report.py
```
Outputs go to `results/`: `metrics.csv`, `run_summary.json`, `pairs.csv`, `feature_importance.csv`,
`nn_gridsearch.csv`, figures (`fig_*.png`) and the saved models.

## Method summary
1. Keep records with magnitude ≥ 5.0.
2. For each record, find the orientation of maximum oscillator response (RotD100) from the two horizontal
   waveforms (damped SDOF oscillator, T = 1 s by default, rotated 0–179°).
3. Pair all stations that recorded the same earthquake and are ≤ 5 km apart; Δθ = |θi − θj| folded to [0°, 90°].
4. Pair features: earthquake attributes, mean and absolute difference of each site attribute, site-class pair,
   and inter-site distance. Numeric → zero mean / unit variance, categorical → one-hot.
5. 80 / 10 / 10 train / validation / test split.
6. Benchmark `Δθ = c1 + c2·d`; ridge with backward ranking → forward selection (stop when validation R² drops)
   alternating with α tuning; MLP (ReLU, Adam) on all features with grid search on layers, neurons, L2 α.
7. Metrics on the test set: R² and mean absolute error.

## Implementation notes / deviations from the paper (be ready to explain)
- Feature selection uses the **validation** set (the paper describes the test set), to keep the test set untouched.
- Symmetric pair features (mean / absolute difference) so the prediction doesn't depend on site order.
- The default MLP grid is small so it runs on a laptop; the paper's best network was 25 layers × 400 neurons.
- The oscillator period for RotD100 is not stated in the paper; it is configurable (`PSA_PERIOD_S`).
