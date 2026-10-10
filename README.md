# Predicting Earthquake Ground-Motion Directionality (Δθ) with Machine Learning

UE24CS352A – Machine Learning · Mini-Project

**Team:** Deeksha G (PES1UG24AM078) · Harshitha B (PES1UG24AM118)

Based on: *Detecting and predicting earthquake ground motion directionality patterns using machine learning tools* (Poulos & Silva-Lopez, Stanford).

## Problem
Shaking at a site is strongest along one horizontal orientation (RotD100). Given two nearby sites and an earthquake, predict the absolute difference Δθ (0–90°) between their orientations: Δθ = f(site_i, site_j, earthquake). Three models are compared: a distance-only benchmark, ridge regression (backward/forward feature selection) and a neural network (MLP).

## Results (final run, test set)
| Model | R² train | R² test | MAE test |
|---|---|---|---|
| Benchmark (distance only) | 0.005 | 0.002 | 23.0° |
| Ridge regression | 0.005 | 0.002 | 23.0° |
| Neural network | 0.084 | 0.014 | 22.9° |

Data: 198 records with waveforms → 2,010 site pairs from 17 earthquakes (1,608 / 201 / 201 train/val/test).
Mean Δθ rises with distance (about 37° below 10 km, about 47° at 50–100 km), but no model beats a constant prediction.
Metrics, figures and the run summary are in `results_final/`. The full write-up is `Report_Earthquake_Directionality.pdf`.

**Limitation:** PEER limits waveform downloads to about 200 records per two weeks. Only 20 pairs were ≤ 5 km apart (the paper's limit), so pairs up to 100 km were used (`MAX_PAIR_DIST_KM` in `src/config.py`). Results are therefore not comparable with the paper's.

## Project layout
```
run_pipeline.py     whole pipeline (data -> pairs -> features -> models -> metrics -> plots)
demo.py             predict Δθ for two chosen sites of one earthquake
make_report.py      builds the PDF write-up from results/
check_columns.py    converts the NGA-West2 Excel flatfile to CSV and checks column names
check_data.py, check_downloads.py, select_rsn_subset.py   data inspection helpers
src/config.py       flatfile column names and settings (distance limit, magnitude, ...)
src/waveform.py     .AT2 reader, oscillator response, RotD100 orientation
src/pairs.py        orientations, site pairing, Δθ
src/modeling.py     split, scaling/one-hot, feature selection, benchmark, ridge, MLP
src/plots.py        figures
src/synthetic.py    fake data, ONLY to test that the code runs
```

## Setup
```bash
pip install -r requirements.txt     # tested with Python 3.13
```

## Data (not included: licence and size)
1. Download the NGA-West2 flatfile from https://peer.berkeley.edu/research/databases/databases and convert it to `data/NGAWest2_flatfile.csv` (`python check_columns.py` does this for the Excel file).
2. Register at https://ngawest2.berkeley.edu and download the time series (`.AT2` files) into `data/timeseries/`.
3. Component directions are read from the file names (e.g. `...180`, `...NS`/`...EW`), so all sites share one reference frame.

## Run
```bash
# real run (settings used for the reported results)
python run_pipeline.py --flatfile data/NGAWest2_flatfile.csv --ts-dir data/timeseries --layers 1 2 --neurons 10 20 --nn-alphas 1 10 --max-iter 1000

# demo: predict Δθ for two sites of one earthquake
python demo.py --artifacts results/model_artifacts.joblib --random

# PDF write-up with the real numbers
python make_report.py

# test that the code runs (fake data, meaningless results)
python run_pipeline.py --synthetic --layers 2 --neurons 30 --max-iter 100
```

## Method summary
1. Keep records with magnitude ≥ 5.0.
2. RotD100 orientation per record: both horizontal waveforms through a damped oscillator (T = 1 s, 5 % damping), scan 0–179°.
3. Pair stations that recorded the same earthquake (≤ 100 km here); Δθ = |θi − θj| folded to [0°, 90°].
4. Pair features: earthquake attributes, mean and absolute difference of site attributes, site-class pair, distance. Numeric → standardised, categorical → one-hot.
5. 80 / 10 / 10 split. Benchmark Δθ = c1 + c2·d; ridge with backward ranking → forward selection and α tuning; MLP (ReLU, Adam) with grid search.
6. Test-set metrics: R² and mean absolute error.

## Deviations from the paper
- Distance limit 100 km instead of 5 km (download limit, see above).
- Feature selection uses the validation set (the paper describes the test set).
- Symmetric pair features so predictions do not depend on site order.
- Small MLP grid (small dataset); the oscillator period is not given in the paper and is configurable (`PSA_PERIOD_S`).
