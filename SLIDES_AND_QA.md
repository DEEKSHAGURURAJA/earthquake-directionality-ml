# Slide outline (8–10 slides) and likely Q&A

1. **Title** – project title, UE24CS352A, Deeksha G (PES1UG24AM078), Harshitha B (PES1UG24AM118)
2. **Problem** – what is directionality / RotD100; why nearby sites correlate; goal: predict Δθ for two sites (use Fig. 1 idea: arrows on a map)
3. **Dataset** – NGA-West2: flatfile + waveforms; M ≥ 5 filter; counts (records, pairs, earthquakes) from `run_summary.json`
4. **Data pipeline** – waveforms → oscillator response → RotD100 angle per record → pair stations ≤ 5 km → Δθ (draw as a flow diagram)
5. **Features** – earthquake, distance, site, intensity features; mean/diff of site features; scaling + one-hot
6. **Models** – benchmark (Δθ = c1 + c2·d), ridge (+ backward/forward selection, α tuning), MLP (grid search)
7. **Results** – metrics table + `fig_*_trend_distance.png`, `fig_*_trend_pga.png`, `fig_neural_true_vs_pred.png`
8. **Discussion** – mean trend captured, low R² because of inherent scatter, overfitting of the NN
9. **Live demo** – `python demo.py --artifacts results/model_artifacts.joblib --random`
10. **Conclusion & future work** – predict mean + variance; more than two sites; team contributions

## Questions the panel may ask
- *What is Δθ and why is it folded to 0–90°?* Orientations repeat every 180°, so the largest possible difference is 90°.
- *Why does a 2-station-only model start from a distance benchmark?* It shows how much the extra features add.
- *Why ridge? What does α do?* L2 penalty shrinks coefficients to reduce overfitting; α picked on validation data.
- *Why one-hot encoding / scaling?* Categorical features have no order; scaling makes the penalty fair across features.
- *Why is test R² much lower than train R² for the NN?* Overfitting + heavy inherent noise in Δθ.
- *Why validation set for feature selection?* Keeps the test set untouched (the paper used the test set).
- *Could pairs from the same earthquake leak between train and test?* Yes with a random split – `--by-event-split` is the stricter option.
- *How is the RotD100 angle found?* Rotate the two components through 0–179°, take the angle with the largest oscillator response; linearity means only two oscillator runs are needed.
- *Why only pairs within 5 km?* The use case is city-scale analysis; beyond 5 km orientations look uncorrelated.
- *Why is R² low overall and is that bad?* The data has large irreducible scatter; the models still follow the mean trend.
