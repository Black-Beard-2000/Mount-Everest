# Everest Summit Prediction — Model Audit & Correction

A logistic regression model predicting whether a climber will successfully summit Mount Everest, built on a synthetic dataset of 45,000 climbing attempts.

**Final result:** 71.4% accuracy / 0.769 AUC, vs. a 62% majority-class baseline.

This project isn't just "build a model" — it's a record of finding and fixing five separate data/pipeline issues across iterative review, several of which were silently inflating or invalidating the original results.

## Issues found and corrected

1. **Target leakage** — `turnaround_reason` (a column describing why an attempt ended) included `"summited"` as one of its categories, which corresponded 1:1 with the target variable. Including it in the feature set inflated AUC to 0.82; dropping it brought results down to the honest 0.769.
2. **Train/test contamination** — the feature scaler was originally fit on the full dataset before the train/test split, leaking test-set statistics into training. Fixed by fitting the scaler on the training set only.
3. **Improper encoding** — nominal categorical variables (`season`, `route`, `operator_tier`, `sex`) were originally ordinal-encoded, imposing a false numeric order on unordered categories. Replaced with one-hot encoding.
4. **Multicollinearity** — `uses_oxygen`/`o2_start_altitude_m` (r = 0.98) and `weather_score`/`jetstream_risk` (r = -0.85) were near-duplicate pairs. One feature from each pair was dropped.
5. **A downstream bug in feature-selection validation** — a "reduced 3-feature" model built via RFE was silently absorbing 11 extra columns due to a copy-paste error in a later scaling step, making its reported accuracy meaningless until caught and fixed.

## Files

- `mount_everest.ipynb` — full exploratory analysis, model build, and evaluation, with the debugging narrative preserved in markdown cells.
- `everest_logistic_regression.py` — clean, runnable script version of the final pipeline.
- `requirements.txt` — dependencies to run either file.
- `everest_summit_master.csv` — dataset.

## How to run

```bash
pip install -r requirements.txt
python everest_logistic_regression.py
```

## Results summary

| Metric | Value |
|---|---|
| Test accuracy | 0.714 |
| Test AUC | 0.769 |
| Majority-class baseline | 0.62 |
| 5-fold CV accuracy | 0.712 ± 0.003 |

Regularization (L2, C=0.1) and RFE-based feature reduction were both evaluated as part of the analysis; neither meaningfully changed performance, consistent with the base model not overfitting (train accuracy 0.713 ≈ test accuracy 0.714).
