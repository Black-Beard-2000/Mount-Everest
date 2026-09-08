# Model Comparison Report: Baseline vs. L2-Regularized vs. RFE-Reduced Model

**Task:** Binary classification (0 = negative class, 1 = positive class)
**Test set:** 9,000 samples — 5,557 in class 0 (61.7%), 3,443 in class 1 (38.3%)
**Models compared:** three variants of a logistic regression model, evaluated on the same test set

I ran three versions of the model — a baseline with no enhancements, a version with L2 regularization, and a version with features reduced via RFE — to see which approach actually improves performance.

---

## Summary Table

| Metric | Baseline (No Enhancements) | L2-Regularized | RFE |
|---|---|---|---|
| Accuracy | 0.71 | 0.71 | 0.68 |
| Class 0 Precision | 0.74 | 0.74 | 0.70 |
| Class 0 Recall | 0.83 | 0.83 | 0.85 |
| Class 0 F1 | 0.78 | 0.78 | 0.77 |
| Class 1 Precision | 0.66 | 0.66 | 0.63 |
| Class 1 Recall | 0.53 | 0.52 | 0.42 |
| Class 1 F1 | 0.58 | 0.58 | 0.50 |
| Macro F1 | 0.68 | 0.68 | 0.64 |
| Weighted F1 | 0.71 | 0.71 | 0.67 |

---

## Model-by-Model Analysis

### 1. Baseline Model (No Enhancements)
Accuracy came out to 0.71, macro F1 0.68. The model favors the majority class (0): recall is 0.83 on class 0 vs. only 0.53 on class 1. It misses roughly half of all true class-1 cases. I'm treating this as my reference point for everything else.

### 2. L2-Regularized Model
This is virtually indistinguishable from the baseline. Every metric is identical except class-1 recall, which drops by a single point (0.53 → 0.52), and class-1 F1 stays flat at 0.58 despite that drop (likely just a rounding artifact rather than a real change). **This regularization step didn't achieve anything meaningful on this test set.** If my intent was to improve generalization, reduce overfitting, or lift class-1 performance, it didn't happen here. It's possible this model still helps if I compare train vs. validation performance (i.e., it could be reducing overfitting in a way this single test report doesn't show) — but on held-out performance alone, there's no measurable benefit.

### 3. RFE (Recursive Feature Elimination) Model
This model is **worse across every single metric**, not just on the minority class:
- Accuracy drops from 0.71 to 0.68
- Class 1 recall collapses from ~0.53 to 0.42 — the model now misses **58% of all actual positive cases**, up from ~47-48% in the other two models
- Class 1 F1 drops from 0.58 to 0.50
- Even class 0 recall, its one improving metric (0.83 → 0.85), comes at the cost of everything else — this is consistent with a model that has become more biased toward predicting the majority class, not a genuinely better model

Reducing features via RFE evidently stripped out signal that mattered specifically for identifying class 1, without a compensating gain anywhere else. This wasn't a good trade.

---

## Overall Verdict

None of my three models solve the actual problem, which is **weak detection of class 1**. Ranked by macro F1 (the fairest single-number summary given the class imbalance):

1. **Baseline** and **L2-regularized model** — tied at 0.68, effectively identical
2. **RFE model** — 0.64, meaningfully worse than the other two

**The regularization step wasn't useful** — it changed nothing of substance. **The RFE step actively hurt performance** — I shouldn't adopt it based on these results. If the goal was dimensionality reduction for interpretability or speed, that's a separate tradeoff I'd need to weigh consciously, but it shouldn't be framed as a model-quality improvement, because it isn't one.

None of these three variants meaningfully address the core weakness (poor class-1 recall). If class 1 is the class that matters most to detect, none of this experimentation moved the needle — I need to look into class-imbalance handling (class weighting, resampling, threshold tuning) or a different feature set entirely, rather than iterating further on regularization strength or RFE feature count.

---

## Real-World Applicability

While this model is weak at identifying successful summits, it's consistently stronger at identifying failed attempts, and that asymmetry is worth using deliberately rather than discarding.

**What the numbers actually say about failure detection:**
- Baseline / L2 models: precision 0.74, recall 0.83 on class 0 — when the model flags an attempt as likely to fail, it's right 74% of the time, and it catches 83% of all attempts that do fail.
- RFE model: precision 0.70, recall 0.85 — slightly lower precision but higher recall; it flags a few more real failures at the cost of a few more false alarms.

Both are meaningfully better than the class-1 numbers, and better than chance given the ~62% base rate of failure in this dataset.

**Where this could be genuinely useful:**
- **Risk screening, not outcome prediction.** The model shouldn't be marketed as "predicts who will summit" — it should be positioned as "flags attempts at elevated risk of failure," which is a real and useful distinction. High recall on class 0 (83–85%) means it's good at *not missing* risky attempts, which matters more than precision in a screening context.
- **Pre-expedition risk triage.** Expedition operators or guide services could use this as one input (not the sole input) to flag climbers or planned attempts that share characteristics with historical failures — prompting closer review of oxygen plans, acclimatization schedules, weather windows, or guide-to-climber ratios before departure.
- **Resource and support allocation.** If an operator has limited high-altitude support staff, supplemental oxygen, or rescue capacity, flagging higher-risk attempts (with 83–85% recall) allows resources to be weighted toward those attempts rather than distributed evenly.
- **Insurance or permit-review contexts.** A model that reliably flags likely-failure profiles (with ~74% precision) could support human decision-making in permit review or insurance risk assessment — as a supporting signal, not an automated denial mechanism, given that a quarter of its failure flags are still wrong.

**Where it should not be used, and why:**
- **Not for denying permits or attempts outright.** 74–70% precision means roughly 1 in 4 to 3 in 10 "predicted failure" flags are wrong — a real climber capable of summiting would be misclassified. Using this as a hard gate rather than a decision-support input would unfairly block real attempts.
- **Not as the only input to any consequential decision.** Recall of 83–85% is good but not near-certain — 15–17% of actual failures still slip through undetected, so it can't be relied on as a safety guarantee.
- **Not for predicting individual success.** Given the earlier analysis, the model should never be used to tell a specific climber "you will summit" — that's exactly the class it performs worst on (53% recall, meaning it misses roughly half of actual successes).

**Bottom line:** this model has a real, usable skill — flagging likely-failure attempts with decent precision and good recall — but it's a risk-screening tool, not a prediction engine, and it should stay a supporting input alongside human judgment (weather data, climber experience, physical readiness, guide assessment), not a standalone decision-maker.