# Predicting VM job durations for cloud server scheduling

How much is knowing a job's duration worth to an online scheduler, and how much of that can a learned predictor recover on real data?

In the cloud server renting problem, jobs (virtual machines) arrive online and are placed on servers immediately. Each server is paid for while it runs, and the goal is to minimise total rental time. The Greedy algorithm uses each job's duration at arrival, which real systems do not know. This project trains a model to predict durations from information available at arrival, feeds the predictions to Greedy, and measures the effect on a real Azure trace.

## Summary

- **A validated simulator.** Capacity and cost follow true job end times, and predictions only rank servers, so a wrong prediction can never overload a server (checked by tests). On the full trace it reproduces the published First Fit result (1.2997 against 1.30) and Greedy to within 0.02 (1.2004 against 1.22).
- **A predictor that halves the error, mostly on short jobs.** Test MAE on the log scale falls from 1.29 (median baseline) to 0.67, and R2 rises from -0.09 to 0.66. On the 107 long test jobs it does no better than the baseline.
- **Duration information is worth a lot, and the predictor recovers part of it.** Exact durations cut Greedy's cost by 2.1% (validation days) and 18.1% (test days) compared with a constant guess. The model's predictions recover about 80% of that gap on the validation days and about 27% on the test days.
- **The bottleneck is long jobs.** They decide how long servers stay open, and they are the jobs the model cannot recognise.

## Data

The Azure VM trace from the Protean paper (Hadary et al., OSDI 2020): machine 4, priority 0, 94,361 jobs submitted over 14 days. All jobs are kept, including very short ones. The data is not in this repository. Place the file at `data/raw/Azure_Data2.csv`.

Jobs are split by submission day, with no shuffling: train days 0 to 9 (67,675 jobs), validation days 10 to 11 (12,946 jobs), test days 12 to 13 (13,740 jobs).

## Method

- **Predictor.** Gradient Boosting (scikit-learn) on log(1 + duration in minutes). Features known at submission: the four resource sizes (core, memory, ssd, nic) and the hour of day as sine and cosine. The settings come from a random search of 10 configurations scored on the validation days. The baseline predicts the training median for every job.
- **Schedulers.** First Fit; Greedy as implemented in the C++ experiments of the related paper (place the job on the active server it would extend the least, ties to the server opened first); and Greedy driven by predicted durations.
- **Metric.** Cost ratio: total server rental time divided by a lower bound on the optimal cost (the bound used in the related paper). Sizes are whole numbers with server capacity 1000, as in the paper.

## Results

Prediction quality on the test days (13,740 jobs):

| | Median baseline | Gradient Boosting |
|---|---|---|
| MAE, log scale | 1.2906 | 0.6666 |
| RMSE, log scale | 1.8700 | 1.0384 |
| R2, log scale | -0.0932 | 0.6629 |
| Mean absolute error (minutes) | 135.3 | 122.8 |
| Median absolute error (minutes) | 47.7 | 26.3 |
| MAE, short jobs under 30 min (4,139 jobs) | 57.5 | 24.7 |
| MAE, medium jobs 30 min to 12 h (9,494 jobs) | 67.4 | 64.0 |
| MAE, long jobs over 12 h (107 jobs) | 9,169.5 | 9,136.5 |

Scheduling cost ratio (total rental time divided by the lower bound; lower is better):

| Scheduler | Validation (days 10 to 11) | Test (days 12 to 13) |
|---|---|---|
| First Fit | 1.2839 | 1.4911 |
| Greedy, true durations | 1.3036 | 1.3357 |
| Greedy, model durations | 1.3092 | 1.5512 |
| Greedy, median duration | 1.3320 | 1.6317 |

## What the results say

- With exact durations, Greedy is 10% cheaper than First Fit on the test days (cost 666.3 against 743.8), while First Fit wins on the validation days. The ranking of the two algorithms depends on the window.
- With the model's durations, Greedy is 4.9% cheaper than Greedy with the median on the test days, and 4.0% more expensive than First Fit.
- The model's gain is concentrated on short jobs. A likely reason for the weaker test result is the long jobs, since the test days have twice as many as the validation days (107 against 50). This is a hypothesis that has not been tested yet.

## Scope and limits

- Each scheduling number is one run on a two-day window with no confidence interval, and the two windows disagree. The results describe this trace and these windows, not a general ranking of the algorithms.
- Every window starts with empty servers, because jobs still running from earlier days are not simulated.
- Some training jobs end after the split day, so their durations would not yet have been known.
- Greedy is reproduced to within 0.02 of the published number, not exactly.
- The ratio is against a lower bound on the optimum, not the true optimum.
- The test days were used once for the final scores. The median baseline and the whole-trace simulator check, which use no model and no tuning, were also run on them.
- The data is one machine pool (machine 4, priority 0).

## Next steps

- Several windows with bootstrap confidence intervals, to separate the effects from window-to-window variation.
- A prediction-error sweep: Greedy with true durations plus controlled noise, to measure how accurate a predictor must be.
- Short/long labels with the weakly clairvoyant FirstFit-FirstFit algorithm from the related paper.
- A predictor aimed at long jobs, and a scheduler that uses its uncertainty.

## Running it

Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m scripts.prepare_data                  # builds data/processed/jobs.csv
python -m scripts.run_median_baseline           # median baseline
python -m scripts.train_model                   # search on validation, saves the model
python -m scripts.evaluate_model validation     # prediction scores (use "test" for the final run)
python -m scripts.run_scheduling validation     # scheduling comparison (use "test" for the final run)
python -m pytest -q                             # tests
```

Results are saved as JSON files in `results/`. Seeds are fixed (42).

## Project layout

```
scheduler/    Job, Server, First Fit, Greedy, Greedy with predicted durations
ml/           feature construction
evaluation/   lower bound and prediction metrics
scripts/      data preparation, training, evaluation, scheduling runs
tests/        28 tests, including checks that no server is ever overloaded
results/      saved results (JSON)
```

## Related work

- Y. Li, M. Masoori, L. Narayanan, D. Pankratov. Renting Servers for Multi-Parameter Jobs in the Cloud. ICDCN 2025.
- M. Masoori, L. Narayanan, D. Pankratov. Renting Servers in the Cloud: Empirical Study on Real-World Data. ALGOCLOUD 2025.
- O. Hadary et al. Protean: VM Allocation Service at Scale. OSDI 2020.