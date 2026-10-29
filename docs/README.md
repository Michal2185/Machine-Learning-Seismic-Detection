# Continuous-Time Neural Networks for Extraterrestrial Seismic Detection

**Research question.** Can compact continuous-time neural networks (CfC, LTC) give a better
accuracy–resource–generalization trade-off than conventional onboard seismic detection (STA/LTA and
classical feature-based models) under extraterrestrial conditions?

**Hypothesis (to be tested, not assumed).** At an equal or smaller parameter budget, a CfC reaches detection
performance comparable to a CNN / GRU and degrades less on cross-mission and irregular-sampling tests.
STA/LTA is almost the cheapest possible detector, so a network has to win on accuracy at matched false alarms,
on detection delay, or on generalization, not on size.

**Current state (end of phase 1).** Data, evaluation, baselines, the shared model infrastructure (normalisation,
features, training harness, resource accounting, plots, waveform-bootstrap statistics) and the first models
are done and tested: amplitude controls, RFC, DNN, 1D CNN. In-domain (Apollo 12) the feature models and the CNN
beat the frozen STA/LTA with a fixed scale (N0), and the learned models beat a bare amplitude threshold on weak
and deep events. **Next: GRU and CfC** (see `docs/HANDOFF_GRU_CFC.md`), then stress tests and cross-mission.

---

## Pipeline

```
Apollo 12 catalogue + 6.625 Hz waveforms
   │  data branch: causal Butterworth highpass 0.2 Hz (order 4), interpolated-gap map, event zones
   │               (visible onset, coda end), 60 s windows (398 samples, step 10 s), labels 1 / 0 / -1,
   │               5 folds over waveforms
   ▼
DatasetBundle  ──  bundle.fold_split(k), bundle.eval_streams(ids)
   │
   │  src/modules/normalization    N0 fixed scale (per fold)   N1 per window
   │                               N2 causal adaptive scale (v3: dead-block rule + floor)   N3 N2 + asinh
   │  src/modules/features         6 features: energy, RMS, peak-to-peak, crest, dominant frequency, zero crossings
   ▼
Inputs  (RawWindowInput | FeatureInput)
   │
   │  src/modules/training         LearnerDetector: fresh weights per fold, Sampler draws new background windows
   │                               every epoch, early stopping on held-out TRAINING waveforms, resource report
   │  src/modules/models           amplitude control, RFC, DNN, CNN; next: GRU, CfC (LTC as ablation)
   ▼
Detector.score(windows, streams) -> one score per window (higher = event)
   │
   │  evaluation branch            cross_validate_budgets: threshold from validation at a false-alarm budget
   │                               (strict 0.25 / lenient 2 per day), test applied once, pooled over 5 folds
   ▼
EvalResult (per-event outcomes)  ──>  tests/run_dnn_features.py writes JSON incl. outcomes
   │
   │  src/modules/reporting + tests/analyze_results.py, visualize_*   waveform bootstrap (intervals, paired
   │                               differences), alarm timeline vs catalogue arrival, confusion matrices
   ▼
Tables, figures, CSV, handoff notes
```

**Protocol (applies to every detector).** Fit on Apollo 12 training / validation waveforms only, freeze, evaluate on
unseen waveforms (nested CV, 5 folds), later zero-shot. Thresholds come from validation under a false-alarm budget.
At least 5 seeds per configuration; every seed is its own `cross_validate_budgets` call (the function selects the best
candidate per fold, so seeds in one call would be selected, not averaged). Results carry 95 % intervals from resampling
WAVEFORMS (events and false alarms of one day are not independent); detectors are compared with a paired bootstrap on the
same resamples.

**Detection rule.** Alarm time = end of the window that triggered it; first score at or above the threshold, then
300 s of silence. An alarm in [catalogue − 120 s, catalogue + 600 s] detects the event; alarms in ignore intervals or
gaps are neutral; the rest are false alarms.

## Repository map

| Path | Content |
|---|---|
| `src/modules/pipeline`, `labeling`, `sampler`, `detectors/sta_lta.py`, `evaluation` | data and evaluation branches (see `docs/*HANDOFF*.md`) |
| `src/modules/normalization` | N0–N3, `Normalizer`, `ScaleCache` (causal N2 estimator; v3 needs the fold's reference level, set by `Normalizer.fit`) |
| `src/modules/features` | 6-feature extractor, `FeatureScaler` |
| `src/modules/training` | `LearnerDetector`, learners (numpy, torch), inputs, waveform-level split, resource counting |
| `src/modules/models` | `dnn.py`, `rfc.py`, `cnn.py`, `amplitude.py`, `report.py` |
| `src/modules/reporting` | alarm / arrival comparison, confusion metrics, waveform bootstrap (`stats.py`) |
| `tests/test_*.py` | unit tests on synthetic data; run with plain `python tests/<file>.py` (torch tests skip without torch) |
| `tests/run_dnn_features.py` | experiment runner: nested CV per configuration and seed, table + JSON with per-event outcomes |
| `tests/analyze_results.py` | intervals, event-group detail, paired comparisons, CSV and recall plot from the runner JSON |
| `tests/visualize_normalization.py`, `inspect_normalization.py`, `visualize_alarms.py` | figures and checks on the real bundle |
| `docs/` | `DATA_BRANCH_HANDOFF`, `EVALUATION_BRANCH_HANDOFF`, `NORMALIZATION_HANDOFF`, `HARNESS_HANDOFF`, `DNN_FEATURES_HANDOFF`, `CNN_CONTROLS_HANDOFF`, `STATS_HANDOFF`, `MODELS_PLAN`, **`HANDOFF_GRU_CFC`** |

Set the data paths once at the top of the `tests/` scripts that need the real bundle. Run scripts as modules from the repo root
(`python -m tests.<script>`).

## Results (Apollo 12, nested CV, 5 seeds, all 75 events; 95 % intervals over waveforms)

| Detector | Recall at 0.25 FA/day | FA/day | Recall at 2 FA/day | FA/day | Parameters | Model cost per window |
|---|---|---|---|---|---|---|
| STA/LTA (frozen, STA 120 s / LTA 86 400 s) | 0.67 [0.56, 0.76] | 0.30 | 0.81 [0.72, 0.89] | 0.43 | – | a few running sums |
| Amplitude control: RMS threshold, N0 | 0.68 [0.57, 0.79] | 0.31 | 0.97 [0.93, 1.00] | 0.87 | 1 threshold | 0 |
| Amplitude control: peak-to-peak, N0 | 0.64 [0.53, 0.74] | 0.43 | 0.96 [0.91, 1.00] | 1.05 | 1 threshold | 0 |
| DNN 1k, N0 | 0.76 [0.68, 0.84] | 0.12 | 0.98 [0.96, 0.99] | 0.57 | 1 037 | 980 MACs |
| DNN 10k, N0 | 0.82 [0.75, 0.89] | 0.20 | 0.99 [0.98, 1.00] | 0.72 | 10 081 | 9.9 k MACs |
| CNN 10k, N0 | 0.84 [0.78, 0.89] | 0.18 | 0.99 [0.97, 1.00] | 0.43 | 9 729 | 624 k MACs |
| CNN 1k, N0 | 0.85 [0.78, 0.91] | 0.14 | 0.99 [0.97, 1.00] | 0.53 | 1 025 | 77 k MACs |
| **RFC, N0** (100 trees, depth 8) | **0.91 [0.86, 0.96]** | 0.16 | 0.97 [0.93, 0.99] | **0.26** | 4 280 nodes (29 KiB int8 est.) | 733 comparisons |
| CNN 10k / 1k, N2 (v3) | 0.50 / 0.45 | 0.17 / 0.19 | 0.73 / 0.69 | 1.11 / 0.99 | 9 729 / 1 025 | 624 k / 77 k MACs |
| DNN 10k / 1k, N2 (v3) | 0.43 / 0.42 | 0.19 / 0.17 | 0.59 / 0.56 | 0.81 / 0.76 | 10 081 / 1 037 | |
| RFC, N2 (v3) | 0.30 [0.23, 0.38] | 0.23 | 0.73 [0.65, 0.80] | 1.76 | ~4 000 nodes | 800 comparisons |
| Amplitude control, N2 (v3), RMS / peak-to-peak | 0.27 / 0.25 | 0.09 / 0.14 | 0.52 / 0.53 | 0.93 / 1.20 | 1 threshold | 0 |

The feature front end (normalisation + 6 features) costs about 12 600 operations per window, more than the DNN / RFC models;
the amplitude control about 1 200. Full tables, event groups, delays and paired differences: `results/analysis.csv`,
`results/analysis_recall.png` (made by `tests/analyze_results.py` from `results/final.json`).

Event groups at the strict budget (recall): weak events (no usable onset) RFC 0.87, CNN-N0 0.73–0.74, DNN-N0 0.62–0.70, amplitude
control 0.54, STA/LTA 0.58; deep events RFC 0.87, CNN-N0 0.76–0.78, amplitude control 0.56, STA/LTA 4 of 9. Delay of the first alarm vs
the visible onset (median): N0 models 42–59 s, STA/LTA 112 s (strict) / 81 s (lenient).

### Findings

* **A bare RMS threshold on the fixed scale already matches the frozen STA/LTA** at the strict budget (0.68 vs 0.67) and exceeds it in
  recall at the lenient budget (0.97 vs 0.81, at more false alarms). STA/LTA's advantage is its cost, not its accuracy under this protocol.
* **Learned models add real signal beyond amplitude**: paired differences against the RMS threshold (strict) are RFC +0.23 [0.13, 0.34],
  CNN +0.16 to +0.17, DNN-10k +0.14 [0.04, 0.25] (intervals exclude 0), DNN-1k +0.08 [−0.03, 0.20] (does not); the gain is largest on weak
  events (RFC +0.33, CNN +0.19 to +0.20). The CNN is not better than the RFC (recall about 0.07 lower; the direct paired difference was not run: `--compare cnn-10k-N0 rfc-medium-N0`) and costs 80–600x the MACs of the MLP.
* **In-domain results are saturated at the lenient budget** (0.97–0.99 recall); models then differ in false alarms (RFC 0.26 vs amplitude
  control 0.87).
* **N0 beats N2 for every model** (+0.34 to +0.61 recall, all significant). The noise level is a plateau (about 2.5e-10) with frequent
  quiet dips, not a slowly varying background, so a causal adaptive scale follows the dips. N2 v3 (dead-block rule + scale floor) removed the
  collapse of v2 but stays far below N0 and is frozen as a documented baseline.
* **The feature models act largely as amplitude detectors**: peak-to-peak, RMS and energy carry about 89 % of the RFC importance (Reviewer 1,
  point 3). This works because the noise floor is steady across days; the stress tests are the test of that.

## Plan

**Done**
- [x] Data branch, evaluation branch, frozen STA/LTA baseline
- [x] Input normalisation N0–N3 (N2 v3), 6-feature extractor
- [x] Training harness, early stopping, resource accounting (parameters, int8 size, MACs, state)
- [x] `models/dnn-features`: DNN (1k / 10k / 100k presets) and RFC, 5-seed runner
- [x] Amplitude controls and 1D CNN
- [x] Waveform bootstrap, paired comparisons, event-group detail, alarm-vs-arrival and confusion plots

**Next (details in `docs/HANDOFF_GRU_CFC.md`)**
1. `models/gru-cfc`, step 1: GRU and CfC on the raw normalised window (variant W) at 1k / 10k parameters on N0 through the existing harness.
2. Variant E (12 sub-window envelope features per window, time step as CfC input), 1k / 10k / 100k.
3. Variant S (stateful stream, hidden state carried over hours, truncated backpropagation, `StreamDetector`), matched GRU control; LTC as ablation.
4. `feat/stress-tests`: gain scaling, SNR scaling by inserting events into noise, gaps and timestamp jitter, sampling-rate change; frozen thresholds.
5. `feat/downlink-metrics`: snippet policy, data sent / wasted / lost, pre-roll, merging of repeated alarms.
6. `feat/mars-insight`: zero-shot with frozen detectors on other Apollo stations and InSight (MQS catalogue).

**Open items**
- Paired comparison with STA/LTA (needs its per-event outcomes in the runner format; waiting for the `frozen.py` / `sta_lta.py` signatures).
- Decide: trim file edges (first / last minutes) from training negatives and the N0 fit; treatment of dead stretches in labels and false-alarm accounting.
- Tune the feature models and the CNN on validation folds only (all are deliberately untuned); rerun N1 / N3 with N2 v3 if wanted.
- Per-event-type and per-group results; sensitivity to the detect interval (+600 vs +900 s) and the refractory time.

## Known limitations

- Only 49 of 75 events carry positive windows (198 windows); the effective number of independent positives is about 49.
- Delay against the catalogue is capped by the detect interval (+10 min) and the 60 s window (alarm = window end); timing below about 60 s is
  not meaningful. It cannot show the hours-scale timing errors criticised in review.
- Early stopping uses 20 % of the training waveforms, which costs positives per fit.
- Resource numbers: MACs cover Linear, Conv1d, GRU, LSTM and GRUCell; int8 size is an estimate; CfC / LTC layers need a verified formula before
  their cost is reported; no MAC figure is quoted while the report's `notes` list is non-empty.
- The noise-level plateau makes N0 strong on this dataset; conclusions about robustness need the stress tests and other stations.

## Running

```bash
python tests/test_normalization.py      # also: test_features, test_training, test_models, test_alarms, test_confusion, test_stats, test_analyze
python tests/test_training_torch.py     # needs torch (also test_models_torch)
python -m tests.run_dnn_features --only dnn rfc cnn amp --rfc --cnn-sizes 1k 10k --norms N0 N2 --out results/final.json
python -m tests.analyze_results results/final.json
python -m tests.visualize_alarms
```