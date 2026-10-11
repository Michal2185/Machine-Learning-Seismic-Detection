# Evaluation and baselines handoff

State at closing of the evaluation / baselines branch. Builds on `docs/DATA_BRANCH_HANDOFF.md`
(labels, folds and the evaluation contract are defined there and not repeated).

## Protocol (applies to every detector, classical or neural)

1. **Fit on source data only.** Parameters, hyper-parameters and the decision threshold are
   chosen from training / validation waveforms of Apollo 12. Nothing is retuned on the data
   a result is reported on, and nothing is retuned for another mission.
2. **Freeze.** The chosen detector (parameters + thresholds) is stored as an artifact and
   applied unchanged.
3. **Two numbers.**
   * *Nested cross-validation*: for each fold the selection uses only that fold's
     training + validation waveforms, the result is pooled over the five test folds.
     This is the honest estimate for unseen waveforms.
   * *Final frozen detector*: selected once on all 70 waveforms (in-sample, optimistic);
     this is the artifact used for zero-shot tests on other stations / missions.
4. **Threshold-free view for zero-shot tests.** A threshold fixed on the Moon can drift on
   another body (calibration). Report the frozen-threshold result AND the
   recall-versus-false-alarm curve on the target.

## Code map

| Module | Purpose |
|---|---|
| `src/modules/evaluation/module.py` | `Evaluator`: alarms, classification, false alarms per scored hour, curves, threshold selection, bootstrap over waveforms, `paired_bootstrap`, `build_series`, `average_precision`, `pool_results` |
| `src/modules/evaluation/types.py` | `EvalConfig`, `ScoreSeries`, `EventOutcome`, `StreamResult`, `EvalResult`, `Curve` |
| `src/modules/evaluation/cv.py` | `cross_validate_budgets`, `cross_validate`: validation-selected candidate and threshold per fold, test applied once |
| `src/modules/detectors/types.py` | `Detector` interface: `fit(train_windows)`, `score(windows, streams_by_id)` -> one score per window |
| `src/modules/detectors/sta_lta.py` | `StaLtaDetector` (continuous, causal, gap-aware), `WindowStaLtaDetector`, tuning grids |
| `src/modules/detectors/frozen.py` | `FrozenSpec` (JSON), `freeze_sta_lta_sets`, `evaluate_frozen` |
| `tests/test_evaluator.py`, `test_detectors.py`, `test_frozen.py` | unit tests on synthetic data (22 / 10 / 5), run with plain `python` |
| `tests/test_evaluator_sanity.py` | oracle and random detectors on the real streams |
| `tests/freeze_sta_lta.py` | selection + nested CV + final frozen detector (about 18 min) |
| `tests/visualize_detection.py` | day and zoom figures with catalogue arrival, visible onset, alarms |
| `tests/test_sta_lta_cv.py` | per-fold tuned STA/LTA variants with paired comparisons |

New detectors implement `Detector` and are scored with `cross_validate_budgets`.

## Evaluator conventions

* Alarm time = END time of the window that triggered it; first threshold crossing, then
  `refractory_s` = 300 s of silence (sensitivity not studied yet).
* Classification of an alarm: inside an event's detect interval ([arrival - 120 s,
  arrival + 600 s]) -> detection (later alarms there are neutral duplicates); else inside an
  ignore interval or an interpolated gap -> neutral; else false alarm.
* False alarms per 24 h of SCORED time (stream minus ignore intervals, gaps, detect intervals).
* Operating point = highest validation recall under the false-alarm budget; ties -> highest
  threshold. If the budget is not binding the point is the knee of the curve.
* Uncertainty by resampling WAVEFORMS (not events or windows); paired bootstrap for comparing
  two detectors on the same waveforms. Differences below about 0.15 recall are inconclusive.
* Timing: quote delays against the visible onset; against the catalogue time they include the
  median +79 s offset of the visible onset and the 20-30 s of signal the labels require.

## Results (Apollo 12, 75 events on 70 waveforms)

Per-fold tuned continuous STA/LTA, threshold from validation at 1 false alarm per day:

| Grid | Recall all (usable / weak) | Test FA per day |
|---|---|---|
| first (STA <= 120 s, LTA <= 7,200 s) | 0.59 (0.71 / 0.35) | 0.96 |
| wide (STA <= 480 s, LTA <= 28,800 s) | 0.89 (0.94 / 0.81) | 1.51 |
| window-restricted (60 s) | 0.01 | 0.20 |

Gap masking made no difference for the best configuration (0.60 vs 0.60).

**Frozen STA/LTA** (`data/lunar/derived/baselines/sta_lta_frozen.json`): STA 120 s,
LTA 86,400 s (whole day so far), energy, gaps masked; one parameter set, two thresholds.

| | strict (budget 0.25 / day) | lenient (budget 2 / day) |
|---|---|---|
| threshold | 25.02 | 17.55 |
| nested CV recall, all events | 0.67 [0.56, 0.76] | 0.81 [0.72, 0.89] |
| usable / weak | 0.71 / 0.58 | 0.86 / 0.73 |
| test false alarms per day | 0.30 [0.14, 0.50] | 0.43 [0.24, 0.64] |
| delay vs visible onset, median | 112 s | 81 s |
| deep / impact / shallow detected | 4/9, 44/63, 2/3 | 6/9, 53/63, 2/3 |
| in-sample recall (final detector) | 0.65 | 0.83 |

Reading: no optimism gap between nested CV and in-sample; the lenient budget is not binding
(recall saturates near 0.8 at about 0.4 false alarms per day), so lenient is the knee of the
curve. False alarms at these rates cost roughly 0.1 % of the recording, so strictness saves
almost no bandwidth while costing 14 points of recall.

## Known limitations

* `alarms_per_day` and `downlink_share` count repeated alarms inside a coda as separate
  transmissions (provisional). `episode_precision` (detected / (detected + false alarms),
  about 0.7 for both points) is the meaningful share. A real data-sent / data-loss metric
  needs a snippet policy (pre-roll, post-roll, merging) and coverage of each event's signal.
* The STA/LTA grid optimum lies at LTA = whole day (57,600-86,400 s). After a strong event
  the LTA stays inflated for hours, which may suppress a second event on the same day
  (check pairs such as `evid00031` / `evid00032` with the visualizer).
* Per-fold thresholds vary (strict 16-31); test false alarms by fold ranged 0.3-3.6 per day.
  About 64 days of data means the strict point rests on roughly 16 false alarms.
* STA/LTA gap masking uses the interpolated gaps of the zone table (equivalent to a
  missing-packet flag onboard).
* Refractory period (300 s) and the detect interval ([-120, +600] s) are modelling choices.

## Next branches

1. `feat/input-normalization`: fixed scale fitted per fold, per-window standardisation,
   causal adaptive scale (+ optional asinh); feature extractor for DNN / RFC.
2. `models/cnn-gru-baselines`, `models/cfc`: matched parameter budgets, window and stateful
   stream; `Detector` interface and `cross_validate_budgets`.
3. `feat/downlink-metrics`: snippet policy, data sent / wasted / lost, onset captured by the
   pre-roll, resource accounting (parameters, MACs, memory, latency; STA/LTA needs only a few
   running sums).
4. `feat/stress-tests`: SNR scaling, gaps and jitter, sampling-rate change.
5. `feat/mars-insight` (and other Apollo stations): zero-shot with the frozen detectors.