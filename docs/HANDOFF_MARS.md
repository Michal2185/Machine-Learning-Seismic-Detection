# Handoff: Mars zero-shot (feat/mars-insight) and the all-results comparison

State at the close of the Mars branch (two catalogued InSight events). Read with `HANDOFF_BASELINE.md`, `HANDOFF_DATA_BRANCH.md`,
`HANDOFF_MODELS.md`, `HANDOFF_STRESS.md`, `MARS_PREREGISTRATION.md`. Everything below was run on the user's machine
(Windows, CPU, torch 2.14.1+cpu) unless marked "sandbox".

## 1. What was done

| Piece | Where | Status |
|---|---|---|
| Dependency-free miniSEED reader (float64 / Steim / int), gap filling | `src/modules/mars/miniseed.py`, `_mseed_records.py` | tested (sandbox + real files) |
| Chain: `resample_poly(53, 160)` 20 -> 6.625 Hz, causal Butterworth highpass 0.2 Hz, optional lowpass 1.3 Hz (arms `lp1.3`, `nolp`) | `chain.py` | tested; **identical to the Apollo `apply_filter`** (same steady-state initial condition; verified by the repo test) |
| Streams, targets (Apollo envelope rule restated), windows, fold rotation | `streams.py` | tested; **window grid verified against Apollo** (`--check-windowing`: 8 667 windows, end_sample difference 0, `end_time = end_sample / fs`) |
| N0 recalibration: `calibrate_n0`, `set_fixed_scale` | `scale.py` | tested |
| Frozen-model runner (policy D + all-fold audit + frozen STA/LTA) | `run/run_mars.py` | ran on the user's machine (smoke and full run) |
| Timelines | `tests/plot_mars_timelines.py` | smoke-tested on fabricated scores; the real plot was not inspected by me |
| Comparison of everything | `tests/compare_all.py` -> `results/comparison/` (fig1-6, CSVs, `COMPARISON.md`) | run in the sandbox on `mars_1.json` |
| Tests | `tests/test_mars.py` | 21/21 pass on the user's machine |

Window convention learned: Apollo step = `round(10 s x 6.625 Hz) = 66` samples (9.96 s), length 398; 24 h file -> 8 667 windows.
Filter convention learned: `apply_filter` starts from the steady state of the first sample.

## 2. Data (what exists, what it is)

* Catalogue `Mars_InSight_training_catalog_final.csv`: columns filename, time_abs, time_rel (s since file start), evid. **Only two rows**:
  evid0005 (2022-02-03T08:08:27, 507 s) and evid0006 (2022-01-02T04:35:30, 2130 s). Nine more event hours exist without catalogue rows
  (examples inspected: evid0041 2019-05-23 hour 02, evid0033 2019-07-26 hour 12); they contain events at unknown times and are NOT background.
* Files: XB.ELYSE.02.BHV (raw V axis of the VBB, **not the vertical**), 20 Hz, 1 h, float64, no gaps, tapered edges (~20 s), units unknown
  (robust std 20-50, Apollo N0 1.32e-10: ratio 1.78e11 for arm lp1.3). Pre-event noise is non-stationary; glitch spikes and 1.0 Hz / 2.4 Hz tonal lines are present.
* evid0005: low-frequency event; catalogue time = small first arrival, main burst about 3 min later; 0.4-0.8 Hz contrast 81x.
  evid0006: 2.4 Hz-type event (energy 1-3 Hz; the 1.3 Hz lowpass halves its contrast), censored coda.
* Targets (shared by both arms): evid0005 detect (387, 1107) s, ignore (325, 1470); evid0006 detect (2010, 2730), ignore (1947, 3570, censored).
  Both onsets usable (-2 s, -3 s). Edges (first 120 s, last 30 s) registered as gaps. **Scored background about 1.15 h in total.**
* N0 constants (median over files of the robust std of background samples): 23.43 (lp1.3), 31.93 (nolp).

## 3. Results (full run: 7 models, 2 arms, N0 x0.5 / x1 / x2, 5 seeds x 5 folds; policy D + audit)

Files: `results/mars.json` (records, audit, targets, meta), `results/mars_scores.npz`; summary in `results/comparison/`.
Detection share = share of (fold model x seed) runs that detect the event; false alarms are COUNTS per run over 1.15 h, not rates.

| Strict thresholds, lp1.3 | N0 x0.5 | x1 (registered) | x2 |
|---|---|---|---|
| CfC-E 1k | 1.00 / 0.96, FA 0.12 | 1.00 / 0.16, FA 0 | 0.32 / 0.00 |
| GRU-E 1k | 1.00 / 1.00, FA 1.64 | 0.96 / 0.72, FA 0.36 | 0.44 / 0.52 |
| CNN 1k | 1.00 / 0.64, FA 0 | 1.00 / 0.08, FA 0 | 0.64 / 0.00 |
| RFC | 1.00 / 0.88, FA 0.24 | 0.24 / 0.60, FA 0 | 0.16 / 0.00 |
| Amplitude (RMS) | 1.00 / 0.60, FA 0 | 0.80 / 0.20, FA 0 | 0.80 / 0.00 |

(columns: evid0005 / evid0006). Findings:
1. The N0 scale dominates every number; a factor of 2 changes detection by up to 0.7 and the ranking. The N0 estimate rests on two files.
2. Without the 1.3 Hz lowpass GRU-E fires about 6.2 false alarms per run (others 0-0.08) at x1 strict: spectral mismatch.
3. Neural scores barely clear the Apollo thresholds on Mars (event maximum +0.1 to +0.4 above the strict threshold; RFC +0.00; background 0.4-2.3 below): detections are fragile.
4. Delays on evid0005 are about 180-200 s (the models fire on the main burst, not the first arrival); amplitude controls fire earlier at the lenient budget (about 60 s).
5. No false alarms on the glitch spikes (checked informally by eye-picked times, not by a defined glitch finder).
6. Frozen STA/LTA found 0/2: an artefact of 1 h files (LTA needs 600 s of valid history, available only from 720 s; pre-event noise bursts inflate the LTA). It is not comparable on these snippets.

Registered predictions: (1) low-frequency event detected by most models: yes at the lenient budget, mixed at strict (RFC 0.24, amplitude 0.80);
(2) GRU-E most exposed to spectral mismatch: supported for false alarms, not for detection; (3) glitches trigger amplitude models: not supported (informal);
(4) no continuous-time advantage claimed: consistent; (5) delays capped at +10 min: stands.

## 4. Comparison across all phases

`tests/compare_all.py --mars results/mars.json --out results/comparison`. Apollo in-domain and stress numbers are TRANSCRIBED from the
handoff tables (no raw JSONs were available to the script author); intervals only where given. To recompute with intervals, feed the
result JSONs to `analyze_results` and replace the dictionaries at the top of the script. CNN/DNN/RFC use the phase-1 schedule, GRU/CfC the long one.
Summary: CfC-E 1k ties RFC at the strict Apollo budget (0.92 vs 0.91) at about 1 KB vs 29 KiB; beats GRU-E and CNN (+0.07, intervals exclude 0);
generalization is mixed (better than GRU-E under spectral change; no advantage under gain or SNR loss); no evidence that continuous-time dynamics are the cause.

## 5. Commands

```
python -m tests.test_mars
python -m run.run_mars --check-windowing                      # gate; must print WINDOWING MATCHES
python -m run.run_mars                                        # full run (minutes)
python -m tests.plot_mars_timelines --catalog <csv> --data-dir <dir> --arm lp1.3
python -m tests.compare_all --mars results/mars.json --out results/comparison
python -m tests.analyze_results results/mars.json --reference amp-rms-N0@mars-lp1.3-x1   # not yet tried on Mars records
```
Paths: catalogue under `data/mars/catalogs/`, files under `data/mars/data/`; the tests also honour the `MARS_DATA` environment variable.
Adding the other nine events = add rows to the catalogue CSV (`time_rel` = MQS start time minus file start) and rerun; nothing else changes.

## 6. Known limitations (state them in the report)

* Two events, about 1.15 h scored background: no intervals, no significance, no false-alarm RATES; counts only.
* BHV is not vertical; units unknown (the N0 recalibration absorbs a gain, not a different physical quantity); MQS family/quality unknown;
  MQS is incomplete, so some counted false alarms may be real unlisted events; the envelope coda rule ended the evid0005 coda at 1470 s although
  energy stays elevated (some post-coda alarms may be real coda).
* The resample FIR looks about 1.5 s ahead (not causal at that scale). Delays are capped at +10 min by the detect interval and cannot show hours-scale errors (Reviewer 2).
* STA/LTA is not comparable on 1 h snippets; the Mars threshold-free view (recall-vs-false-alarm curves, protocol item 4) was NOT produced.
* The Mars runner uses the same E-model checkpoints as the stress tests; `cfc0E`, CNN, DNN, RFC schedules differ (see section 4).

## 7. Open items, by value

1. MQS table for the 11 files (times, family, quality) -> n = 11; then the whole MQS catalogue (low-frequency, quality A/B) for a real zero-shot sample.
2. Long continuous Mars data (full sols, event-free and event-bearing) for false-alarm RATES and a fair STA/LTA.
3. Scale-robustness: the N0 constant is the failure point (Apollo gain tests, Mars x0.5/x2). Options: gain augmentation in training, a better N0 estimator, the stateful S variant (not built).
4. Threshold-free Mars curves; glitch finder defined without detector outputs; Mars score timelines inspected.
5. Irregular-step test for the CfC (`timespans`), E-feature spectral ablation, true 2.65 Hz decimation, int8 accuracy, latency on a target-like CPU, downlink metrics, other Apollo stations.