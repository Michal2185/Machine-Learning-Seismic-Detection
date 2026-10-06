# Data branch handoff

State at closing of the data-handling branch (Apollo 12, Grade A catalogue).
Everything below was verified on the real data with `tests/test_pipeline.py`.

## Entry point

```python
from src.modules.pipeline import DataPipeline, PipelineConfig

config = PipelineConfig(catalog_path=..., data_dir=..., derived_dir=...)
bundle = DataPipeline(config).build()

split   = bundle.fold_split(k)                         # DatasetSplit: train / validation / test windows
streams = bundle.eval_streams(split.test_events)       # continuous waveforms + targets for the evaluator
```

`split.*_events` hold PRIMARY WAVEFORM ids. Zone table and folds are cached in
`derived_dir/zones/` and `derived_dir/folds/folds.csv` (`rebuild=True` recomputes).

## Dataset

| | |
|---|---|
| Catalogue events | 75 (63 impact, 9 deep, 3 shallow) |
| Distinct waveforms | 70 (5 days carry two events; identical copies are merged) |
| Sampling rate | 6.625 Hz, one trace per file, almost all files 24 h (one is 14.4 h) |
| Windows | 603,214 of 60 s (398 samples), step 10 s |
| Preprocessing | causal Butterworth highpass 0.2 Hz (order 4), no scaling, float32 |

Scaling is NOT applied by the pipeline: `fixed_scale` must be fitted on the
training waveforms of each fold (`Preprocessor.fit`), so the model code does it.
`DataPipeline` refuses `fixed_scale` without an explicit scale.

## Labels (window level)

| label | meaning |
|---|---|
| 1 | window holds >= 20 s of VISIBLE signal after the visible onset (window starts at most `positive_span_s` = 0 s after it) |
| -1 | ignore: onset region (from 60 s before the earliest plausible onset), the coda up to the per-event coda end, and windows >= 50 % inside an interpolated gap |
| 0 | background |

* Visible onset = first time the 20 s RMS envelope of the 0.2 Hz highpassed signal stays above 3x the local background for 30 s (search -900..+300 s around the catalogue time). Usable if the offset is within -120..+600 s.
* Events without a usable visible onset (26 of 75: 17 none, 9 offset outside bounds; impact 24, deep 2) get NO positives. They still count in event-level evaluation.
* Coda end = first time after the peak the envelope stays below 2x the day background for 600 s (dropouts skipped); capped at 6 h (`max_coda_s`). 4 events are censored at the file end.
* Result: 198 positive windows (4-5 per event, 49 events), 49,195 ignore (8.16 %, of which 3,946 gaps), 553,821 background.
* Without a zone table `Labeling` falls back to catalogue-anchored positives and per-type coda values.

All values live in `LabelConfig`, `ZoneConfig`, `PipelineConfig`.

## Evaluation contract (for the event-level evaluator)

* Alarm time = END time of the window that triggered the alarm.
* An event is detected if an alarm falls in `EventTarget.detect` = [arrival - 120 s, arrival + 600 s] (catalogue-anchored, independent of onset anchoring).
* False alarms: alarms outside every `EventTarget.ignore` interval and outside interpolated gaps (`EvalStream.gaps`).
* Normalise false alarms per hour of SCORED data (stream length minus ignore intervals minus gaps), not per 24 h: files differ in length and 8 % of the data is ignored.
* Report recall for all 75 events AND for the 49 with `has_positive`; also by event type with the out-of-fold predictions pooled over the 5 folds (only 3 shallow events).
* Catalogue times are whole minutes and the visible onset is a median +79 s later (p25 +29, p75 +151); timing errors below about 60 s are not meaningful.

## Folds

5 folds over distinct waveforms, stratified into shallow / deep / impact with positives / impact without positives, dealt round-robin (seed 42). Fold k is test, fold k+1 validation, the rest training (42 / 14 / 14 waveforms). Test positives per fold: 41, 44, 40, 33, 40. Training folds hold only about 113-125 positive windows from 28-31 events.

## Data findings that matter for modelling

* Background noise level differs between days by 15-300x (robust std after one global scale: 0.003..2.27). Only `evid00120` is partly explained by gaps. -> compare fixed scale, per-window standardisation and a causal adaptive scale (with optional asinh compression).
* Largest event peaks are about 25x the median event peak -> input compression helps recurrent models.
* Raw data contain linearly interpolated stretches (no exact zeros): runs >= 10 s in every waveform (median 21 per day, max 281), runs >= 60 s in 8 waveforms (one single 18,702 s gap in `evid00030`).
* Event/background spectral contrast: none below about 0.15 Hz, strongest at 0.4-0.8 Hz, sharp power drop at about 1.05 Hz (decimation to about 2.65 Hz is a possible resource experiment).
* Coda decay is close to exponential (tau median 1,924 s, IQR 1,617-2,432 s). Outliers with a coda far above the prediction: `evid00015`, `evid00026`.
* Weak events by peak ratio: some "weak" events are strong (e.g. `evid00121`, peak/background 25.9) because the onset was unusable, not because the signal is weak.

## Open items (next branches)

1. Event-level evaluator, tested with STA/LTA first.
2. Normalisation variants incl. the causal adaptive scale; feature extractor for the 6 features (DNN / RFC).
3. Models at matched parameter budgets: small 1D CNN, GRU, CfC (LTC as ablation), plus MLP on features; windows vs stateful stream for the CfC.
4. Stress tests for the robustness hypothesis: SNR scaling by inserting events into noise, gaps / timestamp jitter, sampling-rate change.
5. Cross-mission: other Apollo stations, then InSight (MQS catalogue, common rate and band, own labels).
6. Optional sensitivities: `ZoneConfig(k_on=2.0)`, `LabelConfig(positive_span_s=60)`.