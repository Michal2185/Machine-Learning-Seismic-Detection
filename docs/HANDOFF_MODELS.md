# Handoff: GRU and CfC (next block)

State at the close of phase 1 (controls, feature models, CNN, normalisation, statistics). Read with `README.md`,
`docs/MODELS_PLAN.md`, `docs/HARNESS_HANDOFF.md`, `docs/NORMALIZATION_HANDOFF.md`, `docs/STATS_HANDOFF.md`.
Goal of the next block: **get a presentable GRU / CfC table fast, polish later.** Everything below is written for a
new chat that starts from these files.

## 1. What the new chat has to deliver, in order

| Step | Deliverable | Why first | Effort |
|---|---|---|---|
| 1 | **W variant**: GRU and CfC on the normalised 398-sample window, 1k and 10k parameters, N0, 5 seeds, through the existing harness | needs no new infrastructure: a model factory + `TorchLearner` + `RawWindowInput("NTC")` | small |
| 2 | **E variant**: GRU and CfC on a sequence of 12 sub-window envelope features (5 s each), 1k / 10k / 100k | cheap, probably the most favourable accuracy-per-cost point; needs one new input builder | small-medium |
| 3 | **S variant**: stateful stream models (hidden state carried over hours), matched GRU control | the variant that tests the hypothesis (long background memory, cheap per 10 s step); needs a new detector class | large |
| 4 | **Stress tests** (gain, SNR, gaps, sampling rate) for the best models of 1-3 against RFC / CNN / amplitude control | in-domain results are saturated under N0; this is where models can differ | medium |
| 5 | Table + figures (`tests/analyze_results.py` already gives intervals and paired differences) | the thing to present | small |

**Present after step 1 or 2** if time is short: a table of GRU-W / CfC-W / GRU-E / CfC-E against RFC-N0, CNN-N0, amplitude
control, with intervals, parameters and cost. S and the stress tests then become the "what we do next" slide.

## 2. State of the repository (what exists and works)

All of this is tested; run `python tests/test_<name>.py` (normalization 43, features 12, training 25, models 12, alarms 7,
confusion 7, stats 11, analyze 2, plus torch tests `test_training_torch.py` 6 and `test_models_torch.py` 5).

| Piece | Where | Notes |
|---|---|---|
| Data, labels, folds, evaluator, frozen STA/LTA | data / evaluation branches | `bundle.fold_split(k)`, `bundle.eval_streams(ids)`, `cross_validate_budgets(bundle, [cand], evaluator, budgets)` |
| Normalisation N0-N3 | `src/modules/normalization` | `Normalizer(NormConfig("N0"))`; N0 = one constant per fold (about 1.2e-10); **N2 / N3 need `Normalizer.fit`** (reference level), shared `ScaleCache`; N1 is window-only |
| 6 features, `FeatureScaler` | `src/modules/features` | energy, rms, ptp, crest, dominant frequency, zero crossings per second |
| Harness | `src/modules/training` | `LearnerDetector(name, learner, input_builder, TrainConfig)`; `TorchLearner(model_factory, lr, weight_decay)`; fresh weights per fold; Sampler draws new background windows each epoch; early stopping on held-out TRAINING waveforms (20 %); `fit_infos` per fold |
| Inputs | `training/inputs.py` | `RawWindowInput(normalizer, layout="NT"/"NCT"/"NTC")`, `FeatureInput` |
| Models | `src/modules/models` | `dnn.py`, `rfc.py`, `cnn.py` (pattern to copy: `make_*`, `PRESETS`, `*_param_count`, `*_macs`, `build_*_detector`), `amplitude.py` |
| Resource report | `training/resources.py` | params, fp32 / int8 bytes, MACs per window, state floats, notes; hook counter knows Linear, Conv1d, GRU, LSTM, GRUCell |
| Runner | `tests/run_dnn_features.py` | `--only dnn rfc cnn amp`, `--norms`, `--seeds`, writes JSON incl. per-event outcomes; **one `cross_validate_budgets` call per seed** |
| Statistics | `tests/analyze_results.py`, `reporting/stats.py` | waveform bootstrap, paired comparison, CSV, recall plot |
| Plots | `tests/visualize_alarms.py` | alarm vs arrival, confusion matrices (accepts `dnn-`, `cnn-`, `rfc-`, `amp-` names; extend `make_detector` for new kinds) |

Interfaces to respect: `Detector.fit(train_windows)` and `Detector.score(windows, streams_by_id) -> one score per window, same order`
(higher = event). Window records have `.data` (398 float32), `.label` (1 event / 0 background / -1 ignore), `.event.evid` (waveform id),
`.end_time`, and an end sample reachable through `end_sample_of(w)` from `src.modules.normalization`. `streams_by_id[wid]` is an
`EvalStream` (`.data`, `.sampling_rate`, `.gaps`, `.events`). Alarm time = end of the triggering window; the evaluator chooses thresholds.

## 3. Results to beat (Apollo 12, nested CV, 5 seeds, 75 events, 95 % intervals over waveforms)

Strict budget 0.25 false alarms/day; lenient 2/day. Detectors in the same table must use the same protocol.

| Detector | Recall strict | FA/day strict | Recall lenient | FA/day lenient | Params | Model cost per window |
|---|---|---|---|---|---|---|
| RFC, N0 | **0.91** [0.86, 0.96] | 0.16 | 0.97 [0.93, 0.99] | **0.26** | 4 280 nodes (29 KiB) | 733 comparisons |
| CNN 1k, N0 | 0.85 [0.78, 0.91] | 0.14 | 0.99 | 0.53 | 1 025 | 77 k MACs |
| CNN 10k, N0 | 0.84 [0.78, 0.89] | 0.18 | 0.99 | 0.43 | 9 729 | 624 k MACs |
| DNN 10k, N0 | 0.82 [0.75, 0.89] | 0.20 | 0.99 | 0.72 | 10 081 | 9.9 k MACs |
| DNN 1k, N0 | 0.76 [0.68, 0.84] | 0.12 | 0.98 | 0.57 | 1 037 | 980 MACs |
| Amplitude control (RMS), N0 | 0.68 [0.57, 0.79] | 0.31 | 0.97 | 0.87 | 1 threshold | 0 |
| STA/LTA (frozen, handoff) | 0.67 [0.56, 0.76] | 0.30 | 0.81 [0.72, 0.89] | 0.43 | - | a few running sums |
| CNN 10k, N2 v3 | 0.50 [0.39, 0.60] | 0.17 | 0.73 | 1.11 | 9 729 | 624 k MACs |
| DNN 10k / RFC, N2 v3 | 0.43 / 0.30 | 0.19 / 0.23 | 0.59 / 0.73 | 0.81 / 1.76 | | |

Front end (normalisation + 6 features) costs about 12 600 operations per window; the amplitude control about 1 200.

Weak events (no usable onset) at the strict budget: RFC-N0 0.87, CNN-N0 0.73-0.74, DNN-N0 0.62-0.70, amplitude control 0.54, STA/LTA 0.58.
Deep events: RFC-N0 0.87, CNN-N0 0.76-0.78, amplitude 0.56, STA/LTA 4 of 9. Delay vs visible onset (median): N0 models 42-59 s,
amplitude control 51 s, STA/LTA 112 s (strict) / 81 s (lenient).

Paired findings (strict, intervals exclude 0): RFC, both CNNs and DNN-10k beat the amplitude control by +0.14 to +0.23 recall (largest on weak
events); DNN-1k does not (+0.08 [-0.03, 0.20]); N0 beats N2 for every model by +0.34 to +0.61; CNN vs RFC differ by about 0.07 in recall (the direct paired difference was not run; do `--compare cnn-10k-N0 rfc-medium-N0` before claiming either way).

## 4. Findings that shape the GRU / CfC work

1. **In-domain N0 is close to ceiling** (0.97-0.99 recall lenient). A CfC is unlikely to win in-domain recall; its case is cost per step,
   memory over hours (variant S), and robustness (stress tests). Say so in the presentation.
2. **Amplitude is most of the signal**: RFC importances are about 89 % peak-to-peak + RMS + energy. A learned model must be compared with the
   amplitude control, not only with STA/LTA (which a bare RMS threshold already matches at the strict budget).
3. **The noise level is a plateau (about 2.5e-10) with frequent quiet dips**, not a slowly varying background. N2 (causal adaptive scale) loses to
   N0 for every model because it follows the dips. This is exactly what a stateful model (S) could learn to handle internally. Use N0 as the
   working scaling; N2 v3 is frozen as a documented baseline that loses.
4. **Window models are expensive in sequence form**: a W-variant recurrent model re-reads 398 samples every 10 s. The S variant reads each sample
   once. Report cost per 10 s of signal, not only per window.
5. **Emergent events** (no usable onset, no positive training windows) are found by the N0 models; check GRU / CfC on weak events separately
   (`recall_weak` in `analyze_results`).

## 5. Variants, inputs and budgets

**Matched budgets.** Every CfC gets a GRU with the same input and about the same parameter count (1k, 10k, 100k), otherwise a CfC advantage cannot be
attributed to continuous-time dynamics. GRU parameters (PyTorch, one layer, input size F, hidden h, linear head): `3(F*h + h^2 + 2h) + (h + 1)`.
For F = 1 (variant W) h = 17 / 56 / 181 give 1 038 / 9 969 / 100 094. For the CfC write `cfc_param_count(...)` by building the model and counting
parameters, then search the hidden size per budget; keep the backbone small (or none) at 1k.

**W (window).** Input (B, 398, 1) = normalised window, `RawWindowInput(normalizer, "NTC")`, decision at the last step, time step 1/fs.
GRU: `nn.GRU(1, h, batch_first=True)` + Linear on the last hidden state. CfC: wrap the `ncps` CfC and a Linear head in an `nn.Module` that returns
`(B, 1)` logits. This runs through `LearnerDetector` + `TorchLearner` unchanged. Cost warning: scoring is (number of windows) x 398 steps (about
240 k windows per fold and candidate), so 100k-parameter W models are too slow on CPU; run 1k and 10k first. The ncps CfC loops over time in Python,
which is slow; check the training time per epoch before launching 5 seeds.

**E (envelope sequence).** Each 60 s window is split into 12 sub-windows of 5 s (33 samples); per sub-window the features log energy, RMS,
peak-to-peak, zero crossings (4 features, after N0 scaling) form one step; input (B, 12, 4), plus a time-step input of 5 s for the CfC.
New builder `EnvelopeInput(normalizer, fs)` next to `FeatureInput` (vectorised, no Python loop per window), with tests first. Resource note: the
window can be recomputed from scratch (12 steps per window) or maintained in a streaming way (2 new steps per 10 s); report both.

**S (stream).** Hidden state carried across consecutive 10 s steps (66 samples) over hours. Per step input: log energy, RMS, peak-to-peak and zero
crossings of the newest 10 s, after N0 scaling (log features already compress the 25x event peaks). Target of step t = label of the window ending at t
(1 / 0 / -1: ignore steps get no loss). Training: truncated backpropagation over segments of a few hours (start with 2-4 h = 720-1440 steps), carry the
state between segments of the same file, new random segment draw each epoch (all segments with positives + random background segments, same spirit
as the Sampler, ratio chosen on the held-out training waveforms). Inference: `score(windows, streams_by_id)` runs each stream once in order,
caches per-step scores, and maps every window to the step of its end sample; compare continuous inference (24 h in one pass) with state resets every
training-segment length to see if the state drifts. S needs a new `StreamDetector(Detector)` (not `LearnerDetector`): it uses `streams_by_id`,
which the window models ignore. Only the validation split of the TRAINING fold may be used for early stopping, as in the harness.
Gaps: interpolated stretches (`stream.gaps`) become missing steps; for the CfC pass their true time step instead of resampling (also the hook for the
irregular-sampling stress test).

**Normalisation for the new models.** N0 first. N2 is not needed for S (the network adapts); run N2 for W / E only if time permits (it is expected
to lose; the plan requires N0 and N2 for every model, so one N2 row per variant is enough).

## 6. CfC specifics (verify against the installed version)

* Library: `ncps` (`pip install ncps`, torch backend `ncps.torch.CfC` / `LTC`); confirm availability on the user's machine before anything else
  (my sandbox had no torch, so nothing about ncps was executed in this project).
* Check from the installed source, not from memory: the constructor arguments (`in_features`, `units`, `proj_size`, `return_sequences`,
  `batch_first`, `mixed_memory`, `mode`, `backbone_units`, `backbone_layers`), whether `forward` takes `timespans`, and the cell's layers.
* **Cost accounting**: `count_model_resources` hooks leaf layers. If the CfC cell is built from `nn.Linear` and called once per step, the hooks
  fire per step and the MAC count comes out right; `state_floats` (hidden size) must be set by hand. Verify by comparing with a closed-form MAC count
  derived from the cell code (test it). Elementwise gates and the activation are not counted; say so. The `notes` list names any layer that is not counted,
  and **no MAC figure may be quoted while `notes` is non-empty**.
* LTC (ODE solver, several unfolds per step) is an ablation only: report cost with the number of unfolds.
* Seeds: the harness calls `torch.manual_seed(seed)`; CfC / GRU runs should use the same five seeds as the existing models.

## 7. Cost metrics to report

For every model: parameters, fp32 and int8 size, **MACs per 10 s of signal** (W: per window; E: per window and streaming; S: per step), recurrent state
floats, and input operations (normalisation + features). Add a `macs_per_10s` field to `ResourceReport` if the existing `macs_per_window` is ambiguous.
Cost story to tell: STA/LTA needs a few running sums, the amplitude control about 1 200 operations, RFC about 13 000 operations plus 733 comparisons,
CNN 77 k-624 k MACs; a stateful model's cost per 10 s is one cell update plus the 4 features.

## 8. Protocol reminders (do not change between models)

* Fit on Apollo 12 training / validation waveforms only; thresholds from validation at the false-alarm budget; test applied once; pooled nested CV (5 folds).
* Strict 0.25 and lenient 2 false alarms per day; 5 seeds; **one candidate per `cross_validate_budgets` call**.
* Report recall with waveform-bootstrap intervals (`analyze_results`) and paired differences, not the old "0.15 rule".
* Never tune on the test folds; hyper-parameters (lr, weight decay, dropout, hidden sizes) are chosen on validation folds only; the existing
  models were deliberately untuned, so keep new defaults modest and identical for GRU and CfC.
* Scores are logits; many N0 detectors hit a ceiling at the lenient budget, so also look at false alarms and at weak / deep events.

## 9. Open items

1. **STA/LTA paired comparison** is missing: needs STA/LTA's per-event outcomes in the runner format. Waiting for the signatures of
   `src/modules/detectors/frozen.py` (`FrozenSpec`, `evaluate_frozen`) and `sta_lta.py` (`StaLtaDetector` constructor), or the top of
   `tests/freeze_sta_lta.py`. Today only the handoff's interval is drawn as a band.
2. **Stress tests** (not built): gain 0.3x / 3x / 10x on test streams with thresholds frozen from the 1x validation, added noise, gaps and timestamp
   jitter, decimation to about 2.65 Hz (spectral energy drops sharply above 1.05 Hz), then InSight zero-shot. Build on `cross_validate_budgets` with
   modified `EvalStream.data`.
3. **User decisions pending**: trim file edges (first / last minutes ramp towards 1e-15) from training negatives and the N0 fit only; how dead stretches
   enter labels and false-alarm accounting (data branch; would change STA/LTA numbers).
4. N1 and N3 were not rerun with N2 v3 (ablations only). Downlink metrics (snippet policy, merging of repeated alarms) and InSight remain later items.
5. Known data caveats: only 49 of 75 events carry positives; delays against the catalogue are capped at +10 min by the detect interval and cannot show
   hours-scale timing errors; the alarm is the end of a 60 s window, so differences below about 30-60 s are not meaningful.

## 10. Practicalities learned the hard way

* Windows paths: run scripts as modules from the repo root (`python -m tests.<script>`); every package folder needs its `__init__.py`.
* The path constants at the top of `tests/*.py` scripts (catalogue, data, derived directory) must be edited once; they were guessed.
* `Normalizer.fit` must be called before N2 / N3 use (the harness and runner do it); `ScaleCache` computes tracks on demand (about 100 s per fold the first time).
* Use `--out results/<name>.json` for every run; old result files lack outcomes and cannot be analysed with the bootstrap.
* A full run of all current configurations takes about 1.5-2 hours; use `--seeds 1` and `--only` for smoke runs.
* Keep `results/final.json`, `results/analysis.csv` and `results/analysis_recall.png` as the frozen phase-1 artifacts.

## 11. First message for the new chat

"Attach the README, this handoff, `src/modules/detectors/types.py`, `src/modules/training/harness.py`, `learners.py`, `inputs.py`,
`resources.py`, `src/modules/models/cnn.py`, `tests/run_dnn_features.py`, `tests/analyze_results.py`. We are starting `models/gru-cfc` as described in
`docs/HANDOFF_GRU_CFC.md`, step 1 (W variant, GRU and CfC at 1k and 10k parameters on N0, through the existing harness), test-first, in the same style
as the earlier branches (synthetic unit tests, torch tests, handoff note). Tell me first whether `ncps` is installed and send the CfC constructor
signature of the installed version."