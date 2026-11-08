# Handoff: GRU / CfC models, stress tests, frozen models, and the Mars zero-shot (next branch)

State at the close of the models + stress-test phase (Apollo 12 only). Read with `docs/HANDOFF_BASELINE.md`,
`docs/DATA_BRANCH_HANDOFF.md`, `docs/HANDOFF_MODELS.md`, `docs/MODELS_PLAN.md`. Goal of the next branch:
**zero-shot evaluation of the FROZEN detectors on InSight (Mars) data**, in a new chat on a new branch
(e.g. `feat/mars-insight`). Everything below was run on the user's machine (Windows, PowerShell, CPU only,
torch 2.14.1+cpu, ncps 0.0.2) unless marked "not run".

## 1. Research question and where it stands

Question: can compact continuous-time networks give a better accuracy-resource-generalization trade-off than
conventional onboard detection under extraterrestrial conditions?

| Part | Status after this phase |
|---|---|
| Accuracy (in-domain) | CfC-E 1k ties RFC and beats CNN-1k, GRU-E and the amplitude control at the strict budget |
| Resources | CfC-E 1k: 950 parameters, about 1 KB int8, about 10.6k MACs per window + about 3.2k input operations; RFC is about 29 KiB |
| Generalization | mixed: CfC-E better than GRU-E under spectral change, no advantage under gain or SNR loss, no evidence that continuous-time dynamics are the cause |
| Continuous time specifically | untested where it could matter (irregular steps, changed step duration): open item |

## 2. Protocol (unchanged, do not change between models)

Fit on Apollo 12 training / validation waveforms only; thresholds from validation at the false-alarm budget; test
applied once; pooled nested CV (5 folds). Strict 0.25 and lenient 2 false alarms per day. 5 seeds (amplitude control
1). One candidate per `cross_validate_budgets` call. Recall with waveform-bootstrap intervals and paired differences
(`analyze_results`), not the old "0.15 rule". Never tune on test folds.

Training schedule: CNN / DNN / RFC use the phase-1 schedule (60 epochs cap). GRU / CfC use a longer one:
`--rec-epochs 150 --rec-min-epochs 60 --rec-patience 40`, lr 1e-3, weight decay 1e-3. **The CNN was not rerun on the
long schedule** (decision pending); say so wherever the tables are compared.

## 3. What was built (repo map)

| Piece | Path | Notes |
|---|---|---|
| GRU / CfC models | `src/modules/models/recurrent.py` | `PRESETS`, `*_param_count`, `*_macs`, `make_gru`, `make_cfc(dt)`, `build_recurrent_detector(variant="W"/"E", fs=...)`; export in `models/__init__.py` |
| Variant E input | `src/modules/training/inputs.py` | `envelope_features`, `EnvelopeInput` (12 sub-windows of 33 samples aligned to the window end, log energy / RMS / ptp / zero crossings per second, standardised on the training windows) |
| State size in cost report | `src/modules/training/resources.py` | `state_floats = hook count or model.state_floats` |
| Fold callback | `src/modules/evaluation/cv.py` | `cross_validate_budgets(..., fold_callback)` |
| Stress module | `src/modules/stress/` | perturbations, window re-cutting, frozen-threshold loop with oracle option and load / save hooks |
| Checkpoints | `src/modules/checkpoints/` | save / load fitted detectors with thresholds |
| Runners | `run/run_dnn_features.py`, `run/run_stress.py` | flags in section 9 |
| Tests (all plain `python -m tests.<name>`) | `test_models_recurrent_torch`, `test_envelope_input` (12), `test_stress` (23), `test_checkpoints` (12) | the last three were run in a sandbox with stubs and then on the user's machine |
| Diagnostics | `tests/time_recurrent.py`, `tests/diagnose_recurrent.py`, `tests/probe_ncps.py` | timing, per-epoch trajectories, ncps signatures |

ncps facts (verified from the installed version): `CfC(input_size, units, proj_size=None, return_sequences=True,
batch_first=True, mixed_memory=False, mode="default", activation="lecun_tanh", backbone_units=None,
backbone_layers=None, ...)`; `forward(input, hx=None, timespans=None)` returns `(output, hx)`. The default backbone
(128 units) makes 1k parameters impossible, so models use `backbone_layers=0` (four `Linear(F+h, h)`:
`4h(F+h+1) + h + 1` parameters with the head). `timespans` is indexed `[:, t]` and squeezed inside ncps, so the wrapper
passes a `(B, T, h)` expanded view (a `(B, T)` tensor fails to broadcast when B differs from h).

Matched budgets (hidden size, GRU / CfC): W (F=1): 1k 17 / 15, 10k 56 / 49, 100k 181 / 157; E (F=4): 1k 15 / 13,
10k 55 / 48, 100k 180 / 156. Parameter counts are asserted by test; hook MAC counts equal the closed forms with an
empty `notes` list.

## 4. Variants

* **W**: 398 raw normalised samples, decision at the last step, CfC time step 1/fs.
* **E**: 12 steps of 4 features (5 s each), CfC time step about 4.98 s (`cfcE`); `cfc0E` is the same CfC with the
  ncps default time step 1 (ablation). A constant time step is a rescaling of a learnable weight, so a gain from it
  is an initialisation / optimisation effect, not evidence of continuous-time dynamics.
* **S (stream)**: NOT built. Hidden state carried across consecutive 10 s steps over hours, input = four features of the
  newest 10 s, ~850 MACs per 10 s at 1k; needs a `StreamDetector` and truncated BPTT over 2-4 h segments (see
  `HANDOFF_MODELS.md` section 5).

## 5. Results (Apollo 12, N0, nested CV, 5 seeds, 95 % intervals over waveforms)

Strict budget (0.25 FA/day); weak = events without a usable onset.

| Detector | Recall all | FA/day | Weak | Lenient recall | FA/day | Delay vs onset (s) |
|---|---|---|---|---|---|---|
| CfC-E 1k | 0.92 [0.86, 0.96] | 0.15 | 0.83 | 0.98 | 0.39 | 38 |
| CfC-E 1k, dt = 1 (`cfc0E`) | 0.86 [0.81, 0.92] | 0.13 | 0.75 | 0.98 | 0.48 | 35 |
| GRU-E 1k | 0.85 [0.78, 0.91] | 0.20 | 0.72 | 0.98 | 0.40 | 46 |
| RFC | 0.91 [0.86, 0.96] | 0.16 | 0.87 | 0.97 | 0.26 | 47 |
| CNN 1k | 0.85 [0.78, 0.91] | 0.14 | 0.74 | 0.99 | 0.53 | 52 |
| GRU-W 1k (long) | 0.79 [0.70, 0.86] | 0.23 | 0.67 | 0.99 | 0.60 | 52 |
| CfC-W 1k (long) | 0.74 [0.67, 0.82] | 0.26 | 0.58 | 0.94 | 0.72 | 107 |
| Amplitude control (RMS) | 0.68 [0.57, 0.79] | 0.31 | 0.54 | 0.97 | 0.87 | 51 |
| STA/LTA frozen (baseline handoff) | 0.67 [0.56, 0.76] | 0.30 | 0.58 | 0.81 | 0.43 | 112 |

Paired differences, strict, CfC-E 1k minus the other (95 % interval): GRU-E +0.069 [+0.028, +0.118]; `cfc0E`
+0.053 [+0.019, +0.091]; RFC +0.003 [-0.039, +0.051] (tie); CNN-1k +0.067 [+0.014, +0.125]; amplitude control
+0.237 [+0.147, +0.330] (weak events +0.29 [+0.14, +0.45]).
10k: CfC-E 0.89 [0.85, 0.93], GRU-E 0.78, `cfc0E` 0.81; CfC-E 10k minus GRU-E 10k +0.117 [+0.069, +0.169]; 10k minus 1k
for CfC-E -0.024 [-0.048, +0.000] (no gain from more parameters). GRU-E 10k overfits (8 of 25 fits best at epoch 0).

W findings: CfC-W trains badly under the phase-1 schedule (validation AP volatile, early stopping cuts it in its first
dip); lr 1e-3 is right (3e-4 too slow, 3e-3 unstable); a 600-epoch run on fold 3 gains the CfC only about 0.05 AP and
the GRU overfits. Conclusion for W: CfC at most equal to GRU. **Best epoch 0 means the weights after the first epoch
were restored, not the untrained weights.**

Cost per window: GRU-W 1k 365 381 MACs (17 state floats), CfC-W 1k 382 095 (15), GRU-W 10k 3.81M; GRU-E 1k 10 275
(15), CfC-E 1k 10 621 (13), GRU-E 10k 116 875, CfC-E 10k 119 856; input operations for E about 3 242 (an estimate);
CNN-1k 77 384 MACs / 1 025 parameters; RFC 4 280 nodes (about 29 KiB, 733 comparisons, 12 573 input operations);
amplitude control 1 194 operations. For E only the features can be updated incrementally; the recurrent part runs
12 steps per window every 10 s (carrying state across windows is S).

## 6. Stress tests (frozen detector + frozen thresholds, perturbed TEST streams, N0 only)

Strict budget, frozen thresholds, recall / false alarms per day, 5 seeds (amplitude control 1 seed):

| | CfC-E 1k | GRU-E 1k | RFC | CNN 1k | Amplitude |
|---|---|---|---|---|---|
| clean | 0.92 / 0.15 | 0.85 / 0.20 | 0.91 / 0.16 | 0.85 / 0.14 | 0.68 / 0.31 |
| gain 0.3 | 0.31 / 0.01 | 0.30 / 0.04 | 0.35 / 0.00 | 0.27 / 0.01 | 0.24 / 0.09 |
| gain 3 | 0.98 / 90.6 | 1.00 / 51.6 | 0.98 / 76.6 | 1.00 / 19.9 | 0.88 / 66.6 |
| white noise 1 | 0.92 / 8.7 | 0.87 / 121.7 | 0.92 / 2.9 | 0.85 / 0.14 | 0.68 / 0.31 |
| matched noise 1 | 0.92 / 0.19 | 0.85 / 0.26 | 0.92 / 0.38 | 0.85 / 0.14 | 0.68 / 0.31 |
| matched noise 2 | 0.93 / 12.8 | 0.90 / 5.2 | 0.93 / 19.4 | 0.85 / 1.7 | 0.69 / 6.1 |
| gaps 10 % | 0.89 / 0.14 | 0.83 / 0.19 | 0.91 / 0.15 | 0.84 / 0.13 | 0.68 / 0.30 |
| jitter 0.25 | 0.90 / 0.12 | 0.83 / 0.17 | 0.90 / 0.14 | 0.82 / 0.12 | 0.67 / 0.31 |
| lowpass 1.3 Hz | 0.92 / 0.14 | 0.83 / 0.16 | 0.92 / 0.15 | 0.84 / 0.13 | 0.68 / 0.31 |
| lowpass 0.8 Hz | 0.66 / 0.03 | 0.48 / 0.11 | 0.60 / 0.01 | 0.51 / 0.04 | 0.56 / 0.31 |

Findings:
1. **Amplitude calibration is the shared failure**: after a gain change every model's frozen threshold breaks (20-90
   FA/day at x3, recall collapse at x0.3). Recall at gain 3 is meaningless (alarms everywhere).
2. **Oracle threshold** (re-chosen on perturbed test data, 2 seeds, diagnostic only): gain 3 is pure calibration for the
   neural models (oracle recall within about 0.07 of clean); RFC cannot reach the budget at gain 3 (13.8 FA/day at the
   top threshold); at gain 0.3 CfC-E, `cfc0E` and RFC lose about 0.3 recall even with a perfect threshold (CNN and
   amplitude control lose nothing); at white noise 1 GRU-E loses 0.25.
3. **White-noise failures were spectral shape, not SNR**: with noise of the same power and the stream's own background
   spectrum (`mnoise`), GRU-E drops from 122 to 0.26 FA/day, CfC-E from 8.7 to 0.19, RFC from 2.9 to 0.38. The CfC-E
   seed split (seeds 2, 3 at about 20 FA/day under white noise) disappears. The explanation (high-frequency content moving
   zero-crossing-type features) is plausible but NOT tested directly.
4. **The CNN's robustness to white noise was spectral blindness**: it detects exactly the same events under gain and
   white noise (0 differences), and under a true SNR loss (matched noise 2) its oracle recall falls 0.87 -> 0.73.
5. At matched noise 2 (SNR loss) all models degrade (oracle recall CfC-E 0.82, `cfc0E` 0.81, RFC 0.82, GRU-E 0.76,
   CNN 0.73, amplitude 0.59); differences not tested for significance.
6. Gaps, jitter and the 1.3 Hz lowpass cost almost nothing (the 1.3 Hz result supports decimation to about 2.65 Hz as a
   resource experiment; true decimated input NOT tested). A 0.8 Hz lowpass hurts everyone.
7. `cfc0E` is as robust to noise as `cfcE`, so the noise robustness of the CfC does not come from the 5 s time step.

Caveats: matched noise is stationary and Gaussian with a day-level spectrum; paired bootstrap exists only for gain 3
and for the in-domain comparisons (commands in section 9 for the rest); oracle rows are never results.

## 7. Frozen model set (artifacts for the Mars test)

`results/models/<config>/s<seed>/fold<k>.pkl`, one file per configuration, seed and fold, each with the weights, the
N0 constant, feature / envelope standardisation, and the thresholds for budgets 0.25 and 2 chosen on the clean
validation streams of that fold. Configurations: `cfcE-1k-N0-long`, `gruE-1k-N0-long`, `cfc0E-1k-N0-long`
(seeds 0-4), `rfc-medium-N0`, `cnn-1k-N0` (seeds 0-4), `amp-rms-N0`, `amp-ptp-N0` (seed 0). DNN is not in the set.
STA/LTA frozen: `data/lunar/derived/baselines/sta_lta_frozen.json` (STA 120 s, LTA 86 400 s, energy, gaps masked;
thresholds 25.02 strict / 17.55 lenient) and `evaluate_frozen`.

Verified: the saving run reproduced the earlier JSON exactly (0.0000 difference in recall and false alarms, all
seeds, so training is bit-reproducible on CPU); a loaded stress run reproduced the training run exactly (recall,
false alarms, per-fold thresholds) and the earlier gain-3 result. Every checkpoint is reload-checked on 100 windows
(`--verify-models`; a mismatch raises, so a finished run means all passed). Checkpoints support N0 and N1 only
(N2 / N3 depend on whole-recording scale tracks) and are pickles: load only your own files. RFC and the amplitude
control are stored by pickling the detector object.

For Mars the 5 folds give 5 trained models per seed that have never seen Mars: apply each fold's detector with its
own thresholds and pool, or choose one policy and declare it (see section 8).

## 8. What the Mars branch needs (decisions BEFORE looking at results)

Proposals, not yet agreed unless stated:

1. **N0 recalibration rule (the largest finding of the stress tests).** The N0 constant is the median per-waveform
   robust std (`estimate_fixed_scale`); under a gain change it scales exactly, so recomputing it with the same
   estimator on the target data restores the normalised input (first-order argument, not tested on real mismatched
   data). Proposal: set `normalizer.fixed_scale` after `load_checkpoint` from InSight quiet-time data with the same
   estimator, declared in advance; the feature standardisation of E stays frozen at the Apollo values. A helper to
   override the constant on a loaded detector is not written yet. Without recalibration expect the gain-row failures.
2. **Common rate and band.** Apollo data are 6.625 Hz, causal Butterworth highpass 0.2 Hz (order 4), windows of 60 s
   (398 samples), step 10 s. InSight must be resampled (anti-aliased) to the same grid and filtered the same way,
   and windows cut identically. Event contrast is strongest at 0.4-0.8 Hz with a sharp drop at about 1.05 Hz; the
   0.8 Hz lowpass test shows content up to about 1 Hz matters. Note E windows always span 12 steps of 1/12 of the
   window, so E is insensitive to the sampling rate itself.
3. **Catalogue, events and labels.** Select MQS events and state which (a reviewer asked for exactly this); own
   labels / onset anchoring; decide whether the detect interval [arrival - 120, arrival + 600] s and the ignore zones
   apply unchanged to marsquakes (different durations and families). No folds on the target: every Mars waveform is
   test only.
4. **Report**: frozen-threshold result AND the recall-versus-false-alarm curve on the target (protocol item 4);
   weak vs usable events; waveform bootstrap; STA/LTA frozen as the reference; the oracle threshold only as a labelled
   diagnostic.
5. **Predictions registered now** (from the stress tests): the N0 constant must be recomputed; spectral mismatch is the
   main risk, with GRU-E most exposed, RFC and the CfC variants moderate, CNN least affected but weaker on weak events;
   no continuous-time advantage is expected on regular sampling.
6. **Reviewer items**: Reviewer 1 asked which MQS events were used; Reviewer 2 wants engagement with Civilini et al.
   (2021) and Dahmen et al. (2022, MarsQuakeNet) and objected that hours-scale timing errors are unacceptable. Here
   delays are capped at +10 min by the detect interval and cannot show hours-scale errors; say so.

## 9. Commands (PowerShell: no backslash continuation; use one line or a trailing backtick)

```
python -m tests.run_dnn_features --only gruE cfcE cfc0E --norms N0 --rec-sizes 1k --rec-epochs 150 --rec-min-epochs 60 --rec-patience 40 --rec-tag long --save-models results/models --verify-models --out results/frozen_e.json
python -m tests.run_dnn_features --only rfc cnn amp --norms N0 --cnn-sizes 1k --seeds 5 --save-models results/models --verify-models --out results/frozen_baselines.json
python -m tests.run_stress --models cfcE:1k gruE:1k rfc:medium cnn:1k amp:rms --perturb clean gain3 --load-models results/models --rec-tag long --out results/stress_loaded.json
python -m tests.analyze_results results/stress_matched.json --compare cfcE-1k-N0-long@noise1 gruE-1k-N0-long@noise1 --compare cfcE-1k-N0-long@mnoise2 gruE-1k-N0-long@mnoise2 --compare cfcE-1k-N0-long@mnoise2 rfc-medium-N0@mnoise2 --out results/analysis_stress_matched
```

Flags: `--rec` (W), `--rec-e` (E: `gruE`, `cfcE`, `cfc0E`), `--only` kinds `gru cfc gruE cfcE cfc0E dnn rfc cnn amp`,
`--rec-sizes`, `--rec-epochs/--rec-patience/--rec-min-epochs/--rec-lr` (recurrent kinds only), `--rec-tag` (config
suffix), `--save-models DIR`, `--verify-models`; stress: `--perturb` names `clean gain0.3 gain3 gain10 noise0.5 noise1
noise2 mnoise0.5 mnoise1 mnoise2 gaps2 gaps10 jitter0.25 lp1.3 lp0.8`, `--oracle`, `--save-models`, `--load-models`.

## 10. Practicalities learned

* Run scripts as modules from the repo root; every package folder needs `__init__.py`.
* `analyze_results` merges records by config name AND seed: runs with different schedules need different tags.
  Paired differences are SECOND minus FIRST of `--compare`. The default reference `amp-rms-N0` does not exist for
  perturbed rows: use `--compare` or `--reference amp-rms-N0@gain3`.
* Config names must match between saving and loading (same `--rec-tag`).
* Every seed is its own `cross_validate_budgets` call (the function selects the best candidate by validation recall).
* The runner writes its JSON only at the end of a run; `run_stress.py` writes after every seed.
* My patched copies of project files have Unix line endings (the originals are CRLF): use `git diff -w`.
* ncps `timespans` shape and `backbone_layers=0` (section 3); `best_epoch 0` meaning (section 5).

## 11. Open items (not done)

1. Irregular-step test for the CfC (dropped sub-window steps with the true elapsed time vs a GRU without it): needs
   `timespans` input in the wrappers and a custom predict path. This is where continuous time could matter.
2. Zero-crossing / spectral-feature ablation of E to test the spectral explanation directly.
3. S variant (stream) and its stress tests.
4. CNN / DNN / RFC on the long schedule; DNN-1k in the frozen set; stress of 10k models; paired bootstrap for the
   remaining stress rows.
5. True decimated input (about 2.65 Hz), a helper to override the N0 constant on a loaded detector, stress with
   N2 / N3 (not supported by checkpoints), downlink metrics, STA/LTA under the stress perturbations (needs the
   continuous detector wired in).
6. User decisions pending: trim file edges from training negatives and the N0 fit; how dead stretches enter labels and
   false-alarm accounting (data branch).

## 12. First message for the new chat

"Same claude.ai Project. Attach `docs/HANDOFF_STRESS_MARS.md`, `src/modules/checkpoints/module.py`,
`src/modules/stress/module.py`, `src/modules/detectors/frozen.py`, `src/modules/evaluation/module.py`,
`src/modules/pipeline/types.py` and `tests/run_stress.py`. We are starting `feat/mars-insight` as described in
section 8. First tell me which InSight / MQS files and catalogue you will provide, then propose the N0 recalibration
rule, the common rate and band, and the event selection and labelling for approval, test-first, in the same style as
the earlier branches (synthetic unit tests, manual runs, handoff note)."