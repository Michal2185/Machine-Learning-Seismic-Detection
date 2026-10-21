# Models plan and kickoff brief

Read together with `docs/DATA_BRANCH_HANDOFF.md` (data, labels, folds, evaluation contract) and
`docs/EVALUATION_BRANCH_HANDOFF.md` (evaluator, protocol, baselines, frozen STA/LTA).

## Research question

Can compact continuous-time networks give a better accuracy-resource-generalization trade-off
than conventional onboard detection under extraterrestrial conditions?

Hypothesis (to be tested, not assumed): at equal or smaller parameter budget, a CfC reaches
detection performance comparable to a CNN / GRU and degrades less on cross-mission and
irregular-sampling tests.

Reference to beat: frozen STA/LTA (STA 120 s / LTA 86,400 s). Nested CV recall 0.67 (strict,
0.30 false alarms per day) and 0.81 (lenient, 0.43 per day). STA/LTA is also close to the
cheapest possible detector, so a network has to win on accuracy at matched false alarms,
delay or generalization, not on size.

## Protocol (unchanged)

Fit on Apollo 12 training / validation waveforms only, freeze, evaluate on unseen waveforms
(nested CV, 5 folds) and later zero-shot. Thresholds from validation under a false-alarm budget;
report strict (0.25 per day) and lenient (2 per day) points and the recall-versus-false-alarm
curve. At least 5 seeds per configuration; mean and spread; paired bootstrap for comparisons.
Models implement `Detector` (`fit(train_windows)`, `score(windows, streams_by_id)`) and run
through `cross_validate_budgets`.

## Who needs what

| Model | Input | Hand-crafted features? | Context seen |
|---|---|---|---|
| STA/LTA (done) | continuous stream | no | up to a whole day |
| RFC (optional) | 6 features of the window | yes | 60 s |
| DNN (MLP) | 6 features of the window | yes | 60 s |
| 1D CNN | normalised 60 s window, 398 samples | no | 60 s |
| GRU (control) | same sequence as the CfC | no | 60 s or stream |
| CfC (main) | same sequence as the GRU, plus time steps | no | 60 s or stream |
| LTC | as CfC, ODE solver (ablation only) | no | as CfC |

Features (thesis, section 5.2.2): signal energy, RMS, peak-to-peak amplitude, crest factor
(peak / RMS), dominant frequency (power spectrum), zero-crossing rate. They are computed on the
NORMALISED window.

Continuous time concerns how time is handled, not which features are used: a CfC takes the
time step between inputs, so it handles irregular sampling (interpolated gaps treated as
missing steps) and a different sampling rate (InSight) without resampling tricks. A single
6-vector per window has no sequence to evolve over; if features are used for the CfC they must
come as a sequence (for example 12 sub-windows of 5 s, each with its features and a time stamp).

## CfC / GRU input variants

* W (window, raw): 398 normalised samples, time step 1 / fs, decision at the last step.
* E (envelope sequence): sub-window features (log energy, RMS, peak-to-peak, ZCR) every few
  seconds; short sequences, cheap.
* S (stream): the hidden state is carried across consecutive 10 s steps over hours (truncated
  back-propagation over segments of a few hours, continuous inference). This is the variant
  that can match the long background memory that makes STA/LTA work, and the one the
  streaming hypothesis refers to.

Every CfC variant gets a GRU with the same input and the same parameter budget (about 1k, 10k
and 100k parameters). Without the GRU control a CfC advantage cannot be attributed to
continuous-time dynamics.

## Input normalisation (shared, first piece to build)

The window sees 60 s, STA/LTA sees hours of background; normalisation is how a window model can
see the background level.

| Variant | Definition | Note |
|---|---|---|
| N0 fixed scale | one constant, fitted on the training waveforms of each fold | keeps amplitude; day-to-day noise level differs 15-300x |
| N1 per window | zero mean, unit std of every window | removes absolute amplitude |
| N2 causal adaptive | divide by a running robust background estimate over hours (gaps masked, slow update while an event is active) | deployable, cross-mission friendly |
| N3 N2 + asinh | compress the dynamic range (event peaks reach 25x the median event peak) | helps recurrent models |

Compare at least N0 and N2 for every model; N1 and N3 as ablations.

## Order of work

0. `feat/input-normalization`: the four variants, the 6-feature extractor, a torch dataset on
   top of `Sampler` (new negatives every epoch), a training harness behind `Detector.fit`
   (weighted loss, early stopping on validation, fixed seeds), resource accounting
   (parameters, int8 size, MACs per window, state memory).
1. `models/dnn-features`: DNN on the 6 features (properly sized; the thesis DNN was 105 MB).
   Smallest end-to-end test of the harness, normalisation and evaluator; first neural number.
   RFC alongside if cheap.
2. `models/cnn`: small 1D CNN on the normalised window.
3. `models/gru-cfc`: GRU and CfC at matched budgets, variants W and E, then S; LTC as ablation.
4. `feat/downlink-metrics`: snippet policy, data sent / wasted / lost, pre-roll.
5. `feat/stress-tests`, `feat/mars-insight`: SNR scaling, gaps and jitter, sampling-rate change,
   zero-shot with frozen detectors; sensitivity of recall to the detect tolerance (+600 vs +900 s).

## Constraints to keep in mind

* Only 49 of 75 events carry positives (198 windows, 4-5 per event); effective independent
  positives are about 49. Small models, strong regularisation; event injection into noise
  (stress branch) may be needed as augmentation.
* Training folds hold about 113-125 positive windows each; differences below about 0.15 recall
  are inconclusive.
* Positives are anchored to the visible onset; delays against the catalogue time include about
  100 s from that definition.

## Starting a new chat

Use the same claude.ai Project (it carries the Project files and memory). Add the three docs to
the Project files. Attach `src/modules/detectors/types.py`, `src/modules/evaluation/cv.py`,
`src/modules/pipeline/types.py` and `src/modules/sampler/module.py` so the interfaces are exact.
First message: "We are starting `feat/input-normalization` as described in `docs/MODELS_PLAN.md`;
implement the normalisation variants and the 6-feature extractor test-first, in the same style
as the data and evaluation branches (manual test scripts, synthetic unit tests, handoff note)."