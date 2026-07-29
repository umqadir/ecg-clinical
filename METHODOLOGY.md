# Methodology

## 1. Study design and question

- Two sequence architectures, trained only on PTB-XL, evaluated on hospitals not seen during training. Primary object of study: cross-hospital shift; internal test performance is the reference point for the shift estimand.
- Design: prospective. Sealed plan: `preregistration/PREREGISTRATION.md`; machine-readable protocol `configs/preregistered_protocol.json`.
- Plan fixes, before any model fitting: targets, splits, architectures, training recipe, metrics, calibration, subgroups, uncertainty method.

Registration seal (`preregistration/SEAL.json`): registration commit `f34661970a4bdf9986b9cf006f6bd8713f017fff`; SHA-256 digests of the analysis plan, protocol, and label manifest; committed before any training, any inference on fold 10, or any external waveform matrix read.

Evaluation seal: covers selected checkpoints, per-label thresholds, and one scalar temperature per architecture; committed to `results/evaluation_choices.json`, recorded in `results/EVALUATION_SEAL.json`, before the single inference pass over the protected cohorts.

Guards (`src/ecg_clinical/integrity.py`, invoked by training, protected inference, and analysis):
- `verify_preregistration_seal`: fails unless every sealed artifact hashes to its registered digest in the working tree; when an object store holding the sealed commit is resolvable, also requires that digest to match the corresponding blob inside the sealed commit.
- `verify_evaluation_seal`: requires the choices file to match its own seal and committed blob, the choices commit to be a descendant of the registration commit, study and registration identifiers to agree across both seals, and `protected_inference_completed: false`; also invokes the registration check.
- Frozen choices file also records digests of normalization constants, protocol, label manifest, waveform audit, each cohort's store files, and each cohort's expected record set; protected inference re-verifies all of these before a model is loaded.
- Scope: catches accidental execution out of order, against wrong inputs, or against drifted artifacts. Does not catch tampering by someone with repository write access; not externally anchored or notarized.

Digest verifiability: committed and independently recomputable: registration seal, protocol, label manifest, frozen choices file, waveform audit. Not committed, for file size: checkpoint digests in `results/EVALUATION_SEAL.json` and store digests in the frozen choices file (internally consistent, unverifiable externally without a full retrain and rebuild). Committed at `results/inference_receipts/`: SHA-256 digest of every per-seed prediction file, cohort ensemble, target array, and selected checkpoint, for each of the three sealed inference passes.

History rewrite: both seal files record `history_rewritten: true` (working history squashed before publication). Under that flag, working-tree digests are verified unconditionally, and blob comparison against the sealed commit runs only if an object store holding it is resolvable. In this repository both sealed commits are present in `.git` as unreachable objects, so blob comparison runs and passes and the registration commit is confirmed an ancestor of the choices commit. `git clone`/`git push` do not transfer unreachable objects and garbage collection will eventually prune them, so a published clone carries the working-tree digest check only: it catches an artifact edited after sealing but not one edited in step with its seal entry. Both seal files record that the guarantee differs between this working copy and a clone.

Quotation convention: `preregistration/PREREGISTRATION.md`, `configs/preregistered_protocol.json`, `preregistration/SEAL.json`, `results/EVALUATION_SEAL.json` are frozen and never edited (the seal guards fail otherwise). Quotations of them in this file, `README.md`, `RESULTS.md`, or `reviews/pre_seal_protocol_review.md` reproduce the registered wording verbatim, exempt from this document's prose conventions.

`commit_store` anomaly: registration seal records `commit_store: ".git-shadow"`. At registration the execution environment exposed the primary `.git` directory read-only, so the pre-registration was committed to a shadow object store before any protected access. That store was later reconciled into the normal repository and archived; sealed commit `f34661970a4b` is present in `.git` and the guard resolves it there. It is not an ancestor of current history, which was squashed to a single commit before publication. The `commit_store` field is left exactly as sealed and names a store that no longer exists.

## 2. Data sources and versions

| Source | Version | License | Role |
|---|---|---|---|
| PTB-XL | 1.0.3 | CC BY 4.0 | train, validation, internal test |
| PhysioNet/CinC Challenge 2021 | 1.0.3 label snapshot | CC BY 4.0 | label harmonization; external cohort labels |
| Chapman-Shaoxing / Ningbo (ECG Arrhythmia) | 1.0.0 | CC BY 4.0 | external cohorts |

- None of the three requires credentialed access.
- PTB-XL 1.0.3: 21,799 records, 18,869 patients.
- Challenge snapshot: 21,837 PTB-XL headers, joined by ECG ID. 38 records absent from PTB-XL 1.0.3 (removed by PTB-XL as duplicates of retained waveforms, enumerated in the label manifest).
- Chapman-Shaoxing: 10,247 records. Ningbo: 34,905 records. Combined external: 45,152 records.
- External cohorts evaluated separately at every stage; never pooled with each other or with PTB-XL. Neither contributes to training, selection, thresholds, or calibration.
- Challenge release used in place of the combined ECG Arrhythmia provenance card.

## 3. Label harmonization

- Mapping target: Challenge 2021 scored code space, fixed at mapping commit `e2a75fc01f729cb74cc4e853e054ce81e28381fc` of `physionetchallenges/evaluation-2021`.
- PTB-XL SCP statements and external SNOMED-coded headers harmonized into that space.
- 30 raw SNOMED codes collapse to 26 groups under four equivalences: complete/unqualified LBBB; complete/unqualified RBBB; premature atrial contraction/supraventricular premature beats; premature ventricular contractions/ventricular premature beats.
- 17 of 26 groups have positive support in all three sources simultaneously.
- Excluded before training: `251146004`. Chapman-Shaoxing maps both poor R-wave progression and low QRS voltage to this code; Ningbo maps poor R-wave progression separately under `365413008`.
- Remaining 16 groups are model targets; all 16 are trained.
- Headline set: 13 of 16, selected by a registered support rule: at least 100 positives in the training folds, and at least 25 positives and 25 negatives in each of fold 9, fold 10, Chapman-Shaoxing, and Ningbo.

| Label key | Diagnosis | Train pos | Val pos | Test pos | Chapman pos | Ningbo pos | Headline |
|---|---|---:|---:|---:|---:|---:|---|
| 164890007 | atrial flutter | 59 | 7 | 7 | 445 | 7615 | no |
| 733534002\|164909002 | complete LBBB / LBBB | 428 | 54 | 54 | 205 | 248 | yes |
| 713427006\|59118001 | complete RBBB / RBBB | 433 | 55 | 54 | 454 | 1291 | yes |
| 270492004 | first-degree AV block | 636 | 80 | 79 | 247 | 893 | yes |
| 39732003 | left axis deviation | 4129 | 489 | 528 | 382 | 1163 | yes |
| 164947007 | prolonged PR interval | 272 | 34 | 34 | 12 | 40 | no |
| 111975006 | prolonged QT interval | 95 | 12 | 11 | 57 | 337 | no |
| 698252002 | nonspecific IVCD | 630 | 78 | 79 | 235 | 536 | yes |
| 426783006 | sinus rhythm | 14452 | 1784 | 1822 | 1826 | 6299 | yes |
| 284470004\|63593006 | PAC / SVPB | 446 | 55 | 54 | 258 | 1063 | yes |
| 164917005 | abnormal Q wave | 438 | 55 | 55 | 235 | 828 | yes |
| 47665007 | right axis deviation | 281 | 36 | 26 | 215 | 638 | yes |
| 426177001 | sinus bradycardia | 509 | 64 | 64 | 3889 | 12670 | yes |
| 427084000 | sinus tachycardia | 661 | 83 | 82 | 1568 | 5687 | yes |
| 164934002 | T-wave abnormality | 1875 | 235 | 231 | 1876 | 5167 | yes |
| 59931005 | T-wave inversion | 235 | 30 | 29 | 157 | 2720 | yes |

- Non-headline exclusions and reason: atrial flutter (59 training positives), prolonged PR interval (12 Chapman-Shaoxing positives), prolonged QT interval (95 training positives).
- All three remain training targets; reported under sparse-coverage analysis; never in a headline metric.

## 4. Preprocessing

Input format:
- Leads: 12, fixed order I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
- Units: physical millivolts
- Sampling rate: 100 Hz
- Record length: 10 s
- Model input window: 2.5 s (250 samples)

External resampling:
- Source rate: 500 Hz
- Method: polyphase resampling, `scipy.signal.resample_poly`, up=1, down=5, applied before normalization
- Output shape: 12 x 1000, matching PTB-XL

Normalization:
- Per-lead mean/SD computed on training folds only, over 17,418,000 per-lead samples of the 17,418 training records
- Stored at `data/derived/training_normalization.json`
- Same constants applied unchanged to fold 9, fold 10, and both external cohorts
- No statistic from fold 9, fold 10, or either external cohort enters training or preprocessing
- No external adaptation, no external recalibration
- Demographics are never model inputs

Waveform validation (post-registration, pre-inference), required per record:
- 12 leads, 10 s duration, finite physical values throughout, interpretable gain and unit metadata

| Cohort | Records retained |
|---|---|
| PTB-XL | 21,799 |
| Chapman-Shaoxing | 10,247 |
| Ningbo | 34,905 |

- Attrition: zero in all three cohorts.
- Total stored samples finite: 803,412,000.
- Digests, ranges, and label-count reconciliation: `data/derived/waveform_audit.json`.
- Preserved source anomaly, not repaired: Ningbo header `S23074` declares checksummed matrix `JS23074.mat`; the header ID remains the record ID.

## 5. Splits

Split source: PTB-XL recommended stratified fold assignment, used unchanged.

| Split | Folds | Records | Patients |
|---|---|---|---|
| Train | 1-8 | 17,418 | 15,023 |
| Validation | 9 | 2,183 | 1,942 |
| Internal test | 10 | 2,198 | 1,904 |

- Patient cross-fold leakage: 0 of 18,869 patients.

## 6. Architectures

- Both implemented from scratch in `src/ecg_clinical/models.py`, not loaded from released checkpoints.
- Both take identical inputs and produce identical 16-target outputs.

### xResNet1d101 (1,873,680 trainable parameters)

| Component | Value |
|---|---|
| Stem | 3 convolutions, base width 64, followed by max pooling |
| Stages | 4 bottleneck stages, block layout (3, 4, 23, 3) |
| Expansion factor | 4 |
| Kernel size | 5 |
| Shortcuts | anti-aliased average-pooling |
| Final-block batch-norm scale | zero-initialized |
| Head pooling | concatenated adaptive mean + max pooling over time |
| Head layers | batch norm, dropout 0.25, linear(128), ReLU, batch norm, dropout 0.5, linear output |

Stage-width audit:
- Stage widths held at 64 across all four stages, not the ImageNet-style doubling to [64, 128, 256, 512].
- Checked against `helme/ecg_ptbxl_benchmarking` commit `cdbf4e66d7e57d9b6a2657b6024716212b8d0afa`: in `code/models/xresnet1d.py` the `[64,128,256,512]` line is commented out and `[64,64,64,64]` is active.
- Parameter count pinned by test at 1,873,680 for the 16-output head.

### S4D classifier (2,165,264 trainable parameters)

| Component | Value |
|---|---|
| Encoder | pointwise Conv1d, 12 to 512 channels |
| Blocks | 4 bidirectional residual S4D blocks |
| Model dimension | 512 |
| Real state dimension | 8 |
| Activation | GELU |
| Dropout | 0.2 |
| Output projection | gated linear unit |
| Normalization | post-residual LayerNorm |
| Pooling | temporal mean pooling |
| Head | linear multilabel |
| Kernel application | FFT convolution, both directions fused into one zero-padded two-sided kernel |

State-representation audit:
- Reference: official S4D formulation, `state-spaces/s4`, commit `e757cef57d89e448c413de7325ed5601aceaac13`.
- Real state dimension 8 represented as four complex conjugate modes; kernel taken as twice the real part over conjugate pairs.
- Implementation enforces even real state size, retains the exact zero-order-hold diagonal discretization.
- Per-channel timescales sampled log-uniformly between 1e-3 and 1e-1.

S4D vs. full S4:
- S4D selected over full S4 as a registered portability decision: the released full S4 implementation requires a custom CUDA Cauchy kernel extension, unsupported on Apple Metal.
- Model here is an extension-free diagonal approximation at the published ECG block scale, named as such throughout.

## 7. Training protocol

| Parameter | Value |
|---|---|
| Loss | unweighted binary cross-entropy with logits, all 16 targets |
| Seeds | 17, 29, 43 |
| Max epochs | 50 (fixed) |
| Crop | one uniformly random 2.5 s crop per record per epoch, seeded from run seed + epoch index + record index |

| Architecture | Optimizer | Schedule | Learning rate | Weight decay | Batch size |
|---|---|---|---|---|---|
| xResNet1d101 | AdamW | one-cycle | 0.01 max | 0.01 | 128 |
| S4D | AdamW | constant | 0.001 | 0.01 | 32 |

Checkpoint selection:
- Retained checkpoint: epoch with highest fold-9 macro-AUROC over the 13 headline labels; ties broken by lower fold-9 negative log-likelihood.
- Selection uses only architecture/seed checkpoints; fold 10 and external cohorts are never consulted.

Execution:
- Hardware: Apple M4, PyTorch MPS backend.
- `torch.use_deterministic_algorithms(True)`; thread counts pinned.
- Each run writes a manifest recording device, torch version, trainable parameter count, record counts, and digests of the protocol, label manifest, and normalization files.
- Checkpointing per epoch; runs resumable; an interruption costs at most one epoch.
- Training takes an exclusive device lock by default, disabled with `--share-device`. Lock affects scheduling only; results do not depend on it.

## 8. Evaluation protocol and the seals

Inference:
- Each record scored on 10 deterministic equidistant 2.5 s windows spanning the full 10 s; per-window probabilities averaged.
- Architecture prediction = arithmetic mean of its 3 seed predictions.
- Batch size affects throughput only; window averaging is strictly per record.

Freeze step (`scripts/freeze_evaluation_choices.py`):
- Reads fold-9 predictions only.
- Writes per architecture: 3 run identifiers, checkpoint digests, per-label thresholds, one scalar temperature, digests of validation targets and record order.
- File committed and sealed before `scripts/run_protected_inference.py` may run.

Protected inference (`scripts/run_protected_inference.py`):
- One pass per cohort.
- Verifies both seals; re-verifies each checkpoint against its sealed digest; refuses a cohort whose receipt already reports completion; writes a receipt recording the sealed choices commit and prediction digests.
- No result-driven iteration after the seal.
- An error found afterward is reported as a deviation with the invalid output retained; never silently substituted.

## 9. Metrics

Primary:
- Term-centric macro-AUROC over the 13 headline labels, per cohort.
- Shift delta: external macro-AUROC minus fold-10 macro-AUROC, per hospital.
- Model contrast: S4D minus xResNet1d101 macro-AUROC, within each cohort.

Secondary:
- macro-AUPRC, micro-AUROC, micro-AUPRC, per-label AUROC and AUPRC.
- All 16 targets reported with coverage; the three sparse targets flagged.
- Thresholded metrics (fold-9 thresholds: smallest value attaining maximum fold-9 F1 per label): macro-F1, micro-F1, sensitivity, specificity, per cohort.

Undefined-value handling:
- AUROC on a stratum with no positives or no negatives is reported as undefined, never as 0.5 or 0.
- Macro averages are taken over defined labels only; the contributing-label count travels with the estimate.

## 10. Calibration

- Method: one positive scalar temperature per architecture ensemble, fitted on fold 9 alone by bounded minimization of binary negative log-likelihood over log temperature, applied to the logit of the mean seed probabilities.
- No recalibration performed on either external cohort in the primary protocol.
- Primary metric: macro average, over headline labels, of per-label expected calibration error, 15 equal-width bins on the unit interval.
- Secondary metrics: pooled expected calibration error, macro Brier score, per-label expected calibration error.
- Pooled and per-label reliability curves reported for every cohort and architecture, before and after temperature scaling.

## 11. Subgroups

- Age strata: `under_40`, `40_to_64`, `65_plus`. PTB-XL sentinel age value 300 (90-and-older marker) assigned to `65_plus`.
- Sex strata: female, male.
- Records missing a stratifier excluded only from analyses using that stratifier.
- Eligible-label rule: at least 10 positives and 10 negatives in every subgroup of a stratification, per cohort; counts generated at registration in `subgroup_label_counts.csv`.
- Eligible set differs by cohort and stratification; each subgroup result reports the contributing-label count.
- Reported quantities: macro-AUROC, macro-AUPRC, macro per-label expected calibration error, Brier score.

## 12. Uncertainty

- Method: nonparametric cluster bootstrap, 2,000 replicates, percentile bounds at the 2.5th and 97.5th percentiles.
- Base seed: 20260716, distinct derived seed per cohort and stratum.
- Resampling unit: patient cluster for PTB-XL; record for the external cohorts (each source record is one patient ECG).
- Replicates shared across architectures within a cohort, so model contrasts are paired.
- Shift deltas and hospital-delta differences formed replicate-wise from independent cohort draws.
- AUROC computed directly on resample weights, not materialized resamples: scores sorted once, tied scores grouped, each positive group credited the negative weight strictly below it plus half the tied negative weight.
- Percentile intervals are not constructed to contain their own point estimate. Expected calibration error is positively biased at small effective sample size; a bootstrap resample holds about 63 percent of its clusters distinct, so every replicate is computed at a smaller effective sample than the full cohort. Where the error is small and the cohort is small, the replicate distribution can sit entirely above the point estimate. Percentile intervals were registered and are kept as specified.
- `scripts/diagnose_ece_bootstrap_bias.py` measures this effect; writes `results/ece_bootstrap_bias_diagnostic.json`. Exploratory; changes no registered quantity.

## 13. Deviations from the registration

Deviation 001: PTB-XL sex encoding.
- Cause: the metadata-only registration generator read PTB-XL `sex=0` as female and `sex=1` as male; PTB-XL convention is 0=male, 1=female.
- Release 1.0.3: 11,354 zero-coded records (52.1%), 10,445 one-coded records (47.9%), matching the dataset's documented 52% male / 48% female.
- Found after registration and waveform validation, before any fold-10 or external model inference.
- Effect: confined to the row labels of two PTB-XL subgroup rows in the sealed `subgroup_label_counts.csv` (female/male exchanged). Eligibility invariant to the exchange, since the common-label rule requires the support threshold in both sex groups.
- Unaffected: waveforms, targets, training, thresholds, calibration, headline metrics, age subgroups, external cohorts' sex subgroups (Challenge header strings were read correctly).
- Sealed artifacts left unchanged. Analysis code maps PTB-XL 0 to male, 1 to female. Every PTB-XL sex subgroup row carries a correction flag (`deviations/001_ptb_sex_encoding.md`).

## 14. Reproducibility

Environment (pinned in `pyproject.toml` and `uv.lock`):
- Python 3.12, torch 2.12.1, numpy 2.2.6, scipy 1.16.3, scikit-learn 1.8.0, pandas 2.3.3, wfdb 4.3.1
- Source metadata files: SHA-256 digests in the label manifest
- Waveform stores: digest in the waveform audit

Pipeline order:

1. `scripts/fetch_source_metadata.py` and `scripts/fetch_challenge_headers.py`, then `scripts/build_preregistration_artifacts.py`, for the label manifest, counts, mapping audit, and subgroup counts.
2. `scripts/build_waveform_store.py` and `scripts/audit_waveform_stores.py`, to stream and validate the 100 Hz stores.
3. `scripts/compute_training_statistics.py`, for the training-fold normalization constants.
4. `scripts/train_all.py`, driving `scripts/train_model.py` over the six architecture and seed combinations.
5. `scripts/freeze_evaluation_choices.py`, then commit and seal.
6. `scripts/run_protected_inference.py`, once per cohort (`ptb_test`, `chapman_shaoxing`, `ningbo`).
7. `scripts/analyze_results.py` and `scripts/plot_results.py`, writing the metric tables, uncertainty intervals, bootstrap distributions, reliability bins, an analysis manifest recording both seal commits, and the five result figures.

Stages 8-14 (optional, exploratory, sit outside the frozen analysis):

| Stage | Script | Added | Function | Outputs (`results/...`) |
|---|---|---|---|---|
| 8 | `diagnose_ece_bootstrap_bias.py` | n/a | characterizes finite-sample ECE bias under the registered bootstrap; reports nothing registered | `ece_bootstrap_bias_diagnostic.json` |
| 9 | `compute_per_label_intervals.py` | 2026-07-25 | attaches percentile intervals to per-label AUROC estimates and shift deltas; rebuilds registered resamples; aborts unless macro-averaged replicates reproduce the registered macro-AUROC distribution bit for bit; intervals marginal, uncorrected | (none listed here) |
| 10 | `exploratory_label_commensurability.py` | 2026-07-26 | tests whether the 13 headline labels denote the same finding in the three cohorts; label-density/co-occurrence computed from label matrices alone, no model output; two reconciled estimands rescore sealed predictions under a redefined target; competing-label sets fixed on the clinical relation between SNOMED groups, 2 of 4 are negative controls; reconciled estimands reported beside the registered ones, never in place of them | `exploratory_label_structure.csv`, `exploratory_label_cooccurrence.csv` (13x13 per cohort), `exploratory_reconciled_auroc.csv`, `exploratory_reconciled_shift_deltas.csv`, `exploratory_macro_variants.csv`, `exploratory_label_commensurability.json` |
| 11 | `exploratory_transfer_mechanism.py` | 2026-07-26 | attributes each per-label shift delta to the positive or negative class by a Shapley split over 4 AUROCs from the two cohorts' positive/negative record sets, summing exactly to the registered delta; rank-matched variant removes score-distribution differences; also tests prevalence shift, demographic shift (standardized to PTB-XL fold 10 age/sex mix), SNOMED code composition for the 2 bundle branch block groups; reports 18 per-seed macro-AUROCs | `exploratory_class_side_decomposition.csv`, `exploratory_demographic_standardization.csv`, `exploratory_bundle_branch_codes.csv`, `exploratory_bundle_branch_scores.csv`, `exploratory_seed_spread.csv`, `exploratory_macro_contributions.csv`, `exploratory_transfer_mechanism.json` |
| 12 | `exploratory_clinical_utility.py` | 2026-07-26 | net benefit over a 50-point threshold grid, 0.01 to 0.50, vs. the better of treat-all/treat-none, on calibrated probabilities (uncalibrated alongside); 2 threshold sets: sealed fold-9 F1 set, exploratory fold-9 sensitivity-0.90 set; neither derived from a test or external cohort | `exploratory_decision_curves.csv`, `exploratory_operating_points.csv`, `exploratory_clinical_utility.json` |
| 13 | `exploratory_calibration_ladder.py` | 2026-07-26 | 9 logit transforms scored with the registered estimator; rungs 3-6 fit on the evaluation cohort's own labels (oracle upper bounds, not deployable methods); rungs 7-9 deployable, not registered, selected after the registered outcome was known; aborts unless rung 1 and rung 2 reproduce the registered uncalibrated/calibrated values | `exploratory_calibration_ladder.csv`, `exploratory_calibration_ladder.json` |
| 14 | `exploratory_feature_baseline.py` | 2026-07-26 | 1 L2-penalized logistic regression per target, 22 handcrafted waveform measurements, one code path, no per-cohort tuning; standardization constants, regularization strength, coefficients from PTB-XL folds 1-8 fitting, fold 9 selecting; fold 10 and both external cohorts scored once; verifies registration seal only (reads no sealed prediction file); reads `exploratory_per_label_shift_deltas.csv` for the model comparison, left untouched | `exploratory_feature_baseline.csv`, `exploratory_feature_baseline.json` |

Exploratory-stage properties:
- Stages 10-14 change no registered quantity, rewrite no sealed artifact, re-run no inference.
- Stages 9-13 rebuild the registered cluster resamples from the registered per-cohort seeds, cluster units, replicate count, and batch size; abort unless those resamples reproduce the registered macro-AUROC distribution in `results/bootstrap_distributions.npz` bit for bit. Stage 13 additionally requires its rung 1 and rung 2 macro error replicate distributions to match the registered ones bit for bit.
- Stage 14 uses the same bootstrap design but different resample values, since a different model's scores are resampled on the same records.
- Outputs of stages 9, 10, 12, 13, and 14 are committed under `results/`. Stage 11 writes its outputs when it runs; they are not committed.
- Every interval these stages produce is marginal and uncorrected, matching the registered plan.

Seal enforcement:
- Steps 4, 6, and `analyze_results.py` (step 7): require the registration seal. Steps 6 and `analyze_results.py`: additionally require the evaluation seal.
- Steps 9, 10, 11, 12, 13: require both seals. Step 14: requires the registration seal only.
- Steps calling neither guard: `scripts/plot_results.py` (reads the metric tables `analyze_results.py` wrote); step 8 (aborts unless its regenerated bootstrap reproduces the percentiles in `results/uncertainty.json` within a fixed tolerance, a weaker check than the guards perform).

Test coverage:
- Suite covers: harmonization rules, data pipeline, both parameter counts, metrics, bootstrap, both integrity guards, freeze step, protected inference, source metadata digests, and each exploratory stage.
- Exploratory-stage test counts: 5 (per-label intervals), 16 (two shift-diagnosis stages together), 17 (clinical utility), 23 (calibration ladder), 30 (feature baseline).
- Total: 202 tests, runs from a clean checkout with no data present.

Source acquisition:
- No dataset file is committed. Step 1 fetches from primary sources over unauthenticated HTTPS.
- `scripts/fetch_source_metadata.py` writes 5 files, 23.6 MB total, under `data/raw-metadata/`: the PTB-XL 1.0.3 record database and release checksums, the ECG Arrhythmia 1.0.0 release checksums, and the Challenge 2021 weights matrix and scored-code mapping at commit `e2a75fc0`. Verifies each download against the digest recorded in the sealed label manifest, or, for the two release checksum files, against the digest observed at registration. Idempotent.
- `scripts/fetch_challenge_headers.py` retrieves 66,989 `.hea` header files; requests no signal-bearing file.
- Step 2 streams waveforms from `physionet-open.s3.amazonaws.com`, checks every record against the release checksums, produces 1.5 GB of float16 stores at 100 Hz.

Compute:
- Hardware: one Apple M4, PyTorch MPS backend (`device: mps` in every run manifest).
- Training: six architecture/seed combinations, run sequentially, frozen 50-epoch maximum, 23.5 hours wall clock total.
- Per-epoch elapsed time, from each run's `history.json`: xResNet1d101 (batch 128) 18-104 s median; S4D (batch 32) 385-441 s median. Within-architecture spread attributed to contention with unrelated jobs on the same machine.
- Protected inference: three passes, back to back, same device, on 2026-07-20, six checkpoints applied per cohort.

| Cohort | Records | Duration |
|---|---|---|
| PTB-XL fold 10 | 2,198 | 4 minutes |
| Chapman-Shaoxing | 10,247 | 24 minutes |
| Ningbo | 34,905 | 80 minutes |

- Analysis and figures: CPU-only, dominated by the 2,000-replicate bootstrap.
- Device lock: exclusive, path fixed at `/tmp/ml-train.lock` in the registered protocol; overridden by `scripts/run_protected_inference.py --training-lock`. Affects scheduling only, never a computed value.

Results are reported in `RESULTS.md`, which draws every number from the artifacts written by step 7.

## 15. Benchmark context

Three published results are quoted for context. None is a comparator for this study. Each quoted value is labeled with its source table, since two of the three papers also print columns from leakage or oracle conditions.

Architecture-family benchmarks:

| Source | Table | Model | Task | Value |
|---|---|---|---|---|
| Strodthoff et al. (arXiv:2004.13701; DOI 10.1109/JBHI.2020.3022989) | Table 2 | xResNet1d101 | native 71-statement PTB-XL "all", 100 Hz, patient-disjoint stratified folds | macro-AUROC 0.925 (0.008) |
| Strodthoff et al. | Table 2 | heterogeneous ensemble | same | macro-AUROC 0.929 (0.008) |
| Strodthoff et al. | Table 6, `rnd. 100Hz` column | xResNet1d101 | same task, random (non-patient-disjoint) train-test split | macro-AUROC 0.929 (0.007); leakage demonstration, not a benchmark value |
| Strodthoff et al. | Table 6, patient-disjoint column | xResNet1d101 | same | macro-AUROC 0.928 (0.006) |
| Mehari and Strodthoff (arXiv:2211.07579) | Table 1 | full S4, 4 S4 blocks + 1 fully connected head, supervised signal-only | same native task | mean macro-AUROC 0.9417, SD 0.0016, 10 runs |

- Registration text states the uncertainty on both Strodthoff values as 0.007; Table 2 prints 0.008 for both. The 0.007 figure matches only the Table 6 `rnd. 100Hz` leakage-column entry. `preregistration/PREREGISTRATION.md` is sealed by digest and left exactly as registered. Values quoted in this section are the Table 2 values.
- Both architecture-family numbers use the native 71-statement task over the original 21,837-record release. This study's headline uses a 13-label subset of the harmonized Challenge 2021 SNOMED scored space on the duplicate-cleaned PTB-XL 1.0.3 release: a different task on a different label space.
- The S4D model here is a diagonal, extension-free approximation at the published block scale, not the full S4 model that produced 0.9417; no result here reproduces that number.

Size of the transfer loss, Leinonen et al. 2024 (Computers in Biology and Medicine, 183: 109271; DOI 10.1016/j.compbiomed.2024.109271; arXiv:2403.15012):

- Table 4, single-source experiment, ResNet trained on PTB and PTB-XL: macro-AUC mean error 0.0529, SD 0.0263, RMSE 0.0591. Mean error = 5-fold cross-validation estimate minus test score, averaged over held-out sources (positive value = optimism).
- Table 5, multi-source experiment, ResNet: 4-fold column (pools all sources before splitting) macro-AUC mean error 0.1121; leave-source-out column macro-AUC mean error -0.0061.
- No cross-validation estimate from either table is quoted here as a performance value; only the mean-error values are quoted.
- Figure 3 readings, not printed in the paper text, used in `RESULTS.md` section 11: ResNet trained and 5-fold cross-validated within Chapman-Shaoxing and Ningbo reaches macro-AUC of about 0.97; ResNet trained on PTB and PTB-XL, tested on Chapman-Shaoxing and Ningbo, reaches about 0.91. Reading accuracy about 0.005; scale fixed by panel gridlines.
- Cross-check: 5-fold bar minus the mean of that panel's four test-source bars gives 0.0438 for the Chapman-Shaoxing/Ningbo training source against the printed Table 4 value 0.0434, and 0.0531 for the PTB/PTB-XL training source against the printed 0.0529.
- Label space, sources, and architecture all differ from this study's; these values bound plausibility and are not differenced against any number reported here.

## 16. Data sources, attribution, and references

- All three data sources: Creative Commons Attribution 4.0 International Public License (https://creativecommons.org/licenses/by/4.0/); none requires credentialed access. No dataset file is redistributed here.
- Committed artifacts under `data/derived/` are adapted material under that license: aggregate counts and constants computed from PTB-XL 1.0.3 metadata and the Challenge 2021 header snapshot; no record-level data. Adaptations: the SNOMED harmonization of section 3, the per-cohort and per-subgroup positive and negative counts, the banded age and sex tabulation, the per-lead normalization constants of section 4, and the waveform store audit of section 4.
- The code in this repository is separately licensed under the MIT License in `LICENSE`.

Data resources:

- Wagner P, Strodthoff N, Bousseljot R, Samek W, Schaeffter T. PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet, 2022. DOI 10.13026/kfzx-aw45.
- Reyna MA, Sadr N, Gu A, Perez Alday EA, Liu C, Seyedi S, Shah A, Clifford G. Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021 (version 1.0.3). PhysioNet, 2022. DOI 10.13026/34va-7q14.
- Zheng J, Guo H, Chu H. A large scale 12-lead electrocardiogram database for arrhythmia study (version 1.0.0). PhysioNet, 2022. DOI 10.13026/wgex-er52.

Original publications for those resources:

- Wagner P, Strodthoff N, Bousseljot RD, Kreiseler D, Lunze FI, Samek W, Schaeffter T. PTB-XL: A large publicly available ECG dataset. Scientific Data, 2020. DOI 10.1038/s41597-020-0495-6.
- Reyna MA, Sadr N, Perez Alday EA, Gu A, Shah AJ, Robichaux C, Bahrami Rad A, Elola A, Seyedi S, Ansari S, Ghanbari H, Li Q, Sharma A, Clifford GD. Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021. Computing in Cardiology, 2021; 48: 1-4. DOI 10.23919/CinC53138.2021.9662687.
- Zheng J, Zhang J, Danioko S, Yao H, Guo H, Rakovski C. A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients. Scientific Data, 2020; 7(1): 48. DOI 10.1038/s41597-020-0386-x.
- Zheng J, Chu H, Struppa D, Zhang J, Yacoub SM, El-Askary H, Chang A, Ehwerhemuepha L, Abudayyeh I, Barrett AS, Fu G, Yao H, Li D, Guo H, Rakovski C. Optimal Multi-Stage Arrhythmia Classification Approach. Scientific Reports, 2020; 10: 2898. DOI 10.1038/s41598-020-59821-7.
- Pollard T, Moody BE, Lehman L, Gow B, Fernandes C, Xie C, Johnson A, Mark RG, Heldt T. PhysioNet as a global platform for biomedical research. Nature Health, 2026. DOI 10.1038/s44360-026-00096-z.

Label space and mapping tables:

- PhysioNet/Computing in Cardiology Challenge 2021 evaluation code, `physionetchallenges/evaluation-2021`, commit `e2a75fc01f729cb74cc4e853e054ce81e28381fc`. The scored-code weights matrix and diagnosis mapping table define the 26 scored groups used in section 3. Neither file is committed here; `scripts/fetch_source_metadata.py` retrieves both at that commit.

Reference implementations that the architectures follow:

- Strodthoff N, Wagner P, Schaeffter T, Samek W. Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL. IEEE Journal of Biomedical and Health Informatics, 2021; 25(5): 1519-1528. DOI 10.1109/JBHI.2020.3022989. Repository `helme/ecg_ptbxl_benchmarking` at commit `cdbf4e66d7e57d9b6a2657b6024716212b8d0afa` fixes the xResNet1d101 stage widths used here.
- Mehari T, Strodthoff N. Advancing the State-of-the-Art for ECG Analysis through Structured State Space Models. arXiv:2211.07579, 2022.
- Gu A, Goel K, Re C. Efficiently Modeling Long Sequences with Structured State Spaces. International Conference on Learning Representations, 2022. arXiv:2111.00396. The Apache-2.0 reference implementation at commit `e757cef57d89e448c413de7325ed5601aceaac13` supplies the S4D kernel definition reimplemented in `src/ecg_clinical/models.py`.

Benchmark context for the size of the transfer loss:

- Leinonen T, Wong D, Vasankari A, Wahab A, Nadarajah R, Kaisti M, Airola A. Empirical investigation of multi-source cross-validation in clinical ECG classification. Computers in Biology and Medicine, 2024; 183: 109271. DOI 10.1016/j.compbiomed.2024.109271; arXiv:2403.15012. Section 15 quotes the macro-AUC mean errors for the PTB and PTB-XL and the Chapman-Shaoxing and Ningbo training sources from their Table 4, the ResNet macro-AUC mean errors from their Table 5, and reads two values off their Figure 3.
