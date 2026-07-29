# Preregistration: External Generalization and Architecture Comparison for 12-Lead ECG Diagnostic Classification

**Date of registration: 2026-07-16**

## Registration status

This document is a prospective analysis plan. It is committed before any model training, before any inference on PTB-XL fold 10, and before any inference on the external cohorts. No project results exist at the time of writing. No project performance numbers, thresholds, or calibration parameters have been observed. All 66,989 source headers have been parsed for label and demographic counts, but no external waveform matrix has been accessed.

All quantitative provenance cited below is fixed at registration. The harmonized label definitions, cohort counts, mapping decisions, and subgroup counts are generated deterministically and stored in:

- `data/derived/preregistration/harmonized_labels.json` (SHA256 `3f553708a045c6b1f08203d13b5dbff7ee41e8ad3f21401e48a2b900a7d5daf7`)
- `data/derived/preregistration/harmonized_label_counts.csv`
- `data/derived/preregistration/label_mapping_audit.csv`
- `data/derived/preregistration/subgroup_label_counts.csv`
- `configs/preregistered_protocol.json`

Every count reproduced in prose in this document is drawn from those files. Where prose and files disagree, the files are authoritative.

## Question and hypotheses

The study asks how well two distinct sequence architectures, trained only on PTB-XL, transfer to two independent hospital cohorts that were never seen during training, and whether the two architectures rank the same way internally and externally.

The registered hypotheses are:

1. Both architectures lose headline macro-AUROC when moved from the internal PTB-XL test fold to the external cohorts, and the size of that loss differs between the two hospitals.
2. The S4D ensemble exceeds the xResNet ensemble on the internal test fold and on each external hospital.
3. External calibration worsens relative to internal calibration. A single scalar temperature fitted on fold 9 improves calibration on the internal test fold, but no assumption is made that it improves calibration externally.
4. Discrimination and calibration vary across age groups and across sex.

## Sources and splits

Three public sources are used. PTB-XL is the only training source. The external cohorts are used for inference only and are never pooled with PTB-XL or with each other.

PTB-XL version 1.0.3 contains 21,799 records from 18,869 patients, after removal of 38 records that were duplicated relative to version 1.0.1. The recommended stratified fold assignment is used unchanged. Training uses folds 1 through 8, comprising 17,418 records from 15,023 patients. Validation uses fold 9, comprising 2,183 records from 1,942 patients. The held-out internal test set is fold 10, comprising 2,198 records from 1,904 patients. Patient identity does not cross folds.

The CinC Challenge 2021 label snapshot provides 21,837 headers, which are joined by ECG ID to the current PTB-XL records to supply the scored diagnostic codes.

Chapman-Shaoxing contributes 10,247 records as one external cohort. Ningbo contributes 34,905 records as the second external cohort. These two cohorts are analyzed separately at all stages.

Data sources:

- PTB-XL 1.0.3: https://physionet.org/content/ptb-xl/1.0.3/
- CinC Challenge 2021 1.0.3: https://physionet.org/content/challenge-2021/1.0.3/
- Chapman-Shaoxing and Ningbo (ECG Arrhythmia) 1.0.0: https://physionet.org/content/ecg-arrhythmia/1.0.0/

## Harmonization

Labels are harmonized to the official CinC 2021 scored code space. The official scoring set contains 30 scored raw SNOMED codes. These collapse to 26 groups under four documented clinical equivalences, in which each pair is treated as a single group:

- complete left bundle branch block and left bundle branch block (`733534002|164909002`)
- complete right bundle branch block and right bundle branch block (`713427006|59118001`)
- premature atrial contraction and supraventricular premature beats (`284470004|63593006`)
- premature ventricular contractions and ventricular premature beats (`427172004|17338001`)

The mapping commit that defines these equivalences is fixed at https://github.com/physionetchallenges/evaluation-2021/tree/e2a75fc01f729cb74cc4e853e054ce81e28381fc.

Of the 26 groups, 17 have positive support in all three sources (PTB-XL, Chapman-Shaoxing, and Ningbo) simultaneously. One of these 17, `251146004`, is excluded before training. The reason is a mapping inconsistency between the external cohorts: Chapman-Shaoxing collapses both poor R-wave progression and low QRS voltage under `251146004`, while Ningbo maps poor R-wave progression separately under `365413008`. Because the underlying finding differs by cohort, this group is not a coherent target and is removed. The remaining 16 groups are the model targets.

From the 16 trained targets, a headline set of 13 labels is defined by support thresholds that guarantee estimable metrics in every cohort. A label enters the headline set only if it has at least 100 positive examples in the PTB-XL training folds and at least 25 positives and 25 negatives in each of the fold 9 validation set, the fold 10 test set, Chapman-Shaoxing, and Ningbo. The three trained targets that do not meet these thresholds remain full training targets but are reported only under sparse-coverage analysis and are never used to compute primary headline metrics.

| Key | Diagnosis | Train pos | Val pos | Test pos | Chapman pos | Ningbo pos | Headline |
|---|---|---:|---:|---:|---:|---:|---|
| 164890007 | atrial flutter | 59 | 7 | 7 | 445 | 7615 | no |
| 733534002\|164909002 | complete left bundle branch block / left bundle branch block | 428 | 54 | 54 | 205 | 248 | yes |
| 713427006\|59118001 | complete right bundle branch block / right bundle branch block | 433 | 55 | 54 | 454 | 1291 | yes |
| 270492004 | first-degree AV block | 636 | 80 | 79 | 247 | 893 | yes |
| 39732003 | left axis deviation | 4129 | 489 | 528 | 382 | 1163 | yes |
| 164947007 | prolonged PR interval | 272 | 34 | 34 | 12 | 40 | no |
| 111975006 | prolonged QT interval | 95 | 12 | 11 | 57 | 337 | no |
| 698252002 | nonspecific intraventricular conduction disorder | 630 | 78 | 79 | 235 | 536 | yes |
| 426783006 | sinus rhythm | 14452 | 1784 | 1822 | 1826 | 6299 | yes |
| 284470004\|63593006 | premature atrial contraction / supraventricular premature beats | 446 | 55 | 54 | 258 | 1063 | yes |
| 164917005 | abnormal Q wave | 438 | 55 | 55 | 235 | 828 | yes |
| 47665007 | right axis deviation | 281 | 36 | 26 | 215 | 638 | yes |
| 426177001 | sinus bradycardia | 509 | 64 | 64 | 3889 | 12670 | yes |
| 427084000 | sinus tachycardia | 661 | 83 | 82 | 1568 | 5687 | yes |
| 164934002 | T-wave abnormality | 1875 | 235 | 231 | 1876 | 5167 | yes |
| 59931005 | T-wave inversion | 235 | 30 | 29 | 157 | 2720 | yes |

Atrial flutter, prolonged PR interval, and prolonged QT interval are the three trained but non-headline targets. Atrial flutter has only 59 training positives and 7 positives in each internal evaluation fold. Prolonged PR interval has only 12 Chapman-Shaoxing positives. Prolonged QT interval has 95 training, 12 validation, and 11 test positives. These failures are defined entirely by the registered support rule.

## Models and training

Both architectures receive identical inputs: the 12 standard leads in physical millivolt units, resampled to 100 Hz, in 2.5-second windows. External signals recorded at 500 Hz are resampled to 100 Hz by polyphase resampling. Per-lead standardization uses the mean and standard deviation computed on the training folds only. No statistics from validation, test, or external data enter training. There is no external adaptation. Demographics are never used as model inputs.

The first architecture is xResNet1d101, independently reimplemented rather than taken from a released checkpoint. It uses stage blocks `[3, 4, 23, 3]`, a three-convolution stem, kernel size 5, base width 64, a concatenated mean and max pooling head, a 128-unit head, and dropout of 0.25 and 0.5 in the head. Optimization is AdamW with a one-cycle schedule at maximum learning rate 0.01, weight decay 0.01, and batch size 128.

The second architecture is an S4D model. It uses a Conv1d embedding from 12 to 512 channels with kernel size 1, model dimension 512, state dimension 8, four bidirectional residual diagonal S4D blocks, dropout 0.2, post-residual LayerNorm, temporal mean pooling, and a linear head. Optimization is AdamW at a constant learning rate 0.001, weight decay 0.01, and batch size 32. The released full S4 implementation requires a custom CUDA Cauchy extension that is incompatible with this environment. This model is therefore named an S4D portability fallback at the published ECG block scale. It is not a reproduction of the 0.9417 full-S4 result and is not claimed to be one.

Both architectures are trained with unweighted binary cross-entropy over all 16 targets, using seeds 17, 29, and 43, for a maximum of 50 epochs. Each epoch draws one uniform random 2.5-second crop per record. For each seed, the retained checkpoint is the one with the highest fold 9 macro-AUROC over the 13 headline labels, with lower fold 9 negative log-likelihood as the tie-break.

## Inference, selection, and seal

At inference, each record is scored with 10 equidistant 2.5-second windows that span the full 10 seconds, and the per-window probabilities are averaged. The prediction for an architecture is the mean of its three seed predictions.

The seal is procedural and strict. Run identifiers, per-label thresholds, and calibration temperatures are all fixed on the training and fold 9 data before any test or external inference. After the seal, exactly one inference pass is run on fold 10 and one on each external cohort. There is no result-driven iteration after the seal.

## Estimands and metrics

The primary estimand is term-centric macro-AUROC computed over the 13 headline labels, reported per cohort. Two derived contrasts are registered as primary. The shift delta is external macro-AUROC minus internal macro-AUROC, computed separately for each external cohort. The model contrast is S4D macro-AUROC minus xResNet macro-AUROC, computed within each cohort.

Secondary metrics are macro-AUPRC, micro-AUROC, micro-AUPRC, and the full set of per-label AUROCs and AUPRCs. All 16 trained targets are reported with coverage, and the three non-headline targets are explicitly flagged as sparse. An undefined metric, such as an AUROC with no positives or no negatives in a stratum, is never imputed. It is reported as undefined.

Per-label operating thresholds are chosen to maximize F1 on fold 9, taking the smallest threshold in case of ties. The fixed thresholds are used to report macro-F1, micro-F1, sensitivity, and specificity in every evaluation cohort.

## Calibration

Calibration uses one positive scalar temperature per architecture ensemble. The temperature is fitted on fold 9 by minimizing binary negative log-likelihood on the logit of the mean seed probabilities. No recalibration is performed on either external cohort.

The primary calibration metric is equal-width 15-bin macro per-label expected calibration error. Secondary calibration metrics are pooled expected calibration error, macro Brier score, and per-label expected calibration error. Pooled and per-label reliability diagrams are reported before and after temperature scaling.

## Subgroups

The subgroup analysis is descriptive, not causal. Age groups are under 40, 40 to 64, and 65 and older. In PTB-XL, the sentinel age value of 300 denotes 90 and older and is placed in the 65-and-older group. Sex is analyzed as female and male. Records with missing stratifiers are excluded only from the analysis in which the stratifier is used.

Within each cohort and each stratification, subgroup metrics are computed only for the common labels that have at least 10 positives and at least 10 negatives in every subgroup of that stratification, using the counts generated in `subgroup_label_counts.csv`. Registered subgroup metrics are macro-AUROC, macro-AUPRC, macro per-label expected calibration error, and Brier score. Results are reported as observed differences without causal or clinical-fairness interpretation.

## Uncertainty

All interval estimates use 2,000 bootstrap replicates with fixed seed 20260716 and percentile 95% confidence intervals. For PTB-XL, the resampling unit is the patient cluster, so that all records from a patient are drawn together. For the external cohorts, the resampling unit is the record because each source record is described as one patient ECG. Model contrasts within a cohort are computed on paired resamples of the two architectures. Shift differences between cohorts use independent draws from each cohort.

## Integrity and attrition

Waveform validation runs after this preregistration is committed but before any model inference. Each record must have the 12 expected leads, a duration of 10 seconds, finite samples throughout, and interpretable gain and unit metadata. Every attrition reason is reported with counts. Analyses use the paired set of valid records. There is no external pooling, external adaptation, or external recalibration at any point.

## Benchmark context

The peer-reviewed Strodthoff et al. xResNet1d101 result reports test macro-AUROC of 0.925 with compressed test-bootstrap uncertainty of 0.007, and a heterogeneous ensemble result of 0.929 with uncertainty 0.007, on the native 71-label task over the original 21,837-record release at 100 Hz, with folds 1 through 8 for training, fold 9 for validation, and fold 10 for test (https://doi.org/10.1109/JBHI.2020.3022989).

The Mehari and Strodthoff supervised signal-only full bidirectional S4 result reports mean macro-AUROC 0.9417 with run-to-run standard deviation 0.0016 over 10 runs on the same native task (https://doi.org/10.1109/JBHI.2023.3310989).

Neither benchmark is directly comparable to this study. Both are computed on the native 71-label task, whereas this study uses a 13-label headline subset of the harmonized CinC 2021 scored space and the duplicate-cleaned PTB-XL release. The S4D fallback here is a diagonal, extension-free approximation at the published block scale and is not the full-S4 model that produced 0.9417. These numbers frame plausibility, not a target to match.

## Interpretation and deviation policy

Hypothesis 1 is supported for an architecture when the 95% interval for its external-minus-internal shift delta lies below zero; a hospital difference is supported when the interval for the difference between the two hospital-specific deltas excludes zero. Hypothesis 2 is supported in a cohort when the 95% paired interval for S4D minus xResNet macro-AUROC lies above zero. Hypothesis 3 is assessed from the registered before-versus-after calibration estimates and their intervals, separately by cohort. Hypothesis 4 remains descriptive and has no binary pass criterion. No multiplicity-adjusted null-hypothesis tests are planned.

Any departure from this plan, whether in data handling, model configuration, thresholding, calibration, or analysis, is recorded as a deviation with its reason and expected direction of effect. It is reported alongside the preregistered analysis rather than in place of it. Exploratory analyses not named here are labeled exploratory. If an implementation error invalidates a sealed run, the invalid output is retained, the error is documented, and any corrected rerun is labeled a protocol deviation rather than silently substituted.

## Registered limitations

The S4D model is a portability fallback and cannot be read as evidence about full S4 performance. The external cohorts differ from PTB-XL in acquisition, population, and labeling convention, and one otherwise shared label (`251146004`) is excluded precisely because its clinical meaning is not stable across cohorts. Three trained targets have sparse support and yield unstable per-label estimates that are reported but not used for headline conclusions. Subgroup analyses are descriptive and constrained by subgroup support, so some labels and strata will have undefined metrics that are reported as undefined rather than imputed. The single-pass seal removes post-hoc tuning but also means that any error discovered after inference is reported as a deviation rather than silently corrected. No claim about clinical deployment is made or implied by this study.
