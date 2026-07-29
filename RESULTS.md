# Results

## 1. Summary

- Two architectures trained on PTB-XL only; evaluated once on PTB-XL fold 10 and once on each of two hospitals not seen during training (Chapman-Shaoxing, Ningbo).
- Internal-to-external macro-AUROC loss: approximately 0.045 for both architectures. All four shift intervals lie entirely below zero (section 3, H1).
- Hospital-vs-hospital and model-vs-model contrasts: every interval contains zero (section 3, H1 second half; H2).
- Per-label split (section 4): rate/axis findings, right bundle branch block, first-degree AV block, and abnormal Q wave transfer to within 0.03 AUROC, two higher externally than internally. Nonspecific intraventricular conduction disorder, T-wave abnormality, T-wave inversion, and sinus rhythm lose 0.10 to 0.23 at their worst external cohort; nonspecific IVCD reaches 0.536 on Ningbo (chance = 0.5).
- Calibration (section 3, H3): macro expected calibration error 6 to 8 times larger externally than internally; fold-9 temperature scaling changes it by 0.0004 to 0.0007 against an error near 0.08.
- Sections 7-10 are post-seal exploratory additions:
  - Sinus rhythm recovers under a restricted (annotation-controlled) reanalysis; three control labels given the same treatment do not (section 7).
  - A 22-feature logistic-regression baseline reproduces the geometric/interpretive split: macro-AUROC 0.8562 internally, 0.8119 and 0.8294 externally (section 8).
  - Calibration failure is dominantly a shift of the prior: one additive per-label logit offset removes 73.5 percent of external error on average, versus 21.8 to 26.5 percent for the most flexible multiplicative correction (section 9).
  - Nonspecific IVCD on Ningbo has negative net benefit at threshold probability 0.10: -0.0045 [-0.0049, -0.0041] (section 10).

## 2. Headline discrimination

Primary estimand: term-centric macro-AUROC over the 13 headline labels, per cohort, 95% percentile cluster bootstrap, 2,000 replicates.

| Cohort | Architecture | macro-AUROC [95% CI] | macro-AUPRC | micro-AUROC | micro-AUPRC | contributing labels |
|---|---|---|---|---|---|---|
| PTB-XL fold 10 | xResNet1d101 | 0.9371 [0.9290, 0.9447] | 0.6264 | 0.9821 | 0.9118 | 13 |
| PTB-XL fold 10 | S4D | 0.9331 [0.9250, 0.9410] | 0.6034 | 0.9809 | 0.9037 | 13 |
| Chapman-Shaoxing | xResNet1d101 | 0.8919 [0.8864, 0.8971] | 0.4775 | 0.8851 | 0.3911 | 13 |
| Chapman-Shaoxing | S4D | 0.8912 [0.8854, 0.8960] | 0.4600 | 0.8818 | 0.3769 | 13 |
| Ningbo | xResNet1d101 | 0.8869 [0.8843, 0.8893] | 0.5113 | 0.8790 | 0.3657 | 13 |
| Ningbo | S4D | 0.8887 [0.8860, 0.8912] | 0.4798 | 0.8790 | 0.3544 | 13 |

- All 13 headline labels contribute in every cohort; contributing count never fell below 13 in any of the 12,000 bootstrap replicates (2,000 x 3 cohorts x 2 architectures). No subgroup replicate fell below its eligible set size.
- Figure: `figures/headline_distribution_shift.png`.

## 3. Registered hypotheses

### H1: loss on transfer, and a difference between the hospitals

First half: supported for both architectures, every shift interval lies below zero.

| External cohort | Architecture | delta macro-AUROC [95% CI] |
|---|---|---|
| Chapman-Shaoxing | xResNet1d101 | -0.0452 [-0.0542, -0.0354] |
| Chapman-Shaoxing | S4D | -0.0420 [-0.0515, -0.0324] |
| Ningbo | xResNet1d101 | -0.0503 [-0.0585, -0.0417] |
| Ningbo | S4D | -0.0445 [-0.0528, -0.0357] |

- Second half (registered criterion: interval for the difference between the two hospital-specific deltas excludes zero): not supported. xResNet1d101: -0.0051 [-0.0106, +0.0008]. S4D: -0.0025 [-0.0082, +0.0036]. Ningbo is the numerically harder cohort for both.

### H2: S4D above xResNet1d101 internally and at each hospital

Not supported in any cohort. Registered criterion: paired S4D-minus-xResNet1d101 interval above zero. No interval clears it.

| Cohort | S4D minus xResNet1d101 [95% CI] |
|---|---|
| PTB-XL fold 10 | -0.0040 [-0.0094, +0.0013] |
| Chapman-Shaoxing | -0.0008 [-0.0047, +0.0027] |
| Ningbo | +0.0018 [-0.0001, +0.0037] |

- Ranking flips: xResNet1d101 leads internally and at Chapman-Shaoxing; S4D leads at Ningbo. Ningbo interval closest to excluding zero (lower bound -0.0001).

### H3: external calibration and the effect of temperature scaling

Primary calibration metric: macro average over headline labels of 15-bin per-label expected calibration error.

| Cohort | Architecture | uncalibrated [95% CI] | calibrated [95% CI] | calibrated minus uncalibrated [95% CI] |
|---|---|---|---|---|
| PTB-XL fold 10 | xResNet1d101 | 0.01020 [0.01106, 0.01442] | 0.00994 [0.01104, 0.01414] | -0.00026 [-0.00089, +0.00065] |
| PTB-XL fold 10 | S4D | 0.01271 [0.01377, 0.01723] | 0.01200 [0.01328, 0.01655] | -0.00071 [-0.00200, +0.00089] |
| Chapman-Shaoxing | xResNet1d101 | 0.08025 [0.07906, 0.08202] | 0.07980 [0.07860, 0.08155] | -0.00045 [-0.00067, -0.00026] |
| Chapman-Shaoxing | S4D | 0.08152 [0.08045, 0.08333] | 0.08089 [0.07977, 0.08266] | -0.00063 [-0.00123, -0.00010] |
| Ningbo | xResNet1d101 | 0.08414 [0.08341, 0.08498] | 0.08367 [0.08299, 0.08453] | -0.00047 [-0.00056, -0.00031] |
| Ningbo | S4D | 0.08619 [0.08549, 0.08714] | 0.08546 [0.08475, 0.08631] | -0.00072 [-0.00116, -0.00044] |

- External:internal uncalibrated macro ECE ratio: xResNet1d101 7.87 (Chapman-Shaoxing), 8.25 (Ningbo); S4D 6.41, 6.78.
- Fold-9 temperature scaling: internal contrast interval contains zero for both architectures (no improvement established). External contrast interval excludes zero (real but ~0.0005 against ~0.08 error, about half a percent of the quantity it corrects).
- Fitted temperatures: xResNet1d101 1.0407; S4D 1.1201.
- Uncalibrated pooled reliability curves (`figures/pooled_reliability.png`): internal top-bin observed frequency 0.9690 (xResNet1d101), 0.9535 (S4D). External: top six of fifteen bins (predicted probability above 0.6) observed frequency 0.3635-0.4938 across architectures/hospitals; top bin 0.4322 (Chapman-Shaoxing), 0.4040 (Ningbo) for xResNet1d101.
- Pooled ECE (xResNet1d101): 0.0036 internal to 0.0644 (Chapman-Shaoxing) and 0.0682 (Ningbo). Macro Brier: 0.0265 to 0.0812 and 0.0846. Both secondary (mix labels of differing prevalence).

### H4: subgroup variation

Registered as descriptive, no pass criterion. 65-plus stratum is worst of the three age strata in every cohort, both architectures.

| Cohort | Age stratum | records | eligible labels | xResNet1d101 macro-AUROC [95% CI] | S4D macro-AUROC [95% CI] |
|---|---|---|---|---|---|
| PTB-XL fold 10 | under 40 | 284 | 4 | 0.9465 [0.9206, 0.9712] | 0.9401 [0.9065, 0.9705] |
| PTB-XL fold 10 | 40 to 64 | 866 | 4 | 0.9433 [0.9285, 0.9566] | 0.9404 [0.9268, 0.9525] |
| PTB-XL fold 10 | 65 plus | 1048 | 4 | 0.9219 [0.9022, 0.9370] | 0.9159 [0.8944, 0.9320] |
| Chapman-Shaoxing | under 40 | 1262 | 7 | 0.9303 [0.9159, 0.9432] | 0.9247 [0.9097, 0.9381] |
| Chapman-Shaoxing | 40 to 64 | 4503 | 7 | 0.9244 [0.9197, 0.9287] | 0.9264 [0.9222, 0.9303] |
| Chapman-Shaoxing | 65 plus | 4482 | 7 | 0.9222 [0.9182, 0.9262] | 0.9185 [0.9146, 0.9227] |
| Ningbo | under 40 | 5970 | 14 | 0.8801 [0.8645, 0.8942] | 0.8737 [0.8558, 0.8896] |
| Ningbo | 40 to 64 | 14023 | 14 | 0.8829 [0.8776, 0.8882] | 0.8850 [0.8790, 0.8907] |
| Ningbo | 65 plus | 14857 | 14 | 0.8394 [0.8353, 0.8434] | 0.8395 [0.8355, 0.8435] |

- Ningbo: 65-plus 0.8394 vs. under-40 0.8801, non-overlapping intervals. Chapman-Shaoxing: same ordering, spread 0.0081, two older-stratum intervals nearly touching. PTB-XL fold 10: 65-plus interval below under-40 point estimate, intervals overlap.
- Sex differences small. Largest: Ningbo xResNet1d101, female 0.8796 [0.8752, 0.8840] vs. male 0.8672 [0.8635, 0.8710], gap 0.0124, non-overlapping. PTB-XL fold 10 and Chapman-Shaoxing: sex intervals overlap, both architectures.
- Subgroup values computed over each cohort's own eligible label set; not comparable across cohorts. `figures/subgroup_macro_auroc.png` prints the label count per row. PTB-XL sex rows carry the deviation 001 correction flag (section 12).

## 4. Per-label transfer

Labels ordered by xResNet1d101 change from PTB-XL fold 10 to worst external cohort.

| Diagnosis | xRN PTB-XL | xRN Chapman | xRN Ningbo | worst drop | S4D PTB-XL | S4D Chapman | S4D Ningbo | prev PTB-XL | prev Chapman | prev Ningbo |
|---|---|---|---|---|---|---|---|---|---|---|
| nonspecific IVCD | 0.765 | 0.700 | 0.536 | -0.229 | 0.753 | 0.728 | 0.619 | 0.036 | 0.023 | 0.015 |
| T-wave abnormality | 0.920 | 0.758 | 0.771 | -0.162 | 0.916 | 0.761 | 0.779 | 0.105 | 0.183 | 0.148 |
| sinus rhythm | 0.938 | 0.848 | 0.816 | -0.122 | 0.931 | 0.850 | 0.819 | 0.829 | 0.178 | 0.180 |
| T-wave inversion | 0.959 | 0.853 | 0.865 | -0.105 | 0.955 | 0.851 | 0.876 | 0.013 | 0.015 | 0.078 |
| LBBB | 0.997 | 0.898 | 0.996 | -0.100 | 0.997 | 0.922 | 0.997 | 0.025 | 0.020 | 0.007 |
| PAC / SVPB | 0.905 | 0.856 | 0.871 | -0.049 | 0.885 | 0.829 | 0.832 | 0.025 | 0.025 | 0.030 |
| right axis deviation | 0.994 | 0.971 | 0.967 | -0.028 | 0.991 | 0.969 | 0.963 | 0.012 | 0.021 | 0.018 |
| RBBB | 0.998 | 0.977 | 0.995 | -0.021 | 0.998 | 0.977 | 0.993 | 0.025 | 0.044 | 0.037 |
| abnormal Q wave | 0.833 | 0.825 | 0.828 | -0.009 | 0.836 | 0.810 | 0.814 | 0.025 | 0.023 | 0.024 |
| sinus tachycardia | 0.995 | 0.995 | 0.988 | -0.007 | 0.995 | 0.992 | 0.986 | 0.037 | 0.153 | 0.163 |
| first-degree AV block | 0.980 | 0.973 | 0.973 | -0.007 | 0.979 | 0.964 | 0.957 | 0.036 | 0.024 | 0.026 |
| sinus bradycardia | 0.956 | 0.982 | 0.965 | +0.009 | 0.958 | 0.980 | 0.967 | 0.029 | 0.380 | 0.363 |
| left axis deviation | 0.943 | 0.961 | 0.958 | +0.015 | 0.937 | 0.952 | 0.949 | 0.240 | 0.037 | 0.033 |

- Group A, point estimates within about 0.03 in both external cohorts: sinus tachycardia, sinus bradycardia, first-degree AV block, left axis deviation, right axis deviation, RBBB, abnormal Q wave. Intervals for the sparser labels in this group are wider than 0.03.
- Left axis deviation: +0.018 [+0.006, +0.031] Chapman-Shaoxing, +0.015 [+0.004, +0.028] Ningbo, separated from zero at both.
- Sinus bradycardia: +0.026 [+0.005, +0.053] Chapman-Shaoxing (separated from zero); +0.009 [-0.012, +0.037] Ningbo (not separated from zero).
- Abnormal Q wave: never exceeds 0.836. Internal AUROC (55 positives) 0.833 [0.774, 0.887]. Changes: -0.009 [-0.068, +0.054] Chapman-Shaoxing; -0.005 [-0.061, +0.054] Ningbo.
- PAC/SVPB (outside group A, xResNet1d101): -0.049 [-0.094, +0.005] Chapman-Shaoxing; -0.034 [-0.074, +0.015] Ningbo, neither separated from zero.
- Group B, degraded point estimates at both hospitals, worst-case loss 0.10 or more: nonspecific IVCD, T-wave abnormality, T-wave inversion, sinus rhythm.
- Nonspecific IVCD (xResNet1d101): 0.765 internal, 0.700 Chapman-Shaoxing, 0.536 [0.514, 0.559] Ningbo (interval excludes 0.5). S4D: 0.619 [0.597, 0.642] Ningbo. Ningbo change -0.229 [-0.294, -0.159], the largest per-label loss in the study; Chapman-Shaoxing change -0.066 [-0.138, +0.013], not separated from zero.
- T-wave abnormality: 0.920 to 0.758, 0.771. T-wave inversion: 0.959 to 0.853, 0.865. Sinus rhythm: 0.938 to 0.848, 0.816. All six changes separated from zero.
- LBBB: hospital-specific behavior, not category-specific. xResNet1d101: 0.997 internal, 0.996 Ningbo, 0.898 Chapman-Shaoxing. S4D: 0.997, 0.997, 0.922. Chapman-Shaoxing change -0.100 [-0.125, -0.076], separated from zero; Ningbo change -0.001 [-0.005, +0.003], not separated from zero. Positives: 205 Chapman-Shaoxing, 248 Ningbo.
- Figures: `figures/per_label_transfer.png`, `figures/per_label_auroc_heatmap.png`.
- Group A labels ("geometric") are measured from waveform geometry directly: heart rate, frontal-plane axis, QRS width/morphology, PR interval. Group B labels ("interpretive"): nonspecific IVCD is defined by a QRS duration cutoff applied when no specific block pattern is present; T-wave abnormality and T-wave inversion are defined by reader-annotated repolarization deviation.

### Per-label intervals: a post-seal exploratory addition, 2026-07-25

- Per-label AUROC was a registered secondary metric, emitted as a point estimate only (macro quantities alone were bootstrapped). Intervals added 2026-07-25, after the seal was broken; exploratory, not folded into the sealed analysis.
- `scripts/compute_per_label_intervals.py` rebuilds the registered cluster resamples (same seeds, cluster units, replicate count, batch size); aborts unless macro-averaging its per-label replicates reproduces the registered macro-AUROC distribution in `results/bootstrap_distributions.npz` bit for bit. Check passes for all three cohorts, both architectures. Per-label shift deltas formed replicate-wise from independent cohort draws.
- Intervals marginal, uncorrected, matching the registered plan. 13 labels x 2 hospitals = 26 changes per architecture; separated from zero: 16/26 xResNet1d101, 18/26 S4D.

| Diagnosis | change to Chapman-Shaoxing | change to Ningbo |
|---|---|---|
| nonspecific IVCD | -0.066 [-0.138, +0.013] | -0.229 [-0.294, -0.159] |
| T-wave abnormality | -0.162 [-0.183, -0.141] | -0.148 [-0.167, -0.127] |
| sinus rhythm | -0.089 [-0.105, -0.073] | -0.122 [-0.136, -0.106] |
| T-wave inversion | -0.105 [-0.139, -0.072] | -0.094 [-0.112, -0.074] |
| LBBB | -0.100 [-0.125, -0.076] | -0.001 [-0.005, +0.003] |
| PAC / SVPB | -0.049 [-0.094, +0.005] | -0.034 [-0.074, +0.015] |
| right axis deviation | -0.023 [-0.033, -0.013] | -0.028 [-0.034, -0.019] |
| RBBB | -0.021 [-0.030, -0.014] | -0.003 [-0.006, -0.001] |
| abnormal Q wave | -0.009 [-0.068, +0.054] | -0.005 [-0.061, +0.054] |
| sinus tachycardia | -0.001 [-0.005, +0.005] | -0.007 [-0.011, -0.002] |
| first-degree AV block | -0.007 [-0.018, +0.003] | -0.007 [-0.014, +0.002] |
| sinus bradycardia | +0.026 [+0.005, +0.053] | +0.009 [-0.012, +0.037] |
| left axis deviation | +0.018 [+0.006, +0.031] | +0.015 [+0.004, +0.028] |

- xResNet1d101, ordered as above. S4D rows, per-label AUROC intervals in every cohort, and the three sparse non-headline targets: `results/exploratory_per_label_shift_deltas.csv`, `results/exploratory_per_label_intervals.csv`.
- Of the eight interpretive-label changes, 7/8 separated from zero (exception: nonspecific IVCD at Chapman-Shaoxing). Geometric labels stay above 0.95 externally except abnormal Q wave.
- Figures show point estimates only (78 dumbbell endpoints); intervals live in the tables and the two CSV files above.
- Prevalence: sinus rhythm positive on 82.9% of PTB-XL fold 10 records, 17.8% Chapman-Shaoxing, 18.0% Ningbo. Sinus bradycardia: 2.9% internal, 38.0% Chapman-Shaoxing, 36.3% Ningbo.
- Labeling convention not manipulated in this study; its effect is not separated from differences in acquisition, patient mix, and disease severity.

## 5. Operating points and the difference between ranking and probability

Per-label thresholds fixed on fold 9 as the smallest value attaining maximum fold-9 F1, transported unchanged.

| Cohort | Architecture | macro-F1 | micro-F1 | macro sensitivity | macro specificity |
|---|---|---|---|---|---|
| PTB-XL fold 10 | xResNet1d101 | 0.6088 | 0.8249 | 0.6391 | 0.9528 |
| PTB-XL fold 10 | S4D | 0.5882 | 0.7903 | 0.6503 | 0.9420 |
| Chapman-Shaoxing | xResNet1d101 | 0.4206 | 0.4478 | 0.4752 | 0.9345 |
| Chapman-Shaoxing | S4D | 0.4203 | 0.4425 | 0.4964 | 0.9290 |
| Ningbo | xResNet1d101 | 0.4143 | 0.4262 | 0.5050 | 0.9315 |
| Ningbo | S4D | 0.4223 | 0.4238 | 0.5181 | 0.9249 |

- Macro-F1: 0.6088 to ~0.42. Micro-F1: 0.8249 to 0.4478, 0.4262. Micro-AUPRC: 0.9118 to 0.3911, 0.3657 (against a macro-AUROC loss of about 0.045).
- AUROC is prevalence-invariant; AUPRC and fixed-threshold metrics are not. Thresholded metrics here use the fold-9-fixed operating policy carried to each hospital unchanged, not a hospital-tuned operating point.

## 6. Intervals excluding their own point estimate

- 27 intervals in the analysis do not contain their point estimate; all are expected-calibration-error quantities carrying the PTB-XL fold-10 term.
- 12 cohort-level (in the tables above): 5 PTB-XL fold-10 calibration estimates, interval above the point; 7 external-minus-internal calibration shift deltas, interval below the point.
- 15 PTB-XL fold-10 subgroup rows in `results/subgroup_metrics.csv`: 7 uncalibrated + 8 calibrated macro ECE, all above their point.
- No subgroup macro-AUROC interval affected; no external-cohort subgroup row affected.
- Cause: a bias in expected calibration error under resampling.
- Distinct-cluster fraction of a bootstrap resample: 0.6323 (PTB-XL fold 10), 0.6320 (Chapman-Shaoxing), 0.6322 (Ningbo); asymptotic value 1 - 1/e = 0.63212.
- Binned ECE is positively biased at small effective sample size (sparse bins contribute sampling noise that cannot cancel inside an absolute value).
- Subsampling sweep (xResNet1d101, PTB-XL fold 10): mean macro ECE 0.01020 at full sample, 0.01183 at 63.2%, 0.01824 at 20%.
- PTB-XL fold 10: error near 0.01; bias ~0.0016; 1,999/2,000 replicates land above the point estimate, both architectures.
- External cohorts: error 6-8x larger, more clusters (10,247 and 34,905 record clusters vs. 1,904 patient clusters); replicate-mean shift <0.0004; fraction of replicates above point 0.562-0.697.
- The 15 subgroup rows are PTB-XL fold-10 strata of 866-1,132 records, smaller than the 2,198-record fold.

| Cohort | Architecture | point | bootstrap mean | fraction of replicates above point | subsample mean at 0.632 |
|---|---|---|---|---|---|
| PTB-XL fold 10 | xResNet1d101 | 0.01020 | 0.01269 | 0.9995 | 0.01183 |
| PTB-XL fold 10 | S4D | 0.01271 | 0.01545 | 0.9995 | 0.01457 |
| Chapman-Shaoxing | xResNet1d101 | 0.08025 | 0.08058 | 0.668 | 0.08048 |
| Chapman-Shaoxing | S4D | 0.08152 | 0.08189 | 0.697 | 0.08179 |
| Ningbo | xResNet1d101 | 0.08414 | 0.08420 | 0.562 | 0.08420 |
| Ningbo | S4D | 0.08619 | 0.08632 | 0.628 | 0.08627 |

- Diagnostic exploratory, changes no registered quantity. Regenerated bootstrap reproduces the registered percentiles to a difference of 0.00e+00 for all six cohort/architecture pairs.
- No macro-AUROC quantity subject to this bias. All six paired calibrated-minus-uncalibrated contrasts contain their own point estimates.

## 7. Label commensurability: a post-seal exploratory addition, 2026-07-26

Tests whether the 13 headline labels denote the same finding in the three cohorts, using label matrices alone (no model output). Fails for one label.

- Sinus-rhythm co-annotation with sinus bradycardia/tachycardia: PTB-XL fold 10, 36/64 (56.25%) and 24/82 (29.27%); Chapman-Shaoxing, 0/3,889 and 0/1,568; Ningbo, 6/12,670 (0.05%) and 0/5,687.
- Mean headline labels per record: 1.44 (PTB-XL fold 10), 1.13 (Chapman-Shaoxing), 1.12 (Ningbo). Records with no headline label: 6.0%, 15.5%, 17.0%.
- `scripts/exploratory_label_commensurability.py` rescores sealed predictions under two redefined targets: `union` (disjunction of the label and its competing labels, all cohorts including PTB-XL) and `restricted` (records with no competing label only, all cohorts including PTB-XL). Internal reference recomputed under each definition. Competing-label sets fixed on the clinical relation between the SNOMED groups.

xResNet1d101 shift deltas under the three estimands:

| Diagnosis | External cohort | as registered | union | restricted |
|---|---|---|---|---|
| sinus rhythm | Chapman-Shaoxing | -0.089 [-0.105, -0.073] | +0.053 [+0.035, +0.072] | +0.062 [+0.045, +0.079] |
| sinus rhythm | Ningbo | -0.122 [-0.136, -0.106] | -0.052 [-0.070, -0.034] | -0.007 [-0.023, +0.010] |
| T-wave abnormality | Chapman-Shaoxing | -0.162 [-0.183, -0.141] | -0.162 [-0.183, -0.140] | -0.156 [-0.177, -0.134] |
| T-wave abnormality | Ningbo | -0.148 [-0.167, -0.127] | -0.162 [-0.181, -0.141] | -0.118 [-0.138, -0.097] |
| T-wave inversion | Chapman-Shaoxing | -0.105 [-0.139, -0.072] | +0.076 [+0.044, +0.109] | -0.057 [-0.095, -0.020] |
| T-wave inversion | Ningbo | -0.094 [-0.112, -0.074] | +0.070 [+0.039, +0.100] | -0.088 [-0.108, -0.066] |
| nonspecific IVCD | Chapman-Shaoxing | -0.066 [-0.138, +0.013] | -0.036 [-0.101, +0.026] | -0.066 [-0.140, +0.010] |
| nonspecific IVCD | Ningbo | -0.229 [-0.294, -0.159] | -0.126 [-0.190, -0.069] | -0.228 [-0.294, -0.159] |

- Sinus rhythm recovers under `restricted`: Chapman-Shaoxing change +0.062 [+0.045, +0.079] (sign reversed); Ningbo change -0.007 [-0.023, +0.010] (not separated from zero). Restricted external AUROC: 0.9981 [0.9972, 0.9990] Chapman-Shaoxing, 0.9298 [0.9259, 0.9337] Ningbo; restricted internal reference 0.9365 [0.9194, 0.9527]. Restriction drops 5,457/10,247 Chapman-Shaoxing records, 18,357/34,905 Ningbo records, 146/2,198 PTB-XL fold-10 records (sinus bradycardia/tachycardia carriers in every case). S4D: +0.065 [+0.048, +0.082] Chapman-Shaoxing, -0.003 [-0.019, +0.014] Ningbo.
- Controls (do not recover): T-wave abnormality scored against T-wave inversion, T-wave inversion against T-wave abnormality, nonspecific IVCD against the two bundle branch blocks. Under `restricted`, all four T-wave changes (xResNet1d101 and S4D) keep sign and stay separated from zero; both nonspecific IVCD changes are within 0.001 of their registered values, and the Chapman-Shaoxing change remains not separated from zero, as under the registered estimand.
- `union` recovers T-wave inversion by absorbing it into a label with 8.8x the internal positives, dropping the internal reference from 0.959 to 0.727.
- Dropping sinus rhythm from the 13-label macro (xResNet1d101 shift delta): -0.0452 to -0.0415 (Chapman-Shaoxing); -0.0503 to -0.0443 (Ningbo). Whole macro under `union`: -0.0180, -0.0254; internal reference under `union`: 0.9371 to 0.9018. Registered estimand remains the primary result; these figures do not replace it.
- Stage verifies both seals, changes no registered quantity, rebuilds the registered cluster resamples, reproducing the registered macro-AUROC distribution bit for bit in all six cohort/architecture pairs. Intervals marginal and uncorrected.

## 8. A handcrafted-feature baseline: a post-seal exploratory addition, 2026-07-26

Tests whether the section 4 transfer split reflects the findings rather than the networks, using a logistic-regression baseline over explicit geometric measurements.

- `scripts/exploratory_feature_baseline.py` extracts 22 measurements per record, one code path, no per-cohort tuning: heart rate + 2 RR dispersion statistics, QRS duration, frontal axis (sine, cosine), PR interval (+ missingness indicator), QT interval (+ missingness indicator), 8 named R/S amplitudes, 2 T amplitudes, R-peak detection flag.
- One L2-penalized logistic regression per target, fitted on PTB-XL folds 1-8, penalty selected on fold 9, standardization constants from the fitting split alone. Fold 10 and both external cohorts scored once.

| Cohort | feature baseline macro-AUROC [95% CI] | xResNet1d101 | S4D |
|---|---|---|---|
| PTB-XL fold 10 | 0.8562 [0.8450, 0.8673] | 0.9371 | 0.9331 |
| Chapman-Shaoxing | 0.8119 [0.8056, 0.8182] | 0.8919 | 0.8912 |
| Ningbo | 0.8294 [0.8263, 0.8324] | 0.8869 | 0.8887 |

- xResNet1d101 leads the feature baseline by 0.081 (internal), 0.080 (Chapman-Shaoxing), 0.057 (Ningbo). S4D leads by 0.077, 0.079, 0.059.
- Feature-baseline shift delta: -0.0442 [-0.0570, -0.0311] Chapman-Shaoxing; -0.0267 [-0.0381, -0.0150] Ningbo (versus -0.0452, -0.0503 for xResNet1d101).
- Spearman correlation, feature-baseline vs. xResNet1d101 per-label shift deltas (13 headline labels): 0.593, p = 0.033 (Chapman-Shaoxing); 0.516, p = 0.071 (Ningbo).
- By finding class, Ningbo loss: feature baseline 0.008 (7 geometric labels) / 0.089 (4 interpretive labels); xResNet1d101 0.003 / 0.148. Chapman-Shaoxing loss: feature baseline 0.046 / 0.075; xResNet1d101 0.015 / 0.106. Two unclassified labels gain in the feature baseline at both hospitals.
- LBBB (feature baseline): -0.258 [-0.304, -0.216] Chapman-Shaoxing; -0.014 [-0.030, +0.002] Ningbo, the same asymmetry as xResNet1d101's -0.100/-0.001; baseline's only bundle-branch-relevant inputs are QRS duration and lead amplitudes.
- Sinus rhythm falls further in the baseline than either deep model: -0.230 Chapman-Shaoxing, -0.248 Ningbo.
- Detection failure rates: P-wave 52.0% (PTB-XL fold 10), 55.4% (Chapman-Shaoxing), 55.2% (Ningbo); R-peak 0%; T-wave end 3.0-5.1%.
- Single-feature AUROC: heart rate alone / sinus bradycardia 0.9875 (Chapman-Shaoxing), 0.9757 (Ningbo); QRS duration alone / RBBB 0.9499, 0.9555; frontal axis alone / left axis deviation 0.9445, 0.9368; heart rate alone / sinus tachycardia excluding competing rhythm negatives, 0.9824 (Ningbo).
- Intervals use the registered bootstrap design (same seeds, cluster units, replicate count, batch size) but are not the registered resamples bit for bit, since a different model's scores are resampled on the same records.

## 9. Decomposing the calibration failure: a post-seal exploratory addition, 2026-07-26

Tests which part of the external miscalibration a temperature cannot reach. `scripts/exploratory_calibration_ladder.py` applies nine transforms to the logit of the uncalibrated ensemble probability, scored with the registered estimator (15 equal-width bins per label, macro-averaged over the 13 headline labels). Aborts unless rungs 1 and 2 reproduce the registered uncalibrated/calibrated values; all 12 reproduce exactly.

| Rung | Definition |
|---|---|
| 1 | no transform |
| 2 | sealed fold-9 scalar temperature |
| 3 | one temperature per cohort and architecture, fitted on the cohort's own labels, 13 headline labels pooled |
| 4 | one temperature per label, fitted on that label's own cohort labels |
| 5 | one additive logit offset per label, slope fixed at one, fitted |
| 6 | slope and offset per label, fitted |
| 7 | offset = target log-odds minus source log-odds; needs target prevalence only, no target scores or fitting |
| 8 | same offset, prevalence estimated by Saerens-Latinne-Decaestecker expectation-maximization on the target cohort's unlabeled scores |
| 9 | same offset, prevalence estimated by black-box shift estimation from the fold-9 confusion matrix at the registered threshold and the target cohort's flagged rate |

Source prevalence for rungs 7-9: from the 17,418 records of PTB-XL folds 1-8, checked against `data/derived/preregistration/harmonized_label_counts.csv`.

Macro per-label expected calibration error by rung:

| Rung | fold 10 xRN | fold 10 S4D | Chapman xRN | Chapman S4D | Ningbo xRN | Ningbo S4D |
|---|---|---|---|---|---|---|
| 1 uncalibrated | 0.01020 | 0.01271 | 0.08025 | 0.08152 | 0.08414 | 0.08619 |
| 2 registered temperature | 0.00994 | 0.01200 | 0.07980 | 0.08089 | 0.08367 | 0.08546 |
| 3 oracle global temperature | 0.00995 | 0.01227 | 0.09859 | 0.10410 | 0.10184 | 0.10688 |
| 4 oracle per-label temperature | 0.00886 | 0.00971 | 0.06278 | 0.06215 | 0.06317 | 0.06336 |
| 5 oracle per-label offset | 0.00937 | 0.01109 | 0.02348 | 0.02153 | 0.02155 | 0.02133 |
| 6 oracle per-label affine | 0.00806 | 0.00836 | 0.00737 | 0.00767 | 0.00720 | 0.00714 |
| 7 prevalence-only intercept | 0.00990 | 0.01289 | 0.04370 | 0.04725 | 0.04251 | 0.04573 |
| 8 SLD expectation-maximization | 0.01577 | 0.02546 | 0.07109 | 0.07419 | 0.07570 | 0.07384 |
| 9 black-box shift estimation | 0.01438 | 0.01595 | 0.07040 | 0.07449 | 0.07340 | 0.07902 |

- Rungs 3-6 fit parameters on the evaluation cohort's own labels: oracle upper bounds, not deployable methods.
- Rung 5 (additive offset) removes 70.7-75.3% of external error across the four external cohort/architecture pairs, 73.5% on average. Rung 4 (per-label temperature) removes 21.8-26.5%. Rung 3 (per-cohort temperature) makes error 21.0-27.7% worse. Rung 6 (per-label affine) removes 90.6-91.7%; the offset alone accounts for 78-82% of what fitting both parameters achieves.
- Mean absolute prevalence shift, PTB-XL training folds vs. cohort: 0.1116 (Chapman-Shaoxing), 0.1143 (Ningbo), 0.0011 (fold 10). A constant predictor emitting the training prevalence has macro per-label ECE equal to these values by construction.
- Rungs 7-9 deployable, not registered. Rung 7 (needs target prevalence, no target scores): removes 42.0-49.5%. Rungs 8/9 (prevalence estimated from unlabeled target scores): remove 8.3-14.3%. Rungs 8/9 prevalence-estimate error over the 13 headline labels: mean 0.065-0.073, max 0.318-0.429.
- Per-label Platt scaling on k labeled evaluation-cohort records (xResNet1d101, Chapman-Shaoxing, 50 draws): mean macro per-label ECE 0.0202 (k=100), 0.0141 (k=250), 0.0115 (k=500), 0.0101 (k=1,000); versus 0.0803 uncalibrated and 0.0102 for the same model on fold 10. Ningbo and S4D within 0.0004 of these values at k=1,000. Degenerate label fits (no positive or negative in sample, kept at identity transform): 5.2-9.1% at k=100; 2/650 at k=500; 0 at k=1,000.
- Equal-mass (15-bin) vs. equal-width rescoring of rungs 1-2: external macro per-label error moves by at most 0.00068; internal by at most 0.00175.
- Stage verifies both seals, rebuilds registered resamples; macro-averaged per-label AUROC replicates and rung 1/rung 2 macro error replicates both reproduce the registered distributions bit for bit. Transform parameters held at full-cohort fitted values inside every replicate, never refitted.

## 10. Decision value: a post-seal exploratory addition, 2026-07-26

`scripts/exploratory_clinical_utility.py` computes net benefit = TP/n - (FP/n) x t/(1-t), 50-point threshold grid 0.01 to 0.50, against the better of treat-all/treat-none. Primary curves use temperature-calibrated probabilities; uncalibrated curve alongside in `results/exploratory_decision_curves.csv`.

- xResNet1d101 clears both defaults at all 50 thresholds, all three cohorts: first-degree AV block, LBBB, RBBB, sinus tachycardia. Abnormal Q wave clears them at all 50 thresholds in both external cohorts, 16/50 internally.
- Largest external gains, net benefit over better default at best threshold: sinus tachycardia +0.143 / +0.140; T-wave abnormality +0.063 / +0.059.
- Nonspecific IVCD on Ningbo clears both defaults at 0/50 thresholds (xResNet1d101), 2/50 (S4D). At t = 0.10 (treat-all -0.094, treat-none 0): net benefit -0.004479 [-0.004874, -0.004078] xResNet1d101, -0.004310 [-0.004772, -0.003807] S4D, both intervals entirely below zero. These are the only two of 78 label/cohort/architecture rows whose net-benefit interval excludes zero from below at that threshold.
- Sealed fold-9 F1 thresholds, median over 13 headline labels: alerts per true positive 1.48 (PTB-XL fold 10), 2.36 (Chapman-Shaoxing), 2.49 (Ningbo); median sensitivity 0.680, 0.415, 0.467.
- Exploratory sensitivity-0.90 threshold set (fold-9-derived): median external sensitivity 0.767 / 0.855 at 3.77 / 3.65 alerts per true positive. Both sets attainable on fold 9 for all 13 labels.
- Nonspecific IVCD on Ningbo: sealed threshold finds 1.7% of cases at PPV 0.022, 44.6 alerts per true positive; sensitivity-0.90 threshold finds 47.8% at 60.3 alerts per true positive.
- Stage verifies both seals, changes no registered quantity, rebuilds registered resamples, reproducing the registered macro-AUROC distribution bit for bit in all six cohort/architecture pairs. Every count inside a replicate is weighted against the same cluster draw, so the treat-all comparator moves with resampled prevalence. Intervals marginal and uncorrected.

## 11. Published-benchmark context

- Leinonen et al. 2024, Table 4 (single-source): ResNet trained on PTB and PTB-XL, macro-AUC mean error 0.0529, SD 0.0263, RMSE 0.0591 (mean error = 5-fold CV estimate minus held-out test AUC, averaged over 4 held-out sources).
- This study: internal-to-external drops 0.042-0.050 across two architectures and two hospitals; 0.045 and 0.050 for xResNet1d101.
- Figure 3 readings, not printed in the paper's tables: ResNet trained and 5-fold cross-validated within Chapman-Shaoxing and Ningbo, macro-AUC about 0.97; ResNet trained on PTB and PTB-XL, tested on Chapman-Shaoxing and Ningbo, macro-AUC about 0.91. Accuracy about 0.005.
- Cross-check: 5-fold bar minus the mean of that panel's four test-source bars gives 0.0438 (Chapman-Shaoxing/Ningbo training source) against printed Table 4 value 0.0434, and 0.0531 (PTB/PTB-XL training source) against printed 0.0529.
- The 0.97 in-distribution value is not a comparator for this study's 0.8869/0.8919 (section 2): different label set.
- `METHODOLOGY.md` section 15 records the two PTB-XL architecture-family benchmarks, states which table column each quoted number comes from, and names the published columns that demonstrate an inflated estimate rather than report a benchmark value.

## 12. Limitations

- Study compares two prospectively specified training recipes, not two architectures each at its own optimum.
- xResNet1d101 selected checkpoints: epochs 41, 45, 49. Fold-9 macro-AUROC at epoch 50 within 0.0022 of best, every seed.
- S4D peaked at epochs 13, 10, 11; decayed by 0.0289, 0.0316, 0.0310 fold-9 macro-AUROC by epoch 50. Training loss: 0.062-0.070 at selected epochs to 0.0150-0.0159 at epoch 50, under the registered constant-learning-rate recipe (no schedule, dropout 0.2, weight decay 0.01).
- Selection on fold 9 for both architectures, the registered rule.
- S4D is an extension-free diagonal portability fallback at the published ECG block scale (full S4 requires a CUDA Cauchy kernel unsupported on this hardware). Not the full S4 model that produced 0.9417; no number here reproduces or compares to that result.
- Headline 13-label set chosen by label support in fold 10 and both external cohorts. Three sparse trained targets reported separately, never in a headline metric.
- Micro-AUROC and pooled ECE mix labels of differing prevalence; secondary throughout. Primary claims rest on macro quantities over a fixed label set.
- Subgroup macro-AUROC eligible-label counts, age stratification: 4 (PTB-XL fold 10), 7 (Chapman-Shaoxing), 14 (Ningbo). Comparable within a cohort only; no cross-cohort subgroup comparison made.
- Deviation 001 (PTB-XL sex subgroups): registration generator read `sex=0` as female (dataset codes 0 as male). Found after registration, before fold-10/external inference. Analysis maps the encoding correctly; every PTB-XL sex row carries a correction flag. Only the row names for two PTB-XL sex rows were affected.
- Seal permits a single inference pass per cohort. Post-inference errors reported as deviations; invalid output retained, never silently corrected. Integrity guards prevent out-of-order/wrong-input execution; not proof against tampering by someone with repository write access.
- Sections 7-10 written after the evaluation seal was broken, with targets, estimands, and transforms chosen after the registered outcome was known. None strengthens a registered claim.
- Section 9 oracle rungs fit on the evaluation cohort's own labels: upper bounds, not methods. Section 7 reconciled estimands change the estimand; reported beside the registered one, never in place of it.
- Intervals in sections 7-10 are marginal and uncorrected across a large simultaneous family.
- No claim about clinical deployment is made or implied. Every number here describes a research evaluation on retrospective public data.

## 13. Reproducing these results

- All tables and figures regenerated by `scripts/analyze_results.py` then `scripts/plot_results.py`, against the sealed predictions in `artifacts/predictions/protected/`. `analyze_results.py` verifies both seals before any work; `plot_results.py` calls neither guard and reads the metric tables `analyze_results.py` wrote.
- Outputs: `results/cohort_metrics.csv`, `results/per_label_metrics.csv`, `results/subgroup_metrics.csv` (metric tables); `results/uncertainty.json` (intervals); `results/bootstrap_distributions.npz` (replicate distributions); `results/reliability_bins.csv` (reliability curves); `results/ece_bootstrap_bias_diagnostic.json` (section 6 diagnostic).
- Seven stages are exploratory additions outside the frozen analysis. None recomputes anything the registered analysis reports; each writes only to files carrying the `exploratory_` prefix, except the first.

- `scripts/diagnose_ece_bootstrap_bias.py` writes the section-6 diagnostic to `results/ece_bootstrap_bias_diagnostic.json`.
- `scripts/compute_per_label_intervals.py` writes the section-4 per-label intervals to `results/exploratory_per_label_intervals.csv`, `results/exploratory_per_label_shift_deltas.csv`, and a summary with the resample cross-check to `results/exploratory_per_label_intervals.json`.
- `scripts/exploratory_label_commensurability.py` writes section 7 to `results/exploratory_label_structure.csv`, `results/exploratory_label_cooccurrence.csv`, `results/exploratory_reconciled_auroc.csv`, `results/exploratory_reconciled_shift_deltas.csv`, `results/exploratory_macro_variants.csv`, and a summary to `results/exploratory_label_commensurability.json`.
- `scripts/exploratory_feature_baseline.py` writes section 8 to `results/exploratory_feature_baseline.csv` and `results/exploratory_feature_baseline.json`.
- `scripts/exploratory_calibration_ladder.py` writes section 9 to `results/exploratory_calibration_ladder.csv` and `results/exploratory_calibration_ladder.json`.
- `scripts/exploratory_clinical_utility.py` writes section 10 to `results/exploratory_decision_curves.csv`, `results/exploratory_operating_points.csv`, and `results/exploratory_clinical_utility.json`.
- `scripts/exploratory_transfer_mechanism.py` attributes each per-label shift delta to the positive or negative class and tests three candidate explanations of the loss. Results not discussed in this document; outputs not committed. `METHODOLOGY.md` section 14 item 11 states what it computes and which files it writes.

- 5 of 7 stages require both the registration seal and the evaluation seal. The feature baseline requires the registration seal only (reads no sealed prediction file). The section-6 diagnostic calls neither guard; it aborts unless its regenerated bootstrap reproduces the percentiles in `results/uncertainty.json` within a fixed tolerance.
