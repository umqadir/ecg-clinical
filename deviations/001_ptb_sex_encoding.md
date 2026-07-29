# Deviation 001: corrected PTB-XL numeric sex encoding

Date discovered: 2026-07-16
Stage discovered: after pre-registration and waveform validation; before PTB-XL fold-10 or external model inference

## Error

The metadata-only pre-registration generator interpreted PTB-XL `sex=0` as female and `sex=1` as male. PTB-XL uses the opposite coding: `0` is male and `1` is female. In release 1.0.3, the field contains 11,354 zero-coded records (52.1%) and 10,445 one-coded records (47.9%), reproducing the official dataset description of 52% male and 48% female.

Challenge header strings for Chapman-Shaoxing and Ningbo were already interpreted correctly and are unaffected.

## Impact

The PTB-XL rows labeled `female` and `male` in the sealed `subgroup_label_counts.csv` are exchanged. This does not change the pre-registered common-label eligibility set: eligibility requires the support threshold in both sex groups, so exchanging the group names leaves the intersection invariant.

The error has no effect on waveforms, diagnostic targets, training, thresholds, calibration, headline metrics, age subgroups, or external sex subgroups. If left uncorrected, it would reverse the names attached to the two PTB-XL sex-specific result rows.

## Correction

The sealed pre-registration artifacts remain unchanged. Final analysis maps PTB-XL `0` to male and `1` to female, generates corrected post-registration subgroup counts, and labels this correction wherever subgroup results are presented. This is an implementation correction with a known direction of effect: the two PTB-XL sex-group names are swapped, while their numerical results and eligibility are otherwise unchanged.

Source: PTB-XL dataset card and paper, <https://doi.org/10.1038/s41597-020-0495-6>.
