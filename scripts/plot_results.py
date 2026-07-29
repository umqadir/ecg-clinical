#!/usr/bin/env python3
"""Render publication-quality shift, calibration, label, transfer, and subgroup figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

ARCHITECTURE_NAMES = {"xresnet1d101": "xResNet1d101", "s4d": "S4D"}
COHORT_ORDER = ["ptb_test", "chapman_shaoxing", "ningbo"]
COHORT_NAMES = {
    "ptb_test": "PTB-XL fold 10",
    "chapman_shaoxing": "Chapman-Shaoxing",
    "ningbo": "Ningbo",
}
COHORT_TICK_NAMES = {
    "ptb_test": "PTB-XL\nfold 10",
    "chapman_shaoxing": "Chapman-\nShaoxing",
    "ningbo": "Ningbo",
}
PALETTE = {"xresnet1d101": "#2864A8", "s4d": "#D05A3A"}

INTERNAL_COHORT = "ptb_test"
EXTERNAL_COHORTS = ["chapman_shaoxing", "ningbo"]
COHORT_MARKERS = {"ptb_test": "o", "chapman_shaoxing": "s", "ningbo": "^"}

DIAGNOSIS_NAMES = {
    "1st degree av block": "First-degree AV block",
    "complete left bundle branch block / left bundle branch block": "LBBB",
    "complete right bundle branch block / right bundle branch block": "RBBB",
    "left axis deviation": "Left axis deviation",
    "nonspecific intraventricular conduction disorder": "Nonspecific IVCD",
    "premature atrial contraction / supraventricular premature beats": "PAC / SVPB",
    "qwave abnormal": "Abnormal Q wave",
    "right axis deviation": "Right axis deviation",
    "sinus bradycardia": "Sinus bradycardia",
    "sinus rhythm": "Sinus rhythm",
    "sinus tachycardia": "Sinus tachycardia",
    "t wave abnormal": "T-wave abnormality",
    "t wave inversion": "T-wave inversion",
}

SUBGROUP_ORDER = ["under_40", "40_to_64", "65_plus", "female", "male"]
SUBGROUP_NAMES = {
    "under_40": "age under 40",
    "40_to_64": "age 40-64",
    "65_plus": "age 65+",
    "female": "sex female",
    "male": "sex male",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("results"))
    parser.add_argument("--output-root", type=Path, default=Path("figures"))
    return parser.parse_args()


def save_figure(figure: plt.Figure, output_root: Path, name: str) -> None:
    figure.savefig(output_root / f"{name}.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_root / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def diagnosis_name(raw: str) -> str:
    """Return the clinical presentation name, falling back to the manifest string."""

    return DIAGNOSIS_NAMES.get(raw, raw)


def headline_transfer_table(per_label: pd.DataFrame, architecture: str) -> pd.DataFrame:
    """Pivot headline per-label AUROC to one row per diagnosis with a worst-external drop.

    The drop is the internal fold-10 AUROC minus the weaker of the two external
    AUROCs, so a large positive number means the label degraded badly at the
    hospital where it degraded most.
    """

    selected = per_label[per_label.headline & (per_label.architecture == architecture)]
    table = selected.pivot(index="diagnosis", columns="cohort", values="auroc")
    table = table[COHORT_ORDER]
    table["worst_external"] = table[EXTERNAL_COHORTS].min(axis=1)
    table["drop"] = table[INTERNAL_COHORT] - table["worst_external"]
    return table.sort_values("drop", ascending=False)


def shift_figure(metrics: pd.DataFrame, output_root: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    offsets = {"xresnet1d101": -0.12, "s4d": 0.12}
    selected = metrics.set_index(["cohort", "architecture"])
    extremes = []
    for architecture in ("xresnet1d101", "s4d"):
        point = np.asarray(
            [selected.loc[(cohort, architecture), "macro_auroc"] for cohort in COHORT_ORDER]
        )
        lower = np.asarray(
            [
                selected.loc[(cohort, architecture), "macro_auroc_lower_95"]
                for cohort in COHORT_ORDER
            ]
        )
        upper = np.asarray(
            [
                selected.loc[(cohort, architecture), "macro_auroc_upper_95"]
                for cohort in COHORT_ORDER
            ]
        )
        extremes.extend([lower.min(), upper.max()])
        x = np.arange(len(COHORT_ORDER)) + offsets[architecture]
        axis.errorbar(
            x,
            point,
            yerr=np.vstack((point - lower, upper - point)),
            fmt="o-",
            capsize=4,
            linewidth=2,
            markersize=7,
            color=PALETTE[architecture],
            label=ARCHITECTURE_NAMES[architecture],
        )
    floor, ceiling = min(extremes), max(extremes)
    margin = 0.12 * (ceiling - floor)
    axis.set_ylim(floor - margin, ceiling + margin)
    axis.set_xlim(-0.5, len(COHORT_ORDER) - 0.5)
    axis.set_xticks(
        np.arange(len(COHORT_ORDER)), [COHORT_TICK_NAMES[value] for value in COHORT_ORDER]
    )
    axis.set_ylabel("Headline macro-AUROC")
    axis.set_title("Cross-hospital discrimination shift", pad=40)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.005))
    sns.despine(ax=axis)
    save_figure(figure, output_root, "headline_distribution_shift")


def calibration_error_inset(
    axis: plt.Axes, metrics: pd.DataFrame, cohort: str, architecture: str
) -> None:
    """Draw the bootstrapped macro per-label ECE with its 95% interval.

    The reliability curve shows the shape of the miscalibration; this inset
    carries the registered summary statistic with the uncertainty that the
    calibration hypothesis is assessed against, so no calibration number in the
    figure set is a bare point estimate.
    """

    row = metrics[(metrics.cohort == cohort) & (metrics.architecture == architecture)]
    if row.empty:
        return
    row = row.iloc[0]
    inset = axis.inset_axes([0.52, 0.09, 0.44, 0.26])
    # Colours follow the reliability curves in the same panel: the default
    # cycle's first colour is the uncalibrated curve, the second the calibrated.
    for position, (calibration, colour) in enumerate(
        (("uncalibrated", "C0"), ("calibrated", "C1"))
    ):
        prefix = f"{calibration}_macro_per_label_ece"
        if prefix not in row:
            continue
        point = float(row[prefix])
        lower = float(row.get(f"{prefix}_lower_95", np.nan))
        upper = float(row.get(f"{prefix}_upper_95", np.nan))
        error = np.asarray([[max(point - lower, 0.0)], [max(upper - point, 0.0)]])
        inset.errorbar(
            [point],
            [position],
            xerr=np.nan_to_num(error),
            fmt="o",
            markersize=4,
            capsize=3,
            linewidth=1.4,
            color=colour,
        )
    inset.set_yticks([0, 1], ["Uncal.", "Cal."], fontsize=6)
    inset.set_ylim(-0.6, 1.6)
    inset.tick_params(axis="x", labelsize=6)
    inset.set_xlabel("Macro ECE (95% CI)", fontsize=6, labelpad=1)
    inset.grid(axis="x", alpha=0.2)
    inset.patch.set_alpha(0.85)


def calibration_figure(reliability: pd.DataFrame, metrics: pd.DataFrame, output_root: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(11, 6.8), sharex=True, sharey=True)
    for row, architecture in enumerate(("xresnet1d101", "s4d")):
        for column, cohort in enumerate(COHORT_ORDER):
            axis = axes[row, column]
            selected = reliability[
                (reliability.architecture == architecture)
                & (reliability.cohort == cohort)
                & (reliability.label_key == "pooled")
                & (reliability["count"] > 0)
            ]
            for calibration, style in (("uncalibrated", "--"), ("calibrated", "-")):
                line = selected[selected.calibration == calibration]
                axis.plot(
                    line.mean_probability,
                    line.observed_frequency,
                    linestyle=style,
                    marker="o",
                    markersize=3,
                    label=calibration.capitalize(),
                )
            axis.plot([0, 1], [0, 1], color="0.55", linewidth=1, alpha=0.7)
            axis.set_title(f"{ARCHITECTURE_NAMES[architecture]} · {COHORT_NAMES[cohort]}")
            axis.grid(alpha=0.18)
            calibration_error_inset(axis, metrics, cohort, architecture)
    for axis in axes[-1]:
        axis.set_xlabel("Mean predicted probability")
    for axis in axes[:, 0]:
        axis.set_ylabel("Observed frequency")
    axes[0, 0].legend(frameon=False, loc="upper left")
    figure.suptitle("Pooled reliability before and after fold-9 temperature scaling", y=1.01)
    figure.tight_layout()
    save_figure(figure, output_root, "pooled_reliability")


def label_shift_figure(per_label: pd.DataFrame, output_root: Path) -> None:
    """Draw one heatmap block per architecture, rows ranked by transfer loss."""

    primary = headline_transfer_table(per_label, "xresnet1d101")
    order = [diagnosis_name(value) for value in primary.index]
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 5.8), sharey=True)
    figure.subplots_adjust(left=0.24, right=0.87, top=0.84, bottom=0.24, wspace=0.07)
    colourbar_axis = figure.add_axes([0.895, 0.24, 0.021, 0.60])
    for column, architecture in enumerate(("xresnet1d101", "s4d")):
        table = headline_transfer_table(per_label, architecture)[COHORT_ORDER]
        table.index = [diagnosis_name(value) for value in table.index]
        table = table.loc[order]
        table.columns = [COHORT_NAMES[value] for value in table.columns]
        axis = axes[column]
        sns.heatmap(
            table,
            annot=True,
            fmt=".2f",
            cmap="mako",
            vmin=0.5,
            vmax=1.0,
            linewidths=0.4,
            cbar=column == 1,
            cbar_ax=colourbar_axis if column == 1 else None,
            cbar_kws={"label": "AUROC"} if column == 1 else None,
            annot_kws={"fontsize": 10},
            ax=axis,
        )
        axis.set_title(ARCHITECTURE_NAMES[architecture], pad=8)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_xticklabels(
            axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor", fontsize=10
        )
        axis.tick_params(axis="y", labelsize=10)
    figure.suptitle("Per-label discrimination reveals heterogeneous shift", y=0.955)
    figure.text(
        0.555,
        0.055,
        "Ordered by the xResNet1d101 drop from PTB-XL fold 10 to the worst external cohort.",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    save_figure(figure, output_root, "per_label_auroc_heatmap")


def transfer_figure(per_label: pd.DataFrame, output_root: Path) -> None:
    """Show, per headline diagnosis, how far internal AUROC falls at each external hospital."""

    primary = headline_transfer_table(per_label, "xresnet1d101")
    tables = {
        architecture: headline_transfer_table(per_label, architecture).loc[primary.index]
        for architecture in ("xresnet1d101", "s4d")
    }
    order = list(primary.index)
    offsets = {"xresnet1d101": -0.16, "s4d": 0.16}
    figure, axis = plt.subplots(figsize=(9.4, 7.0))
    for position in range(len(order)):
        if position % 2 == 0:
            axis.axhspan(position - 0.5, position + 0.5, color="0.5", alpha=0.06, zorder=0)
    for architecture, table in tables.items():
        colour = PALETTE[architecture]
        for position, diagnosis in enumerate(order):
            row = table.loc[diagnosis]
            y = position + offsets[architecture]
            values = [row[cohort] for cohort in COHORT_ORDER]
            axis.plot(
                [min(values), max(values)],
                [y, y],
                color=colour,
                linewidth=2.2,
                alpha=0.4,
                solid_capstyle="round",
                zorder=1,
            )
            for cohort in COHORT_ORDER:
                internal = cohort == INTERNAL_COHORT
                axis.plot(
                    [row[cohort]],
                    [y],
                    marker=COHORT_MARKERS[cohort],
                    markersize=8 if internal else 7,
                    markeredgewidth=1.5,
                    markerfacecolor=colour if internal else "white",
                    markeredgecolor=colour,
                    linestyle="none",
                    zorder=3,
                )
    axis.axvline(0.5, color="0.4", linewidth=1.1, linestyle=":", zorder=2)
    axis.text(
        0.505,
        0.02,
        "Chance",
        transform=axis.get_xaxis_transform(),
        fontsize=9,
        color="0.45",
        va="bottom",
        ha="left",
    )
    labels = [f"{diagnosis_name(value)}  ({-primary.loc[value, 'drop']:+.2f})" for value in order]
    axis.set_yticks(np.arange(len(order)), labels)
    axis.set_ylim(len(order) - 0.5, -0.5)
    axis.set_xlim(0.47, 1.01)
    axis.set_xlabel("AUROC")
    axis.grid(axis="x", alpha=0.25)
    axis.grid(axis="y", visible=False)
    axis.set_title("Which diagnoses survive the move to a new hospital", pad=36)
    cohort_handles = [
        Line2D(
            [],
            [],
            marker=COHORT_MARKERS[cohort],
            linestyle="none",
            markersize=8,
            markeredgewidth=1.5,
            markerfacecolor="0.35" if cohort == INTERNAL_COHORT else "white",
            markeredgecolor="0.35",
            label=COHORT_NAMES[cohort] + (" (internal)" if cohort == INTERNAL_COHORT else ""),
        )
        for cohort in COHORT_ORDER
    ]
    architecture_handles = [
        Line2D(
            [],
            [],
            color=PALETTE[architecture],
            linewidth=3,
            label=ARCHITECTURE_NAMES[architecture],
        )
        for architecture in ("xresnet1d101", "s4d")
    ]
    cohort_legend = axis.legend(
        handles=cohort_handles,
        loc="lower left",
        bbox_to_anchor=(-0.005, 1.005),
        ncol=3,
        frameon=False,
        fontsize=9,
        handletextpad=0.4,
        columnspacing=1.4,
    )
    axis.add_artist(cohort_legend)
    axis.legend(
        handles=architecture_handles,
        loc="lower right",
        bbox_to_anchor=(1.005, 1.005),
        ncol=2,
        frameon=False,
        fontsize=9,
        handletextpad=0.6,
    )
    sns.despine(ax=axis, left=True)
    figure.text(
        0.5,
        0.015,
        "Ordered by the xResNet1d101 change from PTB-XL fold 10 to its worst external cohort; "
        "that change is in parentheses.",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(figure, output_root, "per_label_transfer")


def subgroup_figure(subgroups: pd.DataFrame, output_root: Path) -> None:
    """Draw subgroup macro-AUROC in cohort blocks, each row labelled with its label count."""

    subgroups = subgroups.copy()
    subgroups["cohort_rank"] = subgroups.cohort.map(COHORT_ORDER.index)
    subgroups["subgroup_rank"] = subgroups.subgroup.map(SUBGROUP_ORDER.index)
    subgroups = subgroups.sort_values(["cohort_rank", "subgroup_rank"])
    subgroups["row"] = [
        f"{COHORT_NAMES[cohort]} · {SUBGROUP_NAMES[subgroup]} ({count} labels)"
        for cohort, subgroup, count in zip(
            subgroups.cohort, subgroups.subgroup, subgroups.eligible_labels, strict=True
        )
    ]
    order = list(dict.fromkeys(subgroups.row))
    matrix = subgroups.pivot(index="row", columns="architecture", values="macro_auroc")
    matrix = matrix.loc[order, ["xresnet1d101", "s4d"]].rename(columns=ARCHITECTURE_NAMES)
    floor = np.floor(matrix.to_numpy().min() * 100) / 100
    ceiling = np.ceil(matrix.to_numpy().max() * 100) / 100
    figure, axis = plt.subplots(figsize=(8.4, 7.6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="crest",
        vmin=floor,
        vmax=ceiling,
        linewidths=0.4,
        cbar_kws={"label": "Macro-AUROC"},
        annot_kws={"fontsize": 11},
        ax=axis,
    )
    block_size = len(SUBGROUP_ORDER)
    for boundary in range(block_size, len(order), block_size):
        axis.axhline(boundary, color="0.15", linewidth=2.4)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_xticklabels(axis.get_xticklabels(), rotation=0, fontsize=11)
    axis.tick_params(axis="y", labelsize=10)
    axis.set_title("Prespecified age and sex subgroup discrimination", pad=12)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.text(
        0.5,
        0.015,
        "Macro-AUROC is computed over each cohort's own eligible label set (count in each row "
        "label),\nso values are not comparable across cohorts.",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    save_figure(figure, output_root, "subgroup_macro_auroc")


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.82)
    metrics = pd.read_csv(args.result_root / "cohort_metrics.csv")
    reliability = pd.read_csv(args.result_root / "reliability_bins.csv")
    per_label = pd.read_csv(args.result_root / "per_label_metrics.csv")
    subgroups = pd.read_csv(args.result_root / "subgroup_metrics.csv")
    shift_figure(metrics, args.output_root)
    calibration_figure(reliability, metrics, args.output_root)
    label_shift_figure(per_label, args.output_root)
    transfer_figure(per_label, args.output_root)
    subgroup_figure(subgroups, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
