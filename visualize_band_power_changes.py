#!/usr/bin/env python3
"""
Create easy-to-compare patient-level visualizations of pre/post EEG band power.

Reads the CSV produced by the EEG_v33 pipeline:
    <POP>_global_absolute_relative_band_power_patient_values.csv

Required columns:
    patient, session, scope, band, power_type, value

Outputs:
    Individual Power Changes/
        <POP>_Global_relative_power_change_heatmap.png
        <POP>_Global_absolute_power_change_dB_heatmap.png
        <POP>_Global_relative_power_pre_post_dumbbells.png
        <POP>_Global_absolute_power_pre_post_dumbbells.png
        ... same for Left / Right scopes ...
        <POP>_patient_band_power_change_values.csv

Recommended use:
    1) Run your main EEG_v33 pipeline.
    2) Point INPUT_CSV below to the generated
       <POP>_global_absolute_relative_band_power_patient_values.csv
    3) Run this script.
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================================================
# USER SETTINGS
# ================================================================
INPUT_CSV = Path(r"C:\path\to\P3_global_absolute_relative_band_power_patient_values.csv")
OUTPUT_DIR = INPUT_CSV.parent / "Individual Power Changes"
SELECTED_SCOPES = ["Global", "Left", "Right"]
BAND_ORDER = [
    "Delta", "Theta", "Alpha", "Alpha 1", "Alpha 2",
    "Beta", "Beta 1", "Beta 2", "Gamma 1", "Gamma 2",
]
PATIENT_SORT_MODE = "alphabetical"  # "alphabetical", "mean_change", "band_change"
SORT_BAND = "Alpha 1"               # Used only if PATIENT_SORT_MODE = "band_change"
ANNOTATE_HEATMAP = True
FIG_DPI = 220

# Relative power is shown as percentage-point change: Post - Pre.
# Absolute power is shown as dB change: 10*log10(Post / Pre).


# ================================================================
# HELPERS
# ================================================================
def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_")


def load_patient_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"patient", "session", "scope", "band", "power_type", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing)}\n"
            f"Found: {list(df.columns)}"
        )
    df = df.copy()
    df["session"] = df["session"].astype(str).str.strip()
    df["scope"] = df["scope"].astype(str).str.strip()
    df["band"] = df["band"].astype(str).str.strip()
    df["power_type"] = df["power_type"].astype(str).str.strip().str.lower()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def build_change_table(df: pd.DataFrame, scope: str, power_type: str) -> pd.DataFrame:
    sub = df[(df["scope"] == scope) & (df["power_type"] == power_type)].copy()
    if sub.empty:
        return pd.DataFrame()

    pivot = sub.pivot_table(
        index=["patient", "band"],
        columns="session",
        values="value",
        aggfunc="first",
    ).reset_index()

    # Allow either Pre/Post or V1/V16-like labels if needed later. For now expect Pre/Post.
    pre_col = "Pre"
    post_col = "Post"
    if pre_col not in pivot.columns or post_col not in pivot.columns:
        raise ValueError(
            f"Scope={scope!r}, power_type={power_type!r}: expected sessions 'Pre' and 'Post'. "
            f"Found session columns: {list(pivot.columns)}"
        )

    pivot = pivot.rename(columns={pre_col: "pre", post_col: "post"})
    pivot["raw_change"] = pivot["post"] - pivot["pre"]

    if power_type == "relative":
        pivot["change_display"] = pivot["raw_change"]        # percentage points
        pivot["change_label"] = "Post - Pre (percentage points)"
        pivot["unit"] = "%"
    elif power_type == "absolute":
        pre = pd.to_numeric(pivot["pre"], errors="coerce").to_numpy(dtype=float)
        post = pd.to_numeric(pivot["post"], errors="coerce").to_numpy(dtype=float)
        out = np.full_like(pre, np.nan, dtype=float)
        valid = np.isfinite(pre) & np.isfinite(post) & (pre > 0) & (post > 0)
        out[valid] = 10.0 * np.log10(post[valid] / pre[valid])
        pivot["change_display"] = out                        # dB change
        pivot["change_label"] = "10*log10(Post / Pre) (dB)"
        pivot["unit"] = "µV²"
    else:
        raise ValueError(f"Unsupported power_type: {power_type!r}")

    return pivot


def get_patient_order(change_table: pd.DataFrame) -> list[str]:
    patients = sorted(change_table["patient"].dropna().unique().tolist())
    if PATIENT_SORT_MODE == "alphabetical":
        return patients

    wide = change_table.pivot(index="patient", columns="band", values="change_display")
    for band in BAND_ORDER:
        if band not in wide.columns:
            wide[band] = np.nan
    wide = wide[BAND_ORDER]

    if PATIENT_SORT_MODE == "mean_change":
        score = wide.mean(axis=1, skipna=True)
        return score.sort_values(ascending=False).index.tolist()

    if PATIENT_SORT_MODE == "band_change":
        if SORT_BAND not in wide.columns:
            return patients
        score = wide[SORT_BAND]
        return score.sort_values(ascending=False, na_position="last").index.tolist()

    return patients


def plot_heatmap(change_table: pd.DataFrame, scope: str, power_type: str, out_path: Path) -> None:
    if change_table.empty:
        return

    patient_order = get_patient_order(change_table)
    wide = change_table.pivot(index="patient", columns="band", values="change_display")
    for band in BAND_ORDER:
        if band not in wide.columns:
            wide[band] = np.nan
    wide = wide[BAND_ORDER]
    wide = wide.reindex(patient_order)

    data = wide.to_numpy(dtype=float)
    if np.all(~np.isfinite(data)):
        return

    finite = data[np.isfinite(data)]
    vmax = np.nanmax(np.abs(finite)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    n_patients, n_bands = data.shape
    fig_w = max(8.5, 1.1 * n_bands + 2)
    fig_h = max(4.0, 0.42 * n_patients + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(n_bands))
    ax.set_xticklabels(BAND_ORDER, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_patients))
    ax.set_yticklabels(wide.index.tolist())

    title_metric = "Relative power change" if power_type == "relative" else "Absolute power change"
    units = "percentage points" if power_type == "relative" else "dB"
    ax.set_title(f"{scope} — {title_metric} (Post − Pre)\nPatients × Bands heatmap")
    ax.set_xlabel("Frequency band")
    ax.set_ylabel("Patient")

    # grid lines
    ax.set_xticks(np.arange(-0.5, n_bands, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_patients, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    if ANNOTATE_HEATMAP:
        for r in range(n_patients):
            for c in range(n_bands):
                val = data[r, c]
                if np.isfinite(val):
                    txt_color = "black" if abs(val) < 0.60 * vmax else "white"
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=7.5, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.94)
    cbar.set_label(units)

    note = (
        "Relative: Post − Pre in percentage points"
        if power_type == "relative"
        else "Absolute: 10*log10(Post / Pre) in dB"
    )
    fig.text(0.01, 0.01, note, fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_dumbbell_panels(change_table: pd.DataFrame, scope: str, power_type: str, out_path: Path) -> None:
    if change_table.empty:
        return

    patient_order = get_patient_order(change_table)
    n_bands = len(BAND_ORDER)
    ncols = 2
    nrows = math.ceil(n_bands / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(10, 2.4 * nrows)), squeeze=False)
    axes_flat = axes.ravel()

    units = "%" if power_type == "relative" else "µV²"
    title_metric = "Relative power" if power_type == "relative" else "Absolute power"

    for i, band in enumerate(BAND_ORDER):
        ax = axes_flat[i]
        sub = change_table[change_table["band"] == band].copy()
        if sub.empty:
            ax.axis("off")
            continue

        sub = sub.set_index("patient").reindex(patient_order).reset_index()
        y = np.arange(len(sub))
        pre = pd.to_numeric(sub["pre"], errors="coerce").to_numpy(dtype=float)
        post = pd.to_numeric(sub["post"], errors="coerce").to_numpy(dtype=float)

        # lines
        for yy, a, b in zip(y, pre, post):
            if np.isfinite(a) and np.isfinite(b):
                ax.plot([a, b], [yy, yy], linewidth=1.2, alpha=0.65)

        ax.scatter(pre, y, marker="o", s=32, label="Pre" if i == 0 else None)
        ax.scatter(post, y, marker="s", s=32, label="Post" if i == 0 else None)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["patient"].tolist(), fontsize=8)
        ax.invert_yaxis()
        ax.set_title(band)
        ax.set_xlabel(units)
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    for j in range(n_bands, len(axes_flat)):
        axes_flat[j].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"{scope} — {title_metric}: patient-level Pre vs Post", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_change_csv(change_tables: list[pd.DataFrame], out_path: Path) -> None:
    if not change_tables:
        return
    df = pd.concat(change_tables, ignore_index=True)
    # Keep a clean export order
    cols_first = [
        "patient", "band", "scope", "power_type", "pre", "post",
        "raw_change", "change_display", "change_label", "unit",
    ]
    other_cols = [c for c in df.columns if c not in cols_first]
    df = df[cols_first + other_cols]
    df.to_csv(out_path, index=False)


# ================================================================
# MAIN
# ================================================================
def main() -> None:
    df = load_patient_table(INPUT_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Derive population code from filename if possible.
    population_code = INPUT_CSV.name.split("_")[0] if "_" in INPUT_CSV.name else "POP"

    exported_tables: list[pd.DataFrame] = []

    for scope in SELECTED_SCOPES:
        for power_type in ("relative", "absolute"):
            table = build_change_table(df, scope=scope, power_type=power_type)
            if table.empty:
                print(f"Skipping {scope} / {power_type}: no rows found")
                continue
            table = table.copy()
            table["scope"] = scope
            table["power_type"] = power_type
            exported_tables.append(table)

            heatmap_name = (
                f"{population_code}_{_safe_slug(scope)}_"
                f"{'relative_power_change_heatmap' if power_type == 'relative' else 'absolute_power_change_dB_heatmap'}.png"
            )
            dumbbell_name = (
                f"{population_code}_{_safe_slug(scope)}_"
                f"{'relative_power_pre_post_dumbbells' if power_type == 'relative' else 'absolute_power_pre_post_dumbbells'}.png"
            )

            plot_heatmap(table, scope, power_type, OUTPUT_DIR / heatmap_name)
            plot_dumbbell_panels(table, scope, power_type, OUTPUT_DIR / dumbbell_name)
            print(f"Saved visuals for {scope} / {power_type}")

    export_csv = OUTPUT_DIR / f"{population_code}_patient_band_power_change_values.csv"
    save_change_csv(exported_tables, export_csv)
    print(f"Saved change table: {export_csv}")
    print(f"Done. Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
