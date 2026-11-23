"""Event-level EEG ratios and ROI band powers from cached spectral topomap data.

All powers are ABSOLUTE Welch integrated band powers in microvolt^2.
The input is the list returned by replicate_eeg_topomaps_v2.export_all_topomap_figures.
No additional PSD estimate is performed.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROI_CHANNELS = {
    'Frontal': ('F3', 'F4', 'F7', 'F8', 'Fz'),
    'Parietal': ('P3', 'P4', 'P7', 'P8', 'Pz'),
    'Posterior': ('P3', 'P4', 'P7', 'P8', 'Pz', 'O1', 'O2', 'Oz'),
}


def _ratio(numerator, denominator):
    if not (np.isfinite(numerator) and np.isfinite(denominator)) or denominator <= 0:
        return float('nan')
    return float(numerator / denominator)


def compute_event_ratios(topomap_results):
    """Compute one row per event, preserving order and duplicate event labels.

    TAR = Theta(Fz) / Alpha(Pz), exact electrodes, no substitutions.
    Global EI = mean EEG Beta / (mean EEG Theta + mean EEG Alpha).
    ROI EI = posterior mean Beta / (frontal mean Theta + parietal mean Alpha).

    ROI mean power is the arithmetic mean of available electrodes in each ROI.
    Missing ROI electrodes are listed in the output. No ROI estimate is made
    if all of its designated electrodes are absent.
    """
    rows = []
    for idx, item in enumerate(topomap_results, start=1):
        channels = list(item['info']['ch_names'])
        abs_power = item['absolute']
        needed = ('Theta', 'Alpha', 'Beta')
        if any(band not in abs_power for band in needed):
            raise ValueError(f"Event {idx}: expected bands {needed}; got {list(abs_power)}")
        if any(len(abs_power[band]) != len(channels) for band in needed):
            raise ValueError(f'Event {idx}: power vectors and EEG channel names do not match')
        lookup = {name.lower(): i for i, name in enumerate(channels)}

        def channel_power(ch, band):
            index = lookup.get(ch.lower())
            return float(abs_power[band][index]) if index is not None else float('nan')

        def roi_power(roi, band):
            values = [channel_power(ch, band) for ch in ROI_CHANNELS[roi]]
            good = [v for v in values if np.isfinite(v)]
            return float(np.mean(good)) if good else float('nan')

        def global_power(band):
            x = np.asarray(abs_power[band], dtype=float)
            x = x[np.isfinite(x)]
            return float(np.mean(x)) if x.size else float('nan')

        fz_theta = channel_power('Fz', 'Theta')
        pz_alpha = channel_power('Pz', 'Alpha')
        frontal_theta = roi_power('Frontal', 'Theta')
        parietal_alpha = roi_power('Parietal', 'Alpha')
        posterior_alpha = roi_power('Posterior', 'Alpha')
        posterior_beta = roi_power('Posterior', 'Beta')
        g_theta = global_power('Theta')
        g_alpha = global_power('Alpha')
        g_beta = global_power('Beta')

        absent = {
            roi: [c for c in members if c.lower() not in lookup]
            for roi, members in ROI_CHANNELS.items()
        }
        for roi, names in absent.items():
            if names:
                warnings.warn(f"Event {idx} ({item['label']}): missing {roi} channels: {names}", stacklevel=2)

        rows.append({
            'Event_index': idx,
            'Event': item['label'],
            'Theta_Fz_uV2': fz_theta,
            'Alpha_Pz_uV2': pz_alpha,
            'TAR_FzTheta_over_PzAlpha': _ratio(fz_theta, pz_alpha),
            'Theta_Frontal_ROI_uV2': frontal_theta,
            'Alpha_Parietal_ROI_uV2': parietal_alpha,
            'Alpha_Posterior_ROI_uV2': posterior_alpha,
            'Beta_Posterior_ROI_uV2': posterior_beta,
            'Theta_Global_uV2': g_theta,
            'Alpha_Global_uV2': g_alpha,
            'Beta_Global_uV2': g_beta,
            'Engagement_Global_Beta_over_ThetaPlusAlpha': _ratio(g_beta, g_theta + g_alpha),
            'Engagement_ROI_PosteriorBeta_over_FrontalThetaPlusParietalAlpha': _ratio(
                posterior_beta, frontal_theta + parietal_alpha
            ),
            'Frontal_ROI_channels_used': ', '.join(c for c in ROI_CHANNELS['Frontal'] if c.lower() in lookup),
            'Parietal_ROI_channels_used': ', '.join(c for c in ROI_CHANNELS['Parietal'] if c.lower() in lookup),
            'Posterior_ROI_channels_used': ', '.join(c for c in ROI_CHANNELS['Posterior'] if c.lower() in lookup),
            'Fz_available': 'fz' in lookup,
            'Pz_available': 'pz' in lookup,
        })
    return pd.DataFrame(rows)


def export_event_ratios(topomap_results, output_dir, show=False):
    """Save a wide CSV and two figures; return the DataFrame."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = compute_event_ratios(topomap_results)
    if df.empty:
        raise ValueError('No events were provided')
    file_path = output_dir / 'EEG_Event_TAR_Engagement_ROI_Power.csv'
    df.to_csv(file_path, index=False, encoding='utf-8-sig')

    labels = [f"{i}. {label}" for i, label in zip(df['Event_index'], df['Event'])]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.9), 5.4))
    ax.plot(x, df['TAR_FzTheta_over_PzAlpha'], 'o-', label='TAR Fz/Pz')
    ax.plot(x, df['Engagement_Global_Beta_over_ThetaPlusAlpha'], 's-', label='Engagement (all EEG channels)')
    ax.plot(x, df['Engagement_ROI_PosteriorBeta_over_FrontalThetaPlusParietalAlpha'], '^-', label='Engagement (selected ROIs)')
    ax.set_xticks(x, labels, rotation=60, ha='right')
    ax.set_ylabel('Ratio (dimensionless)')
    ax.set_title('TAR and Engagement by surgical event')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'EEG_Event_TAR_Engagement.png', dpi=220, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

    roi_cols = [
        ('Theta_Frontal_ROI_uV2', 'Theta — Frontal'),
        ('Alpha_Parietal_ROI_uV2', 'Alpha — Parietal'),
        ('Alpha_Posterior_ROI_uV2', 'Alpha — Posterior'),
        ('Beta_Posterior_ROI_uV2', 'Beta — Posterior'),
    ]
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.9), 5.4))
    for col, label in roi_cols:
        ax.plot(x, df[col], 'o-', label=label)
    ax.set_xticks(x, labels, rotation=60, ha='right')
    ax.set_ylabel('Absolute power (µV²)')
    ax.set_title('Regional absolute EEG band power by surgical event')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'EEG_Event_ROI_Absolute_Power.png', dpi=220, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f'Saved event ratios and ROI powers: {file_path}')
    return df
