"""Generate 6 EEG event x band topomap summary figures.

Outputs exactly these files in the chosen output directory:
- ABP_Global_Scale.png
- ABP_Event_Level_Scale.png
- ABP_Frequency_Band_Level_Scale.png
- RBP_Global_Scale.png
- RBP_Event_Level_Scale.png
- RBP_Frequency_Band_Level_Scale.png

ABP = absolute band power (µV²)
RBP = relative band power (%)

Compatible with event containers that are either:
- a list of dicts with keys including {'raw', 'description' or 'event_label'}
- a list of tuples/lists like (label, raw)
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import mne
from scipy.signal import welch

BANDS = {
    'Delta': (1.0, 4.0),
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 13.0),
    'Beta': (13.0, 30.0),
    'Gamma': (30.0, 40.0),
}


def _standardize_events(events):
    standardized = []
    for i, entry in enumerate(events):
        if isinstance(entry, dict):
            if 'raw' not in entry:
                raise ValueError(f"Event {i} has no 'raw' key. Keys: {list(entry.keys())}")
            raw = entry['raw']
            label = entry.get('description', entry.get('event_label', f'Event {i+1}'))
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            label, raw = entry
        else:
            raise TypeError(
                f"Entry {i}: expected dict with 'raw' or (label, Raw); got {type(entry).__name__}"
            )
        if not isinstance(raw, mne.io.BaseRaw):
            raise TypeError(f"Event {i} ({label}): raw must be an MNE Raw object")
        standardized.append((str(label), raw))
    if not standardized:
        raise ValueError('No events provided')
    return standardized


def _eeg_only(raw):
    picks = [
        ch_name for ch_name, ch_type in zip(raw.ch_names, raw.get_channel_types())
        if ch_type == 'eeg' and not ch_name.upper().startswith('EOG')
    ]
    if len(picks) < 4:
        raise ValueError(f'Need at least 4 EEG channels for scalp maps; found {picks}')

    eeg = raw.copy().pick(picks)

    # Ensure valid positions.
    pos = np.array([ch['loc'][:3] for ch in eeg.info['chs']])
    if (not np.all(np.isfinite(pos))) or np.any(np.linalg.norm(pos, axis=1) < 1e-8):
        eeg.set_montage(mne.channels.make_standard_montage('standard_1020'), on_missing='warn')
        pos = np.array([ch['loc'][:3] for ch in eeg.info['chs']])
    if (not np.all(np.isfinite(pos))) or np.any(np.linalg.norm(pos, axis=1) < 1e-8):
        raise ValueError('Some EEG channels have no valid locations. Check channel names/montage.')

    return eeg


def _integrate_band(psd, freqs, low, high):
    mask = (freqs >= low) & (freqs <= high)
    if np.count_nonzero(mask) < 2:
        raise ValueError(
            f'Not enough Welch frequency bins inside {low}-{high} Hz. '
            'Increase Welch window length or use longer events.'
        )
    return np.trapezoid(psd[:, mask], freqs[mask], axis=1)


def compute_event_band_powers(events, bands=BANDS, total_band=(1.0, 40.0), welch_window_sec=4.0):
    standardized = _standardize_events(events)
    results = []

    for label, raw in standardized:
        eeg = _eeg_only(raw)
        data_uV = eeg.get_data() * 1e6
        sfreq = float(eeg.info['sfreq'])
        nperseg = min(int(round(welch_window_sec * sfreq)), data_uV.shape[1])
        if nperseg < 8:
            raise ValueError(f'Event {label}: too few samples for Welch PSD')
        if max([total_band[1]] + [hi for _, hi in bands.values()]) > sfreq / 2:
            raise ValueError(f'Event {label}: upper frequency exceeds Nyquist ({sfreq/2:g} Hz)')

        freqs, psd = welch(
            data_uV,
            fs=sfreq,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            axis=-1,
            detrend='constant'
        )

        total_power = _integrate_band(psd, freqs, total_band[0], total_band[1])
        absolute = {}
        relative = {}
        for band_name, (low, high) in bands.items():
            abs_power = _integrate_band(psd, freqs, low, high)
            rel_power = 100.0 * np.divide(
                abs_power,
                total_power,
                out=np.full_like(abs_power, np.nan),
                where=total_power > 0
            )
            absolute[band_name] = abs_power
            relative[band_name] = rel_power

        results.append({
            'label': label,
            'info': eeg.info,
            'absolute': absolute,
            'relative': relative,
        })

    return results


def _range(values):
    arr = np.concatenate([np.asarray(v).ravel() for v in values])
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi


def plot_topomap_grid(results, bands=BANDS, kind='absolute', scale='global',
                      output_path=None, cmap='RdBu_r', show=False, dpi=200):
    if kind not in ('absolute', 'relative'):
        raise ValueError("kind must be 'absolute' or 'relative'")
    if scale not in ('global', 'event', 'band'):
        raise ValueError("scale must be 'global', 'event', or 'band'")

    band_names = list(bands.keys())
    nrows = len(results)
    ncols = len(band_names)

    figsize = (2.25 * ncols + 3.4, 1.78 * nrows + 1.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.subplots_adjust(
        left=0.19,
        right=0.87 if scale != 'band' else 0.91,
        top=0.92,
        bottom=0.09 if scale == 'band' else 0.055,
        wspace=0.12,
        hspace=0.33,
    )

    global_limits = _range([r[kind][b] for r in results for b in band_names])
    row_limits = [_range([r[kind][b] for b in band_names]) for r in results]
    col_limits = {b: _range([r[kind][b] for r in results]) for b in band_names}

    label = 'Absolute power (µV²)' if kind == 'absolute' else 'Relative power (%)'
    heading = 'Absolute Power' if kind == 'absolute' else 'Relative Power'
    scale_heading = {
        'global': 'Global Scale',
        'event': 'Event-Level Scale',
        'band': 'Frequency-band-Level Scale',
    }[scale]
    fig.suptitle(f'{heading} {scale_heading}', fontsize=13)

    for j, band_name in enumerate(band_names):
        axes[0, j].set_title(band_name, fontsize=11, pad=10)

    for i, result in enumerate(results):
        for j, band_name in enumerate(band_names):
            ax = axes[i, j]
            if scale == 'global':
                vmin, vmax = global_limits
            elif scale == 'event':
                vmin, vmax = row_limits[i]
            else:
                vmin, vmax = col_limits[band_name]

            mne.viz.plot_topomap(
                result[kind][band_name],
                result['info'],
                axes=ax,
                show=False,
                cmap=cmap,
                vlim=(vmin, vmax),
                contours=6,
                sensors=True,
                outlines='head',
                extrapolate='head',
                image_interp='cubic',
            )

            if j == 0:
                ax.text(
                    -0.27, 0.5, result['label'],
                    transform=ax.transAxes,
                    va='center', ha='right', fontsize=9, clip_on=False
                )

    if scale == 'global':
        cax = fig.add_axes([0.90, 0.18, 0.018, 0.65])
        fig.colorbar(
            ScalarMappable(norm=Normalize(*global_limits), cmap=cmap),
            cax=cax,
            label=label
        )
    elif scale == 'event':
        for i, limits in enumerate(row_limits):
            box = axes[i, -1].get_position()
            cax = fig.add_axes([0.90, box.y0 + 0.01, 0.012, max(box.height - 0.02, 0.025)])
            fig.colorbar(
                ScalarMappable(norm=Normalize(*limits), cmap=cmap),
                cax=cax
            ).ax.tick_params(labelsize=6)
        fig.text(0.938, 0.51, label, rotation=90, va='center', fontsize=9)
    else:
        for j, band_name in enumerate(band_names):
            box = axes[-1, j].get_position()
            cax = fig.add_axes([box.x0 + 0.012, 0.055, max(box.width - 0.024, 0.03), 0.012])
            fig.colorbar(
                ScalarMappable(norm=Normalize(*col_limits[band_name]), cmap=cmap),
                cax=cax,
                orientation='horizontal'
            ).ax.tick_params(labelsize=6)
        fig.text(0.94, 0.055, label, fontsize=9, va='center')

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f'Saved {output_path}')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def export_all_topomap_figures(events, output_dir='Spectral_Topomaps', show=False):
    """Compute spectral values once, then save exactly 6 topomap summary figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute once and reuse for all 6 figure types.
    results = compute_event_band_powers(events, bands=BANDS)

    configs = [
        ('ABP_Global_Scale.png', 'absolute', 'global'),
        ('ABP_Event_Level_Scale.png', 'absolute', 'event'),
        ('ABP_Frequency_Band_Level_Scale.png', 'absolute', 'band'),
        ('RBP_Global_Scale.png', 'relative', 'global'),
        ('RBP_Event_Level_Scale.png', 'relative', 'event'),
        ('RBP_Frequency_Band_Level_Scale.png', 'relative', 'band'),
    ]

    for filename, kind, scale in configs:
        plot_topomap_grid(
            results,
            bands=BANDS,
            kind=kind,
            scale=scale,
            output_path=output_dir / filename,
            show=show,
        )

    print(f'Finished: {len(configs)} figures in {output_dir.resolve()}')
    return results  # reusable numerical values, no second PSD computation


if __name__ == '__main__':
    print('Import this file in your EEG script AFTER generating event-specific MNE Raw segments.')
    print('Example: from replicate_eeg_topomaps_v2 import export_all_topomap_figures')
    print('         export_all_topomap_figures(eeg_events)')
