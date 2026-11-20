"""Re-create the event x frequency-band EEG topomap grids from supplied PNGs.

Compatible with NumPy 2.x, recent MNE releases, and event containers that are
lists of (label, mne.Raw) tuples or {'description': label, 'raw': Raw} dicts.

Usage AFTER obtaining the correct event-specific, preprocessed recordings:

    from replicate_eeg_topomaps import export_all_topomap_figures
    export_all_topomap_figures(eeg_events, output_dir='Spectral_Topomaps')

Or, when your cleaned segments are (label, Raw) tuples:

    export_all_topomap_figures(ica_applied_segments)

Do not run this module by itself: it needs your EEG recordings. It does not
perform ASR, ICA, or event segmentation.
"""
from pathlib import Path
import re
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
SUB_BANDS = {
    'Theta': (4.0, 8.0),
    'Low Alpha': (8.0, 10.0),
    'High Alpha': (10.0, 13.0),
    'Low Beta': (13.0, 20.0),
    'High Beta': (20.0, 30.0),
}


def _standardize_events(events):
    result = []
    for i, entry in enumerate(events):
        if isinstance(entry, dict):
            if 'raw' not in entry:
                raise ValueError(f'Event {i} has no raw key; got keys {list(entry)}')
            raw = entry['raw']
            label = entry.get('description', entry.get('event_label', f'Event {i + 1}'))
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            label, raw = entry
        else:
            raise TypeError(f'Entry {i}: expected (label, Raw) or dict with raw, got {type(entry).__name__}')
        if not isinstance(raw, mne.io.BaseRaw):
            raise TypeError(f'Event {i} ({label}): raw must be an MNE Raw object')
        result.append((str(label), raw))
    if not result:
        raise ValueError('No EEG events provided')
    return result


def _eeg_only(raw):
    """Only EEG sensors, dropping EOG even if incorrectly classified as EEG."""
    picks = [name for name, typ in zip(raw.ch_names, raw.get_channel_types())
             if typ == 'eeg' and not name.upper().startswith('EOG')]
    if len(picks) < 4:
        raise ValueError(f'Need >=4 EEG electrodes for scalp maps; found {picks}')
    eeg = raw.copy().pick(picks)
    # A standard montage fills in coordinates for conventional 10-20 names.
    # Existing montage positions are preserved when already present.
    pos = np.array([ch['loc'][:3] for ch in eeg.info['chs']])
    if not np.all(np.isfinite(pos)) or np.any(np.linalg.norm(pos, axis=1) < 1e-8):
        eeg.set_montage(mne.channels.make_standard_montage('standard_1020'), on_missing='warn')
    pos = np.array([ch['loc'][:3] for ch in eeg.info['chs']])
    if not np.all(np.isfinite(pos)) or np.any(np.linalg.norm(pos, axis=1) < 1e-8):
        raise ValueError('Some EEG channels have no locations. Check channel names and montage.')
    return eeg


def compute_event_band_powers(events, bands=BANDS, total_band=(1.0, 40.0),
                              welch_window_sec=4.0):
    """Compute integrated PSD band power in µV², and relative power in %.

    Relative power is 100 * band integrated PSD / integrated PSD(total_band)
    per EEG electrode. More precise than dividing by the sum of a set of bands
    when bands overlap or omit frequency ranges.
    """
    standardized = _standardize_events(events)
    output = []
    for label, raw in standardized:
        eeg = _eeg_only(raw)
        data_uV = eeg.get_data() * 1e6
        sfreq = float(eeg.info['sfreq'])
        nperseg = min(int(round(welch_window_sec * sfreq)), data_uV.shape[1])
        if nperseg < 8:
            raise ValueError(f'Event {label}: fewer than 8 samples')
        fs, power_spectrum = welch(data_uV, fs=sfreq, nperseg=nperseg,
                                   noverlap=nperseg // 2, axis=-1,
                                   detrend='constant')
        if max([total_band[1]] + [high for low, high in bands.values()]) > sfreq / 2:
            raise ValueError(f'Event {label}: upper frequency exceeds Nyquist ({sfreq/2:g} Hz)')

        def band_integral(low, high):
            keep = (fs >= low) & (fs <= high)
            if np.count_nonzero(keep) < 2:
                raise ValueError(f'{label}: not enough Welch bins in {low}-{high} Hz. '
                                 'Increase welch_window_sec or event duration.')
            return np.trapezoid(power_spectrum[:, keep], fs[keep], axis=1)

        denominator = band_integral(*total_band)
        abs_powers = {band: band_integral(*limits) for band, limits in bands.items()}
        rel_powers = {band: 100 * np.divide(power, denominator,
                                              out=np.full_like(power, np.nan),
                                              where=denominator > 0)
                      for band, power in abs_powers.items()}
        output.append({'label': label, 'info': eeg.info,
                       'channel_names': eeg.ch_names, 'absolute': abs_powers,
                       'relative': rel_powers})
    return output


def _range(values):
    numbers = np.concatenate([np.asarray(v).ravel() for v in values])
    numbers = numbers[np.isfinite(numbers)]
    if not numbers.size:
        return 0., 1.
    lo, hi = float(np.min(numbers)), float(np.max(numbers))
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi


def plot_topomap_grid(results, bands, kind='absolute', scale='global',
                      output_path=None, cmap='RdBu_r', show=True,
                      figsize=None, dpi=200):
    """Plot event rows x band columns, using one of three shared-scale modes.

    scale='global': all cells share same limits, one colorbar.
    scale='event': each event row has a shared scale, one colorbar per row.
    scale='band': each band column has a shared scale, one colorbar per column.
    """
    if scale not in ('global', 'event', 'band'):
        raise ValueError('scale must be global, event, or band')
    if kind not in ('absolute', 'relative'):
        raise ValueError('kind must be absolute or relative')
    names = list(bands)
    nrows, ncols = len(results), len(names)
    if figsize is None:
        figsize = (2.25*ncols + 3.4, 1.78*nrows + 1.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.subplots_adjust(left=.19, right=.87 if scale != 'band' else .91,
                        top=.92, bottom=.09 if scale == 'band' else .055,
                        wspace=.12, hspace=.33)

    globally = _range([r[kind][band] for r in results for band in names])
    row_ranges = [_range([r[kind][band] for band in names]) for r in results]
    col_ranges = {band: _range([r[kind][band] for r in results]) for band in names}
    label = 'Absolute power (µV²)' if kind == 'absolute' else 'Relative power (%)'
    heading = 'Absolute Power (µV²)' if kind == 'absolute' else 'Relative Power (%)'
    scale_heading = {'global': 'Global Scale', 'event': 'Event-Level Scale',
                     'band': 'Frequency-band-Level Scale'}[scale]
    fig.suptitle(f'{heading} {scale_heading}', fontsize=13)

    for j, name in enumerate(names):
        axes[0,j].set_title(name, fontsize=11, pad=10)

    for i, result in enumerate(results):
        for j, band in enumerate(names):
            ax = axes[i, j]
            limits = (globally if scale == 'global' else
                      row_ranges[i] if scale == 'event' else col_ranges[band])
            vmin, vmax = limits
            # MNE's topomap maps power measured at electrodes onto an interpolated scalp.
            mne.viz.plot_topomap(
                result[kind][band], result['info'], axes=ax, show=False,
                cmap=cmap, vlim=(vmin, vmax), contours=6, sensors=True,
                outlines='head', extrapolate='head', image_interp='cubic'
            )
            if j == 0:
                ax.text(-0.27, 0.5, result['label'], transform=ax.transAxes,
                        va='center', ha='right', fontsize=9, clip_on=False)
    if scale == 'global':
        axbar = fig.add_axes([.90, .18, .018, .65])
        fig.colorbar(ScalarMappable(norm=Normalize(*globally), cmap=cmap),
                     cax=axbar, label=label)
    elif scale == 'event':
        # Each event has its own bar to avoid misleading comparisons between rows.
        for i, limits in enumerate(row_ranges):
            box = axes[i,-1].get_position()
            axbar = fig.add_axes([.90, box.y0+.01, .012, max(box.height-.02, .025)])
            fig.colorbar(ScalarMappable(norm=Normalize(*limits), cmap=cmap),
                         cax=axbar).ax.tick_params(labelsize=6)
        fig.text(.938, .51, label, rotation=90, va='center', fontsize=9)
    else:
        for j, band in enumerate(names):
            box = axes[-1,j].get_position()
            axbar = fig.add_axes([box.x0+.012, .055, max(box.width-.024,.03), .012])
            fig.colorbar(ScalarMappable(norm=Normalize(*col_ranges[band]), cmap=cmap),
                         cax=axbar, orientation='horizontal').ax.tick_params(labelsize=6)
        fig.text(.94,.055,label, fontsize=9, va='center')

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches='tight', facecolor='white')
        print('Saved', p)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def export_all_topomap_figures(events, output_dir='Spectral_Topomaps', show=False):
    """Export the five comparable maps shown in the ZIP (not MODASR).

    Produces: global absolute, event-scaled absolute, band-scaled absolute,
    absolute sub-bands, and relative global. Modified-ASR maps require a
    separately processed recording and cannot be recreated from PNGs alone.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = compute_event_band_powers(events, bands=BANDS)
    configs = [
        ('ABP_Global_Scale.png', base, BANDS, 'absolute', 'global'),
        ('ABP_Event_Level_Scale.png', base, BANDS, 'absolute', 'event'),
        ('ABP_Frequency_Band_Level_Scale.png', base, BANDS, 'absolute', 'band'),
        ('RBP_Global_Scale.png', base, BANDS, 'relative', 'global'),
    ]
    sub = compute_event_band_powers(events, bands=SUB_BANDS)
    configs.append(('ABP_Global_Scale_Sub_Bands.png', sub, SUB_BANDS, 'absolute', 'global'))
    for filename, data, bands, kind, scale in configs:
        plot_topomap_grid(data, bands, kind=kind, scale=scale,
                          output_path=output / filename, show=show)
    print(f'Finished: {len(configs)} figures in {output.resolve()}')


if __name__ == '__main__':
    print('Import this file in your EEG script AFTER generating event-specific MNE Raw segments.')
    print('Example: from replicate_eeg_topomaps import export_all_topomap_figures')
    print('         export_all_topomap_figures(eeg_events)')
