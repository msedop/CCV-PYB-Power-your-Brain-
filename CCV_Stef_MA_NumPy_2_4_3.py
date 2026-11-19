import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt

import mne
from mne_icalabel import label_components
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from mne.viz import plot_topomap
from mne.channels import Layout
from mne import create_info
from mne.channels import find_ch_adjacency
from mne.channels import find_layout, make_eeg_layout
from mne.annotations import Annotations
from mne import events_from_annotations
from mne.time_frequency import tfr_array_morlet
from mne.datasets import ssvep

from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer

from scipy.signal import correlate
from scipy.signal import welch
from scipy.signal import find_peaks, peak_prominences
import scipy.stats as stats
from scipy.stats import iqr
from scipy.stats import zscore
import scipy.sparse

from collections import defaultdict

from datetime import datetime
from datetime import timedelta

from math import sqrt
import asrpy
# NumPy 2.4.3: before running, apply the one-time asrpy patch supplied
# in patch_asrpy_numpy2.py (fixes scalar conversion in fit_eeg_distribution).

from pathlib import Path
from copy import deepcopy
from scipy.integrate import simpson

import warnings
from mne_connectivity import spectral_connectivity_epochs
import seaborn as sns


plt.close("all")

# ================================ Loading Data ==============================
# -------------------------  Physiological Data ------------------------------

param_df = pd.read_csv(r"C:\Users\msedo\Documents\Power Your Brain\Prueba 4 10_09_2025\ParameterData.csv")

# Ensure time column is parsed as datetime
param_df['Time'] = pd.to_datetime(param_df['Time'], utc=True)
param_df.columns = param_df.columns.str.strip()


# -------------------------  EEG Data ------------------------------

edf_path = r"C:\Users\msedo\Documents\Power Your Brain\Prueba 4 10_09_2025\CCV_004_100920205_1.edf"

raw = mne.io.read_raw_edf(edf_path, preload=True)

prueba = os.path.basename(os.path.dirname(raw.filenames[0]))

# Print basic info
print(raw.info)

# =============================== Setup =======================================

# ------------------------- EEG Channel Setup -------------------------------

channels = raw.info['ch_names'][:20]

first_20_chs = raw.ch_names[:20]
raw.pick(first_20_chs)

# Your custom names for channels 1–20 (in order)
new_names = [
    'P8', 'T8', 'F8', 'F4', 'C4', 'P4',
    'EOG2 (FP2)', 'EOG1 (FP1)',
    'Fz', 'Cz', 'O1', 'Oz', 'O2',
    'Pz', 'P3', 'C3', 'F3', 'F7', 'T7', 'P7'
]

# Get current channel names (just the first 20 you want to rename)
original_names = raw.ch_names[:20]

# Create a mapping: old name → new name
rename_dict = dict(zip(original_names, new_names))

# Apply renaming
raw.rename_channels(rename_dict)

n_channels = len(raw.ch_names[:20])

sfreq = raw.info['sfreq']


# -------------------- Annotations from Events Excel File ---------------------

# Read the Excel file
df = pd.read_excel(r"C:\Users\msedo\Documents\Power Your Brain\Prueba 4 10_09_2025\CCV 004 Etapas Cirugía CCV Stefano.xlsx")

print(df.to_string(index=False))

# Get the recording start time from meas_date
meas_date = raw.info['meas_date']
if isinstance(meas_date, (tuple, list)):
    from datetime import datetime, timezone
    meas_date = datetime.fromtimestamp(meas_date[0], tz=timezone.utc)
elif hasattr(meas_date, 'to_pydatetime'):
    meas_date = meas_date.to_pydatetime()
elif meas_date is None:
    raise ValueError("meas_date is missing from the EDF file.")


start_seconds = meas_date.hour * 3600 + meas_date.minute * 60 + meas_date.second

onset_list = []
description_list = []
for idx, row in df.iterrows():
    event = row['Evento']
    time_obj = row['Hora 1']  # datetime.time object
    event_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + getattr(time_obj, 'second', 0)
    onset = event_seconds - start_seconds
    onset_list.append(onset)
    description_list.append(event)

duration_list = []

for i in range(len(onset_list)):
    if i < len(onset_list)-1:
        duration = onset_list[i+1] - onset_list[i]
    else:
        duration = 10
    
    duration_list.append(duration)


# Add annotations
annotations = mne.Annotations(onset=onset_list,
                              duration=duration_list,
                              description=description_list)

# Set annotation to EEG data
raw.set_annotations(annotations)

# Plot (vertical lines at H:M:S points)

raw.plot(time_format='%H:%M:%S', scalings='auto')

# if your data has EOG1/EOG2 and DOES NOT already have Fp1/Fp2, rename them to Fp1/Fp2
rename_map = {}
if 'EOG1 (FP1)' in raw.ch_names and 'Fp1' not in raw.ch_names:
    rename_map['EOG1 (FP1)'] = 'Fp1'
if 'EOG2 (FP2)' in raw.ch_names and 'Fp2' not in raw.ch_names:
    rename_map['EOG2 (FP2)'] = 'Fp2'

if rename_map:
    raw.rename_channels(rename_map)

montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, on_missing='ignore')

# rename back so channels keep the Fp1 positions but keep their EOG1/EOG2 names
if rename_map:
    inv_map = {v: k for k, v in rename_map.items()}
    raw.rename_channels(inv_map)



# ---------------- Set annotations to physiological data --------------------

# Filter to keep only physiological data from EEG start time onwards
param_df = param_df[param_df['Time'] >= meas_date].reset_index(drop=True)
param_df['offset_sec'] = (param_df['Time'] - meas_date).dt.total_seconds()


baseline = param_df[(param_df['offset_sec'] >= onset_list[0]) & (param_df['offset_sec'] < onset_list[0] + duration_list[0])]  # see earlier event_mask for first event

limits = {}

for param in ['HR(bpm)', 'SpO2(%)', 'PI(%)', 'RR(rpm)', 'PR(bpm)']:
    col_vals = pd.to_numeric(baseline[param], errors='coerce').dropna()
    mean_val = col_vals.mean()
    std_val = col_vals.std()
    lower_lim = mean_val - std_val
    upper_lim = mean_val + std_val
    limits[param] = {'low': lower_lim, 'high': upper_lim}

print(limits)

def count_crossings(values, thresh, direction='up'):
    # direction='up' for crossing above thresh
    # direction='down' for crossing below thresh
    count = 0
    last_val = None
    for val in values:
        if np.isnan(val):
            continue
        if last_val is not None:
            if direction == 'up':
                # Crossing: last below or at thresh, current above
                if last_val <= thresh and val > thresh:
                    count += 1
            elif direction == 'down':
                # Crossing: last above or at thresh, current below
                if last_val >= thresh and val < thresh:
                    count += 1
        last_val = val
    return count


event_physio_dict = {}

for onset, duration, desc in zip(onset_list, duration_list, description_list):
    event_mask = (param_df['offset_sec'] >= onset) & (param_df['offset_sec'] < onset + duration)
    event_phys = param_df[event_mask]
    phys_means = event_phys[['HR(bpm)', 'SpO2(%)', 'PI(%)', 'RR(rpm)', 'PR(bpm)']].apply(pd.to_numeric, errors='coerce').mean()

    exceed_count = {}
    fall_count = {}
    std = {}
    var = {}
    quantile = {}
    Range = {}
    
    
    for col, lim in limits.items():
        numeric_col = pd.to_numeric(event_phys[col], errors='coerce').values
        numeric_col_valid = numeric_col[np.isfinite(numeric_col)]  # ignore all NaN/inf
        
        # Print debugging counts
        print(f'Event: {desc}, Data: {col}, size: {numeric_col.size}, NaNs: {np.isnan(numeric_col).sum()}, Valid: {numeric_col_valid.size}')
        
        if numeric_col_valid.size == 0:
            # No valid data, so assign NaN
            exceed_count[col] = np.nan
            fall_count[col]   = np.nan
            std[col]          = np.nan
            var[col]          = np.nan
            quantile[col]     = np.nan
            Range[col]        = np.nan
            continue
        
        cross_up = count_crossings(numeric_col_valid, lim['high'], direction='up')
        cross_down = count_crossings(numeric_col_valid, lim['low'], direction='down')
        exceed_count[col] = cross_up
        fall_count[col]   = cross_down
        std[col]          = np.std(numeric_col_valid)
        var[col]          = np.var(numeric_col_valid)
        quantile[col]     = np.percentile(numeric_col_valid, 75) - np.percentile(numeric_col_valid, 25)
        Range[col]        = np.max(numeric_col_valid) - np.min(numeric_col_valid)
        
    event_physio_dict[desc] = {
        'mean': phys_means,
        'exceed_count': exceed_count,
        'fall_count': fall_count,
        'std': std,
        'var': var,
        'quantile': quantile,
        'range': Range
    }

duration_list_minutes = [d / 60 for d in duration_list]

summary_rows = []
for idx, (desc, values) in enumerate(event_physio_dict.items()):
    row = {'Event': desc}
    row['Time_on_task (min)'] = duration_list[idx] / 60  # duration in minutes
    # If you want seconds: row['Duration_sec'] = duration_list[idx]
    for param in values['mean'].keys():
        row[f'{param}_mean'] = values['mean'][param]
        row[f'{param}_std'] = values['std'][param]
        row[f'{param}_var'] = values['var'][param]
        row[f'{param}_iqr'] = values['quantile'][param]     # Interquartile range
        row[f'{param}_range'] = values['range'][param]
        row[f'{param}_above_high'] = values['exceed_count'][param]
        row[f'{param}_below_low']  = values['fall_count'][param]
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows).set_index('Event')
pd.set_option('display.max_columns', None)
print(df_summary.round(2))

# Correlation Matrix

# 1. If 'Event' is the index, reset it to make a fully numeric matrix
df_corr = df_summary.reset_index(drop=True)

# 2. Correlation matrix
corr_matrix = df_corr.corr(method='pearson')  # or 'spearman'

# 3. Plot
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Physiological Summary Variables")
plt.tight_layout()
plt.show()




# -------------- Plot Physiological data mean for each event -------------------

event_order = df_summary.index.tolist()  # already as Event index
# Make a list of parameter names
param_names = ['HR(bpm)', 'SpO2(%)', 'PI(%)', 'RR(rpm)', 'PR(bpm)']

# Get the columns for means
mean_cols = [f"{param}_mean" for param in param_names]

# Extract matrix for plotting
means_matrix = df_summary[mean_cols].values


plt.figure(figsize=(16, 7))  # Wider plot

for idx, param in enumerate(param_names):
    plt.plot(event_order, df_summary[f"{param}_mean"], marker='o', label=param)

# Optionally wrap long labels:
def wrap_labels(labels, width=15):
    return ['\n'.join([label[i:i+width] for i in range(0, len(label), width)]) for label in labels]

wrapped_labels = wrap_labels(event_order, width=18)
plt.xticks(range(len(wrapped_labels)), wrapped_labels, rotation=30, ha='right', fontsize=10)

plt.ylabel("Mean Value")
plt.xlabel("Event")
plt.title("Mean Physiological Parameters per Event")
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# --------------- Plot Threshold crossings for each physiological variable ------------

plt.figure(figsize=(16, 7))

for param in param_names:
    # Plot upper crossings (above_high) as positive
    plt.plot(event_order, df_summary[f"{param}_above_high"], marker='o', label=f'{param} Above (Positive)')
    # Plot lower crossings (below_low) as negative
    plt.plot(event_order, -df_summary[f"{param}_below_low"], marker='x', label=f'{param} Below (Negative)')

plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add a horizontal zero line for reference
plt.xticks(rotation=30, ha='right', fontsize=10)
plt.ylabel("Threshold Crossings")
plt.xlabel("Event")
plt.title("Number of Threshold Crossings per Event")
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# ======================== EEG Pre-processing ================================

# ------------------- Frequency content visualisation -------------------

# Compute and plot PSD
fig = raw.compute_psd(tmax=np.inf, fmax=250).plot(
    average=True, amplitude=False, picks="data", exclude="bads"
)

# Change the title
fig.axes[0].set_title("PSD before filters")

raw.plot_psd(fmax=120, dB=False)

# -------------------- Bandpass filter application -------------------------

# Optional bandpass and notch filtering (re-apply if unsure of quality)
raw.filter(l_freq=1.0, h_freq=80.0, fir_design='firwin')
#raw.notch_filter(freqs=[50], fir_design='firwin')  # Notch for 50 Hz power-line noise

# Plot to see filter changes
raw.plot(time_format='%M:%S', scalings = 'auto')

# Compute and plot PSD
fig = raw.compute_psd(tmax=np.inf, fmax=250).plot(
    average=True, amplitude=False, picks="data", exclude="bads"
)

# Change the title
fig.axes[0].set_title("PSD after filters")

raw.plot(time_format='%H:%M:%S', scalings = 'auto', duration=3)

# Apply average reference to the good channels only
raw.set_eeg_reference(ref_channels='average', projection=False)
raw.plot(time_format='%H:%M:%S', scalings = 'auto', duration=3)

# ================= Bad channel detection (Tukey) ================

# Calculate standard deviation for each channel
data = raw.get_data()
channel_std = np.std(data, axis=1)

def tukey_outliers(data, k=1.5):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return np.where((data < lower_bound) | (data > upper_bound))[0]

# Detect outlier channels
outlier_indices = tukey_outliers(channel_std)


# Define bad channels
bad_channels = [raw.ch_names[i] for i in outlier_indices if 'EOG' not in raw.ch_names[i]]

if bad_channels:
    bad_indices = [raw.ch_names.index(ch) for ch in bad_channels]
    bad_data, times = raw[bad_indices, :]

    # Plot Bad channels 
    plt.figure(figsize=(12, 6))
    for i, channel_data in enumerate(bad_data):
        plt.plot(times, channel_data * 1e6 + i * 100, label=bad_channels[i])

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude + offset (µV)')
    plt.title('Bad EEG Channels (Stacked)')
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()
else:
    print("No bad channels were detected.")



# Get the list of good channels
good_channels = [ch for ch in raw.ch_names if ch not in bad_channels]

# Get data for the good channels
good_indices = [raw.ch_names.index(ch) for ch in good_channels]
good_data, times = raw[good_indices, :]


# Plot good channels
plt.figure(figsize=(14, 6))
for i, ch_data in enumerate(good_data):
    plt.plot(times, ch_data * 1e6 + i * 100, label=good_channels[i])  # µV scaling and stacking

plt.xlabel('Time (s)')
plt.ylabel('Amplitude + offset (µV)')
plt.title('Good EEG Channels (Stacked)')
plt.legend(loc='upper right', fontsize='x-small', ncol=2)
plt.tight_layout()
plt.show()

all_channels = [ch for ch in raw.ch_names]
all_data, times = raw[: , :]


raw_g = raw.copy().pick(good_channels)

# =========================== Segmentation ==========================================

# Parameters
min_segment_duration = 15 * 60  # 15 minutes in seconds

# Start from third event --> First two events correspond to baseline (EC, EO)
start_idx = 2
segments = []

i = start_idx
while i < len(onset_list):
    seg_start = onset_list[i]
    seg_end = seg_start
    j = i
    # Group consecutive events until segment is >= 15 min or run out of events
    while j < len(onset_list):
        seg_end = onset_list[j] + duration_list[j]
        total_duration = seg_end - seg_start
        if total_duration >= min_segment_duration:
            break
        j += 1
    # Store segment index range
    segments.append((i, j))
    i = j + 1  # Start next segment after current

n_events = len(onset_list)
max_time = raw_g.times[-1]


    
# =========================== ASR ==========================================

# -------------------------- ASR Calibration --------------------------------
# Get the onsets and durations of the first two annotations
annotations = raw.annotations
onset1 = annotations.onset[0]
duration1 = annotations.duration[0]
# onset2 = annotations.onset[1]
# duration2 = annotations.duration[1]

# Calibration interval: first annotation only (as in the original script).
start_time = onset1
end_time = onset1 + duration1

eeg_chan_names = [ch for ch in raw_g.ch_names if ch not in [
    'EOG1 (FP1)', 'EOG2 (FP2)']]
raw_calib = raw_g.copy().crop(tmin=start_time, tmax=end_time)
raw_calib.pick(eeg_chan_names)

raw_calib.plot(title="First annotation (ASR calibration)",
               time_format='%H:%M:%S', scalings='80e-6')


# Fit ASR using this calibration segment
# Lower cuttoff values (5-10) are more restrictive. Higher values (20-50) are more permissive
asr = asrpy.ASR(sfreq=raw_calib.info['sfreq'], cutoff=20)
asr.fit(raw_calib)


# -------------------- ASR computation and result storage ---------------------

# Decimation to reduce computational complexity of ASR and ICA
raw_g.resample(256)

eog_channels = ['EOG1 (FP1)', 'EOG2 (FP2)']

asr_cleaned_segments = []
raw_segments = []

max_time = raw_g.times[-1]
for seg_num, (seg_start_idx, seg_end_idx) in enumerate(segments, 1):
    seg_onset = onset_list[seg_start_idx]
    if seg_end_idx >= n_events:
        seg_end = max_time
    else:
        seg_end = onset_list[seg_end_idx] + duration_list[seg_end_idx]
        # Fix the tmax issue here:
        if seg_end > max_time:
            seg_end = max_time
    segment = raw_g.copy().crop(tmin=seg_onset, tmax=seg_end)
    
    segment_channel_order = segment.ch_names.copy()
    
    # Separate EEG and EOG channels by name
    eeg_channel_names = [ch for ch in segment.ch_names if ch not in eog_channels]
    eog_channel_names = [ch for ch in segment.ch_names if ch in eog_channels]

    # Extract EEG and EOG subsegments
    segment_eeg = segment.copy().pick(eeg_channel_names)
    segment_eog = segment.copy().pick(eog_channel_names)
    
    # Apply ASR only to EEG channels
    cleaned_eeg = asr.transform(segment_eeg, mem_splits=100)

    # Get corresponding data arrays
    cleaned_eeg_data = cleaned_eeg.get_data()
    untouched_eog_data = segment_eog.get_data()

    # Find indices of EEG and EOG channels in original order
    eeg_indices = [segment_channel_order.index(ch) for ch in eeg_channel_names]
    eog_indices = [segment_channel_order.index(ch) for ch in eog_channel_names]
    
    # Now create a list where each element is either from cleaned eeg or untouched eog, in the order of segment_channel_order
    combined_data_ordered = []
    for ch in segment_channel_order:
        if ch in eeg_channel_names:
            idx = eeg_channel_names.index(ch)
            combined_data_ordered.append(cleaned_eeg_data[idx])
        elif ch in eog_channel_names:
            idx = eog_channel_names.index(ch)
            combined_data_ordered.append(untouched_eog_data[idx])
    combined_data_ordered = np.array(combined_data_ordered)
    
    sfreq_re = 256
    
    # Create info and RawArray
    ch_types = ['eeg' for ch in segment_channel_order]
    info = mne.create_info(segment_channel_order, sfreq_re, ch_types=ch_types)
    combined_segment = mne.io.RawArray(combined_data_ordered, info)
    
    # if your data has EOG1/EOG2 and DOES NOT already have Fp1/Fp2, rename them to Fp1/Fp2
    rename_map = {}
    if 'EOG1 (FP1)' in combined_segment.ch_names and 'Fp1' not in combined_segment.ch_names:
        rename_map['EOG1 (FP1)'] = 'Fp1'
    if 'EOG2 (FP2)' in combined_segment.ch_names and 'Fp2' not in combined_segment.ch_names:
        rename_map['EOG2 (FP2)'] = 'Fp2'

    if rename_map:
        combined_segment.rename_channels(rename_map)

    combined_segment.set_montage(montage, on_missing='ignore')

    # rename back so channels keep the Fp1 positions but keep their EOG1/EOG2 names
    if rename_map:
        inv_map = {v: k for k, v in rename_map.items()}
        combined_segment.rename_channels(inv_map)
    
    asr_cleaned_segments.append(combined_segment)
    raw_segments.append(segment.copy())  # Optionally, keep original for plotting etc.

    print(f"ASR applied to segment {seg_num} (events {seg_start_idx+1}-{min(seg_end_idx+1, n_events)})")
    
    # Plot using segment_channel_order
    plot_channel_names = segment_channel_order
    picks = [combined_segment.ch_names.index(ch) for ch in plot_channel_names]
    raw_data = segment.get_data(picks=picks)
    clean_data = combined_segment.get_data(picks=picks)

    times = segment.times

    # Convert to µV for plotting
    raw_data = raw_data * 1e6
    clean_data = clean_data * 1e6

    n_channels = raw_data.shape[0]
    channel_names_ds = plot_channel_names[:n_channels]
    peak = max(np.max(np.abs(raw_data)), np.max(np.abs(clean_data)))
    offset = peak if peak > 0 else 1.0

    plt.figure(figsize=(16, 8))
    for i in range(n_channels):
        plt.plot(times, raw_data[i] + i * offset,
                 color='red', alpha=0.5, linewidth=0.8, label='Raw' if i == 0 else "")
        plt.plot(times, clean_data[i] + i * offset,
                 color='black', alpha=0.9, linewidth=0.8, label='ASR Cleaned' if i == 0 else "")

    plt.yticks([i * offset for i in range(n_channels)], channel_names_ds)
    plt.xlabel("Time (s)")
    plt.ylabel("EEG & EOG Channels (stacked)")
    plt.title(f"Segment {seg_num}: Raw (Red) vs ASR-Cleaned (Black)")
    plt.legend(loc='upper right')
    plt.xlim(times[0], times[-1])
    plt.tight_layout()
    plt.show()


# ================================= ICA ====================================


def infer_region(ch_name):
    
    # Simplified region mapping
    region_map = {
        "AF": "Frontal", "Fp": "Frontal", "F": "Frontal", "FC": "Frontal",
        "T": "Temporal", "C": "Central", "CP": "Parietal", "P": "Parietal",
        "PO": "Occipital", "O": "Occipital"
    }    
    
    for prefix, region in region_map.items():
        if ch_name.startswith(prefix):
            return region
    return "Unknown"

def compute_frontal_correlation(component_ts, raw_data, sfreq, ch_names, start_time=10, duration=None):
    # raw_data: shape (channels, samples)
    if duration is None:
        duration = raw_data.shape[1] / sfreq  # use all samples

    start_sample = int(start_time * sfreq)
    end_sample = int((start_time + duration) * sfreq)

    eog_indices = [i for i, ch in enumerate(ch_names) if ch in ['EOG1 (FP1)', 'EOG2 (FP2)']]
    if not eog_indices:
        return np.nan

    ica_segment = component_ts[start_sample:end_sample]
    raw_segment = raw_data[eog_indices, start_sample:end_sample]
    
    corrs = [np.corrcoef(ica_segment, raw_segment[i])[0, 1] for i in range(len(eog_indices))]
    return np.mean(np.abs(corrs))


def fast_mi_matrix(sources, eeg, n_bins=16):
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    sources_disc = est.fit_transform(sources.T).T
    eeg_disc = est.fit_transform(eeg.T).T

    mi = np.zeros((sources.shape[0], eeg.shape[0]))
    for i in range(sources.shape[0]):
        for j in range(eeg.shape[0]):
            mi[i, j] = mutual_info_score(sources_disc[i], eeg_disc[j])
    return mi


def plot_clean_signals_with_labels(cleaned_data, times, channel_names, offset=100):
    """
    Plots cleaned EEG signals with channel names labeled on the left.

    Parameters:
        cleaned_data (np.ndarray): EEG data (n_channels, n_times), in µV
        times (np.ndarray): Time vector in seconds
        channel_names (list): List of channel names
        offset (int): Vertical offset between channel traces
    """
    plt.figure(figsize=(14, 8))
    n_channels = len(channel_names)

    for i in range(n_channels):
        y_offset = i * offset
        plt.plot(times, cleaned_data[i] + y_offset, color='black', linewidth=0.5)
        plt.text(times[0] - 1, y_offset, channel_names[i], va='center', ha='right', fontsize=8)

    plt.yticks(np.arange(n_channels) * offset, [])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude + offset (µV)")
    plt.title("Cleaned EEG Signals")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()


def compute_ica(raw, sfreq):
    
    # data = raw.copy().drop_channels(['EOG1 (FP1)', 'EOG2 (FP2)'])
    data = raw.copy()
    
    # map every channel name to 'eeg' and apply
    mapping = {ch: 'eeg' for ch in data.ch_names}
    data.set_channel_types(mapping)
        
    n_channels = len(raw.ch_names)

    # Perform ICA
    ica = ICA(n_components=int(len(data.ch_names)-1), random_state=97, max_iter=800)
    ica.fit(data)
    
    # Apply ICLabel to classify the ICA components
    labels = label_components(data, ica, method='iclabel')
    ica.labels_ = labels['labels'] #if you want to attach the labels to the ica object
    scores = labels['y_pred_proba'] #class probabilities per component
    
    # Visualize the ICA components
    #ica.plot_components()
    
    # ---------------- ICA components info ------------------------

    # Define frequency bands
    bands = {
        "Delta": (1, 4),
        "Theta": (4.25, 8),
        "Alpha": (8.25, 13),
        "Beta": (13.25, 30),
        "Gamma": (30.25, 45)
    }
    
    # Create output folder
    output_dir = r'C:\Users\msedo\Documents\CCV Stefano\Prueba 4 10_09_2025\ICA'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get ICA sources as a NumPy array
    sources_array = ica.get_sources(data).get_data()
    
    #--------------------- MI data -------------------------------------------- 
    
    sources_sub = sources_array[:, ::10]
    eeg_sub = data.get_data()[:, ::10]
    mi_matrix = fast_mi_matrix(sources_sub, eeg_sub)

    #--------------------------------------------------------------------------        

    times = data.times
    components = ica.get_components()
    ch_names = data.ch_names
    
    # Analysis loop per component
    summary = []
    
    for i in range(ica.n_components_):
        # time sequence
        ts = sources_array[i]
        
        # Peak-to-peak amplitude (µV)
        ptp = np.ptp(ts) * 1e6
        
        # Welch computation for frequency power
        power_freqs, power = welch(ts, fs=sfreq, nperseg=2048)
        
        # Dominant frequency
        dom_freq = power_freqs[np.argmax(power)]
    
        # Frequency band powers
        band_powers = {
            band: round(np.trapezoid(power[(power_freqs >= low) & (power_freqs <= high)],
                                 power_freqs[(power_freqs >= low) & (power_freqs <= high)]), 2)
            for band, (low, high) in bands.items()
        }
        
        max_band = max(band_powers, key=band_powers.get)
        
        # Top 3 channels contributing to the component
        weights = components[:, i]
        abs_weights = np.abs(weights)
        top_indices = abs_weights.argsort()[-3:][::-1]
        top_channels = [ch_names[idx] for idx in top_indices]
    
        # Dominant region
        max_ch = ch_names[np.argmax(abs_weights)]
        region = infer_region(max_ch)
        
        # Infer regions of the top 3 channels
        top_regions = [infer_region(ch) for ch in top_channels]
        
        # Count how many are in the same region as max_ch
        same_region_count = top_regions.count(region)
        
        
        eog_ch_names = ['EOG1 (FP1)', 'EOG2 (FP2)']
        raw_data_eog = raw_data_eog = raw.get_data(picks=eog_ch_names) * 1e6
        
        frontal_corr = compute_frontal_correlation(ts, raw_data_eog, sfreq, eog_ch_names)
    
        # Classification confidence
        label = ica.labels_[i]
        conf = labels['y_pred_proba'][i][0] if isinstance(labels['y_pred_proba'][i], (list, np.ndarray)) else labels['y_pred_proba'][i]
        
        # # Plotting component time series, topography and power spectrum (plots saved in separate files)
        # fig, axs = plt.subplots(1, 3, figsize=(15, 3))
        # axs[0].plot(times, ts)
        # axs[0].set_title(f'Component {i} Time Series')
        # plot_topomap(weights, data.info, axes=axs[1], show=False)
        # axs[1].set_title(f'Component {i} Topography')
        # axs[2].semilogy(power_freqs, power)
        # axs[2].set_xlim(0, 100)
        # axs[2].set_title('Power Spectrum')
        # plt.tight_layout()
        # plt.savefig(os.path.join(output_dir, f'ica_component_{i}_Stefano_CCV.png'), dpi=300)


        # ------------ MI values ----------------------------------------------
        
        mi_values = mi_matrix[i]
        mean_mi = np.mean(mi_values)
        max_mi = np.max(mi_values)
        top_mi_channel = ch_names[np.argmax(mi_values)]
        
        # -------------- ECG correlation components --------------------------



        # ---------- Summary --------------------------------------------------
        
        summary.append({
            "Component": i,
            "Predicted Label": label,
            "Confidence": round(conf, 3),
            "Peak-to-Peak (µV)": round(ptp, 2),
            "Dominant Frequency (Hz)": round(dom_freq, 2),
            "Delta Power": band_powers["Delta"],
            "Theta Power": band_powers["Theta"],
            "Alpha Power": band_powers["Alpha"],
            "Beta Power": band_powers["Beta"],
            "Gamma Power": band_powers["Gamma"],
            "Max Power Band": max_band,
            "Top Channels": ", ".join(top_channels),
            "Max Channel": max_ch,
            "Region": region,
            "Same Region": same_region_count,
            "Mean MI": round(mean_mi, 4),
            "Max MI": round(max_mi, 4),
            "Top MI Channel": top_mi_channel,
            "FC": frontal_corr
        })
        
   
    # Save summary
    df = pd.DataFrame(summary)
    print(df.to_string(index=False))
    # df.to_excel(os.path.join(output_dir, f'Stefano_CCV_{prueba}.xlsx'), index=True)
    

    #--------------------- ICA component Rejection ----------------------------------

    custom_reject = []
    for comp in summary:
        if (comp["Predicted Label"] == 'eye blink' or comp["Predicted Label"] == 'other') :
            if (
                (comp["Confidence"] > 0.5 or comp["FC"] > 0.3) and
                comp["Dominant Frequency (Hz)"] < 4 and
                comp["Max Power Band"] == "Delta" and
                (comp["Region"] == "Frontal" or comp["Top MI Channel"].startswith(("Fp", "AF", "F")) or comp["FC"] == np.max(df["FC"]))
            ):
                custom_reject.append(comp["Component"])

                
        elif comp["Predicted Label"] == 'muscle artifact':
            if (
                comp["Confidence"] > 0.5 and
                (comp["Max Power Band"] == "Beta" or comp["Max Power Band"] == "Gamma")
            ):
                custom_reject.append(comp["Component"])
                
                
        elif comp["Predicted Label"] == 'channel noise':
            custom_reject.append(comp["Component"])        


    # Apply ICA
    ica.exclude = custom_reject

    # Apply ICA to the EEG-only data
    cleaned_raw = ica.apply(raw.copy())
    
    print("Rejected components:", custom_reject)

    print("ica.exclude =", ica.exclude)


    print(f"Rejected components: {ica.exclude}")
    for i in ica.exclude:
        print(f"Component {i}: Label = {labels['labels'][i]}, Confidence = {labels['y_pred_proba'][i]}")

    
    # Before ICA cleaning
    raw.plot(title="Re-Referenced EEG pre ICA", n_channels = n_channels)

    # After ICA cleaning
    cleaned_raw.plot(title="Cleaned EEG (ICA applied)", n_channels = n_channels)
    
    
    # ------------------------ Manual Visualisation of blink component ----------------

    # Define EOG channels
    eog_channels = ['EOG1 (FP1)', 'EOG2 (FP2)']
    
    # Filter only the components in custom_reject with valid FC
    valid_fc_components = [
        comp for comp in summary
        if comp["Component"] in custom_reject and not np.isnan(comp["FC"])
    ]
    
    if valid_fc_components:
        # Select the component with highest FC from the rejected ones
        blink_component = max(valid_fc_components, key=lambda comp: comp["FC"])["Component"]
        max_fc_value = max(valid_fc_components, key=lambda comp: comp["FC"])["FC"]
        print(f"Selected rejected component with highest EOG correlation: {blink_component} (FC = {max_fc_value:.3f})")
    else:
        blink_component = None
        print("No rejected component had valid frontal correlation (FC).")
    

    scale_factor = 50
    
    # Time window
    start_time = 100  # seconds
    duration = 30    # seconds
    sfreq = data.info['sfreq']
    start_sample = int(start_time * sfreq)
    end_sample = int((start_time + duration) * sfreq)
    times = data.times[start_sample:end_sample]
    
    # ICA component time series
    ica_ts = sources_array[blink_component, start_sample:end_sample]
    
    # Get raw and cleaned EEG data
    raw_eeg_data = raw.get_data() * 1e6  # original in µV
    cleaned_eeg_data = cleaned_raw.get_data() * 1e6  # cleaned in µV
    
    # Plotting
    fig, axs = plt.subplots(len(eog_channels), 1, figsize=(12, 3 * len(eog_channels)), sharex=True)
    
    for i, ch in enumerate(eog_channels):
        idx = raw.ch_names.index(ch)
        raw_ts = raw_eeg_data[idx, start_sample:end_sample]
        clean_ts = cleaned_eeg_data[idx, start_sample:end_sample]
    
        axs[i].plot(times, raw_ts, label=f'{ch} (raw EEG)', color='black')
        axs[i].plot(times, ica_ts * scale_factor, label=f'ICA Component {blink_component} ×{scale_factor}', color='blue', alpha=0.7)
        axs[i].plot(times, clean_ts, label=f'{ch} (cleaned EEG)', color='green', linestyle='--')
        axs[i].set_ylabel("µV / scaled a.u.")
        axs[i].set_title(f"{ch} vs ICA Component {blink_component}")
        axs[i].legend()
        axs[i].grid(True)
    
    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
    
        
    # ------------------- Plot of clean signals overlapped with signals before ICA --------------

    # Get EEG data and times
    car_data, times = raw.get_data(return_times=True)
    cleaned_data = cleaned_raw.get_data()
    
    # Convert from Volts to microvolts
    car_data *= 1e6
    cleaned_data *= 1e6
    
    # Define vertical offset
    offset = 100
    
    # Plot
    plt.figure(figsize=(14, 6))
    
    for i, ch_name in enumerate(raw.ch_names):
        y_offset = i * offset
        plt.plot(times, car_data[i] + y_offset, color='red', linewidth=0.5)
        plt.plot(times, cleaned_data[i] + y_offset, color='black', linewidth=0.5)
        # Add channel label (optional)
        plt.text(times[0] - 1, y_offset, ch_name, va='center', ha='right', fontsize=8)
    
    # Add labeled y-axis ticks at channel positions
    yticks = np.arange(len(raw.ch_names)) * offset
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude + offset (µV)')
    plt.title('EEG Before (Red) and After ICA (Black)')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()
    
    chs = data.ch_names
    
    plot_clean_signals_with_labels(cleaned_data, times, chs)
    
    
    return summary, cleaned_raw


# asr_cleaned_segments: list of preprocessed, ASR-cleaned MNE Raw objects (one per segment)
# sfreq: sampling frequency (you can get it from each segment with segment.info['sfreq'])

all_ica_summaries = []
all_cleaned_ica = []

for seg_num, asr_segment in enumerate(asr_cleaned_segments, 1):
    print(f"\n=== Computing ICA for ASR-cleaned Segment {seg_num} ===")
    # Compute ICA on this segment
    summary, cleaned_raw = compute_ica(asr_segment, asr_segment.info['sfreq'])
    # Store results
    all_ica_summaries.append(summary)
    all_cleaned_ica.append(cleaned_raw)

#=================== Save each segmnt into fif ==========================

out_dir = r"C:\Users\msedo\Documents\CCV Stefano\Prueba 4 10_09_2025\Event_segments"
os.makedirs(out_dir, exist_ok=True)

for idx, raw_seg in enumerate(all_cleaned_ica, start=1):
    fname = os.path.join(out_dir, f"cleaned_seg_{idx:03d}.fif")
    raw_seg.save(fname, overwrite=True)
    print("Saved:", fname)
    
raws = all_cleaned_ica

# ============================ Read all fif files =============================

# Skip the first two events
start_idx = 2
durations = duration_list[start_idx:]
descriptions = description_list[start_idx:]

original_onsets = onset_list[start_idx:]



# # --- Read FIF files into a list ---
fif_path = r"C:\Users\msedo\Documents\CCV Stefano\Prueba 4 10_09_2025\Event_segments"
p = Path(fif_path)
folder = p.parent if p.is_file() else p

files = sorted(folder.glob("*.fif")) + sorted(folder.glob("*.fif.gz"))
if not files:
    raise FileNotFoundError(f"No .fif or .fif.gz files found in {folder}")

raws = []
file_paths = []
for f in files:
    try:
        raw = mne.io.read_raw_fif(str(f), preload=True)
    except Exception as e:
        print(f"Failed to read {f}: {e}")
        continue
    raws.append(raw)
    file_paths.append(str(f))
print(f"Loaded {len(raws)} raw file(s).")



for seg in raws:
    seg.info['bads'] = []



class EventExtractionError(Exception):
    pass


def split_segments_by_event_durations(segments, descriptions, durations, tol=5.0):
    """
    Extract sequential events from a list of segment Raw objects.

    Parameters
    ----------
    segments : list of mne.io.Raw
        Cleaned segment raws, in order.
    descriptions : list of str
        Labels for each event, in order.
    durations : list of float
        Duration (seconds) for each event, in order.
    tol : float
        Tolerance in seconds used when deciding if the remaining portion of a segment is
        "close enough" to the remaining event duration (default 5.0s).

    Returns
    -------
    events : list of dict
        Each dict has keys:
         - 'raw': mne.io.Raw of the extracted event (onset reset at 0)
         - 'description': str label
         - 'duration': float (requested event duration)
         - 'source_segments': list of (segment_index, start_sample_in_seg, stop_sample_in_seg)
           describing which pieces were stitched together (stop is exclusive).
    """
    if len(descriptions) != len(durations):
        raise ValueError("descriptions and durations must have the same length")

    if len(segments) == 0:
        raise ValueError("no segments provided")

    # Basic consistency checks: sampling frequency and channel order
    base_sfreq = segments[0].info['sfreq']
    base_chs = segments[0].ch_names
    for i, seg in enumerate(segments):
        if seg.info['sfreq'] != base_sfreq:
            raise ValueError(f"sampling frequency mismatch: segment 0 has {base_sfreq}, segment {i} has {seg.info['sfreq']}")
        if seg.ch_names != base_chs:
            # try to reorder if same set of names but different order
            if set(seg.ch_names) == set(base_chs):
                seg.reorder_channels(base_chs)
            else:
                raise ValueError(f"channel mismatch between segment 0 and segment {i}")

    sfreq = float(base_sfreq)
    tol_samples = int(round(tol * sfreq))

    seg_idx = 0
    seg_time_pos = 0.0  # seconds into current segment
    seg_sample_pos = 0  # sample index into current segment

    events = []

    # helper to take a piece from a segment by sample indices (relative to that segment)
    def take_piece_from_segment(seg, start_sample, stop_sample):
        """
        start_sample inclusive, stop_sample exclusive, indices relative to seg.
        Returns a RawArray containing the piece and the number of samples taken.
        """
        if start_sample < 0:
            start_sample = 0
        if stop_sample > seg.n_times:
            stop_sample = seg.n_times
        if stop_sample <= start_sample:
            return None, 0
        # get_data returns shape (n_channels, n_times)
        data_piece = seg.get_data(start=start_sample, stop=stop_sample)
        # Create a new RawArray with a deep-copy of info to avoid shared mutable state
        info_copy = deepcopy(seg.info)
        piece_raw = mne.io.RawArray(data_piece, info=info_copy)
        return piece_raw, (stop_sample - start_sample)

    # iterate over events
    for ev_idx, (desc, dur) in enumerate(zip(descriptions, durations)):
        if dur <= 0:
            raise ValueError(f"event {ev_idx} has non-positive duration: {dur}")

        remaining_needed_sec = float(dur)
        remaining_needed_samples = int(round(remaining_needed_sec * sfreq))
        collected_pieces = []
        source_segments = []

        while remaining_needed_samples > 0:
            # get current segment; if none left, we cannot fulfil the events
            if seg_idx >= len(segments):
                raise EventExtractionError(
                    f"Ran out of segments while extracting event {ev_idx} (description={desc}). "
                    f"{remaining_needed_samples} samples still needed."
                )

            seg = segments[seg_idx]
            seg_total_samples = seg.n_times

            # compute current sample pos in this segment
            # seg_sample_pos tracks sample index at which we should start in this seg
            start_sample = seg_sample_pos
            available_samples = seg_total_samples - start_sample
            if available_samples <= 0:
                # move to next segment
                seg_idx += 1
                seg_sample_pos = 0
                continue

            # Decide whether we can finish the event in this segment, considering tol
            # If available >= needed - tol -> finish here
            if available_samples >= remaining_needed_samples - tol_samples:
                # we will take the remaining_needed_samples (but not more than available)
                take_samples = min(remaining_needed_samples, available_samples)
                stop_sample = start_sample + take_samples
                piece_raw, n_taken = take_piece_from_segment(seg, start_sample, stop_sample)
                if n_taken == 0 or piece_raw is None:
                    raise EventExtractionError("failed to take piece from segment despite available samples")
                collected_pieces.append(piece_raw)
                source_segments.append((seg_idx, start_sample, stop_sample))
                # update pointers
                remaining_needed_samples -= n_taken
                seg_sample_pos = stop_sample
                # if we consumed the rest of the segment, move to next one
                if seg_sample_pos >= seg_total_samples:
                    seg_idx += 1
                    seg_sample_pos = 0
                # done for this event if remaining_needed_samples == 0
            else:
                # not enough here (and difference larger than tol) -> take entire remainder and continue
                stop_sample = seg_total_samples
                piece_raw, n_taken = take_piece_from_segment(seg, start_sample, stop_sample)
                if n_taken == 0 or piece_raw is None:
                    raise EventExtractionError("failed to take remainder piece from segment")
                collected_pieces.append(piece_raw)
                source_segments.append((seg_idx, start_sample, stop_sample))
                remaining_needed_samples -= n_taken
                # move to next segment
                seg_idx += 1
                seg_sample_pos = 0

        # At this point we have collected duration equal to the requested event duration (within rounding)
        # Concatenate pieces if needed
        if len(collected_pieces) == 0:
            raise EventExtractionError(f"no data collected for event {ev_idx}")
        elif len(collected_pieces) == 1:
            event_raw = collected_pieces[0]
        else:
            # Ensure identical channel order / types already checked earlier.
            event_raw = mne.concatenate_raws(collected_pieces, on_mismatch='ignore')

        # Reset annotation so the event starts at 0.0 in the returned raw
        event_raw.set_annotations(mne.Annotations(onset=[0.0], duration=[dur], description=[desc]))

        events.append({
            'raw': event_raw,
            'description': desc,
            'duration': float(dur),
            'source_segments': source_segments,
            'event_index': ev_idx
        })

    return events



events = split_segments_by_event_durations(asr_cleaned_segments, descriptions, durations, tol=5.0)
for e in events:
    print(e['event_index'], e['description'], e['duration'], e['source_segments'])

# ======================= Definition of frequency bands =======================

bands = {
    "Theta": (4.25, 8),
    "Low Alpha": (8.25, 10),
    "High Alpha": (10.25, 13),
    "Alpha": (8.25, 13),
    "Low Beta": (13.25, 20),
    "High Beta": (20.25, 30)
}    

# bands = {
#     "Delta": (0.5, 4),
#     "Theta": (4.25, 8),
#     "Alpha": (8.25, 13),
#     "Beta": (13.25, 30),
#     "Gamma": (30.25, 45)
# }   


#=============================================================================
# ============================  CONNECTIVITY  ================================
# ============================================================================

eog_channels = ['EOG1 (FP1)', 'EOG2 (FP2)']


EPOCH_DURATION = 5.0  # seconds
ALL_EVENT_RESULTS = []  # will collect per-event connectivity results
event_band_rows = []    # collect per-event x band average coherence (lower triangle)


#=============================================================================
# wPLI connectivity
# ============================================================================

def compute_stage_connectivity_summary(epochs, sfreq, fmin, fmax, method="wpli2_debiased"):
    n_chan = len(epochs.ch_names)
    if n_chan < 2:
        return np.nan, None

    # unique pairs only (lower triangle indices)
    idx = np.tril_indices(n_chan, k=-1)

    conn = spectral_connectivity_epochs(
        epochs,                  # pass Epochs object (recommended)
        method=method,           # "wpli2_debiased" or "imcoh"
        indices=idx,             # compute only unique pairs
        sfreq=sfreq,
        mode="multitaper",
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        n_jobs=1
    )

    # shape: (n_connections, 1) because faverage=True and one band
    vals = conn.get_data()
    vals = np.squeeze(vals)

    # For imcoh, values can be signed; use abs for magnitude summaries
    avg_pairs = float(np.nanmean(np.abs(vals)))

    # Optional: reconstruct a symmetric matrix for plotting
    mat = np.full((n_chan, n_chan), np.nan, dtype=float)
    mat[idx] = vals
    mat[(idx[1], idx[0])] = vals  # mirror

    # Diagonal handling:
    # coherence: diag=1.0
    # wpli2_debiased / wpli / imcoh: better to set diag=0 or NaN
    np.fill_diagonal(mat, 0.0)

    return avg_pairs, mat


for seg_idx, ev in enumerate(events, start=1):
    raw_obj = ev.get('raw')
    if raw_obj is None:
        print(f"Event {seg_idx}: no raw found, skipping")
        continue

    raw_seg = raw_obj.copy()
    seg_ch_names = raw_seg.ch_names.copy()
    eeg_names = [ch for ch in seg_ch_names if ch not in eog_channels]
    n_chan = len(eeg_names)
    if n_chan == 0:
        print(f"Event {seg_idx}: no EEG channels found, skipping")
        continue

    desc = ev.get('description', f'event_{seg_idx}')
    dur = ev.get('duration', f'event_{seg_idx}')
    print(f"\nEvent {seg_idx}: {desc} — computing connectivity on {n_chan} channels")

    data_seg = raw_seg.copy().pick(eeg_names)
    sfreq = data_seg.info['sfreq']

    epochs = mne.make_fixed_length_epochs(
        data_seg, duration=EPOCH_DURATION, overlap=0.0,
        preload=True, reject_by_annotation=True
    )

    # IMPORTANT: reset per-event dict here
    per_band_matrices = {}

    for band_name, (fmin, fmax) in bands.items():
        avg_val, mat = compute_stage_connectivity_summary(
            epochs=epochs,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            method="wpli2_debiased"
        )
        # store a copy of the matrix to avoid any accidental mutation later
        per_band_matrices[band_name] = None if mat is None else mat.copy()

    # IMPORTANT: append INSIDE the event loop
    ALL_EVENT_RESULTS.append({
        "event_index": seg_idx,
        "event_label": desc,
        "eeg_names": eeg_names,
        "connectivity": per_band_matrices
    })

print(f"Stored {len(ALL_EVENT_RESULTS)} events.")


def plot_all_events_connectivity(all_event_results,
                                 bands_order=None,
                                 vmin=0.0,
                                 vmax=1.0,
                                 cmap="viridis",
                                 mask_upper=True,
                                 show=True,
                                 save_dir=None,
                                 dpi=300):
    """
    all_event_results: list of dicts with keys:
      - 'event_index'
      - 'event_label'
      - 'eeg_names'
      - 'connectivity' : dict band_name -> (n_chan x n_chan) matrix (dense, symmetric)
    bands_order: list of band names to enforce a consistent order
    mask_upper: if True, show only lower triangle (unique pairs) in the heatmap
    """

    if not all_event_results:
        raise RuntimeError("No event results to plot.")

    if bands_order is None:
        bands_order = list(all_event_results[0]["connectivity"].keys())

    for evt in all_event_results:
        event_label = evt["event_label"]
        eeg_names = evt["eeg_names"]
        n_chan = len(eeg_names)

        n_bands = len(bands_order)
        fig_w = max(3, 3 * n_bands)
        fig_h = max(3, 3)
        fig, axes = plt.subplots(1, n_bands, figsize=(fig_w, fig_h), squeeze=False)
        axes = axes.ravel()

        for j, band in enumerate(bands_order):
            ax = axes[j]
            mat = evt["connectivity"].get(band, None)

            if mat is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                ax.set_axis_off()
                continue

            mat = np.asarray(mat)

            # Optional masking: display only lower triangle
            mask = None
            if mask_upper:
                mask = np.triu(np.ones_like(mat, dtype=bool), k=1)

            sns.heatmap(
                mat,
                ax=ax,
                xticklabels=eeg_names,
                yticklabels=eeg_names,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                square=True,
                cbar=True if j == n_bands - 1 else False,  # colorbar only on last plot
                mask=mask,
                linewidths=0.2
            )

            ax.set_title(band, fontsize=10)

            # Tidy tick labels
            plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)

        fig.suptitle(event_label, fontsize=12)
        fig.tight_layout(rect=[0, 0, 0.90, 0.95])

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(save_dir, f"connectivity_event_{evt['event_index']}.png")
            fig.savefig(fname, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(fig)

plot_all_events_connectivity(
    ALL_EVENT_RESULTS,
    bands_order=list(bands.keys()),
    vmin=0.0,
    vmax=1.0,
    cmap="viridis",
    mask_upper=True,                 # shows only the lower triangle
    show=True,
    save_dir=r"C:\Users\msedo\Documents\CCV Stefano\Prueba 4 10_09_2025\Coherence Results wPLI 2"
)


#=============================================================================
# Coherence connectivity
# ============================================================================


def avg_lower_triangle(matrix, exclude_diagonal=True):
    """
    Compute the average of the lower triangle (unique pairs) of a square matrix.

    Parameters
    - matrix: 2D square array-like (n_channels x n_channels)
    - exclude_diagonal: if True, exclude the diagonal when averaging

    Returns
    - float average of absolute values of the lower-triangle entries, or np.nan
      if there are no valid (finite) entries to average.
    """
    mat = np.asarray(matrix)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("avg_lower_triangle expects a square 2D array")
    n = mat.shape[0]
    if n < 2:
        return np.nan

    if exclude_diagonal:
        idx = np.tril_indices(n, k=-1)
    else:
        idx = np.tril_indices(n, k=0)

    vals = mat[idx]
    # keep finite values only
    finite_mask = np.isfinite(vals)
    if not np.any(finite_mask):
        return np.nan
    vals = vals[finite_mask]
    # coherence should be real and non-negative; be defensive and take absolute
    vals = np.abs(vals)
    if vals.size == 0:
        return np.nan
    return float(np.nanmean(vals))


# 1) Compute & store connectivity per event and per band
for seg_idx, ev in enumerate(events, start=1):
    raw_obj = ev.get('raw')
    if raw_obj is None:
        print(f"Event {seg_idx}: no raw found, skipping")
        continue

    raw_seg = raw_obj.copy()
    seg_ch_names = raw_seg.ch_names.copy()
    eeg_names = [ch for ch in seg_ch_names if ch not in eog_channels]
    n_chan = len(eeg_names)
    if n_chan == 0:
        print(f"Event {seg_idx}: no EEG channels found, skipping")
        continue

    desc = ev.get('description', f'event_{seg_idx}')
    print(f"\nEvent {seg_idx}: {desc} — computing connectivity on {n_chan} channels")

    # pick channels (preserve order in eeg_names)
    data_seg = raw_seg.copy().pick(eeg_names)
    sfreq = data_seg.info['sfreq']

    # Make fixed-length epochs (non-overlapping)
    epochs = mne.make_fixed_length_epochs(data_seg, duration=EPOCH_DURATION, overlap=0.0, preload=True)
    if len(epochs) == 0:
        print(f"  Event {seg_idx}: no {EPOCH_DURATION}s epochs created (data too short), skipping")
        continue

    data_epochs = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    print(f"  Created {data_epochs.shape[0]} epochs x {data_epochs.shape[1]} channels x {data_epochs.shape[2]} samples")

    # Compute connectivity for each band and store dense matrices
    per_band_matrices = {}
    per_band_freqs = {}
    for band_name, (fmin, fmax) in bands.items():
        try:
            conn = spectral_connectivity_epochs(
                data_epochs,
                method='coh',
                sfreq=sfreq,
                mode='multitaper',  
                fmin=fmin,
                fmax=fmax,
                faverage=True,       # average within the band -> returns single-frequency slice
                n_jobs=1
            )
        except Exception:
            # Try alternate mode (some installs don't accept multitaper)
            try:
                conn = spectral_connectivity_epochs(
                    data_epochs,
                    method='coh',
                    sfreq=sfreq,
                    mode='fourier',
                    fmin=fmin,
                    fmax=fmax,
                    faverage=True,
                    n_jobs=1
                )
            except Exception as exc2:
                raise RuntimeError(f"spectral_connectivity_epochs failed for band {band_name} in event {seg_idx}: {exc2}") from exc2


        # After get_data()
        dense = conn.get_data(output='dense')
        # Squeeze 3D to 2D if needed
        if dense.ndim == 3 and dense.shape[2] == 1:
            dense2d = dense[:, :, 0]
        elif dense.ndim == 2:
            dense2d = dense
        else:
            raise RuntimeError(f"Unexpected shape {dense.shape}")
        
        # Only set diagonal!
        np.fill_diagonal(dense2d, 1.0)
        
        print("Dense2D min:", dense2d.min(), "max:", dense2d.max(), "mean:", dense2d.mean())
        print("Sample:", dense2d[:5, :5])

        per_band_matrices[band_name] = dense2d
        # store freqs if available on object
        per_band_freqs[band_name] = getattr(conn, "freqs") if hasattr(conn, "freqs") else None

        print(f"  {band_name}: stored connectivity matrix shape {per_band_matrices[band_name].shape}")
        print(f"Band {band_name}, Event {desc}: min={dense2d.min():.3f}, max={dense2d.max():.3f}, mean={dense2d.mean():.3f}")
        print("Matrix sample (first 5 rows):\n", np.around(dense2d[:5, :5], 3))
    
    # ---------------------------
    # Compute average coherence per band using the lower triangle
    # ---------------------------
    n_epochs = data_epochs.shape[0]
    for band_name, mat in per_band_matrices.items():
        try:
            avg = avg_lower_triangle(mat, exclude_diagonal=True)
        except Exception:
            avg = np.nan
        event_band_rows.append({
            "event_index": seg_idx,
            "event_label": desc,
            "conn_band": f"conn_{band_name}",
            "avg_coherence_lower_triangle": avg,
            "n_channels": n_chan,
            "n_epochs": n_epochs
        })

    # append event-level results to global list
    ALL_EVENT_RESULTS.append({
        "event_index": seg_idx,
        "event_label": desc,
        "eeg_names": eeg_names,
        "connectivity": per_band_matrices,
        "freqs": per_band_freqs
    })

print(f"\nDone computing connectivity for {len(ALL_EVENT_RESULTS)} events.")

# Build DataFrames from collected averages
df_long = pd.DataFrame(event_band_rows)  # one row per event x band
if not df_long.empty:
    df_wide = df_long.pivot_table(
        index=["event_index", "event_label", "n_channels", "n_epochs"],
        columns="conn_band",
        values="avg_coherence_lower_triangle"
    ).reset_index()
    df_wide.columns.name = None
else:
    df_wide = pd.DataFrame()

# Example quick prints
print("\nAverage coherence (wide format) head:")
print(df_wide.head())


save_dir=r"C:\Users\msedo\Documents\CCV Stefano\Prueba 4 10_09_2025\Coherence Results"

# 2) Plot function: one figure per event, subplots for bands
def plot_all_events_connectivity(all_event_results, bands_order=None, vmin=0.0, vmax=1.0, cmap="viridis", show=True, save_dir=save_dir):
    """
    all_event_results: list of dicts with keys:
      - 'event_label'
      - 'eeg_names'
      - 'connectivity' : dict band_name -> (n_chan x n_chan) matrix
    bands_order: optional list of band names (if None uses keys from first event)
    """
    if len(all_event_results) == 0:
        raise RuntimeError("No event results to plot")
        
    band_names = bands_order or list(all_event_results[0]['connectivity'].keys())
    n_bands = len(band_names)
    for evt in all_event_results:
        event_label = evt['event_label']
        eeg_names = evt['eeg_names']
        n_chan = len(eeg_names)

        # build figure: one row with n_bands columns
        fig_w = max(3, 3 * n_bands)
        fig_h = max(3, 3)
        fig, axes = plt.subplots(1, n_bands, figsize=(fig_w, fig_h), squeeze=False)
        axes = axes.ravel()

        for j, band in enumerate(band_names):
            ax = axes[j]
            mat = evt['connectivity'].get(band, None)
            if mat is None:
                ax.text(0.5, 0.5, "no data", ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(band)
                continue

            # ensure 2D array
            mat = np.asarray(mat)
            if mat.ndim != 2:
                raise ValueError(f"Connectivity matrix for band {band} has unexpected shape {mat.shape}")

            # heatmap without colorbar (shared colorbar added later)
            sns.heatmap(mat, ax=ax, xticklabels=eeg_names, yticklabels=eeg_names,
                        vmin=vmin, vmax=vmax, cmap=cmap, square=True, cbar=False,
                        linewidths=0.2)

            ax.set_title(band, fontsize=10)
            if n_chan > 30:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
                plt.setp(ax.get_yticklabels(), fontsize=7)

        # shared colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        mappable = plt.cm.ScalarMappable(cmap=cmap)
        mappable.set_clim(vmin, vmax)
        fig.colorbar(mappable, cax=cbar_ax, label='Connectivity (coherence)')

        fig.suptitle(event_label, fontsize=12)
        fig.tight_layout(rect=[0, 0, 0.90, 0.95])

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(save_dir, f"connectivity_event_{evt['event_index']}.png")
            fig.savefig(fname, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        plt.close(fig)


# 3) plot all computed events
plot_all_events_connectivity(ALL_EVENT_RESULTS, vmin=0.0, vmax=1.0, cmap='viridis')


#=============================================================================
# ==========================  SPECTRAL ANALYSIS  =============================
# ============================================================================

def compute_psd_band_power(raw, bands, desc, n_fft = 2048):
    """Compute absolute and relative band power for each band."""
    # Compute mean PSD across epochs and channels
    fmin_all = min(b[0] for b in bands.values())
    fmax_all = max(b[1] for b in bands.values())
    psd_a = raw.compute_psd(method='welch', fmin=fmin_all, fmax=fmax_all, n_fft=n_fft)
    psd_avg = psd_a.get_data()  # shape: (n_channels, n_freqs)
    freqs = psd_a.freqs

    # Total power for each channel in full freq range
    total_power = simpson(psd_avg, freqs, axis=1)  # shape: (n_channels,)

    abs_powers = {}
    rel_powers = {}
    label = {}

    for band, (fmin, fmax) in bands.items():
        # Find freq indices for this band
        idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
        # Simpson integration across the band for each channel
        abs_power = simpson(psd_avg[:, idx_band], freqs[idx_band], axis=1) 
        rel_power = abs_power / total_power
        abs_powers[band] = abs_power
        rel_powers[band] = rel_power
        label[band] = desc

    return abs_powers, rel_powers

# In-place rename: if source channel exists, drop any existing target channel first to avoid duplicates,
# then rename the source to the target name.


for i, ev in enumerate(events):
    raw = ev['raw']
    raw.drop_channels(['EOG1 (FP1)', 'EOG2 (FP2)'])
        
    raw.set_montage(montage, on_missing='ignore')
    
    

n_events = len(events)
n_bands = len(bands)
if n_events == 0:
    raise RuntimeError("No events to plot")

# PSD settings (tweak n_fft and fmin/fmax as needed)
n_fft = 2048
fmin_all = min(b[0] for b in bands.values())
fmax_all = max(b[1] for b in bands.values())


# ========================== Absolute spectral power ===========================

# ======================== Shared colorbar ====================

# parameters
fmin_all = 0.5
fmax_all = 45.0
n_fft = 2048  # tune as needed
# bands dict already defined; example below if needed:
# bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 45)}

# storage
all_band_powers = []           # flattened list of all per-channel band powers across events
per_event_band_powers = []     # list of dicts to hold per-event results for plotting

for i, ev in enumerate(events):
    raw = ev.get('raw')
    if raw is None:
        print(f"Event {i}: no raw, skipping")
        continue
    desc = ev.get('description', f'event_{i}')

    # compute PSD (MNE returns PSD in V^2/Hz)
    psd_obj = raw.compute_psd(method='welch', fmin=fmin_all, fmax=fmax_all, n_fft=n_fft)
    psd = psd_obj.get_data()   # shape: (n_channels, n_freqs)
    freqs = psd_obj.freqs      # shape: (n_freqs,)

    # sanity
    if psd.size == 0 or freqs.size == 0:
        print(f"Event {i}: empty PSD, skipping")
        continue

    n_channels = psd.shape[0]
    ch_names = raw.ch_names.copy()

    # compute band powers per channel
    for band_name, (bmin, bmax) in bands.items():
        band_idx = np.logical_and(freqs >= bmin, freqs <= bmax)
        if not np.any(band_idx):
            band_power_uV2 = np.full(n_channels, np.nan)
        else:
            # integrate PSD (V^2/Hz) over freq to get V^2
            band_power_v2 = simpson(psd[:, band_idx], freqs[band_idx], axis=1)  # shape (n_channels,)
            band_power_uV2 = band_power_v2 * 1e12  # convert V^2 -> µV^2

        # collect for global vmin/vmax
        all_band_powers.extend(band_power_uV2.tolist())

        # store per-event, per-band for plotting later
        per_event_band_powers.append({
            'event_index': i,
            'event_label': desc,
            'band': band_name,
            'power_uV2': band_power_uV2,
            'channel_names': ch_names,
            'raw_info': raw.info  # keep info for plotting (montage, channel order)
        })

# compute global color limits
all_vals = np.array(all_band_powers, dtype=float)
all_vals = all_vals[~np.isnan(all_vals)]
if all_vals.size == 0:
    raise RuntimeError("No band-power values computed; check PSD / frequency band settings")

# Option A: exact min/max (use if you want absolute range)
global_vmin = float(np.nanmin(all_vals))
global_vmax = float(np.nanmax(all_vals))

# Option B: percentile-based (robust to outliers)
global_vmin_pct = float(np.nanpercentile(all_vals, 1))
global_vmax_pct = float(np.nanpercentile(all_vals, 99))

print("Global vmin/vmax (min/max):", global_vmin, global_vmax)
print("Global vmin/vmax (1/99 percentile):", global_vmin_pct, global_vmax_pct)

# choose which global scale to use
vmin, vmax = global_vmin_pct, global_vmax_pct




# parameters
fmin_all = 0.5
fmax_all = 45.0
n_fft = 2048

total_power_uV2_all_events = []  # accumulated total power in V^2

for i, ev in enumerate(events):
    raw = ev.get('raw')
    if raw is None:
        # skip events without raw data
        continue

    psd_obj = raw.compute_psd(method='welch', fmin=fmin_all, fmax=fmax_all, n_fft=n_fft)
    psd = psd_obj.get_data()   # shape: (n_channels, n_freqs)
    freqs = psd_obj.freqs

    if psd.size == 0 or freqs.size == 0:
        continue

    freq_idx = np.logical_and(freqs >= fmin_all, freqs <= fmax_all)
    if not np.any(freq_idx):
        continue

    # integrated power per channel (V^2)
    power_per_channel_v2 = simpson(psd[:, freq_idx], freqs[freq_idx], axis=1)

    # sum across channels to get event total (V^2) and add to grand total
    event_total_v2 = np.nansum(power_per_channel_v2)
    total_power_uV2_all_events.append(event_total_v2 * 1e12)

# convert to microvolt^2 for easier interpretation
total_power = max(total_power_uV2_all_events)

print(f"Total power across all events (0.5-45 Hz): {total_power:.6e} µV^2 ")


# ========== 2nd pass: Plotting topomaps in μV² with global vlim =========

fig, axes = plt.subplots(
    nrows=n_events,
    ncols=n_bands,
    figsize=(n_bands * 3, max(2.5, n_events * 2)),
    squeeze=False
)

# set a global title (adjust fontsize as needed)
fig.suptitle("Absolute Power (μV²) Global Scale", fontsize=16)

for i, ev in enumerate(events):
    raw = ev['raw']
    desc = ev.get('description', f'event_{i}')
    psd_obj = raw.compute_psd(method='welch', fmin=fmin_all, fmax=fmax_all, n_fft=n_fft)
    psd = psd_obj.get_data()
    freqs = psd_obj.freqs

    for j, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        ax = axes[i, j]
        band_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        if not np.any(band_idx):
            ax.text(0.5, 0.5, "no freqs", ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        band_power = simpson(psd[:, band_idx], freqs[band_idx], axis=1)  * 1e12   # μV²

        im, cn = mne.viz.plot_topomap(
            band_power,
            raw.info,
            axes=ax,
            show=False,
            names=None,
            cmap='RdBu_r',
            vlim=(vmin, vmax)
        )

        # column titles for the first row
        if i == 0:
            ax.set_title(band_name, fontsize=9)
        if j == 0:
            ax.set_ylabel(desc, rotation=0, labelpad=40, va='center', fontsize=9)

fig.tight_layout(rect=[0, 0, 0.95, 1])
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Absolute Power (μV²)')
plt.show()



# ================== Individual colorbars for events ====================


fig, axes = plt.subplots(
    nrows=n_events,
    ncols=n_bands,
    figsize=(n_bands * 3.2 + 0.7, max(3, n_events * 2.5)),
    squeeze=False
)

fig.suptitle("Absolute Power (μV²) Event-Level Scale", fontsize=14)

plt.subplots_adjust(left=0.08, right=0.92, wspace=0.25, hspace=0.1)

colorbar_axes = []
for i, ev in enumerate(events):
    raw = ev['raw']
    desc = ev.get('description', f'event_{i}')
    psd_obj = raw.compute_psd(method='welch', fmin=fmin_all, fmax=fmax_all, n_fft=n_fft)
    psd = psd_obj.get_data()
    freqs = psd_obj.freqs

    row_band_powers = []
    band_powers_list = []
    for band_name, (fmin, fmax) in bands.items():
        band_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        if not np.any(band_idx):
            band_powers_list.append(None)
            continue
        band_power = simpson(psd[:, band_idx], freqs[band_idx], axis=1)    # V²
        band_power_uV2 = band_power * 1e12                                 # μV²
        row_band_powers.append(band_power_uV2)
        band_powers_list.append(band_power_uV2)
    row_band_powers_concat = np.concatenate([bp for bp in band_powers_list if bp is not None])
    vmin = np.percentile(row_band_powers_concat, 5)
    vmax = np.percentile(row_band_powers_concat, 95)

    im = None
    for j, (band_name, band_power_uV2) in enumerate(zip(bands.keys(), band_powers_list)):
        ax = axes[i, j]
        if band_power_uV2 is None:
            ax.text(0.5, 0.5, "no freqs", ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        im, cn = mne.viz.plot_topomap(
            band_power_uV2,
            raw.info,
            axes=ax,
            show=False,
            names=None,
            cmap='RdBu_r',
            vlim=(vmin, vmax)
        )

        if i == 0:
            ax.set_title(band_name, fontsize=9)
        if j == 0:
            ax.set_ylabel(desc, rotation=0, labelpad=45, va='center', fontsize=9)

    # Add an unlabeled colorbar for this event row
    cbar_ax = fig.add_axes([0.92, ax.get_position().y0, 0.015, ax.get_position().height])
    fig.colorbar(im, cax=cbar_ax)
    colorbar_axes.append(cbar_ax)
    cbar_ax.yaxis.set_label_position('right')
    cbar_ax.tick_params(labelsize=8)

# Add one global colorbar label in the middle right
# Use the middle event as reference to center the label vertically
middle = n_events // 2
mid_ax = axes[middle, -1]
label_ax = fig.add_axes([0.95, mid_ax.get_position().y0, 0.03, mid_ax.get_position().height])
label_ax.axis('off')
label_ax.text(
    0.2, 0.5, "Absolute Power (μV²)",
    fontsize=13,
    ha='left', va='center', rotation=90,
    color='black', fontweight='bold'
)

plt.show()



# ====================== Individual colorbars for frequency bands =================

# Step 1: Calculate vlim for each band across all events and channels
band_vlims = {}
band_maps = {band_name: [] for band_name in bands}
for i, ev in enumerate(events):
    raw = ev['raw']
    psd_obj = raw.compute_psd(method='welch', fmin=fmin_all, fmax=fmax_all, n_fft=n_fft)
    psd = psd_obj.get_data()
    freqs = psd_obj.freqs
    for band_idx, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        freq_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        if not np.any(freq_idx):
            band_maps[band_name].append(None)
            continue
        band_power = simpson(psd[:, freq_idx], freqs[freq_idx], axis=1) * 1e12  # μV²
        band_maps[band_name].append(band_power)

# Calculate vlims for each band
for band_name in bands:
    band_powers = [arr for arr in band_maps[band_name] if arr is not None]
    band_powers_all = np.concatenate(band_powers) if band_powers else np.array([0])
    vmin = np.percentile(band_powers_all, 5)
    vmax = np.percentile(band_powers_all, 95)
    band_vlims[band_name] = (vmin, vmax)

# Step 2: Plot topomaps
fig, axes = plt.subplots(
    nrows=n_events,
    ncols=n_bands,
    figsize=(n_bands * 3.2 + 0.7, max(3, n_events * 2.5) + 1.4),
    squeeze=False
)
fig.suptitle("Absolute Power (μV²) Frequency-band-Level Scale", fontsize=14)

plt.subplots_adjust(left=0.08, right=0.92, wspace=0.25, hspace=0.1, bottom=0.12)


im_handles = [None for _ in range(n_bands)]
for i, ev in enumerate(events):
    raw = ev['raw']
    desc = ev.get('description', f'event_{i}')
    psd_obj = raw.compute_psd(method='welch', fmin=fmin_all, fmax=fmax_all, n_fft=n_fft)
    psd = psd_obj.get_data()
    freqs = psd_obj.freqs
    for j, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        ax = axes[i, j]
        freq_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        if not np.any(freq_idx):
            ax.text(0.5, 0.5, "no freqs", ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        band_power = simpson(psd[:, freq_idx], freqs[freq_idx], axis=1) * 1e12  # μV²
        vmin, vmax = band_vlims[band_name]
        im, cn = mne.viz.plot_topomap(
            band_power,
            raw.info,
            axes=ax,
            show=False,
            names=None,
            cmap='RdBu_r',
            vlim=(vmin, vmax)
        )
        im_handles[j] = im
        if i == 0:
            ax.set_title(band_name, fontsize=9)
        if j == 0:
            ax.set_ylabel(desc, rotation=0, labelpad=45, va='center', fontsize=9)

for j, band_name in enumerate(bands.keys()):
    anchor_ax = axes[-1, j]
    bbox = anchor_ax.get_position()
    # Move colorbar further down and give it a bit more height
    cbar_y = bbox.y0 - 0.09
    cbar_height = 0.05
    cbar_ax = fig.add_axes([bbox.x0, cbar_y, bbox.width, cbar_height])
    cb = fig.colorbar(im_handles[j], cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=8)

    # If this is the last column, add the 'Absolute Power' label to the right of the bar
    if j == (n_bands - 1):
        # get updated bbox of colorbar axis (in figure coords)
        cbar_bbox = cbar_ax.get_position()
        # Slight offset to place the label to the right of the cbar
        x_label = cbar_bbox.x1 + 0.01
        y_label = cbar_bbox.y0 + cbar_bbox.height / 2.0
        # place the label in figure coordinates
        fig.text(x_label, y_label, "Absolute Power (μV²)", fontsize=11,
                 ha='left', va='center', fontweight='bold')

plt.show()


# ================ Relative Power ====================

rel_pow = []

fig, axes = plt.subplots(
    nrows=n_events,
    ncols=n_bands,
    figsize=(n_bands * 3, max(2.5, n_events * 2)),
    squeeze=False
)

fig.suptitle("Relative Power (%)", fontsize=14)

plt.subplots_adjust(left=0.08, right=0.92, wspace=0.25, hspace=0.1)

for i, ev in enumerate(events):
    raw = ev['raw']
    desc = ev.get('description', f'event_{i}')

    # Compute PSD once per event across the full frequency range of interest
    psd_obj = raw.compute_psd(method='welch', fmin=fmin_all, fmax=fmax_all, n_fft=n_fft)
    psd = psd_obj.get_data()   # (n_channels, n_freqs)
    freqs = psd_obj.freqs
    
    tot_pow = simpson(psd, freqs, axis=1)

    for j, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        ax = axes[i, j]
        band_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        if not np.any(band_idx):
            ax.text(0.5, 0.5, "no freqs", ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        band_power =  simpson(psd[:, band_idx], freqs[band_idx], axis=1)
        
        # --- RELATIVE BAND POWER ---
        rel_power = (band_power / tot_pow) *100 # element-wise: per channel
        
        im, cn = mne.viz.plot_topomap(
            rel_power,  # use rel_power_db for dB scale, rel_power for [0-1] ratio
            raw.info,
            axes=ax,
            show=False,
            names=None,
            cmap='RdBu_r',
            vlim=(np.percentile(rel_power, 5), np.percentile(rel_power, 95))
        )

        # column titles for first row
        if i == 0:
            ax.set_title(band_name, fontsize=9)
        if j == 0:
            ax.set_ylabel(desc, rotation=0, labelpad=40, va='center', fontsize=9)
  
    # Add an unlabeled colorbar for this event row
    cbar_ax = fig.add_axes([0.92, ax.get_position().y0, 0.015, ax.get_position().height])
    fig.colorbar(im, cax=cbar_ax)
    colorbar_axes.append(cbar_ax)
    cbar_ax.yaxis.set_label_position('right')
    cbar_ax.tick_params(labelsize=8)

# Add one global colorbar label in the middle right
# Use the middle event as reference to center the label vertically
middle = n_events // 2
mid_ax = axes[middle, -1]
label_ax = fig.add_axes([0.95, mid_ax.get_position().y0, 0.03, mid_ax.get_position().height])
label_ax.axis('off')
label_ax.text(
    0.2, 0.5, "Relative Power (%)",
    fontsize=13,
    ha='left', va='center', rotation=90,
    color='black', fontweight='bold'
)

plt.show()    
  

# fig.tight_layout(rect=[0, 0, 0.95, 1])
# cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
# fig.colorbar(im, cax=cbar_ax, label='Relative Power (%)')

# plt.show()

# ===================== Summary table ====================================

rows = []

for i, ev in enumerate(events):
    raw = ev['raw']
    desc = ev.get('description', f'event_{i}')
    
    # Compute band power
    abs_power, rel_power = compute_psd_band_power(raw, bands, desc)

    for band in bands:
        rows.append({
            "Event": desc,
            "Band": band,
            "Absolute Power": abs_power[band][i]* 1e12,
            "Relative Power": rel_power[band][i]*100
        })
    
    
# Build DataFrame and export
power_topomap_df = pd.DataFrame(rows)
#power_topomap_df.to_excel(os.path.join(output_dir, f"{patient}_Band_Power_Per_Channel.xlsx"), index=False)
print(power_topomap_df.head())




# ==================== FREQUENCY ANALYSIS =================================

# ---------------- Time-Frequency Representation --------------------------

# ----------------- Grouping channels by region ---------------------------

bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 45)
}  

def group_channels_by_region(ch_names):
    """Group EEG channels by anatomical region based on first letter."""
    region_map = {
        'F': 'Frontal',
        'P': 'Parietal',
        'T': 'Temporal',
        'O': 'Occipital',
        'C': 'Central'
    }
    grouped = {region: [] for region in region_map.values()}
    for ch in ch_names:
        if not ch or not ch[0].upper() in region_map: continue
        region = region_map[ch[0].upper()]
        grouped[region].append(ch)
    return grouped

def compute_mean_band_powers(raw, bands, desc):

    ch_regions = group_channels_by_region(raw.ch_names)
    psd_obj = raw.compute_psd(method='welch', fmin=min(b[0] for b in bands.values()),
                              fmax=max(b[1] for b in bands.values()))
    psd = psd_obj.get_data()    # shape: (n_channels, n_freqs), units: V²/Hz
    freqs = psd_obj.freqs

    abs_powers = {}   # region -> band -> mean absolute power (μV²)
    rel_powers = {}   # region -> band -> mean relative power (unitless)

    for region, chs in ch_regions.items():
        if not chs: 
            abs_powers[region] = {}
            rel_powers[region] = {}
            continue

        # Get indices for channels in this region
        ch_idxs = [raw.ch_names.index(ch) for ch in chs if ch in raw.ch_names]
        if not ch_idxs:
            abs_powers[region] = {}
            rel_powers[region] = {}
            continue

        # Total regional power: sum power across all region channels and all frequencies
        # This is a scalar (μV²)
        region_psd = psd[ch_idxs]  # shape: (n_channels_in_region, n_freqs)
        region_total_power = simpson(region_psd, freqs, axis=1).sum() * 1e12

        abs_powers[region] = {}
        rel_powers[region] = {}

        for band, (fmin, fmax) in bands.items():
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
            if not np.any(idx_band):
                abs_powers[region][band] = np.nan
                rel_powers[region][band] = np.nan
                continue
            # Absolute band power: mean is average across channels in region
            region_band_power_by_ch = simpson(region_psd[:, idx_band], freqs[idx_band], axis=1) * 1e12
            region_band_power_total = region_band_power_by_ch.sum()  # sum across region channels

            abs_powers[region][band] = np.mean(region_band_power_by_ch)
            # Region relative power: total band power in region / total power in region
            rel_powers[region][band] = region_band_power_total / region_total_power if region_total_power > 0 else np.nan

    return abs_powers, rel_powers, desc

results = []
for i, ev in enumerate(events):
    raw = ev['raw']
    desc = ev.get('description', f'event_{i}')
    abs_powers, rel_powers, desc = compute_mean_band_powers(raw, bands, desc)
    for region in abs_powers:
        for band in bands:
            results.append({
                "Event": desc,
                "Region": region,
                "Band": band,
                "Mean Absolute Power (μV²)": abs_powers[region].get(band, np.nan),
                "Mean Regional Relative Power (%)": rel_powers[region].get(band, np.nan) * 100
            })


per_event_band_data = pd.DataFrame(results)

print(per_event_band_data.to_string(index=False))

alpha_bands = per_event_band_data[per_event_band_data['Band'] == 'Alpha']


# ====================== Theta/ Alpha ratios ===============================


# New bands: include combined "Alpha" plus Low/High sub-bands
bands = {
    "Theta": (4.25, 8),
    "Low Alpha": (8.25, 10),
    "High Alpha": (10.25, 13),
    "Alpha": (8.25, 13),        # combined alpha band to be used for the normal alpha ratio
    "Low Beta": (13.25, 20),
    "High Beta": (20.25, 30)
}

output_dir_res = r"C:\Users\msedo\Documents\CCV Stefano\Prueba 4 10_09_2025\Results"
os.makedirs(output_dir_res, exist_ok=True)

results = []               # detailed per-region x band results (long)
ratios_rows = []           # one row per event with all ratios (wide)
regional_ratios_list = []  # regional ratios (one row per event)
global_ratios_list = []    # global ratios (one row per event)

for i, ev in enumerate(events):
    raw = ev['raw']
    desc = ev.get('description', f'event_{i}')

    # compute_mean_band_powers must exist in your environment:
    # it should accept (raw, bands, desc) and return (abs_powers, rel_powers, desc)
    # Where abs_powers[region][band] gives the mean absolute power (μV²) for that region and band.
    abs_powers, rel_powers, desc = compute_mean_band_powers(raw, bands, desc)

    # ---------- Regional ratios ----------
    frontal_theta = abs_powers.get("Frontal", {}).get("Theta", np.nan)
    parietal_high_alpha = abs_powers.get("Parietal", {}).get("High Alpha", np.nan)
    parietal_alpha = abs_powers.get("Parietal", {}).get("Alpha", np.nan)  # combined alpha band

    # Regional ratio using combined Alpha
    if (not np.isnan(frontal_theta)) and (not np.isnan(parietal_alpha)) and (parietal_alpha != 0):
        reg_ratio_alpha = frontal_theta / parietal_alpha
    else:
        reg_ratio_alpha = np.nan

    # Regional ratio using High Alpha
    if (not np.isnan(frontal_theta)) and (not np.isnan(parietal_high_alpha)) and (parietal_high_alpha != 0):
        reg_ratio_highalpha = frontal_theta / parietal_high_alpha
    else:
        reg_ratio_highalpha = np.nan

    regional_ratios_list.append({
        "Event": desc,
        "FrontalTheta_ParietalAlpha": reg_ratio_alpha,
        "FrontalTheta_ParietalHighAlpha": reg_ratio_highalpha
    })

    # ---------- Global ratios (via PSD integration) ----------
    fmin_all = min(b[0] for b in bands.values())
    fmax_all = max(b[1] for b in bands.values())
    psd_obj = raw.compute_psd(method='welch', fmin=fmin_all, fmax=fmax_all)
    psd = psd_obj.get_data()    # (n_channels, n_freqs)
    freqs = psd_obj.freqs

    mean_abs = {}
    for band_name, (fmin, fmax) in bands.items():
        idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
        if not np.any(idx_band):
            mean_abs[band_name] = np.nan
            continue
        # integrate PSD over band for each channel, convert V^2/Hz -> V^2 by integrating, then to μV²
        band_power = simpson(psd[:, idx_band], freqs[idx_band], axis=1) * 1e12  # μV² per channel
        mean_abs[band_name] = np.mean(band_power)  # average across channels

    global_theta = mean_abs.get("Theta", np.nan)
    global_high_alpha = mean_abs.get("High Alpha", np.nan)
    global_alpha = mean_abs.get("Alpha", np.nan)  # combined alpha

    # Global ratio using combined Alpha
    if (not np.isnan(global_theta)) and (not np.isnan(global_alpha)) and (global_alpha != 0):
        glob_ratio_alpha = global_theta / global_alpha
    else:
        glob_ratio_alpha = np.nan

    # Global ratio using High Alpha
    if (not np.isnan(global_theta)) and (not np.isnan(global_high_alpha)) and (global_high_alpha != 0):
        glob_ratio_highalpha = global_theta / global_high_alpha
    else:
        glob_ratio_highalpha = np.nan

    global_ratios_list.append({
        "Event": desc,
        "GlobalTheta_Alpha": glob_ratio_alpha,
        "GlobalTheta_HighAlpha": glob_ratio_highalpha
    })

    # ---------- Detailed per-region x band results (long) ----------
    for region in abs_powers:
        # record every band value present in bands dict
        for band in bands.keys():
            abs_val = abs_powers[region].get(band, np.nan)
            rel_val = rel_powers[region].get(band, np.nan) if rel_powers is not None else np.nan
            results.append({
                "Event": desc,
                "Region": region,
                "Band": band,
                "Mean Absolute Power (μV²)": abs_val,
                "Mean Relative Power (%)": (rel_val * 100) if (not np.isnan(rel_val)) else np.nan
            })

    # ---------- Single-row wide ratios summary ----------
    ratios_rows.append({
        "Event": desc,
        "FrontalTheta_ParietalAlpha": reg_ratio_alpha,
        "FrontalTheta_ParietalHighAlpha": reg_ratio_highalpha,
        "GlobalTheta_Alpha": glob_ratio_alpha,
        "GlobalTheta_HighAlpha": glob_ratio_highalpha
    })

# ---------- Build DataFrames ----------
df = pd.DataFrame(results)                      # detailed region x band table (long)
regional_ratio_df = pd.DataFrame(regional_ratios_list)   # regional ratios (one row per event)
global_ratio_df = pd.DataFrame(global_ratios_list)       # global ratios (one row per event)
ratios_summary_df = pd.DataFrame(ratios_rows)            # wide ratios summary (one row per event with all ratios)

# Normalize key column name to match your other dfs
df = df.rename(columns={"Event": "event_label"})
regional_ratio_df = regional_ratio_df.rename(columns={"Event": "event_label"})
global_ratio_df = global_ratio_df.rename(columns={"Event": "event_label"})
ratios_summary_df = ratios_summary_df.rename(columns={"Event": "event_label"})

# Save summary files
df.to_excel(os.path.join(output_dir_res, "band_powers_by_region_with_alpha_and_highalpha.xlsx"), index=False)
ratios_summary_df.to_excel(os.path.join(output_dir_res, "ratios_summary_wide_alpha_and_highalpha.xlsx"), index=False)
regional_ratio_df.to_excel(os.path.join(output_dir_res, "regional_theta_parietal_alpha_ratios.xlsx"), index=False)
global_ratio_df.to_excel(os.path.join(output_dir_res, "global_theta_alpha_ratios.xlsx"), index=False)

# Quick preview
print("Detailed band-power table (head):")
print(df.head())
print("\nRatios summary (head):")
print(ratios_summary_df.head())


# ================================ GLOBAL SUMMARY TABLE ========================================

# Preserve original Event and Band order when pivoting power_topomap_df
# Minimal change to your existing code.

import pandas as pd

# ensure working with strings to avoid dtype mismatches
power_topomap_df['Event'] = power_topomap_df['Event'].astype(str)
power_topomap_df['Band']  = power_topomap_df['Band'].astype(str)

# capture the original order of first occurrence (preserves the order seen in the long DF)
event_order = power_topomap_df['Event'].drop_duplicates().tolist()
band_order  = power_topomap_df['Band'].drop_duplicates().tolist()




# pivot as before
power_abs = power_topomap_df.pivot(index="Event", columns="Band", values="Absolute Power")

# reindex rows and columns to the original order
power_abs = power_abs.reindex(index=event_order, columns=band_order)

# rename columns (keeps the band order you had in the long DF)
power_abs.columns = [f"Absolute_{band}" for band in power_abs.columns]

# Example: if you also want relative power and then combine, do the same reindexing there
power_rel = power_topomap_df.pivot(index="Event", columns="Band", values="Relative Power")
power_rel = power_rel.reindex(index=event_order, columns=band_order)
power_rel.columns = [f"Relative_{band}" for band in power_rel.columns]

# combine into wide power dataframe
power_wide = pd.concat([power_abs, power_rel], axis=1).reset_index().rename(columns={"Event": "event_label"})

# sanity check: event order matches original
assert power_wide['event_label'].tolist() == event_order, "Event order not preserved"

# 2) build desired conn column names in that order
desired_conn_cols = [f"conn_{b}" for b in band_order]

# 3) pick only those conn columns that actually exist in df_wide (defensive)
existing_conn_cols = [c for c in desired_conn_cols if c in df_wide.columns]

# 4) define meta columns to keep at the front (only include those present)
meta_cols = [c for c in ["event_index", "event_label", "n_channels", "n_epochs"] if c in df_wide.columns]

# 5) remaining columns (preserve their current order)
remaining = [c for c in df_wide.columns if c not in meta_cols + existing_conn_cols]

# 6) final column order: meta -> conn (in band_order) -> remaining
final_cols = meta_cols + existing_conn_cols + remaining

# 7) reindex dataframe columns
df_wide = df_wide.reindex(columns=final_cols)


if existing_conn_cols:
    seq = [c.replace("conn_", "") for c in existing_conn_cols]
    assert seq == [b for b in band_order if f"conn_{b}" in existing_conn_cols], "Conn columns not in expected band order"


# Merge with connectivity dataframe (df_wide) on event_label
# keep df_wide order (simple)
global_df = pd.merge(df_wide, power_wide, on="event_label", how="left", validate="one_to_one")

# Inspect and save
print(global_df.head())


# Drop the first two rows from df_summary before creating event_label and merging.
# Minimal change to your existing pipeline.

# If df_summary already has 'Event' as a column this is a no-op; if 'Event' is the index it becomes a column.
if df_summary.index.name == "Event" or "Event" not in df_summary.columns:
    df_summary = df_summary.reset_index()

# Drop the first two event rows (ignore them) and reset the index
df_summary = df_summary.iloc[2:].reset_index(drop=True)

# Now safely create the string key used for merging
df_summary["event_label"] = df_summary["Event"].astype(str).str.strip()

# Now, before merging, ensure global_df has the same key column and type/format:
if 'event_label' not in global_df.columns:
    if 'Event' in global_df.columns:
        global_df = global_df.rename(columns={'Event': 'event_label'})
    else:
        global_df = global_df.reset_index().rename(columns={global_df.columns[0]: 'event_label'})

global_df['event_label'] = global_df['event_label'].astype(str).str.strip()

# Finally, the simple merge (left merge keeps all rows from global_df)
merged_df = global_df.merge(df_summary, on='event_label', how='left')

print(merged_df)
# merged_df.to_csv("global_with_summary.csv", index=False)

# drop 'Event' if present (no error if it's missing)
merged_df = merged_df.drop(columns=['Event','n_channels','n_epochs'], errors='ignore')


import unicodedata

# 1) Ensure ratios_summary_df has an 'event_label' column
if "event_label" not in ratios_summary_df.columns:
    if "Event" in ratios_summary_df.columns:
        ratios_summary_df = ratios_summary_df.rename(columns={"Event": "event_label"})
    else:
        ratios_summary_df = ratios_summary_df.reset_index().rename(columns={ratios_summary_df.index.name or 0: "event_label"})

# 2) Ensure merged_df has an 'event_label' column (your code likely already did this)
if "event_label" not in merged_df.columns:
    if "Event" in merged_df.columns:
        merged_df = merged_df.rename(columns={"Event": "event_label"})
    else:
        merged_df = merged_df.reset_index().rename(columns={merged_df.index.name or 0: "event_label"})

# 3) Minimal normalization function: strip, lowercase, remove accents
def _norm_label(s):
    s = "" if s is None else str(s)
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

merged_df["event_key"] = merged_df["event_label"].apply(_norm_label)
ratios_summary_df["event_key"] = ratios_summary_df["event_label"].apply(_norm_label)

# 4) Merge on the normalized key and keep all rows from merged_df
#    Drop the duplicate readable label from the right-hand side to avoid column name collisions.
right = ratios_summary_df.drop(columns=["event_label"])
merged_with_ratios = merged_df.merge(right, on="event_key", how="left", suffixes=("", "_ratios"))

# Cleanup: drop the helper key column
merged_with_ratios = merged_with_ratios.drop(columns=["event_key"])

# rounds all numeric columns to 2 decimal places
merged_with_ratios = merged_with_ratios.round(2)

# Display with 2 decimals in notebooks (optional, affects only display)
pd.options.display.float_format = "{:.2f}".format

print(merged_with_ratios.head())

# Save if desired
merged_with_ratios.to_excel(r"C:\Users\msedo\Documents\CCV Stefano\Prueba 4 10_09_2025\Results\merged_df.xlsx", index=False)


num = merged_df.select_dtypes(include=[np.number]).copy()

# Optionally drop columns you don't want to include (indices / counts)
# exclude = ["event_index", "n_channels", "n_epochs"]
# num = num.drop(columns=[c for c in exclude if c in num.columns])

# Compute correlation matrix (Pearson)
corr = num.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
mask = np.zeros_like(corr, dtype=bool)  # no mask by default
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, linewidths=.5)
plt.title("Correlation matrix across events (one row per event)")
plt.tight_layout()
plt.show()



"""
Event-level feature analysis pipeline
Reads 'merged_df.csv' (one row per chronologic event) and runs a set of
analyses to identify which variables change most per event and which
variables co-vary across events.

Produces:
 - z-score heatmap (events x variables)
 - percent-change vs previous-event table & top-K lists per event
 - PCA (scores, loadings) and PC1/PC2 trajectory plot
 - Mahalanobis distance per event and per-variable contributions
 - Correlation matrices (Pearson + Spearman) and clustermap of variables
 - CSV summary files (top-K changed variables, PCA loadings, Mahalanobis outputs)

Usage:
 - Place this file in the same folder as your merged_df.csv (or update csv_path)
 - Run: python event_analysis_full_pipeline.py
 - Outputs are saved in the 'event_analysis_results' directory.

Notes:
 - With only ~9 events, p-values will be noisy. Emphasize effect sizes (z, pct)
 - If you have per-epoch data later, repeat analyses at that level for within-event inference.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, linalg
from scipy.stats import median_abs_deviation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ----------------- Config -----------------
out_dir = r"C:\Users\marti\Documents\HSJD\CCV Stefano\Prueba_7_19_11_2025\Results"
os.makedirs(out_dir, exist_ok=True)

# plotting defaults
sns.set(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150})

# number of top features to list per event
TOP_K = 8

# Permutation test config (used if requested in code below)
PERM_N = 2000  # reduce if slow; default 2000


# ----------------- Load data -----------------
df = merged_with_ratios
# Keep original event order and labels
if "event_index" in df.columns:
    # assume event_index already in chronological order; otherwise sort by it
    df = df.sort_values("event_index").reset_index(drop=True)

events = df["event_label"].astype(str).tolist()

# Drop event_index to avoid it polluting numeric analysis (but keep label)
df = df.drop(columns=["event_index"], errors="ignore")

# Select numeric features
num = df.select_dtypes(include=[np.number]).copy()

# Defensive: drop columns that are constant (zero variance) - they can't be correlated
zero_var_cols = num.columns[(num.std(ddof=0) == 0)].tolist()
if zero_var_cols:
    print("Dropping zero-variance columns:", zero_var_cols)
    num = num.drop(columns=zero_var_cols)

# Optionally transform very skewed columns (comment/uncomment as needed)
# skewed_candidates = [c for c in num.columns if (num[c] > 0).all() and (num[c].skew() > 3)]
# for c in skewed_candidates:
#     num[c] = np.log1p(num[c])

# Keep a DataFrame copy for outputs
num_df = num.copy()
num_df.index = events  # set index for nicer output CSVs

# ----------------- Standardization -----------------
# Standard z-score (for visualization & PCA)
zscore = (num - num.mean()) / num.std(ddof=0)
zscore.index = events

# Robust z-score (median / MAD) - uses scipy median_abs_deviation
mad = num.apply(lambda col: median_abs_deviation(col, scale='normal'), axis=0)
robust_z = (num - num.median()) / mad.replace(0, np.nan)
robust_z.index = events

# Save z-score CSVs
zscore.to_csv(os.path.join(out_dir, "zscore_across_events.csv"))
robust_z.to_csv(os.path.join(out_dir, "robust_z_across_events.csv"))

# ----------------- Univariate changes -----------------
# Percent change vs previous event (chronological)
pct_vs_prev = num.pct_change().fillna(0) * 100
pct_vs_prev.index = events
pct_vs_prev.to_csv(os.path.join(out_dir, "percent_change_vs_previous.csv"))

# Absolute change vs previous
abs_vs_prev = num.diff().fillna(0)
abs_vs_prev.index = events
abs_vs_prev.to_csv(os.path.join(out_dir, "absolute_change_vs_previous.csv"))

# Change vs across-event mean (percent)
with np.errstate(divide='ignore', invalid='ignore'):
    pct_vs_mean = (num - num.mean()) / num.mean().replace({0: np.nan}) * 100
pct_vs_mean.index = events
pct_vs_mean.to_csv(os.path.join(out_dir, "percent_change_vs_mean.csv"))

# For each event, top-K variables by absolute percent change vs previous
topk_pct_prev = []
for idx, ev in enumerate(events):
    row = pct_vs_prev.iloc[idx].abs().sort_values(ascending=False).head(TOP_K)
    for rank, (var, val) in enumerate(row.items(), start=1):
        topk_pct_prev.append({"event_label": ev, "rank": rank, "variable": var, "abs_pct_change": val,
                              "signed_pct_change": pct_vs_prev.iloc[idx][var]})
topk_pct_prev_df = pd.DataFrame(topk_pct_prev)
topk_pct_prev_df.to_csv(os.path.join(out_dir, f"top{TOP_K}_pct_change_vs_prev_by_event.csv"), index=False)

# Top-K by absolute robust z
topk_robust_z = []
for idx, ev in enumerate(events):
    row = robust_z.iloc[idx].abs().sort_values(ascending=False).head(TOP_K)
    for rank, (var, val) in enumerate(row.items(), start=1):
        topk_robust_z.append({"event_label": ev, "rank": rank, "variable": var, "abs_robust_z": val,
                              "signed_robust_z": robust_z.iloc[idx][var]})
topk_robust_z_df = pd.DataFrame(topk_robust_z)
topk_robust_z_df.to_csv(os.path.join(out_dir, f"top{TOP_K}_robust_z_by_event.csv"), index=False)

# ----------------- Heatmap of standardized values (chronological rows) -----------------
# Reorder columns to group related features (conn_, Absolute_, Relative_, then the rest)
def grouped_col_order(cols):
    # group keys in desired order, then remaining alphabetical
    groups = ["conn_", "Absolute_", "Relative_"]
    ordered = []
    used = set()
    for g in groups:
        for c in cols:
            if c.startswith(g) and c not in used:
                ordered.append(c); used.add(c)
    # then add the rest preserving original order
    for c in cols:
        if c not in used:
            ordered.append(c); used.add(c)
    return ordered

cols_ordered = grouped_col_order(zscore.columns.tolist())

plt.figure(figsize=(14, 6))
sns.heatmap(zscore[cols_ordered], cmap="vlag", center=0, yticklabels=True, xticklabels=True)
plt.title("Z-scores (standardized) across chronological events")
plt.ylabel("Event")
plt.xlabel("Variables")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "zscore_heatmap.png"))
plt.close()

# ----------------- PCA (dimensionality reduction) -----------------
scaler = StandardScaler()
X = scaler.fit_transform(num)  # rows = events
pca = PCA(n_components=min(6, X.shape[1]))
scores = pca.fit_transform(X)  # event scores in PC space
loadings = pd.DataFrame(pca.components_.T, index=num.columns, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
explained = pca.explained_variance_ratio_

# Save PCA results
pd.DataFrame(scores, index=events, columns=[f"PC{i+1}" for i in range(pca.n_components_)]).to_csv(
    os.path.join(out_dir, "pca_scores.csv"))
loadings.to_csv(os.path.join(out_dir, "pca_loadings.csv"))

# Plot PC1 time series
plt.figure(figsize=(9, 3.5))
plt.plot(events, scores[:, 0], marker="o", linestyle='-')
plt.xticks(rotation=45, ha='right')
plt.title(f"PC1 time series (explained {explained[0]*100:.1f}%)")
plt.ylabel("PC1 score")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pca_pc1_timeseries.png"))
plt.close()

# PC1-PC2 trajectory (with chronological arrows)
if pca.n_components_ >= 2:
    plt.figure(figsize=(6, 6))
    plt.plot(scores[:, 0], scores[:, 1], "-o")
    for i, ev in enumerate(events):
        plt.text(scores[i, 0], scores[i, 1], str(i + 1), fontsize=9)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Event trajectory in PC1-PC2 space")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_trajectory_pc1_pc2.png"))
    plt.close()

# ----------------- Mahalanobis distance (multivariate deviation) -----------------
# Use standardized X for Mahalanobis (scale to unit variance)
X_for_md = X  # already standardized by StandardScaler
cov = np.cov(X_for_md, rowvar=False)
# regularize if near-singular
eps = 1e-6 * np.eye(cov.shape[0])
try:
    cov_inv = linalg.inv(cov)
except np.linalg.LinAlgError:
    cov_inv = linalg.inv(cov + eps)

diff = X_for_md - X_for_md.mean(axis=0)
md2 = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
md = np.sqrt(md2)
mahal_df = pd.DataFrame({"event_label": events, "Mahalanobis": md, "Mahalanobis_sq": md2})
mahal_df.to_csv(os.path.join(out_dir, "mahalanobis_per_event.csv"), index=False)

# Per-variable contributions to D^2 (raw and percent)
# contribution_i = (x_i - mu_i) * sum_j (S_inv[i,j] * (x_j - mu_j))
contrib = (diff @ cov_inv) * diff  # shape (n_events, n_vars)
contrib_df = pd.DataFrame(contrib, columns=num.columns, index=events)
contrib_df.to_csv(os.path.join(out_dir, "mahalanobis_raw_contributions.csv"))
# percent contribution (per event, normalized by md2)
with np.errstate(divide='ignore', invalid='ignore'):
    percent_contrib = contrib / md2[:, None]
percent_contrib_df = pd.DataFrame(np.nan_to_num(percent_contrib), columns=num.columns, index=events)
percent_contrib_df.to_csv(os.path.join(out_dir, "mahalanobis_percent_contrib.csv"))

# Plot Mahalanobis timeline
plt.figure(figsize=(9, 3.5))
plt.bar(events, md, color="C1")
plt.xticks(rotation=45, ha='right')
plt.title("Mahalanobis distance per event (multivariate deviation)")
plt.ylabel("Mahalanobis distance")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "mahalanobis_timeline.png"))
plt.close()

# Save top contributors for top M most-deviant events
M = min(4, len(events))
topM_events = mahal_df.sort_values("Mahalanobis", ascending=False)["event_label"].head(M).tolist()
top_mahal_contrib = []
for ev in topM_events:
    row = percent_contrib_df.loc[ev].abs().sort_values(ascending=False).head(TOP_K)
    for rank, (var, pct) in enumerate(row.items(), start=1):
        top_mahal_contrib.append({"event_label": ev, "rank": rank, "variable": var, "pct_contrib": pct * 100})
pd.DataFrame(top_mahal_contrib).to_csv(os.path.join(out_dir, "top_mahalanobis_contributors.csv"), index=False)

# ----------------- Correlations and Clustering -----------------
# Pearson correlations with p-values (robustly computed pairwise)
def corr_with_pvalues(df_numeric):
    cols = df_numeric.columns.tolist()
    n = len(cols)
    rmat = pd.DataFrame(np.nan, index=cols, columns=cols)
    pmat = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i in range(n):
        for j in range(i, n):
            a = df_numeric.iloc[:, i]
            b = df_numeric.iloc[:, j]
            mask = a.notnull() & b.notnull()
            if mask.sum() < 3:
                r = np.nan; p = np.nan
            else:
                # skip if constant on the masked values
                if a[mask].std(ddof=0) == 0 or b[mask].std(ddof=0) == 0:
                    r = np.nan; p = np.nan
                else:
                    r, p = stats.pearsonr(a[mask], b[mask])
            rmat.iloc[i, j] = r; rmat.iloc[j, i] = r
            pmat.iloc[i, j] = p; pmat.iloc[j, i] = p
    return rmat, pmat

pearson_r, pearson_p = corr_with_pvalues(num_df.reset_index(drop=True))  # use num matrix with events rows
pearson_r.to_csv(os.path.join(out_dir, "pearson_correlation_matrix.csv"))
pearson_p.to_csv(os.path.join(out_dir, "pearson_pvalues_matrix.csv"))

# Spearman correlation (ranks)
spearman_r = num_df.corr(method="spearman")
spearman_r.to_csv(os.path.join(out_dir, "spearman_correlation_matrix.csv"))

# Mask insignificant Pearson correlations (alpha)
alpha = 0.05
mask_insig = (pearson_p > alpha) | pearson_p.isna()

plt.figure(figsize=(12, 10))
sns.heatmap(pearson_r, mask=mask_insig, annot=True, fmt=".2f", cmap="vlag", center=0, linewidths=.5)
plt.title("Pearson correlation (insignificant p>0.05 masked)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pearson_heatmap_masked.png"))
plt.close()

plt.figure(figsize=(12, 10))
sns.heatmap(spearman_r, annot=True, fmt=".2f", cmap="vlag", center=0, linewidths=.5)
plt.title("Spearman correlation across events")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "spearman_heatmap.png"))
plt.close()

# Clustermap of variables by profile across events (transpose: rows = variables)
# Fill any tiny NaNs by column mean (should be none)
data_for_clust = num_df.fillna(num_df.mean()).T
sns.clustermap(data_for_clust, metric='correlation', method='average', cmap="vlag", center=0,
               figsize=(10, 10), xticklabels=events)
plt.suptitle("Clustered variables by profile across events (correlation distance)", y=1.02)
plt.savefig(os.path.join(out_dir, "variables_clustermap.png"))
plt.close()

# Save top correlated pairs
pairs = []
cols = pearson_r.columns.tolist()
ncols = len(cols)
for i in range(ncols):
    for j in range(i+1, ncols):
        r = pearson_r.iloc[i, j]; p = pearson_p.iloc[i, j]
        if (pd.notna(r) and pd.notna(p)):
            pairs.append((cols[i], cols[j], r, p))
pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
top_pairs_df = pd.DataFrame(pairs_sorted, columns=['var1','var2','pearson_r','pval']).head(50)
top_pairs_df.to_csv(os.path.join(out_dir, "top_correlated_pairs.csv"), index=False)

# ----------------- Quick plotting helpers for report -----------------
# Plot top N variables across events (raw values)
def plot_top_vars(var_list, fname, title=None):
    plt.figure(figsize=(10, 4))
    for v in var_list:
        if v in num_df.columns:
            plt.plot(events, num_df[v], marker='o', label=v)
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    if title is None:
        title = "Top variables across events"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()

# Example: plot top 6 variables by variance
top_by_var = num_df.var().sort_values(ascending=False).head(6).index.tolist()
plot_top_vars(top_by_var, "top6_by_variance_timeseries.png", "Top 6 variables by variance across events")

# Example: plot top variables for the most deviant event by Mahalanobis
most_dev = mahal_df.sort_values("Mahalanobis", ascending=False).iloc[0]["event_label"]
top_vars_most_dev = percent_contrib_df.loc[most_dev].abs().sort_values(ascending=False).head(6).index.tolist()
plot_top_vars(top_vars_most_dev, "top_vars_most_deviant_event.png", f"Top vars for event: {most_dev}")

# ----------------- Optional: permutation test for specific candidate changes ---------------
# This is expensive. Use only on a few candidate variables/events.
def perm_test_pct_change(col, event_idx, n_perm=PERM_N):
    """
    Permutation test: is the absolute percent change at event_idx larger than by chance?
    Shuffle event order (preserving values) and recompute percent change; get empirical p.
    """
    observed = abs(pct_vs_prev.iloc[event_idx][col])
    vals = num[col].values.copy()
    count = 0
    for _ in range(n_perm):
        np.random.shuffle(vals)
        perm_pct = np.concatenate([[0], np.diff(vals) / (vals[:-1].astype(float) + 1e-12)]) * 100
        if abs(perm_pct[event_idx]) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)

# Example usage (commented out by default):
# pval_example = perm_test_pct_change('HR(bpm)_mean', 2, n_perm=1000)
# print("Permutation p-value example:", pval_example)

# ----------------- Summary outputs -----------------
print("Analysis complete. Results saved in:", out_dir)
print("Files include: zscore heatmap, PCA outputs, Mahalanobis outputs, correlation matrices, top-K lists.")



# ============================= Median and sd plot per region frequency bands ===========================================


def group_channels_by_region(ch_names):
    """Group EEG channels by anatomical region based on first letter."""
    region_map = {
        'F': 'Frontal',
        'P': 'Parietal',
        'T': 'Temporal',
        'O': 'Occipital',
        'C': 'Central'
    }
    grouped = {region: [] for region in region_map.values()}
    for ch in ch_names:
        if not ch:
            continue
        first = ch[0].upper()
        if first in region_map:
            grouped[region_map[first]].append(ch)
    return grouped


def compute_channel_band_powers(raw, bands, method='welch', fmin=None, fmax=None):
    """
    Compute absolute band power per channel using Simpson integration over PSD.

    Returns:
      per_channel_band_power: dict band -> np.array shape (n_channels,) in μV²
      total_power: np.array shape (n_channels,) total power across fmin..fmax in μV²
      ch_names: list of channel names (same order as returned PSD)
      freqs: frequency vector from PSD object
    """
    if fmin is None:
        fmin = min(b[0] for b in bands.values())
    if fmax is None:
        fmax = max(b[1] for b in bands.values())

    # Use MNE Raw/Epochs compute_psd; adapt if your API differs
    psd_obj = raw.compute_psd(method=method, fmin=fmin, fmax=fmax)
    psd = psd_obj.get_data()    # shape (n_channels, n_freqs), units V^2/Hz
    freqs = psd_obj.freqs
    ch_names = list(raw.ch_names)

    # integrate PSD to get absolute power per channel, convert to μV²
    total_power = simpson(psd, freqs, axis=1) * 1e12  # μV²

    per_channel_band_power = {}
    for band, (bmin, bmax) in bands.items():
        idx_band = np.logical_and(freqs >= bmin, freqs <= bmax)
        if not np.any(idx_band):
            per_channel_band_power[band] = np.full(psd.shape[0], np.nan)
            continue
        band_power = simpson(psd[:, idx_band], freqs[idx_band], axis=1) * 1e12  # μV²
        per_channel_band_power[band] = band_power

    return per_channel_band_power, total_power, ch_names, freqs


def _compute_stats_from_values(vals):
    """Helper: given 1D array of values, return (mean, std(ddof=1), sem) handling NaNs and small n."""
    vals = np.asarray(vals, dtype=float)
    mask = np.isfinite(vals)
    n = int(np.sum(mask))
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.nanmean(vals))
    if n > 1:
        std = float(np.nanstd(vals, ddof=1))
        sem = float(std / sqrt(n))
    else:
        # with a single value we treat std=0 and sem=0 (you may prefer np.nan instead)
        std = 0.0
        sem = 0.0
    return mean, std, sem


def aggregate_region_stats(per_channel_band_power, total_power, ch_names, bands, group_func=group_channels_by_region):
    """
    Aggregate per-channel band powers by anatomical region and compute mean/std/sem.

    Returns:
      region_band_stats: dict region -> band -> stats dict with keys:
         'values'   : np.array of per-channel absolute band power (μV²) for the region
         'mean_abs' : float mean absolute power (μV²)
         'std_abs'  : float sample std (μV²)
         'sem_abs'  : float sem (μV²)
         'mean_rel' : float mean relative power (%)  (if total_power provided)
         'std_rel'  : float std of relative (%)     (if total_power provided)
         'sem_rel'  : float sem of relative (%)     (if total_power provided)
    """
    ch_regions = group_func(ch_names)
    region_band_stats = defaultdict(dict)

    # map channel name -> index
    name_to_idx = {name: idx for idx, name in enumerate(ch_names)}

    for region, ch_list in ch_regions.items():
        # convert channel names to indices; ignore missing names
        ch_idxs = [name_to_idx[ch] for ch in ch_list if ch in name_to_idx]
        for band in bands:
            if not ch_idxs:
                region_band_stats[region][band] = {
                    'values': np.array([], dtype=float),
                    'mean_abs': np.nan, 'std_abs': np.nan, 'sem_abs': np.nan,
                    'mean_rel': np.nan, 'std_rel': np.nan, 'sem_rel': np.nan
                }
                continue
            vals = per_channel_band_power[band][ch_idxs]
            mean_abs, std_abs, sem_abs = _compute_stats_from_values(vals)

            # compute relative per-channel if total_power available
            if total_power is not None:
                total_vals = total_power[ch_idxs]
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel_vals = np.where(np.isfinite(total_vals) & (total_vals != 0),
                                        (vals / total_vals) * 100.0,
                                        np.nan)
                mean_rel, std_rel, sem_rel = _compute_stats_from_values(rel_vals)
            else:
                rel_vals = np.full_like(vals, np.nan)
                mean_rel = std_rel = sem_rel = np.nan

            region_band_stats[region][band] = {
                'values': np.asarray(vals, dtype=float),
                'mean_abs': mean_abs,
                'std_abs': std_abs,
                'sem_abs': sem_abs,
                'mean_rel': mean_rel,
                'std_rel': std_rel,
                'sem_rel': sem_rel
            }

    return region_band_stats


def compute_and_group_events(events, bands, group_func=group_channels_by_region, psd_method='welch'):
    """
    Process a list of events and produce per_event_band_data plus a summary DataFrame.

    Args:
      events: list of dicts with keys:
          - 'raw': MNE Raw/Epochs object
          - 'description' (optional): string label for the event/group
      bands: dict of band definitions
    Returns:
      df: pandas.DataFrame with columns:
          Event, Region, Band, mean_abs, std_abs, sem_abs, mean_rel_%, std_rel_%, sem_rel_%
      per_event_band_data: nested dict event -> region -> band -> stats (same as aggregate_region_stats)
    """
    results = []
    per_event_band_data = {}

    for i, ev in enumerate(events):
        raw = ev['raw']
        desc = ev.get('description', f'event_{i}')
        # compute per-channel band powers
        per_channel_band_power, total_power, ch_names, freqs = compute_channel_band_powers(raw, bands, method=psd_method)
        # aggregate by region
        region_stats = aggregate_region_stats(per_channel_band_power, total_power, ch_names, bands, group_func=group_func)
        per_event_band_data[desc] = region_stats

        # build rows for dataframe
        for region, band_dict in region_stats.items():
            for band, d in band_dict.items():
                results.append({
                    'Event': desc,
                    'Region': region,
                    'Band': band,
                    'mean_abs': d['mean_abs'],
                    'std_abs': d['std_abs'],
                    'sem_abs': d['sem_abs'],
                    'mean_rel_%': d['mean_rel'],
                    'std_rel_%': d['std_rel'],
                    'sem_rel_%': d['sem_rel']
                })

    df = pd.DataFrame(results)
    col_order = ['Event', 'Region', 'Band', 'mean_abs', 'std_abs', 'sem_abs', 'mean_rel_%', 'std_rel_%', 'sem_rel_%']
    df = df[col_order]
    return df, per_event_band_data


def plot_band_powers(per_event_band_data, bands, metric='absolute',
                     region_order=None, one_row=True, figsize=None,
                     cmap='tab10', marker_cycle=None, savepath=None):
    """
    Plot mean ± SEM per-event, per-region for each band.

    - per_event_band_data: nested dict event -> region -> band -> stats (with 'mean_abs' and 'sem_abs' present)
    - bands: dict of band definitions
    - metric: 'absolute' or 'relative' (plots mean_abs/sem_abs or mean_rel/sem_rel in %)
    - one_row: if True, put all bands in one row (recommended)
    - figsize: override default figure size
    """
    events = list(per_event_band_data.keys())
    if len(events) == 0:
        raise ValueError("per_event_band_data is empty.")
    n_events = len(events)
    band_names = list(bands.keys())

    sample_event = events[0]
    regions = list(per_event_band_data[sample_event].keys())
    if region_order:
        regions = [r for r in region_order if r in regions]
    else:
        preferred = ['Total', 'Frontal', 'Central', 'Parietal', 'Temporal', 'Occipital']
        regions = [r for r in preferred if r in regions] + [r for r in regions if r not in preferred]

    n_regions = len(regions)

    # Layout decisions
    if one_row:
        n_cols = len(band_names)
        n_rows = 1
        if figsize is None:
            figsize = (26, 4.5)
    else:
        n_cols = min(3, len(band_names))
        n_rows = int(np.ceil(len(band_names) / n_cols))
        if figsize is None:
            figsize = (14, 10)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    cmap = plt.get_cmap(cmap)
    colors = [cmap(i % cmap.N) for i in range(n_events)]
    if marker_cycle is None:
        marker_cycle = ['o', 's', 'D', '^', 'v', 'P', 'X', '>', '<']
    markers = [marker_cycle[i % len(marker_cycle)] for i in range(n_events)]

    x = np.arange(n_regions)
    width = 0.6
    offsets = np.linspace(-width/2, width/2, n_events) if n_events > 1 else [0]
    marker_size = 6 if n_events <= 6 else 5
    capsize = 3

    for b_i, band in enumerate(band_names):
        ax = axes[b_i]
        for e_i, event in enumerate(events):
            means = []
            sems = []
            for region in regions:
                d = per_event_band_data[event].get(region, {}).get(band, None)
                if d is None:
                    means.append(np.nan)
                    sems.append(np.nan)
                else:
                    if metric == 'absolute':
                        means.append(d.get('mean_abs', np.nan))
                        sems.append(d.get('sem_abs', np.nan))
                    else:
                        # relative metric in percent
                        means.append(d.get('mean_rel', np.nan))
                        sems.append(d.get('sem_rel', np.nan))
            means = np.array(means, dtype=float)
            sems = np.array(sems, dtype=float)

            xi = x + offsets[e_i]
            ax.errorbar(xi, means, yerr=sems, fmt=markers[e_i], markersize=marker_size,
                        label=event if b_i == 0 else None,
                        color=colors[e_i], capsize=capsize, linestyle='None', elinewidth=1, markeredgewidth=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=25, ha='right', fontsize=9)
        ylabel = "Absolute power (μV²)" if metric == 'absolute' else "Relative power (%)"
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(band_names[b_i], fontsize=12)
        ax.grid(axis='y', linestyle=':', alpha=0.35)

        # Compute sensible y-limits with small padding
        y_vals = []
        for event in events:
            for region in regions:
                d = per_event_band_data[event].get(region, {}).get(band_names[b_i], None)
                if d:
                    if metric == 'absolute':
                        m = d.get('mean_abs', np.nan)
                        s = d.get('sem_abs', np.nan)
                    else:
                        m = d.get('mean_rel', np.nan)
                        s = d.get('sem_rel', np.nan)
                    y_vals.extend([m, m + (s if np.isfinite(s) else 0), m - (s if np.isfinite(s) else 0)])
        if y_vals:
            ymin = np.nanmin(y_vals)
            ymax = np.nanmax(y_vals)
            if np.isfinite(ymin) and np.isfinite(ymax) and (ymax - ymin) > 0:
                yrange = ymax - ymin
                ax.set_ylim([max(0, ymin - 0.12 * yrange), ymax + 0.12 * yrange])

        # Thin spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    # remove any unused axes
    for j in range(len(band_names), n_rows * n_cols):
        fig.delaxes(axes[j])

    # global legend above
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        ncol_legend = min(9, n_events)
        fig.legend(handles, labels, loc='upper center', ncol=ncol_legend,
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.99))

    plt.subplots_adjust(top=0.88 if not one_row else 0.82, left=0.05, right=0.98,
                        wspace=0.35 if one_row else 0.3, hspace=0.45 if not one_row else 0.3)

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Example run: compute and plot if 'events' exists in environment.
    # Replace this with your actual events list (MNE Raw/Epochs).
    try:
        events  # noqa: F821
    except NameError:
        print("No 'events' variable found in the environment. Create a list `events` and re-run:")
        print("events = [{'raw': raw0, 'description': 'event0'}, {'raw': raw1, 'description': 'event1'}, ...]")
    else:
        print("Computing band powers and aggregating by region for events...")
        df, per_event_band_data = compute_and_group_events(events, bands)
        print("Summary table:")
        pd.set_option('display.max_rows', 200)
        print(df.to_string(index=False))
        print("Plotting...")
        plot_band_powers(per_event_band_data, bands=bands, metric='absolute', one_row=True,
                         figsize=(26, 4.5), savepath='band_wide.png')






# ============== Statistical Significance evaluation =================

