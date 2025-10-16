# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 10:26:20 2025

@author: msedo
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, time
import mne

# --- Helper function to convert seconds to real datetimes
def sec_to_datetime(seconds, eeg_start_datetime):
    return [eeg_start_datetime + pd.Timedelta(seconds=s) for s in seconds]

# --- Load Data ---
event_df = pd.read_excel(r"C:\Users\msedo\Documents\CCV Stefano\Prueba 4\CCV 004 Etapas Cirugía CCV Stefano.xlsx")
param_df = pd.read_csv(r"C:\Users\msedo\Documents\CCV Stefano\Prueba 4\CMSViewer\ParameterData.csv")
param_df.columns = param_df.columns.str.strip()

edf_path = r"C:\Users\msedo\Documents\CCV Stefano\Prueba 4\CCV_004_100920205_1.edf"
raw = mne.io.read_raw_edf(edf_path, preload=True)

# --- Keep only first 20 EEG channels, rename ---
first_20_chs = raw.ch_names[:20]
raw.pick(first_20_chs)
new_names = [
    'P8', 'T8', 'F8', 'F4', 'C4', 'P4',
    'EOG2 (FP2)', 'EOG1 (FP1)',
    'Fz', 'Cz', 'O1', 'Oz', 'O2',
    'Pz', 'P3', 'C3', 'F3', 'F7', 'T7', 'P7'
]
rename_dict = dict(zip(raw.ch_names[:20], new_names))
raw.rename_channels(rename_dict)

# --- Get recording start time (datetime) ---
meas_date = raw.info['meas_date']
if isinstance(meas_date, (tuple, list)):
    meas_date = datetime.fromtimestamp(meas_date[0], tz=timezone.utc)
elif hasattr(meas_date, 'to_pydatetime'):
    meas_date = meas_date.to_pydatetime()
elif meas_date is None:
    raise ValueError("meas_date is missing from the EDF file.")
meas_date = meas_date.replace(tzinfo=None)

# --- Prepare event onsets and descriptions ---
onset_list = []
description_list = []
for idx, row in event_df.iterrows():
    event = row['Evento']
    time_obj = row['Hora 1']
    if not isinstance(time_obj, time):
        time_obj = pd.to_datetime(time_obj).time()
    event_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + getattr(time_obj, 'second', 0)
    start_seconds = meas_date.hour * 3600 + meas_date.minute * 60 + meas_date.second
    onset = event_seconds - start_seconds
    onset_list.append(onset)
    description_list.append(event)

# --- Calculate durations for annotations ---
duration_list = []
for i in range(len(onset_list)):
    if i < len(onset_list)-1:
        duration = onset_list[i+1] - onset_list[i]
    else:
        duration = 300
    duration_list.append(duration)

# --- Add Annotations to EEG ---
annotations = mne.Annotations(onset=onset_list,
                              duration=duration_list,
                              description=description_list)
raw.set_annotations(annotations)

# ------------------ Cropping relevant time window ----------------------

# --- Crop window based on events ---
crop_start_sec = onset_list[0]
crop_end_sec = onset_list[-1] + duration_list[-1]

# Crop EEG
raw.crop(tmin=crop_start_sec, tmax=crop_end_sec)

# Crop physiological data
param_df = param_df[(param_df['Time_sec'] >= crop_start_sec) & (param_df['Time_sec'] <= crop_end_sec)].reset_index(drop=True)

# Update event times (relative to new cropped data)
# After cropping, the new meas_date is the same, but raw.times[0] = crop_start_sec
# So, for plotting, subtract crop_start_sec from all event/EEG/physio times
event_times_sec = np.array(onset_list) - crop_start_sec
event_labels = description_list
param_df['Time_sec'] = param_df['Time_sec'] - crop_start_sec

# Prepare EEG data again (now cropped)
eeg_data, eeg_times = raw[:, :]
eeg_data = eeg_data * 1e6  # µV
eeg_times = eeg_times - eeg_times[0]  # Now starts at 0

# For plotting in real time, adjust base datetime:
new_meas_date = meas_date + pd.Timedelta(seconds=crop_start_sec)

eeg_times_dt = sec_to_datetime(eeg_times, new_meas_date)
physio_times_dt = sec_to_datetime(param_df['Time_sec'], new_meas_date)
event_times_dt = sec_to_datetime(event_times_sec, new_meas_date)


# --- Settings for plotting ---
eeg_offset = 200      # µV offset for each EEG channel
physio_offset = 30    # offset for each physiological signal
max_eeg_chs = 3       # Number of EEG channels to plot
measures = ['HR(bpm)', 'SpO2(%)', 'PI(%)', 'RR(rpm)', 'PR(bpm)']

# --- Plot EEG and Physiology together with real time x-axis ---
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# Plot EEG
num_eeg_chs = eeg_data.shape[0]
for ch in range(num_eeg_chs):
    axes[0].plot(eeg_times_dt, eeg_data[ch] + ch*eeg_offset, label=f'EEG {raw.ch_names[ch]}')
axes[0].set_ylabel('EEG (µV)')
axes[0].legend(loc='upper right')
axes[0].set_title('EEG and Physiological Signals with Event Markers')

# Plot Physiological Signals, offset
for i, measure in enumerate(measures):
    y = pd.to_numeric(param_df[measure].replace('--', np.nan), errors='coerce')
    axes[1].plot(physio_times_dt, y + i*physio_offset, label=measure)
axes[1].set_yticks([i*physio_offset for i in range(len(measures))])
axes[1].set_yticklabels(measures)
axes[1].set_ylabel('Physiological\nsignals (offset)')
axes[1].legend(loc='upper right')

# Add event lines to both subplots
for ax in axes:
    for ev_dt, ev_label in zip(event_times_dt, event_labels):
        ax.axvline(ev_dt, color='red', linestyle='--', alpha=0.6)
        ax.text(ev_dt, ax.get_ylim()[1], ev_label, rotation=90, color='red',
                va='top', ha='right', fontsize=7)

# Format x-axis as time (HH:MM)
axes[-1].set_xlabel('Time (HH:MM)')
time_fmt = mdates.DateFormatter('%H:%M')
axes[-1].xaxis.set_major_formatter(time_fmt)
fig.autofmt_xdate()

plt.tight_layout()
plt.show()



'''
# ====================== Physiological measures subplots =======================

# Assume param_df['Time'] is already datetime
param_times = param_df['Time']

# If event_df['Hora 1'] is not datetime.time, convert it:
if not isinstance(event_df['Hora 1'].iloc[0], time):
    event_df['Hora 1'] = pd.to_datetime(event_df['Hora 1']).dt.time

event_times = []
for t in event_df['Hora 1']:
    # Find the parameter datetime closest in time-of-day to the event time
    candidates = param_times[param_times.dt.time == t]
    if not candidates.empty:
        event_times.append(candidates.iloc[0])
    else:
        # If exact time not found, find the closest
        abs_diffs = abs(param_times.dt.hour*60 + param_times.dt.minute - (t.hour*60 + t.minute))
        closest_idx = abs_diffs.idxmin()
        event_times.append(param_times.loc[closest_idx])

event_labels = event_df['Evento'].tolist()

# Create subplots
fig, axes = plt.subplots(len(measures), 1, figsize=(14, 3 * len(measures)), sharex=True)

# Plot all measures
for i, measure in enumerate(measures):
    ax = axes[i]
    y = pd.to_numeric(param_df[measure].replace('--', pd.NA), errors='coerce')
    ax.plot(param_df['Time'], y, label=measure, color='tab:blue')
    ax.set_ylabel(measure)
    ax.legend(loc='upper right')
    ax.grid(True)

# Only draw event lines on the bottom axis for clarity
bottom_ax = axes[-1]
for ev_time, ev_label in zip(event_times, event_labels):
    # Draw event lines close together (short line, small font)
    bottom_ax.axvline(ev_time, color='red', linestyle='--', alpha=0.7, ymin=0.90, ymax=1.0, linewidth=1.5)
    # Position labels tightly above x-axis, rotate for compactness
    bottom_ax.text(ev_time, bottom_ax.get_ylim()[0], ev_label, rotation=90, color='red',
                   va='bottom', ha='center', fontsize=9, clip_on=True, fontweight='bold')

axes[0].set_title("Physiological Measures Over Time with Shared Event Markers")
bottom_ax.set_xlabel("Time")
plt.tight_layout()
plt.show()
'''