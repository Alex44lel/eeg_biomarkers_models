"""
==============================================================================
Exploring the three .mat files in data/recordings/S01AS/DMT/
==============================================================================

WHAT IS A .MAT FILE?
--------------------
A .mat file is MATLAB's native binary format for saving variables (arrays,
structs, etc.). Python can read them using:
  - scipy.io.loadmat()  -> for MATLAB v5/v7 files
  - h5py                -> for MATLAB v7.3 files (HDF5-based)

Our three files are all v5 format, so we use scipy.io.loadmat().

THE THREE FILES AT A GLANCE:
----------------------------
1. data_ref.mat         -> The RAW continuous EEG recording, split into
                           variable-length segments (0.1s to 56s each).
                           106 segments, 36 EEG channels, 1000 Hz sampling.

2. data_trials3s.mat    -> The recording re-cut into FIXED 3-second trials,
                           grouped into 4 CONDITIONS (e.g. baseline, peak, etc.).
                           Each trial = 36 channels x 3000 timepoints.

3. data_trialsmxm_3s.mat -> Same 3-second trial format, but split into 21
                            FINER segments (minute-by-minute or similar),
                            giving higher temporal resolution of how the
                            brain state evolves over time.

Think of it as: raw tape -> 4 big bins -> 21 small bins.
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================================
# LOADING .MAT FILES
# ============================================================================
# squeeze_me=True removes unnecessary single-element dimensions,
# making the data easier to work with in Python.

print("Loading .mat files (this may take a moment, they are large)...")
mat_ref = sio.loadmat(
    'data/recordings/S01AS/DMT/data_ref.mat', squeeze_me=True
)
mat_trials = sio.loadmat(
    'data/recordings/S01AS/DMT/data_trials3s.mat', squeeze_me=True
)
mat_mxm = sio.loadmat(
    'data/recordings/S01AS/DMT/data_trialsmxm_3s.mat', squeeze_me=True
)
print("Done!\n")


# ============================================================================
# FILE 1: data_ref.mat  —  The raw continuous recording
# ============================================================================
print("=" * 70)
print("FILE 1: data_ref.mat  —  Raw continuous EEG recording")
print("=" * 70)

# The file contains a single MATLAB struct stored under the key 'data_ref'.
# In Python, scipy loads MATLAB structs as numpy structured arrays.
# Access fields with: data['field_name'].item()
data_ref = mat_ref['data_ref']

# Show all available fields in the struct
print(f"\nFields in data_ref struct: {data_ref.dtype.names}")
print("""
  hdr        -> Header info (recording device metadata)
  fsample    -> Sampling frequency in Hz
  sampleinfo -> Start/end sample indices for each segment
  trial      -> Array of EEG data segments (each is channels x timepoints)
  time       -> Array of time vectors (one per segment, in seconds)
  label      -> Channel names (electrode positions on the scalp)
  cfg        -> Configuration/processing history from MATLAB FieldTrip
""")

fsample = data_ref['fsample'].item()
labels = data_ref['label'].item()
trials_ref = data_ref['trial'].item()      # array of 106 segments
times_ref = data_ref['time'].item()         # corresponding time vectors
sampleinfo = data_ref['sampleinfo'].item()  # [start_sample, end_sample] per segment

print(f"Sampling rate: {fsample} Hz  (= {fsample} data points per second)")
print(f"Number of EEG channels: {len(labels)}")
print(f"Channel names: {list(labels)}")
print(f"\nNumber of segments: {len(trials_ref)}")

# Compute duration of each segment
durations = np.array([times_ref[i][-1] - times_ref[i][0] for i in range(len(trials_ref))])
print(f"Segment durations: min={durations.min():.2f}s, max={durations.max():.2f}s, "
      f"mean={durations.mean():.2f}s")
print(f"Total recording time: {durations.sum():.1f}s = {durations.sum()/60:.1f} minutes")
print(f"\nKey point: segments have VARIABLE lengths (not uniform trials).")
print(f"This is the raw data before it was cut into uniform analysis windows.\n")


# ============================================================================
# FILE 2: data_trials3s.mat  —  Fixed 3-second trials, 4 conditions
# ============================================================================
print("=" * 70)
print("FILE 2: data_trials3s.mat  —  3-second trials in 4 conditions")
print("=" * 70)

data_trials = mat_trials['data_trials3s']  # shape (4,) — 4 conditions
print(f"\nNumber of conditions: {data_trials.shape[0]}")
print("Each condition is a FieldTrip struct with the same fields as data_ref.\n")

for i in range(data_trials.shape[0]):
    cond = data_trials[i]
    trial_data = cond['trial'].item()
    n_trials = len(trial_data)
    shape = trial_data[0].shape  # (channels, timepoints)
    duration = shape[1] / fsample
    print(f"  Condition {i}: {n_trials} trials, "
          f"each {shape[0]} channels x {shape[1]} samples = {duration:.1f}s")

print(f"""
Each trial is exactly 3 seconds (3000 samples at 1000 Hz).
The 4 conditions likely correspond to experimental phases, e.g.:
  0 = Pre-injection baseline
  1 = Onset / early effects
  2 = Peak effects
  3 = Post-peak / comedown
(Exact labels depend on the experimental protocol.)

Total trials across conditions: {sum(len(data_trials[i]['trial'].item()) for i in range(4))}
""")


# ============================================================================
# FILE 3: data_trialsmxm_3s.mat  —  3-second trials, 21 fine segments
# ============================================================================
print("=" * 70)
print("FILE 3: data_trialsmxm_3s.mat  —  3-second trials, 21 segments")
print("=" * 70)

data_mxm = mat_mxm['data_trialsmxm_3s']  # shape (21,) — 21 segments
print(f"\nNumber of segments: {data_mxm.shape[0]}")
print("Same trial format (36 ch x 3000 samples), but now split into 21 bins.\n")

seg_counts = []
for i in range(data_mxm.shape[0]):
    seg = data_mxm[i]
    trial_data = seg['trial'].item()
    n_trials = len(trial_data)
    seg_counts.append(n_trials)
    print(f"  Segment {i:2d}: {n_trials:3d} trials")

print(f"\nTotal trials: {sum(seg_counts)}")
print("""
This gives minute-by-minute (or similar) temporal resolution.
Segment 0 has the most trials (85) — likely the baseline period.
The remaining 20 segments have 9-18 trials each — the drug period
split into fine time bins to track how brain activity evolves.
""")


# ============================================================================
# SUMMARY COMPARISON TABLE
# ============================================================================
print("=" * 70)
print("SUMMARY: How the three files relate")
print("=" * 70)
print("""
 File                 | Segments | Trials/seg  | Trial length | Purpose
 ---------------------|----------|-------------|--------------|------------------
 data_ref             | 106      | 1 each      | VARIABLE     | Raw recording
 data_trials3s        | 4        | 81-95 each  | 3.0 seconds  | Coarse conditions
 data_trialsmxm_3s    | 21       | 9-85 each   | 3.0 seconds  | Fine time bins

All files: 36 EEG channels, 1000 Hz sampling rate.
Same underlying recording, just organized differently.
""")


# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("Generating visualizations...")

fig = plt.figure(figsize=(18, 22))
fig.suptitle('Comparison of the three .mat files — Subject S01AS / DMT',
             fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(4, 2, hspace=0.45, wspace=0.3,
                       top=0.94, bottom=0.04, left=0.08, right=0.95)

# --- Plot 1: data_ref segment durations ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(range(len(durations)), durations, color='steelblue', alpha=0.8)
ax1.set_xlabel('Segment index')
ax1.set_ylabel('Duration (seconds)')
ax1.set_title('data_ref: Variable-length segment durations')
ax1.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, label='3s reference')
ax1.legend()

# --- Plot 2: Trial counts across the 3 files ---
ax2 = fig.add_subplot(gs[0, 1])
categories = ['data_ref\n(106 segs)', 'data_trials3s\n(4 conds)', 'data_trialsmxm\n(21 segs)']
totals = [106, sum(len(data_trials[i]['trial'].item()) for i in range(4)), sum(seg_counts)]
bars = ax2.bar(categories, totals, color=['steelblue', 'coral', 'seagreen'], alpha=0.8)
for bar, val in zip(bars, totals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(val), ha='center', fontweight='bold')
ax2.set_ylabel('Total number of trials/segments')
ax2.set_title('Total trial counts per file')

# --- Plot 3: Raw EEG from data_ref (first segment, first 5 channels) ---
ax3 = fig.add_subplot(gs[1, :])
seg_idx = 5  # pick a medium-length segment
raw_data = trials_ref[seg_idx]
raw_time = times_ref[seg_idx]
n_ch_show = 6
offsets = np.arange(n_ch_show) * 150  # vertical spacing in microvolts
for ch in range(n_ch_show):
    ax3.plot(raw_time, raw_data[ch, :] + offsets[ch],
             linewidth=0.5, label=labels[ch])
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Amplitude (uV, offset for clarity)')
ax3.set_title(f'data_ref: Raw EEG — segment {seg_idx} '
              f'({raw_data.shape[1]/fsample:.1f}s, variable length)')
ax3.legend(loc='upper right', fontsize=8, ncol=2)
ax3.set_xlim(raw_time[0], raw_time[-1])

# --- Plot 4: 3s trials from data_trials3s (condition 0 vs 2) ---
ax4 = fig.add_subplot(gs[2, 0])
cond0_trials = data_trials[0]['trial'].item()
trial_example = cond0_trials[0]  # first trial of condition 0
time_3s = np.arange(trial_example.shape[1]) / fsample
ch_idx = 0  # Fp1
# plot a few trials overlaid for condition 0
for t_idx in range(min(10, len(cond0_trials))):
    ax4.plot(time_3s, cond0_trials[t_idx][ch_idx, :],
             alpha=0.3, color='steelblue', linewidth=0.5)
ax4.set_xlabel('Time (seconds)')
ax4.set_ylabel(f'Amplitude (uV) — channel {labels[ch_idx]}')
ax4.set_title(f'data_trials3s: Condition 0 — 10 trials overlaid ({labels[ch_idx]})')
ax4.set_xlim(0, 3)

ax5 = fig.add_subplot(gs[2, 1])
cond2_trials = data_trials[2]['trial'].item()
for t_idx in range(min(10, len(cond2_trials))):
    ax5.plot(time_3s, cond2_trials[t_idx][ch_idx, :],
             alpha=0.3, color='coral', linewidth=0.5)
ax5.set_xlabel('Time (seconds)')
ax5.set_ylabel(f'Amplitude (uV) — channel {labels[ch_idx]}')
ax5.set_title(f'data_trials3s: Condition 2 — 10 trials overlaid ({labels[ch_idx]})')
ax5.set_xlim(0, 3)

# --- Plot 5: data_trialsmxm_3s — trial count per segment (timeline) ---
ax6 = fig.add_subplot(gs[3, 0])
colors = ['steelblue'] + ['seagreen'] * 20
ax6.bar(range(21), seg_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax6.set_xlabel('Segment index (time progression →)')
ax6.set_ylabel('Number of 3s trials')
ax6.set_title('data_trialsmxm_3s: Trials per segment over time')
ax6.annotate('Baseline\n(pre-drug)', xy=(0, seg_counts[0]),
             xytext=(3, seg_counts[0] + 5),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=9, ha='center')

# --- Plot 6: Mean amplitude (power) evolution across 21 segments ---
ax7 = fig.add_subplot(gs[3, 1])
mean_power = []
for i in range(21):
    seg_trials = data_mxm[i]['trial'].item()
    # compute mean absolute amplitude across all trials and channels
    all_data = np.concatenate([seg_trials[t] for t in range(len(seg_trials))], axis=1)
    mean_power.append(np.mean(np.abs(all_data)))

ax7.plot(range(21), mean_power, 'o-', color='darkgreen', markersize=6)
ax7.fill_between(range(21), mean_power, alpha=0.2, color='seagreen')
ax7.set_xlabel('Segment index (time progression →)')
ax7.set_ylabel('Mean |amplitude| (uV)')
ax7.set_title('data_trialsmxm_3s: Brain activity evolution over time')
ax7.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Drug administration')
ax7.legend()

plt.savefig('mat_files_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'mat_files_comparison.png'")
print("\nDone! Check the figure window for the visualizations.")
