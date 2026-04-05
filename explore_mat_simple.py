"""
Simple visual comparison: what does the EEG signal look like in each .mat file?

All three files contain the SAME brain recording from ONE person.
The only difference is HOW the recording was chopped up.

Think of it like cutting a long ribbon:
  - File 1: irregular cuts (different sizes)
  - File 2: uniform 3s cuts, sorted into 4 piles
  - File 3: uniform 3s cuts, sorted into 21 piles
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

print("Loading files (this takes ~30s, they are big)...")
mat_ref = sio.loadmat('data/recordings/S01AS/DMT/data_ref.mat', squeeze_me=True)
mat_trials = sio.loadmat('data/recordings/S01AS/DMT/data_trials3s.mat', squeeze_me=True)
mat_mxm = sio.loadmat('data/recordings/S01AS/DMT/data_trialsmxm_3s.mat', squeeze_me=True)
print("Done!\n")

data_ref = mat_ref['data_ref']
data_trials = mat_trials['data_trials3s']
data_mxm = mat_mxm['data_trialsmxm_3s']

# We'll always plot channel 0 ("Fp1", a forehead electrode)
CH = 0
ch_name = data_ref['label'].item()[CH]

# ==========================================================================
# FIGURE 1: The structure of each file — what's inside?
# ==========================================================================
fig, axes = plt.subplots(3, 1, figsize=(16, 14))
fig.suptitle(f'What does the signal look like in each file? (channel: {ch_name})',
             fontsize=15, fontweight='bold')

# --- FILE 1: data_ref ---
ax = axes[0]
ax.set_title('FILE 1: data_ref.mat — The raw recording, cut into IRREGULAR pieces',
             fontsize=13, fontweight='bold', color='steelblue')

trials_ref = data_ref['trial'].item()
times_ref = data_ref['time'].item()

# Plot the first 8 pieces side by side with gaps between them
colors = plt.cm.tab10(np.linspace(0, 1, 10))
x_offset = 0
for i in range(8):
    signal = trials_ref[i][CH, :]
    t = np.arange(len(signal)) / 1000.0 + x_offset
    ax.plot(t, signal, color=colors[i % 10], linewidth=0.5)
    duration = len(signal) / 1000.0
    ax.axvline(x=x_offset, color='gray', linestyle='--', alpha=0.3)
    ax.text(x_offset + duration / 2, ax.get_ylim()[0] if i > 0 else -25,
            f'piece {i}\n{duration:.1f}s', ha='center', fontsize=8,
            color=colors[i % 10], fontweight='bold')
    x_offset += duration + 0.5  # small gap between pieces

ax.set_xlabel('Time (seconds) — gaps are artificial, just to separate pieces')
ax.set_ylabel('Amplitude (µV)')
ax.text(0.98, 0.95,
        f'Total: 106 pieces\nEach piece = different length\n(0.1s to 56s)',
        transform=ax.transAxes, fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# --- FILE 2: data_trials3s ---
ax = axes[1]
ax.set_title('FILE 2: data_trials3s.mat — Same recording, re-cut into 3-second pieces, '
             'sorted into 4 GROUPS',
             fontsize=13, fontweight='bold', color='coral')

group_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
group_labels = ['Group 0', 'Group 1', 'Group 2', 'Group 3']
x_offset = 0

for g in range(4):
    group_trials = data_trials[g]['trial'].item()
    n = len(group_trials)

    # Plot first 3 pieces from each group
    for j in range(min(3, n)):
        signal = group_trials[j][CH, :]
        t = np.arange(3000) / 1000.0 + x_offset
        ax.plot(t, signal, color=group_colors[g], linewidth=0.5, alpha=0.8)
        x_offset += 3.3  # 3s + small gap

    # Label and bracket
    bracket_start = x_offset - 3.3 * min(3, n)
    bracket_end = x_offset - 0.3
    ax.annotate('', xy=(bracket_start, 28), xytext=(bracket_end, 28),
                arrowprops=dict(arrowstyle='|-|', color=group_colors[g], lw=2))
    ax.text((bracket_start + bracket_end) / 2, 32,
            f'{group_labels[g]}\n({n} pieces)',
            ha='center', fontsize=9, color=group_colors[g], fontweight='bold')

    x_offset += 2  # gap between groups

ax.set_xlabel('Each piece is exactly 3 seconds — gaps separate pieces and groups')
ax.set_ylabel('Amplitude (µV)')
ax.text(0.98, 0.95,
        'Total: 342 pieces\nAll pieces = exactly 3s\nSorted into 4 groups',
        transform=ax.transAxes, fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# --- FILE 3: data_trialsmxm_3s ---
ax = axes[2]
ax.set_title('FILE 3: data_trialsmxm_3s.mat — Same recording, re-cut into 3-second pieces, '
             'sorted into 21 GROUPS',
             fontsize=13, fontweight='bold', color='seagreen')

x_offset = 0
cmap = plt.cm.viridis(np.linspace(0, 0.9, 21))

for g in range(21):
    group_trials = data_mxm[g]['trial'].item()
    n = len(group_trials)

    # Plot first 2 pieces from each group
    for j in range(min(2, n)):
        signal = group_trials[j][CH, :]
        t = np.arange(3000) / 1000.0 + x_offset
        ax.plot(t, signal, color=cmap[g], linewidth=0.5, alpha=0.8)
        x_offset += 3.2

    # Label
    label_x = x_offset - 3.2 * min(2, n) / 2
    ax.text(label_x, 30, f'{g}', ha='center', fontsize=7,
            color=cmap[g], fontweight='bold')
    x_offset += 1  # gap between groups

ax.set_xlabel('Each piece is exactly 3 seconds — numbers = group index (0 to 20)')
ax.set_ylabel('Amplitude (µV)')
ax.text(0.98, 0.95,
        'Total: 375 pieces\nAll pieces = exactly 3s\nSorted into 21 groups\n(finer time resolution)',
        transform=ax.transAxes, fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('mat_files_simple.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved as mat_files_simple.png")


# ==========================================================================
# FIGURE 2: Zoom into ONE piece from each file — they look the same!
# ==========================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
fig.suptitle(f'Zooming into ONE piece from each file (channel: {ch_name})\n'
             f'They all look the same — it\'s the same type of brain signal!',
             fontsize=14, fontweight='bold')

# File 1: one piece (pick a ~3s one for fair comparison)
# Find a piece close to 3s
durations = [trials_ref[i].shape[1] / 1000.0 for i in range(len(trials_ref))]
idx_3s = min(range(len(durations)), key=lambda i: abs(durations[i] - 3.0))
signal1 = trials_ref[idx_3s][CH, :]
t1 = np.arange(len(signal1)) / 1000.0
axes[0].plot(t1, signal1, color='steelblue', linewidth=0.7)
axes[0].set_title(f'data_ref — piece #{idx_3s} (duration: {durations[idx_3s]:.2f}s)',
                  fontweight='bold', color='steelblue')
axes[0].set_ylabel('µV')

# File 2: one piece from group 0
signal2 = data_trials[0]['trial'].item()[0][CH, :]
t2 = np.arange(3000) / 1000.0
axes[1].plot(t2, signal2, color='coral', linewidth=0.7)
axes[1].set_title('data_trials3s — group 0, piece #0 (duration: 3.00s)',
                  fontweight='bold', color='coral')
axes[1].set_ylabel('µV')

# File 3: one piece from group 0
signal3 = data_mxm[0]['trial'].item()[0][CH, :]
t3 = np.arange(3000) / 1000.0
axes[2].plot(t3, signal3, color='seagreen', linewidth=0.7)
axes[2].set_title('data_trialsmxm_3s — group 0, piece #0 (duration: 3.00s)',
                  fontweight='bold', color='seagreen')
axes[2].set_ylabel('µV')
axes[2].set_xlabel('Time (seconds)')

plt.tight_layout()
plt.savefig('mat_files_zoom.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved as mat_files_zoom.png")


# ==========================================================================
# Print a plain-English summary
# ==========================================================================
print("""
==========================================================================
PLAIN ENGLISH SUMMARY
==========================================================================

Imagine you recorded 21 minutes of brain activity on a long tape.

FILE 1 (data_ref):
  The tape was cut into 106 IRREGULAR pieces.
  Some pieces are 0.1 seconds, some are 56 seconds.
  This is the raw, unprocessed recording.

FILE 2 (data_trials3s):
  Someone took that same tape and re-cut it into pieces of
  EXACTLY 3 seconds each. Then they sorted those pieces into
  4 GROUPS (like 4 folders).
  Result: 342 pieces, each 3 seconds, in 4 groups.

FILE 3 (data_trialsmxm_3s):
  Same idea — 3-second pieces — but sorted into 21 GROUPS
  instead of 4. More groups = finer time resolution.
  Result: 375 pieces, each 3 seconds, in 21 groups.

WHY DIFFERENT NUMBER OF PIECES?
  When you cut a 7.4-second segment into 3-second pieces,
  you get 2 pieces and throw away the leftover 1.4 seconds.
  Files 2 and 3 may define group boundaries differently,
  so they lose different amounts of leftover data.

WHAT DOES EACH PIECE CONTAIN?
  A matrix of numbers: 36 rows (one per electrode on the scalp)
  × N columns (one per time sample at 1000 samples/second).
  For 3-second pieces: 36 × 3000.

THE KEY INSIGHT:
  All three files are the SAME brain recording.
  The only difference is how it was sliced and organized.
""")
