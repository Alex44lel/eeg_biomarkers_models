# Findings: .mat files in data/recordings/{subject}/{condition}/

Analysis based on subject S01AS, DMT condition. All subjects follow the same structure.

## General info

- All three files contain the **same brain recording** from one person, just sliced differently.
- 36 channels (electrodes on the scalp), of which 32 are EEG and 4 are physiological references (ECG, VEOG, EMGfront, EMGtemp).
- Sampling rate: 1000 Hz (1000 data points per second).
- Total recording duration: ~30 minutes.
- Format: MATLAB v5 (readable with `scipy.io.loadmat()`).
- Created with FieldTrip, a MATLAB EEG analysis toolbox.

### Common fields in every struct

| Field        | Description                                                        |
| ------------ | ------------------------------------------------------------------ |
| `hdr`        | Header info (recording device metadata)                            |
| `fsample`    | Sampling frequency in Hz (always 1000)                             |
| `sampleinfo` | Start/end sample indices for each piece (Nx2 matrix)               |
| `trial`      | Array of EEG data pieces, each is a matrix of (channels x samples) |
| `time`       | Array of time vectors (one per piece, in seconds)                  |
| `label`      | Channel names (electrode positions)                                |
| `cfg`        | Configuration/processing history from FieldTrip                    |

---

## File 1: data_ref.mat

The raw continuous recording, cut into **irregular** pieces.

- **Key**: `data_ref`
- **Number of pieces**: 106
- **Piece length**: variable (0.1s to 56s, mean 11.95s)
- **Total data**: 1,266,945 samples per channel (1266.9s = 21.1 min)
- **Time range**: 1.1 min – 30.6 min

Each piece is a matrix of shape `(36, N)` where N varies per piece.

This is the unprocessed starting point from which the other two files were derived.

---

## File 2: data_trials3s.mat

The same recording re-cut into **fixed 3-second pieces**, sorted into **4 groups**.

- **Key**: `data_trials3s` (shape: 4 elements, one per group)
- **Piece length**: exactly 3 seconds (36 channels x 3000 samples each)
- **Total pieces**: 342
- **Total data**: 1,026,000 samples per channel (1026.0s, 81% of raw)

### Group breakdown

| Group | Pieces | Time range          | Meaning                  |
| ----- | ------ | ------------------- | ------------------------ |
| 0     | 85     | 1.1 min – 6.1 min   | Pre-drug baseline        |
| —     | —      | 6.1 min – 11.1 min  | GAP: drug administration |
| 1     | 81     | 11.1 min – 16.6 min | Early drug effects       |
| 2     | 81     | 16.8 min – 22.4 min | Mid/peak drug effects    |
| 3     | 95     | 22.7 min – 28.6 min | Late effects / comedown  |

**Note**: There are no explicit group labels stored in the file. The meaning was inferred from the `sampleinfo` time ranges. The 5-minute gap between Group 0 and Group 1 corresponds to the drug injection period.

### Group breakdown for all subjects (DMT, data_trials3s)

| Subject | Group | Pieces | Time range |
| ------- | ----: | -----: | ---------- |
| S01AS | 0 | 85 | 1.1 – 6.1 min |
| S01AS | 1 | 81 | 11.1 – 16.6 min |
| S01AS | 2 | 81 | 16.8 – 22.4 min |
| S01AS | 3 | 95 | 22.7 – 28.6 min |
| S02WT | 0 | 87 | 1.0 – 6.0 min |
| S02WT | 1 | 70 | 8.9 – 14.3 min |
| S02WT | 2 | 83 | 14.5 – 20.3 min |
| S02WT | 3 | 92 | 20.5 – 26.3 min |
| S03BS | 0 | 97 | 1.2 – 6.2 min |
| S03BS | 1 | 86 | 10.1 – 15.6 min |
| S03BS | 2 | 89 | 15.8 – 21.6 min |
| S03BS | 3 | 89 | 21.8 – 27.6 min |
| S04SG | 0 | 36 | 1.1 – 6.0 min |
| S04SG | 1 | 86 | 10.1 – 15.6 min |
| S04SG | 2 | 93 | 15.7 – 21.6 min |
| S04SG | 3 | 95 | 21.7 – 27.6 min |
| S05LM | 0 | 100 | 1.5 – 6.5 min |
| S05LM | 1 | 79 | 9.2 – 14.6 min |
| S05LM | 2 | 94 | 14.9 – 20.7 min |
| S05LM | 3 | 96 | 20.8 – 26.7 min |
| S06ET | 0 | 79 | 1.5 – 6.3 min |
| S06ET | 1 | 54 | 11.1 – 16.3 min |
| S06ET | 2 | 86 | 16.7 – 22.2 min |
| S06ET | 3 | 98 | 22.5 – 28.2 min |
| S07CS | 0 | 86 | 1.2 – 6.1 min |
| S07CS | 1 | 49 | 10.4 – 15.8 min |
| S07CS | 2 | 72 | 16.0 – 21.9 min |
| S07CS | 3 | 96 | 22.0 – 27.9 min |
| S08EK | 0 | 94 | 1.2 – 6.2 min |
| S08EK | 1 | 68 | 13.6 – 19.1 min |
| S08EK | 2 | 79 | 19.3 – 25.1 min |
| S08EK | 3 | 86 | 25.2 – 31.1 min |
| S09BB | 0 | 96 | 1.2 – 6.2 min |
| S09BB | 1 | 67 | 10.6 – 16.1 min |
| S09BB | 2 | — | — |
| S09BB | 3 | — | — |
| S10DL | 0 | 83 | 1.1 – 6.1 min |
| S10DL | 1 | 81 | 10.1 – 15.6 min |
| S10DL | 2 | 70 | 15.7 – 21.5 min |
| S10DL | 3 | 91 | 21.8 – 27.5 min |
| S11NW | 0 | 90 | 1.3 – 6.2 min |
| S11NW | 1 | 75 | 9.8 – 15.2 min |
| S11NW | 2 | 92 | 15.4 – 21.2 min |
| S11NW | 3 | 94 | 21.5 – 27.2 min |
| S12AI | 0 | 98 | 1.4 – 6.4 min |
| S12AI | 1 | 84 | 10.5 – 16.0 min |
| S12AI | 2 | 94 | 16.1 – 22.0 min |
| S12AI | 3 | 95 | 22.1 – 28.0 min |
| S13MBJ | 0 | 99 | 1.5 – 6.5 min |
| S13MBJ | 1 | 72 | 10.3 – 15.7 min |
| S13MBJ | 2 | 89 | 15.9 – 21.7 min |
| S13MBJ | 3 | 103 | 21.8 – 27.7 min |

**Notable observations**:
- **S09BB** only has groups 0 and 1 — groups 2 and 3 are missing, meaning the recording was cut short or data was lost after ~16 min.
- **S04SG** has an unusually small baseline (group 0): only 36 pieces vs 79–100 for other subjects.
- **S08EK** has a later drug administration (group 1 starts at 13.6 min vs ~9–11 min for others), shifting all subsequent groups later.
- The gap between group 0 and group 1 varies across subjects (~3–7.5 min), reflecting different drug administration timing.

### Why fewer total samples than data_ref?

When a variable-length piece is re-cut into 3-second windows, the leftover tail (< 3s) is discarded. For example, a 7.4s piece yields 2 pieces of 3s, and 1.4s is thrown away.

---

## File 3: data_trialsmxm_3s.mat

Same 3-second pieces, but sorted into **21 groups** for finer temporal resolution.

- **Key**: `data_trialsmxm_3s` (shape: 21 elements, one per group)
- **Piece length**: exactly 3 seconds (36 channels x 3000 samples each)
- **Total pieces**: 375
- **Total data**: 1,125,000 samples per channel (1125.0s, 88.8% of raw)

### Group breakdown

| Group | Pieces | Time range          | Meaning                   |
| ----- | ------ | ------------------- | ------------------------- |
| 0     | 85     | 1.1 min – 6.1 min   | Pre-drug baseline         |
| 1     | 11     | 11.1 min – 11.8 min | Post-drug, ~minute 1      |
| 2     | 11     | 11.8 min – ...      | Post-drug, ~minute 2      |
| ...   | 9–18   | ~1 min each         | Each group ≈ 1 minute bin |
| 20    | 16     | 29.7 min – 30.6 min | Final minute of recording |

Group 0 is identical to Group 0 in `data_trials3s` (same 85 pieces, same time range). Groups 1–20 subdivide the post-drug period into ~1-minute bins, whereas `data_trials3s` groups them into just 3 coarse blocks.

### Why more total pieces than data_trials3s?

`data_trialsmxm_3s` covers more of the recording. `data_trials3s` stops at ~28.6 min, while `data_trialsmxm_3s` goes all the way to ~30.6 min (the end of the raw recording). Specifically, mxm groups 19 and 20 contain data that `data_trials3s` does not include at all.

### Mapping between the two files

| mxm groups | trials3s group      | Time range       |
|------------|---------------------|------------------|
| 0          | 0 (baseline)        | 1.1 – 6.1 min   |
| 1–6        | 1 (early effects)   | 11.1 – 16.6 min |
| 7–12       | 2 (mid/peak)        | 16.8 – 22.4 min |
| 13–18      | 3 (late/comedown)   | 22.7 – 28.6 min |
| **19–20**  | **not covered**     | 28.7 – 30.6 min |

---

## Summary comparison

| Property         | data_ref       | data_trials3s     | data_trialsmxm_3s   |
| ---------------- | -------------- | ----------------- | ------------------- |
| Piece length     | Variable       | 3 seconds         | 3 seconds           |
| Number of pieces | 106            | 342               | 375                 |
| Number of groups | 1 (ungrouped)  | 4                 | 21                  |
| Total duration   | 1266.9s (100%) | 1026.0s (81%)     | 1125.0s (89%)       |
| Use case         | Raw data       | Coarse conditions | Fine time evolution |

All three are the same recording — raw tape vs 4 big bins vs 21 small bins.

---

## Piece counts across all subjects

Each subject has two conditions: DMT and PLA (placebo). The columns show:

- **trials3s total**: total pieces in `data_trials3s.mat`
- **g0 (3s)**: pieces in group 0 (baseline) of `data_trials3s.mat`
- **3s - g0**: post-drug pieces in `data_trials3s.mat`
- **mxm total**: total pieces in `data_trialsmxm_3s.mat`
- **g0 (mxm)**: pieces in group 0 (baseline) of `data_trialsmxm_3s.mat`
- **mxm - g0**: post-drug pieces in `data_trialsmxm_3s.mat`

| Subject    | trials3s total | g0 (3s) | 3s - g0 | mxm total | g0 (mxm) | mxm - g0 |
| ---------- | -------------: | ------: | ------: | --------: | -------: | -------: |
| S01AS/DMT  |            342 |      85 |     257 |       375 |       85 |      290 |
| S01AS/PLA  |            354 |      95 |     259 |       386 |       95 |      291 |
| S02WT/DMT  |            332 |      87 |     245 |       364 |       87 |      277 |
| S02WT/PLA  |            351 |      98 |     253 |       382 |       98 |      284 |
| S03BS/DMT  |            361 |      97 |     264 |       390 |       97 |      293 |
| S03BS/PLA  |            222 |      49 |     173 |       246 |       49 |      197 |
| S04SG/DMT  |            310 |      36 |     274 |       342 |       36 |      306 |
| S04SG/PLA  |            376 |      97 |     279 |       410 |       97 |      313 |
| S05LM/DMT  |            369 |     100 |     269 |       405 |      100 |      305 |
| S05LM/PLA  |            386 |      97 |     289 |       421 |       97 |      324 |
| S06ET/DMT  |            317 |      79 |     238 |       348 |       79 |      269 |
| S06ET/PLA  |            367 |      89 |     278 |       398 |       89 |      309 |
| S07CS/DMT  |            303 |      86 |     217 |       338 |       86 |      252 |
| S07CS/PLA  |            361 |      86 |     275 |       387 |       86 |      301 |
| S08EK/DMT  |            327 |      94 |     233 |       354 |       94 |      260 |
| S08EK/PLA  |            334 |      94 |     240 |       369 |       94 |      275 |
| S09BB/DMT  |            163 |      96 |      67 |       383 |       96 |      287 |
| S09BB/PLA  |            160 |      96 |      64 |       164 |       96 |       68 |
| S10DL/DMT  |            325 |      83 |     242 |       360 |       83 |      277 |
| S10DL/PLA  |            309 |      75 |     234 |       338 |       75 |      263 |
| S11NW/DMT  |            351 |      90 |     261 |       383 |       90 |      293 |
| S11NW/PLA  |            375 |      93 |     282 |       409 |       93 |      316 |
| S12AI/DMT  |            371 |      98 |     273 |       406 |       98 |      308 |
| S12AI/PLA  |            358 |      82 |     276 |       391 |       82 |      309 |
| S13MBJ/DMT |            363 |      99 |     264 |       394 |       99 |      295 |
| S13MBJ/PLA |            391 |      94 |     297 |       423 |       94 |      329 |
