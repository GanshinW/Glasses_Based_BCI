# utils/load_bdf.py

import mne
import numpy as np

def extract_trials_from_bdf(bdf_path, mrk_path, trial_duration_s=12.5):
    """
    Read a .bdf file and its .mrk marker file, then split into EEG trials.
    Only EEG channels are retained. From the .mrk file with lines of the form:
        <sample_index> <sample_index> "<marker_label>"
    We only select those lines where marker_label == "255" (block onset).
    Then we use sample_index from those lines as trial starts.
    
    Parameters:
        bdf_path (str): Path to the .bdf file
        mrk_path (str): Path to the .mrk file (each line: "<sample> <sample> \"<label>\"")
        trial_duration_s (float): Length of each trial in seconds (default 12.5s)

    Returns:
        trials (np.ndarray): Array of shape (n_trials, n_channels, n_samples)
            containing EEG data for each trial.
    """
    # 1. Load raw BDF data (contains EEG + peripheral + triggers)
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
    raw.pick_types(eeg=True)      # Keep only EEG channels
    sfreq = raw.info['sfreq']     # Sampling frequency (e.g., 1024.0 Hz)

    # 2. Read .mrk file; select lines where marker = "255"
    triggers = []
    with open(mrk_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            # parts example: ["19841", "19841", "\"255\""] or ["TL02"]
            if parts[0] == 'TL02':
                continue
            # marker_label is parts[2], e.g. "\"255\""
            label = parts[2].strip('"')  # remove the quotes
            if label == '255':
                try:
                    sample_idx = int(parts[0])
                    triggers.append(sample_idx)
                except ValueError:
                    continue

    # Verify we found expected number of triggers
    if len(triggers) == 0:
        raise RuntimeError(f"No '255' markers found in '{mrk_path}'.")
    # 3. Calculate number of samples per trial
    samples_per_trial = int(trial_duration_s * sfreq)

    # 4. Extract each trial segment from raw data
    trials = []
    for start in triggers:
        stop = start + samples_per_trial
        # Avoid out-of-bounds
        if stop > raw.n_times:
            break
        data, _ = raw[:, start:stop]
        # data shape: (n_channels, samples_per_trial)
        trials.append(data)

    if len(trials) == 0:
        raise RuntimeError(f"No trials extracted. Check trial_duration_s and raw.n_times.")

    # Stack into array shape (n_trials, n_channels, n_samples)
    trials = np.stack(trials, axis=0)
    return trials
