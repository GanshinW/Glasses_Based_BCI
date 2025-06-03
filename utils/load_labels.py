# utils/load_labels.py

import numpy as np

def load_labels(label_txt_path, subject_idx=0):
    """
    Read IAPS_Classes_EEG_FNIRS.txt and map each textual label to an integer.
    Parameters:
        label_txt_path (str): Path to IAPS_Classes_EEG_FNIRS.txt
        subject_idx (int): Column index (0-based) for which subject’s labels to load
                            (0 for Part1, 1 for Part2, etc.)
    Returns:
        labels (List[int]): List of integer labels for each trial
            0 -> Calm, 1 -> Pos, 2 -> Neg
    """
    label_map = {"Calm": 0, "Pos": 1, "Neg": 2}

    # Read all lines, each line has three entries (one per subject)
    with open(label_txt_path, 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]

    # Select the column corresponding to subject_idx and map to integers
    labels = [label_map[row[subject_idx]] for row in lines]
    return labels
