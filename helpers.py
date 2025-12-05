import numpy as np
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt

def preprocess_subject(data, fs=100, lp_cutoff=5.0, lp_order=2):
    """
    Apply the same preprocessing pipeline as in Part 1 – Q1
    to one subject, exercise 1.

    Parameters
    ----------
    data : dict
        Output of scipy.io.loadmat for a single subject/exercise file.
        Must contain keys 'emg', 'restimulus', 'rerepetition'.
    fs : float
        Sampling frequency (Hz). For Ninapro DB1, fs = 100 Hz.
    lp_cutoff : float
        Low-pass cutoff frequency in Hz (default: 5 Hz).
    lp_order : int
        Order of the Butterworth low-pass filter (default: 2).

    Returns
    -------
    out : dict
        {
          "emg_lp":         (T, C) low-pass filtered EMG,
          "restimulus":     (T,) movement labels (0 = rest),
          "rerepetition":   (T,) repetition indices (0 = no repetition),
          "unique_stimuli": (S,) sorted movement labels (no rest),
          "unique_reps":    (R,) sorted repetition indices (no 0),
          "trial_masks":    list of boolean arrays over time (one per trial),
          "trial_labels":   (n_trials,) movement label for each trial,
          "trial_reps":     (n_trials,) repetition index for each trial,
          "channel_var":    (C,) variance of emg_lp per channel
        }
    """

    # --- 1) Extract raw arrays from the .mat dict ---
    emg = data["emg"].astype(float)                  # shape (T, C)
    restimulus = data["restimulus"].ravel()          # shape (T,)
    rerepetition = data["rerepetition"].ravel()      # shape (T,)

    # --- 2) Low-pass filter at 5 Hz (Butterworth, zero-phase) ---
    nyq = fs / 2.0
    wn = lp_cutoff / nyq
    sos = butter(lp_order, wn, btype="low", output="sos")
    emg_lp = sosfiltfilt(sos, emg, axis=0)

    # --- 3) Unique movements and repetitions (ignore rest/0) ---
    unique_stimuli = np.unique(restimulus)
    unique_stimuli = unique_stimuli[unique_stimuli != 0]

    unique_reps = np.unique(rerepetition)
    unique_reps = unique_reps[unique_reps != 0]

    # --- 4) Build trial masks (same logic as Part 1 – Q1) ---
    trial_masks = []
    trial_labels = []
    trial_reps = []

    for s in unique_stimuli:
        for r in unique_reps:
            mask = (restimulus == s) & (rerepetition == r)
            if not np.any(mask):
                continue  # no samples for this (movement, repetition) pair

            trial_masks.append(mask)
            trial_labels.append(s)
            trial_reps.append(r)

    trial_labels = np.asarray(trial_labels, dtype=int)
    trial_reps   = np.asarray(trial_reps, dtype=int)

    # --- 5) Channel variance (QC only, we do not remove channels here) ---
    channel_var = emg_lp.var(axis=0)

    out = {
        "emg_lp": emg_lp,
        "restimulus": restimulus,
        "rerepetition": rerepetition,
        "unique_stimuli": unique_stimuli,
        "unique_reps": unique_reps,
        "trial_masks": trial_masks,
        "trial_labels": trial_labels,
        "trial_reps": trial_reps,
        "channel_var": channel_var,
    }
    return out

def compute_trial_activation_scores(emg_lp, trial_masks, trial_labels, trial_reps):
    """
    Compute simple activation scores for each trial.

    Returns a dict with:
        "scores": (n_trials,) mean EMG over time and channels
        "labels": (n_trials,) movement label
        "reps":   (n_trials,) repetition index
    """
    n_trials = len(trial_masks)
    scores = np.zeros(n_trials)

    for k, mask in enumerate(trial_masks):
        trial = emg_lp[mask, :]  # (T_k, C)
        # mean over time and channels
        scores[k] = trial.mean()

    return {
        "scores": np.asarray(scores),
        "labels": np.asarray(trial_labels),
        "reps":   np.asarray(trial_reps),
    }

def find_suspicious_trials(scores, thr_low, z_thr):
    """
    Return indices of suspicious trials (very low activation).
    """
    median_score = np.median(scores)
    low_mask = scores < thr_low * median_score

    # z-score option
    mu = scores.mean()
    sigma = scores.std()
    if sigma == 0:
        z_mask = np.zeros_like(scores, dtype=bool)
    else:
        z = (scores - mu) / sigma
        z_mask = z < z_thr

    return np.where(low_mask | z_mask)[0]

def plot_suspicious_trials_for_subject(
    sid, suspicious, subjects_preprocessed, max_trials=None,
    fs=100.0, pre_seconds=1.0, post_seconds=1.0
):
    """
    Plot EMG for suspicious trials of one subject, with an extended time window
    around the movement to better assess whether channels/trials are truly dead.

    Parameters
    ----------
    sid : int
        Subject ID.
    suspicious : dict
        Suspicious-trial info as built earlier (per subject).
    subjects_preprocessed : dict
        Mapping sid -> output of preprocess_subject_ex1.
    max_trials : int or None
        Optional limit on how many suspicious trials to plot.
    fs : float
        Sampling frequency in Hz (100 Hz for Ninapro DB1).
    pre_seconds : float
        Extra time (in seconds) to include *before* the movement segment.
    post_seconds : float
        Extra time (in seconds) to include *after* the movement segment.
    """
    info = suspicious[sid]
    idx_list = info["indices"]
    labels   = info["labels"]
    reps     = info["reps"]
    scores   = info["scores"]

    if len(idx_list) == 0:
        print(f"No suspicious trials for subject {sid}")
        return

    res = subjects_preprocessed[sid]
    emg_lp       = res["emg_lp"]        # shape (T_total, C)
    trial_masks  = res["trial_masks"]   # list of boolean masks over time

    n_to_plot = len(idx_list) if max_trials is None else min(len(idx_list), max_trials)

    n_samples = emg_lp.shape[0]
    pre_samples  = int(pre_seconds * fs)
    post_samples = int(post_seconds * fs)

    for i in range(n_to_plot):
        k   = idx_list[i]     # index into trial_masks
        lab = labels[i]
        rep = reps[i]
        sc  = scores[i]

        mask_mov = trial_masks[k]              # True only during movement for this (mov, rep)
        mov_idx = np.where(mask_mov)[0]

        if mov_idx.size == 0:
            print(f"Warning: empty mask for subject {sid}, trial_idx {k}")
            continue

        # Extended window around the movement segment
        start_idx = max(0, mov_idx[0] - pre_samples)
        end_idx   = min(n_samples - 1, mov_idx[-1] + post_samples)

        # Slice EMG and build time axis
        trial_ext = emg_lp[start_idx:end_idx + 1, :]   # (T_ext, C)
        T_ext, C = trial_ext.shape
        t = np.arange(start_idx, end_idx + 1) / fs     # in seconds

        # Indices (in extended window) corresponding to the movement part
        mov_idx_ext_start = mov_idx[0] - start_idx
        mov_idx_ext_end   = mov_idx[-1] - start_idx

        fig, axes = plt.subplots(C, 1, figsize=(10, 1.5 * C), sharex=True)
        if C == 1:
            axes = [axes]

        for ch in range(C):
            axes[ch].plot(t, trial_ext[:, ch])
            # Optional: shade the true movement segment
            axes[ch].axvspan(
                t[mov_idx_ext_start],
                t[mov_idx_ext_end],
                alpha=0.2
            )
            axes[ch].set_ylabel(f"ch {ch+1}")

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(
            f"Subject {sid} – mov {lab}, rep {rep}, trial_idx {k}, score={sc:.4f}\n"
            f"Extended window: {pre_seconds:.1f}s before, {post_seconds:.1f}s after movement"
        )
        plt.tight_layout()
        plt.show()



def build_trial_index(restimulus, rerepetition):
    """
    Build trial index for a single subject / exercise.

    Returns
    -------
    trial_labels : (n_trials,) int
        Movement label (restimulus) for each trial.
    trial_reps : (n_trials,) int
        Repetition index for each trial.
    trial_masks : list of boolean arrays
        trial_masks[k] is a boolean mask over time samples selecting trial k.
    """
    unique_stimuli = np.unique(restimulus)
    unique_stimuli = unique_stimuli[unique_stimuli != 0]  # ignore rest

    unique_reps = np.unique(rerepetition)
    unique_reps = unique_reps[unique_reps != 0]  # ignore 0 = no repetition

    trial_labels = []
    trial_reps = []
    trial_masks = []

    for s in unique_stimuli:
        for r in unique_reps:
            mask = (restimulus == s) & (rerepetition == r)
            if np.any(mask):
                trial_labels.append(s)
                trial_reps.append(r)
                trial_masks.append(mask)

    trial_labels = np.asarray(trial_labels)
    trial_reps   = np.asarray(trial_reps)

    return trial_labels, trial_reps, trial_masks

def split_trials_by_repetition(trial_reps, train_reps, val_reps, test_reps):
    """
    Split trials into train / validation / test sets based on repetition index.
    """
    train_mask = np.isin(trial_reps, train_reps)
    val_mask   = np.isin(trial_reps, val_reps)
    test_mask  = np.isin(trial_reps, test_reps)

    # Sanity checks: masks must be disjoint
    assert not np.any(train_mask & val_mask)
    assert not np.any(train_mask & test_mask)
    assert not np.any(val_mask & test_mask)

    return train_mask, val_mask, test_mask

def mav(x):
    """
    Mean Absolute Value (per channel).

    x: array of shape (T, C) for one trial
    returns: array of shape (C,)
    """
    return np.mean(np.abs(x), axis=0)

def std_feature(x):
    """
    Standard deviation (per channel).
    """
    return np.std(x, axis=0, ddof=1)

def max_av(x):
    """
    Maximum absolute value (per channel).
    """
    return np.max(np.abs(x), axis=0)

def rms(x):
    """
    Root Mean Square (per channel).
    """
    return np.sqrt(np.mean(x**2, axis=0))

def waveform_length(x):
    """
    Waveform Length (per channel).

    Sum of absolute differences over time.
    """
    return np.sum(np.abs(np.diff(x, axis=0)), axis=0)

def build_dataset_from_ninapro_lp(emg_lp, stimulus, repetition, features):
    """
    Build a trial-level feature dataset from Ninapro EMG (low-pass filtered).

    Parameters
    ----------
    emg_lp : array, shape (T, C)
        Preprocessed EMG (e.g. 5 Hz low-pass filtered envelope).
    stimulus : array, shape (T,)
        Restimulus labels (0 = rest, 1..S = movements).
    repetition : array, shape (T,)
        Repetition indices (0 = no repetition, 1..R = reps).
    features : list of callables
        Each f(trial) returns an array of shape (C,) with one value per channel.

    Returns
    -------
    X : array, shape (n_trials, C * n_features)
        Feature matrix, one row per trial.
    y : array, shape (n_trials,)
        Movement labels (1..S).
    trial_reps : array, shape (n_trials,)
        Repetition index for each trial.
    """

    # Unique movements (ignore rest = 0)
    unique_stimuli = np.unique(stimulus)
    unique_stimuli = unique_stimuli[unique_stimuli != 0]

    # Unique repetitions (ignore 0)
    unique_reps = np.unique(repetition)
    unique_reps = unique_reps[unique_reps != 0]

    n_stimuli = len(unique_stimuli)
    n_reps = len(unique_reps)
    n_channels = emg_lp.shape[1]
    n_feat_per_channel = len(features)

    # Maximum possible number of trials (some (stim,rep) could be empty in theory)
    n_trials_max = n_stimuli * n_reps
    n_features_total = n_channels * n_feat_per_channel

    X = np.zeros((n_trials_max, n_features_total))
    y = np.zeros(n_trials_max, dtype=int)
    trial_reps = np.zeros(n_trials_max, dtype=int)

    trial_idx = 0

    for s in unique_stimuli:
        for r in unique_reps:
            mask = (stimulus == s) & (repetition == r)
            if not np.any(mask):
                # No samples for this (movement, repetition) pair
                continue

            trial = emg_lp[mask, :]  # shape (T_sr, C)

            feat_vec = []
            for f in features:
                fvals = f(trial)  # shape (C,)
                feat_vec.append(fvals)

            feat_vec = np.concatenate(feat_vec, axis=0)  # shape (C * n_features,)

            X[trial_idx, :] = feat_vec
            y[trial_idx] = s
            trial_reps[trial_idx] = r
            trial_idx += 1

    # Trim unused rows (if any)
    X = X[:trial_idx, :]
    y = y[:trial_idx]
    trial_reps = trial_reps[:trial_idx]

    return X, y, trial_reps


def get_feature_matrix_for_feature_index(X, feature_index, n_channels):
    """
    Extract a (n_trials, n_channels) matrix for one feature.

    feature_index: 0 for MAV, 1 for STD, 2 for MAX, 3 for RMS, 4 for WL
    """
    start = feature_index * n_channels
    end   = (feature_index + 1) * n_channels
    return X[:, start:end]