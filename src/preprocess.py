"""
STEP 2: Clean up the signal.

Two jobs here:
1. Bandpass filter - keep only the 8-30 Hz range (the "mu" and "beta"
   bands), where motor imagery signals actually show up. Everything
   outside that range (eye blinks, muscle noise, mains hum, etc.) gets
   filtered out.
2. Epoching - cut the continuous recording into short clips ("epochs"),
   each one time-locked to a single imagery trial (e.g. the 4 seconds
   after the subject was told "imagine left fist now").
"""

import mne


def bandpass_filter(raw, l_freq: float = 8.0, h_freq: float = 30.0):
    """
    Keep only the frequency band relevant to motor imagery.

    l_freq / h_freq define the band we keep (8-30 Hz by default).
    Filtering happens in place and also returns raw for convenience.
    """
    raw.filter(l_freq, h_freq, fir_design="firwin", skip_by_annotation="edge")
    return raw


def extract_epochs(raw, tmin: float = -1.0, tmax: float = 4.0):
    """
    Cut the continuous recording into per-trial epochs.

    tmin/tmax are in seconds, relative to each event marker. tmin=-1
    means "include 1 second before the cue" (useful as a baseline),
    tmax=4 means "include 4 seconds after the cue" (long enough to
    cover the imagined movement).

    Returns
    -------
    epochs : mne.Epochs
        One epoch per trial, each labeled 'left' or 'right'.
    """
    events, event_id = mne.events_from_annotations(raw)

    # Only keep the 'left' and 'right' events - drop any 'rest' (T0)
    # trials, since we're building a binary left-vs-right classifier.
    event_id = {k: v for k, v in event_id.items() if k in ("left", "right")}

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")

    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )
    return epochs
