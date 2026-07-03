"""
STEP 1: Get the data.

We're using the PhysioNet EEG Motor Movement/Imagery Dataset.
It contains EEG recordings from 109 subjects performing (and imagining)
left/right fist movements and hands/feet movements.

mne has this dataset built in - calling eegbci.load_data() will
automatically download the files the first time you run it (and cache
them locally so you don't re-download every time).

Specifically, we want "runs 4, 8, 12" - these are the recordings where
the subject was told to IMAGINE opening/closing their left or right fist
(not actually move it). Each run contains many short trials.
"""

from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws


def load_subject_data(subject: int = 1, runs=(4, 8, 12)):
    """
    Download (if needed) and load EEG data for one subject.

    Parameters
    ----------
    subject : int
        Which subject to load (1-109). Start with subject 1 while you're
        building the pipeline - it's faster to iterate on one person's
        data before scaling up to many.
    runs : tuple of int
        Which recording runs to use. (4, 8, 12) = imagined left/right
        fist movement, which is what we want for this project.

    Returns
    -------
    raw : mne.io.Raw
        The concatenated raw EEG recording, with all three runs joined
        into one continuous object.
    """
    # This line does the downloading. First run: slow (fetches files).
    # After that: instant (reads from local cache).
    raw_fnames = eegbci.load_data(subject, runs)

    # Each run is a separate .edf file - read them all in, then stitch
    # them together into one continuous recording.
    raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raws)

    # The channel names in this dataset use an old naming convention.
    # This renames them to the modern standard so we can attach a
    # standard electrode layout (montage) in the next line.
    eegbci.standardize(raw)
    raw.set_montage("standard_1005")

    # The events in this dataset are labeled T0 (rest), T1, T2.
    # For these specific runs, T1 = imagined LEFT fist, T2 = imagined
    # RIGHT fist. We rename them here so the rest of our code is
    # readable (we deal with "left"/"right" instead of "T1"/"T2").
    raw.annotations.rename(dict(T1="left", T2="right"))

    return raw


if __name__ == "__main__":
    # Quick sanity check: run this file directly to confirm the download
    # and loading works before building the rest of the pipeline.
    raw = load_subject_data(subject=1)
    print(raw)
    print("Channel names:", raw.ch_names)
    print("Events found:", raw.annotations.description)
