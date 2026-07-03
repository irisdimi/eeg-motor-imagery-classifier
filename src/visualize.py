"""
Generates figures for the report. Reuses the same pipeline pieces from
load_data.py / preprocess.py / features.py / classify.py, so nothing
here changes how the classifier itself works - it just LOOKS at what's
happening at each stage and saves a picture of it.

Run this after main.py has worked at least once (so you know the
pipeline runs cleanly). Figures are saved into a "figures/" folder as
.png files, ready to drop straight into a Word doc or report.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, cross_val_predict

from load_data import load_subject_data
from preprocess import bandpass_filter, extract_epochs
from features import make_csp

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def save(fig, name):
    """Small helper so every plot gets saved the same way."""
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_raw_vs_filtered(raw_unfiltered, raw_filtered, channel="C3"):
    """
    STEP 2 visual: shows a few seconds of one electrode's signal before
    and after filtering, so the reader can SEE what "cleaning the
    signal" actually did to the waveform.
    """
    sfreq = raw_unfiltered.info["sfreq"]
    start, stop = 0, int(5 * sfreq)  # first 5 seconds

    ch_idx = raw_unfiltered.ch_names.index(channel)
    raw_data = raw_unfiltered.get_data(picks=[ch_idx], start=start, stop=stop)[0]
    filt_data = raw_filtered.get_data(picks=[ch_idx], start=start, stop=stop)[0]
    times = np.arange(len(raw_data)) / sfreq

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(times, raw_data, color="tab:gray")
    axes[0].set_title(f"Raw EEG signal ({channel}) - unfiltered")
    axes[0].set_ylabel("Amplitude (V)")

    axes[1].plot(times, filt_data, color="tab:blue")
    axes[1].set_title(f"After 8-30 Hz bandpass filter ({channel})")
    axes[1].set_ylabel("Amplitude (V)")
    axes[1].set_xlabel("Time (s)")

    fig.tight_layout()
    save(fig, "01_raw_vs_filtered.png")


def plot_psd(raw, fmax=60.0):
    """
    Power Spectral Density: shows how much signal power exists at each
    frequency. This is the evidence for WHY we picked 8-30 Hz -
    you should be able to see elevated power in that range.
    """
    fig = raw.compute_psd(fmax=fmax).plot(show=False)
    fig.suptitle("Power Spectral Density (before epoching)")
    save(fig, "02_psd.png")


def plot_csp_patterns(epochs):
    """
    The headline figure: shows CSP as a brain-shaped topomap, i.e.
    WHERE on the scalp the model is finding the left-vs-right
    difference. Motor imagery should show up strongest over the
    motor cortex (central electrodes, roughly where C3/C4 sit).
    """
    X = epochs.get_data(copy=False)
    y = epochs.events[:, -1]

    csp = make_csp(n_components=4)
    csp.fit(X, y)

    fig = csp.plot_patterns(epochs.info, ch_type="eeg", components=[0, 1, 2, 3], show=False)
    fig.suptitle("CSP Spatial Patterns (left vs. right motor imagery)")
    save(fig, "03_csp_patterns.png")


def plot_accuracy_per_split(scores):
    """
    Bar chart of accuracy per cross-validation split, with the mean
    and chance level (50%) marked - makes the variability visible at
    a glance instead of just listing numbers.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    splits = np.arange(1, len(scores) + 1)
    ax.bar(splits, scores, color="tab:blue")
    ax.axhline(scores.mean(), color="black", linestyle="--", label=f"Mean = {scores.mean():.2f}")
    ax.axhline(0.5, color="red", linestyle=":", label="Chance level = 0.50")
    ax.set_xlabel("Cross-validation split")
    ax.set_ylabel("Accuracy")
    ax.set_title("Classifier accuracy across cross-validation splits")
    ax.set_ylim(0, 1)
    ax.legend()
    save(fig, "04_accuracy_per_split.png")


def plot_confusion_matrix(epochs):
    """
    Shows exactly what the model gets right/wrong: of all the times
    the true label was 'left', how often did it correctly say 'left'
    vs. wrongly say 'right' (and vice versa).
    """
    X = epochs.get_data(copy=False)
    y = epochs.events[:, -1]

    csp = make_csp(n_components=4)
    lda = LinearDiscriminantAnalysis()
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([("CSP", csp), ("LDA", lda)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)

    labels = list(epochs.event_id.keys())
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    save(fig, "05_confusion_matrix.png")


def run(subject: int = 1):
    print(f"\n--- Generating figures for subject {subject} ---")

    raw_unfiltered = load_subject_data(subject=subject)
    raw_filtered = raw_unfiltered.copy()
    raw_filtered = bandpass_filter(raw_filtered)

    plot_raw_vs_filtered(raw_unfiltered, raw_filtered)
    plot_psd(raw_filtered)

    epochs = extract_epochs(raw_filtered)
    plot_csp_patterns(epochs)
    plot_confusion_matrix(epochs)

    # Re-run cross-validation here just to get scores for the bar chart
    from classify import evaluate
    scores = evaluate(epochs)
    plot_accuracy_per_split(scores)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    run(subject=1)
