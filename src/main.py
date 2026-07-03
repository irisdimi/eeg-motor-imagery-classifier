"""
Runs the full pipeline, start to finish:
  load data -> filter -> epoch -> extract features -> classify -> evaluate

This is the file you actually run. Each step below corresponds to one
of the other files in src/ - if you want to understand or change a
step, that's the file to go to.
"""

from load_data import load_subject_data
from preprocess import bandpass_filter, extract_epochs
from classify import evaluate


def run(subject: int = 1):
    print(f"\n--- Loading data for subject {subject} ---")
    raw = load_subject_data(subject=subject)

    print("\n--- Filtering (8-30 Hz) ---")
    raw = bandpass_filter(raw)

    print("\n--- Extracting epochs ---")
    epochs = extract_epochs(raw)
    print(f"Got {len(epochs)} trials: {epochs.event_id}")

    print("\n--- Training + evaluating CSP+LDA classifier ---")
    evaluate(epochs)


if __name__ == "__main__":
    run(subject=1)
