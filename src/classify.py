"""
STEP 4 & 5: Teach a model to guess, then check how well it worked.

We chain CSP (feature extraction) + LDA (the actual classifier) into a
single scikit-learn Pipeline, then use cross-validation to test it
fairly: repeatedly split the data into "train" and "test" chunks, fit
on train, score on test, and average the results. This avoids the
mistake of testing the model on data it already memorized.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from features import make_csp


def build_pipeline():
    """
    CSP -> LDA pipeline.

    LDA (Linear Discriminant Analysis) is a simple, fast, easy-to-explain
    classifier - a good baseline before reaching for anything fancier
    like an SVM or a neural network. Real BCI papers still use CSP+LDA
    as a standard baseline, so this isn't just a toy choice.
    """
    csp = make_csp(n_components=4)
    lda = LinearDiscriminantAnalysis()
    return Pipeline([("CSP", csp), ("LDA", lda)])


def evaluate(epochs, n_splits: int = 10, test_size: float = 0.2):
    """
    Run cross-validated evaluation of the CSP+LDA pipeline.

    Parameters
    ----------
    epochs : mne.Epochs
        The labeled trial data from preprocess.extract_epochs().
    n_splits : int
        How many random train/test splits to average over. More splits
        = more reliable estimate, but slower.
    test_size : float
        Fraction of trials held out for testing each split (0.2 = 20%).

    Returns
    -------
    scores : np.ndarray
        Accuracy for each split.
    """
    X = epochs.get_data(copy=False)  # shape: (n_trials, n_channels, n_times)
    y = epochs.events[:, -1]         # the label (left/right) for each trial

    pipeline = build_pipeline()
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)

    scores = cross_val_score(pipeline, X, y, cv=cv, n_jobs=1)

    print(f"Accuracy per split: {np.round(scores, 3)}")
    print(f"Mean accuracy: {scores.mean():.3f}  (chance level = 0.5)")

    return scores
