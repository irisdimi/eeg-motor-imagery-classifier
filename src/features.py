"""
STEP 3: Turn raw squiggly signals into a small set of useful numbers.

We use CSP - Common Spatial Patterns - the classic feature extraction
method for exactly this kind of task (two-class motor imagery). In
plain terms: CSP finds combinations of electrode channels where the
"left" trials and "right" trials look MOST different from each other,
and boils each trial down into a handful of numbers describing how
strongly that difference shows up.

mne has CSP built in - we don't need to implement the maths ourselves.
"""

from mne.decoding import CSP


def make_csp(n_components: int = 4):
    """
    Build a CSP feature extractor.

    n_components controls how many "spatial filter" outputs we keep per
    trial - i.e. how many numbers each trial gets boiled down to.
    4 is a common, reasonable starting point: enough to capture the
    useful pattern without overfitting on a small dataset.
    """
    return CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
