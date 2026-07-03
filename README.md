# EEG Motor Imagery Classifier

Classifies imagined **left-hand vs. right-hand** movement from EEG signals, using Common Spatial Patterns (CSP) for feature extraction and Linear Discriminant Analysis (LDA) for classification. Built as a side project to explore brain–computer interface (BCI) signal processing, alongside coursework in neural interfaces and bioelectronics.

📄 **[Read the full report (PDF)](report_ieee/report_ieee.pdf)** - IEEE-format writeup with methodology, results, and discussion.

## Overview

When a person imagines moving their left or right hand (without actually moving), distinct patterns emerge in EEG recorded over the motor cortex. This project builds a full pipeline - from raw EEG to a trained classifier - to detect that difference automatically.

**Result:** 62.2% mean classification accuracy across 10-fold cross-validation on a single subject (chance level = 50%), with CSP spatial patterns showing activity concentrated over central electrodes - consistent with known motor cortex topography.

## Dataset

[PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) — 64-channel EEG recordings collected via the BCI2000 system. Downloads automatically on first run via `mne`.

## Pipeline

| Stage | File | What it does |
|---|---|---|
| 1. Load data | `src/load_data.py` | Downloads and loads raw EEG for a subject |
| 2. Preprocess | `src/preprocess.py` | Bandpass filters (8–30 Hz) and epochs the signal |
| 3. Feature extraction | `src/features.py` | Common Spatial Patterns (CSP) |
| 4. Classify | `src/classify.py` | CSP + LDA pipeline, cross-validated |
| 5. Run everything | `src/main.py` | Runs the full pipeline end to end |
| 6. Visualize | `src/visualize.py` | Generates report figures (signal plots, PSD, CSP topomaps, accuracy, confusion matrix) |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
cd src
python3 main.py         # runs the classifier, prints accuracy
python3 visualize.py    # generates figures into ../figures/
```

First run downloads EEG data automatically (cached locally afterward).

## Results

- **Mean accuracy:** 62.2% (chance level: 50%)
- **Per-split range:** 33.3% - 77.8% (10 cross-validation splits, single subject)
- CSP spatial patterns show strongest activity over central electrodes, consistent with sensorimotor cortex involvement in hand motor imagery

See `figures/` for generated plots, or the [full report](report_ieee/report_ieee.pdf) for detailed discussion and limitations.

## Repository Structure

```
eeg-motor-imagery-classifier/
├── README.md
├── requirements.txt
├── src/                    # pipeline code
├── figures/                # generated plots
└── report_ieee/            # IEEE-format report (.tex, .pdf, figures)
```

## Future Work

- Evaluate across multiple subjects for a more robust accuracy estimate
- Compare LDA against SVM / filter-bank CSP variants
- Extend toward real-time/online classification
- Pair with an EMG-controlled Arduino actuator (in progress, separate side project)

## Author

Aya Banna - Biomedical Engineering, University of Sydney
