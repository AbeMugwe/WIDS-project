
# Machine Learning-Based Wireless Intrusion Detection System (WIDS)

## Project Overview

This project implements a Python-based Wireless Intrusion Detection System (WIDS) designed to detect suspicious Wi-Fi deauthentication activity using supervised machine learning.

The system processes recorded wireless traffic from the AWID3 dataset, extracts traffic features, trains and compares machine-learning classifiers, generates intrusion alerts, and displays results through a Streamlit dashboard.

**Current implementation:** Offline traffic analysis and replay. The system does not yet capture live wireless traffic or actively block attacks.

## Objectives

- Identify IEEE 802.11 deauthentication frames.
- Extract traffic features using one-second observation windows.
- Train and compare Decision Tree and Random Forest classifiers.
- Detect suspicious deauthentication activity using a trained model.
- Generate attack alerts and CSV reports.
- Visualize detected activity using Streamlit.
- Evaluate classification performance using accuracy, precision, recall, F1-score and a confusion matrix.

## Technologies Used

- Python
- Pandas and NumPy
- scikit-learn
- Joblib
- Matplotlib
- Streamlit
- AWID3 wireless intrusion dataset
- Kali Linux (development environment)

## Project Structure

```text
wids-project/
├── prepare_awid3.py     # Extract training features
├── prepare_test.py      # Extract validation features
├── compare_models.py    # Train, compare and select models
├── wids_v2.py           # Run offline intrusion detection
├── dashboard.py         # Streamlit monitoring dashboard
├── requirements.txt     # Python dependencies
├── .gitignore
└── data/                # Place AWID3 CSV files here
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AbeMugwe/WIDS-project.git
cd wids-project
```

### 2. Create a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Dataset Setup

The original AWID3 datasets are not included in this repository because of their size and distribution considerations.

Obtain the relevant datasets from an authorized AWID3 distribution. This implementation was developed with files named:

- `Deauth_29.csv` — Training data
- `Deauth_30.csv` — Cross-chunk validation data

Place the files in the `data/` directory:

```text
data/
├── Deauth_29.csv
└── Deauth_30.csv
```

The code expects AWID3-style columns, including:

- `frame.time_epoch`
- `wlan.fc.type`
- `wlan.fc.subtype`
- `wlan.ta`
- `radiotap.dbm_antsignal`
- `Label`

## How to Run the System

Run all commands from the project root with the virtual environment activated.

### Step 1: Prepare training features

```bash
python prepare_awid3.py
```

This processes `data/Deauth_29.csv` and creates:

```text
features.csv
```

### Step 2: Prepare validation features

```bash
python prepare_test.py
```

This processes `data/Deauth_30.csv` and creates:

```text
test_features_30.csv
```

### Step 3: Train and compare the models

```bash
python compare_models.py
```

The program trains Decision Tree and Random Forest classifiers, evaluates their performance, and saves the selected model.

Output:

```text
model_v2.joblib
```

The updated model-selection implementation prioritizes attack recall.

### Step 4: Run WIDS attack detection

```bash
python wis_v2.py --csv data/Deauth_30.csv --delay 0.2
```

To show normal traffic alongside attack alerts:

```bash
python wis_v2.py --csv data/Deauth_30.csv --delay 0.2 --show-normal
```

The detector loads the trained model, processes recorded traffic, and prints attack alerts in the terminal.

### Step 5: Launch the dashboard

```bash
python -m streamlit run dashboard.py
```

Open the local URL shown in the terminal, typically:

```text
http://localhost:8501
```

Use the dashboard controls to start, pause and reset the recorded traffic replay.

## Generated Outputs

Running the detector with `Deauth_30.csv` produces:

- `Deauth_30_wids_predictions.csv` — All classified observation windows.
- `Deauth_30_wids_alerts.csv` — Windows classified as attacks.
- `Deauth_30_wids_confusion_matrix.png` — Evaluation confusion matrix.

To open the confusion matrix on Kali Linux:

```bash
xdg-open Deauth_30_wids_confusion_matrix.png
```

## Feature Engineering

Captured frames are grouped by observed transmitter MAC address and one-second time interval.

The model uses three features:

| Feature | Description |
|---|---|
| `frames_per_second` | Total observed frames in a one-second window |
| `deauth_count` | Number of deauthentication frames in that window |
| `management_ratio` | Proportion of management frames |

IEEE 802.11 deauthentication frames are identified using management frame type `0` and subtype `12`.

Each window is assigned an attack label if at least one constituent frame is labelled `Deauth` in the dataset.

## Model Evaluation

During development, the Decision Tree achieved the following results on `Deauth_30.csv`:

| Metric | Result |
|---|---:|
| Accuracy | 99.67% |
| Attack precision | 100% |
| Attack recall | 98.65% |
| True positives | 73 |
| True negatives | 230 |
| False positives | 0 |
| False negatives | 1 |

These figures represent **cross-chunk development-validation performance**, not independently verified real-world detection accuracy.

The training and validation CSV files may contain related traffic from the same original capture. The validation chunk was also used for model comparison.

## Limitations

- The current system analyzes recorded traffic rather than live Wi-Fi captures.
- The trained Decision Tree relies heavily on deauthentication counts.
- MAC addresses can be spoofed and do not prove attacker identity.
- Performance on independent networks and capture sessions has not been established.
- The system provides detection and alerts, not active wireless intrusion prevention.

## Future Improvements

- Independent capture-session and cross-dataset evaluation.
- Live monitor-mode wireless packet capture.
- Additional features for low-rate or stealthier attacks.
- Temporal sequence analysis using LSTM.
- Evaluation of false-positive rates and detection latency.
- Support for additional Wi-Fi attack categories.

## Ethical Use

This project is intended for cybersecurity education, defensive monitoring and authorized security testing.

Only monitor networks for which you have appropriate permission.

## Dataset Attribution

AWID3: Aegean Wi-Fi Intrusion Dataset, developed for wireless intrusion detection research.

Dataset project: https://icsdweb.aegean.gr/awid/
