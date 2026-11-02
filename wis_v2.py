
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

REQUIRED_COLUMNS = [
    "frame.time_epoch",
    "wlan.fc.type",
    "wlan.fc.subtype",
    "wlan.ta",
    "radiotap.dbm_antsignal",
    "Label",
]


def extract_features(csv_path):
    print(f"\n[INFO] Reading {csv_path}")

    df = pd.read_csv(
        csv_path,
        usecols=REQUIRED_COLUMNS,
        low_memory=False,
    )

    print(f"[INFO] Loaded {len(df)} frames")

    for col in [
        "frame.time_epoch",
        "wlan.fc.type",
        "wlan.fc.subtype",
        "radiotap.dbm_antsignal",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Label"] = df["Label"].astype("string").str.strip()

    # Only use the classes included in our training.
    valid = df["Label"].isin(["Normal", "Deauth"])
    df = df.loc[valid].copy()

    df["wlan.ta"] = df["wlan.ta"].astype("string").str.strip()
    df["wlan.ta"] = df["wlan.ta"].replace("", pd.NA)

    df = df.dropna(subset=[
        "frame.time_epoch",
        "wlan.fc.type",
        "wlan.fc.subtype",
        "wlan.ta",
    ])

    if df.empty:
        raise ValueError("No usable wireless frames found.")

    # One-second observation windows, per transmitter.
    df["second"] = np.floor(
        df["frame.time_epoch"]
    ).astype("int64")

    df["is_deauth"] = (
        (df["wlan.fc.type"] == 0)
        & (df["wlan.fc.subtype"] == 12)
    ).astype(int)

    df["is_management"] = (
        df["wlan.fc.type"] == 0
    ).astype(int)

    df["is_attack"] = (
        df["Label"] == "Deauth"
    ).astype(int)

    grouped = df.groupby(
        ["wlan.ta", "second"],
        dropna=True
    )

    result = grouped.agg(
        frames_per_second=("is_deauth", "size"),
        deauth_count=("is_deauth", "sum"),
        management_count=("is_management", "sum"),
        avg_signal=("radiotap.dbm_antsignal", "mean"),
        label=("is_attack", "max"),
    ).reset_index()

    result["management_ratio"] = (
        result["management_count"]
        / result["frames_per_second"]
    )

    # Same missing-signal handling as training.
    result["avg_signal"] = result["avg_signal"].fillna(-100)

    result = result.rename(columns={"wlan.ta": "mac"})

    result = result.drop(columns=["management_count"])

    # Sort by time first, then transmitter.
    result = result.sort_values(
        ["second", "mac"]
    ).reset_index(drop=True)

    return result


def run_detection(csv_path, model_path, delay, show_normal):
    # Only load models you created or trust.
    saved = joblib.load(model_path)

    if not isinstance(saved, dict):
        raise ValueError(
            "Expected model_v2.joblib containing "
            "a model and its feature names."
        )

    model = saved["model"]
    features = saved["features"]

    print("[INFO] Model loaded successfully")
    print("[INFO] Features:", features)

    df = extract_features(csv_path)

    df["prediction"] = model.predict(df[features])

    df["timestamp"] = pd.to_datetime(
        df["second"], unit="s", utc=True
    ).dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    source_name = Path(csv_path).stem

    predictions_file = f"{source_name}_wids_predictions.csv"
    alerts_file = f"{source_name}_wids_alerts.csv"
    matrix_file = f"{source_name}_wids_confusion_matrix.png"

    df.to_csv(predictions_file, index=False)

    alerts = df[df["prediction"] == 1].copy()
    alerts.to_csv(alerts_file, index=False)

    print("\n" + "=" * 55)
    print("WIRELESS INTRUSION DETECTION SYSTEM - WIDS V2")
    print("=" * 55)

    print("\n[INFO] Replaying captured traffic chronologically\n")

    for row in df.itertuples(index=False):
        timestamp = row.timestamp
        mac = row.mac
        rate = int(row.frames_per_second)
        deauth = int(row.deauth_count)

        if row.prediction == 1:
            print(
                f"[ALERT] {timestamp} | "
                f"MAC: {mac} | "
                f"Frames/s: {rate} | "
                f"Deauth/s: {deauth}",
                flush=True,
            )

        elif show_normal:
            print(
                f"[NORMAL] {timestamp} | "
                f"MAC: {mac} | "
                f"Frames/s: {rate} | "
                f"Deauth/s: {deauth}",
                flush=True,
            )

        # Optional delay for a presentation replay.
        if delay > 0:
            time.sleep(delay)

    normal_count = int((df["prediction"] == 0).sum())
    attack_count = int((df["prediction"] == 1).sum())

    print("\n" + "=" * 55)
    print("DETECTION SUMMARY")
    print("=" * 55)

    print("Total windows:", len(df))
    print("Predicted normal:", normal_count)
    print("Predicted attacks:", attack_count)
    print("Unique flagged MACs:", alerts["mac"].nunique())

    print("\nMODEL EVALUATION")

    y_true = df["label"]
    y_pred = df["prediction"]

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        zero_division=0,
    ))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    print("Confusion matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Attack"],
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(f"WIDS Evaluation: {source_name}")
    plt.tight_layout()
    plt.savefig(matrix_file, dpi=200)
    plt.close()

    print("\nFILES SAVED")
    print(predictions_file)
    print(alerts_file)
    print(matrix_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AWID3 CSV-based WIDS detector"
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to labelled AWID3 CSV file",
    )

    parser.add_argument(
        "--model",
        default="model_v2.joblib",
        help="Path to trained model",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between displayed windows",
    )

    parser.add_argument(
        "--show-normal",
        action="store_true",
        help="Print normal predictions as well",
    )

    args = parser.parse_args()

    run_detection(
        args.csv,
        args.model,
        args.delay,
        args.show_normal,
    )
