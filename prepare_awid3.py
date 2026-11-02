
import pandas as pd
import numpy as np

INPUT = "data/Deauth_29.csv"

columns = [
    "frame.time_epoch",
    "wlan.fc.type",
    "wlan.fc.subtype",
    "wlan.ta",
    "radiotap.dbm_antsignal",
    "Label"
]

df = pd.read_csv(INPUT, usecols=columns, low_memory=False)

# Convert network features into numeric values
for col in [
    "frame.time_epoch",
    "wlan.fc.type",
    "wlan.fc.subtype",
    "radiotap.dbm_antsignal"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Label"] = df["Label"].astype(str).str.strip()

# Retain the normal and confirmed deauth labels
df = df[df["Label"].isin(["Normal", "Deauth"])].copy()

# Remove frames missing essential information
df = df.dropna(subset=[
    "frame.time_epoch", "wlan.fc.type",
    "wlan.fc.subtype", "wlan.ta"
])

# Group traffic into one-second windows
df["second"] = np.floor(df["frame.time_epoch"]).astype("int64")

df["is_deauth"] = (
    (df["wlan.fc.type"] == 0) &
    (df["wlan.fc.subtype"] == 12)
).astype(int)

df["is_management"] = (
    df["wlan.fc.type"] == 0
).astype(int)

df["is_attack"] = (df["Label"] == "Deauth").astype(int)

grouped = df.groupby(["wlan.ta", "second"])

features = grouped.agg(
    frames_per_second=("is_deauth", "size"),
    deauth_count=("is_deauth", "sum"),
    management_count=("is_management", "sum"),
    avg_signal=("radiotap.dbm_antsignal", "mean"),
    label=("is_attack", "max")
).reset_index()

features["management_ratio"] = (
    features["management_count"] /
    features["frames_per_second"]
)

features["avg_signal"] = features["avg_signal"].fillna(-100)

features = features.rename(columns={"wlan.ta": "mac"})

features = features[[
    "mac",
    "second",
    "frames_per_second",
    "deauth_count",
    "management_ratio",
    "avg_signal",
    "label"
]]

features.to_csv("features.csv", index=False)

print("Feature extraction completed!")
print("\nWindow labels:")
print(features["label"].value_counts())
print("\nFirst five rows:")
print(features.head())
