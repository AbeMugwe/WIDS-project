
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="WIDS Security Monitor",
    page_icon="🛡️",
    layout="wide",
)

MODEL_FILE = "model_v2.joblib"
DATA_FILE = "data/Deauth_30.csv"

st.title("🛡️ Wireless Intrusion Detection System")
st.caption(
    "Machine Learning-Based Wi-Fi Deauthentication Detection "
    "| AWID3 Dataset | Offline Replay"
)


@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_FILE)
    return saved["model"], saved["features"]


@st.cache_data
def load_traffic(path):
    columns = [
        "frame.time_epoch",
        "wlan.fc.type",
        "wlan.fc.subtype",
        "wlan.ta",
        "radiotap.dbm_antsignal",
        "Label",
    ]

    df = pd.read_csv(path, usecols=columns, low_memory=False)

    for col in [
        "frame.time_epoch",
        "wlan.fc.type",
        "wlan.fc.subtype",
        "radiotap.dbm_antsignal",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Label"] = df["Label"].astype("string").str.strip()

    df["wlan.ta"] = (
        df["wlan.ta"].astype("string").str.strip()
    )

    df["wlan.ta"] = df["wlan.ta"].replace("", pd.NA)

    df = df[df["Label"].isin(["Normal", "Deauth"])].copy()

    df = df.dropna(subset=[
        "frame.time_epoch",
        "wlan.fc.type",
        "wlan.fc.subtype",
        "wlan.ta",
    ])

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

    grouped = df.groupby(["wlan.ta", "second"])

    features = grouped.agg(
        frames_per_second=("is_deauth", "size"),
        deauth_count=("is_deauth", "sum"),
        management_count=("is_management", "sum"),
        avg_signal=("radiotap.dbm_antsignal", "mean"),
        label=("is_attack", "max"),
    ).reset_index()

    features["management_ratio"] = (
        features["management_count"]
        / features["frames_per_second"]
    )

    features["avg_signal"] = (
        features["avg_signal"].fillna(-100)
    )

    features = features.rename(columns={"wlan.ta": "mac"})

    features["timestamp"] = pd.to_datetime(
        features["second"], unit="s", utc=True
    )

    return features.sort_values(
        ["second", "mac"]
    ).reset_index(drop=True)


if not Path(MODEL_FILE).exists():
    st.error(f"Missing model: {MODEL_FILE}")
    st.stop()

if not Path(DATA_FILE).exists():
    st.error(f"Missing dataset: {DATA_FILE}")
    st.stop()

model, feature_names = load_model()
traffic = load_traffic(DATA_FILE)

if traffic.empty:
    st.error("No usable observation windows found.")
    st.stop()

# Predict with the previously trained Decision Tree.
traffic["prediction"] = model.predict(
    traffic[feature_names]
)

traffic["status"] = traffic["prediction"].map({
    0: "NORMAL",
    1: "ATTACK",
})

# Session state stores progress through the replay.
if "position" not in st.session_state:
    st.session_state.position = 0

if "running" not in st.session_state:
    st.session_state.running = False

st.sidebar.header("Replay controls")

speed = st.sidebar.slider(
    "Windows per update",
    min_value=1,
    max_value=20,
    value=3,
)

if st.sidebar.button("▶ Start / Resume"):
    if st.session_state.position >= len(traffic):
        st.session_state.position = 0
    st.session_state.running = True

if st.sidebar.button("⏸ Pause"):
    st.session_state.running = False

if st.sidebar.button("↺ Reset"):
    st.session_state.position = 0
    st.session_state.running = False

st.sidebar.divider()
st.sidebar.info(
    "This dashboard replays stored AWID3 traffic. "
    "It does not capture live Wi-Fi packets."
)

st.sidebar.write("Model: Decision Tree")
st.sidebar.write("Dataset: Deauth_30.csv")

# Only the display fragment refreshes automatically.
interval = 0.5 if st.session_state.running else None


@st.fragment(run_every=interval)
def monitor():
    if st.session_state.running:
        st.session_state.position = min(
            len(traffic),
            st.session_state.position + speed
        )

        if st.session_state.position >= len(traffic):
            st.session_state.running = False

    position = st.session_state.position
    visible = traffic.iloc[:position]

    alerts = visible[visible["prediction"] == 1]
    normal = visible[visible["prediction"] == 0]

    total_col, normal_col, attack_col, mac_col = st.columns(4)

    total_col.metric(
        "Windows analyzed",
        len(visible)
    )

    normal_col.metric(
        "Normal windows",
        len(normal)
    )

    attack_col.metric(
        "Attack alerts",
        len(alerts)
    )

    mac_col.metric(
        "Flagged MACs",
        alerts["mac"].nunique()
    )

    st.progress(position / len(traffic))

    st.write(
        f"Processed {position} of {len(traffic)} "
        "observation windows"
    )

    if visible.empty:
        st.info("Click Start / Resume to begin the replay.")
        return

    st.subheader("Wireless traffic monitoring")

    timeline = (
        visible.groupby("timestamp")
        .agg(
            Total_frames=("frames_per_second", "sum"),
            Deauth_frames=("deauth_count", "sum"),
        )
        .sort_index()
    )

    st.line_chart(
        timeline,
        height=300,
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Recent attack alerts")

        if alerts.empty:
            st.success("No attacks predicted yet.")
        else:
            st.error(
                f"{len(alerts)} suspicious windows detected"
            )

            st.dataframe(
                alerts[[
                    "timestamp",
                    "mac",
                    "frames_per_second",
                    "deauth_count",
                ]].tail(15).sort_values(
                    "timestamp",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True,
            )

    with right:
        st.subheader("Latest analyzed windows")

        st.dataframe(
            visible[[
                "timestamp",
                "mac",
                "deauth_count",
                "status",
            ]].tail(15).sort_values(
                "timestamp",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Detection statistics")

    c1, c2 = st.columns(2)

    with c1:
        st.bar_chart(
            visible["status"].value_counts()
        )

    with c2:
        if not alerts.empty:
            st.write("Most frequently flagged MACs")
            st.bar_chart(
                alerts["mac"].value_counts().head(5)
            )

    st.subheader("Export results")

    st.download_button(
        "Download detected alerts (CSV)",
        data=alerts.to_csv(index=False),
        file_name="wids_dashboard_alerts.csv",
        mime="text/csv",
    )

    if position >= len(traffic):
        st.success("Replay completed.")
    elif st.session_state.running:
        st.caption("Replay running...")
    else:
        st.caption("Replay paused.")


monitor()
