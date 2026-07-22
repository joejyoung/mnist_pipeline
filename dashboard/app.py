"""ML observability dashboard (Streamlit).

Four panels mirroring what a production ML dashboard shows:
  1. Training curves per run       (experiment comparison)
  2. Per-digit accuracy of runs    (slice metrics — what averages hide)
  3. Live inference monitoring     (volume, confidence, latency)
  4. Input drift                   (PSI on input pixel statistics vs training baseline)

Production equivalents: Grafana panels + Evidently/Arize reports.
"""
import os

import pandas as pd
import plotly.express as px
import psycopg2
import streamlit as st

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://mnist:mnist@localhost:5432/experiments"
)

# MNIST training-set input statistics = the drift baseline.
TRAIN_MEAN_PIXEL = 0.1307
PSI_ALERT = 0.2  # common rule of thumb: <0.1 stable, 0.1–0.2 moderate, >0.2 alert

st.set_page_config(page_title="MNIST ML Ops", layout="wide")
st.title("MNIST Pipeline — ML Observability")


@st.cache_resource
def conn():
    return psycopg2.connect(DATABASE_URL)


def q(sql, params=None):
    return pd.read_sql(sql, conn(), params=params)


# ---------------- 1. Training curves ----------------
st.header("1 · Training curves")
runs = q("SELECT run_id, run_name, test_accuracy, promoted FROM training_runs ORDER BY created_at DESC")
if runs.empty:
    st.info("No runs yet — run `python src/train.py` first.")
else:
    metrics = q(
        """SELECT e.run_id, r.run_name, e.epoch, e.train_loss, e.val_loss, e.val_accuracy
           FROM epoch_metrics e JOIN training_runs r USING (run_id) ORDER BY e.epoch"""
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.line(metrics, x="epoch", y="val_loss", color="run_name",
                    title="Validation loss"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            px.line(metrics, x="epoch", y="val_accuracy", color="run_name",
                    title="Validation accuracy"),
            use_container_width=True,
        )

    # ---------------- 2. Slice metrics ----------------
    st.header("2 · Per-digit accuracy (slice metrics)")
    st.caption("Overall accuracy can improve while one digit regresses. "
               "The quality gate in evaluate.py checks exactly this table.")
    slices = q(
        """SELECT run_name, promoted, per_digit_accuracy FROM training_runs
           WHERE per_digit_accuracy IS NOT NULL"""
    )
    if slices.empty:
        st.info("Run `python src/evaluate.py --run-name <name>` to populate slice metrics.")
    else:
        rows = []
        for _, r in slices.iterrows():
            for digit, acc in r["per_digit_accuracy"].items():
                rows.append({"run": r["run_name"] + (" ★" if r["promoted"] else ""),
                             "digit": digit, "accuracy": acc})
        st.plotly_chart(
            px.bar(pd.DataFrame(rows), x="digit", y="accuracy", color="run",
                   barmode="group", range_y=[0.9, 1.0]),
            use_container_width=True,
        )

# ---------------- 3. Live inference ----------------
st.header("3 · Live inference monitoring")
preds = q("SELECT * FROM predictions ORDER BY ts DESC LIMIT 2000")
if preds.empty:
    st.info("No predictions yet — start the API and run `python tests/send_test_requests.py`.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Predictions (recent)", len(preds))
    c2.metric("Mean confidence", f"{preds.confidence.mean():.3f}")
    c3.metric("p95 latency", f"{preds.latency_ms.quantile(0.95):.1f} ms")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.histogram(preds, x="predicted", nbins=10,
                         title="Predicted class distribution"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            px.histogram(preds, x="confidence", nbins=30,
                         title="Confidence distribution (watch for left drift)"),
            use_container_width=True,
        )

    # ---------------- 4. Input drift ----------------
    st.header("4 · Input drift (PSI on mean pixel intensity)")
    st.caption(f"Baseline: training-set mean pixel = {TRAIN_MEAN_PIXEL}. "
               f"PSI > {PSI_ALERT} ⇒ alert ⇒ investigate / retrain.")

    # Simple PSI: bucket recent mean_pixel values vs a synthetic baseline
    # distribution centered on the training mean. monitoring/drift.py does the
    # authoritative version; this panel is the visual.
    import numpy as np

    recent = preds["mean_pixel"].dropna().values
    if len(recent) >= 50:
        rng = np.random.default_rng(0)
        baseline = rng.normal(TRAIN_MEAN_PIXEL, 0.02, 5000).clip(0, 1)
        bins = np.histogram_bin_edges(np.concatenate([baseline, recent]), bins=10)
        b_pct = np.histogram(baseline, bins=bins)[0] / len(baseline) + 1e-6
        r_pct = np.histogram(recent, bins=bins)[0] / len(recent) + 1e-6
        psi = float(np.sum((r_pct - b_pct) * np.log(r_pct / b_pct)))

        color = "🟢" if psi < 0.1 else ("🟡" if psi < PSI_ALERT else "🔴")
        st.metric("PSI", f"{color} {psi:.3f}")

        df = pd.concat([
            pd.DataFrame({"mean_pixel": baseline, "source": "training baseline"}),
            pd.DataFrame({"mean_pixel": recent, "source": "live inference"}),
        ])
        st.plotly_chart(
            px.histogram(df, x="mean_pixel", color="source", barmode="overlay",
                         histnorm="probability", nbins=40,
                         title="Input distribution: baseline vs live"),
            use_container_width=True,
        )
        if psi >= PSI_ALERT:
            st.error("Drift alert: live inputs no longer match training data. "
                     "In production this pages someone / triggers retraining review.")
    else:
        st.info("Need ≥50 predictions for the drift panel. "
                "Try `python monitoring/simulate_drift.py` to see an alert fire.")
