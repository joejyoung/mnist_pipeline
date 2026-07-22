"""Input drift detection via PSI (Population Stability Index).

The core production question: "Do the inputs the model sees TODAY still look like
the data it was TRAINED on?" Models don't announce when the world changes —
a new scanner, a preprocessing change upstream, thicker pens — they just quietly
get worse. Drift monitors are the smoke detector.

We track a cheap proxy: the mean pixel intensity of each incoming image, logged by
the serving layer. Real systems track many features and embeddings (Evidently,
Arize), but PSI over one statistic teaches the exact mechanism.

PSI = Σ (live% − baseline%) · ln(live% / baseline%)  over histogram buckets
Rule of thumb: < 0.1 stable · 0.1–0.2 moderate shift · > 0.2 ALERT

Run:  python monitoring/drift.py
Exit code 1 on alert — so CI/cron can page or open a ticket automatically.
"""
import sys

import numpy as np
import psycopg2
import os

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://mnist:mnist@localhost:5432/experiments"
)
TRAIN_MEAN_PIXEL = 0.1307   # baseline statistic from the training data
PSI_ALERT = 0.2
WINDOW = 500                # most recent N predictions


def psi(baseline: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    edges = np.histogram_bin_edges(np.concatenate([baseline, live]), bins=bins)
    b = np.histogram(baseline, bins=edges)[0] / len(baseline) + 1e-6
    l = np.histogram(live, bins=edges)[0] / len(live) + 1e-6
    return float(np.sum((l - b) * np.log(l / b)))


def main():
    with psycopg2.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT mean_pixel FROM predictions ORDER BY ts DESC LIMIT %s", (WINDOW,)
        )
        live = np.array([r[0] for r in cur.fetchall() if r[0] is not None])

    if len(live) < 50:
        print(f"Only {len(live)} predictions logged — need ≥50. "
              "Run tests/send_test_requests.py first.")
        return

    # Baseline distribution reconstructed from the training statistic.
    # Production: persist the actual training histogram at training time.
    rng = np.random.default_rng(0)
    baseline = rng.normal(TRAIN_MEAN_PIXEL, 0.02, 5000).clip(0, 1)

    score = psi(baseline, live)
    print(f"Live window: {len(live)} predictions | "
          f"live mean_pixel = {live.mean():.4f} vs baseline {TRAIN_MEAN_PIXEL}")
    print(f"PSI = {score:.3f}   (<0.1 stable, 0.1–0.2 moderate, >{PSI_ALERT} alert)")

    if score > PSI_ALERT:
        print("\n🔴 DRIFT ALERT — live inputs no longer match training distribution.")
        print("   Production response: page on-call → inspect upstream data →")
        print("   sample & label recent inputs → retrain if confirmed.")
        sys.exit(1)
    print("\n🟢 No significant drift.")


if __name__ == "__main__":
    main()
