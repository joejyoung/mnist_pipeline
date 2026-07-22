"""Postgres helpers shared by training, serving, and monitoring.

Real-world note: in production the *inference* event stream usually goes through
Kafka/Kinesis into a warehouse, not straight into Postgres. Same idea, bigger pipes.
"""
import json
import os

import psycopg2

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://mnist:mnist@localhost:5432/experiments"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS training_runs (
    run_id      TEXT PRIMARY KEY,
    run_name    TEXT,
    created_at  TIMESTAMPTZ DEFAULT now(),
    params      JSONB,
    test_accuracy DOUBLE PRECISION,
    per_digit_accuracy JSONB,        -- slice metrics: {"0": 0.99, ..., "9": 0.97}
    promoted    BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS epoch_metrics (
    run_id     TEXT,
    epoch      INT,
    train_loss DOUBLE PRECISION,
    val_loss   DOUBLE PRECISION,
    val_accuracy DOUBLE PRECISION,
    PRIMARY KEY (run_id, epoch)
);

-- Inference event log: the raw material for drift detection and feedback loops
CREATE TABLE IF NOT EXISTS predictions (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ DEFAULT now(),
    predicted   INT,
    confidence  DOUBLE PRECISION,
    mean_pixel  DOUBLE PRECISION,    -- cheap input statistics for drift monitoring
    std_pixel   DOUBLE PRECISION,
    latency_ms  DOUBLE PRECISION,
    model_path  TEXT
);
"""


def get_conn():
    return psycopg2.connect(DATABASE_URL)


def init_schema():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA)


def log_epoch(run_id, epoch, train_loss, val_loss, val_accuracy):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO epoch_metrics (run_id, epoch, train_loss, val_loss, val_accuracy)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (run_id, epoch) DO UPDATE
               SET train_loss = EXCLUDED.train_loss,
                   val_loss = EXCLUDED.val_loss,
                   val_accuracy = EXCLUDED.val_accuracy""",
            (run_id, epoch, train_loss, val_loss, val_accuracy),
        )


def upsert_run(run_id, run_name, params, test_accuracy=None, per_digit=None):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO training_runs (run_id, run_name, params, test_accuracy, per_digit_accuracy)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (run_id) DO UPDATE
               SET test_accuracy = COALESCE(EXCLUDED.test_accuracy, training_runs.test_accuracy),
                   per_digit_accuracy = COALESCE(EXCLUDED.per_digit_accuracy, training_runs.per_digit_accuracy)""",
            (
                run_id,
                run_name,
                json.dumps(params),
                test_accuracy,
                json.dumps(per_digit) if per_digit else None,
            ),
        )
