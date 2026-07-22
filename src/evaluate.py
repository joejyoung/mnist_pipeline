"""Quality gate — the single most important production concept in this repo.

A new model does NOT ship because someone eyeballed the accuracy. It ships because
it passed explicit, automated criteria:

  1. Overall test accuracy >= incumbent's accuracy (no regression).
  2. SLICE CHECK: per-digit accuracy must not drop more than 1 point on ANY digit.
     Why: overall accuracy can improve while one class silently regresses.
     In the bank-check world: "overall +0.3% but digit-7 recall -4%" means
     misread amounts. Slices catch what averages hide.
  3. Absolute floor: accuracy >= 0.95 (never ship below a minimum bar, even if
     the incumbent was worse).

Exit code 0 = PASS (CI can proceed to deploy), 1 = FAIL (CI blocks the release).

Run:  python src/evaluate.py --run-name baseline
"""
import argparse
import json
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import db
from model import MnistCNN

ABSOLUTE_FLOOR = 0.95
MAX_SLICE_REGRESSION = 0.01  # 1 percentage point


def evaluate(model, device):
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test = datasets.MNIST("data", train=False, download=True, transform=tfm)
    loader = DataLoader(test, batch_size=512)

    model.eval()
    correct_per_digit = {d: 0 for d in range(10)}
    total_per_digit = {d: 0 for d in range(10)}
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(1).cpu()
            for digit in range(10):
                mask = y == digit
                total_per_digit[digit] += mask.sum().item()
                correct_per_digit[digit] += (preds[mask] == digit).sum().item()

    per_digit = {
        str(d): correct_per_digit[d] / total_per_digit[d] for d in range(10)
    }
    overall = sum(correct_per_digit.values()) / sum(total_per_digit.values())
    return overall, per_digit


def get_incumbent():
    """The currently promoted model's metrics (None if this is the first model)."""
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT run_name, test_accuracy, per_digit_accuracy
               FROM training_runs WHERE promoted = TRUE
               ORDER BY created_at DESC LIMIT 1"""
        )
        row = cur.fetchone()
    if not row:
        return None
    return {"run_name": row[0], "accuracy": row[1], "per_digit": row[2]}


def promote(run_name):
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute("UPDATE training_runs SET promoted = FALSE WHERE promoted = TRUE")
        cur.execute("UPDATE training_runs SET promoted = TRUE WHERE run_name = %s", (run_name,))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--weights", default="model_latest.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MnistCNN().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    overall, per_digit = evaluate(model, device)
    print(f"Candidate '{args.run_name}': overall accuracy = {overall:.4f}")
    print("Per-digit:", json.dumps({k: round(v, 4) for k, v in per_digit.items()}))

    # Persist metrics on the run row
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """UPDATE training_runs SET test_accuracy=%s, per_digit_accuracy=%s
               WHERE run_name=%s""",
            (overall, json.dumps(per_digit), args.run_name),
        )

    failures = []

    # Gate 3: absolute floor
    if overall < ABSOLUTE_FLOOR:
        failures.append(f"accuracy {overall:.4f} below absolute floor {ABSOLUTE_FLOOR}")

    incumbent = get_incumbent()
    if incumbent:
        print(f"Incumbent '{incumbent['run_name']}': accuracy = {incumbent['accuracy']:.4f}")
        # Gate 1: no overall regression
        if overall < incumbent["accuracy"]:
            failures.append(
                f"overall regression: {overall:.4f} < incumbent {incumbent['accuracy']:.4f}"
            )
        # Gate 2: slice checks
        for digit, inc_acc in incumbent["per_digit"].items():
            if per_digit[digit] < inc_acc - MAX_SLICE_REGRESSION:
                failures.append(
                    f"digit {digit} regressed: {per_digit[digit]:.4f} vs "
                    f"incumbent {inc_acc:.4f}"
                )
    else:
        print("No incumbent — first model only needs to clear the absolute floor.")

    if failures:
        print("\n❌ QUALITY GATE FAILED:")
        for f in failures:
            print(f"   - {f}")
        sys.exit(1)

    promote(args.run_name)
    print(f"\n✅ QUALITY GATE PASSED — '{args.run_name}' promoted to incumbent.")
    print("Next: python src/export_onnx.py --output models/model.onnx")


if __name__ == "__main__":
    main()
