"""Training pipeline step.

Production concepts demonstrated:
  * Every run has an ID tying together code, params, and metrics (reproducibility).
  * Metrics go to TWO places: MLflow (experiment tracking / registry) and
    Postgres (powers the live dashboard) — mirroring how real systems fan out
    to a tracking server and an observability store.
  * Determinism: seeds are set and logged. "Which exact run produced prod model v12?"
    must always be answerable.

Run:  python src/train.py --epochs 3 --lr 0.001 --run-name baseline
"""
import argparse
import time
import uuid

import mlflow
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import db
from model import MnistCNN

MLFLOW_URI = "http://localhost:5000"


def get_loaders(batch_size):
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    full_train = datasets.MNIST("data", train=True, download=True, transform=tfm)
    # Hold out a validation split — the test set stays untouched until evaluate.py.
    train_set, val_set = random_split(
        full_train, [55000, 5000], generator=torch.Generator().manual_seed(42)
    )
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_set, batch_size=512),
    )


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--run-name", default=None)
    args = p.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.run_name or f"run-{uuid.uuid4().hex[:8]}"
    params = {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
              "seed": 42, "device": device}

    db.init_schema()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("mnist")

    train_loader, val_loader = get_loaders(args.batch_size)
    model = MnistCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.log_params(params)
        db.upsert_run(run_id, run_name, params)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, _ = run_epoch(model, train_loader, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, device)

            print(f"epoch {epoch}: train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                  f"({time.time()-t0:.1f}s)")

            # Fan out: tracking server + observability store
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc},
                step=epoch,
            )
            db.log_epoch(run_id, epoch, train_loss, val_loss, val_acc)

        # Log the model artifact to the registry. In production, promotion to
        # "Staging"/"Production" stages happens via evaluate.py's quality gate.
        mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")
        torch.save(model.state_dict(), "model_latest.pt")  # local handoff for eval/export

    print(f"\nDone. run_id={run_id}  →  MLflow: {MLFLOW_URI}  Dashboard: http://localhost:8501")


if __name__ == "__main__":
    main()
