"""Model packaging: PyTorch checkpoint → ONNX artifact.

Production concept: the serving container never imports your training code.
It loads a stable, framework-neutral artifact (ONNX). This decouples the
training stack from the serving stack — you can upgrade PyTorch, or serve
from a Rust/Go service, without touching the artifact contract.

Run:  python src/export_onnx.py --output models/model.onnx
"""
import argparse
import pathlib

import torch

from model import MnistCNN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="model_latest.pt")
    p.add_argument("--output", default="models/model.onnx")
    args = p.parse_args()

    model = MnistCNN()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Exported → {args.output}")
    print("Restart serving to pick it up: docker compose restart api")


if __name__ == "__main__":
    main()
