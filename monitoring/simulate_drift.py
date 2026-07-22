"""Simulate real-world drift so you can watch the monitor catch it.

Scenario: "the bank installed new scanners" — images come in brighter and
lower-contrast than the training data. The model still returns answers
(models always return answers!), but confidence sags and inputs shift.

Sends 200 distorted MNIST test images to the API, then tells you how to see
the effect. Compare dashboard panel 4 before/after, or run monitoring/drift.py.
"""
import numpy as np
import requests
from torchvision import datasets, transforms

API = "http://localhost:8000/predict"


def main():
    test = datasets.MNIST("data", train=False, download=True,
                          transform=transforms.ToTensor())
    rng = np.random.default_rng(7)
    idx = rng.choice(len(test), 200, replace=False)

    sent, conf_sum = 0, 0.0
    for i in idx:
        img, _ = test[int(i)]
        pixels = img.numpy().flatten()

        # "New scanner": brighter background + washed-out contrast + noise
        pixels = pixels * 0.6 + 0.25
        pixels = np.clip(pixels + rng.normal(0, 0.05, pixels.shape), 0, 1)

        r = requests.post(API, json={"pixels": pixels.tolist()}, timeout=5)
        r.raise_for_status()
        conf_sum += r.json()["confidence"]
        sent += 1

    print(f"Sent {sent} drifted images. Mean confidence: {conf_sum/sent:.3f}")
    print("Now: python monitoring/drift.py   (expect a PSI alert)")
    print("And check dashboard panel 4: http://localhost:8501")


if __name__ == "__main__":
    main()
