"""Send 200 clean MNIST test images to the running API.

Purpose: populate the prediction log (dashboard panel 3) and establish a
'normal traffic' window for the drift monitor — plus it doubles as a smoke
test that the whole serve path works end to end.
"""
import numpy as np
import requests
from torchvision import datasets, transforms

API = "http://localhost:8000/predict"


def main():
    test = datasets.MNIST("data", train=False, download=True,
                          transform=transforms.ToTensor())
    rng = np.random.default_rng(0)
    idx = rng.choice(len(test), 200, replace=False)

    correct, sent = 0, 0
    for i in idx:
        img, label = test[int(i)]
        r = requests.post(API, json={"pixels": img.numpy().flatten().tolist()},
                          timeout=5)
        r.raise_for_status()
        body = r.json()
        correct += int(body["digit"] == label)
        sent += 1

    print(f"Sent {sent} images | live accuracy {correct/sent:.3f}")
    print("Dashboard: http://localhost:8501 (panel 3 should now have data)")


if __name__ == "__main__":
    main()
