import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import torch
from PIL import Image

from utils.checkpoint import load_model
from utils.dataset import FaceTransform
from utils.seed import choose_device


def main():
    parser = argparse.ArgumentParser(description="Classify a cropped face image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be in [0, 1]")
    device = choose_device(args.device)
    model, config, _ = load_model(args.checkpoint, device)
    with Image.open(args.image) as image:
        tensor = FaceTransform(config["data"])(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = model(tensor).sigmoid().item()
    print(json.dumps({"forgery_probability": probability,
                      "label": int(probability >= args.threshold),
                      "prediction": "fake" if probability >= args.threshold else "real"}, indent=2))


if __name__ == "__main__":
    main()
