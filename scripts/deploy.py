import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import torch

from models.deploy import convert_to_deploy
from utils.checkpoint import load_model


def main():
    parser = argparse.ArgumentParser(description="Fuse RepClassifier branches")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.resolve() == Path(args.checkpoint).resolve():
        parser.error("output must differ from the training checkpoint")
    model, config, payload = load_model(args.checkpoint)
    model = convert_to_deploy(model, inplace=True)
    config["model"] = model.model_config
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "config": config, "model": model.state_dict(),
                "source_epoch": payload.get("epoch"), "deployment": True}, target)
    print(json.dumps({"checkpoint": str(target)}))


if __name__ == "__main__":
    main()
