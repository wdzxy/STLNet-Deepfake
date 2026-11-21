import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from engine.trainer import train
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train STLNet")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume")
    args = parser.parse_args()
    result = train(load_config(args.config), args.train_csv, args.val_csv,
                   args.output, args.device, args.resume)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
