import random

import numpy as np
import torch

from models import build_model


def load_model(checkpoint, device="cpu"):
    # weights_only avoids arbitrary pickle execution; saves use tensors and basic types.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    config = dict(payload["config"])
    # v0.1/v0.2 checkpoints predate configurable stem strides and used two s2 layers.
    config["model"] = dict(config["model"])
    config["model"].setdefault("stem_strides", [2, 2])
    model = build_model(config["model"])
    model.load_state_dict(payload["model"], strict=True)
    config["model"] = model.model_config
    return model.to(device).eval(), config, payload


def _checkpoint(model, config, optimizer, epoch, best_auc, bad_epochs, generator):
    np_state = np.random.get_state()
    return {"format_version": 1, "model": model.state_dict(), "config": config,
            "optimizer": optimizer.state_dict(), "epoch": epoch, "best_auc": best_auc,
            "bad_epochs": bad_epochs, "scheduler": {"next_epoch": epoch + 1, "power": 0.9},
            "rng": {"python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(),
                     np_state[2], np_state[3], np_state[4]], "torch": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                     "loader": generator.get_state()}}


def _restore_rng(state, generator):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the same number of CUDA devices")
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])
