from .deploy import convert_to_deploy
from .stlnet import STLNet

__all__ = ["STLNet", "build_model", "convert_to_deploy"]


def build_model(config=None):
    if config is None:
        return STLNet()
    model_config = dict(config.get("model", config))
    if model_config.pop("ablation", "full") != "full":
        raise ValueError("Only the full STLNet model is supported")
    return STLNet(**model_config)
