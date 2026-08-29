"""Config -> model. Single entry point used by training, eval and width matching."""
from copy import deepcopy

from .conditioning import ConditionedDATNet, build_encoder
from .datnet import DATNet

_MODEL_KEYS = {
    "in_ch", "out_ch", "width", "topology", "enc_depths", "bottleneck_depth",
    "dec_depths", "refine_depth", "flat_depth", "heads", "mode", "window_size",
    "ffn_expansion", "gate_init", "cond_dim", "bias", "task", "scale", "grad_ckpt",
    "normalise_branches",
}


def build_model(cfg):
    cfg = deepcopy(cfg)
    cond_cfg = cfg.pop("conditioning", None)

    if cond_cfg:
        num_tasks = cond_cfg["num_tasks"]
        cfg["cond_dim"] = num_tasks

    kwargs = {k: v for k, v in cfg.items() if k in _MODEL_KEYS}
    unknown = set(cfg) - _MODEL_KEYS - {"conditioning"}
    if unknown:
        raise ValueError(f"unknown model config keys: {sorted(unknown)}")

    for key in ("enc_depths", "dec_depths", "heads", "gate_init"):
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = tuple(kwargs[key])

    backbone = DATNet(**kwargs)
    if not cond_cfg:
        return backbone

    encoder = build_encoder(
        cond_cfg.get("kind", "oracle"),
        cond_cfg["num_tasks"],
        **{k: v for k, v in cond_cfg.items() if k not in ("kind", "num_tasks")},
    )
    return ConditionedDATNet(backbone, encoder)
