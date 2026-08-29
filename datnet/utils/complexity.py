"""Parameter and FLOP accounting -- required for every table in the paper."""
import torch


def count_parameters(model, trainable_only=True):
    ps = model.parameters()
    if trainable_only:
        ps = (p for p in ps if p.requires_grad)
    return sum(p.numel() for p in ps)


def count_flops(model, input_shape=(1, 3, 256, 256), device="cpu", cond=None):
    """MACs-equivalent FLOPs via PyTorch's built-in FlopCounterMode.

    Reported at 256x256 by convention. Returns None if the counter is unavailable
    so callers can degrade to params-only tables rather than crash.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None

    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(x, cond) if cond is not None else model(x)
    return counter.get_total_flops()


def summarise(model, input_shape=(1, 3, 256, 256), device="cpu"):
    flops = count_flops(model, input_shape, device)
    return {
        "params": count_parameters(model),
        "params_M": round(count_parameters(model) / 1e6, 3),
        "flops": flops,
        "flops_G": round(flops / 1e9, 2) if flops else None,
        "flops_shape": list(input_shape),
    }
