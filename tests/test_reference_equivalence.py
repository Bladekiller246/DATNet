"""Prove our attention branches match the official implementations.

The whole ablation rests on `dual` differing from `channel_only` by the spatial
branch and nothing else. That argument is only as good as the claim that our
MDTA *is* Restormer's MDTA and our window attention *is* SwinIR's. This file
turns that claim into a numerical test against the real source in `reference/`.

Requires: python scripts/fetch_reference.py --extract
          pip install einops timm   (the reference files import them)

Run:  python tests/test_reference_equivalence.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "reference")


def load_module(path, name):
    """Import a reference file directly, without installing its whole package."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mdta_matches_restormer():
    path = os.path.join(REF, "Restormer-main", "basicsr", "models", "archs",
                        "restormer_arch.py")
    if not os.path.exists(path):
        print("[SKIP] Restormer source not found; run scripts/fetch_reference.py --extract")
        return None
    ref = load_module(path, "restormer_arch")

    from datnet.models.attention import ChannelAttention

    torch.manual_seed(0)
    dim, heads = 32, 4
    theirs = ref.Attention(dim, heads, bias=False).eval()
    ours = ChannelAttention(dim, heads, bias=False).eval()

    # map our parameter names onto theirs; the modules are the same three convs
    ours.qkv.load_state_dict(theirs.qkv.state_dict())
    ours.qkv_dw.load_state_dict(theirs.qkv_dwconv.state_dict())
    ours.proj.load_state_dict(theirs.project_out.state_dict())
    with torch.no_grad():
        ours.temperature.copy_(theirs.temperature)

    x = torch.randn(2, dim, 32, 48)
    with torch.no_grad():
        a, b = theirs(x), ours(x)
    err = (a - b).abs().max().item()
    ok = err < 1e-5
    print(f"[{'PASS' if ok else 'FAIL'}] MDTA == Restormer Attention   "
          f"max|diff| = {err:.3e}")
    return ok


def test_gdfn_matches_restormer():
    path = os.path.join(REF, "Restormer-main", "basicsr", "models", "archs",
                        "restormer_arch.py")
    if not os.path.exists(path):
        print("[SKIP] Restormer source not found")
        return None
    ref = load_module(path, "restormer_arch")

    from datnet.models.ffn import GDFN

    torch.manual_seed(0)
    dim, expansion = 32, 2.66
    theirs = ref.FeedForward(dim, expansion, bias=False).eval()
    ours = GDFN(dim, expansion, bias=False).eval()

    ours.project_in.load_state_dict(theirs.project_in.state_dict())
    ours.dwconv.load_state_dict(theirs.dwconv.state_dict())
    ours.project_out.load_state_dict(theirs.project_out.state_dict())

    x = torch.randn(2, dim, 32, 48)
    with torch.no_grad():
        a, b = theirs(x), ours(x)
    err = (a - b).abs().max().item()
    ok = err < 1e-5
    print(f"[{'PASS' if ok else 'FAIL'}] GDFN == Restormer FeedForward  "
          f"max|diff| = {err:.3e}")
    return ok


def test_window_attention_matches_swinir():
    """SwinIR's WindowAttention is token-major (B*nW, N, C); ours is NCHW.

    We therefore compare a single unshifted window: partition by hand, run
    theirs on the tokens, run ours on the equivalent feature map, and check the
    outputs agree after the layout change.
    """
    path = os.path.join(REF, "SwinIR-main", "models", "network_swinir.py")
    if not os.path.exists(path):
        print("[SKIP] SwinIR source not found")
        return None
    try:
        ref = load_module(path, "network_swinir")
    except ImportError as e:
        print(f"[SKIP] SwinIR import failed ({e}); needs `pip install timm`")
        return None

    from datnet.models.attention import WindowAttention

    torch.manual_seed(0)
    dim, heads, m = 32, 4, 8
    theirs = ref.WindowAttention(dim, (m, m), heads, qkv_bias=True).eval()
    ours = WindowAttention(dim, heads, window_size=m, shift=False).eval()

    # theirs uses Linear(dim, 3*dim); ours uses Conv2d 1x1 -- same maths,
    # weights differ only by two trailing singleton dims
    with torch.no_grad():
        ours.qkv.weight.copy_(theirs.qkv.weight.view(3 * dim, dim, 1, 1))
        ours.qkv.bias.copy_(theirs.qkv.bias)
        ours.proj.weight.copy_(theirs.proj.weight.view(dim, dim, 1, 1))
        ours.proj.bias.copy_(theirs.proj.bias)
        ours.relative_position_bias_table.copy_(theirs.relative_position_bias_table)

    x = torch.randn(1, dim, m, m)                    # exactly one window
    tokens = x.flatten(2).transpose(1, 2)            # (1, m*m, dim)
    with torch.no_grad():
        a = theirs(tokens, mask=None)                # (1, m*m, dim)
        b = ours(x)                                  # (1, dim, m, m)
    a = a.transpose(1, 2).reshape(1, dim, m, m)
    err = (a - b).abs().max().item()
    ok = err < 1e-4
    print(f"[{'PASS' if ok else 'FAIL'}] WindowAttention == SwinIR      "
          f"max|diff| = {err:.3e}")
    return ok


if __name__ == "__main__":
    results = [
        test_mdta_matches_restormer(),
        test_gdfn_matches_restormer(),
        test_window_attention_matches_swinir(),
    ]
    ran = [r for r in results if r is not None]
    print(f"\n{sum(ran)}/{len(ran)} equivalence checks passed "
          f"({len(results) - len(ran)} skipped)")
    sys.exit(0 if all(ran) else 1)
