"""Parameter matching across ablation arms.

Removing an attention branch removes parameters, so `channel_only` and
`window_only` must be grown until their parameter counts match `dual`. Depth is
never touched -- changing block counts would confound "which axis" with "how
many blocks", and isolating the former is the entire point of the ablation.

Two knobs, applied in order:

1. **width** -- the coarse knob. Must stay even, because the pixel-unshuffle
   downsample halves channel counts, so the grid step floor is 2.
2. **ffn_expansion (gamma)** -- the fine knob. Parameters grow quadratically in
   width, so one step of the width grid is worth ~2*P/w parameters: about 6% of
   the target at width 56. Width alone therefore cannot land inside 2% at larger
   sizes. Gamma scales the GDFN hidden width, which is where most parameters
   live, and moves the count in much smaller increments.

Gamma differing slightly between arms is the same *kind* of adjustment as width
differing between arms: it changes capacity, not structure. The GDFN is
identical in form in every arm. Report the exact gamma alongside the exact
parameter count in every table.
"""
from copy import deepcopy

from .complexity import count_parameters

DEFAULT_EXPANSION = 2.66


def _build(builder, cfg, mode, width, expansion=None):
    cfg = deepcopy(cfg)
    cfg["mode"] = mode
    cfg["width"] = width
    if expansion is not None:
        cfg["ffn_expansion"] = expansion
    return builder(cfg)


def _count(builder, cfg, mode, width, expansion=None):
    return count_parameters(_build(builder, cfg, mode, width, expansion))


def match_width(builder, cfg, target_params, mode, lo=8, hi=256, step=2, tol=0.02):
    """Closest width on the grid. Returns (width, params, rel_error).

    Parameter count is monotonically increasing in width, so a binary search is
    exact. Does NOT raise when it cannot reach `tol` -- the caller decides
    whether to refine with gamma. `tol` is reported, not enforced.
    """
    grid = list(range(lo, hi + 1, step))
    best = None
    a, b = 0, len(grid) - 1
    while a <= b:
        mid = (a + b) // 2
        w = grid[mid]
        p = _count(builder, cfg, mode, w)
        err = abs(p - target_params) / target_params
        if best is None or err < best[2]:
            best = (w, p, err)
        if p < target_params:
            a = mid + 1
        else:
            b = mid - 1
    return best


def refine_expansion(builder, cfg, target_params, mode, width,
                     lo=1.0, hi=6.0, iters=40):
    """Bisect gamma at a fixed width to close the residual gap.

    Parameter count is monotonically non-decreasing in gamma (the GDFN hidden
    width is `int(dim * gamma)`, so it is a staircase, but never decreasing).
    Bisection converges to the closest reachable step. Returns
    (expansion, params, rel_error).
    """
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        p = _count(builder, cfg, mode, width, mid)
        err = abs(p - target_params) / target_params
        if best is None or err < best[2]:
            best = (round(mid, 4), p, err)
        if p < target_params:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4:
            break
    return best


def match_arm(builder, cfg, target_params, mode, tol=0.02, **kw):
    """Match one arm to `target_params` using width, then gamma if needed.

    Returns {'width', 'ffn_expansion', 'params', 'rel_error', 'knob'}.
    Raises only if even gamma refinement cannot reach `tol`.
    """
    width, params, err = match_width(builder, cfg, target_params, mode, tol=tol, **kw)
    base_gamma = cfg.get("ffn_expansion", DEFAULT_EXPANSION)
    if err <= tol:
        return {"width": width, "ffn_expansion": base_gamma, "params": params,
                "rel_error": err, "knob": "width"}

    gamma, params_g, err_g = refine_expansion(builder, cfg, target_params,
                                              mode, width)
    if err_g > tol:
        raise ValueError(
            f"cannot match {mode} to {target_params:,} params within {tol:.0%}. "
            f"Best by width alone: width={width}, {params:,} ({err:.2%}). "
            f"Best after gamma refinement: gamma={gamma}, {params_g:,} "
            f"({err_g:.2%}). Loosen tol or widen the search range."
        )
    return {"width": width, "ffn_expansion": gamma, "params": params_g,
            "rel_error": err_g, "knob": "width+gamma"}


def match_all(builder, cfg, reference_mode="dual", tol=0.02, **kw):
    """Returns {mode: {'width', 'ffn_expansion', 'params', 'rel_error', 'knob'}}."""
    target = count_parameters(_build(builder, cfg, reference_mode, cfg["width"]))
    base_gamma = cfg.get("ffn_expansion", DEFAULT_EXPANSION)
    out = {reference_mode: {"width": cfg["width"], "ffn_expansion": base_gamma,
                            "params": target, "rel_error": 0.0, "knob": "reference"}}
    for mode in ("channel_only", "window_only", "dual"):
        if mode == reference_mode:
            continue
        out[mode] = match_arm(builder, cfg, target, mode, tol=tol, **kw)
    return out
