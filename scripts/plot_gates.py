"""The central figure: axis preference per block, one line per task.

If the lines separate -- channel-leaning for noise, spatial-leaning for blur --
that is the result the project turns on. If they overlap, then applying both
attentions unconditionally (as X-Restormer does) was already the right answer,
and the conditioning story has to be dropped in favour of the
all-in-one-including-SR contribution. Either way this plot is what decides it,
so it is worth running early and often.

**Plots the EFFECTIVE ratio by default**, not the nominal gate `g`. They agree
only when the branches are magnitude-balanced; on the pre-fix Phase-1 run a
block reporting g=0.803 was running an effective 0.105. Generate the effective
values first:

    python scripts/effective_gate.py --run-dir runs/<run>

Usage:
    python scripts/plot_gates.py runs/phase1_denoise__dual__seed0 \
                                 runs/phase2_deblur__dual__seed0 \
                                 --labels denoise deblur --out figures/gate_flip.png
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
# This runs from a detached overnight process with no display attached. Without
# forcing a non-interactive backend, matplotlib can pick an interactive one and
# fail at import, which would lose the figure after the run already finished.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# encoder -> bottleneck -> decoder -> refinement, which is the order the signal
# actually flows; sorting alphabetically would scramble it
STAGE_ORDER = ["enc1", "enc2", "bottleneck", "dec2", "dec1", "refine"]


def load_last(run_dir, prefer_effective=True):
    """Per-block axis preference for one run.

    Prefers the EFFECTIVE ratio |gA| / (|gA| + |(1-g)B|) written by
    `scripts/effective_gate.py`, falling back to the nominal `g` logged during
    training.

    Why the preference matters: `g` is only trustworthy when the branches are
    magnitude-balanced. On the pre-fix Phase-1 run a block reporting g=0.803 was
    running an effective 0.105 -- so plotting `g` would have plotted a number
    that did not describe the network. The figure this script produces IS the
    contribution, so it must show the measured quantity, not the nominal one.
    """
    eff = os.path.join("results", os.path.basename(run_dir.rstrip("/\\")),
                       "effective_gate.json")
    if prefer_effective and os.path.exists(eff):
        with open(eff, encoding="utf-8") as f:
            d = json.load(f)
        return {"iteration": d["iteration"],
                "gates": {k: v["effective"] for k, v in d["blocks"].items()},
                "source": "effective",
                "warn": d.get("max_abs_delta", 0.0) > 0.05}

    path = os.path.join(run_dir, "gates.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no effective_gate.json and no gates.jsonl for {run_dir}. "
            f"Run: python scripts/effective_gate.py --run-dir {run_dir}")
    last = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    last["source"] = "nominal g"
    last["warn"] = False
    return last


def sort_key(name):
    for i, stage in enumerate(STAGE_ORDER):
        if name.startswith(stage + "."):
            block = int(name.split(".")[2]) if len(name.split(".")) > 2 else 0
            return (i, block)
    return (len(STAGE_ORDER), name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="figures/gate_flip.png")
    ap.add_argument("--nominal", action="store_true",
                    help="plot the raw gate g instead of the effective ratio "
                         "(only honest when branches are magnitude-balanced)")
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(r) for r in args.runs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = None
    sources = set()
    for run, label in zip(args.runs, labels):
        rec = load_last(run, prefer_effective=not args.nominal)
        sources.add(rec["source"])
        if rec.get("warn"):
            print(f"WARNING: {run} -- nominal g and the effective ratio disagree "
                  f"by >0.05. The branches are not magnitude-balanced; g is not "
                  f"a description of this network.")
        gates = rec["gates"]
        ordered = sorted(gates, key=sort_key)
        if names is None:
            names = ordered
        elif ordered != names:
            print(f"warning: {run} has a different block layout; "
                  "the two runs are not directly comparable")
        ax.plot(range(len(ordered)), [gates[n] for n in ordered],
                marker="o", label=f"{label} (it {rec['iteration']:,})")

    ax.axhline(0.5, color="0.6", ls="--", lw=1)
    ax.text(0.01, 0.51, "equal weighting", color="0.4", fontsize=8,
            transform=ax.get_yaxis_transform())
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(".blocks", "") for n in names],
                       rotation=60, ha="right", fontsize=7)
    src = "/".join(sorted(sources)) if sources else "gate"
    ax.set_ylabel(f"axis preference [{src}]\n(1 = channel attention, 0 = spatial)")
    ax.set_xlabel("block, in signal order")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("Learned channel/spatial axis preference per task")
    if sources == {"nominal g"}:
        ax.text(0.99, 0.02, "nominal g -- verify branches are balanced",
                transform=ax.transAxes, ha="right", fontsize=7, color="0.5")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
