"""Download the reference implementations DATNet is built against or compared to.

These are read-only references, not dependencies. Two uses:

1. **Verification.** Our channel branch is a port of Restormer's MDTA and our
   spatial branch is a port of SwinIR's shifted-window MSA. Having the official
   source on disk means "is our block actually equivalent?" is a diff, not a
   recollection.
2. **Prior art.** X-Restormer is the closest existing work and has to be read in
   full before Phase 0 is called finished.

Source zips only -- no pretrained weights. DATNet trains from scratch, so the
checkpoints these repos ship are not needed.

Usage:
    python scripts/fetch_reference.py               # all of them
    python scripts/fetch_reference.py --only Restormer SwinIR
    python scripts/fetch_reference.py --extract     # also unzip
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile

# (local name, owner/repo, default branch, why it is here)
REPOS = [
    ("Restormer",   "swz30/Restormer",            "main",
     "MDTA + GDFN - our channel branch is a direct port"),
    ("SwinIR",      "JingyunLiang/SwinIR",        "main",
     "shifted-window MSA - our spatial branch"),
    ("NAFNet",      "megvii-research/NAFNet",     "main",
     "the no-attention control baseline"),
    ("Uformer",     "ZhendongWang6/Uformer",      "main",
     "window attention inside a U-Net"),
    ("X-Restormer", "Andrew0613/X-Restormer",     "master",
     "closest prior art - read this first"),
    ("PromptIR",    "va1shn9v/PromptIR",          "main",
     "Phase 5 all-in-one competitor"),
    ("AirNet",      "XLearning-SCU/2022-CVPR-AirNet", "main",
     "alternative conditioning approach"),
    ("DAT",         "zhengchen1999/DAT",          "main",
     "Dual Aggregation Transformer - the DAT name clash, Appendix A"),
]

UA = {"User-Agent": "datnet-fetch-reference"}


def resolve_branch(repo, fallback):
    """Ask the API rather than assuming main vs master."""
    try:
        req = urllib.request.Request(f"https://api.github.com/repos/{repo}",
                                     headers=UA)
        return json.load(urllib.request.urlopen(req, timeout=20))["default_branch"]
    except Exception:
        return fallback


def download(name, repo, branch, out_dir):
    branch = resolve_branch(repo, branch)
    url = f"https://github.com/{repo}/archive/refs/heads/{branch}.zip"
    dest = os.path.join(out_dir, f"{name}.zip")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  {name:12s} already present ({os.path.getsize(dest)/1e6:.1f} MB)")
        return dest
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    print(f"  {name:12s} {total/1e6:6.1f} MB  <- {repo}@{branch}")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reference")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--extract", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    wanted = [r for r in REPOS if args.only is None or r[0] in args.only]
    if args.only:
        unknown = set(args.only) - {r[0] for r in REPOS}
        if unknown:
            sys.exit(f"unknown names: {sorted(unknown)}; "
                     f"known: {[r[0] for r in REPOS]}")

    failed = []
    for name, repo, branch, why in wanted:
        try:
            path = download(name, repo, branch, args.out)
            if args.extract:
                with zipfile.ZipFile(path) as z:
                    z.extractall(args.out)
        except urllib.error.HTTPError as e:
            failed.append((name, f"HTTP {e.code}"))
            print(f"  {name:12s} FAILED: HTTP {e.code}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  {name:12s} FAILED: {type(e).__name__}: {e}")

    print(f"\n{len(wanted) - len(failed)}/{len(wanted)} downloaded into {args.out}/")
    for name, err in failed:
        print(f"  failed: {name} ({err})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
