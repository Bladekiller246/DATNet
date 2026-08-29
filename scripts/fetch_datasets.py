"""Fetch the remaining evaluation and paired-training datasets.

Split by mechanism, because the reliable source differs per dataset:

* **direct**   -- a plain URL (Kodak24 individual PNGs, HF tarballs)
* **hf**       -- a HuggingFace dataset repo pulled with `snapshot_download`,
                  which handles directory trees, resume and retries properly.
                  Hand-rolling URLs for these is how you end up with a truncated
                  archive that looks fine until training reads garbage.

Everything resumes. Interrupt freely and re-run the same command.

Priorities, smallest and most urgent first:
    evalsets  ~170 MB   Kodak24, Set14, BSD100, Urban100   -- Phase 1 + 3 eval
    gopro     ~6.5 GB   GoPro blur/sharp pairs             -- Phase 2
    rain13k   ~1 GB     Rain13K pairs                      -- Phase 4

Usage:
    python scripts/fetch_datasets.py --groups evalsets
    python scripts/fetch_datasets.py --groups evalsets gopro rain13k
    python scripts/fetch_datasets.py --list
"""
import argparse
import os
import shutil
import sys
import tarfile
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UA = {"User-Agent": "datnet-fetch-datasets"}

KODAK = [f"https://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png" for i in range(1, 25)]

HF_TARBALLS = {
    "Set14":    "https://huggingface.co/datasets/eugenesiow/Set14/resolve/main/data/Set14_HR.tar.gz",
    "BSD100":   "https://huggingface.co/datasets/eugenesiow/BSD100/resolve/main/data/BSD100_HR.tar.gz",
    "Urban100": "https://huggingface.co/datasets/eugenesiow/Urban100/resolve/main/data/Urban100_HR.tar.gz",
}

HF_REPOS = {
    "gopro":   ("HanzhouLiu/GoPro_Deblur", "data/GoPro_raw",
                "GoPro blur/sharp pairs -- Phase 2 deblurring (~6.5 GB)"),
    "rain13k": ("dronefreak/Rain13K", "data/Rain13K_raw",
                "Rain13K pairs -- Phase 4 deraining (~1 GB)"),
}


def human(n):
    return f"{n/1e9:.2f} GB" if n >= 1e9 else f"{n/1e6:.0f} MB"


def download(url, dest, chunk=1 << 20, quiet=False):
    """Resumable GET via HTTP Range."""
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    try:
        head = urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD", headers=UA), timeout=60)
        total = int(head.headers.get("Content-Length", 0))
    except Exception:
        total = 0
    if total and have == total:
        if not quiet:
            print(f"    already complete ({human(total)})")
        return True

    headers = dict(UA)
    if have:
        headers["Range"] = f"bytes={have}-"
    with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=180) as r:
        mode = "ab" if (have and r.status == 206) else "wb"
        if mode == "wb":
            have = 0
        t0 = last = time.time()
        with open(dest, mode) as f:
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                have += len(buf)
                if not quiet and time.time() - last > 5:
                    rate = have / max(time.time() - t0, 1e-9) / 1e6
                    pct = f"{100*have/total:5.1f}%" if total else "  ?  "
                    print(f"    {pct}  {human(have)}  {rate:.1f} MB/s", flush=True)
                    last = time.time()
    return not total or os.path.getsize(dest) == total


def fetch_kodak(out_root):
    d = os.path.join(out_root, "Kodak24")
    os.makedirs(d, exist_ok=True)
    if len(os.listdir(d)) >= 24:
        print(f"    already present ({len(os.listdir(d))} images)")
        return True
    for i, url in enumerate(KODAK, 1):
        dest = os.path.join(d, f"kodim{i:02d}.png")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        download(url, dest, quiet=True)
    n = len(os.listdir(d))
    print(f"    {n}/24 images -> {d}")
    return n >= 24


def fetch_tarball(name, url, out_root):
    d = os.path.join(out_root, name)
    if os.path.isdir(d) and len(os.listdir(d)) > 5:
        print(f"    already present ({len(os.listdir(d))} files)")
        return True
    tmp = os.path.join(out_root, f"_{name}.tar.gz")
    os.makedirs(out_root, exist_ok=True)
    if not download(url, tmp):
        return False
    os.makedirs(d, exist_ok=True)
    with tarfile.open(tmp) as t:
        for m in t.getmembers():
            if m.isfile():
                m.name = os.path.basename(m.name)   # flatten
                t.extract(m, d)
    os.remove(tmp)
    n = len(os.listdir(d))
    print(f"    {n} images -> {d}")
    return n > 0


def fetch_hf(repo, out_dir, why):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("    huggingface_hub not installed: pip install huggingface_hub")
        return False
    print(f"    {why}")
    os.makedirs(out_dir, exist_ok=True)
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=out_dir,
                      max_workers=4)
    print(f"    -> {out_dir}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=["evalsets"],
                    choices=["evalsets", "gopro", "rain13k", "all"])
    ap.add_argument("--test-out", default="data/test")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("evalsets  Kodak24(24) Set14(14) BSD100(100) Urban100(100)  ~170 MB")
        for k, (r, o, w) in HF_REPOS.items():
            print(f"{k:9s} {r:32s} -> {o}\n          {w}")
        return

    groups = ["evalsets", "gopro", "rain13k"] if "all" in args.groups else args.groups
    failed = []

    if "evalsets" in groups:
        print("\nKodak24  -- colour denoising benchmark (Phase 1)")
        if not fetch_kodak(args.test_out):
            failed.append("Kodak24")
        for name, url in HF_TARBALLS.items():
            print(f"\n{name}  -- benchmark (Phase 1 / Phase 3 eval)")
            try:
                if not fetch_tarball(name, url, args.test_out):
                    failed.append(name)
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {str(e)[:90]}")
                failed.append(name)

    for g in ("gopro", "rain13k"):
        if g in groups:
            repo, out, why = HF_REPOS[g]
            print(f"\n{g}  -- {repo}")
            try:
                if not fetch_hf(repo, out, why):
                    failed.append(g)
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {str(e)[:120]}")
                failed.append(g)

    print(f"\n{'FAILED: ' + ', '.join(failed) if failed else 'all requested groups complete'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
