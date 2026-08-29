"""Download the training datasets, resumably.

DIV2K train HR alone is 3.5 GB. On a laptop that may lose mains power partway
through, a download that cannot resume is a download you may never finish, so
every transfer here uses HTTP Range requests: interrupt it, re-run the same
command, and it continues from the byte it reached.

Only DIV2K is automated. GoPro and Rain13K are distributed through Google Drive
links that change and require confirmation tokens; scripted downloads of those
break silently and hand you a corrupt archive, which is worse than doing it by
hand. See docs/DATA.md for those.

Usage:
  python scripts/fetch_data.py                     # DIV2K HR + LR X4 + valid HR
  python scripts/fetch_data.py --only DIV2K_train_HR
  python scripts/fetch_data.py --extract
"""
import argparse
import os
import sys
import time
import urllib.request
import zipfile

BASE = "http://data.vision.ee.ethz.ch/cvl/DIV2K"

SETS = {
    "DIV2K_train_HR":            (f"{BASE}/DIV2K_train_HR.zip",            "800 HR training images -- Phase 1 and Phase 3"),
    "DIV2K_train_LR_bicubic_X4": (f"{BASE}/DIV2K_train_LR_bicubic_X4.zip", "official bicubic x4 LR -- Phase 3, do not regenerate these"),
    "DIV2K_valid_HR":            (f"{BASE}/DIV2K_valid_HR.zip",            "100 validation images"),
}

UA = {"User-Agent": "datnet-fetch-data"}

# Evaluation sets. Only the CLEAN images are needed -- `TestSet` synthesises the
# noise itself from a per-index seed, so the degraded copies these repos ship are
# redundant and would risk a different noise convention than training uses.
TESTSETS = {
    "CBSD68": {
        "zip": "https://github.com/clausmichele/CBSD68-dataset/archive/refs/heads/master.zip",
        "member_prefix": "CBSD68-dataset-master/CBSD68/original_png/",
        "expect": 68,
        "why": "primary colour denoising benchmark -- Phase 1",
    },
}


def human(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def download(url, dest, chunk=1 << 20):
    """Resume `dest` from wherever it stopped, using a Range request."""
    have = os.path.getsize(dest) if os.path.exists(dest) else 0

    head = urllib.request.urlopen(
        urllib.request.Request(url, method="HEAD", headers=UA), timeout=60)
    total = int(head.headers.get("Content-Length", 0))

    if total and have == total:
        print(f"    already complete ({human(total)})")
        return True
    if have:
        print(f"    resuming at {human(have)} of {human(total)}")

    headers = dict(UA)
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req, timeout=120) as r:
        # 200 means the server ignored the Range header and restarted the file
        mode = "ab" if (have and r.status == 206) else "wb"
        if mode == "wb":
            have = 0
        t0, last = time.time(), time.time()
        with open(dest, mode) as f:
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                have += len(buf)
                if time.time() - last > 5:
                    rate = have / max(time.time() - t0, 1e-9) / 1e6
                    pct = f"{100 * have / total:5.1f}%" if total else "  ?  "
                    print(f"    {pct}  {human(have)}  {rate:.1f} MB/s", flush=True)
                    last = time.time()

    final = os.path.getsize(dest)
    if total and final != total:
        print(f"    INCOMPLETE: {human(final)} of {human(total)} -- re-run to resume")
        return False
    print(f"    done ({human(final)})")
    return True


def fetch_testset(name, spec, out_root):
    """Download a repo zip and extract only the clean-image folder from it."""
    dest_dir = os.path.join(out_root, name)
    if os.path.isdir(dest_dir) and len(os.listdir(dest_dir)) >= spec["expect"]:
        print(f"    already present ({len(os.listdir(dest_dir))} files)")
        return True

    tmp_zip = os.path.join(out_root, f"_{name}.zip")
    os.makedirs(out_root, exist_ok=True)
    if not download(spec["zip"], tmp_zip):
        return False

    os.makedirs(dest_dir, exist_ok=True)
    prefix = spec["member_prefix"]
    n = 0
    with zipfile.ZipFile(tmp_zip) as z:
        for m in z.namelist():
            if m.startswith(prefix) and not m.endswith("/"):
                target = os.path.join(dest_dir, os.path.basename(m))
                with z.open(m) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                n += 1
    os.remove(tmp_zip)
    ok = n >= spec["expect"]
    suffix = "" if ok else f"  -- EXPECTED {spec['expect']}"
    print(f"    extracted {n} images to {dest_dir}{suffix}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/DIV2K")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--testsets", action="store_true",
                    help="fetch evaluation sets into data/test/ instead of DIV2K")
    ap.add_argument("--test-out", default="data/test")
    args = ap.parse_args()

    if args.testsets:
        failed = []
        for name, spec in TESTSETS.items():
            print(f"\n{name}  -- {spec['why']}")
            try:
                if not fetch_testset(name, spec, args.test_out):
                    failed.append(name)
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")
                failed.append(name)
        print(f"\n{len(TESTSETS) - len(failed)}/{len(TESTSETS)} "
              f"test sets in {args.test_out}/")
        sys.exit(1 if failed else 0)

    os.makedirs(args.out, exist_ok=True)
    names = args.only or list(SETS)
    unknown = set(names) - set(SETS)
    if unknown:
        sys.exit(f"unknown: {sorted(unknown)}; known: {list(SETS)}")

    failed = []
    for name in names:
        url, why = SETS[name]
        dest = os.path.join(args.out, name + ".zip")
        print(f"\n{name}  -- {why}")
        try:
            if not download(url, dest):
                failed.append(name)
                continue
            if args.extract:
                print("    extracting...")
                with zipfile.ZipFile(dest) as z:
                    z.extractall(args.out)
                print("    extracted")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            failed.append(name)

    print(f"\n{len(names) - len(failed)}/{len(names)} complete in {args.out}/")
    if failed:
        print(f"incomplete: {failed} -- re-run the same command to resume")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
