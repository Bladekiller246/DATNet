"""Append-only run ledger.

One JSON line per finished segment. This is what turns "train for 200k
iterations" into a schedule you can actually plan a laptop around: it records
measured it/s and peak VRAM, so remaining wall-clock is arithmetic rather than
guesswork.
"""
import json
import os
import time


class Ledger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def append(self, record):
        record = {"ts": time.time(), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def read(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def summary(self):
        rows = [r for r in self.read() if r.get("kind") == "segment_end"]
        if not rows:
            return {}
        done = rows[-1]["iteration"]
        total = rows[-1]["total_iters"]
        secs = sum(r["seconds"] for r in rows)
        iters = sum(r["iters_done"] for r in rows)
        ips = iters / secs if secs else 0.0
        return {
            "iteration": done,
            "total_iters": total,
            "percent": round(100 * done / total, 1) if total else 0.0,
            "mean_it_per_s": round(ips, 2),
            "elapsed_hours": round(secs / 3600, 2),
            "eta_hours": round((total - done) / ips / 3600, 2) if ips else None,
            "peak_vram_gb": max(r.get("peak_vram_gb", 0) for r in rows),
        }
