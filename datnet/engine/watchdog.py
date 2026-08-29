"""In-process training watchdogs.

Soft-alert detectors that run alongside the hard divergence guard in
trainer.py (DivergedError / non-finite loss). A non-finite loss is
unambiguous and worth aborting for; these are not -- a single loss spike or
an early gate stall is not proof of failure the way NaN is. They log loudly
to the ledger instead of aborting, so a human reading the run in the morning
sees the warning without the run being killed over noise.

Each watchdog is a small stateful observer: call `.observe(iteration, ...)`
every time new data is available, and it returns an alert dict the moment it
trips, or None otherwise.
"""
import statistics
from collections import deque


class LossSpikeWatchdog:
    """Flags a loss sustained far above its recent rolling median.

    The gradient/loss NaN that destroyed a 60k run (see trainer.py) was only
    ever checked for finiteness, never for trend -- nothing was watching the
    loss climb before it broke. This applies the same idea earlier: a loss
    that is still finite but has jumped `spike_factor`x above its own recent
    median for `min_streak` consecutive observations is worth a loud note
    well before it might go non-finite.
    """

    def __init__(self, window=50, spike_factor=5.0, min_streak=3):
        self.window = window
        self.spike_factor = spike_factor
        self.min_streak = min_streak
        self._history = deque(maxlen=window)
        self._streak = 0

    def observe(self, iteration, loss):
        """`loss` is the finite scalar total loss for this log point.

        Returns an alert dict the moment a sustained spike is confirmed
        (edge-triggered -- fires once per spike, not once per iteration
        while the spike continues), else None.
        """
        if loss is None or loss != loss:  # NaN is the hard guard's job
            return None
        alert = None
        if len(self._history) >= max(self.window // 2, 5):
            median = statistics.median(self._history)
            if median > 0 and loss > median * self.spike_factor:
                self._streak += 1
                if self._streak == self.min_streak:
                    alert = {
                        "watchdog": "loss_spike", "iteration": iteration,
                        "loss": round(loss, 5),
                        "rolling_median": round(median, 5),
                        "spike_factor": self.spike_factor,
                        "streak": self._streak,
                    }
            else:
                self._streak = 0
        self._history.append(loss)
        return alert


class GateStallWatchdog:
    """Flags per-block axis gates that have not moved since warmup.

    The dual-attention gate IS the hypothesis (CONTEXT.md sec 4f/5): Phase 1
    v1 ran 30,000 iterations with the gate moving 0.5375 -> 0.5429 and nobody
    noticed until the post-hoc effective-gate check, by which point the run
    was already the reported result. This watches the same quantity live so
    a stall is a same-morning finding, not a post-run one. It does not
    replace scripts/effective_gate.py's rescaled-effective-g check -- a gate
    can move in raw terms while its *effective* blend stays frozen (the
    decorative-gate bug); this only catches the raw value never moving at
    all.
    """

    def __init__(self, warmup_iters=5000, min_move=0.02):
        self.warmup_iters = warmup_iters
        self.min_move = min_move
        self._baseline = None
        self._tripped = False

    def observe(self, iteration, gates):
        """`gates`: {block_name: float gate value in [0, 1]}.

        Returns an alert dict the first time it confirms a stall past
        warmup, else None. Trips at most once per Trainer instance (i.e.
        once per segment process).
        """
        if not gates or iteration < self.warmup_iters or self._tripped:
            return None
        if self._baseline is None:
            self._baseline = dict(gates)
            return None
        moved = {k: abs(v - self._baseline.get(k, v)) for k, v in gates.items()}
        largest = max(moved.values())
        if largest < self.min_move:
            self._tripped = True
            return {
                "watchdog": "gate_stall", "iteration": iteration,
                "largest_move": round(largest, 5),
                "min_move_threshold": self.min_move,
                "n_gates": len(gates),
                "stalled_block": max(moved, key=moved.get),
            }
        return None
