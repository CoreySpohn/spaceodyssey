#!/usr/bin/env python3
"""Per-step wall time of the compiled executor, and its cold first call.

Runs the toy baseline study through the compiled executor's scan (`_scan_worlds`, the
path every study without a host service or refresh schedule takes). The first call is
timed on its own (trace, compile and run, with the persistent compilation cache off);
then the compiled program is called `--repeats` more times, each synchronized with
`block_until_ready`, and the median wall time over `--n-steps` gives the per-step
cost. Under `vmap` the handler switch evaluates every branch, so this cost grows with
the handler table.

`--mutant` adds one extra handler pass per step: after each transition the next row
is selected and handled again, and the result is kept only under a condition the
compiler cannot prove false (it never holds). The ratio of its per-step time to the
plain one calibrates the step-cost check: the mutant must exceed its threshold.

With `--baseline-per-step-us` (the earlier table's per-step minimum, measured in the
same job) the plain per-step minimum is compared with it, and the exit status is 1
when the ratio exceeds `--limit` (1.10).

Prints one JSON object. Run from the repository root with the package importable.
"""

import argparse
import json
import platform
import statistics
import time

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_compilation_cache", False)

from spaceodyssey.core.conformance.corpus import make_worlds  # noqa: E402
from spaceodyssey.core.events import select_next, where_tree  # noqa: E402
from spaceodyssey.core.executors.compiled import (  # noqa: E402
    CompiledExecutor,
    _scan_worlds,
)
from spaceodyssey.core.handlers import HANDLERS, step  # noqa: E402
from spaceodyssey.core.transaction import transition, wake_by_audience  # noqa: E402
from spaceodyssey.studies import toy  # noqa: E402

NEVER = -(2**40)  # no tick reaches it; the compiler cannot know that


class ExtraPassExecutor(CompiledExecutor):
    """The compiled executor with one extra select-and-handle pass per step."""

    def transition(self, carry, i, root, study, truth, *, wake_fn=wake_by_audience):
        """Advance one transaction, then handle the next row into a discarded carry."""
        c2, trace = transition(carry, i, root, study, truth, wake_fn=wake_fn)
        c3, ev, _ = select_next(c2, study.caps)
        c4, _, _ = step(c3, ev, study, truth, root)
        return where_tree(c2.tick < NEVER, c4, c2), trace


K3_LIMIT = 1.10  # the final table may cost at most 10 percent more per step


def verdict(per_step_min_us, baseline_us, limit=K3_LIMIT):
    """Ratio of a per-step minimum to its baseline, and whether it is within `limit`."""
    ratio = per_step_min_us / baseline_us
    return {"ratio": round(ratio, 3), "limit": limit, "within_limit": ratio <= limit}


def _call(executor, study, truths, roots, n_steps):
    out = _scan_worlds(executor, study, truths, roots, wake_by_audience, n_steps)
    return jax.block_until_ready(out)


def measure(executor, n_worlds, n_steps, repeats):
    """Cold first call and warm per-step times, in seconds, of one executor."""
    study = toy.survey_1()
    truths, roots = make_worlds(study, n_worlds)
    t0 = time.perf_counter()
    _call(executor, study, truths, roots, n_steps)
    cold = time.perf_counter() - t0
    warm = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _call(executor, study, truths, roots, n_steps)
        warm.append(time.perf_counter() - t0)
    median = statistics.median(warm)
    return {
        "cold_s": round(cold, 3),
        "warm_median_s": round(median, 6),
        "warm_min_s": round(min(warm), 6),
        "warm_max_s": round(max(warm), 6),
        "per_step_us": round(1e6 * median / n_steps, 2),
        "per_step_min_us": round(1e6 * min(warm) / n_steps, 2),
    }


def main(argv=None):
    """Measure and print the step cost as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-worlds", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--mutant", action="store_true")
    parser.add_argument(
        "--baseline-per-step-us",
        type=float,
        help="per-step minimum of the earlier handler table, measured in the same "
        "job; exit 1 if the plain run exceeds it by more than the limit",
    )
    parser.add_argument("--limit", type=float, default=K3_LIMIT)
    args = parser.parse_args(argv)
    out = {
        "handlers": len(HANDLERS),
        "study": "toy survey_1",
        "n_worlds": args.n_worlds,
        "n_steps": args.n_steps,
        "repeats": args.repeats,
        "jax": jax.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "plain": measure(CompiledExecutor(), args.n_worlds, args.n_steps, args.repeats),
    }
    if args.mutant:
        out["mutant"] = measure(
            ExtraPassExecutor(), args.n_worlds, args.n_steps, args.repeats
        )
        out["mutant_ratio"] = round(
            out["mutant"]["per_step_us"] / out["plain"]["per_step_us"], 3
        )
    failed = False
    if args.baseline_per_step_us is not None:
        base = args.baseline_per_step_us
        out["baseline_per_step_min_us"] = base
        out["k3"] = verdict(out["plain"]["per_step_min_us"], base, args.limit)
        failed = not out["k3"]["within_limit"]
        if args.mutant:
            out["k3_mutant"] = verdict(
                out["mutant"]["per_step_min_us"], base, args.limit
            )
    print(json.dumps(out, indent=1))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
