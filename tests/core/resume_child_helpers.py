"""Child process for the kill tests (not a test file): run, and die at a write step."""

import os
import signal
import sys

import jax

jax.config.update("jax_enable_x64", True)

from spaceodyssey.codec.checkpoint import DirectoryCheckpoint  # noqa: E402
from spaceodyssey.core import HostExecutor  # noqa: E402
from spaceodyssey.core.executors.base import index_world  # noqa: E402
from tests.codec.run_cases import KINDS  # noqa: E402
from tests.core import refresh_helpers as rh  # noqa: E402
from tests.core.helpers import worlds  # noqa: E402


def main(directory, op, nth, steps):
    """Run `steps` steps and SIGKILL this process at the `nth` publish's `op` step."""
    study = rh.refresh_study(100)
    truths, roots = worlds(1)
    sink = DirectoryCheckpoint(directory, kinds=KINDS)
    seen = {"publishes": 0}

    def hook(name):
        if name == "payload":
            seen["publishes"] += 1
        if name == op and seen["publishes"] == nth:
            os.kill(os.getpid(), signal.SIGKILL)

    sink.store.hook = hook
    HostExecutor().run(
        study,
        index_world(truths, 0),
        roots[0],
        steps,
        refit=rh.masked(),
        checkpoint=sink,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
