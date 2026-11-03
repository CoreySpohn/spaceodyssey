"""Shared cached corpus runs, including private-event probes."""

import functools

from spaceodyssey.core.conformance import run_case
from spaceodyssey.studies.conformance import CORPUS_BY_NAME

N_WORLDS = 2


@functools.cache
def runs(name, executor):
    """The runs of one case on one executor, with the private-event probe."""
    return tuple(run_case(CORPUS_BY_NAME[name], executor, n_worlds=N_WORLDS))
