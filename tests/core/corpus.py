"""Corpus handles for the tests.

The corpus itself lives in the library (`spaceodyssey.studies.conformance`); the leak
scenarios and the draw-invariance perturbations are test fixtures built on its studies.
"""

import equinox as eqx
import jax.numpy as jnp

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.conformance.corpus import CorpusCase as CorpusCase
from spaceodyssey.core.enums import Audience
from spaceodyssey.studies import builder
from spaceodyssey.studies import toy as kinds
from spaceodyssey.studies.conformance import CORPUS as CORPUS
from spaceodyssey.studies.conformance import CORPUS_BY_NAME as CORPUS_BY_NAME
from spaceodyssey.studies.conformance import HAND_HISTORIES as HAND_HISTORIES
from spaceodyssey.studies.conformance import HAND_INVARIANTS as HAND_INVARIANTS
from spaceodyssey.studies.conformance import HAND_ROWS as HAND_ROWS
from spaceodyssey.studies.conformance import HAND_STUDIES as HAND_STUDIES
from spaceodyssey.studies.conformance import PLAN5 as PLAN5
from spaceodyssey.studies.conformance import frozen as frozen
from spaceodyssey.studies.conformance import stop_grid as stop_grid

# ---- leak scenarios ---------------------------------------------------------------

LEAK_PLAN = PLAN5


def leak_study(process):
    """survey_1 under a policy that ignores wakes with nothing new in its view."""
    return builder.survey_1(
        policies=(kinds.ReactivePolicy(plan=LEAK_PLAN),), process=process
    )


LEAK_SCENARIOS = {
    "heartbeat": kinds.HeartbeatProcess(
        period_ticks=37, n_fires=10, audience=Audience.OBSERVER, writes=False
    ),
    "private_writer": kinds.HeartbeatProcess(
        period_ticks=37, n_fires=10, audience=Audience.OBSERVER, writes=True
    ),
    "releasing": kinds.HeartbeatProcess(
        period_ticks=130, n_fires=1, audience=Audience.POLICY, writes=False
    ),
}


# ---- draw-invariance perturbations --------------------------------------------------
# name -> (baseline factory, perturbed factory). A perturbation changes how the study is
# built or when things happen, never which occupancy a draw belongs to.


def _prepend_kind_study():
    """Renumber the measurement and auxiliary record kinds to 5 and 9."""
    base = builder.survey_1(
        observer=kinds.ToyRunningMean(measurement_kind=5, auxiliary_kind=9)
    )
    kind = jnp.full_like(base.channels.kind, 5)
    return eqx.tree_at(lambda st: st.channels.kind, base, kind)


PERTURBATIONS = {
    "insert_wait": (HAND_STUDIES["baseline"], HAND_STUDIES["wait"]),
    "prepend_kind": (HAND_STUDIES["baseline"], _prepend_kind_study),
    # the cancelled occupancy has to be live at a decision, which takes a bundle, so the
    # baseline is the bundle study (the same keys, none cancelled)
    "cancel_reobserve": (
        lambda: builder.survey_1(policies=(frozen(bundle_at=1),)),
        HAND_STUDIES["cancel_live"],
    ),
    "extra_kinds": (HAND_STUDIES["baseline"], lambda: builder.survey_1()),
    "payload_width": (
        HAND_STUDIES["baseline"],
        lambda: builder.survey_1(caps=Capacities(payload_width=6)),
    ),
}
