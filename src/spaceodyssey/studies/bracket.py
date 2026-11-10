"""The summary observer: a decision summary per star, changed by refits only.

`SummaryObserver` holds a `DecisionSummary` as both the full belief and the policy
summary. It never folds a record: a checkpoint's refit boundary computes each star's
summary from that star's visit reports (and, for the oracle arm, the truth catalog)
and the observer installs it.
"""

import equinox as eqx
import jax.numpy as jnp

from spaceodyssey.adapters.photomancy import (
    CATALOG_KIND_NAME,
    REPORT_KIND_NAME,
    SUMMARY_LEAVES,
    DecisionSummary,
)
from spaceodyssey.core.enums import F64
from spaceodyssey.core.kinds import AbstractObserver, ObserverCapabilities
from spaceodyssey.core.snapshot import BeliefLayout


def project_detections(belief):
    """The trace projection: detections per star, and a zero auxiliary vector."""
    return belief.n_det.sum(axis=-1).astype(F64), jnp.zeros_like(belief.aux)


class SummaryObserver(AbstractObserver):
    """A `DecisionSummary` per star, set only by `install`.

    `prior` is the first-tier summary of each star with no data; it is what the
    observer and the policy hold until the first refit, and it is built from the
    belief's own prior, never from the truth. The study needs `every_ticks > 0` and
    `rows_per_subject > 0`, as every non-incremental observer does.
    """

    subject_leaves = SUMMARY_LEAVES
    prior: DecisionSummary
    catalog_kind: str = eqx.field(default=CATALOG_KIND_NAME, static=True)
    report_kind: str = eqx.field(default=REPORT_KIND_NAME, static=True)

    def capabilities(self):
        """Refit only, from visit reports and the truth catalog."""
        return ObserverCapabilities(
            incremental=False,
            refit=True,
            refit_kinds=(self.report_kind, self.catalog_kind),
        )

    def init(self, caps):
        """The prior summary of every star."""
        return self.prior

    def init_summary(self, caps):
        """The policy starts from the same prior."""
        return self.prior

    def layout(self, caps):
        """One type for belief and summary; `n_obs` counts the visits read."""
        shapes = self.prior
        return BeliefLayout(
            belief=shapes,
            summary=shapes,
            count_leaf="n_obs",
            project=project_detections,
        )

    def install(self, belief, fitted, subject):
        """Replace one star's leaves with those of the refit belief."""
        return eqx.tree_at(
            lambda b: tuple(getattr(b, n) for n in SUMMARY_LEAVES),
            belief,
            tuple(
                getattr(belief, n).at[subject].set(getattr(fitted, n)[subject])
                for n in SUMMARY_LEAVES
            ),
        )
