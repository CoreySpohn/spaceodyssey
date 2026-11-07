"""Tables, observers and independent references for the prior tests (not a test file).

The table has four raw rows on channel 60 and three priors on channel 61:

- rows 0 and 1: public measurements of subject 0 (values 1 and 3);
- row 2: an observer-only measurement of subject 1 (value 2);
- row 3: a public measurement of subject 1 (value 4);
- row 4: a public prior of subject 0 (10, precision 2) whose pedigree is external;
- row 5: a public prior of subject 1 (-5, precision 1) summarizing row 3;
- row 6: a public prior of subject 2 (6, precision 3) summarizing row 2, which only
  the observer receives.

`expected` computes each audience's belief from the table alone, as sufficient
statistics: the precision-weighted mean of the retained priors and the rows that
audience folds, each row with weight one.
"""

import dataclasses
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.conformance.observer_kit import _imported_row
from spaceodyssey.core.enums import Audience
from spaceodyssey.studies import toy
from spaceodyssey.studies.premission import (
    PAD,
    PedigreeRule,
    authored_table,
    resolve_overlap,
)
from spaceodyssey.studies.prior_observer import PRIOR_INDEX, PriorObserver, prior_study

OBS, POL = int(Audience.OBSERVER), int(Audience.POLICY)
MEAS = toy.RECORD_MEASUREMENT
EXTERNAL = (7, 90, 0, 0)  # a source outside the table

ROWS = dict(
    subject=[0, 0, 1, 1, 0, 1, 2],
    local_index=[0, 1, 2, 3, 0, 1, 2],
    channel=[60, 60, 60, 60, 61, 61, 61],
    kind=[MEAS] * 4 + [PRIOR_INDEX] * 3,
    audience=[3, 3, OBS, 3, 3, 3, 3],
    payload=[[1.0, 0], [3.0, 0], [2.0, 0], [4.0, 0], [10.0, 2], [-5.0, 1], [6.0, 3]],
    start_offset=[-5, -5, -5, -5, -10, -10, -10],
    end_offset=[-1] * 7,
    pedigree=[
        (),
        (),
        (),
        (),
        ((*EXTERNAL, True),),
        ((0, 60, 1, 3, False),),
        ((0, 60, 1, 2, False),),
    ],
)


def table(*, pad=(), order=None, edit=None):
    """The table above; `pad` rows become padding, `order` permutes the rows."""
    rows = {k: list(v) for k, v in ROWS.items()}
    if edit:
        rows.update(edit)
    namespace = [PAD if i in pad else 0 for i in range(len(rows["subject"]))]
    rows["namespace"] = namespace
    if order is not None:
        rows = {k: [v[i] for i in order] for k, v in rows.items()}
    return authored_table(**rows, width=2)


def summaries(second):
    """Two public priors of subject 0, the second naming `second` (external)."""
    return authored_table(
        subject=[0, 0],
        local_index=[0, 1],
        channel=[61, 61],
        kind=[PRIOR_INDEX] * 2,
        audience=[3, 3],
        payload=[[1.0, 2.0], [3.0, 2.0]],
        pedigree=[((*EXTERNAL, True),), ((*second, True),)],
        width=1,
    )


def refit_table(policy_value=-4.0):
    """Two public raw rows, then an observer, a policy and a public prior.

    Subject 0 has an observer-only prior (10, precision 2) and a policy-only prior
    (`policy_value`, precision 1). Subject 1 has a public prior (3, precision 1) whose
    pedigree names raw row 1, the raw row of subject 1.
    """
    return authored_table(
        subject=[0, 1, 0, 0, 1],
        local_index=[0, 1, 0, 1, 2],
        channel=[60, 60, 61, 61, 61],
        kind=[MEAS, MEAS] + [PRIOR_INDEX] * 3,
        audience=[3, 3, OBS, POL, 3],
        payload=[[1.0, 0], [5.0, 0], [10.0, 2], [policy_value, 1], [3.0, 1]],
        start_offset=[-5] * 5,
        end_offset=[-1] * 5,
        pedigree=[
            (),
            (),
            ((7, 90, 0, 0, True),),
            ((7, 90, 0, 1, True),),
            ((0, 60, 1, 1, False), (7, 90, 1, 2, True)),
        ],
        width=2,
    )


def refit_study(t=None, every=100, rule=PedigreeRule.EXCLUDE_ROWS):
    """The refresh study importing `t` (the refit table) with `PriorRefitObserver`."""
    return prior_study(refit_table() if t is None else t, rule, refit_every=every)


def _present(t):
    return np.asarray(t.source.namespace) != PAD


def expected(t, rule, audience, caps):
    """Per subject `(mean, n, weight)` of `audience` from the table alone.

    A retained prior adds its precision and precision times value; every raw row the
    audience receives and keeps adds one and its value.
    """
    w = np.asarray(resolve_overlap(t, rule, audience, derived_kinds=caps.derived_kinds))
    kind = np.asarray(t.kind)
    subject = np.asarray(t.source.subject)
    payload = np.asarray(t.payload)
    released = _present(t) & ((np.asarray(t.audience) & audience) != 0)
    raw = released & (kind == MEAS)
    rows = np.flatnonzero(raw & (w > 0))
    priors = np.flatnonzero(released & (kind == PRIOR_INDEX) & (w > 0))
    count = caps.n_subjects
    weight, total, n = np.zeros(count), np.zeros(count), np.zeros(count, int)
    for p in priors:
        weight[subject[p]] += payload[p, 1]
        total[subject[p]] += payload[p, 1] * payload[p, 0]
    for r in rows:
        n[subject[r]] += 1
        total[subject[r]] += payload[r, 0]
    held = weight + n
    mean = np.where(held > 0, total / np.where(held > 0, held, 1.0), 0.0)
    return mean, n, weight


class Beliefs(NamedTuple):
    belief: object
    pbelief: object


def imported_beliefs(observer, caps):
    """Both beliefs after the import: each audience's start, then its raw rows.

    Folds every present measurement row the audience receives, as the import writes
    and the release delivers it, in the order of its channel and local index.
    """
    t = observer.table
    present, audience = _present(t), np.asarray(t.audience)
    kind, key = (
        np.asarray(t.kind),
        list(zip(t.channel.tolist(), t.local_index.tolist(), strict=True)),
    )
    out = []
    for a in (OBS, POL):
        belief = observer.init(caps) if a == OBS else observer.init_summary(caps)
        rows = np.flatnonzero(present & ((audience & a) != 0) & (kind == MEAS))
        for i in sorted(rows.tolist(), key=lambda i: key[i]):
            rec = _imported_row(t, i, caps, a)
            if a == OBS:
                belief = observer.analyze(observer.fold(belief, rec, caps))
            else:
                belief = observer.fold_summary(belief, rec, caps)
        out.append(belief)
    return Beliefs(*out)


class IgnoresPrior(PriorObserver):
    """Control: an observer that installs no prior (it ignores the table's priors)."""

    def prior_belief(self, caps, audience):
        flat = dataclasses.replace(self, table=table(pad=(4, 5, 6)))
        return PriorObserver.prior_belief(flat, caps, audience)


class RuleOff(PriorObserver):
    """Control: every row and prior keeps weight one (no overlap rule)."""

    def weights(self, caps, audience):
        t = self.table
        return (t.present & ((t.audience & audience) != 0)).astype(jnp.int64)


class TableWide(PriorObserver):
    """Control: one resolution for every audience, judged on all imported rows."""

    def weights(self, caps, audience):
        t = self.table
        wide = dataclasses.replace(t, audience=jnp.where(t.present, 3, t.audience))
        w = resolve_overlap(wide, self.rule, audience, derived_kinds=caps.derived_kinds)
        mine = t.present & ((t.audience & audience) != 0)
        return jnp.where(mine, w, 0)


class IgnoresWeights(PriorObserver):
    """Control: weights are reported but folds and priors do not use them."""

    def prior_belief(self, caps, audience):
        everything = RuleOff(table=self.table, rule=self.rule)
        return PriorObserver.prior_belief(everything, caps, audience)

    def _dropped(self, rec, caps, audience):
        return jnp.asarray(False)
