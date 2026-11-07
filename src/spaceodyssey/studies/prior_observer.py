"""The prior observer: a toy Gaussian mean whose initial belief holds a table's priors.

A prior is a row of a derived kind in a `PreMissionTable` (value in payload column 0,
precision in column 1, in units of one measurement). `PriorObserver` resolves the
overlap of each audience separately (`resolve_overlap`, D4): the observer's belief
starts from the priors the observer audience retains and the policy summary from the
priors the policy audience retains, and a raw row a rule drops for an audience is
never folded into that audience's belief. The prior rows in the log are derived: they
are never folded, so a prior counts once, through the initial belief.

Weights come from the observer's own fields (the table and the rule), never from the
log. The study's process imports the same table (`prior_study` builds both from one
table; pre-flight refuses two different tables).

Refresh: the core starts both audiences' refits from `observer.init` and shares an
entry between audiences when their masked rows and keys are equal, with no prior
identity. `AudiencePriorRefit` is the boundary that supports a table prior with
refits: it starts each audience's refit from that audience's own prior and ignores the
core's sharing unless the two audiences resolve to the same prior. Pre-flight
(through the observer's `validate`) reports:

- `D008`: two priors retained for one audience whose pedigrees intersect.
- `D009`: audiences that resolve to different priors on the boundary that starts both
  refits from `observer.init` (`MaskedBatchRefit` with the observer's own kernel).
- `D010`: a retained table prior with any other boundary that does not declare
  audience priors (`OftiRefit`, the updater boundary, a custom kernel or fit); with no
  boundary supplied, a warning that `D009` and `D010` are unverified.
"""

import dataclasses
import itertools
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from spaceodyssey.codec.bindings import BindingKind
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import (
    F64,
    I64,
    PUBLIC,
    Audience,
    RecordStatus,
    RefitStatus,
)
from spaceodyssey.core.events import where_tree
from spaceodyssey.core.kinds import AbstractObserver, ObserverCapabilities
from spaceodyssey.core.preflight import Finding, FindingCode, Severity
from spaceodyssey.core.records import KindRegistry, PayloadColumn, RecordKind
from spaceodyssey.core.refresh import (
    REFRESH_PRODUCT_KIND,
    AbstractRefitBoundary,
    MaskedBatchRefit,
    RefitResult,
    RefreshSchedule,
    observer_refit_kernel,
    take_subject,
)
from spaceodyssey.studies import toy
from spaceodyssey.studies.premission import (
    PedigreeRule,
    PreMissionImporter,
    overlap_roles,
    resolve_overlap,
)

OBSERVER, POLICY = int(Audience.OBSERVER), int(Audience.POLICY)

PRIOR_KIND = RecordKind(
    "prior",
    (PayloadColumn("value", PUBLIC), PayloadColumn("precision", PUBLIC)),
    derived=True,
)
PRODUCT_INDEX, PRIOR_INDEX = 3, 4  # kind indices in `prior_registry`


def prior_registry():
    """The toy kinds, the refresh product kind and the prior kind, at 0 to 4."""
    return KindRegistry(
        (
            RecordKind("none", ()),
            toy.MEASUREMENT_KIND,
            toy.AUXILIARY_KIND,
            REFRESH_PRODUCT_KIND,
            PRIOR_KIND,
        )
    )


class PriorBelief(eqx.Module):
    """Per-subject posterior means, rows folded, auxiliary state, prior precision.

    `n` counts folded rows only (the layout's count leaf); `weight` is the precision
    of the installed prior in units of one row, so the posterior of a subject is the
    mean of `n + weight` equally weighted values.
    """

    mean: jax.Array
    n: jax.Array
    aux: jax.Array
    weight: jax.Array


def _canonical_order(table):
    """Row order by source identity, so sums do not depend on the row order."""
    cols = (table.source.source_index, table.source.subject)
    cols += (table.source.producer, table.source.namespace)
    return jnp.lexsort(cols)


class PriorObserver(AbstractObserver):
    """A running mean whose initial beliefs hold each audience's retained priors.

    `table` is the imported table and `rule` the `PedigreeRule`. Rows of kind
    `prior_kind` carry `[value, precision]`; rows of kind `measurement_kind` are
    folded, each with weight one, unless the rule drops them for the audience.
    Incremental only; `PriorRefitObserver` also refits.
    """

    reads_priors: ClassVar[bool] = True
    subject_leaves = ("mean", "n", "weight")

    table: object
    rule: int = eqx.field(default=int(PedigreeRule.EXCLUDE_PRIOR), converter=int)
    prior_kind: int = eqx.field(default=PRIOR_INDEX)
    measurement_kind: int = eqx.field(default=toy.RECORD_MEASUREMENT)

    def __check_init__(self):
        """Refuse an unknown rule."""
        PedigreeRule(self.rule)

    # ---- resolution ---------------------------------------------------------------

    def weights(self, caps, audience):
        """Per-row weights in {0, 1} the observer installs and folds with."""
        return resolve_overlap(
            self.table, self.rule, audience, derived_kinds=caps.derived_kinds
        )

    def prior_belief(self, caps, audience):
        """The full belief holding `audience`'s retained priors and no row."""
        table, count = self.table, caps.n_subjects
        keep = (self.weights(caps, audience) > 0) & (table.kind == self.prior_kind)
        width = jnp.shape(table.payload)[1]
        value = table.payload[:, 0]
        precision = table.payload[:, 1] if width > 1 else jnp.ones_like(value)
        precision = jnp.where(keep, precision, 0.0)
        subject = jnp.clip(table.source.subject, 0, count - 1)
        order = _canonical_order(table)
        # sum in the canonical order of the source identities: permutation free
        weight, total = jnp.zeros(count, F64), jnp.zeros(count, F64)
        for i in range(table.n_rows):
            r, s = order[i], subject[order[i]]
            weight = weight.at[s].add(precision[r])
            total = total.at[s].add(precision[r] * value[r])
        mean = jnp.where(weight > 0, total / jnp.where(weight > 0, weight, 1.0), 0.0)
        return PriorBelief(
            mean=mean,
            n=jnp.zeros(count, I64),
            aux=jnp.zeros(caps.n_an, F64),
            weight=weight,
        )

    def same_resolution(self, caps):
        """Per subject: both audiences start from one prior and drop the same rows.

        Compares the two prior beliefs bit for bit and the weights of the public raw
        rows (a row only one audience receives cannot reach the other's refit).
        """
        a = self.prior_belief(caps, OBSERVER)
        b = self.prior_belief(caps, POLICY)
        same = (a.mean == b.mean) & (a.weight == b.weight)
        table = self.table
        derived = jnp.isin(table.kind, jnp.asarray(caps.derived_kinds, I64))
        public = table.present & ~derived & (table.audience == PUBLIC)
        differ = public & (self.weights(caps, OBSERVER) != self.weights(caps, POLICY))
        subject = jnp.clip(table.source.subject, 0, caps.n_subjects - 1)
        moved = jnp.zeros(caps.n_subjects, I64).at[subject].max(differ.astype(I64))
        return same & (moved == 0)

    def _dropped(self, rec, caps, audience):
        """Whether `rec` is an imported raw row the rule drops for `audience`."""
        table = self.table
        derived = jnp.isin(table.kind, jnp.asarray(caps.derived_kinds, I64))
        hit = table.present & ~derived & (table.channel == rec.channel)
        hit = hit & (table.local_index == rec.acquisition_index)
        return (hit & (self.weights(caps, audience) == 0)).any()

    # ---- the observer contract ----------------------------------------------------

    def init(self, caps):
        """The observer's belief: the priors the observer audience retains."""
        return self.prior_belief(caps, OBSERVER)

    def init_summary(self, caps):
        """The policy summary: the priors the policy audience retains."""
        return self.prior_belief(caps, POLICY)

    def fold_for(self, belief, rec, caps, audience):
        """Fold one row into `audience`'s belief, unless the rule drops it."""
        s = rec.subject
        take = (
            rec.valid
            & (rec.kind == self.measurement_kind)
            & (rec.status == RecordStatus.COMPLETED)
            & ~self._dropped(rec, caps, audience)
        )
        held = belief.n[s].astype(F64) + belief.weight[s]
        new = (belief.mean[s] * held + rec.payload[0]) / (held + 1.0)
        return dataclasses.replace(
            belief,
            mean=belief.mean.at[s].set(jnp.where(take, new, belief.mean[s])),
            n=belief.n.at[s].add(take.astype(I64)),
        )

    def fold(self, belief, rec, caps, key=None):
        """Fold a row the observer accepted, with the observer's weights."""
        return self.fold_for(belief, rec, caps, OBSERVER)

    def fold_summary(self, summary, rec, caps, key=None):
        """Fold a row released to the policy, with the policy's weights."""
        return self.analyze(self.fold_for(summary, rec, caps, POLICY))

    def refit_for(self, audience, belief0, rows, caps, key, row_keys):
        """Fold `rows` in order into `belief0` with `audience`'s weights."""

        def step(belief, row):
            folded = self.fold_for(belief, row, caps, audience)
            return where_tree(row.valid, folded, belief), None

        return lax.scan(step, belief0, rows)[0]

    def refit(self, belief0, rows, caps, key, row_keys):
        """The observer audience's refit: `refit_for` with the observer's weights."""
        return self.refit_for(OBSERVER, belief0, rows, caps, key, row_keys)

    def install(self, belief, fitted, subject):
        """Take the subject's mean, count and prior precision from a refit."""
        return dataclasses.replace(
            belief,
            mean=belief.mean.at[subject].set(fitted.mean[subject]),
            n=belief.n.at[subject].set(fitted.n[subject]),
            weight=belief.weight.at[subject].set(fitted.weight[subject]),
        )

    # ---- pre-flight ---------------------------------------------------------------

    def validate(self, study, context):
        """`D006` (a table the process does not import), `D008`, `D009`, `D010`."""
        caps = study.caps
        out = _table_mismatch(self.table, study.process)
        out += retained_intersections(self, caps)
        out += boundary_findings(self, study, context)
        return out


class PriorRefitObserver(PriorObserver):
    """The prior observer, also refit-capable on measurement rows."""

    def capabilities(self):
        """Incremental and refit-capable on measurement rows."""
        return ObserverCapabilities(
            incremental=True, refit=True, refit_kinds=("measurement",)
        )


# ---- pre-flight -------------------------------------------------------------------


def _finding(code, member, entry, message, fix, severity=Severity.ERROR):
    where = "".join(f"[{i}]" for i in entry)
    return Finding(
        code,
        severity,
        f"observer.{member}{where}",
        message,
        fix,
        subject="observer",
        member=member,
        entry=tuple(entry),
    )


def _table_mismatch(table, process):
    """`D006`: the observer resolves overlap on a table the process does not import."""
    imported = getattr(process, "table", None)
    if imported is None:
        return []
    mine, theirs = (
        jax.tree_util.tree_flatten(table),
        jax.tree_util.tree_flatten(imported),
    )
    same = mine[1] == theirs[1] and all(
        np.shape(a) == np.shape(b) and np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(mine[0], theirs[0], strict=True)
    )
    if same:
        return []
    return [
        _finding(
            FindingCode.D006,
            "table",
            (),
            "the observer resolves overlap on a table the process does not import",
            "give the observer and the importer the same table",
        )
    ]


def retained_intersections(observer, caps):
    """`D008`: two priors retained for one audience whose pedigrees intersect.

    Every pedigree entry counts, external ones included. Reported once per pair of
    rows, at the first entry of the later row that the earlier row also names.
    """
    table = observer.table
    out, seen = [], set()
    width = np.shape(table.external)[1]
    named = [tuple(k) for k in table.pedigree.keys()]
    count = np.clip(np.asarray(table.n_pedigree), 0, width)
    for audience in (OBSERVER, POLICY):
        prior, _, _ = overlap_roles(table, audience, derived_kinds=caps.derived_kinds)
        weights = np.asarray(observer.weights(caps, audience))
        kept = np.flatnonzero(np.asarray(prior) & (weights > 0))
        for p, q in itertools.combinations(kept.tolist(), 2):
            first = {named[p * width + j] for j in range(int(count[p]))}
            hits = [j for j in range(int(count[q])) if named[q * width + j] in first]
            if not hits or (p, q) in seen:
                continue
            seen.add((p, q))
            j = hits[0]
            out.append(
                _finding(
                    FindingCode.D008,
                    "table.pedigree.source_index",
                    (q, j),
                    f"priors {p} and {q} are both retained for audience {audience} "
                    f"and both summarize source {list(named[q * width + j])}",
                    "keep one of the two summaries, or import the shared source as "
                    "a row so the rule can exclude one side",
                )
            )
    return out


def starts_from_observer_init(boundary):
    """Whether `boundary` is the masked batch of the observer's own refit kernel."""
    return (
        isinstance(boundary, MaskedBatchRefit)
        and boundary.kernel is observer_refit_kernel
    )


def boundary_findings(observer, study, context):
    """`D009` and `D010` for a refit-capable prior observer and its refit boundary."""
    caps = study.caps
    if not observer.capabilities().refit:
        return []
    keeps = [
        np.asarray(observer.weights(caps, a) > 0)
        & (np.asarray(observer.table.kind) == observer.prior_kind)
        for a in (OBSERVER, POLICY)
    ]
    retained = bool(np.any(keeps[0] | keeps[1]))
    boundary = None if context is None else context.boundary
    if boundary is None:
        if not retained or int(study.refresh.every_ticks) <= 0:
            return []
        return [
            _finding(
                FindingCode.D010,
                "table",
                (),
                "a table prior with a refit-capable observer; no refit boundary was "
                "supplied, so D009 and D010 are unverified",
                "pass the run's refit boundary in the pre-flight context",
                Severity.WARNING,
            )
        ]
    if getattr(boundary, "audience_priors", False):
        return []
    name = getattr(boundary, "kind", type(boundary).__name__)
    if starts_from_observer_init(boundary):
        differ = np.flatnonzero(~np.asarray(observer.same_resolution(caps)))
        if differ.size == 0:
            return []
        return [
            _finding(
                FindingCode.D009,
                "table",
                (),
                f"the audiences resolve to different priors for subjects "
                f"{differ.tolist()}, but boundary {name!r} starts both refits from "
                "observer.init and shares them",
                "bind a boundary that declares audience priors (AudiencePriorRefit)",
            )
        ]
    if not retained:
        return []
    return [
        _finding(
            FindingCode.D010,
            "table",
            (),
            f"a table prior with refit boundary {name!r}, which does not consume "
            "it; only AudiencePriorRefit supports a table prior with refits",
            "bind AudiencePriorRefit, or drop the table prior",
        )
    ]


# ---- the toy audience-prior boundary ----------------------------------------------


def _audience_lane(boundary, study, request):
    """One world: each dirty entry refit from its own audience's prior."""
    observer, caps = study.observer, study.caps
    count, names = caps.n_subjects, observer.subject_leaves
    own = jnp.arange(2 * count).reshape(2, count)
    same = boundary.shared(study)
    computed = request.dirty & ((request.source == own) | ~same[None, :])
    per = []
    for a, audience in enumerate((OBSERVER, POLICY)):
        start = boundary.start(study, audience)

        def one(rows, key, row_keys, s, audience=audience, start=start):
            fitted = observer.refit_for(audience, start, rows, caps, key, row_keys)
            return take_subject(fitted, names, s)

        subjects = jnp.arange(count)
        per.append(
            jax.vmap(one)(
                jax.tree_util.tree_map(lambda x, a=a: x[a], request.rows),
                request.keys[a],
                request.row_keys[a],
                subjects,
            )
        )
    take = jnp.where(computed | ~request.dirty, own, request.source).reshape(-1)

    def answer(o, p):
        both = jnp.stack([o, p]).reshape((2 * count, *o.shape[1:]))
        picked = both[take].reshape((2, count, *o.shape[1:]))
        mask = request.dirty.reshape(request.dirty.shape + (1,) * (o.ndim - 1))
        return jnp.where(mask, picked, jnp.zeros_like(picked))

    slices = tuple(answer(o, p) for o, p in zip(per[0], per[1], strict=True))
    status = jnp.where(request.dirty, RefitStatus.OK, RefitStatus.NO_SUPPORT)
    n_rows = request.rows.valid.sum(axis=-1).astype(F64)
    diag = jnp.where(request.dirty, n_rows, 0.0)[..., None]
    return RefitResult(
        slices=slices, status=status.astype(I64), diagnostics=diag, computed=computed
    )


@eqx.filter_jit
def _audience_call(boundary, study, requests):
    return jax.vmap(lambda r: _audience_lane(boundary, study, r))(requests)


class AudiencePriorRefit(AbstractRefitBoundary):
    """The toy refit boundary that starts each audience from its own prior.

    For a `PriorObserver`: every dirty entry of audience `a` is the observer's
    `refit_for(a, ...)` of the entry's rows from `prior_belief(caps, a)`. The core's
    sharing (`RefitRequest.source`) is honored only for subjects whose audiences
    resolve to the same prior (`same_resolution`); elsewhere each audience computes
    its own entry. It is the only boundary that supports a table prior with refits.
    """

    kind = "premission.audience_prior"
    audience_priors = True

    def __init__(self, *, model_hash=0, version=1):
        """A stateless boundary; the priors are read from the study's observer."""
        self.model_hash = model_hash
        self.version = version

    def __eq__(self, other):
        """Equal to a boundary of the same class and declaration (one compile)."""
        return type(other) is type(self) and other.binding() == self.binding()

    def __hash__(self):
        """Hash of the class and declaration, as `__eq__` compares them."""
        return hash((type(self), tuple(sorted(self.binding().items()))))

    def start(self, study, audience):
        """The belief an entry of `audience` is refit from."""
        return study.observer.prior_belief(study.caps, audience)

    def shared(self, study):
        """Per subject: the core's sharing between audiences may be honored."""
        return study.observer.same_resolution(study.caps)

    def refit_batch(self, study, requests):
        """Answer every dirty entry of every world."""
        if not getattr(study.observer, "reads_priors", False):
            raise ValueError("AudiencePriorRefit needs an observer that reads priors")
        return _audience_call(self, study, requests)


def audience_prior_binding():
    """The `BindingKind` a run document names an `AudiencePriorRefit` by."""
    return BindingKind(
        AudiencePriorRefit.kind,
        1,
        cls=AudiencePriorRefit,
        describe=lambda obj, registry: {"model_hash": obj.model_hash},
        build=lambda config, registry: AudiencePriorRefit(
            model_hash=config["model_hash"]
        ),
    )


# ---- studies ----------------------------------------------------------------------


def prior_study(table, rule=PedigreeRule.EXCLUDE_PRIOR, *, refit_every=0, **overrides):
    """The toy survey importing `table`, with a prior observer over the same table.

    With `refit_every > 0` the observer is `PriorRefitObserver` and the study has a
    refresh grid of that period (`rows_per_subject = 8`, `refit_lanes = 2`).
    """
    cls = PriorRefitObserver if refit_every else PriorObserver
    observer = cls(table=table, rule=int(rule))
    caps = overrides.pop("caps", None) or Capacities(
        rows_per_subject=8 if refit_every else 0, refit_lanes=2 if refit_every else 0
    )
    caps = prior_registry().lower(caps, observer)
    if refit_every:
        overrides["refresh"] = RefreshSchedule(
            refit_every, 0, *toy.PRODUCT_CHANNELS, PRODUCT_INDEX
        )
    return toy.survey_1(
        caps, process=PreMissionImporter(table=table), observer=observer, **overrides
    )


PRIOR_OBSERVER_KINDS = {
    ("observer", "premission.prior_observer", 1): PriorObserver,
    ("observer", "premission.prior_refit_observer", 1): PriorRefitObserver,
}
"""The kind references of the prior observers, for `dump_study` and `load_study`."""


__all__ = [
    "PRIOR_INDEX",
    "PRIOR_KIND",
    "PRIOR_OBSERVER_KINDS",
    "PRODUCT_INDEX",
    "AudiencePriorRefit",
    "PriorBelief",
    "PriorObserver",
    "PriorRefitObserver",
    "audience_prior_binding",
    "boundary_findings",
    "prior_registry",
    "prior_study",
    "retained_intersections",
    "starts_from_observer_init",
]
