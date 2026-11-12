"""Known planets: priors with a declared pedigree, and window-valued scheduling.

A pre-mission table brings raw rows and priors into the run at tick 0. A prior is a
row of the derived kind `WINDOW_PRIOR_KIND`: its payload holds the value and precision
the `PriorObserver` installs, and the window, relative to the row's release tick, in
which the policy should start observing the prior's subject. Overlap of a prior's
pedigree with an audience's evidence is resolved per audience (`resolve_overlap`).

`WindowImporter` writes each audience's own copy of a prior: the window columns hold
the table's window when that audience retains the prior and `NO_WINDOW` when its rule
drops it. The row's metadata (audience, release, count) never depends on the
exclusion, so the policy's rows depend only on evidence the policy receives.

`WindowPolicy` reads the window of its target subject's prior from the released rows:
its first decision waits until the window opens, or, with no window, for a fallback
delay; later decisions observe the subject whose posterior mean is largest, then the
campaign drains. Every tick it reads is relative to a released row or to the decision
tick, so shifting the epoch origin changes nothing.

The survey's table (`survey_3_table`) has two public raw rows, an optional raw row of
subject 0, and two public priors:

=== =========== ======= ============================ ===========================
Row Kind        Subject Audience                     Pedigree
=== =========== ======= ============================ ===========================
0   measurement 0       public
1   measurement 1       public
2   measurement 0       observer only, public or not
                        imported
3   prior with  0       public                       an external source, row 2
    a window
4   prior       1       public                       row 1
=== =========== ======= ============================ ===========================

Row 3's pedigree entry for row 2 is external when row 2 is not imported, so prior 3 is
then entirely external.
"""

import equinox as eqx
import jax.numpy as jnp

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import (
    F64,
    I64,
    PUBLIC,
    Audience,
    Disposition,
    EffectKind,
    Scope,
)
from spaceodyssey.core.kinds import AbstractPolicy
from spaceodyssey.core.records import KindRegistry, PayloadColumn, RecordKind
from spaceodyssey.studies import builder, effect_rows, toy
from spaceodyssey.studies.premission import (
    PAD,
    PedigreeRule,
    PreMissionImporter,
    authored_table,
)
from spaceodyssey.studies.prior_observer import PriorObserver

OBSERVER, POLICY = int(Audience.OBSERVER), int(Audience.POLICY)

WINDOW_PRIOR_KIND = RecordKind(
    "windowed_prior",
    (
        PayloadColumn("value", PUBLIC),
        PayloadColumn("precision", PUBLIC),
        PayloadColumn("window_lo_rel", PUBLIC),
        PayloadColumn("window_hi_rel", PUBLIC),
    ),
    derived=True,
)
WINDOW_PRIOR_INDEX = 3  # its index in `survey_3_registry`
LO_COLUMN, HI_COLUMN = 2, 3  # the window's payload columns
NO_WINDOW = (0.0, -1.0)  # an empty window: its start is after its end

RAW_CHANNEL, PRIOR_CHANNEL = 60, 61  # the channels of the imported rows
TARGET = 0  # the subject whose prior carries the window the policy schedules by
WINDOW = (40, 80)  # the target prior's window, relative to its release
FALLBACK_WAIT = 150  # the wait from the first decision when no window is released
OBSERVE_CHANNEL = 10
N_OBSERVATIONS = 4
EXTERNAL = (7, 90, TARGET, 0)  # a source of prior 3 outside the table
OLDEST = 20  # ticks before the start of the oldest imported time


def survey_3_registry():
    """The toy kinds and the windowed prior, at indices 0 to 3."""
    return KindRegistry(
        (
            RecordKind("none", ()),
            toy.MEASUREMENT_KIND,
            toy.AUXILIARY_KIND,
            WINDOW_PRIOR_KIND,
        )
    )


class WindowImporter(PreMissionImporter):
    """The pre-mission importer, writing each audience's window into its prior row.

    A row of `window_kind` gets the table's window in its window columns when the
    observer's overlap resolution (`weights`) retains the prior for the audience of
    the event that writes it, and `NO_WINDOW` when it drops it. A public prior is
    written once per audience, so each copy carries its own audience's resolution.
    """

    window_kind: int = eqx.field(default=WINDOW_PRIOR_INDEX, static=True)

    def handle(self, carry, ev, study):
        """Write the row, then its audience's window (prior rows only)."""
        carry, rec = super().handle(carry, ev, study)
        table = self.table
        if table.n_rows == 0:
            return carry, rec
        weights = getattr(study.observer, "weights", None)
        if weights is None:
            raise TypeError(
                "WindowImporter writes each audience's window under the observer's "
                "overlap resolution; the observer must read priors (weights)"
            )
        caps, g = study.caps, ev.group
        kept = jnp.where(
            ev.audience == OBSERVER,
            weights(caps, OBSERVER)[g],
            weights(caps, POLICY)[g],
        )
        windowed = rec.kind == self.window_kind
        lo = jnp.where(kept > 0, table.window_lo_rel[g].astype(F64), NO_WINDOW[0])
        hi = jnp.where(kept > 0, table.window_hi_rel[g].astype(F64), NO_WINDOW[1])
        payload = rec.payload
        payload = payload.at[LO_COLUMN].set(jnp.where(windowed, lo, payload[LO_COLUMN]))
        payload = payload.at[HI_COLUMN].set(jnp.where(windowed, hi, payload[HI_COLUMN]))
        return carry, eqx.tree_at(lambda r: r.payload, rec, payload)


class WindowPolicy(AbstractPolicy):
    """Start observing in the target prior's window, or after a fallback delay.

    The first decision reads the released rows of `window_kind` about `target` whose
    window is not empty and waits until the earliest opens (`released_tick` plus
    `window_lo_rel`); with no such row it waits `fallback_wait` ticks from the
    decision. The next `n_obs` decisions observe, among `subjects`, the subject whose
    posterior mean is largest (the first on a tie); then the campaign drains. The
    policy never reads a pedigree or a weight, and draws nothing.
    """

    target: int = eqx.field(default=TARGET, static=True)
    subjects: tuple = eqx.field(default=(0, 1), static=True)
    window_kind: int = eqx.field(default=WINDOW_PRIOR_INDEX, static=True)
    fallback_wait: int = eqx.field(default=FALLBACK_WAIT, static=True)
    channel: int = eqx.field(default=OBSERVE_CHANNEL, static=True)
    n_obs: int = eqx.field(default=N_OBSERVATIONS, static=True)

    def declare(self):
        """Observe, wait and the closing campaign drain."""
        return Declaration(
            effects=frozenset((EffectKind.OBSERVE, EffectKind.WAIT, EffectKind.STOP)),
            stops=((Scope.CAMPAIGN, Disposition.DRAIN),),
        )

    def start(self, view):
        """The tick the first decision waits for."""
        log = view.log
        lo, hi = log.payload[:, LO_COLUMN], log.payload[:, HI_COLUMN]
        open_ = log.valid & (log.kind == self.window_kind) & (lo <= hi)
        open_ = open_ & (log.subject == self.target)
        opens = log.released_tick + lo.astype(I64)
        earliest = jnp.min(jnp.where(open_, opens, jnp.iinfo(jnp.int64).max))
        return jnp.where(open_.any(), earliest, view.tick + self.fallback_wait)

    def decide(self, pstate, view, study, key):
        """Wait for the window, then observe by the posterior, then drain."""
        first = view.n_dec == 0
        stop = view.n_dec > self.n_obs
        subjects = jnp.asarray(self.subjects, I64)
        best = subjects[jnp.argmax(view.belief.mean[subjects])]
        act = jnp.where(stop, EffectKind.STOP, EffectKind.OBSERVE)
        channel = jnp.argmax(study.channels.id == self.channel).astype(I64)
        eff = effect_rows.effects(
            study.caps,
            kind=jnp.stack(
                [
                    jnp.where(first, EffectKind.NONE, act),
                    jnp.where(first, EffectKind.WAIT, EffectKind.NONE),
                ]
            ).astype(I64),
            subject=best,
            channel_index=channel,
            group=view.n_dec,
            tick=jnp.stack([jnp.asarray(0, I64), self.start(view)]),
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
        )
        return eff, pstate, jnp.asarray(0.0, F64)


def survey_3_table(*, row_2=OBSERVER, prior_3=True, prior_4=True, order=None):
    """The survey's table (see the module docstring).

    `row_2` is the audience of the raw row of subject 0 that prior 3 summarizes, or
    `None` to leave it out (its pedigree entry is then external). `prior_3` and
    `prior_4` keep or pad the two priors; `order` permutes the rows.
    """
    imported = row_2 is not None
    rows = dict(
        subject=[0, 1, 0, TARGET, 1],
        local_index=[0, 1, 2, 0, 1],
        channel=[RAW_CHANNEL] * 3 + [PRIOR_CHANNEL] * 2,
        kind=[toy.RECORD_MEASUREMENT] * 3 + [WINDOW_PRIOR_INDEX] * 2,
        audience=[PUBLIC, PUBLIC, row_2 if imported else OBSERVER, PUBLIC, PUBLIC],
        payload=[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 2.0], [-3.0, 1.0]],
        start_offset=[-12, -5, -8, -OLDEST, -OLDEST],
        end_offset=[-6, -1, -3, 0, 0],
        window_lo_rel=[0, 0, 0, WINDOW[0], WINDOW[0]],
        window_hi_rel=[0, 0, 0, WINDOW[1], WINDOW[1]],
        namespace=[
            0,
            0,
            *(0 if keep else PAD for keep in (imported, prior_3, prior_4)),
        ],
        pedigree=[
            (),
            (),
            (),
            ((*EXTERNAL, True), (0, RAW_CHANNEL, 0, 2, not imported)),
            ((0, RAW_CHANNEL, 1, 1, False),),
        ],
    )
    if order is not None:
        rows = {k: [v[i] for i in order] for k, v in rows.items()}
    return authored_table(**rows, width=2)


def survey_3(
    table=None, rule=PedigreeRule.EXCLUDE_PRIOR, *, observer=None, **overrides
):
    """The survey: the toy channels, the windowed import, the prior observer.

    `table` defaults to `survey_3_table()`; `rule` is the `PedigreeRule` the observer
    resolves overlap with. `observer` replaces the `PriorObserver` built from them (a
    control); other overrides follow `builder.survey_1`.
    """
    table = survey_3_table() if table is None else table
    if observer is None:
        observer = PriorObserver(
            table=table, rule=int(rule), prior_kind=WINDOW_PRIOR_INDEX
        )
    caps = overrides.pop("caps", None) or Capacities()
    caps = survey_3_registry().lower(caps, observer)
    overrides.setdefault("policies", (WindowPolicy(),))
    return builder.survey_1(
        caps, process=WindowImporter(table=table), observer=observer, **overrides
    )


__all__ = [
    "EXTERNAL",
    "FALLBACK_WAIT",
    "HI_COLUMN",
    "LO_COLUMN",
    "NO_WINDOW",
    "N_OBSERVATIONS",
    "OLDEST",
    "PRIOR_CHANNEL",
    "RAW_CHANNEL",
    "TARGET",
    "WINDOW",
    "WINDOW_PRIOR_INDEX",
    "WINDOW_PRIOR_KIND",
    "WindowImporter",
    "WindowPolicy",
    "survey_3",
    "survey_3_registry",
    "survey_3_table",
]
