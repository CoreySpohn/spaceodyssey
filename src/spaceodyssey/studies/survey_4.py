"""RV precursor (prescribed): measurements exported from an upstream run, imported.

An upstream run (the toy survey, `survey_4_upstream`) observes two planets. Its
measurements of one of them are exported as a precursor table (`export_precursor`,
through `precursor.export_records`): one row per datum, at the latest revision
released to the chosen audience by the end of the upstream run, with the upstream
run's end placed at the downstream's tick 0 and the upstream document's content hash
declared as the table's lineage. The downstream survey (`survey_4`) imports the table
at tick 0 with the pre-mission importer, so the observer folds the precursor rows
like its own measurements.

The downstream policy (`LeastKnownPolicy`) observes, at each of its first `n_obs`
decisions, whichever target planet its summary holds the fewest measurements of
(the first on a tie), then drains the campaign. The value (`LeastKnownValue`) is the
number of measurements the observer holds of its least-known target. Each precursor
row of a target is one more measurement of it, so with `k` precursor rows of one
target and `n_obs` observations the value is `(n_obs + k) // 2` (for `k <= n_obs`)
against `n_obs // 2` without: the precursor helps, the toy's sign.

`without_precursor` pads every row of a table (the same shape, nothing imported), so
the runs with and without the precursor share one compiled program;
`zero_information` turns every row into an auxiliary row of value 0, which the toy
observer and the policy read as nothing.
"""

import dataclasses

import equinox as eqx
import jax.numpy as jnp

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import F64, I64, PUBLIC, Disposition, EffectKind, Scope
from spaceodyssey.core.kinds import AbstractObjective, AbstractPolicy
from spaceodyssey.studies import toy
from spaceodyssey.studies.precursor import export_records
from spaceodyssey.studies.premission import PAD, PreMissionImporter

UPSTREAM_CHANNEL, PRECURSOR_CHANNEL = 10, 60
PRODUCERS = {UPSTREAM_CHANNEL: PRECURSOR_CHANNEL}  # upstream channel to downstream
SUBJECTS = {1: 0}  # the upstream run's planet 1 is the downstream's target 0
TARGETS = (0, 1)
OBSERVE_CHANNEL = 10
N_OBSERVATIONS = 4


def survey_4_upstream(**overrides):
    """The upstream study: the toy survey, which observes planet 1 twice."""
    return toy.survey_1(**overrides)


def export_precursor(run_result, upstream_hash, *, audience=PUBLIC, study=None):
    """The precursor `Export` of an upstream run: its measurements of planet 1.

    `upstream_hash` is the content hash of the upstream run's document; `study` is
    the upstream study (`survey_4_upstream()` by default).
    """
    return export_records(
        run_result,
        toy.RECORD_MEASUREMENT,
        audience,
        upstream_hash,
        SUBJECTS,
        PRODUCERS,
        study=survey_4_upstream() if study is None else study,
    )


def without_precursor(table):
    """`table` with every row padded: the same shape, nothing imported."""
    pad = jnp.full_like(table.source.namespace, PAD)
    return eqx.tree_at(lambda t: t.source.namespace, table, pad)


def zero_information(table):
    """`table` with every row an auxiliary row of value 0 (read as nothing)."""
    kind = jnp.full_like(table.kind, toy.RECORD_AUXILIARY)
    return eqx.tree_at(
        lambda t: (t.kind, t.payload), table, (kind, jnp.zeros_like(table.payload))
    )


class LeastKnownPolicy(AbstractPolicy):
    """Observe the target the summary holds the fewest measurements of, `n_obs` times.

    The first `n_obs` decisions each observe, among `targets`, the subject with the
    smallest `view.belief.n` (the first on a tie); the next drains the campaign. It
    draws nothing.
    """

    targets: tuple = eqx.field(default=TARGETS, static=True)
    channel: int = eqx.field(default=OBSERVE_CHANNEL, static=True)
    n_obs: int = eqx.field(default=N_OBSERVATIONS, static=True)

    def declare(self):
        """Observe and the closing campaign drain."""
        return Declaration(
            effects=frozenset((EffectKind.OBSERVE, EffectKind.STOP)),
            stops=((Scope.CAMPAIGN, Disposition.DRAIN),),
        )

    def decide(self, pstate, view, study, key):
        """Observe the least-known target, or drain after `n_obs` decisions."""
        targets = jnp.asarray(self.targets, I64)
        subject = targets[jnp.argmin(view.belief.n[targets])]
        stop = view.n_dec >= self.n_obs
        channel = jnp.argmax(study.channels.id == self.channel).astype(I64)
        eff = toy._effects(
            study.caps,
            kind=jnp.stack(
                [
                    jnp.where(stop, EffectKind.STOP, EffectKind.OBSERVE),
                    jnp.asarray(EffectKind.NONE, I64),
                ]
            ).astype(I64),
            subject=subject,
            channel_index=channel,
            group=view.n_dec,
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
        )
        return eff, pstate, jnp.asarray(0.0, F64)


class LeastKnownValue(AbstractObjective):
    """The measurements the observer holds of its least-known target."""

    targets: tuple = eqx.field(default=TARGETS, static=True)

    def evaluate(self, carry, study):
        """`min(belief.n[targets])` of the final carry."""
        return jnp.min(carry.belief.n[jnp.asarray(self.targets, I64)])


def survey_4(table, **overrides):
    """The downstream survey: the toy channels, the precursor import, the policy.

    `table` is the precursor (an `Export.table`, or it padded by `without_precursor`);
    other overrides follow `toy.survey_1`.
    """
    caps = overrides.pop("caps", None) or Capacities()
    overrides.setdefault("policies", (LeastKnownPolicy(),))
    study = toy.survey_1(caps, process=PreMissionImporter(table=table), **overrides)
    return dataclasses.replace(study, objective=LeastKnownValue())


SURVEY_4_KINDS = {
    ("policy", "survey_4.least_known", 1): LeastKnownPolicy,
    ("objective", "survey_4.least_known_value", 1): LeastKnownValue,
}
"""The kind references of the survey's policy and value, for `dump_study`."""


__all__ = [
    "N_OBSERVATIONS",
    "PRECURSOR_CHANNEL",
    "PRODUCERS",
    "SUBJECTS",
    "SURVEY_4_KINDS",
    "TARGETS",
    "LeastKnownPolicy",
    "LeastKnownValue",
    "export_precursor",
    "survey_4",
    "survey_4_upstream",
    "without_precursor",
    "zero_information",
]
