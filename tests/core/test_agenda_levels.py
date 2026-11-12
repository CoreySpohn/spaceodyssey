"""Agenda levels: numbered, nested effect masks intersected with the phase masks."""

import dataclasses
import itertools

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey import core
from spaceodyssey.core.enums import (
    ALL_EFFECTS,
    LEVEL_EFFECTS,
    AdmissionCode,
    EffectKind,
    effect_mask,
)
from spaceodyssey.core.study import Agenda
from spaceodyssey.studies import builder
from spaceodyssey.studies import toy as kinds
from tests.core.helpers import admit_j, effects_of, one_world

TRUTH, _ROOT = one_world()

# The adopted level table, written out kind by kind.
ADOPTED = (
    {EffectKind.NONE, EffectKind.CANCEL},
    {EffectKind.NONE, EffectKind.CANCEL}
    | {EffectKind.OBSERVE, EffectKind.WAIT, EffectKind.QUERY, EffectKind.STOP},
    {EffectKind.NONE, EffectKind.CANCEL}
    | {EffectKind.OBSERVE, EffectKind.WAIT, EffectKind.QUERY, EffectKind.STOP},
    set(EffectKind),
)


def with_levels(study, levels, permitted_effects=None):
    """The study with its agenda rebuilt at the given levels (validated)."""
    agenda = study.agenda
    rebuilt = Agenda(
        agenda.policies,
        agenda.phase_until,
        agenda.permitted_effects if permitted_effects is None else permitted_effects,
        levels,
    )
    return eqx.tree_at(lambda s: s.agenda, study, rebuilt)


def carry0(study):
    return eqx.filter_jit(core.init_carry)(study, TRUTH)


@pytest.mark.parametrize(
    ("level", "code"), [(1, AdmissionCode.FORBIDDEN), (3, AdmissionCode.UNSUPPORTED)]
)
def test_allocate_outside_the_level_is_forbidden_and_changes_nothing(level, code):
    """Fails if a level-1 policy's Allocate reaches the unsupported branch or moves
    the carry, or if level 3 (which permits Allocate) refuses it as forbidden."""
    study = with_levels(builder.survey_1(), (level,))
    c0 = carry0(study)
    c1, codes = admit_j(c0, effects_of(EffectKind.ALLOCATE), study)
    assert int(codes[0]) == code
    r = core.tree_equal_report(c0, c1)
    assert r["max_int_diff"] == 0 and r["max_abs"] == 0.0


@pytest.mark.parametrize("level", [1, 3])
def test_awards_enabled_allocate_obeys_the_agenda_level(level):
    """Fails if awards bypass level 1 or a valid Allocate at level 3 fails to credit
    its recipient.
    """
    from tests.core.award_helpers import award_study

    study = with_levels(
        award_study((), (), caps_extra={"snapshot_balances": True}), (level,)
    )
    before = carry0(study)
    effects = core.effects_none(study.caps)
    effects = dataclasses.replace(
        effects,
        kind=effects.kind.at[0].set(EffectKind.ALLOCATE),
        amount=effects.amount.at[0, 1].set(5.0),
        recipient=effects.recipient.at[0].set(2),
        window_from=effects.window_from.at[0].set(30),
        window_until=effects.window_until.at[0].set(200),
    )
    after, codes = admit_j(before, effects, study)
    assert int(codes[0]) == (
        AdmissionCode.FORBIDDEN if level == 1 else AdmissionCode.OK
    )
    if level == 1:
        report = core.tree_equal_report(before, after)
        assert (
            report["structure_equal"]
            and report["max_int_diff"] == 0
            and report["max_abs"] == 0.0
        )
    else:
        expected = np.asarray(before.ledger.balance).copy()
        expected[2, 1] += 5.0
        assert np.array_equal(after.ledger.balance, expected)
        assert float(after.ledger.granted[2, 1]) == 5.0


def test_level_masks_match_the_table_and_are_nested():
    """Fails if a level drops an effect the level below permits, a mask departs from
    the adopted table, or the default level stops permitting every effect (which
    would change Slice 1 studies)."""
    for level, kinds_permitted in enumerate(ADOPTED):
        assert LEVEL_EFFECTS[level] == effect_mask(*kinds_permitted), level
    for lower, upper in itertools.pairwise(LEVEL_EFFECTS):
        assert lower & upper == lower
    agenda = builder.survey_1().agenda
    assert agenda.levels.tolist() == [len(LEVEL_EFFECTS) - 1]
    assert agenda.effective_effects().tolist() == [ALL_EFFECTS]


def test_effective_mask_is_the_phase_mask_and_the_level_mask():
    """Fails if admission reads the phase mask or the level mask alone, or applies
    one phase's level to another phase."""
    study = with_levels(
        builder.survey_1(
            policies=(kinds.NullPolicy(), kinds.NullPolicy(), kinds.NullPolicy()),
            phase_until=(300, 600, 1000),
        ),
        (0, 3, 1),
        permitted_effects=(ALL_EFFECTS, effect_mask(EffectKind.WAIT), ALL_EFFECTS),
    )
    assert study.agenda.effective_effects().tolist() == [
        LEVEL_EFFECTS[0],
        effect_mask(EffectKind.WAIT),
        LEVEL_EFFECTS[1],
    ]
    c0 = carry0(study)
    stop = effects_of(EffectKind.STOP)
    want = [AdmissionCode.FORBIDDEN, AdmissionCode.FORBIDDEN, AdmissionCode.OK]
    for tick, code in zip((0, 400, 700), want, strict=True):
        c = eqx.tree_at(lambda c: c.tick, c0, jnp.asarray(tick, jnp.int64))
        _, codes = admit_j(c, stop, study)
        assert int(codes[0]) == code, tick


@pytest.mark.parametrize("levels", [(4,), (-1,), (3, 3)])
def test_unknown_level_or_count_is_refused(levels):
    """Fails if a level outside the table is accepted (indexing would clip it to a
    real mask) or a phase has no level."""
    with pytest.raises(ValueError, match="level"):
        with_levels(builder.survey_1(), levels)
