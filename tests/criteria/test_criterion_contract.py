"""Criterion state is folded once per released measurement and survives refits."""

import dataclasses
import pathlib
import subprocess
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.adapters.measurement import measurement_template
from spaceodyssey.adapters.modes import ModeSpec, ModeTable
from spaceodyssey.adapters.photomancy import DecisionSummary
from spaceodyssey.adapters.skyscapes import Band
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.carry import init_carry
from spaceodyssey.core.enums import EVERYONE, F64, I64, RecordStatus
from spaceodyssey.core.events import empty_event_row
from spaceodyssey.core.handlers import h_release
from spaceodyssey.core.records import pack, record_none, unpack, write_log
from spaceodyssey.core.refresh import RefreshSchedule
from spaceodyssey.criteria.base import AbstractCriterion, StarOptions
from spaceodyssey.studies import bracket, builder

N_TRACKS, N_BANDS, BAND = 3, 2, 1


class _SumState(eqx.Module):
    sum_counts: jax.Array


class _TwoLeafState(_SumState):
    sum_variance: jax.Array


class _SumCriterion(AbstractCriterion):
    mode_names: tuple = eqx.field(static=True, default=("characterization",))
    labels: tuple = eqx.field(static=True, default=("met",))
    label_values: jax.Array = eqx.field(default_factory=lambda: jnp.ones(1))
    claims: tuple = ()

    def state_template(self, n_tracks, n_bands):
        return _SumState(jax.ShapeDtypeStruct((n_tracks, n_bands), F64))

    def init_state(self, n_tracks, n_bands):
        return _SumState(jnp.zeros((n_tracks, n_bands), F64))

    def update(self, state, row, modes):
        take = (row.assignment >= 0) & (row.valid_science > 0)
        take &= row.mode == modes._mode_index(self.mode_names[0])
        counts = state.sum_counts.at[row.assignment].add(
            jnp.where(take[:, None], row.counts, 0.0), mode="drop"
        )
        return dataclasses.replace(state, sum_counts=counts)

    def value(self, state, p_claim, tick):
        met = state.sum_counts.sum(-1) >= 6.0
        return met.astype(F64), jnp.where(met, 0, -1).astype(I64)

    def options(self, state, summary_star, tick, modes, settings):
        return StarOptions(
            jnp.zeros((1, 1)), jnp.zeros((1, 1, 1)), jnp.zeros((1, 1), I64)
        )


class _TwoLeafCriterion(_SumCriterion):
    def state_template(self, n_tracks, n_bands):
        leaf = jax.ShapeDtypeStruct((n_tracks, n_bands), F64)
        return _TwoLeafState(leaf, leaf)

    def init_state(self, n_tracks, n_bands):
        leaf = jnp.zeros((n_tracks, n_bands), F64)
        return _TwoLeafState(leaf, leaf)

    def update(self, state, row, modes):
        state = super().update(state, row, modes)
        variance = state.sum_variance.at[row.assignment].add(row.variance)
        return dataclasses.replace(state, sum_variance=variance)


def zeros_summary():
    return DecisionSummary.zeros(2, N_TRACKS, 1, 1, n_an=1)


@pytest.fixture(scope="module")
def fixture_study():
    modes = ModeTable(
        (
            ModeSpec("detection", "imaging", Band(500.0, 100.0), 0.2, 100.0, None, 7.0),
            ModeSpec(
                "characterization",
                "spectrograph",
                Band(910.0, 6.5),
                0.2,
                100.0,
                140.0,
                None,
            ),
        ),
        (1.0,),
    )
    observer = bracket.SummaryObserver(
        prior=zeros_summary(), criterion=_SumCriterion(), mode_table=modes
    )
    caps = bracket.bracket_registry("replan", N_TRACKS, N_BANDS).lower(
        Capacities(n_subjects=2, n_an=1, rows_per_subject=8, refit_lanes=2), observer
    )
    return builder.survey_1(
        caps,
        observer=observer,
        refresh=RefreshSchedule(100, 0, 90, 91, bracket.KIND_PRODUCT),
    )


def with_criterion(study, criterion):
    return dataclasses.replace(
        study, observer=dataclasses.replace(study.observer, criterion=criterion)
    )


def spec_row(slot, counts, variance):
    row = jax.tree.map(
        lambda x: jnp.zeros(x.shape, x.dtype), measurement_template(N_TRACKS, N_BANDS)
    )
    return dataclasses.replace(
        row,
        mode=jnp.asarray(1, I64),
        valid_science=jnp.asarray(1.0, F64),
        assignment=jnp.asarray([slot, -1, -1], I64),
        counts=row.counts.at[0, BAND].set(counts),
        variance=row.variance.at[0, BAND].set(variance),
        true_counts=row.true_counts + 1000.0,
        true_variance=row.true_variance + 1000.0,
    )


_release = eqx.filter_jit(h_release)


def fold_rows(study, rows):
    carry = init_carry(study, None)
    for i, row in enumerate(rows):
        rec = dataclasses.replace(
            record_none(study.caps),
            kind=jnp.asarray(bracket.KIND_MEASUREMENT, I64),
            valid=jnp.asarray(True),
            acquisition_id=jnp.asarray(i + 1, I64),
            audience=jnp.asarray(EVERYONE, I64),
            status=jnp.asarray(RecordStatus.COMPLETED, I64),
            released_tick=jnp.asarray(-1, I64),
            accepted_tick=jnp.asarray(-1, I64),
            superseded_by=jnp.asarray(-1, I64),
            payload=jnp.pad(pack(row), (0, study.caps.payload_width - pack(row).size)),
            channel_index=jnp.asarray(1, I64),
        )
        carry, _ = write_log(carry, rec, study.caps)
        event = dataclasses.replace(
            empty_event_row(),
            slot=jnp.asarray(i, I64),
            audience=jnp.asarray(EVERYONE, I64),
            tick=jnp.asarray(i + 1, I64),
        )
        for _ in range(2):
            carry = _release(carry, event, study, None, jax.random.key(0))[0]
    np.testing.assert_array_equal(carry.belief.state, carry.pbelief.state)
    return carry.pbelief


def unpack_state(study, summary):
    return unpack(
        study.observer.criterion.state_template(N_TRACKS, N_BANDS), summary.state
    )


def install(study, summary, fitted):
    return study.observer.install(summary, fitted, 0)


@pytest.mark.medium(reason="compiles the release handler for a two-star criterion fold")
def test_the_observer_folds_released_rows_into_criterion_state(fixture_study):
    """Fails if a row is folded twice, folded into another track, or lost at an
    install.
    """
    s0 = fold_rows(fixture_study, rows=[spec_row(slot=1, counts=3.0, variance=1.0)] * 2)
    state = unpack_state(fixture_study, s0)
    np.testing.assert_allclose(state.sum_counts[0, 1, BAND], 6.0)
    expected = np.zeros((2, N_TRACKS, N_BANDS))
    expected[0, 1, BAND] = 6.0
    np.testing.assert_array_equal(state.sum_counts, expected)
    after = install(fixture_study, s0, fitted=zeros_summary())
    np.testing.assert_array_equal(after.state, s0.state)


def test_a_new_criterion_changes_no_observer_layout(fixture_study):
    """Fails if a criterion's state leaves enter the observer's tree structure."""
    a, b = (
        with_criterion(fixture_study, c).observer.layout(fixture_study.caps).summary
        for c in (_SumCriterion(), _TwoLeafCriterion())
    )
    assert jax.tree.structure(a) == jax.tree.structure(b)
    assert b.state.shape[-1] == 2 * a.state.shape[-1]
    study = with_criterion(fixture_study, _TwoLeafCriterion())
    before = study.observer.init(study.caps)
    rec = dataclasses.replace(
        record_none(study.caps),
        valid=jnp.asarray(True),
        kind=jnp.asarray(bracket.KIND_MEASUREMENT, I64),
        status=jnp.asarray(RecordStatus.COMPLETED, I64),
        subject=jnp.asarray(1, I64),
        payload=pack(spec_row(2, 4.0, 2.0)),
    )
    after = eqx.filter_jit(study.observer.fold_summary)(before, rec, study.caps)
    state = unpack_state(study, after)
    np.testing.assert_allclose(state.sum_counts[1, 2, BAND], 4.0)
    np.testing.assert_allclose(state.sum_variance[1, 2, BAND], 2.0)
    np.testing.assert_array_equal(state.sum_counts[0], 0)
    expected_bytes = (
        sum(
            getattr(after, f.name).nbytes
            for f in dataclasses.fields(after)
            if f.name != "aux"
        )
        // 2
    )
    assert after.bytes_per_star() == expected_bytes
    ignored = dataclasses.replace(rec, kind=jnp.asarray(bracket.KIND_CATALOG, I64))
    np.testing.assert_array_equal(
        study.observer.fold(after, ignored, study.caps).state, after.state
    )


def test_plugins_import_without_the_physical_extra():
    """Fails if registering the criterion family pulls in a physical dependency."""
    blocked = (
        "skyscapes",
        "orbix",
        "numpyro",
        "jaxedith",
        "optixstuff",
        "yippy",
        "interpax",
        "astropy",
    )
    # Isolated Python ignores the worktree's PYTHONPATH; explicitly select its source.
    source = str(pathlib.Path(__file__).resolve().parents[2] / "src")
    code = (
        f"import sys; sys.path.insert(0, {source!r}); "
        f"sys.modules.update(dict.fromkeys({blocked!r})); "
        "import spaceodyssey.plugins as p; assert 'criterion' in p.FAMILIES"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)
