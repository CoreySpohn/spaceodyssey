"""Ladder execution shares programs and noise while keeping truth private."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.testbudget.compiles import cache_hit_count, compile_count

from spaceodyssey.core.battery.result import tree_bits_equal
from spaceodyssey.studies import bracket as br
from tests.studies import test_ladder_documents as documents
from tests.studies.ladder_helpers import run

document_union_study = documents.union_study
single_step_910 = documents.single_step_910


def example_table(requirement=(5.0, 5.0)):
    return dataclasses.replace(
        documents.example_table(), requirement_snr=jnp.asarray(requirement)
    )


@pytest.fixture(scope="module")
def union_study(document_union_study):
    base = document_union_study
    return br.bracket_study(
        base.world_source,
        base.channel_model,
        horizon_d=30.0,
        budget_d=3.0,
        observer=dataclasses.replace(base.observer, grants=br.GRANT_PRESETS["oracle"]),
        policy=br.PackagePolicy(max_visits=3, max_spectra=3),
    )


def compiles_during(call):
    """Count backend programs, including persistent-cache loads, during one call."""
    before = compile_count() + cache_hit_count()
    jax.block_until_ready(call())
    return compile_count() + cache_hit_count() - before


def runs_equal(a, b):
    return tree_bits_equal((a.carry, a.trace, a.status), (b.carry, b.trace, b.status))


def row_log(result, study):
    """Attach the executor's band ordinal to the scorer's completed rows."""
    rows = br.row_log(result, study)
    modes, log = study.channel_model.mode_table, result.carry.log
    rows["band"] = np.asarray(
        [modes.band_index(modes.modes[m].name) for m in rows["mode"]], int
    )
    ordinals = []
    for w, s, t, c, b in zip(
        rows["world"],
        rows["star"],
        rows["end_tick"],
        rows["channel"],
        rows["band"],
        strict=True,
    ):
        hit = np.flatnonzero(
            np.asarray(log.valid[w])
            & (np.asarray(log.subject[w]) == s)
            & (np.asarray(log.end_tick[w]) == t)
            & (np.asarray(log.channel_index[w]) == c)
        )
        assert hit.size == 1
        ordinals.append(int(log.group_ordinal[w, hit[0], b]))
    rows["ordinal"] = np.asarray(ordinals, int)
    return rows


def matched_unit_variates(a, b):
    """Join uncensored count residuals by world, host, identity, band and ordinal."""

    def indexed(rows):
        out = {}
        for r, band in enumerate(rows["band"]):
            for slot, identity in enumerate(rows["identity"][r]):
                variance = rows["true_variance"][r, slot, band]
                counts = rows["counts"][r, slot, band]
                truth = rows["true_counts"][r, slot, band]
                if variance <= 0 or not np.isfinite([variance, counts, truth]).all():
                    continue
                key = (
                    rows["world"][r],
                    rows["star"][r],
                    identity,
                    band,
                    rows["ordinal"][r],
                )
                assert key not in out
                out[key] = (counts - truth) / np.sqrt(variance)
        return out

    left, right = indexed(a), indexed(b)
    shared = sorted(left.keys() & right.keys())
    return np.asarray([left[k] for k in shared]), np.asarray([right[k] for k in shared])


@pytest.mark.slow(reason="two ladder entries on one compiled program")
def test_a_traced_ladder_compiles_once_and_is_deterministic(union_study):
    """Fails if a traced requirement recompiles or a rerun differs."""
    a, b = br.ladder_studies(
        union_study,
        (example_table(requirement=(5.0, 5.0)), example_table(requirement=(7.0, 5.0))),
    )
    n0 = compiles_during(lambda: run(a))
    n1 = compiles_during(lambda: run(b))
    assert n0 >= 1 and n1 == 0 and runs_equal(run(a), run(a))


@pytest.mark.slow(reason="two ladder entries that diverge in sequence")
def test_ladder_entries_share_the_draw_of_each_observation_identity(union_study):
    """Fails if two entries draw different noise for the same star, slot, band
    and ordinal.
    """
    a, b = (
        row_log(run(s), s)
        for s in br.ladder_studies(union_study, (example_table(), single_step_910()))
    )
    assert a["mode"].tolist() != b["mode"].tolist()
    za, zb = matched_unit_variates(a, b)
    assert za.size > 0
    np.testing.assert_allclose(za, zb, rtol=1e-12)
