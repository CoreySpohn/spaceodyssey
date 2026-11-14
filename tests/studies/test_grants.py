"""Observer grants declare the columns released and the summary kernel."""

import dataclasses

import numpy as np
import pytest

from spaceodyssey.codec import dump_study, load_study
from spaceodyssey.core.enums import Audience
from spaceodyssey.studies.bracket import (
    GRANT_PRESETS,
    ActivityTable,
    ArmsResult,
    Grants,
    bracket_boundary,
    bracket_registry,
    grants_in_force,
    grants_line,
)
from tests.codec.test_document import TOY_KINDS
from tests.criteria import test_criterion_contract
from tests.studies.bracket_helpers import (
    BAND,
    BUDGET_D,
    build,
    charged_days,
    columns_of,
    fixture_channel,
    fixture_source,
    fold_rows,
    spec_row,
    telescope_rows,
    unpack_state,
    with_grants,
)

base_study = test_criterion_contract.fixture_study


@pytest.fixture(scope="module")
def fixture_study(base_study):
    return dataclasses.replace(
        base_study,
        channel_model=fixture_channel(fixture_source()),
        observer=dataclasses.replace(
            base_study.observer, criterion=test_criterion_contract._TwoLeafCriterion()
        ),
    )


@pytest.mark.parametrize("name", sorted(GRANT_PRESETS))
def test_each_preset_opens_exactly_its_columns(name):
    """Fails if a preset releases a column it does not grant.

    Replaces the block-reveal audience test.
    """
    g = GRANT_PRESETS[name]
    m = columns_of(bracket_registry(g, 3, 2), "measurement")
    assert all(a == Audience.SCORER for c, a in m.items() if c.startswith("identity"))
    assert any(
        a & Audience.OBSERVER for c, a in m.items() if c.startswith("true_")
    ) == (g.delivery == "true_snr")
    assert bool(m["counts_0_0"] & Audience.POLICY) == g.policy_reads_measurements
    cat = columns_of(bracket_registry(g, 3, 2), "truth_catalog")
    for group, column in (
        ("existence", "present_0"),
        ("ephemeris", "params.ln_period_d_0"),
        ("class", "params.radius_rearth_0"),
    ):
        assert bool(cat[column] & Audience.OBSERVER) == (group in g.catalog_groups)
    true = columns_of(
        bracket_registry(Grants(delivery="true_snr"), 3, 2), "measurement"
    )
    assert true["identity_0"] == Audience.SCORER
    assert true["true_counts_0_0"] & Audience.OBSERVER


@pytest.mark.medium(
    reason="loads the shared physical channel and round-trips a study document"
)
def test_the_printed_line_is_decoded_from_the_document(fixture_study, base_study):
    """Fails if the grants line is a label rather than what the study holds."""
    with_model = dataclasses.replace(
        fixture_study, planning_model=fixture_study.channel_model
    )
    assert grants_in_force(fixture_study)["model"] == "truth"
    assert grants_in_force(with_model)["model"] == "declared"
    with_population = dataclasses.replace(
        fixture_study, belief_population=fixture_source().generator
    )
    assert grants_in_force(fixture_study)["population"] == "truth"
    assert grants_in_force(with_population)["population"] == "declared"
    shrink = with_grants(fixture_study, GRANT_PRESETS["shrinkage"])
    assert (
        grants_in_force(shrink)["kernel"] == "shrinkage"
        and bracket_boundary(shrink).kernel == "shrinkage"
    )
    observer = base_study.observer
    kinds = {
        **TOY_KINDS,
        **{
            ("test", type(x).__name__, 1): type(x)
            for x in (
                observer,
                observer.grants,
                observer.criterion,
                observer.prior,
                observer.mode_table,
                observer.mode_table.modes[0],
                observer.mode_table.modes[0].band,
            )
        },
    }
    original = dump_study(base_study, kinds)
    kernel_only = dataclasses.replace(
        base_study,
        observer=dataclasses.replace(observer, grants=Grants(kernel="shrinkage")),
    )
    changed = dump_study(kernel_only, kinds)
    assert changed.content_hash != original.content_hash
    loaded = load_study(changed.to_json(), kinds)
    assert loaded.observer.grants.kernel == "shrinkage"
    decoded = grants_in_force(shrink)
    without_paths = dataclasses.replace(
        shrink, caps=dataclasses.replace(shrink.caps, payload_paths=())
    )
    assert grants_in_force(without_paths) == decoded
    assert set(decoded) == {
        "association",
        "model",
        "population",
        "delivery",
        "catalog",
        "policy_reads",
        "kernel",
    }
    assert decoded["catalog"] == "ephemeris,class"
    relabeled = dataclasses.replace(
        shrink, observer=dataclasses.replace(shrink.observer, grants=Grants())
    )
    assert grants_in_force(relabeled)["catalog"] == "ephemeris,class"
    private = with_grants(fixture_study, GRANT_PRESETS["prior_planned"])
    private = dataclasses.replace(private, observer=fixture_study.observer)
    assert grants_in_force(private)["policy_reads"] == "no"
    true_signal = with_grants(fixture_study, Grants(delivery="true_snr"))
    true_signal = dataclasses.replace(true_signal, observer=fixture_study.observer)
    assert grants_in_force(true_signal)["delivery"] == "true_snr"
    assert grants_line(shrink) == " ".join(f"{k}={v}" for k, v in decoded.items())


@pytest.mark.medium(
    reason="compiles the release handler for measured and true signal delivery"
)
def test_true_snr_delivery_folds_the_true_columns(fixture_study):
    """Fails if `delivery: true_snr` folds measured counts, or vice versa."""
    row = spec_row(slot=0, counts=3.0, variance=1.0, true_counts=5.0, true_variance=2.0)
    for delivery, want, variance in (("measured", 3.0, 1.0), ("true_snr", 5.0, 2.0)):
        s = fold_rows(with_grants(fixture_study, Grants(delivery=delivery)), rows=[row])
        state = unpack_state(fixture_study, s)
        assert float(state.sum_counts[0, 0, BAND]) == want
        assert float(state.sum_variance[0, 0, BAND]) == variance


@pytest.mark.slow(reason="two short fixture runs on the chunked refresh driver")
def test_a_policy_without_measurements_still_spends_and_never_waits_on_a_refit():
    """Fails if `prior_planned` stalls waiting for a refit it can never read."""
    run = build(
        "prior_planned",
        budget_d=BUDGET_D,
        p_exist=0.3,
        n_round=2**9,
        n_rounds=1,
        n_centers=16,
        n_constants=1,
        max_rounds=2,
        n_first=64,
        n_particles=128,
        ess_floor=1.0,
    ).run
    assert charged_days(run) > 0 and telescope_rows(run) >= 2


def test_csv_exports_keep_each_studys_grants(fixture_study, tmp_path):
    """Fails if CSV export omits grants or overwrites another arm's grants."""
    line = grants_line(fixture_study)
    table = ActivityTable(
        star={"visited": np.ones((1, 1))},
        world={"charged": np.ones(1)},
        grants=line,
    )
    table.to_csv(tmp_path, "one")
    assert (tmp_path / "grants.txt").read_text() == f"one: {line}\n"
    other = dataclasses.replace(table, grants="kernel=shrinkage")
    ArmsResult(activity={"one": table, "two": other}).to_csv(tmp_path)
    assert (
        tmp_path / "grants.txt"
    ).read_text() == f"one: {line}\ntwo: kernel=shrinkage\n"
