"""The complete Slice 1 acceptance gate; run this module without exclusions."""

# Some existing builders create capacities at import time. Scope their import and
# execution to x64 without changing the caller's process setting at collection.
import dataclasses
import re
import types

import jax
import numpy as np
import pytest

from spaceodyssey.core import (
    GO_TOLERANCE,
    KILL_TOLERANCE,
    Audience,
    CompiledExecutor,
    HostExecutor,
    RunStatus,
    battery,
    wake_by_audience,
)
from spaceodyssey.core.conformance import EXECUTORS, check_run, failures_of, run_case
from spaceodyssey.core.conformance.invariants import (
    no_extra_wakes_under_private_events,
    no_scorer_column_in_snapshot,
)
from spaceodyssey.core.study import Clock
from tests.core.contract_helpers import assert_study_typed
from tests.core.gate_helpers import stale_allowances, study_branches
from tests.core.lint_helpers import CORE, FIXTURES, fixture_core, run_lint

with jax.enable_x64():
    from spaceodyssey.studies.conformance import CORPUS, CORPUS_BY_NAME
    from tests.core import test_fork as fork_tests
    from tests.core import test_replay_log as replay_tests
    from tests.core.acquisition_helpers import drive
    from tests.core.battery_helpers import AbsolutePlanPolicy, survey
    from tests.core.conformance_helpers import runs
    from tests.core.corpus import LEAK_SCENARIOS
    from tests.core.draw_helpers import compare_draws, compare_policy_draws, run_pair
    from tests.core.helpers import worlds
    from tests.core.leak_helpers import historical_leaky_wake, leaky_wake, run_leak
    from tests.core.mutants import (
        PerWakeHostExecutor,
        decrementing_advance_jit,
        initial_belief_replay,
        live_belief_replay,
        log_order_replay,
    )
    from tests.core.parity_helpers import corpus_parity_reports
    from tests.core.record_helpers import with_leaking_layout
    from tests.core.status_helpers import (
        OVERFLOW_CASES,
        overflow_study,
        run_status_case,
    )

# Independent list: removing an invariant from check_run must break this gate.
EXPECTED_INVARIANTS = (
    "order_keys_monotone",
    "exactly_once_settlement",
    "one_fold_per_acquisition",
    "no_violation",
    "balance_conserved",
    "unique_acquisition_id",
    "dense_acquisition_index",
    "release_not_before_available",
    "no_scorer_column_in_snapshot",
    "no_extra_wakes_under_private_events",
    "refit_inputs_equal_revision_set",
    "installs_match_product_rows",
)


# Independent of the runner: the corpus runs two worlds, numbered from zero.
EXPECTED_WORLDS = (0, 1)
# Matched identities the draw controls must compare, from the known fixtures
# (two worlds; fewer would mean the comparison no longer reaches the draws).
MIN_MEASUREMENT_MATCHES = 10
MIN_POLICY_MATCHES = 12
# Measured changed draws of the mutants out of those matches, on both executors.
CHANGED_DRAWS = {"measurement": 8, "policy": 10}

# Independent of the study: the payload columns only the scorer may read, by record
# kind index (1 is the toy measurement, whose second column is the raw noise variate;
# 3 is the catalog record of its case). Every other column is readable by everyone.
SCORER_ONLY_COLUMNS = {"default": {1: (1,)}, "catalog_record": {1: (1,), 3: (1,)}}
N_KINDS, N_COLUMNS = 4, 4


def reference_layout(name):
    """The layout that judges snapshots, written here and not read from the study."""
    table = np.full((N_KINDS, N_COLUMNS), 7, dtype=np.int64)
    for kind, columns in SCORER_ONLY_COLUMNS.get(
        name, SCORER_ONLY_COLUMNS["default"]
    ).items():
        table[kind, list(columns)] = int(Audience.SCORER)
    return types.SimpleNamespace(payload_audience=table)


def corpus_failures(corpus_runs, name):
    """(world, failed results) of corpus runs judged against the independent layout."""
    out = []
    for run in corpus_runs:
        run = dataclasses.replace(run, reference_caps=reference_layout(name))
        results = check_run(run)
        assert tuple(result.name for result in results) == EXPECTED_INVARIANTS
        out.append((run.world, failures_of(results)))
    return out


@pytest.fixture(scope="module", autouse=True)
def enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.parametrize("check", ["K1a"])
def test_no_study_dispatch(check):
    assert not study_branches(CORE)
    assert not stale_allowances(CORE)


@pytest.mark.parametrize("check", ["K1b"])
def test_typed_composition(check):
    assert_study_typed()


@pytest.mark.parametrize("check", ["K2a"])
@pytest.mark.parametrize("tool", ["check_core_imports.py", "check_domain_words.py"])
def test_core_boundaries(check, tool):
    result = run_lint(tool)
    assert result.returncode == 0, result.stdout + result.stderr
    count = re.search(r"Scanned (\d+) Python files", result.stdout)
    assert count and int(count[1]) == len(list(CORE.rglob("*.py"))) > 0


@pytest.mark.parametrize("check", ["K3"])
@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.name)
def test_corpus_invariants(check, case, executor):
    corpus_runs = runs(case.name, executor)
    worlds = sorted(run.world for run in corpus_runs)
    assert worlds == list(EXPECTED_WORLDS), (
        f"world IDs {worlds} differ from the expected {list(EXPECTED_WORLDS)}",
        case.name,
        executor,
    )
    for world, failures in corpus_failures(corpus_runs, case.name):
        assert not failures, (case.name, executor, world, failures)


@pytest.mark.parametrize("executor", EXECUTORS)
def test_negative_study_layout_leak(executor):
    """A study whose own layout shows the variate fails against the independent one."""
    case = CORPUS_BY_NAME["baseline"]
    leaky = dataclasses.replace(
        case, make_study=lambda: with_leaking_layout(case.make_study())
    )
    (run,) = run_case(leaky, executor, n_worlds=1)
    ((_, failures),) = corpus_failures([run], "baseline")
    assert [f.name for f in failures] == ["no_scorer_column_in_snapshot"], failures


@pytest.mark.parametrize("check", ["A2"])
@pytest.mark.parametrize("case", replay_tests.RELEASING, ids=lambda c: c.name)
def test_replay_acceptance(check, case):
    """The focused A2 check: host-log replay reproduces the belief bit for bit."""
    assert replay_tests.replay_matches_live(case), case.name


@pytest.mark.parametrize("check", ["A2"])
def test_replay_acceptance_controls(check):
    """Replay follows the log and the release order, so it is not the live belief."""
    assert replay_tests.replay_follows_the_log()
    assert replay_tests.reordered_replay_matches()


@pytest.mark.parametrize("check", ["A2-negative"])
class TestReplayNegativeControls:
    """Each broken replay must fail the check it is meant to fail."""

    def test_real_replay_passes_all_controls(self, check):
        assert replay_tests.replay_follows_the_log()
        assert replay_tests.reordered_replay_matches()

    @pytest.mark.parametrize("name", ["baseline", "revision"])
    def test_replay_that_ignores_the_run_fails(self, check, name):
        case = CORPUS_BY_NAME[name]
        assert not replay_tests.replay_matches_live(case, initial_belief_replay)

    def test_replay_returning_the_live_belief_fails(self, check):
        # it equals the live belief by construction, so only the log edit exposes it
        assert replay_tests.replay_matches_live(
            CORPUS_BY_NAME["baseline"], live_belief_replay
        )
        assert not replay_tests.replay_follows_the_log(live_belief_replay)

    def test_replay_ignoring_the_log_fails(self, check):
        assert not replay_tests.replay_follows_the_log(initial_belief_replay)

    def test_replay_in_log_order_fails(self, check):
        assert not replay_tests.reordered_replay_matches(log_order_replay)


@pytest.mark.parametrize("check", ["A3"])
def test_fork_acceptance(check):
    """The focused A3 checks: default coupling, then declared coupling."""
    fork_tests.test_fork(worlds(2))
    fork_tests.test_fork_shares_later_draws_only_where_declared(worlds(2))


@pytest.mark.parametrize("check", ["A4"])
def test_latency_acceptance(check):
    """The focused A4 check: a two-day delay moves belief updates only."""
    study = dataclasses.replace(survey(AbsolutePlanPolicy()), clock=Clock(86400.0))
    truths, roots = worlds(2)
    result = battery.latency_shift(study, truths, roots, 40, delay_s=2 * 86400.0)
    assert result, result.diagnostic
    assert result.detail["delay_ticks"] == 2 and result.detail["moved"] >= 2


@pytest.mark.parametrize("check", ["K4-parity"])
@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.name)
def test_executor_parity(check, case):
    assert GO_TOLERANCE == 1e-12 and KILL_TOLERANCE == 1e-9
    reports = corpus_parity_reports(case)
    assert reports
    worst = max(report.max_abs for _, _, report in reports)
    go = all(report.go for _, _, report in reports)
    # Print separately so `-rP` reports the tighter band without making it the cutoff.
    print(f"{case.name}: go_band_1e-12={go}, max_abs={worst:.17g}")
    for form, world, report in reports:
        where = (case.name, form, world, report)
        assert report.structure_equal and report.max_int_diff == 0, where
        assert not report.vacuous and report.max_abs <= 1e-9 and report.ok, where


@pytest.mark.parametrize("check", ["K4-overflow"])
@pytest.mark.parametrize(
    "executor", [HostExecutor, CompiledExecutor], ids=lambda c: c.__name__
)
@pytest.mark.parametrize("name", sorted(OVERFLOW_CASES))
def test_overflow_status(check, executor, name):
    caps, bit = OVERFLOW_CASES[name]
    result = run_status_case(executor, overflow_study(name, caps), caps)
    assert result.status.tolist() == [RunStatus.OVERFLOW] * 2
    assert not result.complete.any()
    assert result.carry.overflow_mask.tolist() == [int(bit)] * 2
    for world in range(2):
        history = result.history(world)
        assert history.status == RunStatus.OVERFLOW and not history.complete
        assert history.overflow_mask == bit


@pytest.mark.parametrize("executor", EXECUTORS)
def test_negative_leaky_wake_invariant(executor):
    (run,) = run_case(
        CORPUS_BY_NAME["baseline"], executor, n_worlds=1, probe_wake_fn=leaky_wake
    )
    result = no_extra_wakes_under_private_events(run)
    assert no_extra_wakes_under_private_events(runs("baseline", executor)[0]).passed
    assert result.name == "no_extra_wakes_under_private_events"
    assert not result.passed and result.diagnostic


@pytest.mark.parametrize(
    ("scenario", "executor", "wake_fn", "extra_wakes"),
    [
        pytest.param(
            "heartbeat", HostExecutor, leaky_wake, 80, id="current-heartbeat-80"
        ),
        pytest.param(
            "private_writer", HostExecutor, leaky_wake, 80, id="current-writer-80"
        ),
        pytest.param(
            "heartbeat",
            PerWakeHostExecutor,
            historical_leaky_wake,
            80,
            id="historical-heartbeat-80",
        ),
        pytest.param(
            "private_writer",
            PerWakeHostExecutor,
            historical_leaky_wake,
            160,
            id="historical-writer-160",
        ),
    ],
)
def test_negative_leaky_wake_counts(scenario, executor, wake_fn, extra_wakes):
    process = LEAK_SCENARIOS[scenario]
    clean = run_leak(process, wake_by_audience, n_worlds=8, executor=executor)
    assert clean.wakes == clean.wakes_baseline and clean.views_equal
    broken = run_leak(process, wake_fn, n_worlds=8, executor=executor)
    assert broken.n_private_events > 0
    assert broken.wakes - broken.wakes_baseline == extra_wakes
    assert broken.wakes != broken.wakes_baseline, "wake isolation must fail"


@pytest.mark.parametrize("executor", ["host", "scan"])
@pytest.mark.parametrize("mutant", ["measurement", "policy"])
def test_negative_decision_count_keys(executor, mutant):
    policy = mutant == "policy"
    name = "insert_wait_alongside" if policy else "insert_wait"
    compare = compare_policy_draws if policy else compare_draws
    minimum = MIN_POLICY_MATCHES if policy else MIN_MEASUREMENT_MATCHES
    base, other = run_pair(name, executor, n_seeds=2)
    _, _, changed = compare(base, other, minimum)
    assert changed == 0
    base, other = run_pair(
        name, executor, n_seeds=2, mutant=not policy, policy_mutant=policy
    )
    matched, _, changed = compare(base, other, minimum)
    assert matched >= minimum
    assert changed == CHANGED_DRAWS[mutant], "draw invariance must fail"


@pytest.mark.parametrize("executor", EXECUTORS)
def test_negative_leaking_payload(executor):
    run = runs("baseline", executor)[0]
    assert no_scorer_column_in_snapshot(run).passed
    broken = dataclasses.replace(
        run, study=with_leaking_layout(run.study), reference_caps=run.study.caps
    )
    result = no_scorer_column_in_snapshot(broken)
    assert result.name == "no_scorer_column_in_snapshot"
    assert not result.passed and "snapshot shows a hidden column" in result.diagnostic


def test_negative_decrement():
    ops = [("observe", 0, 0), ("cancel", 0), ("observe", 0, 0)]
    assert drive(ops, steps=[0, 3, 0]) == []
    problems = drive(ops, advance=decrementing_advance_jit, steps=[0, 3, 0])
    assert any(problem[1] == "decrement" for problem in problems), problems
    assert any(problem[0] == "dense" for problem in problems), problems


@pytest.mark.parametrize(
    ("fixture", "token"),
    [
        ("identifier", "planet"),
        ("docstring", "telescope"),
        ("comment", "orbit"),
        ("error_message", "exposure"),
        ("preflight_sentence", "target"),
        ("keyword", "band"),
        ("plural", "facilities"),
    ],
)
def test_negative_domain_lint(tmp_path, fixture, token):
    root, path = fixture_core(tmp_path, (FIXTURES / f"{fixture}.txt").read_text())
    result = run_lint("check_domain_words.py", "--core-dir", root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert re.search(rf"{re.escape(str(path))}:\d+:{token}\b", result.stdout)


@pytest.mark.parametrize("fixture", ["import_direct", "import_from"])
def test_negative_import_lint(tmp_path, fixture):
    root, path = fixture_core(tmp_path, (FIXTURES / f"{fixture}.txt").read_text())
    result = run_lint("check_core_imports.py", "--core-dir", root)
    library = "orbix" if fixture == "import_direct" else "photomancy"
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"{path}:1:{library} is not allowed" in result.stdout


@pytest.mark.parametrize(
    ("filename", "source", "rule"),
    [
        ("sample.py", 'if name == "survey_1":\n    pass\n', "survey-name"),
        ("sample.py", 'if "survey 25" in names:\n    pass\n', "survey-name"),
        ("sample.py", "if study.name == other:\n    pass\n", "study.name"),
        (
            "handlers.py",
            "if isinstance(study.channel_model, ToyPhotometry):\n    pass\n",
            "study-kind",
        ),
        (
            "handlers.py",
            "if isinstance(model, ToyPhotometry):\n    pass\n",
            "study-kind",
        ),
        (
            "handlers.py",
            "if builtins.isinstance(study.channel_model, ToyPhotometry):\n    pass\n",
            "study-kind",
        ),
        (
            "transaction.py",
            "if isinstance(policy, CustomPolicy):\n    pass\n",
            "study-kind",
        ),
    ],
    ids=[
        "survey-literal",
        "survey-membership",
        "study-name",
        "handler-kind",
        "concrete-kind",
        "qualified-kind",
        "transaction-kind",
    ],
)
def test_negative_ast(tmp_path, filename, source, rule):
    path = tmp_path / filename
    path.write_text(source, encoding="ascii")
    assert study_branches(tmp_path) == [(path, 1, rule)]


def test_negative_ast_import_alias(tmp_path):
    path = tmp_path / "transaction.py"
    path.write_text(
        "from spaceodyssey.studies.toy import ToyPhotometry as Channel\n"
        "if isinstance(model, Channel):\n    pass\n",
        encoding="ascii",
    )
    assert study_branches(tmp_path) == [(path, 2, "study-kind")]


def test_ast_allows_ordinary_comparisons_and_data_checks(tmp_path):
    (tmp_path / "handlers.py").write_text(
        'text = "label"\n'
        'if event.name == "release" and isinstance(count, int):\n    pass\n',
        encoding="ascii",
    )
    assert study_branches(tmp_path) == []


def test_negative_ast_local_study_alias(tmp_path):
    path = tmp_path / "handlers.py"
    path.write_text(
        "policy = study.agenda.policies[0]\n"
        "if isinstance(policy, CustomPolicy):\n    pass\n",
        encoding="ascii",
    )
    assert study_branches(tmp_path) == [(path, 2, "study-kind")]
