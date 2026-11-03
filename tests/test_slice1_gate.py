"""The complete Slice 1 acceptance gate; run this module without exclusions."""

# Some existing builders create capacities at import time. Scope their import and
# execution to x64 without changing the caller's process setting at collection.
import dataclasses
import re

import jax
import pytest

from spaceodyssey.core import (
    GO_TOLERANCE,
    KILL_TOLERANCE,
    CompiledExecutor,
    HostExecutor,
    RunStatus,
    wake_by_audience,
)
from spaceodyssey.core.conformance import EXECUTORS, check_run, failures_of, run_case
from spaceodyssey.core.conformance.invariants import (
    no_extra_wakes_under_private_events,
    no_scorer_column_in_snapshot,
)
from tests.core.contract_helpers import assert_study_typed
from tests.core.gate_helpers import study_branches
from tests.core.lint_helpers import CORE, FIXTURES, fixture_core, run_lint

with jax.enable_x64():
    from spaceodyssey.studies.conformance import CORPUS, CORPUS_BY_NAME
    from tests.core.acquisition_helpers import drive
    from tests.core.conformance_helpers import runs
    from tests.core.corpus import LEAK_SCENARIOS
    from tests.core.draw_helpers import compare_draws, compare_policy_draws, run_pair
    from tests.core.leak_helpers import historical_leaky_wake, leaky_wake, run_leak
    from tests.core.mutants import PerWakeHostExecutor, decrementing_advance_jit
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
)


@pytest.fixture(scope="module", autouse=True)
def enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.parametrize("check", ["K1a"])
def test_no_study_dispatch(check):
    assert not study_branches(CORE)


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
    for run in runs(case.name, executor):
        results = check_run(run)
        assert tuple(result.name for result in results) == EXPECTED_INVARIANTS
        assert not failures_of(results), (case.name, executor, run.world, results)


@pytest.mark.parametrize("check", ["K4-parity"])
@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.name)
def test_executor_parity(check, case, record_property):
    assert GO_TOLERANCE == 1e-12 and KILL_TOLERANCE == 1e-9
    reports = corpus_parity_reports(case)
    assert reports
    worst = max(report.max_abs for _, _, report in reports)
    go = all(report.go for _, _, report in reports)
    record_property("go_band_1e-12", go)
    record_property("max_abs", worst)
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
    base, other = run_pair(name, executor, n_seeds=2)
    compared, _, changed = compare(base, other)
    assert compared > 0 and changed == 0
    base, other = run_pair(
        name, executor, n_seeds=2, mutant=not policy, policy_mutant=policy
    )
    compared, _, changed = compare(base, other)
    assert compared > 0 and changed > 0, "draw invariance must fail"


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
            "if isinstance(policy, AbstractPolicy):\n    pass\n",
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
        'text = "survey_1"\n'
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
