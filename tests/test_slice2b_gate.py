"""The Slice 2b kill-rule gate: K1 to K5 as one command, and the K6 and K7 wiring.

Run ``pytest tests/test_slice2b_gate.py`` with the full commit history (K2 reads it).

- K1, Slice 1 stays true: the golden histories, fingerprints and fault fixture match
  ``tests/core/golden_pin.json``, the pins name their source commit, and no commit of
  the slice touched a pin file.
- K2, core patches: every Slice 2b commit touches only the allowed paths.
- K3, overlap: the three named overlap checks.
- K4, codec drift: the schema version and every corpus document hash match
  ``tests/codec/hash_pin.json``; the table round-trips through file references,
  tampering is refused and lineage passes and fails.
- K5, the core frame: the domain-word and import lints pass over the core.
- K6 and K7 are enforced by the quick-tier CI step (``ci_summary.py --baseline
  --max-increase-seconds 60 --budget-seconds 540 --require-tier medium``); this gate
  checks that the step carries them and that the script fails as it should.

Named interface checks whose dependency has not merged are listed as pending and fail
the gate; none is skipped.

Which commits are Slice 2b. The range runs from the pins' source commit (recorded in
the R0a change record) to the newest commit that touches a Slice 2b change record:
every Slice 2b task ends with its record, so the slice ends there and later work on
the branch is not judged here. Other work that reached the same history inside that
range (the physical PT4 and PT6 tasks and Slice 3a S1 and S2) is listed below by full
sha with its subject and its own change record; a stale entry, an entry outside the
range, or an entry that touches a Slice 2b file without a listed reason fails. Merge
commits carry no change of their own: each one's combined diff must be empty, and the
commits it brought in are judged one by one.
"""

import importlib
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from tests import conftest
from tests.codec import test_premission_codec as codec_tests
from tests.codec import test_run_document_runs as run_document_tests
from tests.core import test_golden_histories as golden
from tests.core import test_lints as lints
from tests.core import test_premission as premission_tests
from tests.core import test_premission_refit as refit_tests

pytestmark = pytest.mark.clear_caches_between_tests

REPO = Path(__file__).resolve().parents[1]
GOLDEN_PIN = REPO / "tests/core/golden_pin.json"
HASH_PIN = REPO / "tests/codec/hash_pin.json"
CI = REPO / ".github/workflows/ci.yml"
CI_SUMMARY = REPO / ".github/scripts/ci_summary.py"
BASELINE = REPO / ".github/budget/quick_baseline.json"

# The pins' source commit (R0a) and the R0a commits that captured each pin file.
PIN_SOURCE = "96a644728386365177e79f8bca315a746ef2901c"
PIN_CAPTURES = {
    "5b9fbd5f35d2d6be6286f178445fea7c2381d7f3",  # golden_pin.json
    "fc0e2e4ae257c3fef7282747927dac7a03d5a928",  # hash_pin.json
}
RECORDS = "docs/changes/slice-2b-*.md"

# ---- K1 ----------------------------------------------------------------------------

test_k1_pin_names_its_commit = (
    golden.test_the_pin_names_its_commit_and_covers_every_entry
)
test_k1_manifest_equals_pin = golden.test_the_manifest_equals_its_pin
test_k1_history_digest = golden.test_history_matches_its_golden_digest
test_k1_fingerprints = golden.test_fingerprints_match_their_pin
test_k1_fault_fixture = golden.test_the_fault_fixture_matches_its_pin


def test_k1_pins_name_the_source_commit_and_no_slice_commit_touched_them():
    """Fails if a pin names another source commit, or if any commit of the slice
    range other than the two R0a captures changed a pin file.
    """
    assert json.loads(GOLDEN_PIN.read_text())["source_commit"] == PIN_SOURCE
    assert json.loads(HASH_PIN.read_text())["source_commit"] == PIN_SOURCE
    touched = set(
        _git(
            "log",
            "--format=%H",
            f"{PIN_SOURCE}..{slice_end()}",
            "--",
            str(GOLDEN_PIN.relative_to(REPO)),
            str(HASH_PIN.relative_to(REPO)),
        ).split()
    )
    assert touched == PIN_CAPTURES, sorted(touched - PIN_CAPTURES)


# ---- K2 ----------------------------------------------------------------------------

# Paths a Slice 2b commit may touch (spec section 9, K2): the conformance tooling,
# the narrow pre-flight hook, studies, codec and tests, the change records, and the
# CI files of the budget enforcement (Q9): the budget configuration, the workflows
# that run the tests under it, the summary script and its baseline, and the K6
# measurement workflow.
ALLOWED = (
    "src/spaceodyssey/core/battery/",
    "src/spaceodyssey/core/conformance/",
    "src/spaceodyssey/core/preflight/findings.py",
    "src/spaceodyssey/core/preflight/checks.py",
    "src/spaceodyssey/studies/",
    "src/spaceodyssey/codec/",
    "tests/",
    "docs/changes/slice-2b-",
    "pyproject.toml",
    ".github/workflows/ci.yml",
    ".github/scripts/ci_summary.py",
    ".github/budget/",
    ".github/workflows/pilot.yml",
    ".github/scripts/pilot_summary.py",
    ".github/workflows/upstream.yml",
)

# Reviewed exceptions, each for one commit only.
COMMIT_ALLOWANCES = {
    # R0a, shared with Slice 3a S0: the step-cost probe, its record and a word list
    "4a0e39525178b53be5981b3226b1cf9d758f3b9a": {"tools/step_cost.py"},
    "b6c6c08213f0b84a63dc63bd9b33c873b48f107d": {"docs/changes/slice-3a-s0.md"},
    "97753d57564770664fca3b73c26840d0681dacf2": {
        ".codespellrc",
        "docs/changes/slice-3a-s0.md",
        "tools/step_cost.py",
    },
    # R2: the prior test helpers are a support module for the test-name hook
    "93a12afae99b74a223526540eeb2eb4d02a992a5": {".pre-commit-config.yaml"},
}

# Work of other tasks inside the range: full sha -> (task, record, subject prefix).
NOT_SLICE_2B = {
    "a6259078f1b66475fa51df9bcdcb48b276be4d2e": (
        "PT4",
        "physical-pt4.md",
        "fix(adapters): cadence codes as the profile table states them",
    ),
    "b31cf737292807f4034aadd22ee53684eb7c2ec0": (
        "PT4",
        "physical-pt4.md",
        "feat(adapters): summary reporting law, record kind and calibration id",
    ),
    "533ca85ccb380bc02aa0e28148a961a35866fd2e": (
        "PT4",
        "physical-pt4.md",
        "test(adapters): score the B4 refinement with the reporting law",
    ),
    "9449fe28b4e55573e51231b64c28bc1a4b511bb8": (
        "PT4",
        "physical-pt4.md",
        "docs(changes): physical PT4 reporting law record",
    ),
    "88a042ca777e8bb9a020086ca85567aa8d6de9a1": (
        "PT4",
        "physical-pt4.md",
        "fix(adapters): calibration id covers every budget declaration, per-law "
        "survey ids, Monte Carlo panels slow",
    ),
    "64003ee76586f1cece6200d36c759c4b3721891b": (
        "PT4",
        "physical-pt4.md",
        "docs(adapters): literal separation-rate bound in the support docstring",
    ),
    "6ec9e9a0a6ffbd41fab03ae420addb8895bbce80": (
        "PT4",
        "physical-pt4.md",
        "build(adapters): merged library pins, yippy 2.12 floor, fixed-frame default "
        "cadence, blocked anchors become ordinary",
    ),
    "1e83ab69eb91b4ab227167deef29721efe709910": (
        "PT4",
        "physical-pt4.md",
        "docs(changes): PT4 review fixes and library merges",
    ),
    "925e8f8c2e6c360e74a8b73c62058b247d14672e": (
        "PT6",
        "pt6-ofti.md",
        "feat(studies): OFTI stand-in refit of physical summary rows",
    ),
    "7d3e094b7a11be12a5dcb75990972a8564e482a6": (
        "PT6",
        "pt6-ofti.md",
        "test(studies): OFTI stand-in anchors, one real fit, contract",
    ),
    "68891dbb2c1adbf00fafb3b086def4ba5f90ccb3": (
        "PT6",
        "pt6-ofti.md",
        "docs(changes): physical PT6 OFTI stand-in refit record",
    ),
    "d3b24b0b3ba6668b3e3f2cabba3e4c87cb32461b": (
        "PT6",
        "pt6-ofti.md",
        "fix(studies): readable stand-in scope in saved run documents",
    ),
    "6e3ced6c4dcea8b5ea963d6ddd027f69c9e5c899": (
        "PT6",
        "pt6-ofti.md",
        "docs(changes): PT6 review fixes, single-seed survey note and gate results",
    ),
    "f155d0b8c2a4e5db1a0cd5e77e3750eaf7e1c04b": (
        "3a S1",
        "slice-3a-s1.md",
        "feat(core): slice 3a capacities, conditional leaf tables, op",
    ),
    "ab7b1a6b661c2b9c798c93017615167414f33fe8": (
        "3a S1",
        "slice-3a-s1.md",
        "test(core): slice 3a guards, optional codec fields and key_version tests",
    ),
    "1e65b7e41143b2207bd53b6edd83952a05b141e7": (
        "3a S1",
        "slice-3a-s1.md",
        "docs(changes): slice 3a S1 change record",
    ),
    "17976f4539df596208277f0bfe502cc3d4de0183": (
        "3a S1",
        "slice-3a-s1.md",
        "fix(codec): compare tuple defaults element-wise; test(core):",
    ),
    "586e259f3f66647fe98ae3c7d4a8a5ae6445b983": (
        "3a S1",
        "slice-3a-s1.md",
        "test(core): move the idle-asset coordinate guard to the unit tier",
    ),
    "624f25e69f87d81afb90988ba645aaa4503722de": (
        "3a S1",
        "slice-3a-s1.md",
        "fix(core): key_version is a locatable study slot for pre-flight findings",
    ),
    "c61f43ce3b928b2f363547d2500f2af0ce9d0e00": (
        "3a S2",
        "slice-3a-s2.md",
        "feat(core): programs per activity, per-program Stop and P001",
    ),
    "538d2ec0cf1cf21d6fbc4d8abac91b004f98215d": (
        "3a S2",
        "slice-3a-s2.md",
        "fix(core): optional program field, default owner and slow mutants",
    ),
    "794a7157d7621b4285ef8eb65c89dedb08661570": (
        "3a S2",
        "slice-3a-s2.md",
        "test(core): mutate every conditional leaf on a two-program refresh",
    ),
}

# Slice 2b files another task had to adapt, each for one commit, with the reason.
SHARED_EDITS = {
    # a log column the capacities do not hold is None; the export test's log helper
    # skips it
    "794a7157d7621b4285ef8eb65c89dedb08661570": {
        "tests/codec/test_premission_codec.py"
    },
}

# Files only Slice 2b writes; a commit listed as other work must touch none of them.
SLICE_2B_FILES = re.compile(
    r"^(src/spaceodyssey/studies/(premission|prior_observer|precursor|survey_(3|4|6|18))"
    r"\.py|tests/(core/(test_premission\w*|prior_helpers|test_epoch_relation)"
    r"|codec/test_premission_codec|studies/test_survey_(3|4|6|18))\.py"
    r"|docs/changes/slice-2b-.*)$"
)


def _git(*args):
    """Output of a git command in the repository; fails the test without history."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.fail(
            f"git {' '.join(args)} failed ({error}); the gate needs the repository "
            "with its full history (checkout with fetch-depth 0)"
        )
    return result.stdout


def slice_end():
    """The newest commit that touches a Slice 2b change record."""
    end = _git("log", "-1", "--format=%H", "HEAD", "--", RECORDS).strip()
    assert end, "no Slice 2b change record in the history"
    return end


def _range(*extra):
    return _git("rev-list", *extra, f"{PIN_SOURCE}..{slice_end()}").split()


def _subject(sha):
    return _git("log", "-1", "--format=%s", sha).strip()


def _files(sha):
    return _git("show", "--no-renames", "--name-only", "--format=", sha).split()


def _resolve(short):
    return _git("rev-parse", "--verify", f"{short}^{{commit}}").strip()


def not_slice_2b():
    """The listed other-task commits, by full sha."""
    return {_resolve(sha): entry for sha, entry in NOT_SLICE_2B.items()}


def slice_commits():
    """Full shas of the Slice 2b commits in the range, merges excluded."""
    other = not_slice_2b()
    return [sha for sha in _range("--no-merges") if sha not in other]


def violations(shas):
    """Map each commit to the paths it touches outside ALLOWED and its allowances."""
    found = {}
    for sha in shas:
        allowed = COMMIT_ALLOWANCES.get(sha, set())
        bad = [
            path
            for path in _files(sha)
            if not path.startswith(ALLOWED) and path not in allowed
        ]
        if bad:
            found[f"{sha[:7]} {_subject(sha)}"] = bad
    return found


def test_k2_commits_touch_only_allowed_paths():
    """K2. Fails if a Slice 2b commit touches any path outside the allowed list: any
    other core path, or an unreviewed file elsewhere.
    """
    shas = slice_commits()
    # evidence: the range holds the slice's work (R0a to R5 and A10 at least)
    assert len(shas) >= 60, len(shas)
    assert violations(shas) == {}


def test_k2_other_task_commits_are_in_range_and_touch_no_slice_file():
    """Fails if a listed other-task commit is outside the range, its subject or
    record differs from the list, or it touches a file only Slice 2b writes (which
    would hide 2b work from K2); fails if an allowance names a commit outside the
    range.
    """
    in_range = set(_range("--no-merges"))
    tasks = set()
    for sha, (task, record, subject) in not_slice_2b().items():
        assert sha in in_range, (task, sha)
        if subject:
            assert _subject(sha).startswith(subject), sha
        assert (REPO / "docs/changes" / record).exists(), record
        shared = SHARED_EDITS.get(sha, set())
        touched = [p for p in _files(sha) if SLICE_2B_FILES.match(p)]
        assert set(touched) <= shared, (sha, touched)
        tasks.add(task)
    assert tasks == {"PT4", "PT6", "3a S1", "3a S2"}
    assert set(SHARED_EDITS) <= set(not_slice_2b())
    assert set(COMMIT_ALLOWANCES) <= set(slice_commits())


def test_k2_merges_carry_no_change_of_their_own():
    """Fails if a merge commit in the range resolved a conflict with its own edit,
    which no commit-by-commit check would see.
    """
    merges = _range("--merges")
    for sha in merges:
        combined = _git("diff-tree", "--cc", "-p", sha).strip().splitlines()
        assert combined == [sha], f"merge {sha[:7]} carries its own change"


def test_k2_control_a_core_commit_is_caught():
    """Control: Slice 3a S1's core commit, judged as if it were Slice 2b, is caught
    at its core paths; a commit's allowance does not carry to another commit.
    """
    core = _resolve("f155d0b")
    found = violations([core])
    paths = next(iter(found.values()))
    assert "src/spaceodyssey/core/capacities.py" in paths
    assert "src/spaceodyssey/codec/kinds.py" not in paths
    pre_commit = _resolve("93a12af")
    assert violations([pre_commit]) == {}
    assert violations([_resolve("17976f4")]) != {}  # .gitignore is not allowed


# ---- K3 ----------------------------------------------------------------------------

test_k3_permutation_invariant = premission_tests.test_overlap_permutation_invariant
test_k3_observer_only_row = (
    premission_tests.test_overlap_observer_only_row_does_not_move_policy
)
test_k3_retained_priors_disjoint = premission_tests.test_retained_priors_disjoint

# ---- K4 ----------------------------------------------------------------------------

test_k4_hash_pin_names_its_commit = run_document_tests.test_the_hash_pin_names_its_commit_and_covers_every_corpus_document  # noqa: E501
test_k4_corpus_hashes_and_schema = (
    codec_tests.test_every_corpus_document_keeps_its_pinned_hash_and_schema_version
)
test_k4_control_a_changed_study = (
    run_document_tests.test_a_changed_study_misses_its_pinned_hash
)
test_k4_table_round_trip = (
    codec_tests.test_the_table_round_trips_through_file_references
)
test_k4_tampering_refused = (
    codec_tests.test_an_unsafe_or_damaged_file_reference_is_refused_at_its_pointer
)
test_k4_lineage_passes = (
    codec_tests.test_an_exported_table_imports_and_verifies_its_lineage
)
test_k4_lineage_fails = codec_tests.test_lineage_fails_on_a_reordered_or_changed_table

# ---- K5 ----------------------------------------------------------------------------

test_k5_lint_scans_the_core = lints.test_lint_scans_the_core
test_k5_core_has_no_domain_word = lints.test_core_is_clean
test_k5_core_imports_clean = lints.test_core_imports_clean

# ---- named interface checks ----------------------------------------------------------

# check -> (dependency task, "module:name" the dependency adds, the check itself)
NAMED_CHECKS = {
    "test_replay_reproduces_prior_run": (
        "T4c",
        "spaceodyssey.core.executors.replay:replay_run",
        refit_tests.test_replay_reproduces_prior_run,
    ),
    "test_a_refit_observer_keeps_its_prior_across_a_refresh": (
        "T3b",
        "spaceodyssey.core.conformance:check_derived_summary",
        refit_tests.test_a_refit_observer_keeps_its_prior_across_a_refresh,
    ),
}

test_named_t4c_replay = refit_tests.test_replay_reproduces_prior_run
test_named_t3b_derived_summary = (
    refit_tests.test_a_refit_observer_keeps_its_prior_across_a_refresh
)


def _merged(probe):
    """Whether the module and name a dependency adds are importable."""
    module, _, name = probe.partition(":")
    try:
        return hasattr(importlib.import_module(module), name)
    except ImportError:
        return False


def test_no_named_interface_check_is_pending():
    """Fails, listing them, if a named interface check's dependency has not merged
    (the check would be pending), or if a named check is not run by this gate.
    """
    pending = [
        f"{name} (waits on {task})"
        for name, (task, probe, _) in NAMED_CHECKS.items()
        if not _merged(probe)
    ]
    assert not pending, "pending named interface checks: " + ", ".join(pending)
    here = {obj for name, obj in globals().items() if name.startswith("test_named_")}
    assert {check for _, _, check in NAMED_CHECKS.values()} <= here


# ---- K6 and K7 ---------------------------------------------------------------------


def _quick_summary_step():
    text = CI.read_text()
    step = text[text.index('--title "Quick tier"') :].split("\n\n")[0]
    return text, step


def test_k6_k7_the_quick_tier_step_enforces_the_budgets():
    """Fails if the quick-tier CI step drops the 540 s ceiling, the baseline
    comparison with its 60 s limit, or the medium tier, if the quick tier stops
    selecting medium tests, if the budget plugin is not mandatory or not enforcing,
    or if the baseline file lost its wall time, source or status.
    """
    text, step = _quick_summary_step()
    for flag in (
        "--budget-seconds 540",
        "--max-increase-seconds 60",
        "--baseline .github/budget/quick_baseline.json",
        "--require-tier medium",
    ):
        assert flag in step, flag
    assert 'pytest tests/ -m "not slow"' in text
    assert "hwoutils.testbudget.plugin" in conftest.pytest_plugins
    budget = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["testbudget"]
    assert budget["mode"] == "enforce"
    baseline = json.loads(BASELINE.read_text())
    assert baseline["wall_s"] > 0 and baseline["source"] and baseline["status"]


JUNIT = """<?xml version="1.0"?>
<testsuites><testsuite name="pytest" time="{wall}">
<testcase classname="t" name="a" time="1.0">
<properties><property name="tier" value="unit"/></properties></testcase>
<testcase classname="t" name="b" time="2.0">
<properties><property name="tier" value="{tier}"/></properties></testcase>
</testsuite></testsuites>
"""


@pytest.mark.parametrize(
    ("wall", "tier", "ok"),
    [
        (159.0, "medium", True),  # 59 s above the baseline, under the ceiling
        (161.0, "medium", False),  # 61 s above the baseline
        (150.0, "unit", False),  # no medium test: the selection dropped the tier
        (541.0, "medium", False),  # over the 540 s ceiling (baseline raised below)
    ],
)
def test_k6_k7_the_summary_script_fails_over_its_limits(tmp_path, wall, tier, ok):
    """Fails if ci_summary.py passes a run 61 s above its baseline, over 540 s, or
    with no medium test, or fails a run within every limit.
    """
    junit = tmp_path / "junit.xml"
    junit.write_text(JUNIT.format(wall=wall, tier=tier))
    baseline = tmp_path / "baseline.json"
    base = 500.0 if wall > 540 else 100.0
    baseline.write_text(json.dumps({"wall_s": base, "source": "test", "status": "x"}))
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_STEP_SUMMARY"}
    result = subprocess.run(
        [
            sys.executable,
            str(CI_SUMMARY),
            str(junit),
            "--budget-seconds",
            "540",
            "--require-tier",
            "medium",
            "--baseline",
            str(baseline),
            "--max-increase-seconds",
            "60",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert (result.returncode == 0) == ok, result.stdout + result.stderr
