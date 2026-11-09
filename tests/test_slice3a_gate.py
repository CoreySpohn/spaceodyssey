"""The Slice 3a kill-rule gate: K1 to K8 and the acceptance map A1 to A10.

Run ``pytest tests/test_slice3a_gate.py`` with the full commit history (K4 reads it).
Every rule has at least one check whose name starts with its number, and a check fails
when its evidence is empty. Evidence that already lives in a test is invoked here, by
alias or through a thin wrapper that supplies the shared run; it is never rebuilt. The
evidence a rule needs and no test held is built below, each next to its mutant.

- K1, earlier slices stay true: the golden histories, fingerprints, fault fixture and
  corpus document hashes match their pins, no later commit touched a pin file, and four
  mutants (an asset-key change, an always-on empty leaf, a changed ``N_TIES``, lane
  closure on a fault) each move a pinned value.
- K2, conservation: the independent reconciliation holds on every scenario, refunds
  across expiry and re-award included, and the forged and mutant books fail it.
- K3, step cost: measured on the CI runner by ``pilot.yml``; the gate checks the
  measurement is wired and prints a manual-read line.
- K4, a survey needs a core patch: every ``core/`` commit after S0 is attributed to a
  task whose change record names each path it touched and cites a core test, and no
  survey commit touches the core.
- K5, the Q22 frame: the domain-word and import lints pass over the core.
- K6, truth and release discipline: the survey 8 and 10 controls fail as designed,
  refund views are invariant under a private event, the truth-reading committee and the
  visible refund identity are caught.
- K7, quick-tier time: enforced by the CI step; the gate checks the step and prints a
  manual-read line carrying the last measured figure.
- K8, action replay: the survey 10 replay reproduces awards, ``Allocate`` effects and
  admission outcomes, and a dropped award fails it.

The acceptance items A1 to A10 are mapped to their test files below. Those files run in
the quick tier; the gate checks that each exists and still defines tests.

Which commits K4 judges. The range runs from the last S0 commit to HEAD. A core
commit is listed under the task whose record describes it: a 3a feature task (S1 to
S6), Slice 2b (its own records, judged by the 2b gate), or the compile-speed work
(bit-identical, so the K1 pins are its evidence). A core commit that no list names, a
stale entry, a path the record does not name, or a record with no core test fails.
"""

import ast
import json
import re
from pathlib import Path

import equinox as eqx
import jax
import pytest

from spaceodyssey import core
from spaceodyssey.core import Audience
from spaceodyssey.core.conformance import corpus as conformance_corpus
from spaceodyssey.core.conformance import run_case
from spaceodyssey.core.executors.base import canonical_history
from spaceodyssey.core.handlers import HANDLERS
from tests import conftest
from tests import test_slice2b_gate as gate_2b
from tests.codec import test_premission_codec as codec_tests
from tests.codec import test_run_document_runs as run_document_tests
from tests.core import mutants
from tests.core import test_awards as awards
from tests.core import test_conservation as conservation
from tests.core import test_golden_histories as golden
from tests.core import test_lints as lints
from tests.core import test_refund as refund
from tests.core.conformance_helpers import N_WORLDS
from tests.core.conformance_helpers import runs as corpus_runs
from tests.core.corpus import CORPUS_BY_NAME
from tests.core.refund_helpers import refund_study
from tests.core.test_awards import conserved, windowed  # noqa: F401 (fixtures)
from tests.studies import test_allocator as allocator
from tests.studies import test_survey_8 as survey8
from tests.studies import test_survey_10 as survey10
from tests.studies.test_survey_8 import stopped  # noqa: F401 (fixture)

pytestmark = pytest.mark.clear_caches_between_tests

REPO = Path(__file__).resolve().parents[1]
CHANGES = REPO / "docs/changes"
CI = gate_2b.CI
CORE = "src/spaceodyssey/core/"
_git = gate_2b._git
_resolve = gate_2b._resolve


def _pin():
    return json.loads(golden.PIN.read_text())


# ---- K1 ----------------------------------------------------------------------------

test_k1_pin_names_its_commit = (
    golden.test_the_pin_names_its_commit_and_covers_every_entry
)
test_k1_manifest_equals_pin = golden.test_the_manifest_equals_its_pin
test_k1_history_digest = golden.test_history_matches_its_golden_digest
test_k1_fingerprints = golden.test_fingerprints_match_their_pin
test_k1_fingerprints_carry_evidence = golden.test_the_pinned_fingerprints_carry_evidence
test_k1_fault_fixture = golden.test_the_fault_fixture_matches_its_pin
test_k1_hash_pin_names_its_commit = run_document_tests.test_the_hash_pin_names_its_commit_and_covers_every_corpus_document  # noqa: E501
test_k1_corpus_hashes_and_schema = (
    codec_tests.test_every_corpus_document_keeps_its_pinned_hash_and_schema_version
)
test_k1_control_a_changed_study = (
    run_document_tests.test_a_changed_study_misses_its_pinned_hash
)
test_k1_control_a_one_bit_change = golden.test_a_one_bit_change_moves_the_digest
test_k1_mutant_lane_closure_on_a_fault = (
    golden.test_a_lane_closing_fault_breaks_the_fault_pin
)


def test_k1_no_commit_after_the_pins_touched_a_pin_file():
    """Fails if a pin names another source commit, or if any commit from the pins'
    capture to HEAD, other than the two capture commits, changed a pin file: the pins
    hold through the whole of Slice 3a, documented or not.
    """
    assert json.loads(gate_2b.GOLDEN_PIN.read_text())["source_commit"] == (
        gate_2b.PIN_SOURCE
    )
    assert json.loads(gate_2b.HASH_PIN.read_text())["source_commit"] == (
        gate_2b.PIN_SOURCE
    )
    touched = set(
        _git(
            "log",
            "--format=%H",
            f"{gate_2b.PIN_SOURCE}..HEAD",
            "--",
            str(gate_2b.GOLDEN_PIN.relative_to(REPO)),
            str(gate_2b.HASH_PIN.relative_to(REPO)),
        ).split()
    )
    assert touched == gate_2b.PIN_CAPTURES, sorted(touched - gate_2b.PIN_CAPTURES)
    assert len(_pin()["histories"]) == golden.N_PINNED > 0


def _corpus_history(name):
    """The runs of one corpus case on the host executor, without the private probe."""
    out = run_case(CORPUS_BY_NAME[name], "host", n_worlds=N_WORLDS, probe=False)
    assert out
    return {golden.entry_key(name, "host", r.world): r for r in out}


def _digests(runs):
    return {k: golden.history_digests(r.history) for k, r in runs.items()}


def test_k1_mutant_an_asset_key_change_moves_a_pinned_history(monkeypatch):
    """Fails if keying every draw by the layout's asset leaves a pinned history alone.

    `phase_switch` measures on channel 12, the toy asset 1. The unchanged run matches
    its pin (integers), so the comparison is not vacuous; the mutant's draws, and so
    the float payloads, move (float digests are platform dependent, so the mutant is
    compared with the unchanged run on the same machine).
    """
    pinned = _pin()["histories"]
    runs = _corpus_history("phase_switch")
    control = _digests(runs)
    for key, got in control.items():
        assert got["ints"] == pinned[key]["ints"], key
        assert (runs[key].history.ints["channel"] == 12).any(), key  # asset 1 measured
    monkeypatch.setattr(
        conformance_corpus, "HostExecutor", mutants.AssetKeyHostExecutor
    )
    moved = _digests(_corpus_history("phase_switch"))
    assert moved.keys() == control.keys()
    for key in control:
        assert moved[key]["floats"] != control[key]["floats"], key


def test_k1_mutant_an_always_on_empty_leaf_moves_the_pins():
    """Fails if a conditional leaf of size zero, listed always, leaves the pinned
    snapshot fingerprint or the pinned history digest as it was.
    """
    run = corpus_runs("baseline", "host")[0]
    key = golden.entry_key("baseline", "host", 0)
    layout = run.study.observer.layout(run.study.caps)
    view = core.snapshot_view(run.carry, Audience.POLICY, run.study.caps)
    pinned_fp = _pin()["fingerprints"][key]["policy"]
    pinned_ints = _pin()["histories"][key]["ints"]
    history = canonical_history(run.carry, run.trace, layout)
    assert int(core.snapshot_fingerprint(view, layout)) == pinned_fp
    assert golden.history_digests(history)["ints"] == pinned_ints
    assert int(mutants.fingerprint_listing_empty_leaves(view, layout)) != pinned_fp
    with pytest.MonkeyPatch.context() as patch:
        mutants.open_every_conditional_gate(patch, run.carry, run.trace)
        opened = canonical_history(run.carry, run.trace, layout)
    assert golden.history_digests(opened)["ints"] != pinned_ints


def test_k1_mutant_a_changed_n_ties_moves_a_pinned_history():
    """Fails if one more tie rank, which repacks every event order key, leaves the
    pinned history of `baseline` as it was.
    """
    pinned = _pin()["histories"]
    with pytest.MonkeyPatch.context() as patch:
        mutants.change_the_tie_ranks(patch)
        try:
            moved = _digests(_corpus_history("baseline"))
        finally:
            patch.undo()
            eqx.clear_caches()
            jax.clear_caches()
    assert moved
    for key, got in moved.items():
        assert got["ints"] != pinned[key]["ints"], key


# ---- K2 ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def conservation_runs():
    return conservation._Runs()


@pytest.mark.parametrize("name", conservation.CASES)
def test_k2_every_account_agrees(conservation_runs, name):
    """Fails if the balance, the counters and the records disagree on a case."""
    conservation.test_every_account_agrees_after_pricing_cancels_and_refunds(
        conservation_runs, name
    )


def test_k2_a_refund_nets_the_surcharge_out(conservation_runs):
    """Fails if a refund credits the nonrefundable part of the charge."""
    conservation.test_a_refund_nets_the_surcharge_out_of_the_credit(conservation_runs)


def test_k2_a_forged_balance_is_caught(conservation_runs):
    """Fails if a forged balance, alone or with its counter, passes."""
    conservation.test_a_forged_balance_is_caught(conservation_runs)
    conservation.test_a_forged_consistent_balance_and_counter_is_caught_by_the_records(
        conservation_runs
    )


test_k2_every_award_scenario_conserves = (
    awards.test_every_award_scenario_conserves_the_balance
)
test_k2_each_remainder_rule_balance = (
    awards.test_each_remainder_rule_settles_the_unspent_balance
)
test_k2_forged_award_counters = awards.test_forged_award_counters_are_caught
test_k2_a_refund_across_an_expiry_and_a_re_award = (
    awards.test_a_refund_across_an_expiry_and_a_re_award_conserves
)
test_k2_a_cancellation_under_an_award = (
    awards.test_a_cancellation_under_an_award_conserves
)
test_k2_mutant_the_expiry_debit_and_counter_omitted_together = (
    awards.test_a_paired_forged_expiry_is_caught
)
test_k2_mutant_a_refund_of_the_whole_charge = (
    conservation.test_mutant_a_refund_of_the_whole_charge_fails_conservation
)
test_k2_mutant_a_second_receipt_credit = (
    conservation.test_mutant_a_second_receipt_credit_fails_conservation
)
test_k2_mutant_an_old_awards_refund_credited_to_the_newer_award = (
    awards.test_mutant_crediting_an_old_awards_refund_to_the_newer_award_fails
)


# ---- K3 and K7 ---------------------------------------------------------------------

# The last warm quick-tier run of the slice's head commit on the CI runner, read by hand
# from the job summary (CI time cannot be measured from inside pytest).
K7_RUN = {"commit": "ad113d3", "run_id": 42576344367, "wall_s": 391.2, "tests": 2681}
K7_CEILING_S = 540
STEP_COST = REPO / "tools/step_cost.py"
PILOT = REPO / ".github/workflows/pilot.yml"
S0_RECORD = CHANGES / "slice-3a-s0.md"


def test_k3_the_step_cost_measurement_is_wired_for_the_final_table():
    """Fails if the table is not the thirteen-handler table K3 is judged at, if the
    probe, its calibrated mutant or the workflow that runs it on the CI runner is
    gone, or if the S0 record lost the mutant's measured excess.
    """
    assert len(HANDLERS) == 13
    kinds = {k.name for k in core.EventKind}
    assert {"WINDOW_END", "COMMIT_START"} <= kinds
    probe = STEP_COST.read_text()
    assert "--mutant" in probe and "block_until_ready" in probe
    workflow = PILOT.read_text()
    assert "tools/step_cost.py --mutant" in workflow
    record = S0_RECORD.read_text()
    assert "1.48" in record and "1.51" in record  # the mutant beats the 1.10 limit


def test_k7_the_quick_tier_step_enforces_the_ceiling_with_medium_tests_counted():
    """Fails if the quick-tier CI step runs the summary advisory only, drops the
    540 s ceiling or the medium tier, or if the budget plugin is not enforcing.
    """
    step = gate_2b._quick_summary_step()[1]
    assert "--advisory-baseline" not in step
    assert f"--budget-seconds {K7_CEILING_S}" in step
    assert "--require-tier medium" in step
    assert "hwoutils.testbudget.plugin" in conftest.pytest_plugins
    assert K7_RUN["wall_s"] < K7_CEILING_S and K7_RUN["tests"] > 0


test_k7_the_summary_script_fails_over_its_limits = (
    gate_2b.test_k6_k7_the_summary_script_fails_over_its_limits
)
test_k7_the_step_carries_every_budget_flag = (
    gate_2b.test_k6_k7_the_quick_tier_step_enforces_the_budgets
)


def test_k3_k7_manual_reads(capsys):
    """Prints the two numbers the gate cannot measure, and fails if the S10 record
    does not carry the K7 figure it prints.
    """
    run = K7_RUN
    with capsys.disabled():
        print(
            "\nK3 manual read: dispatch .github/workflows/pilot.yml on the final "
            "commit and compare its per-step minimum with the eleven-handler "
            "baseline; the limit is 1.10."
        )
        print(
            f"K7 manual read: warm CI run {run['run_id']} of {run['commit']}, quick "
            f"tier {run['wall_s']} s over {run['tests']} tests, ceiling "
            f"{K7_CEILING_S} s."
        )
    record = (CHANGES / "slice-3a-s10.md").read_text()
    assert str(run["run_id"]) in record and str(run["wall_s"]) in record


# ---- K4 ----------------------------------------------------------------------------

S0_END = "97753d57564770664fca3b73c26840d0681dacf2"

# task -> (records that describe it, the core commits it owns, by short sha)
TASKS = {
    "S1": (
        ("slice-3a-s1",),
        ("f155d0b", "624f25e"),
    ),
    "S2": (
        ("slice-3a-s2",),
        ("c61f43c", "538d2ec"),
    ),
    "S3a": (
        ("slice-3a-s3a",),
        ("33b5eda", "b3c7d91", "6c2e935", "7da9d2b"),
    ),
    "S4": (
        ("slice-3a-s4",),
        ("a33673e", "66f1c3e", "49cd816", "24fe034", "aa04806", "c4f8bf7"),
    ),
    "S5": (
        ("slice-3a-s5",),
        ("61d2e5e", "0c7a4e4", "ea3851b", "b400d37", "d4b2c86"),
    ),
    "S5b": (
        ("slice-3a-s5b",),
        ("7435b80", "55dba07"),
    ),
    "S6": (
        ("slice-3a-s6",),
        (
            "7ec4ad4",
            "098d4ed",
            "3098403",
            "1aad3e2",
            "7a19fdd",
            "efe1f12",
            "7e1b324",
            "79c7231",
        ),
    ),
}
# Work of other tasks that touched the core in the range, with the records that say so.
OTHER_TASKS = {
    # Slice 2b R1 to R3, judged by the 2b gate; the paths are in its records
    "Slice 2b": (
        tuple(sorted(p.stem for p in CHANGES.glob("slice-2b-*.md"))),
        (
            "0c3dbf4",
            "99f36b9",
            "10d31aa",
            "7fe3e24",
            "8eb98eb",
            "5d0f50c",
            "119079f",
            "ec013c3",
            "d115177",
            "1bb9614",
            "76dbe96",
        ),
    ),
    # compile sharing; every history, fingerprint and pin stayed bit-identical (K1)
    "compile speed": (
        ("compile-speed-b", "compile-speed-c"),
        (
            "5b2df1f",
            "e2738bc",
            "dd3b4c0",
            "7b3ae8d",
            "68e23ee",
            "d6ffe93",
            "36cca30",
            "3df12b1",
            "53616a7",
            "3561c5d",
            "958b0b1",
            "cd39e78",
            "85c0b4d",
        ),
    ),
}
SURVEY_FILES = (
    "docs/changes/slice-3a-survey8.md",
    "docs/changes/slice-3a-survey9.md",
    "docs/changes/slice-3a-survey10.md",
    "docs/changes/slice-3a-survey11.md",
    "tests/studies/test_survey_8.py",
    "tests/studies/test_survey_9.py",
    "tests/studies/test_survey_10.py",
    "tests/studies/test_survey_11.py",
)


def _owners():
    """Full sha of each listed core commit -> its task name."""
    owners = {}
    for task, (_, shas) in {**TASKS, **OTHER_TASKS}.items():
        for sha in shas:
            full = _resolve(sha)
            assert full not in owners, f"{sha} is listed under two tasks"
            owners[full] = task
    return owners


def _core_commits():
    """Every non-merge commit from the last S0 commit to HEAD that touches the core."""
    return _git(
        "rev-list", "--no-merges", f"{S0_END}..HEAD", "--", CORE.rstrip("/")
    ).split()


def _core_paths(sha):
    return [
        p
        for p in _git("show", "--no-renames", "--name-only", "--format=", sha).split()
        if p.startswith(CORE)
    ]


def _record_text(names):
    return "\n".join((CHANGES / f"{n}.md").read_text() for n in names)


def _mentions(text, path):
    """Whether a record names a core path: by its module name or, for a subpackage
    file, by `package/file` or the bare `package/` directory.
    """
    rel = path.removeprefix(CORE).removesuffix(".py")
    names = {rel, rel.rsplit("/", 1)[-1]}
    if "/" in rel:
        names.add(rel.split("/", 1)[0] + "/")
    return any(
        re.search(r"(?<![\w])" + re.escape(n) + r"(?![\w])", text) for n in names
    )


def _core_test_cited(text):
    """Whether a record cites an existing file under `tests/core/`."""
    cited = re.findall(r"tests/core/[\w/]+\.py", text)
    return any((REPO / path).exists() for path in cited)


def unattributed(owners, commits, text_of):
    """Problems with the attribution: commits no task lists, stale entries, and tasks
    whose records cite no core test.
    """
    found = []
    for sha in commits:
        if sha not in owners:
            found.append(f"{sha[:7]} {_git('log', '-1', '--format=%s', sha).strip()}")
    found += [
        f"stale {sha[:7]} ({task})"
        for sha, task in owners.items()
        if sha not in commits
    ]
    for task in {owners[s] for s in commits if s in owners}:
        if not _core_test_cited(text_of(task)):
            found.append(f"the {task} record cites no tests/core file")
    return found


def unnamed_paths(owners, commits, text_of):
    """Core paths a commit touched that its task's record does not name (reported,
    not failed: the records describe behavior, and name modules only where it helps).
    """
    return [
        f"{sha[:7]} {path} ({owners[sha]})"
        for sha in commits
        if sha in owners
        for path in _core_paths(sha)
        if not _mentions(text_of(owners[sha]), path)
    ]


def _text_of(task):
    records = {**TASKS, **OTHER_TASKS}[task][0]
    return _record_text(records)


def _touching_the_core(shas):
    return [s for s in shas if _core_paths(s)]


def _survey_commits():
    """The commits that add a survey test or record, from S0 to HEAD."""
    return _git(
        "rev-list", "--no-merges", f"{S0_END}..HEAD", "--", *SURVEY_FILES
    ).split()


def test_k4_every_core_change_is_attributed_to_a_task_record_and_a_core_test(capsys):
    """Fails if a core commit after S0 is listed under no task, a listed commit is not
    in the range, or a task's record cites no existing core test. Core paths a record
    does not name are printed, not failed.
    """
    commits = _core_commits()
    assert len(commits) >= 40  # evidence: the slice's core work is in the range
    owners = _owners()
    assert unattributed(owners, commits, _text_of) == []
    assert {"S1", "S2", "S3a", "S4", "S5", "S5b", "S6"} <= set(owners.values())
    unnamed = unnamed_paths(owners, commits, _text_of)
    with capsys.disabled():
        print(f"\nK4: {len(commits)} core commits, {len(unnamed)} unnamed paths")
        for line in unnamed:
            print(f"K4 unnamed: {line}")


def test_k4_no_survey_commit_touches_the_core():
    """Fails if a commit that adds a survey test or its record touches `core/`."""
    shas = _survey_commits()
    assert len(shas) >= 4  # evidence: surveys 8, 9, 10 and 11 each landed
    assert _touching_the_core(shas) == []


def test_k4_control_a_core_edit_in_a_survey_commit_is_caught():
    """Control: a commit that edited the core is flagged by the survey check."""
    feature = _resolve("f155d0b")
    assert _touching_the_core([feature]) == [feature]


def test_k4_control_an_unlisted_commit_and_an_unnamed_path_are_caught():
    """Control: dropping a commit from the owner table is reported, and a record that
    does not name a path the commit touched is reported.
    """
    commits = _core_commits()
    owners = _owners()
    dropped = _resolve("f155d0b")
    fewer = {s: t for s, t in owners.items() if s != dropped}
    assert any(dropped[:7] in p for p in unattributed(fewer, commits, _text_of))
    blank = lambda task: "tests/core/test_nothing_here.py"  # noqa: E731
    assert unnamed_paths(owners, [dropped], blank)
    assert any("cites no" in p for p in unattributed(owners, [dropped], blank))


# ---- K5 ----------------------------------------------------------------------------

test_k5_lint_scans_the_core = lints.test_lint_scans_the_core
test_k5_core_has_no_domain_word = lints.test_core_is_clean
test_k5_core_imports_clean = lints.test_core_imports_clean
test_k5_control_each_denied_token = lints.test_lint_catches_each_denied_token
test_k5_control_each_import_fixture = lints.test_core_imports_catches_fixtures


# ---- K6 ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def refund_runs():
    """The plain and the private-row-shifted refund runs the privacy checks read."""
    out = {}
    for name in ("applied", "shifted"):
        study = refund_study(
            refund.CANCELLED, **refund.SURCHARGE, **refund.SCENARIOS[name]
        )
        out[name] = (study, refund._run(study))
    return out


@pytest.fixture(scope="module", params=("host", "compiled"))
def survey10_runs(request):
    return survey10._Runs(request.param)


def test_k6_a_refund_view_ignores_a_private_event(refund_runs):
    """Fails if an observer-only receipt moves the policy fingerprint of a refund."""
    refund.test_the_policy_fingerprint_ignores_a_private_event_before_the_refund(
        refund_runs
    )


def test_k6_mutant_a_visible_internal_refund_identity(refund_runs, monkeypatch):
    """Fails if showing the internal `refund_of` leaves the view invariant."""
    refund.test_mutant_a_visible_internal_refund_id_moves_the_fingerprint(
        refund_runs, monkeypatch
    )


def test_k6_survey_10_awards_ignore_unreleased_and_observer_only(survey10_runs):
    """Fails if an award reads a proposal the policy never saw (counts nonzero)."""
    survey10.test_awards_ignore_unreleased_and_observer_only_proposals(survey10_runs)


def test_k6_survey_10_a_release_moves_the_award(survey10_runs):
    """Fails if releasing a proposal does not move its award the expected way."""
    survey10.test_releasing_a_proposal_moves_the_award_the_expected_way(survey10_runs)


def test_k6_mutant_a_truth_reading_committee(survey10_runs):
    """Fails if a committee that reads the carry directly passes the perturbation."""
    survey10.test_a_committee_reading_the_carry_fails_the_perturbation(survey10_runs)


test_k6_survey_8_a_policy_stops_the_dry_asset = (
    survey8.test_a_policy_stops_the_dry_asset_and_the_other_asset_is_unaffected
)
test_k6_control_the_removed_policy_stop = survey8.test_control_without_the_policy_stop_every_refusal_is_no_funds_and_row_b_fails  # noqa: E501


# ---- K8 ----------------------------------------------------------------------------


def test_k8_the_survey_10_replay_reproduces_the_run_and_a_dropped_award_fails(
    survey10_runs,
):
    """Fails if the replay misses an award row, an `Allocate`, an admission outcome, or
    accepts a log with one award dropped.
    """
    survey10.test_replay_reproduces_the_run_and_fails_without_a_recorded_award(
        survey10_runs
    )


test_k8_mutant_a_dropped_award = (
    allocator.test_dropping_a_recorded_award_fails_the_check
)


# ---- A1 to A10 ---------------------------------------------------------------------

ACCEPTANCE = {
    "A1": ("tests/core/test_charging.py",),
    "A2": ("tests/core/test_refund.py", "tests/core/test_awards.py"),
    "A3": ("tests/core/test_awards.py",),
    "A4": ("tests/core/test_frozen_horizon.py",),
    "A5": ("tests/core/test_programs.py",),
    "A6": ("tests/core/test_conservation.py",),
    "A7": ("tests/core/test_snapshot_balances.py",),
    "A8": ("tests/core/test_preflight.py", "tests/codec/test_preflight_json.py"),
    "A9": (
        "tests/studies/test_survey_8.py",
        "tests/studies/test_survey_9.py",
        "tests/studies/test_survey_10.py",
        "tests/studies/test_survey_11.py",
    ),
}
STOP_GOLDENS = tuple(
    f"stop_{scope}_{disposition}"
    for scope in ("asset", "campaign", "lane", "program")
    for disposition in ("abort", "close", "drain")
)


def _test_functions(path):
    tree = ast.parse((REPO / path).read_text())
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]


@pytest.mark.parametrize("item", sorted(ACCEPTANCE))
def test_a1_to_a9_each_acceptance_item_has_its_tests(item):
    """Fails if an acceptance file is gone or defines no test, which would leave the
    item's check empty (the files run in the quick tier).
    """
    for path in ACCEPTANCE[item]:
        assert (REPO / path).exists(), path
        assert _test_functions(path), path


def test_a5_the_twelve_stop_goldens_are_pinned():
    """Fails if a `stop_*` corpus case lost its pin on either executor and world."""
    pinned = _pin()["histories"]
    for name in STOP_GOLDENS:
        for executor in ("host", "compiled"):
            for world in range(N_WORLDS):
                assert golden.entry_key(name, executor, world) in pinned, name
    assert len(STOP_GOLDENS) == 12


def test_a10_the_pins_are_the_k1_checks():
    """Fails if a K1 check this gate invokes went missing: A10 is K1."""
    here = {n for n in globals() if n.startswith("test_k1_")}
    assert len(here) >= 12


# ---- the gate itself ---------------------------------------------------------------

# Each spec mutant and the check in this file that must fail on it.
MUTANT_CHECKS = {
    "K1 an asset-key change for asset 1": test_k1_mutant_an_asset_key_change_moves_a_pinned_history,  # noqa: E501
    "K1 an always-on empty leaf": test_k1_mutant_an_always_on_empty_leaf_moves_the_pins,
    "K1 a changed N_TIES": test_k1_mutant_a_changed_n_ties_moves_a_pinned_history,
    "K1 lane closure on FAULT by default": test_k1_mutant_lane_closure_on_a_fault,
    "K2 omit the expiry debit and its counter together": test_k2_mutant_the_expiry_debit_and_counter_omitted_together,  # noqa: E501
    "K2 refund a nonrefundable surcharge": test_k2_mutant_a_refund_of_the_whole_charge,
    "K2 credit an old award's refund to a newer award": test_k2_mutant_an_old_awards_refund_credited_to_the_newer_award,  # noqa: E501
    "K4 a core edit in a survey commit": test_k4_control_a_core_edit_in_a_survey_commit_is_caught,  # noqa: E501
    "K6 a truth-reading committee": test_k6_mutant_a_truth_reading_committee,
    "K6 a visible internal refund_of": test_k6_mutant_a_visible_internal_refund_identity,  # noqa: E501
    "K8 drop one award from the log": test_k8_mutant_a_dropped_award,
}


def test_the_gate_has_checks_for_every_rule_and_every_mutant():
    """Fails if a rule K1 to K8 has no check in this file, or a spec mutant is not
    tied to a check that exists here. K3 and K7 are the wired measurements.
    """
    for rule in range(1, 9):
        checks = [n for n in globals() if n.startswith(f"test_k{rule}_")]
        assert checks, f"K{rule} has no check"
    here = {id(v) for n, v in globals().items() if n.startswith("test_")}
    for label, check in MUTANT_CHECKS.items():
        assert id(check) in here, label
    assert {label.split()[0] for label in MUTANT_CHECKS} == {
        "K1",
        "K2",
        "K4",
        "K6",
        "K8",
    }
