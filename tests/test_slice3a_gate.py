"""The Slice 3a kill-rule gate: K1 to K8 and the acceptance items A1 to A10.

Run ``pytest tests/test_slice3a_gate.py`` with the full commit history (K4 and the pin
check read it). The gate does not alias other tests. It lists, in `INVENTORY`, the exact
canonical test ids that carry each rule's and each acceptance item's evidence, runs them
once in a child pytest process (same environment, so the compile-cache switch and the
budget plugin apply), and requires every listed id to be collected and to pass: a
missing id, a skip, an xfail or a changed case count fails the gate and names the rule.
What no test held is built here, next to its mutant, and is listed in `OWN`; a recorder
checks that each of those ran and passed in this process.

- K1, earlier slices stay true: the golden histories, fingerprints, fault fixture and
  corpus document hashes match their pins, each pin file is byte-identical to its
  capture commit, no later commit touched one, and three mutants (an asset-key change,
  an always-on empty leaf, a changed ``N_TIES``) plus lane closure on a fault each move
  a pinned value.
- K2, conservation: the independent reconciliation holds on every scenario, refunds
  across expiry and re-award included, and the forged and mutant books fail it.
- K3, step cost: measured by `pilot.yml` in one job for the base tree and this tree
  (`K3_SAME_JOB` and the S10 record, from one pilot job); the gate checks
  the recorded ratio against 1.10, that the recorded mutant ratios exceed it, and that
  the probe's extra pass still runs.
- K4, a survey needs no core patch: every `core/` commit after S0 is listed under a
  task, the task record names each exact path the commit touched and cites a core test,
  and no survey commit touches the core.
- K5, the domain-word frame: the domain-word and import lints pass over the core.
- K6, truth and release discipline, and K8, action replay: survey 8 and 10 controls,
  refund views, the replay and the allocator check.
- K7, quick-tier time: enforced by the CI step; the gate checks the step and prints a
  manual-read line carrying the last measured figure.

Which commits K4 judges. The range runs from the last S0 commit to `K4_END`. A core
commit is listed under the task whose record describes it: a 3a feature task (S1 to
S6), or two other tasks that changed the core in the range, the earlier premission work
and the compile-speed work, which are attributed by their own task records with K1 as
their evidence. A core commit that no list names, a stale entry, a path the record does
not name exactly, or a record with no core test fails.
"""

import ast
import collections
import importlib.util
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
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
from tests import conftest, pin_helpers
from tests.core import mutants
from tests.core import test_golden_histories as golden
from tests.core.conformance_helpers import N_WORLDS
from tests.core.conformance_helpers import runs as corpus_runs
from tests.core.corpus import CORPUS_BY_NAME

# The child run is about 370 s on a laptop and about three times that on a runner, and
# pytest-timeout charges the child fixture's setup to the first test that uses it, so
# the gate tests carry a limit above the nightly job's default of 900 s.
GATE_TIMEOUT_S = 2400
CHILD_TIMEOUT_S = 900
# CI installs pytest-timeout (the `test` extra); a checkout without it runs untimed.
HAS_TIMEOUT = importlib.util.find_spec("pytest_timeout") is not None
pytestmark = [pytest.mark.clear_caches_between_tests]
if HAS_TIMEOUT:
    pytestmark.append(pytest.mark.timeout(GATE_TIMEOUT_S))

REPO = Path(__file__).resolve().parents[1]
CHANGES = REPO / "changes"
CORE = "src/spaceodyssey/core/"
_git = pin_helpers.git
_resolve = pin_helpers.resolve

GOLD = "tests/core/test_golden_histories.py::"
CODEC = "tests/codec/test_premission_codec.py::"
RUNDOC = "tests/codec/test_run_document_runs.py::"
CONS = "tests/core/test_conservation.py::"
AWARDS = "tests/core/test_awards.py::"
REFUND = "tests/core/test_refund.py::"
CHARGING = "tests/core/test_charging.py::"
LINTS = "tests/core/test_lints.py::"
S8 = "tests/studies/test_survey_8.py::"
S10 = "tests/studies/test_survey_10.py::"
ALLOC = "tests/studies/test_allocator.py::"
CISUM = "tests/test_ci_summary.py::"

# rule -> {canonical test id: number of cases that must pass}. A function id runs every
# case of it; a file id runs every test of the file.
_K1 = {
    GOLD + "test_manifest_covers_every_case_executor_and_world": 1,
    GOLD + "test_the_pin_names_its_commit_and_covers_every_entry": 1,
    GOLD + "test_the_manifest_equals_its_pin": 1,
    GOLD + "test_history_matches_its_golden_digest": 58,
    GOLD + "test_fingerprints_match_their_pin": 58,
    GOLD + "test_the_pinned_fingerprints_carry_evidence": 1,
    GOLD + "test_the_fault_fixture_matches_its_pin": 2,
    GOLD + "test_a_one_bit_change_moves_the_digest": 1,
    GOLD + "test_a_lane_closing_fault_breaks_the_fault_pin": 1,
    RUNDOC + "test_the_hash_pin_names_its_commit_and_covers_every_corpus_document": 1,
    RUNDOC + "test_a_changed_study_misses_its_pinned_hash": 1,
    CODEC + "test_every_corpus_document_keeps_its_pinned_hash_and_schema_version": 1,
}
_K2 = {
    CONS + "test_every_account_agrees_after_pricing_cancels_and_refunds": 5,
    CONS + "test_a_refund_nets_the_surcharge_out_of_the_credit": 1,
    CONS + "test_a_forged_balance_is_caught": 1,
    CONS + "test_a_forged_consistent_balance_and_counter_is_caught_by_the_records": 1,
    CONS + "test_a_receipt_row_that_loses_its_negative_charge_is_caught": 1,
    CONS + "test_mutant_a_refund_of_the_whole_charge_fails_conservation": 2,
    CONS + "test_mutant_a_second_receipt_credit_fails_conservation": 2,
    AWARDS + "test_every_award_scenario_conserves_the_balance": 11,
    AWARDS + "test_each_remainder_rule_settles_the_unspent_balance": 4,
    AWARDS + "test_forged_award_counters_are_caught": 2,
    AWARDS + "test_a_paired_forged_expiry_is_caught": 1,
    AWARDS + "test_the_paired_forged_expiry_is_caught_for_a_reallocation": 1,
    AWARDS + "test_a_paired_forged_expiry_is_caught_when_the_run_holds_a_refund": 2,
    AWARDS + "test_a_refund_across_an_expiry_and_a_re_award_conserves": 1,
    AWARDS + "test_a_cancellation_under_an_award_conserves": 1,
    AWARDS + "test_crediting_an_old_awards_refund_to_the_newer_award_is_caught": 1,
    AWARDS + "test_mutant_crediting_an_old_awards_refund_to_the_newer_award_fails": 1,
}
_K5 = {
    LINTS + "test_lint_scans_the_core": 1,
    LINTS + "test_core_is_clean": 1,
    LINTS + "test_core_imports_clean": 1,
    LINTS + "test_lint_catches_each_denied_token": 27,
    LINTS + "test_core_imports_catches_fixtures": 2,
}
_K6 = {
    REFUND + "test_the_policy_fingerprint_ignores_a_private_event_before_the_refund": 1,
    REFUND + "test_mutant_a_visible_internal_refund_id_moves_the_fingerprint": 1,
    S8 + "test_a_policy_stops_the_dry_asset_and_the_other_asset_is_unaffected": 1,
    S8
    + "test_control_without_the_policy_stop_every_refusal_is_no_funds_and_row_b_fails": 1,  # noqa: E501
    S10 + "test_awards_ignore_unreleased_and_observer_only_proposals": 2,
    S10 + "test_releasing_a_proposal_moves_the_award_the_expected_way": 2,
    S10 + "test_a_committee_reading_the_carry_fails_the_perturbation": 2,
    # the snapshot views stay the same when a private event is inserted
    "tests/core/test_core.py::test_observer_only_release_leaves_policy_view_unchanged": 1,  # noqa: E501
    "tests/core/test_audience_firewall.py::test_policy_fold_hides_private_event_identity": 1,  # noqa: E501
}
_K7 = {CISUM + "test_the_summary_script_fails_over_its_limits": 4}
_K8 = {
    S10 + "test_replay_reproduces_the_run_and_fails_without_a_recorded_award": 2,
    ALLOC + "test_dropping_a_recorded_award_fails_the_check": 1,
}
_ACCEPT = {
    "A1": {
        CHARGING + "test_a_halfway_cancel_retains_what_the_pricing_mode_spends": 4,
        CHARGING + "test_without_refunds_the_cost_is_debited_and_no_column_exists": 1,
        CHARGING + "test_charge_parts_reads_the_cost_when_refunds_are_off": 1,
        CHARGING
        + "test_charge_parts_adds_the_surcharge_only_to_a_surcharge_channel": 1,
        CHARGING + "test_the_default_charge_is_the_cost": 1,
        CHARGING + "test_mutant_a_cancel_that_credits_the_surcharge_is_caught": 1,
    },
    "A2": {
        # the refund file whole: deadline, cause, identity and release rules, and the
        # mutants for the deadline and the release
        "tests/core/test_refund.py": 24,
        # the window-refund matrix
        AWARDS + "test_a_late_refund_is_disposed_by_the_window_that_charged_it": 8,
        AWARDS + "test_a_refund_of_five_to_a_row_left_always_keeps_its_balance": 1,
        AWARDS + "test_a_refund_after_the_expiry_is_replayed_and_conserves": 1,
        AWARDS + "test_a_refund_before_the_expiry_is_replayed_and_conserves": 1,
        AWARDS + "test_a_refund_to_the_wrong_row_after_expiry_is_caught": 1,
        AWARDS + "test_a_refund_credited_to_another_programs_row_is_caught": 1,
        AWARDS
        + "test_a_refund_routed_to_another_acquisition_in_both_places_is_caught": 1,
        AWARDS + "test_receipts_that_apply_nothing_do_not_fail_conservation": 4,
        AWARDS + "test_a_refund_across_an_expiry_and_a_re_award_conserves": 1,
    },
    "A3": {
        AWARDS + "test_every_award_scenario_conserves_the_balance": 11,
        AWARDS + "test_each_remainder_rule_settles_the_unspent_balance": 4,
        S10 + "test_releasing_a_proposal_moves_the_award_the_expected_way": 2,
        # window admission
        AWARDS + "test_an_allocate_grants_the_amount_and_opens_the_window": 1,
        AWARDS + "test_an_observation_outside_the_window_is_refused": 1,
        AWARDS + "test_an_observation_at_the_tick_the_window_ends_is_refused": 1,
        AWARDS + "test_the_window_refuses_without_balances_in_the_snapshot": 1,
        AWARDS
        + "test_an_award_onto_an_open_window_is_forbidden_and_changes_nothing": 1,
        AWARDS
        + "test_a_new_award_at_the_expiry_tick_is_admitted_after_the_old_window_ends": 1,  # noqa: E501
        # boundary ordering
        AWARDS + "test_a_window_end_event_exists_only_once_an_award_is_issued": 1,
        AWARDS + "test_the_window_end_is_an_analyze_stage_neutral_tie_event": 1,
        AWARDS
        + "test_a_completion_at_the_expiry_tick_is_handled_before_the_window_end": 1,
        AWARDS + "test_the_expiry_tie_ranks_after_a_same_tick_refresh": 1,
        # the host and compiled executors make the same history of an award
        AWARDS + "test_the_host_and_compiled_executors_make_the_same_history": 1,
        # Allocate admission on both executors, in the allocator study
        ALLOC + "test_the_committee_answers_the_policy_on_the_host": 1,
        ALLOC + "test_the_committee_answers_on_both_executors": 1,
        ALLOC + "test_replay_fails_when_the_run_is_altered": 1,
        # a completion after the expiry still settles
        AWARDS + "test_an_acquisition_that_completes_after_expiry_still_settles": 1,
        # stale expiry
        AWARDS + "test_a_stale_expiry_after_a_re_award_changes_nothing": 1,
        AWARDS
        + "test_a_pending_acquisition_at_the_expiry_is_replayed_without_its_record": 1,
    },
    "A4": {"tests/core/test_frozen_horizon.py": 59},
    "A5": {"tests/core/test_programs.py": 22},
    "A6": {CONS + "test_every_account_agrees_after_pricing_cancels_and_refunds": 5},
    "A7": {"tests/core/test_snapshot_balances.py": 54},
    "A8": {
        "tests/core/test_preflight.py": 175,
        "tests/codec/test_preflight_json.py": 160,
        # optional-field round trips with populated refund, award and commitment columns
        "tests/codec/test_optional_fields.py": 71,
        "tests/codec/test_run_document_features.py": 6,
    },
    "A9": {
        S8 + "test_a_policy_stops_the_dry_asset_and_the_other_asset_is_unaffected": 1,
        "tests/studies/test_survey_9.py": 6,
        S8 + "test_both_executors_make_the_same_history_for_the_dry_asset_study": 1,
        S10 + "test_both_executors_make_the_same_awards": 1,
        "tests/studies/test_survey_11.py": 9,
    },
    "A10": _K1,
}
INVENTORY = {
    "K1": _K1,
    "K2": _K2,
    "K5": _K5,
    "K6": _K6,
    "K7": _K7,
    "K8": _K8,
    **_ACCEPT,
}

# rule -> {name of a check in this file: cases that must pass in this process}
OWN = {
    "K1": {
        "test_k1_every_pin_file_is_the_blob_of_its_capture_commit": 1,
        "test_k1_no_commit_after_the_pins_touched_a_pin_file": 1,
        "test_k1_mutant_an_asset_key_change_moves_a_pinned_history": 1,
        "test_k1_mutant_an_always_on_empty_leaf_moves_the_pins": 1,
        "test_k1_mutant_a_changed_n_ties_moves_a_pinned_history": 1,
    },
    "K3": {
        "test_k3_the_step_cost_measurement_is_wired_for_the_final_table": 1,
        "test_k3_the_same_job_ratio_is_recorded_or_k3_is_marked_pending": 1,
        "test_k3_the_ratio_check_separates_the_limit": 1,
        "test_k3_k7_manual_reads": 1,
    },
    "K4": {
        "test_k4_every_core_change_is_attributed_to_a_task_record_and_a_core_test": 1,
        "test_k4_no_survey_commit_touches_the_core": 1,
        "test_k4_control_a_survey_commit_that_touches_the_core_is_flagged": 1,
        "test_k4_control_an_unlisted_commit_and_an_unnamed_path_are_caught": 1,
    },
    "K7": {
        "test_k7_the_quick_tier_step_enforces_the_ceiling_with_medium_tests_counted": 1,
        "test_the_child_runs_with_a_clean_environment_and_the_budget_plugin": 1,
    },
    "A5": {"test_a5_the_twelve_stop_goldens_are_pinned": 1},
}

# Each spec mutant and the check that must fail on it: an inventory id or an own check.
MUTANT_CHECKS = {
    "K1 an asset-key change for asset 1": "test_k1_mutant_an_asset_key_change_moves_a_pinned_history",  # noqa: E501
    "K1 an always-on empty leaf": "test_k1_mutant_an_always_on_empty_leaf_moves_the_pins",  # noqa: E501
    "K1 a changed N_TIES": "test_k1_mutant_a_changed_n_ties_moves_a_pinned_history",
    "K1 lane closure on FAULT by default": GOLD
    + "test_a_lane_closing_fault_breaks_the_fault_pin",
    "K2 omit the expiry debit and its counter together": AWARDS
    + "test_a_paired_forged_expiry_is_caught",
    "K2 refund a nonrefundable surcharge": CONS
    + "test_mutant_a_refund_of_the_whole_charge_fails_conservation",
    "K2 credit an old award's refund to a newer award": AWARDS
    + "test_mutant_crediting_an_old_awards_refund_to_the_newer_award_fails",
    "K3 an extra handler pass per step": (
        "test_k3_the_same_job_ratio_is_recorded_or_k3_is_marked_pending"
    ),
    "K4 a core edit in a survey commit": "test_k4_control_a_survey_commit_that_touches_the_core_is_flagged",  # noqa: E501
    "K6 a truth-reading committee": S10
    + "test_a_committee_reading_the_carry_fails_the_perturbation",
    "K6 a visible internal refund_of": REFUND
    + "test_mutant_a_visible_internal_refund_id_moves_the_fingerprint",
    "K8 drop one award from the log": S10
    + "test_replay_reproduces_the_run_and_fails_without_a_recorded_award",
}

# The twelve mutants of the slice's acceptance list, one label each. The meta test
# requires exactly these labels in `MUTANT_CHECKS`, each tied to a check that ran and
# passed.
SPEC_MUTANTS = (
    "K1 an asset-key change for asset 1",
    "K1 an always-on empty leaf",
    "K1 a changed N_TIES",
    "K1 lane closure on FAULT by default",
    "K2 omit the expiry debit and its counter together",
    "K2 refund a nonrefundable surcharge",
    "K2 credit an old award's refund to a newer award",
    "K3 an extra handler pass per step",
    "K4 a core edit in a survey commit",
    "K6 a truth-reading committee",
    "K6 a visible internal refund_of",
    "K8 drop one award from the log",
)

# The last warm quick-tier run of the slice's head commit on the CI runner, read by hand
# from the job summary (CI time cannot be measured from inside pytest).
K7_RUN = {"commit": "ad113d3", "run_id": 42576344367, "wall_s": 391.2, "tests": 2681}
K7_CEILING_S = 540
# K3 is judged by one same-job measurement: `pilot.yml` measures the base tree (the
# eleven-handler table of the S0 end commit) and this tree in one job, interleaved, and
# prints their ratio (`tools/step_cost_pair.py`). It is pending until that run exists:
# set `K3_SAME_JOB` to the run's figures (keys below) and the gate then holds the ratio
# to the limit. The two runs in `K3_EARLIER_RUNS` came from separate jobs on separate
# runners, whose speed differs by more than the limit, so their plain figures are not
# compared. They are kept as context: the extra-pass mutant ratio inside each job shows
# the probe resolves a regression, and the cold first calls are the compile times.
K3_SAME_JOB = {
    "run_id": 42601136720,
    "base_commit": "97753d5",
    "final_commit": "a560033",
    "rounds": 5,
    "base_per_step_min_us": 99.11,
    "final_per_step_min_us": 79.76,
    "ratio": 0.805,
    "base_mutant_ratio": 1.526,
    "final_mutant_ratio": 1.618,
}
K3_SAME_JOB_KEYS = (
    "run_id",
    "base_commit",
    "final_commit",
    "rounds",
    "base_per_step_min_us",
    "final_per_step_min_us",
    "ratio",
    "base_mutant_ratio",
    "final_mutant_ratio",
)
K3_EARLIER_RUNS = {
    "base": {
        "run_id": 42584020950,
        "commit": "97753d5",
        "handlers": 11,
        "per_step_min_us": 244.11,
        "mutant_per_step_min_us": 355.99,
        "mutant_ratio": 1.439,
        "cold_s": 11.59,
        "mutant_cold_s": 14.279,
    },
    "final": {
        "run_id": 42584023997,
        "commit": "b851654",
        "handlers": 13,
        "per_step_min_us": 181.28,
        "mutant_per_step_min_us": 277.31,
        "mutant_ratio": 1.531,
        "cold_s": 9.632,
        "mutant_cold_s": 10.592,
    },
}
K3_LIMIT = 1.10  # the final table may cost at most 10 percent more per step
STEP_COST = REPO / "tools/step_cost.py"
STEP_PAIR = REPO / "tools/step_cost_pair.py"
PILOT = REPO / ".github/workflows/pilot.yml"
S0_RECORD = CHANGES / "slice-3a-s0.md"
S10_RECORD = CHANGES / "slice-3a-s10.md"


# ---- running the inventory ---------------------------------------------------------


def _module(nodeid):
    return nodeid.split("::")[0].removesuffix(".py").replace("/", ".")


def _function(nodeid):
    return nodeid.split("::")[1] if "::" in nodeid else None


def passed_counts(root, ids):
    """Number of passed cases for each id, from a junit file, and the other outcomes.

    Each case is counted once by its full node id (class name and test name with its
    parameters): a duplicated entry in the file adds nothing. A skip, an xfail (reported
    as a skip), a failure or an error is an outcome other than passed.
    """
    passed, other = collections.Counter(), collections.Counter()
    seen = set()
    for case in ET.parse(root).getroot().iter("testcase"):
        key = (case.get("classname"), case.get("name"))
        if key in seen:
            continue
        seen.add(key)
        module, name = key[0], key[1].split("[")[0]
        bad = any(case.find(t) is not None for t in ("skipped", "failure", "error"))
        for nodeid in ids:
            fn = _function(nodeid)
            if (module + ".").startswith(_module(nodeid) + ".") and fn in (None, name):
                (other if bad else passed)[nodeid] += 1
    return passed, other


def inventory_problems(inventory, passed, other):
    """Messages naming the rule of each listed id that did not run and pass exactly."""
    problems = []
    for rule, entries in inventory.items():
        for nodeid, n in entries.items():
            if n < 1:
                problems.append(f"{rule}: {nodeid} lists {n} cases; a check must run")
            elif other[nodeid] or passed[nodeid] != n:
                problems.append(
                    f"{rule}: {nodeid} passed {passed[nodeid]} of {n} cases"
                    f" ({other[nodeid]} skipped, failed or errored)"
                )
    return problems


_DEF_LINES = {}


def _line(module, function):
    """Line of a module-level test function in its file (large if it is not there)."""
    if module not in _DEF_LINES:
        tree = ast.parse((REPO / module).read_text())
        _DEF_LINES[module] = {
            n.name: n.lineno for n in tree.body if isinstance(n, ast.FunctionDef)
        }
    return _DEF_LINES[module].get(function, 10**9)


def ordered(ids):
    """The ids as the child should run them: files together, in the order of the file.

    The program budgets and the compile caches are judged per file in file order, so
    functions of one file run contiguously in the order they are written. An id that
    names a whole file covers the functions listed for that file.
    """
    modules = {}
    for nodeid in set(ids):
        modules.setdefault(nodeid.split("::")[0], set()).add(nodeid)
    out = []
    for module in sorted(modules):
        group = modules[module]
        if module in group:
            out.append(module)
        else:
            out += sorted(group, key=lambda i: _line(module, i.split("::")[1]))
    return out


def child_env(environ=None):
    """The environment of the child run: the parent's, without any pytest control.

    `PYTEST_ADDOPTS` (marker or keyword filters, disabled plugins, xdist settings),
    `PYTEST_PLUGINS`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD` and the xdist worker variables
    would change what the child runs without changing its command. Everything else
    stays, so `SPACEODYSSEY_JAX_CACHE` and the JAX settings apply to the child.
    """
    environ = os.environ if environ is None else environ
    return {k: v for k, v in environ.items() if not k.startswith("PYTEST_")}


def child_command(ids, xml, budget):
    """The child's command line: every option that matters is stated here."""
    return [
        sys.executable,
        "-m",
        "pytest",
        *ids,
        "-m",
        "",
        "-n",
        "0",
        "-q",
        "-p",
        "no:cacheprovider",
        "--budget-json",
        str(budget),
        "--junitxml",
        str(xml),
        # a test that passes under a non-strict xfail must fail, not pass quietly
        "-o",
        "xfail_strict=true",
        # each case gets its own limit, sized for a runner about three times slower
        *(["--timeout", str(CHILD_TIMEOUT_S)] if HAS_TIMEOUT else []),
    ]


@pytest.fixture(scope="module")
def child(tmp_path_factory):
    """One child pytest run of every inventory id: its junit counts and exit code."""
    ids = sorted({i for entries in INVENTORY.values() for i in entries})
    folder = tmp_path_factory.mktemp("inventory")
    xml, budget = folder / "junit.xml", folder / "budget.json"
    run = subprocess.run(
        child_command(ordered(ids), xml, budget),
        cwd=REPO,
        env=child_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert xml.exists(), run.stdout[-3000:] + run.stderr[-1000:]
    passed, other = passed_counts(xml, ids)
    return {
        "passed": passed,
        "other": other,
        "code": run.returncode,
        "out": run.stdout,
        "budget": budget,
    }


# The gate's own bookkeeping tests, which `OWN` does not list.
GATE_TESTS = (
    "test_inventory_every_listed_check_was_collected_and_passed",
    "test_zz_every_rule_item_and_mutant_has_a_check_that_ran_and_passed",
)


def _whole_module_selected(config):
    """False when the run narrowed the module by keyword, deselection or node id."""
    narrowed = config.getoption("keyword") or config.getoption("deselect")
    return not narrowed and not any("::" in str(arg) for arg in config.args)


class _Outcomes:
    """Passed cases of the checks of this file, recorded as they report."""

    def __init__(self):
        self.passed = collections.Counter()
        self.other = collections.Counter()

    def pytest_runtest_logreport(self, report):
        name = report.nodeid.split("::")[-1].split("[")[0]
        if report.when == "call" and report.passed:
            self.passed[name] += 1
        elif report.when != "teardown" and not report.passed:
            self.other[name] += 1


@pytest.fixture(scope="module", autouse=True)
def outcomes(request):
    recorder = _Outcomes()
    request.config.pluginmanager.register(recorder)
    yield recorder
    request.config.pluginmanager.unregister(recorder)
    if _whole_module_selected(request.config):
        required = {name for entries in OWN.values() for name in entries} | set(
            GATE_TESTS
        )
        missing = sorted(required - set(recorder.passed))
        assert not missing, f"gate checks that did not run and pass: {missing}"


def test_inventory_every_listed_check_was_collected_and_passed(child):
    """Fails if any canonical id of any rule or acceptance item is missing, skipped,
    xfailed, failed or ran another number of cases, or if the child run exited nonzero.
    """
    assert all(INVENTORY.values())  # no rule with an empty list
    problems = inventory_problems(INVENTORY, child["passed"], child["other"])
    assert problems == [], "\n".join(problems)
    assert child["code"] == 0, child["out"][-3000:]


def test_the_child_runs_with_a_clean_environment_and_the_budget_plugin(child):
    """Fails if a pytest control variable can reach the child, if its command does not
    state the marker, worker and budget options, or if the budget plugin was not loaded
    and enforcing in the child run: its `--budget-json` file must exist, be in enforce
    mode with no violation and hold a record for every module of the inventory.
    """
    leaky = {
        "PYTEST_ADDOPTS": "-m 'not slow' -p no:hwoutils.testbudget.plugin",
        "PYTEST_PLUGINS": "x",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_XDIST_WORKER": "gw1",
        "PYTEST_XDIST_WORKER_COUNT": "2",
        "SPACEODYSSEY_JAX_CACHE": "off",
    }
    assert child_env(leaky) == {"SPACEODYSSEY_JAX_CACHE": "off"}
    cmd = child_command(["a"], "j.xml", "b.json")[3:]
    assert cmd[cmd.index("-m") + 1] == "" and cmd[cmd.index("-n") + 1] == "0"
    assert "--budget-json" in cmd and "--junitxml" in cmd
    assert cmd[cmd.index("-o") + 1] == "xfail_strict=true"
    if HAS_TIMEOUT:
        assert int(cmd[cmd.index("--timeout") + 1]) == CHILD_TIMEOUT_S
        marks = [m for m in pytestmark if getattr(m, "name", "") == "timeout"]
        assert marks and marks[0].args == (GATE_TIMEOUT_S,)
    assert GATE_TIMEOUT_S >= 3 * 1.1 * 376  # the child's cold time at the runner's pace
    budget = json.loads(child["budget"].read_text())
    assert budget["mode"] == "enforce" and budget["module_mode"] == "enforce"
    assert budget["violations"] == []
    ran = {r["nodeid"].split("::")[0] for r in budget["records"]}
    wanted = {i.split("::")[0] for entries in INVENTORY.values() for i in entries}
    assert wanted <= ran, sorted(wanted - ran)


def _pin():
    return json.loads(golden.PIN.read_text())


# ---- K1 ----------------------------------------------------------------------------


def _blob_at_capture(path):
    """The pin file as committed by the capture commit that last wrote it."""
    rel = str(path.relative_to(REPO))
    log = _git(
        "log", "--format=%H", f"{pin_helpers.PIN_SOURCE}..HEAD", "--", rel
    ).split()
    captures = [c for c in log if c in pin_helpers.PIN_CAPTURES]
    assert captures, rel
    return subprocess.run(
        ["git", "cat-file", "blob", f"{captures[0]}:{rel}"],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout


def test_k1_every_pin_file_is_the_blob_of_its_capture_commit(child):
    """Fails if a pin file differs by one byte from the blob its capture commit wrote,
    for instance because a test loader regenerated it. It runs after the child run,
    so a pin rewritten by that run is seen.
    """
    for path in (pin_helpers.GOLDEN_PIN, pin_helpers.HASH_PIN):
        assert path.read_bytes() == _blob_at_capture(path), path.name
    assert len(_pin()["histories"]) == golden.N_PINNED > 0


def test_k1_no_commit_after_the_pins_touched_a_pin_file():
    """Fails if a pin names another source commit, or if any commit from the pins'
    capture to HEAD, other than the two capture commits, changed a pin file.
    """
    assert json.loads(pin_helpers.GOLDEN_PIN.read_text())["source_commit"] == (
        pin_helpers.PIN_SOURCE
    )
    assert json.loads(pin_helpers.HASH_PIN.read_text())["source_commit"] == (
        pin_helpers.PIN_SOURCE
    )
    touched = set(
        _git(
            "log",
            "--format=%H",
            f"{pin_helpers.PIN_SOURCE}..HEAD",
            "--",
            str(pin_helpers.GOLDEN_PIN.relative_to(REPO)),
            str(pin_helpers.HASH_PIN.relative_to(REPO)),
        ).split()
    )
    assert touched == pin_helpers.PIN_CAPTURES, sorted(
        touched - pin_helpers.PIN_CAPTURES
    )


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
    # the mutant table covers handlers 0 to 8 with RESULT stubbed out; the study
    # must handle only event kinds it covers, or the mutant would run the wrong handler
    covered = set(range(len(mutants.ASSET_KEY_HANDLERS))) - {int(core.EventKind.RESULT)}
    pinned = _pin()["histories"]
    runs = _corpus_history("phase_switch")
    control = _digests(runs)
    for key, run in runs.items():
        handled = {kind for _, _, kind in run.history.events}
        assert handled <= covered, (key, sorted(handled - covered))
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


# ---- K3 and K7 ---------------------------------------------------------------------


def _pair_tool():
    """The same-job comparison tool, loaded by path (it does not import JAX)."""
    spec = importlib.util.spec_from_file_location("step_cost_pair", STEP_PAIR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extra_pass_calls():
    """Names called by `ExtraPassExecutor.transition` in the probe, from its source."""
    tree = ast.parse(STEP_COST.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ExtraPassExecutor"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "transition"
    )
    return [
        n.func.id
        for n in ast.walk(method)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]


def test_k3_the_step_cost_measurement_is_wired_for_the_final_table():
    """Fails if the table is not the thirteen-handler table K3 is judged at, if the
    probe or its calibrated mutant (which must still run the extra select-and-handle
    pass) is gone, or if the pilot workflow does not measure the base tree and this
    tree in one job with the exit status of each measurement kept.
    """
    assert len(HANDLERS) == 13
    kinds = {k.name for k in core.EventKind}
    assert {"WINDOW_END", "COMMIT_START"} <= kinds
    probe = STEP_COST.read_text()
    assert "class ExtraPassExecutor" in probe and "block_until_ready" in probe
    calls = _extra_pass_calls()
    assert calls.count("transition") == 1
    assert "select_next" in calls and "step" in calls and "where_tree" in calls
    assert "--baseline-per-step-us" not in probe  # no typed-in baseline
    pair = STEP_PAIR.read_text()
    assert "base_runs.append(run_probe(" in pair
    assert "final_runs.append(run_probe(" in pair
    assert "for _ in range(args.rounds)" in pair and "SystemExit(1)" in pair
    workflow = PILOT.read_text()
    assert "tools/step_cost.py --mutant" in workflow
    assert "tools/step_cost_pair.py --base-tree base-tree" in workflow
    assert "path: base-tree" in workflow and "k3_base_ref" in workflow
    assert "baseline_per_step_min_us" not in workflow
    # a pipe keeps the status of its last command unless pipefail is set
    for line in workflow.splitlines():
        if "| tee step_cost" in line:
            head = workflow[: workflow.index(line)]
            assert head.rstrip().rsplit("\n", 1)[-1].strip() == "set -o pipefail" or (
                "set -o pipefail" in head.rsplit("- name:", 1)[-1]
            ), line


def test_k3_the_same_job_ratio_is_recorded_or_k3_is_marked_pending():
    """Fails if the same-job record has a missing key, a ratio that does not follow
    from its two figures, a ratio over 1.10, a mutant ratio that does not exceed 1.10,
    or figures absent from the S10 record. While no same-job run exists the record must
    say K3 is pending a same-job measurement. The earlier separate-job runs are context:
    their mutant ratios must exceed the limit and their figures stay in the record.
    """
    record = S10_RECORD.read_text()
    for run in K3_EARLIER_RUNS.values():
        for key in ("run_id", "commit", "per_step_min_us", "mutant_ratio", "cold_s"):
            assert str(run[key]) in record, key
        assert str(run["mutant_cold_s"]) in record
        assert run["mutant_ratio"] > K3_LIMIT
    same = K3_SAME_JOB
    if same is None:
        assert "pending a same-job measurement" in record
        return
    assert set(K3_SAME_JOB_KEYS) <= set(same), sorted(set(K3_SAME_JOB_KEYS) - set(same))
    assert (
        round(same["final_per_step_min_us"] / same["base_per_step_min_us"], 3)
        == (same["ratio"])
    )
    assert same["ratio"] <= K3_LIMIT, same["ratio"]
    assert same["base_mutant_ratio"] > K3_LIMIT
    assert same["final_mutant_ratio"] > K3_LIMIT
    for key in ("run_id", "ratio", "base_per_step_min_us", "final_per_step_min_us"):
        assert str(same[key]) in record, key
    assert "pending a same-job measurement" not in record


def test_k3_the_ratio_check_separates_the_limit():
    """Fails if the same-job comparison passes a final tree over 1.10 times the base,
    fails one within it, or compares anything but the lowest minimum of each tree.
    """
    pair = _pair_tool()
    assert pair.LIMIT == K3_LIMIT == 1.10

    def runs(minima):
        return [
            {
                "handlers": 11,
                "plain": {"per_step_min_us": m, "cold_s": 1.0},
                "mutant": {"per_step_min_us": 1.5 * m, "cold_s": 1.0},
                "mutant_ratio": 1.5,
            }
            for m in minima
        ]

    base = runs([100.0, 120.0])
    assert pair.compare(base, runs([110.0, 150.0]))["within_limit"]
    assert not pair.compare(base, runs([111.0, 111.0]))["within_limit"]
    # the lowest minimum of each tree: one noisy round does not decide the ratio
    noisy = pair.compare(base, runs([300.0, 105.0]))
    assert noisy["ratio"] == 1.05 and noisy["within_limit"]


def test_k7_the_quick_tier_step_enforces_the_ceiling_with_medium_tests_counted():
    """Fails if the quick-tier CI step runs the summary advisory only, drops the
    540 s ceiling or the medium tier, or if the budget plugin is not enforcing.
    """
    step = pin_helpers.quick_summary_step()[1]
    assert "--advisory-baseline" not in step
    assert f"--budget-seconds {K7_CEILING_S}" in step
    assert "--require-tier medium" in step
    assert "hwoutils.testbudget.plugin" in conftest.pytest_plugins
    assert K7_RUN["wall_s"] < K7_CEILING_S and K7_RUN["tests"] > 0


def test_k3_k7_manual_reads(capsys):
    """Prints the K3 and K7 figures the gate reads from the CI runner, and fails if
    the S10 record does not carry the K7 run.
    """
    run, same = K7_RUN, K3_SAME_JOB
    with capsys.disabled():
        if same is None:
            print(
                "\nK3 pending a same-job measurement: dispatch .github/workflows/"
                "pilot.yml and record its run id, ratio and mutant ratios in "
                "K3_SAME_JOB and the S10 record."
            )
        else:
            print(
                f"\nK3 same-job read: run {same['run_id']}, ratio {same['ratio']} "
                f"against {K3_LIMIT} over {same['rounds']} rounds; extra-pass mutant "
                f"{same['base_mutant_ratio']} (base) and {same['final_mutant_ratio']} "
                "(final)."
            )
        print(
            f"K7 manual read: warm CI run {run['run_id']} of {run['commit']}, quick "
            f"tier {run['wall_s']} s over {run['tests']} tests, ceiling "
            f"{K7_CEILING_S} s."
        )
    record = S10_RECORD.read_text()
    assert str(run["run_id"]) in record and str(run["wall_s"]) in record


# ---- K4 ----------------------------------------------------------------------------

S0_END = "97753d57564770664fca3b73c26840d0681dacf2"
# the last commit before the gate task: the range the attribution covers ends here, so
# a later commit cannot widen or break it
K4_END = "ad113d30d7e4d2721e718133844c9694da2c0a1d"

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
    # the earlier premission work (R1 to R3), judged by its own gate; the paths are in
    # its records
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
# History names the records under their old directory, `docs/changes/`.
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
    """Every non-merge commit from the last S0 commit to the K4 end that touches the
    core.
    """
    return _git(
        "rev-list", "--no-merges", f"{S0_END}..{K4_END}", "--", CORE.rstrip("/")
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
    """Whether a record names a core path exactly, as the full repository path."""
    pattern = r"(?<![\w/.-])" + re.escape(path) + r"(?!\w|[./-]\w)"
    return bool(re.search(pattern, text))


def _touched_in_range():
    """Every path any commit of the range changed."""
    return {p for paths in commit_paths().values() for p in paths}


def _core_test_cited(text, touched):
    """Whether a record cites an existing `tests/core/test_*.py` file that some commit
    of the range changed. The check does not tie the file to the task's own commits (a
    task whose tests landed in another commit would fail that), so a record could cite a
    core test of another task; reviewers read that part.
    """
    cited = re.findall(r"tests/core/test_\w+\.py", text)
    return any((REPO / path).exists() and path in touched for path in cited)


def unattributed(owners, commits, text_of, touched=None):
    """Problems with the attribution: commits no task lists, stale entries, tasks whose
    records cite no core test, and exact core paths a record does not name.
    """
    touched = _touched_in_range() if touched is None else touched
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
        if not _core_test_cited(text_of(task), touched):
            found.append(f"the {task} record cites no tests/core test file")
    found += unnamed_paths(owners, commits, text_of)
    return found


def unnamed_paths(owners, commits, text_of):
    """Exact core paths a commit touched that its task's record does not name."""
    return [
        f"{sha[:7]} {path} not named in the {owners[sha]} record"
        for sha in commits
        if sha in owners
        for path in _core_paths(sha)
        if not _mentions(text_of(owners[sha]), path)
    ]


def _text_of(task):
    records = {**TASKS, **OTHER_TASKS}[task][0]
    return _record_text(records)


def commit_paths():
    """Paths touched by each non-merge commit from the last S0 commit to the K4 end."""
    out = _git(
        "log",
        "--no-merges",
        "--no-renames",
        "--name-only",
        "--format=@@%H",
        f"{S0_END}..{K4_END}",
    )
    commits, sha = {}, None
    for line in out.splitlines():
        if line.startswith("@@"):
            sha = line[2:]
            commits[sha] = []
        elif line:
            commits[sha].append(line)
    return commits


def survey_commits_touching_the_core(commits):
    """Commits (sha -> touched paths) that add a survey test or record and also touch
    the core, plus the number of survey commits seen.
    """
    survey = {s: p for s, p in commits.items() if set(p) & set(SURVEY_FILES)}
    return [s for s, p in survey.items() if any(f.startswith(CORE) for f in p)], len(
        survey
    )


def test_k4_every_core_change_is_attributed_to_a_task_record_and_a_core_test():
    """Fails if a core commit after S0 is listed under no task, a listed commit is not
    in the range, a task's record cites no existing core test, or a record does not name
    an exact core path its commit touched.
    """
    commits = _core_commits()
    assert len(commits) >= 40  # evidence: the slice's core work is in the range
    owners = _owners()
    assert unattributed(owners, commits, _text_of) == []
    assert {"S1", "S2", "S3a", "S4", "S5", "S5b", "S6"} <= set(owners.values())


def test_k4_no_survey_commit_touches_the_core():
    """Fails if a commit that adds a survey test or its record touches `core/`."""
    touching, n_survey = survey_commits_touching_the_core(commit_paths())
    assert n_survey >= 4  # evidence: surveys 8, 9, 10 and 11 each landed
    assert touching == [], [s[:7] for s in touching]


def test_k4_control_a_survey_commit_that_touches_the_core_is_flagged():
    """Control: a synthetic history with a survey commit that edits the core is flagged
    by the same discovery the real check uses, and a clean survey commit is not.
    """
    history = {
        "aaa": ["tests/studies/test_survey_9.py", CORE + "transaction.py"],
        "bbb": ["tests/studies/test_survey_10.py", "docs/changes/slice-3a-survey10.md"],
        "ccc": [CORE + "carry.py"],
    }
    touching, n_survey = survey_commits_touching_the_core(history)
    assert touching == ["aaa"] and n_survey == 2


def test_k4_control_an_unlisted_commit_and_an_unnamed_path_are_caught():
    """Control: dropping a commit from the owner table is reported, and a record that
    names a path only by its module name or directory does not name it.
    """
    commits = _core_commits()
    owners = _owners()
    dropped = _resolve("f155d0b")
    fewer = {s: t for s, t in owners.items() if s != dropped}
    assert any(dropped[:7] in p for p in unattributed(fewer, commits, _text_of))
    path = _core_paths(dropped)[0]
    stem = path.removeprefix(CORE).removesuffix(".py")
    loose = lambda task: f"tests/core/test_lints.py {stem} core/{stem.split('/')[0]}/"  # noqa: E731
    assert unnamed_paths(owners, [dropped], loose)
    exact = lambda task: "tests/core/test_lints.py " + " ".join(_core_paths(dropped))  # noqa: E731
    assert unnamed_paths(owners, [dropped], exact) == []
    assert any("not named" in p for p in unattributed(owners, [dropped], loose))
    full = _core_paths(dropped)[0]
    assert _mentions(f"`{full}`, then", full) and _mentions(f"see {full}.", full)
    for other in (full + ".bak", full + "c", "x/" + full, "pre-" + full, full + "-old"):
        assert not _mentions(other, full), other
    nothing = lambda task: "no paths here"  # noqa: E731
    assert any("cites no" in p for p in unattributed(owners, [dropped], nothing))


# ---- A5 and the meta check -----------------------------------------------------------

STOP_GOLDENS = tuple(
    f"stop_{scope}_{disposition}"
    for scope in ("asset", "campaign", "lane", "program")
    for disposition in ("abort", "close", "drain")
)


def test_a5_the_twelve_stop_goldens_are_pinned():
    """Fails if a `stop_*` corpus case lost its pin on either executor and world."""
    pinned = _pin()["histories"]
    for name in STOP_GOLDENS:
        for executor in ("host", "compiled"):
            for world in range(N_WORLDS):
                assert golden.entry_key(name, executor, world) in pinned, name
    assert len(STOP_GOLDENS) == 12


def test_zz_every_rule_item_and_mutant_has_a_check_that_ran_and_passed(child, outcomes):
    """Fails if a rule K1 to K8 or an item A1 to A10 has no listed check, a listed
    check of this file did not pass, or a spec mutant is not tied to a check that ran
    and passed (an inventory id from the child run, or a check of this file).
    """
    for rule in [f"K{i}" for i in range(1, 9)] + [f"A{i}" for i in range(1, 11)]:
        assert INVENTORY.get(rule) or OWN.get(rule), f"{rule} has no check"
    own = {name: n for entries in OWN.values() for name, n in entries.items()}
    for name, n in own.items():
        assert n >= 1 and callable(globals().get(name)), name
        assert outcomes.passed[name] == n and not outcomes.other[name], name
    inventory = {i for entries in INVENTORY.values() for i in entries}
    for label, check in MUTANT_CHECKS.items():
        if check in own:
            assert outcomes.passed[check] == own[check], label
        else:
            assert check in inventory, label
            assert child["passed"][check] > 0 and not child["other"][check], label
    assert len(set(SPEC_MUTANTS)) == len(SPEC_MUTANTS) == 12
    assert set(MUTANT_CHECKS) == set(SPEC_MUTANTS)
