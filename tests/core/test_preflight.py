"""Pre-flight: every code catches its seeded mistake and passes the correct studies."""

import dataclasses

import pytest

from spaceodyssey import core
from spaceodyssey.core import FindingCode, Severity, StudyGraph, preflight
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.studies import toy
from tests.core.preflight_cases import CLEAN, SEEDED


def codes(findings):
    return sorted({str(f.code) for f in findings})


@pytest.mark.parametrize("name", sorted(SEEDED))
def test_seeded_mistake_is_caught(name):
    code, make, executor = SEEDED[name]
    found = preflight(make(), executor)
    assert codes(found) == [str(code)], [str(f) for f in found]
    finding = found[0]
    assert finding.path and finding.message and finding.fix
    assert (finding.message + finding.fix + finding.path).isascii()


@pytest.mark.parametrize("name", sorted(CLEAN))
def test_correct_study_is_clean(name):
    make, expected = CLEAN[name]
    found = preflight(make())
    assert codes(found) == sorted(str(c) for c in expected), [str(f) for f in found]
    assert not [f for f in found if f.severity == Severity.ERROR]


def test_every_code_has_a_seeded_mistake():
    assert {c for c, _, _ in SEEDED.values()} == set(FindingCode)


def test_firewall_paths_name_reader_and_source():
    truth = preflight(SEEDED["observer_reads_truth"][1]())[0]
    scorer = preflight(SEEDED["observer_reads_scorer"][1]())[0]
    assert truth.path == "observer <- world"
    assert scorer.path == "observer <- objective"
    assert {truth.severity, scorer.severity} == {Severity.ERROR}


def test_drain_below_campaign_is_a_warning():
    finding = preflight(SEEDED["drain_below_campaign"][1]())[0]
    assert finding.severity == Severity.WARNING


def test_executor_selection_changes_x001():
    make = SEEDED["gradients_on_host"][1]
    assert codes(preflight(make(), "host")) == ["X001"]
    assert codes(preflight(make(), "compiled")) == []
    assert codes(preflight(make())) == []


def test_unknown_declaration_names_fail_at_construction():
    with pytest.raises(ValueError):
        core.Declaration(reads=("oracle",))
    with pytest.raises(ValueError):
        core.Declaration(key_coords=("nothing",))
    with pytest.raises(ValueError):
        core.Declaration(executors=frozenset({"gpu"}))


def test_graph_crosses_truth_only_at_measures():
    graph = StudyGraph.from_study(toy.survey_1())
    kinds = {n.name: (str(n.kind), str(n.side)) for n in graph.nodes.values()}
    assert kinds["world"] == ("source", "truth")
    assert kinds["channel[0]"] == ("measure", "truth")
    assert kinds["observer"] == ("update", "observer")
    assert kinds["policy[0]"] == ("decide", "observer")
    assert kinds["objective"] == ("score", "scorer")
    from_truth = [
        e
        for e in graph.edges
        if str(graph.nodes[e.reader].side) == "observer"
        and str(graph.nodes[e.source].side) == "truth"
    ]
    assert from_truth and all(graph.nodes[e.source].crossing for e in from_truth)


def test_preflight_runs_nothing(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("pre-flight ran the study")

    for cls in (CompiledExecutor, HostExecutor):
        monkeypatch.setattr(cls, "run", refuse)
    monkeypatch.setattr(core.handlers, "step", refuse)
    assert preflight(toy.survey_1()) == []


def test_preflight_reports_counts(capsys):
    """Run every study, print the report and check its counts.

    A false positive is a finding that the study's own fixture does not expect, on a
    correct study or on a seeded one (which carries exactly one mistake).
    """
    errors = warnings = false_positives = missed = 0
    exercised = set()
    for code, make, executor in (SEEDED[n] for n in sorted(SEEDED)):
        found = preflight(make(), executor)
        errors += sum(f.severity == Severity.ERROR for f in found)
        warnings += sum(f.severity == Severity.WARNING for f in found)
        exercised |= {f.code for f in found}
        false_positives += sum(f.code != code for f in found)
        missed += code not in {f.code for f in found}
    clean_errors = 0
    for make, expected in CLEAN.values():
        found = preflight(make())
        clean_errors += sum(f.severity == Severity.ERROR for f in found)
        warnings += sum(f.severity == Severity.WARNING for f in found)
        exercised |= {f.code for f in found}
        false_positives += sum(f.code not in expected for f in found)
        missed += len(set(expected) - {f.code for f in found})
    print(
        f"preflight: {len(SEEDED)} seeded and {len(CLEAN)} correct studies; "
        f"{errors + clean_errors} errors ({clean_errors} on correct studies), "
        f"{warnings} warnings; codes exercised: {', '.join(sorted(exercised))}; "
        f"false positives: {false_positives}; missed: {missed}"
    )
    out = capsys.readouterr().out
    assert "false positives: 0" in out and "missed: 0" in out
    assert clean_errors == 0
    assert exercised == set(FindingCode)
    with capsys.disabled():
        print(out, end="")


# name -> (subject, index, member, entry) every finding of the seeded study carries
LOCATORS = {
    "observer_reads_truth": ("observer", None, "reads", (0,)),
    "policy_reads_truth": ("policy", 0, "reads", (0,)),
    "observer_reads_scorer": ("observer", None, "reads", (0,)),
    "decision_before_release": ("policy", 0, "needs", (0,)),
    "decision_reads_observer_only_channel": ("policy", 0, "needs", (0,)),
    "draws_without_acquisition_index": ("channel_model", None, "key_coords", ()),
    "world_draw_without_world": ("world_source", None, "key_coords", ()),
    "duplicate_channel_ids": ("channels", None, "id", (1,)),
    "allocate": ("policy", 0, "effects", ()),
    "gradients_on_host": ("policy", 0, "requires", ()),
    "host_only_policy_on_compiled": ("policy", 0, "executors", ()),
    "charge_differs_from_cost": ("channels", None, "charge", (0, 0)),
    "frozen_horizon": ("assets", None, "freeze_ticks", (0,)),
    "disruptive_resource": ("assets", None, "disruptive_resource", (0,)),
    "disruptive_effect": ("policy", 0, "disruptive", ()),
    "drain_below_campaign": ("policy", 0, "stops", (1,)),
    "channel_without_order": ("channels", None, "coincident_order", ()),
    "channel_order_out_of_range": ("channels", None, "coincident_order", (2,)),
    "channel_kind_unregistered": ("channels", None, "kind", (2,)),
    "import_upstream_unverified": ("process", None, "table.upstream_hash", ()),
    "import_pedigree_absent": ("process", None, "table.pedigree.source_index", (1, 0)),
    "import_index_repeated": ("process", None, "table.local_index", (1,)),
    "import_offset_positive": ("process", None, "table.end_offset", (0,)),
}


def locator(finding):
    return (finding.subject, finding.index, finding.member, finding.entry)


def test_every_seeded_mistake_has_an_expected_locator():
    assert set(LOCATORS) == set(SEEDED)


@pytest.mark.parametrize("name", sorted(SEEDED))
def test_finding_carries_the_locator_of_its_value(name):
    """Fails if a finding names the wrong slot, policy, member or entry, or none."""
    _, make, executor = SEEDED[name]
    found = preflight(make(), executor)
    assert found
    assert {locator(f) for f in found} == {LOCATORS[name]}


@pytest.mark.parametrize("name", ["duplicate_channel_ids", "charge_differs_from_cost"])
def test_table_locators_name_a_real_field_of_the_slot(name):
    """Fails if a table finding names a slot or field the study does not have."""
    study = SEEDED[name][1]()
    finding = preflight(study)[0]
    slot = getattr(study, finding.subject)
    assert finding.member in {f.name for f in dataclasses.fields(slot)}


def test_repeated_declarations_are_told_apart_by_entry():
    """Fails if repeated reads, needs or drains share one entry, so a tool cannot say
    which of them offends.
    """
    from spaceodyssey.core.declarations import TRUTH, Declaration, Need
    from spaceodyssey.core.enums import Disposition, Scope
    from tests.core.preflight_cases import _observer, _policy

    reads = preflight(_observer(reads=(TRUTH, TRUTH, TRUTH)))
    assert [f.entry for f in reads] == [(0,), (1,), (2,)]
    needs = preflight(_policy(needs=(Need(10, 0, 500), Need(10, 0, 19))))
    assert [(str(f.code), f.entry) for f in needs] == [("C001", (1,))]
    stops = ((Scope.CAMPAIGN, Disposition.DRAIN), (Scope.LANE, Disposition.DRAIN))
    assert [f.entry for f in preflight(_policy(stops=stops))] == [(1,)]
    assert Declaration().reads == ()


def test_a_finding_that_does_not_narrow_has_an_empty_locator():
    """Fails if a hand-built finding does not default to no locator."""
    finding = core.Finding(FindingCode.T001, Severity.WARNING, "p", "m", "f")
    assert locator(finding) == ("", None, "", ())


def _upstream_study(declared):
    """The import study with its table's declared upstream hash set."""
    from tests.core.preflight_cases import import_study

    study = import_study()
    table = dataclasses.replace(study.process.table, upstream_hash=declared)
    return dataclasses.replace(study, process=study.process.__class__(table=table))


def test_upstream_hash_passes_fails_and_is_unverified_through_the_context():
    """Fails if D004 does not pass with the named upstream document, fail with
    another one, and report "declared, unverified" (a warning) with none.
    """
    from spaceodyssey.codec import dump_study
    from spaceodyssey.core.preflight.checks import PreflightContext
    from tests.codec.test_document import TOY_KINDS

    upstream = dump_study(toy.survey_1(), TOY_KINDS)
    other = dump_study(toy.survey_1(horizon_tick=900), TOY_KINDS)
    study = _upstream_study(upstream.content_hash)
    right = PreflightContext(upstream_document=upstream)
    wrong = PreflightContext(upstream_document=other)
    assert preflight(study, context=right) == []
    found = preflight(study, context=wrong)
    assert [(str(f.code), f.severity) for f in found] == [("D004", Severity.ERROR)]
    assert other.content_hash in found[0].message
    for context in (None, PreflightContext()):
        found = preflight(study, context=context)
        assert [(str(f.code), f.severity) for f in found] == [
            ("D004", Severity.WARNING)
        ]
        assert "declared, unverified" in found[0].message
    # no declared hash: nothing to verify, whatever document is supplied
    assert preflight(_upstream_study(""), context=wrong) == []


def _table_of(n_rows):
    from spaceodyssey.studies.premission import PreMissionImporter, authored_table

    table = authored_table(
        subject=[i % 2 for i in range(n_rows)],
        local_index=list(range(n_rows)),
        channel=[60] * n_rows,
        kind=[toy.RECORD_MEASUREMENT] * n_rows,
        audience=[3] * n_rows,
        payload=[[1.0]] * n_rows,
    )
    return toy.survey_1(process=PreMissionImporter(table=table))


@pytest.mark.parametrize(
    ("field", "held", "rows"),
    [
        ("n_events", "queue slots", 9),
        ("n_log", "log rows", 10),
        ("n_steps", "steps", 10),
    ],
)
def test_an_import_one_row_too_large_fails_only_with_the_real_budget(field, held, rows):
    """Fails if a table that just fits the run's queue, log or step budget is
    refused, if one row more is accepted, or if the capacity check runs without a
    budget. The queue holds every process row and the kickoff wait; the steps before
    the first decision are one per event plus the kickoff.
    """
    from spaceodyssey.core.preflight.checks import PreflightContext, RunBudget

    tight = {"n_events": 10, "n_log": 10, "n_steps": 2 * 10 + 1}[field]
    budget = dataclasses.replace(RunBudget(1000, 1000, 1000), **{field: tight})
    context = PreflightContext(budget=budget)
    fits, over = _table_of(rows), _table_of(rows + 1)
    assert preflight(fits, context=context) == []
    found = preflight(over, context=context)
    assert [str(f.code) for f in found] == ["D006"]
    assert held in found[0].message and found[0].member == "table"
    assert preflight(over) == []


def test_a_hook_finding_outside_the_study_falls_back_to_its_component():
    """Fails if a validator's finding that names no slot of the study, or a policy
    that is missing or out of range, keeps that locator instead of falling back to
    the validating component's own slot.
    """
    from tests.codec.test_preflight_json import LocatorProcess

    for subject, index in (("missing", None), ("policy", None), ("policy", 3)):
        study = toy.survey_1(process=LocatorProcess(subject=subject, index=index))
        found = preflight(study)
        assert [locator(f) for f in found] == [("process", None, "", ())]
    kept = preflight(toy.survey_1(process=LocatorProcess(subject="policy", index=0)))
    assert [locator(f) for f in kept] == [("policy", 0, "", ())]


def test_the_locatable_slots_are_the_fields_of_the_study():
    """Fails if a field is added to or removed from `Study` without the pre-flight's
    list of slots a finding may name following it.
    """
    from spaceodyssey.core.preflight.checks import STUDY_SLOTS
    from spaceodyssey.core.study import Study

    assert {f.name for f in dataclasses.fields(Study)} == STUDY_SLOTS
