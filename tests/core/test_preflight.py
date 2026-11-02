"""Pre-flight: every code catches its seeded mistake and passes the correct studies."""

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
