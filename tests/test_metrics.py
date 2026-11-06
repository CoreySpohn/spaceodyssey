"""Tests for the YieldSummary report and summarize scoring."""

from types import SimpleNamespace

import pytest

from spaceodyssey.metrics import YieldSummary, summarize


class TestYieldSummary:
    """The report schema."""

    def test_json_round_trip(self):
        """A summary survives serialization exactly."""
        summary = YieldSummary(
            operator="sample",
            certificate="detection",
            certified_targets=(1, 3),
            false_certified_targets=(3,),
            epochs=6,
            resources_spent=(("time_d", 6.6),),
            wrong_certified_targets=(1,),
        )
        assert YieldSummary.from_json(summary.to_json()) == summary
        assert summary.false_certifications == 1

    def test_reads_reports_without_claim_scoring(self):
        """Reports written before claim-level scoring still load."""
        text = (
            '{"certificate":"detection","certified_targets":[1],"epochs":1,'
            '"false_certified_targets":[],"operator":"sample",'
            '"resources_spent":[["time_d",1.0]]}'
        )
        assert YieldSummary.from_json(text).wrong_certified_targets == ()


def make_state(certified, *, epochs=6, spent=None):
    """A final-state double: `certified` ids, an epoch count and resource ledger."""
    belief = SimpleNamespace(certified=lambda name: tuple(certified))
    ledger = SimpleNamespace(spent=spent or {"time_d": 6.6})
    return SimpleNamespace(belief=belief, ledger=ledger, epoch_index=epochs)


def make_universe(**scenes):
    """A truth double mapping target ids (given as t<N>) to scenes."""
    systems = {int(name[1:]): scene for name, scene in scenes.items()}
    return SimpleNamespace(systems=SimpleNamespace(get=systems.get))


class TestSummarize:
    """Scoring a finished run against its drawn truth."""

    def test_scored_summary_carries_run_accounting(self):
        """Certified targets, epochs and sorted resources pass through."""
        summary = summarize(
            make_state((1, 3), spent={"time_d": 6.6, "cost": 2.0}),
            make_universe(t1="planet", t2=None, t3="planet"),
            operator="sample",
        )
        assert summary.operator == "sample"
        assert summary.certified_targets == (1, 3)
        assert summary.false_certified_targets == ()
        assert summary.certified == 2
        assert summary.epochs == 6
        assert dict(summary.resources_spent)["time_d"] == pytest.approx(6.6)
        assert summary.resources_spent == (("cost", 2.0), ("time_d", 6.6))

    def test_false_certifications_are_counted(self):
        """A certified target with no real source is scored, not hidden."""
        summary = summarize(
            make_state((1, 2, 3)),
            make_universe(t1="planet", t2=None, t3="planet"),
            operator="sample",
        )
        assert summary.certified_targets == (1, 2, 3)
        assert summary.false_certified_targets == (2,)
        assert summary.false_certifications == 1

    def test_wrong_claims_scored_apart_from_absent_sources(self):
        """Claim-level scoring counts real targets only, never absent ones."""
        summary = summarize(
            make_state((1, 2, 3)),
            make_universe(t1="planet", t2=None, t3="planet"),
            operator="sample",
            is_wrong=lambda scene: scene == "planet",
        )
        assert summary.false_certified_targets == (2,)
        assert summary.wrong_certified_targets == (1, 3)
        assert summary.wrong_certifications == 2

    def test_claim_scoring_off_by_default(self):
        """Without a claim test no certification is scored wrong."""
        summary = summarize(
            make_state((1, 3)),
            make_universe(t1="planet", t2=None, t3="planet"),
            operator="sample",
        )
        assert summary.wrong_certified_targets == ()

    def test_custom_presence_test(self):
        """`is_real` decides what counts as a real source."""
        summary = summarize(
            make_state((1, 2)),
            make_universe(t1="dust", t2="planet"),
            operator="sample",
            is_real=lambda scene: scene == "planet",
        )
        assert summary.false_certified_targets == (1,)
