"""Compare the corpus across executors, policy forms, and hand histories."""

import dataclasses

import pytest

from spaceodyssey.core.executors.base import histories_equal
from tests.core.corpus import CORPUS, CORPUS_BY_NAME
from tests.core.helpers import check_corpus_case, history_of


def test_corpus_has_the_planned_cases():
    names = {c.name for c in CORPUS}
    planned = {
        "baseline", "wait", "delayed_release", "cancel_stale", "cancel_live", "bundle",
        "phase_switch", "coincident", "overflow_stall", "overflow_log",
        "coalesced_release", "bundle_unaffordable", "forbidden_effect",
        "catalog_record", "host_service", "revision",
    }  # fmt: skip
    stops = {
        f"stop_{s}_{d}"
        for s in ("lane", "asset", "program", "campaign")
        for d in ("close", "drain", "abort")
    }
    assert names == planned | stops and len(CORPUS) == len(names)


@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.name)
def test_corpus_three_way_parity(case):
    check_corpus_case(case)


def test_coincident_hand_history():
    h = history_of(CORPUS_BY_NAME["coincident"].make_study())
    assert h.events[4:6] == ((160, 0, 1), (160, 0, 1)) and h.lanes[4:6] == (1, 2)
    assert not h.double_settle


def test_parity_detects_difference():

    h = history_of(CORPUS_BY_NAME["wait"].make_study())
    g = dataclasses.replace(
        h, floats=dict(h.floats, payload=h.floats["payload"] + 1e-8)
    )
    assert histories_equal(h, g).max_abs > 1e-9 and not histories_equal(h, g).ok


def test_overflow_stall_stalls_after_one_extra_handled_row():
    h = history_of(CORPUS_BY_NAME["overflow_stall"].make_study())
    assert h.flags[2] and not h.flags[3] and h.n_records == 1


def test_overflow_log_has_four_records_and_flag():
    h = history_of(CORPUS_BY_NAME["overflow_log"].make_study(), n_steps=40)
    assert h.flags[3] and h.n_records == 4
