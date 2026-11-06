"""Survey 5, static checks: pre-flight causality (C001) of the catalog channel."""

import dataclasses

from spaceodyssey.core.declarations import Declaration, Need
from spaceodyssey.core.preflight import preflight
from spaceodyssey.studies import catalog
from spaceodyssey.studies.survey_5 import (
    CANDIDATE_ID_BASE,
    LAST_CANDIDATE_SLOT,
    ROSTER_COUNT_SLOT,
    FollowUpPolicy,
    survey_5,
)

# The catalog acquisition starts at tick 135 (the decision at tick 130 plus the lane's
# 5-tick transition), occupies its lane for 40 ticks and is released 150 ticks after.
CATALOG_START = 135
RELEASE = CATALOG_START + 40 + catalog.CATALOG_DELAY_TICKS


def with_need(decide_tick, study=None, start=CATALOG_START):
    """`study` whose policy declares a read of the catalog product at `decide_tick`."""
    study = survey_5() if study is None else study
    declared = Declaration(
        needs=(Need(catalog.CATALOG_CHANNEL_ID, start, decide_tick),)
    )

    class Reading(FollowUpPolicy):
        def declare(self):
            return declared

    agenda = dataclasses.replace(study.agenda, policies=(Reading(),))
    return dataclasses.replace(study, agenda=agenda)


def first_clean_tick(study):
    """The earliest decision tick pre-flight allows to read the catalog product."""
    low, high = 0, RELEASE
    assert not preflight(with_need(high, study))
    while low < high:
        mid = (low + high) // 2
        if preflight(with_need(mid, study)):
            low = mid + 1
        else:
            high = mid
    return low


def test_the_survey_is_clean_and_names_its_roster():
    """Fails if pre-flight objects to the survey, or a candidate identifier can be a
    truth id (a subject index)."""
    study = survey_5()
    assert not preflight(study, executor="compiled")
    assert not preflight(study, executor="host")
    assert CANDIDATE_ID_BASE >= study.caps.n_subjects
    assert ROSTER_COUNT_SLOT < study.caps.n_an
    assert LAST_CANDIDATE_SLOT < study.caps.n_an
    assert int(study.channels.audience[3]) == 3  # the catalog is public


def test_c001_flags_a_decision_declared_before_the_candidate_is_released():
    """Fails if a decision that reads the catalog product at a tick before the product
    can exist is not reported, or one made at its release tick is.
    """
    flagged = preflight(with_need(130))
    assert [str(f.code) for f in flagged] == ["C001"]
    assert "released no earlier than" in flagged[0].message
    assert not preflight(with_need(RELEASE))
    # pre-flight is conservative (a cancel can end the occupancy early), never late
    first = first_clean_tick(survey_5())
    assert 130 < first <= RELEASE
    assert preflight(with_need(first - 1)) and not preflight(with_need(first))


def test_c001_flags_a_catalog_that_is_never_released_to_the_policy():
    """Negative control: with the catalog observer-only no decision may read it."""
    study = survey_5()
    audience = study.channels.audience.at[3].set(1)
    hidden = dataclasses.replace(
        study, channels=dataclasses.replace(study.channels, audience=audience)
    )
    flagged = preflight(with_need(RELEASE + 500, hidden))
    assert [str(f.code) for f in flagged] == ["C001"]
    assert "never released to the policy" in flagged[0].message
