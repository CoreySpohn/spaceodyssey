"""Capacity exhaustion studies shared by status tests and the gate."""

from spaceodyssey.core import PUBLIC, Capacities, Capacity
from spaceodyssey.studies import toy as kinds
from tests.core.helpers import worlds


def run_status_case(executor, study, caps, n_steps=40, n=2):
    truths, roots = worlds(n, caps)
    return executor().run_worlds(study, truths, roots, n_steps)


OVERFLOW_CASES = {
    "stable_ids": (Capacities(max_sid=6), Capacity.STABLE_IDS),
    "log_rows": (Capacities(n_log=4), Capacity.LOG_ROWS),
    "queue_slots": (Capacities(n_events=3), Capacity.QUEUE_SLOTS),
}


def overflow_study(name, caps):
    # four heartbeat rows plus the kickoff wake need five queue slots
    if name == "queue_slots":
        process = kinds.HeartbeatProcess(
            period_ticks=10, n_fires=4, audience=PUBLIC, writes=False
        )
        return kinds.survey_1(caps=caps, process=process)
    return kinds.survey_1(caps=caps)
