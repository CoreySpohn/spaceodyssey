"""Survey 24: a latent's trajectory is drawn with the world and shown only by a record.

The latent is the trajectory table of `survey_22`'s world source: it is drawn under its
own purpose key, so no other draw moves it and it moves none, and a version-2 channel
model samples it at the start tick of the record it realizes. The policy observes one
subject four times; the observer's belief is the running mean of the sampled values, so
the latent reaches the observer's view only through a released record.
"""

from tests.studies.scenarios.survey_22 import (
    N_PURPOSES,
    TARGET,
    TICKS,
    TrajectoryChannel,
    TrajectoryTruth,
    TrajectoryWorldSource,
    survey_22,
)

__all__ = [
    "N_PURPOSES",
    "TARGET",
    "TICKS",
    "TrajectoryChannel",
    "TrajectoryTruth",
    "TrajectoryWorldSource",
    "survey_24",
]


def survey_24():
    """The clean study of survey 22: four observations of the latent, no finder."""
    return survey_22(leaky=False)
