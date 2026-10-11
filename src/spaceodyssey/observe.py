"""The observation seam: truth or belief in, predictive distribution out.

A measurement model owns the shared physics of an observation (count rates,
error models) and exposes it as a predictive distribution over the dataset
the observation would produce. What happens to that distribution is the
run's outcome operator: the sampling engine draws a realized dataset from it
against a drawn truth, the belief-space engine propagates its moments
against the current belief, and the closed-form engine integrates it against
the population prior. The predictive's ``sample`` returns the full dataset
object (for example :class:`spaceodyssey.data.SummaryDataset`), so the loop never
assembles data itself.

Reference implementations (the summary-level model over the exposure-time
calculator, the image-level model over the image simulator) arrive as
registry entries with their adapters.
"""

import abc
from typing import ClassVar

import equinox as eqx

from spaceodyssey.dist import AbstractPredictive


class AbstractMeasurement(eqx.Module):
    """One observing modality's measurement model."""

    PROTOCOL_VERSION: ClassVar[int] = 1

    @abc.abstractmethod
    def predictive(self, scene, action) -> AbstractPredictive:
        """Return the predictive distribution over the action's dataset.

        Args:
            scene: What the observation looks at: a truth scene on the
                sampling path, or a belief summary on the propagation path.
            action: The action being executed.
        """
