"""The observation seam: truth or belief in, predictive distribution out.

A measurement model owns the shared physics of an observation (count rates,
error models) and exposes it as a predictive distribution over the dataset
the observation would produce. The predictive's ``sample`` returns the full
dataset object (for example :class:`spaceodyssey.data.SummaryDataset`), so a
caller never assembles data itself.
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
