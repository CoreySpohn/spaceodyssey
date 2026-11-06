"""Astronomical analysis and observing campaigns.

Multi-visit observing programs with inference in the loop: an analysis session
fits existing observations, a campaign chooses and executes new visits against
a private simulated world, and an ensemble repeats a campaign over many worlds.
The transaction core is ``spaceodyssey.core``; the document form of a study is
``spaceodyssey.codec``.
"""

from spaceodyssey._version import __version__
from spaceodyssey.certify import (
    AbstractCertificate,
    AbstractProfileCertificate,
    CertificateState,
    SnrThreshold,
    ToyGaussianCertifier,
)
from spaceodyssey.data import Observation, SummaryDataset
from spaceodyssey.dist import AbstractPredictive, GaussianSummary
from spaceodyssey.errors import (
    OutOfRangeError,
    UnsupportedOperator,
    UnsupportedProfile,
)
from spaceodyssey.gaussian_profile import GaussianProfileCertifier
from spaceodyssey.metrics import CertifiedYield, YieldSummary, summarize
from spaceodyssey.observe import AbstractMeasurement
from spaceodyssey.plan import CertificationOutcome, PlanPredictive, PlanSummary
from spaceodyssey.profile import Claim, Edge, Profile
from spaceodyssey.reporting import certified_yield
from spaceodyssey.rng import Purpose, stream
from spaceodyssey.truth import (
    AbstractTruthModel,
    ObservedData,
    PriorSet,
    Truth,
    check_compatible,
)
from spaceodyssey.universe import AbstractUniverseSource, FixedUniverse, Universe

__all__ = [
    "AbstractCertificate",
    "AbstractMeasurement",
    "AbstractPredictive",
    "AbstractProfileCertificate",
    "AbstractTruthModel",
    "AbstractUniverseSource",
    "CertificateState",
    "CertificationOutcome",
    "CertifiedYield",
    "Claim",
    "Edge",
    "FixedUniverse",
    "GaussianProfileCertifier",
    "GaussianSummary",
    "Observation",
    "ObservedData",
    "OutOfRangeError",
    "PlanPredictive",
    "PlanSummary",
    "PriorSet",
    "Profile",
    "Purpose",
    "SnrThreshold",
    "SummaryDataset",
    "ToyGaussianCertifier",
    "Truth",
    "Universe",
    "UnsupportedOperator",
    "UnsupportedProfile",
    "YieldSummary",
    "__version__",
    "certified_yield",
    "check_compatible",
    "stream",
    "summarize",
]
