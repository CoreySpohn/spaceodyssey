"""Astronomical analysis and observing campaigns.

Multi-visit observing programs with inference in the loop: an analysis session
fits existing observations, a campaign chooses and executes new visits against
a private simulated world, and an ensemble repeats a campaign over many worlds.
The transaction core is ``spaceodyssey.core``; the document form of a study is
``spaceodyssey.codec``; physical adapters are in ``spaceodyssey.adapters`` and
example studies in ``spaceodyssey.studies``. Import those explicitly, for
example ``from spaceodyssey import core``.
"""

from spaceodyssey._version import __version__

__all__ = ["__version__"]
