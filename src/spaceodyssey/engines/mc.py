"""The sampling engine: the event-driven loop against drawn truths.

This is the only engine that ever holds a universe, and the only place the
truth and model stacks meet is the observation call inside the loop.
"""

from spaceodyssey.build import Mission
from spaceodyssey.engines import register_engine
from spaceodyssey.loop import DETECTION_CERTIFICATE, run_mission
from spaceodyssey.metrics import YieldSummary, summarize
from spaceodyssey.rng import Purpose, stream


@register_engine("sample")
def run_sample(
    mission: Mission,
    *,
    seed: int = 0,
    is_real=bool,
    max_epochs: int = 10_000,
) -> YieldSummary:
    """Draw one universe, play the mission against it, score the outcome.

    Args:
        mission: The compiled mission; its truth stack must include a scene
            source.
        seed: Root seed; the universe draw and every in-loop draw derive
            from it by meaning, so paired runs stay aligned.
        is_real: Maps a target's truth scene to whether a real source is
            present, for false-certification accounting.
        max_epochs: Hard cap on loop steps.

    Returns:
        The run's YieldSummary.
    """
    universe = mission.truth.scene.draw(stream(seed, 0, 0, Purpose.UNIVERSE))
    state = run_mission(mission, universe, seed=seed, max_epochs=max_epochs)
    return summarize(
        state,
        universe,
        operator="sample",
        certificate=DETECTION_CERTIFICATE,
        is_real=is_real,
    )
