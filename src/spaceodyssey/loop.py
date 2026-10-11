"""The event-driven mission loop.

The loop is plain Python by design: it has data-dependent control flow,
variable-length logs, and heterogeneous per-target work, so the hot paths
live inside the seam implementations (which are free to JIT) rather than in
a compiled whole-mission graph. Ensemble parallelism lives at the process
level.

The certificate maintained by the loop is stored under the name
``"detection"``; richer certificate vocabularies arrive with the
specification's certificate definitions.
"""

from dataclasses import dataclass, field

from planit_py.vocabulary import Action, FixedDuration

from spaceodyssey.account import Ledger
from spaceodyssey.belief import MissionBelief
from spaceodyssey.build import Mission, TruthStack
from spaceodyssey.data import Observation
from spaceodyssey.rng import Purpose, stream
from spaceodyssey.universe import Universe

DETECTION_CERTIFICATE = "detection"


@dataclass
class MissionState:
    """The loop-level running state of one simulated mission.

    Attributes:
        time_d: Mission clock in days.
        ledger: Named-resource accounting.
        belief: The agent-visible belief.
        log: Every observation record, in execution order.
        epoch_index: Number of completed loop steps.
    """

    time_d: float
    ledger: Ledger
    belief: MissionBelief
    log: list = field(default_factory=list)
    epoch_index: int = 0


def observe(truth: TruthStack, universe: Universe, action: Action, key) -> Observation:
    """Execute one action against the truth.

    This is the only function typed to accept the truth side of the
    firewall; everything the agent ever learns passes through its return
    value.

    Args:
        truth: The compiled truth-side stack.
        universe: The drawn ground truth.
        action: The action to execute.
        key: PRNG key for the measurement draw.

    Returns:
        The observation record, carrying the realized duration the ledger
        will charge.

    Raises:
        NotImplementedError: For adaptive integration policies; executing
            those requires certificate evaluation inside the integration,
            which arrives with the adaptive-stopping work.
    """
    scene = universe.systems.get(action.target_id)
    dataset = truth.observation.predictive(scene, action).sample(key)
    integration = action.integration
    if isinstance(integration, FixedDuration):
        duration_d = integration.duration_d
    else:
        raise NotImplementedError(
            f"integration policy {type(integration).__name__} requires "
            "certificate evaluation inside the integration, which is not "
            "implemented yet; use FixedDuration"
        )
    return Observation(
        action=action,
        epoch_d=action.start_time_d,
        duration_d=duration_d,
        dataset=dataset,
    )


def step(
    mission: Mission, universe: Universe, state: MissionState, root
) -> MissionState:
    """Advance the mission by one propose-observe-update-certify cycle.

    Args:
        mission: The compiled mission.
        universe: The drawn ground truth (never shown to the model side).
        state: The current loop state.
        root: The run's root seed; per-draw keys derive from it by meaning.

    Returns:
        The next loop state.

    Raises:
        RuntimeError: If the executed action charges no time; the loop would
            otherwise never advance.
    """
    model = mission.model
    ctx = model.context.context(None, state.time_d)
    action = model.policy.propose(state.belief, ctx, model.cost)
    key = stream(root, action.target_id, state.epoch_index, Purpose.MEASUREMENT)
    obs = observe(mission.truth, universe, action, key)
    belief = state.belief.record(obs)
    certificate = model.certification
    if certificate is not None:
        previous = belief.certificate(action.target_id, DETECTION_CERTIFICATE)
        if previous is None:
            previous = certificate.initial()
        belief = belief.with_certificate(
            action.target_id,
            DETECTION_CERTIFICATE,
            certificate.update(previous, obs.dataset),
        )
    costs = model.cost.realized(action, ctx, obs.duration_d)
    time_cost = float(costs.get("time_d", 0.0))
    if time_cost <= 0.0:
        raise RuntimeError(
            f"action on target {action.target_id} charged no time; "
            "the mission clock would not advance"
        )
    return MissionState(
        time_d=state.time_d + time_cost,
        ledger=state.ledger.charge(costs),
        belief=belief,
        log=[*state.log, obs],
        epoch_index=state.epoch_index + 1,
    )


def run_mission(
    mission: Mission,
    universe: Universe,
    *,
    seed,
    max_epochs: int = 10_000,
) -> MissionState:
    """Run the loop until the mission duration or a budget is exhausted.

    Args:
        mission: The compiled mission.
        universe: The drawn ground truth.
        seed: The run's root seed (integer or PRNG key).
        max_epochs: Hard cap on loop steps, as a runaway guard.

    Returns:
        The final loop state.
    """
    state = MissionState(
        time_d=0.0,
        ledger=Ledger.from_budgets(mission.spec.budgets),
        belief=MissionBelief(),
    )
    while (
        state.time_d < mission.spec.duration_d
        and not state.ledger.exhausted
        and state.epoch_index < max_epochs
    ):
        state = step(mission, universe, state, seed)
    return state
