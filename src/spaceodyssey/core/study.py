"""Study."""

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.capacities import Capacities, require_x64
from spaceodyssey.core.enums import F64, I64, LEVEL_EFFECTS
from spaceodyssey.core.keys import DEFAULT_COUPLING, Coupling, Fork
from spaceodyssey.core.kinds import (
    AbstractChannelModel,
    AbstractDesign,
    AbstractObjective,
    AbstractObserver,
    AbstractPolicy,
    AbstractProcess,
    AbstractWorldSource,
)
from spaceodyssey.core.ledger import AssetTable, ChannelTable
from spaceodyssey.core.records import lower_refit
from spaceodyssey.core.refresh import RefreshSchedule, refresh_enabled
from spaceodyssey.core.snapshot import named_leaves


def _as_horizon_tick(value):
    """Check precision before converting the study horizon to integer ticks."""
    require_x64()
    return jnp.asarray(value, I64)


class Clock(eqx.Module):
    """Duration in seconds of one integer tick."""

    tick_length_s: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, F64))


class Agenda(eqx.Module):
    """Policies, the phase boundaries that select them, and what each phase may ask.

    `permitted_effects` holds one bitmask over `EffectKind` per phase (see
    `effect_mask`), and `levels` one level number (0 to 3) per phase, whose mask is
    `LEVEL_EFFECTS[level]`. Admission refuses an effect whose bit is clear in the
    intersection of the two (`effective_effects`). Levels default to the top level,
    which permits every effect, so the phase masks alone decide.
    """

    policies: tuple[AbstractPolicy, ...]
    phase_until: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, I64))
    permitted_effects: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, I64))
    levels: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, I64))

    def __init__(self, policies, phase_until, permitted_effects, levels=None):
        """Store the fields; `levels=None` puts every phase at the top level."""
        self.policies = policies
        self.phase_until = phase_until
        self.permitted_effects = permitted_effects
        top = len(LEVEL_EFFECTS) - 1
        self.levels = (top,) * len(policies) if levels is None else levels

    def __check_init__(self):
        """Require one mask and one known level per phase."""
        if len(self.permitted_effects) != len(self.policies):
            raise ValueError("permitted_effects needs one mask per policy phase")
        if self.levels.shape != (len(self.policies),):
            raise ValueError("levels needs one level per policy phase")
        if bool(jnp.any((self.levels < 0) | (self.levels >= len(LEVEL_EFFECTS)))):
            raise ValueError(f"levels must lie in 0 to {len(LEVEL_EFFECTS) - 1}")

    def effective_effects(self):
        """Per-phase mask admission applies: the phase mask and the level mask."""
        return self.permitted_effects & jnp.asarray(LEVEL_EFFECTS, I64)[self.levels]


class Study(eqx.Module):
    """A composed transaction study with typed kind contracts."""

    caps: Capacities = eqx.field(static=True)
    clock: Clock
    channels: ChannelTable
    channel_model: AbstractChannelModel
    assets: AssetTable
    budget: jax.Array
    horizon_tick: jax.Array = eqx.field(converter=_as_horizon_tick)
    agenda: Agenda
    process: AbstractProcess
    observer: AbstractObserver
    design: AbstractDesign
    world_source: AbstractWorldSource
    objective: AbstractObjective
    coupling: Coupling = eqx.field(default_factory=lambda: DEFAULT_COUPLING)
    fork: Fork = eqx.field(default_factory=Fork)
    refresh: RefreshSchedule = eqx.field(default_factory=RefreshSchedule)

    def __check_init__(self):
        """Validate precision, kinds, the observer's layout, refit and refresh.

        Payload column audiences come from the registered record kinds. Capacities with
        no registered kinds would leave every column readable by every audience, so a
        study refuses them.
        """
        require_x64()
        if not self.caps.payload_audience:
            raise ValueError(
                "Study capacities have no registered record kinds; lower a registry "
                "into the capacities so payload columns carry their audiences"
            )
        _check_layout(self.observer, self.caps)
        _check_refit(self.observer, self.caps, self.refresh)
        _check_refresh(self.observer, self.caps, self.refresh)


def _check_refit(observer, caps, refresh):
    """Require the observer's refit declaration to be lowered into the capacities.

    A refit kind must name a registered kind that is not derived (`lower_refit`), and
    the capacities must hold the kinds and statuses the observer declares, so a
    registry lowered without this observer is refused. An observer that is not
    incremental changes only by refresh, so it needs a schedule that checkpoints.
    """
    declared = observer.capabilities()
    if not declared.incremental and not (
        refresh_enabled(caps) and int(refresh.every_ticks) > 0
    ):
        raise ValueError(
            "the observer is not incremental; it changes only by refresh, so the "
            "study needs a refresh schedule (every_ticks > 0) and "
            "Capacities.rows_per_subject > 0"
        )
    lowered = lower_refit(declared, caps.kind_names, caps.derived_kinds)
    if lowered != (caps.refit_kinds, caps.refit_statuses):
        raise ValueError(
            "the capacities do not hold the observer's refit kinds and statuses; "
            "lower the registry with the observer (KindRegistry.lower(caps, observer))"
        )


def _check_refresh(observer, caps, refresh):
    """Require a consistent refresh protocol: capacities, observer, schedule, kinds.

    A schedule that checkpoints needs `Capacities.rows_per_subject`; capacities that
    carry the protocol need a refit-capable observer that declares `subject_leaves`
    (leaves of the full belief whose leading axis is the subject), and a schedule whose
    product kind is a registered derived kind with the two product columns and whose
    two product channels differ.
    """
    every, offset = int(refresh.every_ticks), int(refresh.offset_ticks)
    if every < 0 or offset < 0:
        raise ValueError("a refresh schedule has nonnegative ticks")
    if not refresh_enabled(caps):
        if every > 0:
            raise ValueError("a refresh schedule needs Capacities.rows_per_subject > 0")
        return
    if not observer.capabilities().refit:
        raise ValueError("refresh needs a refit-capable observer")
    names = observer.subject_leaves
    if not names:
        raise ValueError("a refit-capable observer declares its subject_leaves")
    leaves = named_leaves(observer.layout(caps).belief)
    for name in names:
        leaf = leaves.get(name)
        if leaf is None or not leaf.shape or leaf.shape[0] != caps.n_subjects:
            raise ValueError(
                f"subject leaf {name!r} is not a belief leaf indexed by subject"
            )
    kind = int(refresh.kind)
    if kind not in caps.derived_kinds:
        raise ValueError("RefreshSchedule.kind must be a registered derived kind")
    if caps.payload_audience and sum(1 for a in caps.payload_audience[kind] if a) < 2:
        raise ValueError("the refresh product kind needs the status and n_rows columns")
    if int(refresh.observer_channel) == int(refresh.policy_channel):
        raise ValueError("the two product channels must differ")


def _shapes(tree):
    """Tree structure (node types and static metadata) and leaf shapes and dtypes."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return treedef, [(leaf.shape, leaf.dtype) for leaf in leaves]


def _check_layout(observer, caps):
    """Require the observer's layout to describe its beliefs and projections.

    The declared shapes must equal those `init` and `init_summary` build, node types
    and static fields included. Both projections must return `(F64[n_subjects],
    F64[n_an])`, the shapes of the trace fields they fill.
    """
    layout = observer.layout(caps)
    belief = jax.eval_shape(lambda: observer.init(caps))
    summary = jax.eval_shape(lambda: observer.init_summary(caps))
    for which, declared, made in (
        ("belief", layout.belief, belief),
        ("summary", layout.summary, summary),
    ):
        if _shapes(declared) != _shapes(made):
            raise ValueError(
                f"the observer's layout does not describe the {which} it initializes"
            )
    want = [((caps.n_subjects,), jnp.dtype(F64)), ((caps.n_an,), jnp.dtype(F64))]
    for which, project, shapes in (
        ("project", layout.project, belief),
        ("summary projection", layout.summary_projection, summary),
    ):
        out = jax.eval_shape(project, shapes)
        got = [(x.shape, x.dtype) for x in jax.tree_util.tree_leaves(out)]
        pair = jax.tree_util.tree_structure((0, 0))
        if jax.tree_util.tree_structure(out) != pair or got != want:
            raise ValueError(
                f"the layout's {which} must return (F64[n_subjects], F64[n_an]), "
                f"not {got}"
            )
