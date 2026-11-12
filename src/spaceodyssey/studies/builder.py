"""The study builder: `survey_1` and the refresh variants studies compose."""

import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import (
    ALL_EFFECTS,
    F64,
    I64,
    PUBLIC,
    CoincidentOrder,
    RecordStatus,
    RefitStatus,
)
from spaceodyssey.core.ledger import AssetTable, ChannelTable
from spaceodyssey.core.records import (
    KindRegistry,
    RecordKind,
)
from spaceodyssey.core.refresh import (
    REFRESH_PRODUCT_KIND,
    MaskedBatchRefit,
    RefreshSchedule,
)
from spaceodyssey.core.study import Agenda, Clock, Study
from spaceodyssey.studies.toy import (
    AUXILIARY_KIND,
    MEASUREMENT_KIND,
    RECORD_MEASUREMENT,
    FixedPlanPolicy,
    NullProcess,
    ToyDesign,
    ToyObjective,
    ToyPhotometry,
    ToyRefitMean,
    ToyRunningMean,
    ToyWorldSource,
    toy_registry,
)


def survey_1(caps=None, **overrides):
    """Build a fixed five-acquisition toy study."""
    caps = Capacities() if caps is None else caps
    if not caps.payload_audience:
        caps = toy_registry().lower(caps)
    c = caps.n_channels
    channels = ChannelTable(
        id=jnp.asarray([10, 11, 12, 13][:c], I64),
        lane=jnp.asarray([0, 1, 2, 2][:c], I64),
        occupancy_ticks=jnp.asarray([100, 150, 50, 40][:c], I64),
        latency_ticks=jnp.asarray([20, 20, 0, 0][:c], I64),
        kind=jnp.full(c, RECORD_MEASUREMENT, I64),
        audience=jnp.full(c, PUBLIC, I64),
        cost=jnp.ones((c, caps.n_resources), F64),
        charge=jnp.ones((c, caps.n_resources), F64),
        coincident_order=jnp.full(c, CoincidentOrder.FAULT_FIRST, I64),
    )
    budget = jnp.concatenate(
        [
            jnp.full((caps.n_assets, caps.n_resources), 10.0, F64),
            jnp.full((caps.n_programs + 1, caps.n_resources), 100.0, F64),
        ]
    )
    base = dict(
        caps=caps,
        channels=channels,
        channel_model=ToyPhotometry(noise=jnp.asarray([0.5, 0.3, 0.8, 0.4][:c], F64)),
        lane_asset=jnp.asarray([0, 0, 1][: caps.n_lanes], I64),
        transition_ticks=jnp.asarray([10, 10, 5][: caps.n_lanes], I64),
        budget=budget,
        horizon_tick=jnp.asarray(1000, I64),
        policies=(FixedPlanPolicy(plan=((0, 10), (1, 10), (0, 10), (1, 10), (0, 10))),),
        phase_until=(1000,),
        process=NullProcess(),
        observer=ToyRunningMean(),
        design=ToyDesign(gain=1.0),
    )
    base.update(overrides)
    base["assets"] = AssetTable(
        lane_asset=base.pop("lane_asset"),
        transition_ticks=base.pop("transition_ticks"),
        freeze_ticks=jnp.zeros(caps.n_assets, I64),
        disruptive_resource=jnp.full(caps.n_assets, -1, I64),
        config_switch_ticks=base.pop("config_switch_ticks", None),
    )
    policies = base.pop("policies")
    base["agenda"] = Agenda(
        policies,
        base.pop("phase_until"),
        base.pop("permitted_effects", (ALL_EFFECTS,) * len(policies)),
    )
    base["clock"] = Clock(1.0)
    base["world_source"] = ToyWorldSource()
    base["objective"] = ToyObjective(threshold=base.pop("detect_thr", 3.0))
    return Study(**base)


PRODUCT_CHANNELS = (
    90,
    91,
)  # observer and policy product channels of the refresh studies


PRODUCT_KIND = 3  # index of `REFRESH_PRODUCT_KIND` in `refresh_registry`


def refresh_registry():
    """The toy kinds and the derived refresh product kind, at indices 0 to 3."""
    return KindRegistry(
        (
            RecordKind("none", ()),
            MEASUREMENT_KIND,
            AUXILIARY_KIND,
            REFRESH_PRODUCT_KIND,
        )
    )


def survey_refresh(every_ticks=100, offset_ticks=0, *, observer=None, **overrides):
    """The five-acquisition toy study with a refresh schedule (a checkpoint grid).

    The observer defaults to `ToyRefitMean`, which changes only by refresh. The
    capacities carry the protocol (`rows_per_subject=8`, `refit_lanes=2`) and the
    registry lowered with the observer's refit kinds.
    """
    observer = ToyRefitMean() if observer is None else observer
    caps = overrides.pop("caps", None) or Capacities(rows_per_subject=8, refit_lanes=2)
    caps = refresh_registry().lower(caps, observer)
    schedule = RefreshSchedule(
        every_ticks, offset_ticks, *PRODUCT_CHANNELS, PRODUCT_KIND
    )
    return survey_1(caps, observer=observer, refresh=schedule, **overrides)


def toy_mean_kernel(study, entry):
    """The toy posterior mean in closed form: a masked kernel for `MaskedBatchRefit`.

    A flat-prior Gaussian posterior with known noise has the sample mean of the
    counting measurements as its mean, whatever the order they are summed in. Returns
    the subject slice `(mean, n)`, status `OK` and the row count as diagnostic.
    """
    observer, rows = study.observer, entry.rows
    take = (
        rows.valid
        & (rows.kind == observer.measurement_kind)
        & (rows.status == RecordStatus.COMPLETED)
    )
    n = jnp.sum(take).astype(I64)
    # Sum left to right, as `toy_refit_fit` does, so the two agree bit for bit.
    total, _ = jax.lax.scan(
        lambda t, x: (t + x, None),
        jnp.zeros((), F64),
        jnp.where(take, rows.payload[:, 0], 0.0),
    )
    mean = total / jnp.maximum(n, 1).astype(F64)
    diag = jnp.zeros(1, F64).at[0].set(n.astype(F64))
    return (mean, n), jnp.asarray(RefitStatus.OK, I64), diag


def toy_refit_fit(study, entry):
    """The same posterior as `toy_mean_kernel`, summed left to right on the host."""
    observer, rows = study.observer, entry.rows
    total, n = 0.0, 0
    for i in range(rows.valid.shape[0]):
        counts = (
            rows.valid[i]
            and rows.kind[i] == observer.measurement_kind
            and rows.status[i] == RecordStatus.COMPLETED
        )
        if counts:
            total += float(rows.payload[i, 0])
            n += 1
    mean = total / n if n else 0.0
    return (np.float64(mean), np.int64(n)), RefitStatus.OK, [float(n)]


def toy_masked_refit():
    """The toy refit as an array kernel: a `MaskedBatchRefit` of `toy_mean_kernel`."""
    return MaskedBatchRefit(toy_mean_kernel)
