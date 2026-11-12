"""Builders for the run document tests (not a test file).

One run: the oracle study on a refresh grid, whose policy asks the toy service once and
whose observer changes only by refresh. Its registry is the toy refresh registry with
the answer kind added, so the run needs a service, a refit boundary and the registry
all at once.
"""

import functools

from spaceodyssey.codec import BindingKind, Bindings
from spaceodyssey.core.refresh import (
    HostLoopRefit,
    MaskedBatchRefit,
    observer_refit_kernel,
)
from spaceodyssey.studies import builder, oracle, toy
from tests.codec.test_document import TOY_KINDS
from tests.core.refresh_helpers import oracle_refresh_study

KINDS = {**TOY_KINDS, ("observer", "toy.refit_mean", 1): toy.ToyRefitMean}
MODEL_HASH = 7


def _service_kind():
    return BindingKind(
        "toy.oracle",
        1,
        cls=oracle.ToyOracleService,
        describe=lambda obj, registry: {"record_kind": registry[obj.kind].name},
        build=lambda config, registry: oracle.ToyOracleService(
            registry.index(config["record_kind"])
        ),
    )


def _masked_kind():
    return BindingKind(
        "toy.mean_kernel",
        1,
        matches=lambda obj: (
            type(obj) is MaskedBatchRefit and obj.kernel is builder.toy_mean_kernel
        ),
        describe=lambda obj, registry: {"n_diag": obj.n_diag},
        build=lambda config, registry: MaskedBatchRefit(
            builder.toy_mean_kernel, n_diag=config["n_diag"], model_hash=MODEL_HASH
        ),
    )


def _observer_kernel_kind():
    """The boundary the refresh corpus case uses: the observer's own refit kernel."""
    return BindingKind(
        "toy.observer_refit",
        1,
        matches=lambda obj: (
            type(obj) is MaskedBatchRefit and obj.kernel is observer_refit_kernel
        ),
        describe=lambda obj, registry: {"n_diag": obj.n_diag},
        build=lambda config, registry: MaskedBatchRefit(n_diag=config["n_diag"]),
    )


def _host_loop_kind():
    return BindingKind(
        "toy.refit_fit",
        1,
        matches=lambda obj: getattr(obj, "fit", None) is builder.toy_refit_fit,
        describe=lambda obj, registry: {"n_diag": obj.n_diag},
        build=lambda config, registry: HostLoopRefit(
            builder.toy_refit_fit, n_diag=config["n_diag"], model_hash=MODEL_HASH
        ),
    )


def bindings():
    """The service and boundary kinds the run uses."""
    return Bindings(
        (_service_kind(), _masked_kind(), _host_loop_kind(), _observer_kernel_kind())
    )


def boundary():
    """The masked refit boundary the run is bound to."""
    return MaskedBatchRefit(builder.toy_mean_kernel, model_hash=MODEL_HASH)


@functools.cache
def refresh_run():
    """The study, its registry and its service: the run the documents describe."""
    study, service = oracle_refresh_study()
    registry = builder.refresh_registry()
    registry.register(oracle.ANSWER_KIND)
    return study, registry, service


def host_loop():
    """The host-loop refit boundary, answering the same model as `boundary()`."""
    return HostLoopRefit(builder.toy_refit_fit, model_hash=MODEL_HASH)
