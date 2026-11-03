"""Every executor output field must have an explicit parity disposition."""

import dataclasses

import numpy as np
import pytest

from spaceodyssey.core import CompiledExecutor, HostExecutor, RunResult
from spaceodyssey.core.carry import Carry, Trace
from spaceodyssey.core.executors import base
from spaceodyssey.core.records import Record, RecordLog
from spaceodyssey.studies import toy
from tests.core.helpers import worlds


def _field_paths(cls, prefix):
    for field in dataclasses.fields(cls):
        path = f"{prefix}.{field.name}"
        if dataclasses.is_dataclass(field.type):
            yield from _field_paths(field.type, path)
        else:
            yield path


RESULT_FIELDS = (*_field_paths(Carry, "carry"), *_field_paths(Trace, "trace"))


@pytest.fixture(scope="module")
def executor_pair():
    study = toy.survey_1()
    truths, roots = worlds(1)
    batch = CompiledExecutor().run_worlds(study, truths, roots, 4)
    compiled = RunResult(
        base.index_world(batch.carry, 0), base.index_world(batch.trace, 0)
    )
    host = HostExecutor().run(study, base.index_world(truths, 0), roots[0], 4)
    assert int(compiled.carry.log.valid.sum()) > 0
    assert bool(compiled.carry.ev.valid.any()), "exercise pending event fields"
    assert base.histories_equal(host.history(), compiled.history()).go
    return host, compiled


def test_every_output_field_has_an_explicit_disposition():
    compared = {
        f"{group}.{name}"
        for group, names in getattr(base, "CANONICAL_FIELDS", {}).items()
        for name in names
    }
    excluded = getattr(base, "CANONICAL_EXCLUSIONS", {})
    assert not compared.intersection(excluded)
    assert set(RESULT_FIELDS) == compared.union(excluded)
    assert all(
        isinstance(reason, str) and reason.strip() for reason in excluded.values()
    )
    # A newly added record column must reach the log, then canonical comparison.
    assert {f.name for f in dataclasses.fields(Record)} == {
        f.name for f in dataclasses.fields(RecordLog)
    }


def _perturb(tree, parts):
    name, *rest = parts
    value = getattr(tree, name)
    if rest:
        changed = _perturb(value, rest)
    else:
        changed = np.array(value, copy=True)
        assert changed.size, f"unexercised field: {name}"
        # Change one element, preserving shape and dtype. Select an active row.
        row = 0
        for mask in ("handled", "valid"):
            if hasattr(tree, mask) and np.ndim(getattr(tree, mask)):
                row = int(np.flatnonzero(np.asarray(getattr(tree, mask)))[0])
                break
        index = () if changed.ndim == 0 else (row,) + (0,) * (changed.ndim - 1)
        if changed.dtype.kind == "b":
            changed[index] = ~changed[index]
        else:
            changed[index] += 0.25 if changed.dtype.kind == "f" else 23
    return dataclasses.replace(tree, **{name: changed})


@pytest.mark.parametrize("field", RESULT_FIELDS)
def test_compiled_field_mutation_fails_parity(executor_pair, field):
    """Any single compared field can reject a compiled-only corruption."""
    host, compiled = executor_pair
    if field in getattr(base, "CANONICAL_EXCLUSIONS", {}):
        pytest.skip(base.CANONICAL_EXCLUSIONS[field])
    mutant = _perturb(compiled, field.split("."))
    assert not base.histories_equal(host.history(), mutant.history()).ok, field


@pytest.mark.parametrize(
    "field", ["lanes", "n_handled", "n_records", "status", "overflow_mask"]
)
def test_history_metadata_is_compared(executor_pair, field):
    host, compiled = executor_pair
    original = compiled.history()
    value = getattr(original, field)
    changed = (*value[:-1], value[-1] + 1) if isinstance(value, tuple) else value + 1
    mutant = dataclasses.replace(original, **{field: changed})
    assert not base.histories_equal(host.history(), mutant).ok


def test_inactive_storage_and_trace_padding_are_ignored(executor_pair):
    """Only occupancy masks, valid rows, and handled trace rows define history."""
    host, compiled = executor_pair
    carry = compiled.carry
    for group in ("ev", "log"):
        table = getattr(carry, group)
        invalid = ~np.asarray(table.valid)
        assert invalid.any()
        updates = {}
        for field in dataclasses.fields(table):
            if field.name != "valid":
                array = np.array(getattr(table, field.name), copy=True)
                array[invalid] += 31
                updates[field.name] = array
        carry = dataclasses.replace(
            carry, **{group: dataclasses.replace(table, **updates)}
        )
    trace = dataclasses.replace(
        compiled.trace,
        **{
            field.name: np.concatenate(
                [
                    np.asarray(getattr(compiled.trace, field.name)),
                    np.full_like(
                        np.asarray(getattr(compiled.trace, field.name))[:2],
                        False if field.name == "handled" else 31,
                    ),
                ]
            )
            for field in dataclasses.fields(compiled.trace)
        },
    )
    assert base.histories_equal(host.history(), RunResult(carry, trace).history()).go
