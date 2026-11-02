"""Construction guards and typed composition contracts."""

import dataclasses
import inspect
import subprocess
import sys
import typing
from collections.abc import Callable
from enum import IntEnum, IntFlag

import jax
import pytest

from spaceodyssey import core
from spaceodyssey.studies.toy import survey_1


@pytest.fixture(scope="module")
def composed_study():
    return survey_1()


def test_x64_guard(composed_study):
    study = composed_study
    with jax.enable_x64(False):
        with pytest.raises(RuntimeError, match="requires JAX x64"):
            core.Capacities()
        with pytest.raises(RuntimeError, match="requires JAX x64"):
            dataclasses.replace(study)
        for executor in (core.CompiledExecutor(), core.HostExecutor()):
            with pytest.raises(RuntimeError, match="requires JAX x64"):
                executor.run(study, None, None, 1)


def test_import_does_not_enable_x64():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import jax; jax.config.update("jax_enable_x64", False); '
            "import spaceodyssey.core; assert not jax.config.x64_enabled",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _is_untyped(annotation):
    if annotation in (object, typing.Any, Callable, typing.Callable):
        return True
    return any(_is_untyped(arg) for arg in typing.get_args(annotation))


def test_study_has_no_untyped_fields():
    bases = {
        cls
        for _, cls in inspect.getmembers(core, inspect.isclass)
        if cls.__name__.startswith("Abstract")
    }
    assert len(bases) == 9
    for cls in bases | {core.Study}:
        annotations = typing.get_type_hints(cls)
        if dataclasses.is_dataclass(cls):
            for field in dataclasses.fields(cls):
                assert field.name in annotations, (cls, field.name)
                assert not _is_untyped(annotations[field.name]), (cls, field.name)
        else:  # the host service is a plain host-side class, outside the study
            for name, annotation in annotations.items():
                assert not _is_untyped(annotation), (cls, name)
    assert {"key_hook", "handler_table", "n_extra_kinds"}.isdisjoint(
        field.name for field in dataclasses.fields(core.Study)
    )


def test_integer_vocabularies():
    for cls in (
        core.Stage,
        core.EventKind,
        core.EffectKind,
        core.Scope,
        core.Disposition,
        core.RecordStatus,
        core.Purpose,
        core.AdmissionCode,
    ):
        assert issubclass(cls, IntEnum)
        assert list(map(int, cls)) == list(range(len(cls)))
    assert issubclass(core.Audience, IntFlag)
    assert int(core.Audience.OBSERVER | core.Audience.POLICY) == 3


def test_collection_preserves_callers_x64_setting():
    code = (
        'import jax, pytest; jax.config.update("jax_enable_x64", False); '
        'result = pytest.main(["--collect-only", "-q", "-p", "no:cacheprovider", '
        '"tests/core/test_keys.py", "tests/studies/test_toy.py"]); '
        "assert result == 0; assert not jax.config.x64_enabled"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
