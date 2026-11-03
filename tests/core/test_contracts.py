"""Construction guards and typed composition contracts."""

import dataclasses
import subprocess
import sys
import typing
from collections.abc import Callable
from enum import IntEnum, IntFlag

import jax
import pytest

from spaceodyssey import core
from spaceodyssey.studies.toy import survey_1
from tests.core.contract_helpers import (
    assert_study_typed,
    assert_typed_fields,
    is_untyped,
)


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


def test_study_has_no_untyped_fields():
    assert_study_typed()


@pytest.mark.parametrize(
    "annotation",
    [
        object,
        typing.Any,
        Callable,
        typing.Callable,
        list[typing.Any],
        Callable[[typing.Any], int],
    ],
)
def test_untyped_fields_are_rejected(annotation):
    assert is_untyped(annotation)
    cls = dataclasses.make_dataclass("AbstractUntyped", [("value", annotation)])
    with pytest.raises(AssertionError, match="untyped field"):
        assert_typed_fields({cls})


def test_plain_abstract_fields_need_annotations():
    class AbstractMissing:
        value = 0

    with pytest.raises(AssertionError, match="missing annotation"):
        assert_typed_fields({AbstractMissing})


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
