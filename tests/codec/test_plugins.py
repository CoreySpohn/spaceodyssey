"""Kind resolution: load by id, version migration, and refused registrations."""

import dataclasses
import json
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.codec import DocumentError, KindResolver, dump_study, load_study
from spaceodyssey.core.kinds import AbstractDesign
from spaceodyssey.core.study import Clock
from spaceodyssey.plugins import RegistrationError, kinds_from_entry_points
from spaceodyssey.studies import toy
from tests.codec.test_document import TOY_KINDS, LabelledDesign, assert_same_pytree


@dataclasses.dataclass(frozen=True)
class Dist:
    name: str


@dataclasses.dataclass(frozen=True)
class Entry:
    """An installed entry point as `importlib.metadata` gives it."""

    group: str
    name: str
    obj: object
    dist: Dist = Dist("spaceodyssey")

    @property
    def value(self):
        return getattr(self.obj, "__qualname__", repr(self.obj))

    def load(self):
        if isinstance(self.obj, Exception):
            raise self.obj
        return self.obj


def entries_for(kinds):
    """One entry point per non-core kind of a mapping."""
    return [
        Entry(f"spaceodyssey.{group}", f"{kind_id}@{version}", cls)
        for (group, kind_id, version), cls in kinds.items()
    ]


class LogGain(AbstractDesign):
    """Version 2 of the toy gain design: the gain is stored as its natural log."""

    log_gain: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, jnp.float64))
    version: int = eqx.field(default=2, kw_only=True)

    @classmethod
    def migrate(cls, from_version, params):
        """Version 1 stored the gain itself."""
        if from_version != 1:
            raise ValueError(f"no migration from version {from_version}")
        return {"log_gain": jnp.log(params["gain"])}


class FrozenGain(AbstractDesign):
    """Version 2 of the toy gain design with no migration from version 1."""

    log_gain: jax.Array
    version: int = eqx.field(default=2, kw_only=True)


class MisnamedGain(LogGain):
    """A migration that returns the old field names."""

    @classmethod
    def migrate(cls, from_version, params):
        return dict(params)


def gain_v2(cls):
    """The toy kinds with `toy.gain` registered at version 2 as `cls`."""
    kinds = {k: v for k, v in TOY_KINDS.items() if k[:2] != ("design", "toy.gain")}
    return {**kinds, ("design", "toy.gain", 2): cls}


def v1_document(version=1):
    """survey_1 with a hand-written toy gain design stored at kind `version`."""
    doc = json.loads(dump_study(toy.survey_1(), TOY_KINDS).to_json())
    doc.pop("content_hash")
    doc["study"]["design"] = {
        "$kind": ["design", "toy.gain", version],
        "gain": {"$array": "float64", "shape": [], "data": 2.0},
    }
    return json.dumps(doc)


def test_load_by_id_without_path_import(tmp_path, monkeypatch):
    """Fails if entry points do not build a resolver that loads a study by id, if
    an entry outside the spaceodyssey groups is loaded, or if a document naming an
    importable module path as its kind id imports it instead of failing with D001.
    """
    foreign = Entry("console_scripts", "tool", ImportError("never loaded"))
    resolver = KindResolver(kinds_from_entry_points([*entries_for(TOY_KINDS), foreign]))
    study = toy.survey_1()
    text = dump_study(study, resolver).to_json()
    assert_same_pytree(study, load_study(text, resolver))

    (tmp_path / "t5c_probe_kind.py").write_text("raise SystemExit('imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    doc = json.loads(text)
    doc.pop("content_hash")
    doc["study"]["design"]["$kind"] = ["design", "t5c_probe_kind.Probe", 1]
    with pytest.raises(DocumentError) as err:
        load_study(json.dumps(doc), resolver)
    assert (err.value.pointer, err.value.code) == ("/study/design/$kind", "D001")
    assert "t5c_probe_kind" not in sys.modules


def test_older_kind_version_migrates_to_the_registered_one():
    """Fails if a hand-written version-1 document does not load through the class's
    `migrate` into the version-2 kind with the migrated value and version 2, or if
    the reloaded study is not written back at version 2.
    """
    kinds = gain_v2(LogGain)
    study = load_study(v1_document(), kinds)
    assert type(study.design) is LogGain and study.design.version == 2
    expected = jnp.log(jnp.asarray(2.0, jnp.float64))
    assert np.asarray(study.design.log_gain).tobytes() == np.asarray(expected).tobytes()
    doc = dump_study(study, kinds)
    assert doc.study["design"]["$kind"] == ["design", "toy.gain", 2]
    assert_same_pytree(study, load_study(doc.to_json(), kinds))


@pytest.mark.parametrize(
    ("cls", "version", "where", "code"),
    [
        (LogGain, 3, "/study/design/$kind", "D001"),
        (FrozenGain, 1, "/study/design/$kind", "D001"),
        (MisnamedGain, 1, "/study/design", None),
    ],
    ids=["newer-version", "no-migrate", "migrate-wrong-fields"],
)
def test_unreadable_kind_version_fails_with_a_pointer(cls, version, where, code):
    """Fails if a document at a kind version newer than the registered one, or an
    older one whose class has no `migrate`, loads or fails without D001 at the
    reference, or if a migration that returns the wrong fields is accepted.
    """
    with pytest.raises(DocumentError) as err:
        load_study(v1_document(version), gain_v2(cls))
    assert (err.value.pointer, err.value.code) == (where, code)


def _twice_from_two_distributions():
    return [
        Entry("spaceodyssey.design", "toy.gain@1", toy.ToyDesign, Dist("alpha")),
        Entry("spaceodyssey.design", "toy.gain@1", LabelledDesign, Dist("beta")),
    ]


@pytest.mark.parametrize(
    ("entries", "words"),
    [
        (_twice_from_two_distributions(), ("registered twice", "alpha", "beta")),
        ([Entry("spaceodyssey.widget", "toy.gain@1", toy.ToyDesign)], ("family",)),
        ([Entry("spaceodyssey.design", "toy.gain", toy.ToyDesign)], ("<id>@",)),
        ([Entry("spaceodyssey.policy", "toy.gain@1", toy.ToyDesign)], ("Policy",)),
        ([Entry("spaceodyssey.design", "toy.gain@2", toy.ToyDesign)], ("defaults",)),
        ([Entry("spaceodyssey.core", "study@1", Clock)], ("core kinds",)),
    ],
    ids=[
        "duplicate",
        "unknown-family",
        "no-version",
        "wrong-family",
        "version-mismatch",
        "core-rebound",
    ],
)
def test_bad_registration_is_refused(entries, words):
    """Fails if one id registered by two distributions is accepted or the error
    does not name both, or if an entry point with an unknown family, a name without
    a version, a class of another family, a version its class does not declare, or
    a core entry bound to another class is accepted.
    """
    with pytest.raises(RegistrationError) as err:
        kinds_from_entry_points(entries)
    for word in words:
        assert word in str(err.value)


@pytest.mark.parametrize(
    "extra",
    [
        {("design", "toy.gain", 2): LogGain},
        {("design", "toy.other_gain", 1): toy.ToyDesign},
    ],
    ids=["id-at-two-versions", "class-under-two-ids"],
)
def test_resolver_refuses_a_double_binding(extra):
    """Fails if an injected mapping may register one id at two versions, which
    would bypass migration, or one class under two ids, which makes its dumped
    reference ambiguous.
    """
    with pytest.raises(ValueError, match=r"registered at versions|bound to both"):
        KindResolver({**TOY_KINDS, **extra})
