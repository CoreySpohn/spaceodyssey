"""The first physical profile as data: conventions, pins, targets, columns, budgets.

A profile fixes every quantity, sign and time choice a physical study makes, so no
choice is implicit. `first-summary-v1` is one companion, one facility, one band
and scalar summary measurements. Its table ships with the package as JSON
(`first_summary_v1.json`) and is read here; the adapters take their target
elements, keepout limits and payload-column audiences from it.

`check_profile` lists what a profile is missing: every payload column and every
error-budget boundary needs a unit, a sign and a measure, and every payload
column an audience built from the core `Audience` flags. An empty list means
complete.
"""

import json
from importlib import resources

from spaceodyssey.core.enums import Audience

FIRST_SUMMARY_V1 = "first-summary-v1"
JATIS_EAC1_V1 = "jatis-eac1-v1"

_FILES = {
    FIRST_SUMMARY_V1: "first_summary_v1.json",
    JATIS_EAC1_V1: "jatis_eac1_v1.json",
}

# Fields every payload column and every error-budget boundary must state.
COLUMN_FIELDS = ("name", "unit", "sign", "measure", "audience")
BOUNDARY_FIELDS = ("name", "observable", "unit", "sign", "measure", "allocation")
# A boundary is either pinned or blocked on a named library item.
_STATUS_BLOCKED = "blocked: "

_AUDIENCES = {
    "observer": Audience.OBSERVER,
    "policy": Audience.POLICY,
    "scorer": Audience.SCORER,
}


def _named(xs):
    """Whether `xs` is a list of objects that each carry a ``name``."""
    return isinstance(xs, list) and all(isinstance(x, dict) and "name" in x for x in xs)


def _merge(base, over):
    """`base` with `over` laid on it: objects merge by key, named-entry lists by name.

    Any other value in `over` replaces the base value.
    """
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for key, value in over.items():
            out[key] = _merge(base[key], value) if key in base else value
        return out
    if _named(base) and _named(over):
        by_name = {x["name"]: i for i, x in enumerate(base)}
        out = list(base)
        for entry in over:
            if entry["name"] in by_name:
                out[by_name[entry["name"]]] = entry
            else:
                out.append(entry)
        return out
    return over


def load_profile(label=FIRST_SUMMARY_V1):
    """The profile table `label` as a plain JSON object.

    A table with an ``extends`` key names a base profile; its own entries are laid on
    the base (objects merge by key, lists of named entries by name) and the result no
    longer carries the key.
    """
    if label not in _FILES:
        raise KeyError(f"unknown profile {label!r}; known: {sorted(_FILES)}")
    text = resources.files(__package__).joinpath(_FILES[label]).read_text("utf-8")
    table = json.loads(text)
    base = table.pop("extends", None)
    return table if base is None else _merge(load_profile(base), table)


def audience_flags(names):
    """The core `Audience` flags for audience names such as ``["observer"]``."""
    if isinstance(names, str) or not names:
        raise ValueError("an audience is a nonempty list of names")
    flags = Audience(0)
    for name in names:
        if name not in _AUDIENCES:
            raise ValueError(f"unknown audience {name!r}; known: {sorted(_AUDIENCES)}")
        flags |= _AUDIENCES[name]
    return flags


def _blank(value):
    return not isinstance(value, str) or not value.strip()


def check_profile(profile):
    """Problems that make `profile` incomplete; an empty list means complete.

    Fails a payload column or a boundary that lacks any of its required fields
    or states one as an empty string, a repeated name, a payload column whose
    audience is not a nonempty list of known names, a boundary whose status is
    neither ``pinned`` nor ``blocked: <library item>``, and an empty column or
    boundary list.
    """
    problems = []
    for section, fields in (
        ("payload_columns", COLUMN_FIELDS),
        ("boundaries", BOUNDARY_FIELDS),
    ):
        entries = profile.get(section)
        if not isinstance(entries, list) or not entries:
            problems.append(f"{section}: missing or empty")
            continue
        names = [e.get("name") for e in entries if isinstance(e, dict)]
        if len(set(names)) != len(names):
            problems.append(f"{section}: repeated name")
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                problems.append(f"{section}[{i}]: not an object")
                continue
            label = entry.get("name", i)
            for field in fields:
                if field == "audience":
                    try:
                        audience_flags(entry.get(field))
                    except ValueError as err:
                        problems.append(f"{section}.{label}.audience: {err}")
                elif _blank(entry.get(field)):
                    problems.append(f"{section}.{label}.{field}: missing")
            if section == "boundaries":
                status = entry.get("status")
                blocked = isinstance(status, str) and status.startswith(_STATUS_BLOCKED)
                if status != "pinned" and not (
                    blocked and status[len(_STATUS_BLOCKED) :].strip()
                ):
                    problems.append(f"{section}.{label}.status: {status!r}")
    return problems


def payload_column_names(profile):
    """The payload column names of `profile`, in record order."""
    return tuple(c["name"] for c in profile["payload_columns"])
