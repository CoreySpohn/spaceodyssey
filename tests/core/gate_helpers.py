"""Static study-dispatch scan used by the close-out gate."""

import ast
import builtins
import re
from pathlib import Path
from typing import NamedTuple

SURVEY_NAME = re.compile(r"\bsurvey[ _-]*\d+\b", re.IGNORECASE)
KIND_NAMES = {
    "Study",
    "AbstractHostService",
    "AbstractChannelModel",
    "AbstractDesign",
    "AbstractObjective",
    "AbstractObserver",
    "AbstractPolicy",
    "AbstractProcess",
    "AbstractWorldSource",
}


def _symbol(node):
    return node.id if isinstance(node, ast.Name) else getattr(node, "attr", None)


def _study_kind_names():
    """Find concrete kind descendants without importing domain implementations."""
    source = Path(__file__).parents[2] / "src/spaceodyssey"
    classes = [
        node
        for directory in (source / "core", source / "studies")
        for path in directory.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef)
    ]
    names = set(KIND_NAMES)
    while True:
        extended = names | {
            node.name
            for node in classes
            if any(_symbol(base) in names for base in node.bases)
        }
        if extended == names:
            return names
        names = extended


def _study_reference(node, study_names):
    return any(
        (isinstance(part, ast.Name) and part.id in study_names)
        or (isinstance(part, ast.Attribute) and part.attr == "study")
        for part in ast.walk(node)
    )


def _study_aliases(tree):
    """Follow simple aliases of the study and its components conservatively."""
    names = {"study"}
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
        and node.value is not None
    ]
    while True:
        extended = set(names)
        for node in assignments:
            if _study_reference(node.value, names):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                extended.update(
                    part.id
                    for target in targets
                    for part in ast.walk(target)
                    if isinstance(part, ast.Name)
                )
        if extended == names:
            return names
        names = extended


CORE_DIR = Path(__file__).parents[2] / "src/spaceodyssey/core"
ENUM_BASES = {"Enum", "IntEnum", "IntFlag", "Flag", "StrEnum"}
INTROSPECTION = {"__name__", "__qualname__", "__class__", "__module__"}
SURVEY_ATTRIBUTES = {"survey_id", "survey", "label", "study_id"}
# a survey word that a later piece completes: "survey_" + str(n), "survey_%d"
SURVEY_FRAGMENT = re.compile(
    r"\bsurvey[ _-]*$|\bsurvey[ _-]*(%\w|\{[^}]*\})", re.IGNORECASE
)
TEXT_OF = {"repr", "str", "format", "ascii"}
KIND_ATTRIBUTES = {"kind", "name", "type", "study", "channel_model", *INTROSPECTION}
LOOKUPS = {"get", "pop", "setdefault"}
CAMEL = re.compile(r"^[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*$")


class Allowance(NamedTuple):
    """One reviewed exception, scoped to the rules it waives and where it sits."""

    reason: str
    rules: tuple[str, ...]
    function: str
    count: int


# Reviewed structural exceptions: (path under the core, stripped source line) ->
# Allowance. An entry waives only the listed rules, only inside the named enclosing
# function, and only for `count` lines; an identical line elsewhere is still reported.
# Each entry must still match exactly, so a moved or duplicated line is noticed.
ENUM_LABEL = "name of a core Purpose enum member, used as a report label"
COLUMN_BY_NAME = (
    "reads one named column or field of a core pytree (carry, log, trace, table, view "
    "or belief); the name comes from the pytree schema, never from a study or kind"
)
FIELD_WALK = (
    "iterates the declared fields of a core pytree, a structural walk over the "
    "transaction schema that never names a study or kind"
)
SCHEMA_TYPE = (
    "type check against a core-owned record-schema class (column descriptor or "
    "record-kind registry type), not a study, survey or channel kind"
)

ALLOWED = {
    ("battery/fork.py", "out[Purpose(purpose).name] = ("): Allowance(
        ENUM_LABEL, ("study.name",), "key_report", 1
    ),
    ("battery/fork.py", "f.name: jnp.asarray(getattr(log, f.name)[i])"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "_log_record", 1
    ),
    ("battery/fork.py", "np.asarray(getattr(trunk, name))[:first],"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "_trace_prefix", 1
    ),
    ("battery/fork.py", "np.asarray(getattr(side, name))[:first],"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "_trace_prefix", 1
    ),
    (
        "battery/fork.py",
        "getattr(trunk.log, name)[i], getattr(side.log, name)[j]",
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "_records", 1),
    (
        "battery/metamorphic.py",
        "a, b = np.asarray(getattr(tb, name)), np.asarray(getattr(tl, name))",
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "_epoch_world", 1),
    (
        "battery/metamorphic.py",
        "a, b = np.asarray(getattr(cb.log, name)), np.asarray(getattr(cl.log, name))",
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "_epoch_world", 1),
    (
        "battery/metamorphic.py",
        "if not tree_bits_equal(getattr(cb, name), getattr(cl, name)):",
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "_epoch_world", 1),
    (
        "battery/metamorphic.py",
        "if not tree_bits_equal(getattr(ca, name), getattr(cb, name)):",
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "zero_time_asset", 1),
    ("battery/replay_latency.py", "n: np.asarray(getattr(trace, n))[keep]"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "_non_release", 1
    ),
    (
        "battery/replay_latency.py",
        "np.asarray(getattr(tb, name))[wb], np.asarray(getattr(tl, name))[wl]",
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "_compare_world", 1),
    ("battery/replay_latency.py", "np.asarray(getattr(cb.log, name)[i]),"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "_compare_world", 1
    ),
    ("battery/replay_latency.py", "np.asarray(getattr(cl.log, name)[j]),"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "_compare_world", 1
    ),
    (
        "battery/replay_latency.py",
        "if not bits_equal(getattr(cb.ledger, name), getattr(cl.ledger, name)):",
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "_compare_world", 1),
    (
        "battery/replay_latency.py",
        "if not bits_equal(getattr(cb.belief, name), getattr(cl.belief, name)):",
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "_compare_world", 1),
    ("battery/result.py", "int(getattr(log, n)[i])"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "log_index", 1
    ),
    (
        "battery/truth_swap.py",
        "a, b = np.asarray(getattr(trace_a, name)), np.asarray(getattr(trace_b, name))",
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "_prefix_equal", 1),
    (
        "battery/views.py",
        "hit &= np.asarray(getattr(log, name)) == getattr(view.log, name)[i]",
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "_payload_matches_log", 1),
    ("battery/views.py", "x = np.asarray(getattr(va, name))"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "compare_views", 1
    ),
    ("battery/views.py", "if not bits_equal(x, getattr(vb, name)):"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "compare_views", 1
    ),
    ("battery/views.py", "x = np.asarray(getattr(va.log, name))"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "compare_views", 1
    ),
    ("battery/views.py", "if not bits_equal(x, getattr(vb.log, name)):"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "compare_views", 1
    ),
    ("carry.py", "o = getattr(o, part)"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "_tree_set.getter", 1
    ),
    (
        "conformance/corpus.py",
        "*(np.asarray(getattr(trace, name))[keep].tolist() for name in names),",
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "public_events", 1),
    ("conformance/invariants.py", "np.asarray(getattr(t, n))[handled]"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "order_keys_monotone", 1
    ),
    (
        "events.py",
        "getattr(table, n).at[i].set(jnp.where(write, v, getattr(table, n)[i]))",
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "_set_row", 1),
    (
        "events.py",
        "return eqx.tree_at(lambda t: tuple(getattr(t, n) for n in names), table, new)",
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "_set_row", 1),
    ("executors/base.py", "obj = getattr(obj, part)"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "_canonical_leaves", 1
    ),
    ("executors/base.py", "leaf = np.asarray(getattr(obj, name))"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "_canonical_leaves", 1
    ),
    ("executors/base.py", "pick(getattr(trace, n)).tolist()"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "canonical_history", 1
    ),
    ("executors/base.py", "bool(getattr(carry, n))"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "canonical_history", 1
    ),
    (
        "executors/replay.py",
        "fields = {n: jnp.asarray(getattr(log, n)[i]) for n in LOG_INT_FIELDS}",
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "_row", 1),
    (
        "executors/replay.py",
        "fields.update({n: jnp.asarray(getattr(log, n)[i]) "
        'for n in ("cost", "charged")})',
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "_row", 1),
    (
        "executors/replay.py",
        '**{n: np.asarray(getattr(carry.log, n)) for n in (*LOG_INT_FIELDS, "valid")},',
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "replay_belief", 1),
    (
        "executors/replay.py",
        "np.asarray(getattr(trace, n))[handled].tolist()",
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "replay_belief", 1),
    (
        "handlers.py",
        "**{f.name: getattr(log, f.name)[slot] for f in dataclasses.fields(Record)}",
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "_log_rec", 1),
    ("records.py", "col = getattr(carry.log, name)"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "write_log", 1
    ),
    (
        "records.py",
        "cols[name] = col.at[s].set(jnp.where(ok, getattr(rec, name), col[s]))",
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "write_log", 1),
    ("snapshot.py", "col = getattr(log, name)"): Allowance(
        COLUMN_BY_NAME, ("dynamic-getattr",), "snapshot_view", 1
    ),
    (
        "snapshot.py",
        "leaves = [getattr(log, n) for n in LOG_INT_FIELDS] + [log.valid]",
    ): Allowance(COLUMN_BY_NAME, ("dynamic-getattr",), "snapshot_fingerprint", 1),
    ("battery/fork.py", "for name in type(trunk).__dataclass_fields__"): Allowance(
        FIELD_WALK, ("introspection",), "_trace_prefix", 1
    ),
    ("battery/fork.py", "for name in type(trunk.log).__dataclass_fields__:"): Allowance(
        FIELD_WALK, ("introspection",), "_records", 1
    ),
    ("battery/metamorphic.py", "for name in type(tb).__dataclass_fields__:"): Allowance(
        FIELD_WALK, ("introspection",), "_epoch_world", 1
    ),
    (
        "battery/metamorphic.py",
        "for name in type(cb.log).__dataclass_fields__:",
    ): Allowance(FIELD_WALK, ("introspection",), "_epoch_world", 1),
    ("battery/metamorphic.py", "new_assets = type(assets)("): Allowance(
        FIELD_WALK, ("introspection",), "with_zero_time_asset", 1
    ),
    (
        "battery/replay_latency.py",
        "for n in type(trace).__dataclass_fields__",
    ): Allowance(FIELD_WALK, ("introspection",), "_non_release", 1),
    (
        "battery/replay_latency.py",
        "for name in type(cb.log).__dataclass_fields__:",
    ): Allowance(FIELD_WALK, ("introspection",), "_compare_world", 1),
    (
        "battery/truth_swap.py",
        "for name in type(trace_a).__dataclass_fields__:",
    ): Allowance(FIELD_WALK, ("introspection",), "_prefix_equal", 1),
    ("battery/views.py", "for name in type(va.log).__dataclass_fields__:"): Allowance(
        FIELD_WALK, ("introspection",), "compare_views", 1
    ),
    (
        "records.py",
        "if not all(isinstance(c, PayloadColumn) for c in self.columns):",
    ): Allowance(SCHEMA_TYPE, ("study-kind",), "RecordKind.__post_init__", 1),
    ("records.py", "if not isinstance(kind, RecordKind):"): Allowance(
        SCHEMA_TYPE, ("study-kind",), "KindRegistry.register", 1
    ),
}


def _core_definitions():
    """Names of core enums and Abstract base classes, found without importing."""
    classes = [
        node
        for path in CORE_DIR.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef)
    ]
    enums = set(ENUM_BASES)
    while True:
        extended = enums | {
            node.name
            for node in classes
            if any(_symbol(base) in enums for base in node.bases)
        }
        if extended == enums:
            break
        enums = extended
    abstract = {node.name for node in classes if node.name.startswith("Abstract")}
    return enums - ENUM_BASES, abstract


def _builtin_types():
    return {
        name
        for name, value in vars(builtins).items()
        if isinstance(value, type) and not name.startswith("_")
    }


def _origins(tree):
    """Map each locally bound import name to the name it was imported as."""
    return {
        alias.asname or alias.name: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }


def _text(node):
    """String content of a constant or f-string, with substitutions as digits."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value if isinstance(part, ast.Constant) else "0"
            for part in node.values
        )
    return None


def _docstrings(tree):
    ids = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        ):
            ids.add(id(body[0].value))
    return ids


def _class_names(node):
    """Names in a class argument, flattening tuples and unions."""
    if isinstance(node, (ast.Tuple, ast.List)):
        return [name for part in node.elts for name in _class_names(part)]
    if isinstance(node, ast.BinOp):
        return _class_names(node.left) + _class_names(node.right)
    return [_symbol(node) or ast.dump(node)]


def _generic_base(node, builtin_types):
    """A subscript such as list[Finding] or Callable[...] is a type, not a lookup."""
    name = _symbol(node)
    return bool(name and (name in builtin_types or CAMEL.match(name)))


def _is_class_key(node, origins):
    if isinstance(node, ast.Call):
        return _symbol(node.func) == "type"
    name = _symbol(node)
    return bool(name and CAMEL.match(origins.get(name, name)))


def _scan(path, tree, enums, abstract, builtin_types, kind_names):
    """Yield (line, rule) for every forbidden dispatch form in one module."""
    origins = _origins(tree)
    study_names = _study_aliases(tree)
    safe = enums | abstract | builtin_types
    docstrings = _docstrings(tree)

    def resolved(node):
        name = _symbol(node)
        return origins.get(name, name) if isinstance(node, ast.Name) else name

    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        text = _text(node)
        if text is not None and id(node) not in docstrings:
            if SURVEY_NAME.search(text) or SURVEY_FRAGMENT.search(text):
                yield line, "survey-name"
            if text in kind_names:
                yield line, "kind-name"
        if isinstance(node, ast.Attribute):
            if node.attr == "name" and _study_reference(node.value, study_names):
                yield line, "study.name"
            if node.attr in INTROSPECTION:
                yield line, "introspection"
            if node.attr == "__dict__" and _study_reference(node.value, study_names):
                yield line, "study-vars"
        elif isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            if any(
                isinstance(side, ast.Attribute) and side.attr in SURVEY_ATTRIBUTES
                for side in sides
            ):
                yield line, "survey-id"
        elif isinstance(node, ast.Call):
            func = resolved(node.func)
            studied = any(_study_reference(arg, study_names) for arg in node.args)
            if func == "type":
                yield line, "introspection"
            elif func in TEXT_OF and studied:
                yield line, "study-text"
            elif func == "vars" and studied:
                yield line, "study-vars"
            elif func == "hasattr" and studied:
                yield line, "study-hasattr"
            elif func == "getattr" and len(node.args) >= 2:
                attribute = node.args[1]
                literal = isinstance(attribute, ast.Constant)
                if not literal or attribute.value in KIND_ATTRIBUTES:
                    yield line, "dynamic-getattr"
            elif func in {"isinstance", "issubclass"} and len(node.args) == 2:
                names = _class_names(node.args[1])
                if any(origins.get(name, name) not in safe for name in names):
                    yield line, "study-kind"
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in LOOKUPS
                and node.args
                and _is_class_key(node.args[0], origins)
            ):
                yield line, "class-dict"
        elif isinstance(node, ast.MatchClass):
            yield line, "class-pattern"
        elif isinstance(node, ast.Dict):
            if any(
                key is not None and _is_class_key(key, origins) for key in node.keys
            ):
                yield line, "class-dict"
        elif (
            isinstance(node, ast.Subscript)
            and not _generic_base(node.value, builtin_types)
            and _is_class_key(node.slice, origins)
        ):
            yield line, "class-dict"


def _functions(tree):
    """(first line, last line, qualified name) of every function and class."""
    spans = []

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                spans.append((child.lineno, child.end_lineno, name))
                visit(child, name + ".")
            else:
                visit(child, prefix)

    visit(tree, "")
    return spans


def _enclosing(spans, line):
    inside = [span for span in spans if span[0] <= line <= span[1]]
    return max(inside)[2] if inside else "<module>"


def _all_hits(root):
    """(path, relative, line, rule, stripped line text, function) with no exceptions."""
    files = sorted(root.rglob("*.py"))
    assert files, f"no Python sources under {root}"
    enums, abstract = _core_definitions()
    builtin_types = _builtin_types()
    kind_names = _study_kind_names() - {"Study"}
    hits = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source, filename=str(path))
        spans = _functions(tree)
        relative = path.relative_to(root).as_posix()
        for line, rule in sorted(
            set(_scan(path, tree, enums, abstract, builtin_types, kind_names))
        ):
            text = lines[line - 1].strip()
            hits.append((path, relative, line, rule, text, _enclosing(spans, line)))
    return hits


def _waived(hits, allowed):
    """Split hits into (reported, matched lines per allowance key)."""
    reported, matched = [], {key: set() for key in allowed}
    for hit in hits:
        _, relative, line, rule, text, function = hit
        entry = allowed.get((relative, text))
        if entry and rule in entry.rules and function == entry.function:
            if len(matched[(relative, text)] | {line}) <= entry.count:
                matched[(relative, text)].add(line)
                continue
        reported.append(hit)
    return reported, matched


def study_branches(root, allowed=ALLOWED):
    """Return (path, line, rule) for forbidden dispatch in a core tree, failing closed.

    Everything is forbidden except the exceptions reasoned in ALLOWED, each scoped to
    its rules, its enclosing function and its occurrence count, and type tests against
    builtins, core enums and core Abstract contracts. This is a lint for accidental
    dispatch on a study, not a proof against code written to evade it.
    """
    reported, _ = _waived(_all_hits(root), allowed)
    return [(hit[0], hit[2], hit[3]) for hit in reported]


def stale_allowances(root, allowed=ALLOWED):
    """Allowance keys whose rules, function or count no longer match the tree."""
    _, matched = _waived(_all_hits(root), allowed)
    return sorted(
        key for key, lines in matched.items() if len(lines) != allowed[key].count
    )
