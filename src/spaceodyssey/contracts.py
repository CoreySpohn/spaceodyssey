"""Contract types that flow across seam boundaries, in one namespace.

Seam implementations declare what they consume and produce as strings naming
public types ("spaceodyssey.SummaryDataset", "planit_py.Action"). Those strings are
resolved to classes here, and compatibility on a dataflow edge is plain
nominal subtyping: an edge is wired correctly when some produced type is a
subclass of some consumed type.
"""

import importlib

from planit_py.vocabulary import Action, ObservingContext

from spaceodyssey.aliases import resolve_qualname_alias
from spaceodyssey.errors import WiringError

__all__ = [
    "Action",
    "ObservingContext",
    "compatible",
    "resolve_contract",
]


def resolve_contract(qualname: str) -> type:
    """Resolve a module-qualified contract-type name to the class it names.

    Args:
        qualname: A name like "spaceodyssey.SummaryDataset": everything before the
            last dot is imported as a module, the last component is looked up
            on it.

    Returns:
        The named class.

    Raises:
        WiringError: If the name is not module-qualified, the module is not
            importable (for example an optional dependency that is not
            installed), the attribute is missing, or it is not a class.
    """
    module_name, _, attr = resolve_qualname_alias(qualname).rpartition(".")
    if not module_name:
        raise WiringError(
            f"contract type '{qualname}' must be module-qualified, "
            "like 'spaceodyssey.SummaryDataset'"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as err:
        raise WiringError(
            f"contract type '{qualname}' is unavailable: "
            f"'{module_name}' is not importable ({err})"
        ) from err
    try:
        obj = getattr(module, attr)
    except AttributeError as err:
        raise WiringError(
            f"contract type '{qualname}' not found: "
            f"'{module_name}' has no attribute '{attr}'"
        ) from err
    if not isinstance(obj, type):
        raise WiringError(f"contract type '{qualname}' is not a class: {obj!r}")
    return obj


def compatible(produces: tuple[str, ...], consumes: tuple[str, ...]) -> bool:
    """Check whether produced contract types satisfy consumed ones.

    An implementation that declares no consumed types accepts anything.
    Otherwise at least one produced type must be a subclass of at least one
    consumed type.

    Args:
        produces: Contract-type names the upstream implementation produces.
        consumes: Contract-type names the downstream implementation consumes.

    Returns:
        True when the edge is type-compatible.

    Raises:
        WiringError: If any named contract type cannot be resolved.
    """
    if not consumes:
        return True
    consumed = [resolve_contract(name) for name in consumes]
    produced = [resolve_contract(name) for name in produces]
    return any(issubclass(prod, cons) for cons in consumed for prod in produced)
