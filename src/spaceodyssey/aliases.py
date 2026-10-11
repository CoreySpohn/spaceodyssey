"""Versioned qualified-name aliases used only at resolution time.

Version 1 (2026-10-10) maps the deprecated package namespace, including all
its submodules and public exports. Match complete dotted components so that
unrelated packages are unaffected. Keep this table when adding later aliases.

Never apply aliases to stored references: the original identifiers remain
part of manifest provenance and content-derived identities.
"""

QUALNAME_ALIASES_VERSION = 1
QUALNAME_ALIASES = {"hwosim": "spaceodyssey"}


def resolve_qualname_alias(qualname: str) -> str:
    """Return the import name for an exact or dotted-prefix alias."""
    for old, new in QUALNAME_ALIASES.items():
        if qualname == old or qualname.startswith(old + "."):
            return new + qualname[len(old) :]
    return qualname
