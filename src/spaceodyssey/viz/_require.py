"""Import guard for the optional plotting dependency."""


def eyepiece():
    """Import eyepiece with an actionable installation hint on failure.

    Returns:
        The imported eyepiece module.

    Raises:
        ImportError: If eyepiece is unavailable.
    """
    try:
        import eyepiece
    except ImportError:
        raise ImportError(
            "spaceodyssey.viz requires eyepiece: pip install 'spaceodyssey[viz]'"
        ) from None
    return eyepiece
