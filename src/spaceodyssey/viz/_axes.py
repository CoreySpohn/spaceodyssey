"""Shared display formatting for physical log axes."""


def decimal_log_ticks(axis, *, zero_floor=None):
    """Use readable major ticks and suppress minor labels on a log axis.

    Retain eyepiece's floor locator when a zero row has been installed.
    """
    from matplotlib.ticker import (
        FormatStrFormatter,
        FuncFormatter,
        MaxNLocator,
        NullFormatter,
    )

    if axis.get_scale() != "log":
        return
    lo, hi = sorted(axis.get_view_interval())
    if zero_floor is None:
        if 0 < lo < hi < 10 * lo:
            axis.set_major_locator(MaxNLocator(nbins=4))
        axis.set_major_formatter(FormatStrFormatter("%g"))
    else:
        axis.set_major_formatter(
            FuncFormatter(
                lambda value, pos: "0" if value == zero_floor else f"{value:g}"
            )
        )
    axis.set_minor_formatter(NullFormatter())
