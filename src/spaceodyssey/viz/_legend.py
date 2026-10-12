"""Legend keys sampled from plotted artists without adding data artists."""


def scatter_key(collection, index=0):
    """Copy one marker's path, size, and rendered colors into a legend proxy."""
    from matplotlib.collections import PathCollection
    from matplotlib.transforms import IdentityTransform

    collection.update_scalarmappable()
    paths = collection.get_paths()
    options = {}
    for name in ("facecolors", "edgecolors", "sizes", "linewidths"):
        values = getattr(collection, f"get_{name}")()
        options[name] = [values[index % len(values)]] if len(values) else []
    return PathCollection(
        [paths[index % len(paths)]], transform=IdentityTransform(), **options
    )


def legend_below(ax, handles=None, labels=None, *, ncols=2):
    """Place a compact key below the axis labels, participating in layout."""
    from matplotlib.transforms import ScaledTranslation

    options = {} if handles is None else {"handles": handles, "labels": labels}
    return ax.legend(
        **options,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        bbox_transform=ax.transAxes
        + ScaledTranslation(0, -42 / 72, ax.figure.dpi_scale_trans),
        ncols=ncols,
        fontsize="small",
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )
