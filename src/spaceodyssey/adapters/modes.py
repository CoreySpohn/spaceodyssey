"""Host-side declarations of observing modes and their channel layout."""

import dataclasses
import json
import math
from pathlib import Path

from hwoutils.constants import d2s

from spaceodyssey.adapters.physical_profile import load_profile
from spaceodyssey.adapters.skyscapes import Band


@dataclasses.dataclass(frozen=True)
class ModeSpec:
    """One observing mode; only imaging carries a reporting threshold."""

    name: str
    kind: str
    band: Band
    optics_throughput: float
    frame_time_s: float
    resolving_power: float | None
    snr_threshold: float | None

    def __post_init__(self):
        """Refuse unsupported kinds and misplaced requirements."""
        if not self.name or self.kind not in ("imaging", "spectrograph"):
            raise ValueError("a mode needs a name and kind imaging or spectrograph")
        if self.kind == "imaging" and self.snr_threshold is None:
            raise ValueError("imaging requires a reporting threshold")
        if self.kind == "spectrograph" and self.snr_threshold is not None:
            raise ValueError("a spectrograph cannot carry a requirement")
        for name in (
            "optics_throughput",
            "frame_time_s",
            "resolving_power",
            "snr_threshold",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.optics_throughput > 1:
            raise ValueError("optics_throughput must not exceed 1")


@dataclasses.dataclass(frozen=True)
class ModeTable:
    """Every mode's durations in mode order, followed by the free catalog channel."""

    modes: tuple[ModeSpec, ...]
    durations_d: tuple[float, ...]

    def __post_init__(self):
        """Refuse ambiguous names, band reports and duration declarations."""
        object.__setattr__(self, "modes", tuple(self.modes))
        object.__setattr__(self, "durations_d", tuple(self.durations_d))
        if not self.modes or not self.durations_d:
            raise ValueError("modes and durations_d must be nonempty")
        if len({m.name for m in self.modes}) != len(self.modes):
            raise ValueError("duplicate mode names")
        if any(not math.isfinite(d) or d <= 0 for d in self.durations_d):
            raise ValueError("durations_d must be finite and positive")
        imaging = tuple(m for m in self.modes if m.kind == "imaging")
        if len({m.band for m in imaging}) > 1:
            raise ValueError("imaging modes must share one band")
        for mode in imaging[1:]:
            if mode.snr_threshold != imaging[0].snr_threshold:
                raise ValueError(
                    f"imaging modes {imaging[0].name!r} ({imaging[0].snr_threshold}) "
                    f"and {mode.name!r} ({mode.snr_threshold}) must share one "
                    "reporting threshold; the refit fits one photometric scale "
                    "and one reporting law per track"
                )
        by_band = {}
        for mode in self.modes:
            optics = (mode.optics_throughput, mode.frame_time_s, mode.resolving_power)
            if mode.band in by_band and by_band[mode.band] != optics:
                raise ValueError(
                    "modes sharing a band must share throughput, frame time "
                    "and resolving power"
                )
            by_band[mode.band] = optics

    @property
    def bands(self) -> tuple[Band, ...]:
        """Distinct bands in declaration order."""
        return tuple(dict.fromkeys(m.band for m in self.modes))

    @property
    def n_bands(self) -> int:
        """Size of the band axis."""
        return len(self.bands)

    def band_index(self, mode_name: str) -> int:
        """Band axis index of the named mode."""
        return self.bands.index(self.modes[self._mode_index(mode_name)].band)

    def _mode_index(self, mode_name: str) -> int:
        return tuple(m.name for m in self.modes).index(mode_name)

    @property
    def imaging_band(self) -> Band:
        """The single photometric band fitted by the report refit."""
        return next(m.band for m in self.modes if m.kind == "imaging")

    def channel(self, mode_name: str, duration_index: int) -> int:
        """Channel of a named mode at one duration index."""
        if not 0 <= duration_index < len(self.durations_d):
            raise IndexError("duration index out of range")
        return self._mode_index(mode_name) * len(self.durations_d) + duration_index

    @property
    def n_channels(self) -> int:
        """Number of observing channels including the catalog."""
        return self.catalog_channel + 1

    @property
    def catalog_channel(self) -> int:
        """The catalog follows every observing channel."""
        return len(self.modes) * len(self.durations_d)

    def mode_of(self, channel: int) -> int:
        """Mode axis index of an observing channel."""
        if not 0 <= channel < self.catalog_channel:
            raise IndexError("not an observing channel")
        return channel // len(self.durations_d)

    def duration_of(self, channel: int) -> int:
        """Duration axis index of an observing channel."""
        self.mode_of(channel)
        return channel % len(self.durations_d)

    def duration_s(self, channel: int) -> float:
        """Science exposure of an observing channel [s]."""
        return self.durations_d[self.duration_of(channel)] * d2s

    def _channels(self, kind):
        return tuple(
            c
            for c in range(self.catalog_channel)
            if self.modes[self.mode_of(c)].kind == kind
        )

    @property
    def imaging_channels(self) -> tuple[int, ...]:
        """Observing channels that write visit reports."""
        return self._channels("imaging")

    @property
    def spectrograph_channels(self) -> tuple[int, ...]:
        """Observing channels that write spectral outcomes."""
        return self._channels("spectrograph")

    @property
    def resources(self) -> tuple[tuple[str, int], ...]:
        """Named ledger columns charged by the modes."""
        return (("telescope_days", 2),)

    def resource(self, name: str) -> int:
        """Ledger column of a named resource."""
        return dict(self.resources)[name]


def profile_modes(profile) -> tuple[ModeSpec, ...]:
    """Survey mode specs; spectral requirements remain reference data only."""
    return tuple(
        ModeSpec(
            name,
            m["kind"],
            Band(m["wavelength_nm"], m["bin_width_nm"]),
            m["optics_throughput"],
            m["frame_time_s"],
            m.get("resolving_power"),
            m["snr_required"] if m["kind"] == "imaging" else None,
        )
        for name, m in load_profile(profile)["survey"]["modes"].items()
    )


def load_mode_specs(path) -> tuple[ModeSpec, ...]:
    """Read a mode list or a documented object with a modes list.

    Each band declares center_nm and width_nm. An object may also document the
    study's assumptions without adding those annotations to the physical modes.
    """
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows["modes"]
    return tuple(ModeSpec(**{**row, "band": Band(**row["band"])}) for row in rows)


def union_modes(*mode_sets: tuple[ModeSpec, ...]) -> tuple[ModeSpec, ...]:
    """Keep modes in first-seen name order, refusing conflicting declarations."""
    modes = {}
    for mode_set in mode_sets:
        for mode in mode_set:
            if mode.name in modes and modes[mode.name] != mode:
                raise ValueError(f"conflicting specifications for mode {mode.name!r}")
            modes.setdefault(mode.name, mode)
    return tuple(modes.values())
