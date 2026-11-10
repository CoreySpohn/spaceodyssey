"""The truth-kernel report refit: lane independence, oracle invariance, summary size.

Each test names the property or number it protects. The study is a stand-in with the
pieces the refit reads (capacities, observer, channel); no record log is run.
"""

import types

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("photomancy")

from spaceodyssey.adapters.photomancy import (
    CATALOG_FIELDS,
    CATALOG_GROUPS,
    DecisionSummary,
    ReportRefit,
    catalog_kind,
    report_kind,
    search_probability_truth,
)
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import EVERYONE, Audience, RefitStatus
from spaceodyssey.core.records import (
    KindRegistry,
    PayloadColumn,
    RecordKind,
    record_none,
)
from spaceodyssey.core.refresh import RefitRequest
from spaceodyssey.studies.bracket import SummaryObserver

SLOTS, GRID, MODES, BLOCKS = 2, 16, 2, 2
N_SUBJECTS, ROWS = 4, 6
WIDTH = len(CATALOG_FIELDS)


class StubChannel(eqx.Module):
    """A smooth sensitivity model with the four methods the summary reads."""

    n_modes: int = eqx.field(static=True, default=MODES)
    n_block_modes: int = eqx.field(static=True, default=BLOCKS)
    tick_days: float = eqx.field(static=True, default=1.0)

    @staticmethod
    def _phase(time_d, el):
        return 2.0 * jnp.pi * (time_d / jnp.exp(el[:, 0]) + el[:, 5])

    def detect_probability(self, subject, mode, time_d, elements):
        c = jnp.cos(self._phase(time_d, elements))
        shape = jnp.exp(-0.5 * ((c - 0.2 * mode) / 0.5) ** 2)
        return 0.9 * shape * (1.0 - 0.1 * subject) * (1.0 - 0.3 * elements[:, 1])

    def block_snr(self, subject, block_mode, start_d, elements):
        c = jnp.cos(self._phase(start_d, elements))
        return 20.0 * c**2 * (1.0 + block_mode) * 0.6 * (1.0 - 0.1 * subject)

    def sky_position(self, subject, time_d, elements):
        ph = self._phase(time_d, elements)
        a = 0.01 * jnp.exp(elements[:, 0] / 1.5)
        return jnp.stack([a * jnp.cos(ph), a * jnp.sin(ph) * elements[:, 2]], axis=-1)

    def luminosity(self, subject):
        return 1.0


def make_study(open_groups=("existence", "ephemeris", "class")):
    audiences = {g: Audience.OBSERVER | Audience.POLICY for g in open_groups}
    registry = KindRegistry(
        [catalog_kind(SLOTS, audiences), report_kind(SLOTS, EVERYONE)]
    )
    prior = DecisionSummary.zeros(N_SUBJECTS, SLOTS, GRID, MODES, BLOCKS, n_an=4)
    observer = SummaryObserver(prior=prior)
    caps = registry.lower(
        Capacities(n_subjects=N_SUBJECTS, rows_per_subject=ROWS, refit_lanes=2),
        observer,
    )
    return types.SimpleNamespace(
        caps=caps, observer=observer, channel_model=StubChannel()
    )


def catalog_payload(rng, present, class1):
    """A catalog payload row: random orbits for the slots, as the layout orders them."""
    row = np.zeros(WIDTH * SLOTS)
    for s in range(SLOTS):
        el = [rng.uniform(4.5, 6.0), rng.uniform(0, 0.5), rng.uniform(-0.8, 0.8)]
        el += [rng.uniform(0, 6.28), rng.uniform(0, 6.28), rng.uniform(0, 1)]
        el += [0.0, np.log(0.01)]
        cells = [present[s], *el[:6], el[6], float(class1[s]), el[7]]
        row[s * WIDTH : (s + 1) * WIDTH] = cells
    return row


def make_request(study, n_worlds, seed=0, source_shared=False):
    """A request over `n_worlds`: every entry dirty, one catalog row, three reports."""
    caps = study.caps
    rng = np.random.default_rng(seed)
    kinds = caps.kind_names
    base = jax.tree_util.tree_map(np.asarray, record_none(caps))
    shape = (n_worlds, 2, N_SUBJECTS, ROWS)
    rows = jax.tree_util.tree_map(
        lambda x: np.broadcast_to(x, shape + x.shape).copy(), base
    )
    payload, valid, kind, end = rows.payload, rows.valid, rows.kind, rows.end_tick
    for w in range(n_worlds):
        for s in range(N_SUBJECTS):
            cat = catalog_payload(rng, [1, 1 - (s % 2)], [1, s % 2])
            seen = rng.integers(0, 2, (3, SLOTS)).astype(float)
            for a in range(2):
                payload[w, a, s, 0] = cat
                valid[w, a, s, 0] = True
                kind[w, a, s, 0] = kinds.index("truth_catalog")
                for j in range(3):
                    payload[w, a, s, 1 + j, :SLOTS] = seen[j]
                    valid[w, a, s, 1 + j] = True
                    kind[w, a, s, 1 + j] = kinds.index("visit_report")
                    end[w, a, s, 1 + j] = 10 * (j + 1)
    own = np.arange(2 * N_SUBJECTS).reshape(2, N_SUBJECTS)
    source = np.broadcast_to(own, (n_worlds, 2, N_SUBJECTS)).copy()
    if source_shared:
        source[:, 1] = own[0]
    return RefitRequest(
        tick=np.full(n_worlds, 60, np.int64),
        ordinal=np.zeros((n_worlds, 2, N_SUBJECTS), np.int64),
        dirty=np.ones((n_worlds, 2, N_SUBJECTS), bool),
        n_rows=np.full((n_worlds, 2, N_SUBJECTS), 4, np.int64),
        rows=rows,
        row_keys=np.zeros((n_worlds, 2, N_SUBJECTS, ROWS, 2), np.uint32),
        keys=rng.integers(0, 2**31, (n_worlds, 2, N_SUBJECTS, 2)).astype(np.uint32),
        source=source,
    )


def same(a, b):
    return all(
        np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)
        for x, y in zip(a.slices, b.slices, strict=True)
    ) and np.array_equal(np.asarray(a.status), np.asarray(b.status))


def take_world(request, w):
    return jax.tree_util.tree_map(lambda x: np.asarray(x)[w : w + 1], request)


def test_bytes_per_star_match_the_budget():
    """Number: bytes per star per audience at S = 4, M = 6, B = 3, G = 16."""
    size = DecisionSummary.zeros(5, 4, 16, 6, 3).bytes_per_star()
    assert size == 5712
    assert abs(size - 5600) / 5600 < 0.03


def test_an_entry_does_not_depend_on_its_lane_or_batch():
    """Property: an entry's slices are bitwise equal alone, in a batch, in any order."""
    study = make_study()
    request = make_request(study, 3)
    full = ReportRefit(lanes=4).refit_batch(study, request)
    assert (np.asarray(full.status) == RefitStatus.OK).all()
    reversed_ = jax.tree_util.tree_map(lambda x: np.asarray(x)[::-1], request)
    back = ReportRefit(lanes=4).refit_batch(study, reversed_)
    for mine, theirs in zip(full.slices, back.slices, strict=True):
        assert np.array_equal(np.asarray(mine), np.asarray(theirs)[::-1])
    one = ReportRefit(lanes=1).refit_batch(study, take_world(request, 1))
    for mine, alone in zip(full.slices, one.slices, strict=True):
        assert np.array_equal(np.asarray(mine)[1], np.asarray(alone)[0])


def test_truth_kernel_ignores_the_refit_key_and_particle_count():
    """Property: under the truth kernel every slice is bitwise equal across keys."""
    study = make_study()
    request = make_request(study, 2)
    other = eqx.tree_at(lambda r: r.keys, request, request.keys + np.uint32(7))
    a = ReportRefit(n_particles=1024).refit_batch(study, request)
    b = ReportRefit(n_particles=8).refit_batch(study, other)
    assert same(a, b)


def test_a_shared_entry_copies_the_computing_entry():
    """Duplicates share one computation through `requests.source`."""
    study = make_study()
    request = make_request(study, 1, source_shared=True)
    out = ReportRefit().refit_batch(study, request)
    assert not np.asarray(out.computed)[:, 1].any()
    for piece in out.slices:
        assert np.array_equal(np.asarray(piece)[:, 1], np.asarray(piece)[:, 0])


def test_the_truth_summary_reads_the_catalog_and_the_reports():
    """The leaves are the true orbit's: detections counted, absent slots empty."""
    study = make_study()
    request = make_request(study, 1)
    out = ReportRefit().refit_batch(study, request)
    names = study.observer.subject_leaves
    leaf = dict(zip(names, out.slices, strict=True))
    reports = np.asarray(request.rows.payload)[0, 0, :, 1:4, :SLOTS]
    assert np.array_equal(leaf["n_det"][0, 0], reports.sum(axis=1).astype(int))
    # subject 1 has its second slot absent: nothing of it reaches the summary
    assert leaf["ess"][0, 0, 1, 1] == 0.0
    assert leaf["p_class"][0, 0, 1, 1] == 0.0
    assert not np.asarray(leaf["p_det"])[0, 0, 1, 1].any()
    # slot 0 is class 1 for every subject
    assert (np.asarray(leaf["p_class"])[0, 0, :, 0] == 1.0).all()
    assert (np.asarray(leaf["scatter"]) == 0.0).all()
    assert (np.asarray(leaf["status"]) == RefitStatus.OK).all()


def test_the_first_detection_probability_is_one_minus_the_miss_product():
    """`p_new` under truth is `1 - prod(1 - p_det)` over the live slots."""
    p = jnp.asarray([[[0.5, 0.1]], [[0.2, 0.3]]])
    live = jnp.asarray([True, True])
    got = search_probability_truth(p, live)
    assert np.allclose(got, [[1 - 0.5 * 0.8, 1 - 0.9 * 0.7]])
    assert np.allclose(search_probability_truth(p, jnp.asarray([True, False])), p[0])


def test_construction_refuses_a_registry_that_hides_a_catalog_group():
    """A hidden column reads as zero, so the registry is checked instead."""
    study = make_study(open_groups=("existence", "ephemeris"))
    with pytest.raises(ValueError, match="hidden"):
        ReportRefit(caps=study.caps)
    with pytest.raises(ValueError, match="hidden"):
        ReportRefit().refit_batch(study, make_request(study, 1))


def test_an_install_changes_only_its_star():
    """A refit installs one star; the prior everywhere else is bitwise unchanged."""
    study = make_study()
    observer = study.observer
    out = ReportRefit().refit_batch(study, make_request(study, 1))
    fitted = observer.init(study.caps)
    names = observer.subject_leaves
    fitted = eqx.tree_at(
        lambda b: tuple(getattr(b, n) for n in names),
        fitted,
        tuple(
            jnp.asarray(np.asarray(p)[0, 0]).astype(getattr(fitted, n).dtype)
            for n, p in zip(names, out.slices, strict=True)
        ),
    )
    after = observer.install(observer.init(study.caps), fitted, 0)
    for n in names:
        assert np.array_equal(getattr(after, n)[1:], getattr(observer.prior, n)[1:])
    assert not np.array_equal(after.n_obs, observer.prior.n_obs)


def test_a_wider_registered_kind_does_not_add_phantom_slots():
    """The slot count is the declared width, not the padded `payload_width`.

    Fails if the check reads columns past the catalog kind and reports a phantom slot
    as hidden when another registered kind is wider than the catalog.
    """
    audiences = {g: Audience.OBSERVER | Audience.POLICY for g in CATALOG_GROUPS}
    wide = RecordKind(
        "wide", tuple(PayloadColumn(f"w{i}", int(EVERYONE)) for i in range(40))
    )
    registry = KindRegistry([catalog_kind(1, audiences), report_kind(1), wide])
    prior = DecisionSummary.zeros(N_SUBJECTS, 1, GRID, MODES, BLOCKS, n_an=4)
    caps = registry.lower(
        Capacities(n_subjects=N_SUBJECTS, rows_per_subject=ROWS, refit_lanes=2),
        SummaryObserver(prior=prior),
    )
    ReportRefit(caps=caps)


def test_an_entry_without_a_catalog_row_installs_nothing():
    """With no catalog row every leaf is zero and the status is `NO_SUPPORT`."""
    study = make_study()
    request = make_request(study, 1)
    rows = request.rows
    valid = np.asarray(rows.valid).copy()
    kind = np.asarray(rows.kind)
    valid &= kind != study.caps.kind_names.index("truth_catalog")
    request = eqx.tree_at(lambda r: r.rows.valid, request, valid)
    out = ReportRefit().refit_batch(study, request)
    assert (np.asarray(out.status) == RefitStatus.NO_SUPPORT).all()
    for name, piece in zip(study.observer.subject_leaves, out.slices, strict=True):
        if name != "status":
            assert not np.asarray(piece).any(), name
