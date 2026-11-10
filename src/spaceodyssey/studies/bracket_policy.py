"""A stand-in decision policy for the bracket study, until the real one is built.

`StandInPolicy` is a minimal rule over the decision summary: catalog kickoff, then
blocks, follow-ups and searches by detection probability per charged day. It reads
the summary leaves `ref_tick`, `n_obs`, `p_new`, `n_det`, `p_det`, `p_block` and
`p_class`, and it is not the bracket policy. Slots are exchangeable to the belief (the
slot index carries the truth's existence ordering), so the rule never uses the index
beyond addressing a slot of the summary.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.adapters.photomancy import GRID_SPACING_D, N_GRID
from spaceodyssey.core.enums import (
    F64,
    I64,
    Disposition,
    EffectKind,
    Scope,
)
from spaceodyssey.core.kinds import AbstractPolicy
from spaceodyssey.studies import bracket, toy

# Future epochs of the summary, their spacing [d], detection and block modes.
N_NODES = N_GRID
NODE_SPACING_D = GRID_SPACING_D
N_DETECT, N_BLOCK = 6, 3


class StandInPolicy(AbstractPolicy):
    """A minimal rule over the summary leaves; one effect per decision.

    State per star in `pstate`: last visit tick, visits, search visits, then a block
    flag per slot; after the stars, the charged days, the kickoff count and a stop
    flag. Kickoff observes the catalog mode of one star per decision (the release of
    its row wakes the policy for the next). After a visit the policy waits for the
    next refresh before deciding, then in order: start a block for a claimed slot
    when the current node is its best; follow up a detected slot at the best node;
    search the star with the best first-detection probability per charged day;
    otherwise wait a day. When nothing affordable is left it attempts one more visit
    (the refusal is the budget binding) and stops.

    Attributes:
        block_mode: ``int[n_stars]`` the block mode each star's reference twin needs.
        budget_d: Telescope days the policy may spend.
        n_slots: Planet slots per star.
        max_visits, max_search: Per-star visit and search limits.
    """

    block_mode: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, I64))
    budget_d: float = eqx.field(static=True)
    n_slots: int = eqx.field(static=True)
    max_visits: int = eqx.field(default=24, static=True)
    max_search: int = eqx.field(default=8, static=True)

    def declare(self):
        """The campaign drain the policy ends with."""
        from spaceodyssey.core.declarations import Declaration

        return Declaration(stops=((Scope.CAMPAIGN, Disposition.DRAIN),))

    def decide(self, pstate, view, study, key):
        """One decision: kickoff, wait for a fresh summary, or act on it."""
        caps, ch = study.caps, study.channels
        n, slots = caps.n_subjects, self.n_slots
        stride = 3 + slots
        per_day = jnp.asarray(86_400_000, I64)
        stars = pstate[: n * stride].reshape(n, stride)
        last_tick, n_vis, n_search = stars[:, 0], stars[:, 1], stars[:, 2]
        flags = stars[:, 3:]
        charged, kick, stopped = pstate[n * stride : n * stride + 3]
        belief = view.belief
        cost = ch.cost[:, bracket.SHARED_RESOURCE]
        remaining = self.budget_d - charged
        tick = view.tick
        horizon = study.horizon_tick

        # Kickoff: the catalog row of star `kick`.
        kicking = kick < n
        # Fresh: every visit made has been read by a refit.
        stale = jnp.any(n_vis > belief.n_obs.astype(F64))
        next_day = ((tick // per_day) + 1) * per_day + 1

        node_ticks = belief.ref_tick[:, None] + jnp.round(
            jnp.arange(N_NODES) * NODE_SPACING_D * per_day
        ).astype(I64)
        g_cur = jnp.clip(
            jnp.round((tick - belief.ref_tick) / (NODE_SPACING_D * per_day)),
            0,
            N_NODES - 1,
        ).astype(I64)
        s_ix = jnp.arange(n)
        weight = jnp.maximum(cost[:N_DETECT], 1e-9)

        # 1. Block.
        claimed = (belief.n_det >= 3) & (belief.p_class >= 0.5)
        want_block = claimed & (flags < 0.5)
        pb_mode = belief.p_block[s_ix, :, :, self.block_mode]
        pb_now = pb_mode[s_ix, :, g_cur]
        best_pb = pb_mode.max(axis=2)
        block_ok = want_block & (pb_now >= 0.8) & (pb_now >= best_pb)
        block_score = jnp.where(block_ok, 1.0, -1.0)
        bs = jnp.argmax(block_score.reshape(-1))
        block_star, block_slot = bs // slots, bs % slots
        do_block = block_score.reshape(-1)[bs] > 0
        block_channel = N_DETECT + self.block_mode[block_star]

        # 2. Follow-up.
        known = (belief.n_det >= 1) | (belief.p_class >= 0.5)
        eligible = known & ~claimed & (belief.p_class >= 0.05)
        late = node_ticks >= (last_tick + 15 * per_day)[:, None]
        pd_val = belief.p_det / weight[None, None, None, :]
        pd_val = jnp.where(
            late[:, None, :, None] & eligible[:, :, None, None], pd_val, -1.0
        )
        flat = pd_val.reshape(n, -1)
        fu_best = flat.argmax(axis=1)
        fu_score = flat.max(axis=1)
        fu_slot = fu_best // (N_NODES * N_DETECT)
        fu_node = (fu_best // N_DETECT) % N_NODES
        fu_mode = fu_best % N_DETECT
        fu_p = belief.p_det[s_ix, fu_slot, fu_node, fu_mode]
        fu_ok = (
            (fu_score > 0)
            & (fu_node == g_cur)
            & (fu_p >= 0.5)
            & (n_vis < self.max_visits)
        )
        fs = jnp.argmax(jnp.where(fu_ok, fu_score, -1.0))
        do_follow = fu_ok[fs]

        # 3. Search.
        rested = (tick - last_tick) >= 15 * per_day
        sp = belief.p_new[s_ix, g_cur, :]
        sp_val = sp / weight[None, :]
        sp_ok = (
            (rested | (n_vis == 0))[:, None]
            & (n_search < self.max_search)[:, None]
            & (n_vis < self.max_visits)[:, None]
            & (sp >= 0.02)
        )
        sp_flat = jnp.where(sp_ok, sp_val, -1.0).reshape(-1)
        sb = jnp.argmax(sp_flat)
        do_search = sp_flat[sb] > 0
        search_star, search_mode = sb // N_DETECT, sb % N_DETECT

        star = jnp.where(do_block, block_star, jnp.where(do_follow, fs, search_star))
        channel = jnp.where(
            do_block, block_channel, jnp.where(do_follow, fu_mode[fs], search_mode)
        )
        act = do_block | do_follow | do_search
        # Budget: a choice that does not fit falls back to the cheapest visit; when not
        # even that fits, one attempt is made and the policy stops.
        fits = cost[channel] <= remaining
        cheapest_fits = cost[0] <= remaining
        channel = jnp.where(fits, channel, 0)
        attempt_last = act & ~fits & ~cheapest_fits
        use = act & ~jnp.asarray(stopped > 0.5)
        done = (tick >= horizon) | (stopped > 0.5)

        observe_cat = kicking
        kind = jnp.select(
            [done, observe_cat, stale & ~kicking, use],
            [EffectKind.STOP, EffectKind.OBSERVE, EffectKind.WAIT, EffectKind.OBSERVE],
            EffectKind.WAIT,
        )
        cat_channel = N_DETECT + N_BLOCK
        eff_star = jnp.where(observe_cat, jnp.minimum(kick.astype(I64), n - 1), star)
        eff_channel = jnp.where(observe_cat, cat_channel, channel)
        wake = jnp.where(stale & ~kicking, next_day, tick + per_day)
        eff = toy._effects(
            caps,
            kind=jnp.stack([kind, jnp.asarray(0, I64)]),
            subject=eff_star,
            channel_index=eff_channel,
            group=view.n_dec,
            tick=jnp.stack(
                [jnp.where(kind == EffectKind.WAIT, wake, 0), jnp.asarray(0, I64)]
            ),
            lane=0,
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
        )
        observing = (kind == EffectKind.OBSERVE) & ~observe_cat
        is_block = observing & do_block
        row = star
        is_search = observing & ~do_block & ~do_follow
        new = pstate
        new = new.at[row * stride].set(
            jnp.where(observing, tick.astype(F64), pstate[row * stride])
        )
        new = new.at[row * stride + 1].add(observing.astype(F64))
        new = new.at[row * stride + 2].add(is_search.astype(F64))
        slot_ix = row * stride + 3 + jnp.where(do_block, block_slot, 0)
        new = new.at[slot_ix].add(is_block.astype(F64))
        base = n * stride
        spent = jnp.where(observing & (cost[channel] <= remaining), cost[channel], 0.0)
        new = new.at[base].add(spent)
        new = new.at[base + 1].add(observe_cat.astype(F64))
        new = new.at[base + 2].set(
            jnp.maximum(pstate[base + 2], (observing & attempt_last).astype(F64))
        )
        return eff, new, jnp.asarray(0.0, F64)


__all__ = ["StandInPolicy"]
