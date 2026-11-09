---
orphan: true
---

# Slice 2b A10: equal resolutions on the observer-init boundary

The Slice 2b spec amends D010 (2026-11-07): a retained table prior is accepted on
`AudiencePriorRefit`, and also on the observer-init boundary (`MaskedBatchRefit` with
the observer's own kernel, `observer_refit_kernel`) when both audiences resolve to the
same prior. Different resolutions on that boundary stay `D009`; every other boundary
with a retained table prior (`OftiRefit`, a host fit, a kernel that never reads the
installed prior, the updater boundary) stays `D010`. R2 had refused the equal case
(`slice-2b-r2.md`, "Review round 2"); this change restores its acceptance.

## What changed

- `boundary_findings` (`studies/prior_observer.py`) returns no finding for the
  observer-init boundary when `same_resolution` holds for every subject. That check
  compares both audiences' prior beliefs bit for bit and, on every raw row the policy
  receives (policy-only rows included), whether each audience's rule drops it. The
  module docstring and the `D010` message name the two accepted boundaries.
- `alike_table` (`tests/core/prior_helpers.py`): the refit table with one public
  subject-0 prior in place of the observer and policy priors.

## Tests

| Test | Tier | Cold time | Fails when |
|---|---|---|---|
| `test_d009_refuses_different_priors_on_the_observer_init_boundary` (changed) | unit | 0.86 s | the observer-init boundary is not `D009` for different resolutions, or is refused for equal ones |
| `test_a_policy_only_row_is_judged_by_the_policy_rule` (changed) | unit | 1.01 s | a policy-only row the policy keeps makes the boundary refused, or one the policy drops is not `D009` |
| `test_equal_resolutions_pass_on_the_observer_init_boundary` (JSON) | unit | 0.96 s | a run document binding the observer's kernel is refused for equal resolutions, or a host fit is accepted |
| `test_equal_resolutions_refit_alike_on_both_boundaries` | medium | 11.0 s | the observer-init run does not hold each audience's prior plus its rows (from the log), differs from the `AudiencePriorRefit` run, or installs no refit; control: the different-resolution study on the observer-init boundary gives the policy the observer's prior and fails the same check |

The `D010` refusals of OFTI, a host fit and a flat kernel
(`test_d010_refuses_a_table_prior_with_a_boundary_that_ignores_it`) and the JSON
boundary cases are unchanged.

## Mutants

| Mutant | Caught by |
|---|---|
| the observer-init boundary accepts different resolutions | the D009 test, the policy-only-row test, the JSON `observer_init` boundary case |
| equal resolutions stay `D010` (the R2 rule) | the D009 test, the policy-only-row test, the JSON equal-resolution test |
| `same_resolution` ignores the rows each rule drops | the policy-only-row test |
| `same_resolution` ignores the priors | the D009 test, the JSON `observer_init` boundary case |
| the observer's own refit starts from a flat belief | the equal-resolution run test |
