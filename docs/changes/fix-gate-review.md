# Gate review

## fix-gate

Items 5-10 strengthen the acceptance harness and core import lint. No runtime
source, stored hand history, random draw, or canonical-history field changes.

- World coverage requires exactly one run for each independently expected world
  ID, before any invariant is called. Empty, truncated, duplicate and substituted
  world sets cannot satisfy K3.
- K1 scans every core module. Survey strings in expressions, study-name reads,
  type introspection, dynamic attributes, class patterns, concrete type tests and
  class-keyed maps fail unless a reviewed structural operation has an entry in the commented
  allowlist, keyed on file and exact source line (not line number) with a stale-entry check. Core enums, builtins and core Abstract contracts
  are the only general type-test exemptions. All six dispatch reproductions
  have permanent controls, supplemented by variations outside those spellings.
- Annotation checking walks the composed field-type graph with cycle detection,
  including concrete classes, static fields and container/callable/union
  arguments. An escape on Agenda or a deeper concrete field fails K1.
- The import boundary rejects dynamic import and loader APIs independently of
  their destination, including unresolved arguments and aliases. The domain-word
  lint remains a separate vocabulary check.
- Draw comparisons require finite paired values and multiple matched identities;
  the gate also requires the known fixture's minimum measurement/policy coverage.
  Correct keys with NaN-only reporting corruption cannot satisfy a negative
  control.
- A2, A3 and A4 are collected by the single advertised gate module. They invoke
  the same replay, fork and latency assertions as the focused acceptance tests.
  The ownership table now states the library-map search for each fixture.

Residual limits: K1 is a conservative syntax check with reviewed structural
exceptions, not a proof of arbitrary Python behavior. The import lint forbids
runtime loading rather than resolving its possible destinations. The independent
invariant and canonical-history gaps remain the responsibility of the sibling
fixes; this branch does not change those implementations.
