# Records

A record is one row of the log: what an acquisition produced, with its timing, cost,
provenance and a typed payload. The log is the same columns with a leading row axis.

## Statuses

| Status | Meaning |
| --- | --- |
| `COMPLETED` | The occupancy ran to its end. |
| `FAILED` | A fault ended the occupancy; charged in full. |
| `PARTIAL` | A cancel ended the occupancy early; the unused cost came back. |
| `CANCELLED` | Declared; no handler writes it yet (a cancel writes `PARTIAL`). |
| `NULL` | The acquisition ran and produced no usable value. It is released like any other row; the observer decides what to do with it. |
| `UNRELEASED` | Logged but held back: never released to the policy, never accepted by the observer, never visible in a snapshot. |

## Identities and provenance

| Field | Meaning |
| --- | --- |
| `acquisition_id` | The physical act, unique per admission. |
| `revision` | Product revision of that acquisition, 0 for the first. |
| `released_tick` | Tick of release to the policy audience, -1 until then. |
| `accepted_tick` | Tick the observer accepted the row, -1 until then. |
| `superseded_by` | The revision that replaced this row, -1 if none. |
| `source`, `vintage`, `config_id` | The channel id, the phase at admission, and the servicing hook (0). |
| `audience` | The `Audience` flags the row is released to; it comes from the channel. |

## Kinds and column audiences

A `RecordKind` names a payload layout: an ordered tuple of `PayloadColumn`, each with a
name and the audiences allowed to read it. A column must declare its audience; an empty
or unknown one fails when the column is built.

| Audience flag | Meaning |
| --- | --- |
| `OBSERVER` | The observer folds the column. |
| `POLICY` | A policy sees the column in its snapshot. |
| `SCORER` | Only the scorer reads the column, from the log. |

A column marked `SCORER` alone stays in the log and is zero in every snapshot and in
every record the observer folds. This is how a channel keeps a value that would let a
reader undo its noise (the raw variate) out of every decision.

A `KindRegistry` holds the kinds of one study in order; a kind's index is the `kind`
value of its records. `KindRegistry.lower(caps)` returns the capacities with
`payload_width` widened to the widest kind and one audience row per kind. Studies build
capacities through it, so a kind never has to be listed twice. Capacities that were never
lowered leave every column readable.

## Release and the fold

The observer accepts every row released to it and folds the first accepted revision of
an acquisition. When a later revision is accepted, the earlier rows are marked
`superseded_by` it, and the later revision is logged but not folded again. The policy
belief follows the same rule over the rows released to the policy. Replacing a fold with
a refit is left to the observer kit.

A snapshot reports `superseded_by` from the rows its own audience can see, so an
observer-only revision does not show through a policy snapshot.

## A second kind

`spaceodyssey.studies.catalog` defines a delayed catalog row. Its channel releases the
row a fixed delay after the occupancy ends, to the observer only. The observer folds it
at that tick and the policy never sees it. The module uses only the public registry
interface; nothing in `spaceodyssey.core` names it.
