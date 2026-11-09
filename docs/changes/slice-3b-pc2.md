---
orphan: true
---

# Slice 3b PC2: checkpoint and resume

A run can now save its state at a boundary and a later call can continue from it. The
sink contract is `src/spaceodyssey/core/checkpoint.py` (no IO), the directory sink is
`src/spaceodyssey/codec/checkpoint.py`, and both executors take `checkpoint=` and `resume=`.
`RunStatus.TRUNCATED_AT_HOLD = 6` is new and follows `AWAITING` in precedence.

## What is saved

One directory per run. Each save publishes an immutable generation `gen-NNNNNN/` holding
`carry.npz` and `manifest.json`, then moves the pointer `CURRENT` (generation number and the
manifest digest) with a write, a rename and a directory sync. Trace rows go in
content-addressed, append-only chunks under `chunks/`, written before the generation that
names them. The last two generations are kept, and a chunk is kept while a kept generation
names it. The manifest records a digest per array and per file, so a payload that was cut
short, edited or swapped is refused (C004) and a pointer to nothing usable is refused (C005,
C006).

A resumed carry is rebuilt on a template built from the study and the truth, so its structure
comes from code and the file supplies array contents only. Each array must match the template
dtype and trailing shape (C007).

## Run identity

The identity is a digest of: the study (the canonical bytes of `dump_study`, so a study that
cannot be dumped is refused, C001), the roots, the truth tree, `key_version`, the codec
format and schema, and the run configuration (executor, row axis, chunk size, policy form,
wake function, service class and the refit boundary's binding record). A changed part is
refused with its name (C002). The dependency fingerprint (jax, jaxlib, numpy, spaceodyssey,
python, x64, backend) is compared separately: a mismatch is refused (C003) unless the call
passes the override flag, and an override is written into every later manifest. A directory
that already holds a checkpoint is refused without `resume` (C008).

The service and the wake function are identified by their qualified names only. A change in
the body of either under the same name is not seen, and neither is a change in the body of the
refit boundary beyond its recorded binding (kind, version and model hash). Hashing a callable
would tie the identity to the bytecode of one Python build, so the limitation is stated here
instead. For the same reason the dependency fingerprint keeps the installed spaceodyssey
version string as it is, including a development suffix, and resuming across a code change
that moves the version requires the override flag.

The identity also records whether the host run keeps a policy view log and an effects log. A
resume that drops a log the saved run kept would leave a silent gap in it, so it is refused
(C002). One difference is allowed on purpose: a run saved with no service that stopped on a
pending request (status `AWAITING`) may be resumed with a service, because the answer is what
it waited for. A run that never raised a request is unaffected by the difference.

## Boundaries, and what "exactly once" rests on

- Host: after acquisition (a handled `DONE` that wrote a record), after each refit install
  (inference), and at the end (`final`, or `hold`).
- Compiled: chunk ends and holds only. A compiled scan has no per-event boundary, so "after
  acquisition" exists only on the host. The comparator for a resumed compiled run is the
  chunked run at the same chunk size, and the chunk size is part of the identity.

Exactly-once on the compiled executor comes from restoring the state and replaying the
chunks deterministically, not from a boundary at each event. A pending request or held
refresh checkpoint is restored with the carry and answered first on resume, once, before any
step is taken. A step budget that ends with a checkpoint held keeps the hold (it is not
dropped) when a sink is attached, and the run reports `TRUNCATED_AT_HOLD`; with no sink the
hold is dropped exactly as before and the new status is never emitted. `n_steps` is a fresh
budget for each call.

## Single scan against chunked

A single-scan compiled run and the chunked run are not bit-equal in the last bit: a program
that takes its carry as an input compiles the same arithmetic slightly differently from one
that builds the carry inside. Measured on the toy five-acquisition survey (two worlds, 60
steps, chunk 7): the canonical histories have the same structure and integers, world 0 is
bit equal and world 1 differs by 1 ulp (1.1e-16 absolute) in a float. This is the existing
behaviour of the chunked driver and this change does not alter it or claim equality. Two chunk
sizes of the same refresh study (7 and 60) were also equal to zero ulps on that run, which is
an observation and not a guarantee, so chunk size stays in the identity. A compiled run that
resumes is compared with the chunked run at the same chunk size, where the final carry is bit
equal. The raw trace rows are not: the empty rows that mark a waiting chunk fall at different
positions once the chunk grid restarts, so the comparison is on the canonical history, which
ignores empty rows, and it is exact (zero ulps).

## Decisions the spec did not make

- A pointer that is missing while generations exist is refused rather than guessed (C005, and
  the message says `recover=True` loads the newest generation that verifies). The exception is
  a first save that died before its pointer committed: such a directory carries a `PENDING`
  marker, written before the first generation and removed after the first pointer commit, and
  loads as empty. The marker is what tells a crash from a deleted pointer.
- Retention counts only generations that have a manifest, so a half-written generation left
  by a crash never costs the last complete one its place. A run clears manifest-less
  generation directories when it begins.
- `resume=True` on a directory with no checkpoint is refused (C010) instead of starting
  fresh. A directory whose first save never committed counts as empty, so it is started
  without `resume`.
- `bucket=` with `checkpoint=` is refused: the chunked driver the checkpoint selects has no
  bucketed program.
- The host manifest `steps` counts every step of the run, including those restored.
- The host `run_worlds` refuses a checkpoint: a host checkpoint holds one world.
- An undumpable study is refused with C001, and a non-core class needs a kinds mapping on the
  sink.
- The dependency fingerprint holds the packages and settings listed above and nothing else.
- Dropping the hold at a budget end is a property of a run without a sink only.
- Held-by-budget is reported per world by the result, and the status of a world that is not
  waiting is untouched.

## The fsync mutant is dropped

Card 10 asked for a mutant that removes the sync order unless it catches something the
in-process injection does not. The injection stops the write at each of its seven durable
steps (chunk, payload, manifest, generation sync, pointer write, pointer rename, pointer
sync) and checks that the previous generation is still the one resumed. A process killed with
SIGKILL keeps the page cache, so a missing sync cannot change what any test reads back, and
the order of the steps is already what the injection varies. The mutant would pass either
way, so it is not built.

## Tests

The store and identity layer: `tests/codec/test_checkpoint_store.py`, 49 tests, unit tier, 1.3 s.

Medium, `tests/core/test_resume.py`, four host tests on one study, 22.8 s cold in total (the
host loop compiles once, in a module fixture that also builds the uninterrupted reference and
the split checkpoint the tests share):

- host resume is the uninterrupted run (trace and carry bit equal), and the manifest step
  count equals the rows after a resume;
- a held refresh reports `TRUNCATED_AT_HOLD`, is answered once on resume, finishes as in the
  uninterrupted run, and a run with no sink drops the same hold and never reports the status;
- a pending request is answered first on resume;
- a changed truth, study, key version, boundary, kept log or roots is refused (C002), a used
  directory without `resume` is refused (C008), and `resume` on an empty directory is
  refused (C010).

Slow, `tests/core/test_resume_compiled.py`, four tests, about 32 s cold (the chunked driver
compiles once for a batch of two worlds):

- compiled resume is the chunked run (carry bit equal, canonical history exact);
- a compiled hold is reported per world and answered on resume;
- a pending request is answered before the first chunk of a resume (that chunk steps);
- another chunk size is refused (C002), and a bucket beside a checkpoint is refused.

Slow, `tests/core/test_resume_kill.py`, three cases: a child process killed by SIGKILL while
writing a chunk, a manifest and the pointer rename, each resumed to the bit-equal
uninterrupted run. About 47 s.

Runner time: the slow tier adds about 79 s (the kill file and the compiled file). The compiled
cases are slow because the chunked driver for a batch costs about 25 s to compile cold, which
no sharing between tests removes.

## Checks at merge

- `tests/core/test_golden_histories.py -m ""` passes with no golden, fingerprint, fault
  fixture or corpus document hash moved.
- `RunStatus.TRUNCATED_AT_HOLD` is never emitted without a sink, so no stored value moves.
- Sibling sites: `HostExecutor.run`, `HostExecutor.run_worlds`, `CompiledExecutor.run`, the
  chunked driver, `answer_refresh`, `answer_refreshes` and `RunResult.status`.
