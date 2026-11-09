# Architecture

```{note}
This page gives the overview and the document form of a study.
The implemented `spaceodyssey.core` package has its own [transaction core guide](core.md),
including the execution layers, study composition and a runnable example.
```

## What spaceodyssey is

spaceodyssey is a closed-loop observing-campaign simulator for exoplanet
direct-imaging missions, built for the Habitable Worlds Observatory concept. It
plays out a campaign against simulated worlds (schedule, observe, update beliefs)
and records time accounting and the released observations.

Classical yield codes such as AYO and EXOSIMS score targets with precomputed
completeness curves and fixed revisit heuristics. spaceodyssey instead carries
inference in the loop: each target holds a belief that every observation
updates, and scheduling policies act on those beliefs.

spaceodyssey owns no physics, no inference, and no policy logic of its own. It is the
composition layer for the HWO simulation suite:

| library | role in spaceodyssey |
| --- | --- |
| skyscapes | astrophysical scenes, target catalogs, population synthesis |
| orbix | orbital propagation, observatory geometry, keepout |
| jaxedith | count rates and integration times |
| photomancy | posteriors, evidence, expected information gain |
| planit-py | scheduling policies and the decision vocabulary |
| yippy / optixstuff / hwoutils | coronagraph performance data, hardware primitives, shared math |

![Suite dependency direction](diagrams/library-stack.svg)

## The environment, the agent, and the belief

The mission loop is a partially observed decision process split across three
libraries. spaceodyssey is the environment: it owns the simulated truth, the clock, the
observation model, and the bookkeeping. planit-py is the agent: a stateless policy
that proposes the next action from the current belief and observing context.
photomancy is the belief update both share.

A strict firewall separates truth from belief. The simulator holds both the drawn
universe and the agent's belief about it. A policy receives a masked snapshot of
its own audience's records and no truth argument, and generated checks swap the
truth to confirm that a decision does not change (see [Transaction core](core.md)).

## Documents

A study is written as a JSON document by `spaceodyssey.codec.dump_study` and read back
by `spaceodyssey.codec.load_study`. Every class a document names is a registered
kind with an integer version; `spaceodyssey.codec.KindResolver` maps the identifier
and version to a class and never imports anything the document names. The package
registers its own kinds, and an external package registers more through entry
points in the groups `spaceodyssey.<family>` (see `spaceodyssey.plugins`).
A kind that changes its fields bumps its version and supplies a `migrate` method
for older documents. Random-number streams are keyed by semantic coordinates rather
than by call order, which keeps paired runs aligned even after adaptive trajectories
diverge.

## Validation

The conformance corpus and its golden histories are the standing test of the
core: every case runs on both executors and reproduces its recorded history.
The generated battery checks null credit, truth swaps, release latency, forks and
epoch shifts on actual runs.

## Extending spaceodyssey

Adding a capability is meant to be a bounded, local change:

1. A new world source, channel model, observer, policy or objective: implement the
   kind's abstract class, give it an integer `version`, and register it under an
   entry point in the family's group. The family's kit runs on each registered kind.
2. A heavyweight dependency: add it behind an optional extra, with its adapter
   module importing lazily.
3. A genuinely new kind family: a deliberate API event with its own abstract class
   and entry-point group.
