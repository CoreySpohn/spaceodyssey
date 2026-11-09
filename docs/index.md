# spaceodyssey

Astronomical analysis and observing campaigns.

spaceodyssey runs multi-visit observing programs with inference in the loop.
An analysis session fits the observations already in hand for a target. A
campaign adds a time horizon, an observatory, a policy for choosing the next
visit, and an execution model that turns admitted visits into new records. An
ensemble repeats a campaign over many simulated worlds and aggregates the
outcomes. The simulated world and the model used to reason about it are
separate by construction.

Pre-alpha. The campaign core provides a declarative vocabulary, a document form
for studies, and host and compiled executors; physics and inference enter
through adapters.

```{toctree}
:caption: Getting started
:maxdepth: 1

getting-started
```

```{toctree}
:caption: Many worlds
:maxdepth: 1

many-worlds
```

```{toctree}
:caption: Concepts
:maxdepth: 1

concepts/truth-and-view
concepts/events-and-admission
concepts/records-and-release
concepts/ledger
```

```{toctree}
:caption: How-to guides
:maxdepth: 1

how-to/channel
how-to/policy
how-to/observer
how-to/save-and-load
```

```{toctree}
:caption: Reference
:maxdepth: 1

architecture
core
executors
records
autoapi/index
```
