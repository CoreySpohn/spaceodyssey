# spaceodyssey

Astronomical analysis and observing campaigns.

spaceodyssey runs multi-visit observing programs with inference in the loop.
An analysis session fits the observations already in hand for a target. A
campaign adds a time horizon, an observatory, a policy for choosing the next
visit, and an execution model that turns admitted visits into new records. An
ensemble repeats a campaign over many simulated worlds and aggregates the
outcomes. The simulated world and the model used to reason about it are
separate by construction, so a campaign can be scored against a truth it never
saw.

The first physical profile is exoplanet direct imaging with the Habitable
Worlds Observatory. The physics comes from the libraries that own it (orbits,
scenes, optics, exposure times, post-processing, inference); this package
composes them and keeps the causal record of what was observed, when, and what
was known at the time.

## Status

Pre-alpha. The public interface is not yet published.

## Installation

```bash
pip install spaceodyssey
```

## License

MIT
