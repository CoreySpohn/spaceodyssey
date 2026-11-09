---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# How to save and load a study

```{admonition} The task
Write a study to a JSON document, read it back, and be sure the copy runs exactly like
the original.
```

A study held only in memory cannot be shared, archived or compared. The codec turns a
study into a canonical JSON document and rebuilds an equal study from it. This guide
saves the toy study with a custom channel in it, loads it, and checks the copy.

```{code-cell} python
import json
import tempfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from spaceodyssey import codec, plugins
from spaceodyssey.core import (
    AbstractChannelModel,
    Capacities,
    HostExecutor,
    histories_equal,
)
from spaceodyssey.studies import toy

caps = Capacities(n_subjects=2, n_an=2, n_assets=1, n_lanes=1, n_channels=1)
study = toy.survey_1(caps=caps)
truth = toy.ToyTruth(signal=jnp.array([1.0, 1.2]))


class ShotNoisePhotometry(AbstractChannelModel):
    """Measured value with noise proportional to the square root of the signal."""

    noise: jax.Array

    def realize(self, truth, subject, channel_index, fraction, key, design, channels):
        """Return the measured value and the standard normal variate behind it."""
        z = jr.normal(key, dtype=jnp.float64)
        signal = design.gain * truth.signal[subject]
        return signal + self.noise[channel_index] * jnp.sqrt(signal) * z, z


custom = eqx.tree_at(
    lambda s: s.channel_model, study, ShotNoisePhotometry(noise=jnp.array([0.5]))
)
```

The channel is the one from [how to write a channel](channel.md), shortened to the
part the document needs.

## Every class in a document is a registered kind

A document never imports anything it names. It refers to each class by a kind
reference, the triple (family, id, version), and a resolver maps the triple to a class.
The family is the component type, such as `channel_model` or `policy`. The core's own
classes, such as `Study` and `Capacities`, resolve without any help.

The toy classes are registered as entry points of the installed package, and
`plugins.installed_kinds()` collects every installed registration into a mapping. A
class of your own is added to that mapping under a name you choose.

```{code-cell} python
kinds = {
    **plugins.installed_kinds(),
    ("channel_model", "docs.shot_noise", 1): ShotNoisePhotometry,
}
document = codec.dump_study(custom, kinds)
text = document.to_json()
print(len(text), "characters")
print(json.dumps(json.loads(text)["study"]["channel_model"], indent=1))
```

The document stores every field with its exact type and array dtype, so the copy is an
equal pytree rather than an approximation.

## A study whose kinds are not named cannot be saved

`dump_study` refuses a class it cannot name, and the error points into the study
where it found one. With no `kinds` argument only the core classes resolve, so the first class outside
the core, here the custom channel, is refused.

```{code-cell} python
try:
    codec.dump_study(custom)
except codec.DocumentError as error:
    print(error)
```

## The content hash identifies the study

The document carries the SHA-256 hash of its canonical bytes. Two dumps of one study
have the same hash, and a change of any field changes it. Quoting the hash in a paper
or a log pins the exact study that produced a result.

```{code-cell} python
again = codec.dump_study(custom, kinds)
louder = eqx.tree_at(
    lambda s: s.channel_model, custom, ShotNoisePhotometry(noise=jnp.array([0.6]))
)
print(document.content_hash == again.content_hash)
print(document.content_hash == codec.dump_study(louder, kinds).content_hash)
```

## A document is text, so it goes in a file

`to_json` returns the canonical text, and `load_study` takes that text back. The loader
checks the format, the schema version and the hash, and a document edited by hand after
it was written fails the hash check.

```{code-cell} python
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "study.json"
    path.write_text(text)
    loaded = codec.load_study(path.read_text(), kinds)
print(type(loaded.channel_model).__name__, float(loaded.channel_model.noise[0]))
```

## The loaded study runs exactly like the original

The check that counts is the one on behavior. Both studies run the same world with the
same key, and their canonical histories must agree: every integer equal and every float
within 1e-12.

```{code-cell} python
key = jr.PRNGKey(0)
original_run = HostExecutor().run(custom, truth, key, 40)
loaded_run = HostExecutor().run(loaded, truth, key, 40)
print(histories_equal(original_run.history(), loaded_run.history()).go)
print(jax.tree_util.tree_structure(custom) == jax.tree_util.tree_structure(loaded))
```

## A document can be checked without loading it

`preflight_document` reads a document and applies the same structural checks as
`preflight`, plus checks of the document itself. It never raises on document content,
and an empty list means no finding. A document that names a kind the resolver does not
hold is reported at its location, which is what a reader without your class sees.

```{code-cell} python
print(codec.preflight_document(text, kinds))
try:
    codec.load_study(text, plugins.installed_kinds())
except codec.DocumentError as error:
    print(error)
```

## Registering a kind lets others load your documents

A mapping works in a notebook, but another reader needs your class to be discoverable.
A package registers each kind as an entry point in the group `spaceodyssey.<family>`.
The entry name is `<id>@<version>`, and the version must equal the default of the
class's `version` field.

```toml
[project.entry-points."spaceodyssey.channel_model"]
"docs.shot_noise@1" = "mypackage.channels:ShotNoisePhotometry"
```

When a class changes its fields, raise its `version` and add a `migrate(from_version,
params)` classmethod that returns the fields of the new version, so that documents
written before the change still load. Arrays larger than a set size are written to
separate files when `dump_study` is given a `files` directory, and `load_study` takes
the same directory.
