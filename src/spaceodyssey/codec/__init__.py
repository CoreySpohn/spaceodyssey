"""Study documents: canonical JSON of a study, its content hash, and loading.

The codec lives outside the core and the core imports none of it. `dump_study`
walks a study's fields generically; `load_study` rebuilds an equal study, which
reuses the programs the original compiled. Both take the kinds as a
`KindResolver` or a mapping of `(group, id, version)` to classes;
`spaceodyssey.plugins.installed_resolver()` builds one from the installed entry
points.

`dump_run` and `load_run` add the run: a `RunDocument` binds the executor, the
`Query` service and the refit boundary by kind, and keeps the record-kind registry.
"""

from spaceodyssey.codec.document import FORMAT as FORMAT
from spaceodyssey.codec.document import RUN_FORMAT as RUN_FORMAT
from spaceodyssey.codec.document import SCHEMA_VERSION as SCHEMA_VERSION
from spaceodyssey.codec.document import CallableRefused as CallableRefused
from spaceodyssey.codec.document import DocumentError as DocumentError
from spaceodyssey.codec.document import RunDocument as RunDocument
from spaceodyssey.codec.document import StudyDocument as StudyDocument
from spaceodyssey.codec.document import canonical_json as canonical_json
from spaceodyssey.codec.dump import dump_study as dump_study
from spaceodyssey.codec.kinds import CORE_KINDS as CORE_KINDS
from spaceodyssey.codec.kinds import KindResolver as KindResolver
from spaceodyssey.codec.load import load_study as load_study
from spaceodyssey.codec.preflight import DocumentFinding as DocumentFinding
from spaceodyssey.codec.preflight import preflight_document as preflight_document
from spaceodyssey.codec.run import BindingKind as BindingKind
from spaceodyssey.codec.run import Bindings as Bindings
from spaceodyssey.codec.run import Run as Run
from spaceodyssey.codec.run import dump_run as dump_run
from spaceodyssey.codec.run import load_run as load_run
