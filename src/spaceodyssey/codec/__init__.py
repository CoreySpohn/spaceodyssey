"""Study documents: canonical JSON of a study, its content hash, and loading.

The codec lives outside the core and the core imports none of it. `dump_study`
walks a study's fields generically; `load_study` rebuilds an equal study, which
reuses the programs the original compiled.
"""

from spaceodyssey.codec.document import CORE_KINDS as CORE_KINDS
from spaceodyssey.codec.document import FORMAT as FORMAT
from spaceodyssey.codec.document import SCHEMA_VERSION as SCHEMA_VERSION
from spaceodyssey.codec.document import DocumentError as DocumentError
from spaceodyssey.codec.document import KindTable as KindTable
from spaceodyssey.codec.document import StudyDocument as StudyDocument
from spaceodyssey.codec.document import canonical_json as canonical_json
from spaceodyssey.codec.dump import dump_study as dump_study
from spaceodyssey.codec.load import load_study as load_study
