"""Generated checks that run a study and report whether a property holds.

Each check takes a study (and the worlds to run it on) and returns a `CheckResult`,
truthy when the property holds, with a diagnostic when it does not. The checks:

- `null_reference`: a world with nothing in it credits nothing;
- `truth_swap`: nothing the policy does depends on truth it has not been given;
- `latency_shift`: a release delay moves belief updates and nothing else;
- `fork_check`: a fork shares its prefix and later draws only as declared;
- `relabel_ids`, `shift_epoch`, `zero_time_asset`: metamorphic relations;
- `declared_matches_run`: the stops and needs components declare match a run.
"""

from spaceodyssey.core.battery.declarations import (
    declared_matches_run as declared_matches_run,
)
from spaceodyssey.core.battery.fork import fork_check as fork_check
from spaceodyssey.core.battery.fork import forked as forked
from spaceodyssey.core.battery.fork import key_report as key_report
from spaceodyssey.core.battery.metamorphic import relabel_ids as relabel_ids
from spaceodyssey.core.battery.metamorphic import shift_epoch as shift_epoch
from spaceodyssey.core.battery.metamorphic import zero_time_asset as zero_time_asset
from spaceodyssey.core.battery.null_reference import (
    false_alarm_interval as false_alarm_interval,
)
from spaceodyssey.core.battery.null_reference import null_reference as null_reference
from spaceodyssey.core.battery.replay_latency import latency_shift as latency_shift
from spaceodyssey.core.battery.result import CheckResult as CheckResult
from spaceodyssey.core.battery.truth_swap import truth_swap as truth_swap
