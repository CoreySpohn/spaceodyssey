"""Findings."""

import dataclasses
from enum import StrEnum


class Severity(StrEnum):
    """How a finding is read: an error blocks the study, a warning is reported."""

    ERROR = "error"
    WARNING = "warning"


class FindingCode(StrEnum):
    """Stable identifiers of the structural checks."""

    F001 = "F001"  # an observer reads truth outside a measure crossing
    F002 = "F002"  # an observer reads a scorer
    C001 = "C001"  # a decision reads a product not yet released to its audience
    K001 = "K001"  # draws without stable-id coordinates
    K002 = "K002"  # two measures with identical key coordinates
    X001 = "X001"  # a capability the selected executors do not provide
    X002 = "X002"  # a reserved field set to a non-default value
    T001 = "T001"  # a drain that equals a close below campaign scope
    P001 = "P001"  # a policy names a program the capacities do not hold
    W002 = "W002"  # a declared failure kind the policy cannot read
    T002 = "T002"  # a channel with no valid coincident order
    R001 = "R001"  # a channel declares a record kind the registry lacks
    D004 = (
        "D004"  # a declared upstream hash differs from the supplied upstream document
    )
    D005 = "D005"  # an in-table pedigree entry names a source the table lacks
    D006 = "D006"  # a table row the study cannot hold: repeated identity or index
    D007 = "D007"  # a row offset above zero, or a start after its end
    D008 = "D008"  # two retained priors of one audience with intersecting pedigrees
    D009 = "D009"  # audiences with different priors on a boundary that shares them
    D010 = "D010"  # a table prior with a refit boundary that does not consume it


@dataclasses.dataclass(frozen=True)
class Finding:
    """One structural finding: a code, where it is, what it says and how to fix it.

    `path` is for people. The locator is for tools: `subject` names the slot of the
    study the finding is about (`observer`, `policy`, `channel_model`, `channels`,
    `assets`, and so on), `index` the position within a slot that holds a sequence
    (the policy), `member` the field of that slot (a dotted path, such as
    `table.local_index`, for a field of a module the slot holds), and `entry` the
    position of the offending item within the member (one integer per axis). Each is
    empty when the finding does not narrow that far, and a finding never names an item
    it did not check.
    """

    code: FindingCode
    severity: Severity
    path: str
    message: str
    fix: str
    subject: str = ""
    index: int | None = None
    member: str = ""
    entry: tuple[int, ...] = ()

    def __str__(self) -> str:
        """One line plus the fix hint."""
        return (
            f"[{self.code}] {self.severity} at {self.path}: {self.message}\n"
            f"        fix: {self.fix}"
        )
