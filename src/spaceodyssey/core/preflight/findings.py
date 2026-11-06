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
    T002 = "T002"  # a channel with no valid coincident order
    R001 = "R001"  # a channel declares a record kind the registry lacks


@dataclasses.dataclass(frozen=True)
class Finding:
    """One structural finding: a code, where it is, what it says and how to fix it.

    `path` is for people. The locator is for tools: `subject` names the slot of the
    study the finding is about (`observer`, `policy`, `channel_model`, `channels`,
    `assets`, and so on), `index` the position within a slot that holds a sequence
    (the policy), `member` the field of that slot, and `entry` the position of the
    offending item within the member (one integer per axis). Each is empty when the
    finding does not narrow that far, and a finding never names an item it did not
    check.
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
