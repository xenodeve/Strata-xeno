"""serve/mip/core.py - what an invocation says and what comes back: the typed envelope of milestone M1 (Roadmap 5.1).

An adapter takes a `MIPInvocation` and gives a `MIPResult`; the work is the adapter's, these two only say what was asked and
what came of it.  The fields this version has nothing to put in (no model binding, no accepted prefix, no value references)
stay empty, so the form does not change when something fills them.

The statuses, for every adapter:
    success      it ran, and this is its result
    refused      it could have run and did not: the arguments, a permission, a budget
    unsupported  nothing here has that name
    cancelled    the request was stopped
    fault        it ran or tried to, and failed: the tool's own error, a timeout, a server that stopped
"""
from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field

PROTOCOL_VERSION = "mip/0.1"
TARGET_KINDS = ("native", "mcp")
STATUSES = ("success", "refused", "unsupported", "cancelled", "fault")


class _Json:
    """The JSON form: a dict of plain values, and back."""

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: dict):
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class MIPInvocation(_Json):
    target_kind: str                               # native | mcp
    capability_id: str                             # for MCP, as the program writes it: <server>.<tool>
    typed_arguments: dict = field(default_factory=dict)
    operation_id: str | None = None
    invocation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    protocol_version: str = PROTOCOL_VERSION
    model_binding: str | None = None
    adapter_binding: str | None = None
    conversation_id: str | None = None
    accepted_prefix_identity: str | None = None
    logical_position: int | None = None
    ownership: str | None = None
    value_references: dict = field(default_factory=dict)
    argument_provenance: str | None = None         # who wrote the arguments
    execution_budget: dict = field(default_factory=dict)
    authorization_context: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.target_kind not in TARGET_KINDS:
            raise ValueError(f"target_kind {self.target_kind!r}: expected one of {TARGET_KINDS}")


@dataclass
class MIPResult(_Json):
    invocation_id: str
    status: str                                    # success | refused | unsupported | cancelled | fault
    typed_content: object = None                   # the value on success, the reason in words otherwise
    content_reference: str | None = None
    result_provenance: dict = field(default_factory=dict)
    identity: str | None = None
    visibility: str = "program"                    # who sees it: the program, and the model only through what it prints
    feedback_strategy: str = "selective_text"      # Roadmap 6.3; the only one of its four this version has
    commit_status: str = "none"                    # nothing here takes part in a commit: every call only reads

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status {self.status!r}: expected one of {STATUSES}")
