"""Agent configuration, reply contract and shared errors; no model imports."""
from __future__ import annotations

from dataclasses import dataclass
import re


class AgentInputError(ValueError):
    pass


class ModelOutputError(ValueError):
    pass


class ContextLimitError(ValueError):
    pass


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 48  # Total model generations; tool-call cap stays separate.
    max_calls: int = 12
    max_invalid_replies: int = 2  # Unified: consecutive rejected proposals; historical v3: per phase.
    max_context_tokens: int = 16384
    max_new_tokens: int = 256
    prefill_chunk_size: int = 512
    max_seconds: float | None = None
    check_reference_provenance: bool = True
    workflow_enabled: bool = True
    unified_protocol: bool = True  # Default: BTC calls only; False opts into historical controllers.
    grounding_enabled: bool = True  # Unified protocol: source facts and conservative pre-call checks.
    terminal_check_enabled: bool = True  # Same BTC output; check final/ask candidates only.
    max_terminal_checks: int = 2  # Total per sample, also charged to max_steps/max_seconds.
    task_reasoning_enabled: bool = True  # One text brief + bounded same-grammar write checks.
    planning_max_new_tokens: int = 512
    review_finish: bool = True  # Review/routing flags below apply to historical controllers.
    review_writes: bool = True
    route_tools: bool = True
    include_examples: bool = True
    control_max_new_tokens: int = 1536
    max_tasks: int = 12
    max_no_progress: int = 3
    adaptive_controller: bool = True
    max_plan_revisions: int = 3
    max_interpretation_revisions: int = 2
    max_action_repairs: int = 3

    def __post_init__(self):
        for key in ("max_steps", "max_calls", "max_invalid_replies", "max_new_tokens", "prefill_chunk_size",
                    "control_max_new_tokens", "max_tasks", "max_no_progress",
                    "max_plan_revisions", "max_interpretation_revisions", "max_action_repairs", "max_terminal_checks",
                    "planning_max_new_tokens"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} phải >= 1")
        if not 1 <= self.max_context_tokens <= 16384:
            raise ValueError("Ngân sách context phải nằm trong 1..16384")
        if self.max_new_tokens >= self.max_context_tokens:
            raise ValueError("Cần dành token cho input")
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise ValueError("max_seconds phải > 0 hoặc None")
        if self.workflow_enabled and not self.unified_protocol and self.control_max_new_tokens >= self.max_context_tokens:
            raise ValueError("Control output must leave room for input")


@dataclass
class ModelReply:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_seconds: float | None = None
    truncated: bool = False


@dataclass(frozen=True)
class ModelLoadConfig:
    """Nạp một Qwen cho cả phiên; mặc định chỉ dùng file/cache local."""
    model_source: str = "Qwen/Qwen3-4B-Instruct-2507"
    device: str = "cuda:0"
    local_files_only: bool = True
    revision: str | None = None

    def __post_init__(self):
        if not isinstance(self.model_source, str) or not self.model_source.strip():
            raise ValueError("model_source phải là model ID hoặc thư mục model local")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("revision phải là tên phiên bản/commit không rỗng hoặc None")
        if not re.fullmatch(r"cuda:\d+", self.device):
            raise ValueError("Bản FP16 này cần device dạng cuda:0 hoặc cuda:1")
