"""LifeMate public API and single-session pipeline coordinator.

Implementations live in lifemate.pipeline. Historical imports and the visible
load_toolenv hook remain here; heavy Torch/Transformers imports stay lazy.
"""
from __future__ import annotations

from copy import deepcopy
import time

from lifemate.paths import default_data_root, agent_output_dir, asr_prediction_file
from lifemate import workflow
from lifemate.pipeline import actor, adaptive, controller, execution, messages, protocol, results, runner
from lifemate.pipeline.state import ControllerState
from lifemate.pipeline.config import (
    AgentConfig, AgentInputError, ContextLimitError, ModelLoadConfig, ModelOutputError, ModelReply,
)
from lifemate.pipeline.inputs import (
    discover_data_root, get_tool_env_factory, load_asr, load_catalog, load_speech_file,
    load_split_items, prepare_audio_input, prepare_request, speech_from_train, validate_speech,
)
from lifemate.pipeline.tools import (
    _description_without_examples, _strings, prompt_schema, qwen_tools,
    reference_lookup_hints, select_available_tools, unverified_references,
)
from lifemate.pipeline.messages import SYSTEM_PROMPT, assistant_call_message, build_messages, tool_result_message
from lifemate.pipeline.replies import parse_model_reply
from lifemate.pipeline.backend import HFBackend, load_qwen_backend
from lifemate.pipeline.results import write_submission


class AgentSession:
    """Own one live session and dispatch planning, execution and bounded recovery.

    Stage handlers receive this context explicitly. All mutable state, budgets,
    model access and the single ToolEnv belong to this instance.
    """

    def __init__(self, request, speech, available_tools, env, backend, config=None, clock=None):
        validate_speech(request, speech)
        if env.get_tool_calls():
            raise AgentInputError("Session mới cần env chưa có call. Để tiếp tục, dùng lại AgentSession cũ.")
        self.request = deepcopy(request)
        self.speech = deepcopy(speech)
        self.tools = {tool["name"]: deepcopy(tool) for tool in available_tools}
        declared = {n for app in request["context"]["installed_apps"].values() for n in app["tools"]}
        declared.add("sys.ask_missing_info")
        if set(self.tools) != declared or len(self.tools) != len(available_tools):
            raise AgentInputError("available_tools không khớp danh sách được phép của mẫu")
        self.native_tools = qwen_tools(available_tools)
        self.tool_directory = workflow.directory([
            {**tool, "description": _description_without_examples(tool.get("description", ""))}
            for tool in available_tools])
        self.messages = build_messages(request, speech)
        self.initial_messages = deepcopy(self.messages)
        self.env, self.backend = env, backend
        self.config = config or AgentConfig()
        self.clock = clock or time.perf_counter
        self.status = "running"
        self.pending = None
        self.trace = []
        self.first_model_input = None
        self.started = None
        self.model_calls = 0
        self.invalid_replies = 0
        self.consecutive_invalid_replies = 0
        self._expected_log_count = 0
        self.known_strings = set(_strings(request["context"])) | set(_strings(request["history"]))
        self.utterances = [speech["current_text"]] + [t["text"] for t in speech["history_user_texts"]]
        self.tasks = None
        self.active_names = set(self.tools)
        self.observations = {}
        self.evidence = {"request": {"source": "request", "text": speech["current_text"]},
                         "context": {"source": "context", "value": deepcopy(request["context"])}}
        for i, turn in enumerate(request["history"]):
            self.evidence[f"history:{i}"] = {"source": "history", "value": deepcopy(turn)}
        for turn in speech["history_user_texts"]:
            self.evidence[f"history:{turn['history_index']}"]["text"] = turn["text"]
        self.review_context = None
        self.control_feedback = ""
        self.control_bad_reply = None
        self.control_failures = 0
        self.no_progress = {}
        self.call_state = (protocol.CallState() if self.config.workflow_enabled and self.config.unified_protocol else None)
        self.controller_state = (ControllerState.from_input(request, speech)
                                 if self.config.workflow_enabled and not self.config.unified_protocol
                                 and self.config.adaptive_controller else None)

    def _routed_tools(self):
        if self.call_state is not None:
            return deepcopy(self.native_tools)
        return [t for t in self.native_tools if t["function"]["name"] in self.active_names]

    def _set_route(self, names):
        selected = workflow.expand_route(names, self.tools,
            lambda schema: reference_lookup_hints(list(self.tools.values()), schema))
        self.active_names.update(selected)

    def _generate_reply(self, phase, messages, tools, control=False, output_tokens=None):
        if self.model_calls >= self.config.max_steps:
            self.status = "step_limit"
            return None
        self.model_calls += 1
        if self.call_state is not None and self.first_model_input is None:
            # Capture exactly what is passed to the backend, not just the base
            # conversation before protocol/grounding guidance is added.
            import hashlib
            import json
            payload = {"messages": deepcopy(messages), "tools": deepcopy(tools)}
            self.first_model_input = {**payload, "phase": phase, "sha256": hashlib.sha256(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()}
        config = self.config
        if control:
            from dataclasses import replace
            config = replace(config, max_new_tokens=config.control_max_new_tokens)
        if output_tokens is not None:
            from dataclasses import replace
            config = replace(config, max_new_tokens=min(output_tokens, config.max_context_tokens - 1))
        try:
            reply = self.backend.generate(messages, deepcopy(tools), config,
                                          remaining_seconds=self._remaining())
        except (ContextLimitError, TimeoutError) as error:
            self.status = "context_limit" if isinstance(error, ContextLimitError) else "timeout"
            self.trace.append({"type": "error", "phase": phase, "error": str(error)})
            return None
        except Exception as error:
            self.status = "model_error"
            self.trace.append({"type": "error", "phase": phase,
                               "error": f"{type(error).__name__}: {error}"})
            return None
        self.trace.append({"type": "control_reply" if control else "model_reply", "phase": phase,
                           "raw": reply.text, "input_tokens": reply.input_tokens,
                           "output_tokens": reply.output_tokens, "generation_seconds": reply.elapsed_seconds,
                           "truncated": reply.truncated})
        return reply if self._budget_ok() else None

    def _start(self):
        if self.started is None:
            self.started = self.clock()

    def elapsed(self):
        return 0.0 if self.started is None else self.clock() - self.started

    def _remaining(self):
        if self.config.max_seconds is None:
            return None
        return self.config.max_seconds - self.elapsed()

    def _budget_ok(self):
        remaining = self._remaining()
        if remaining is not None and remaining <= 0:
            self.status = "timeout"
            self.pending = None
            return False
        return True

    def _check_log(self):
        if len(self.env.get_tool_calls()) != self._expected_log_count:
            raise AgentInputError("Env bị gọi bên ngoài session; dừng để tránh lặp hành động")

    def propose(self):
        self._start()
        self._check_log()
        if self.status != "running":
            return {"type": "stopped", "status": self.status}
        if not self._budget_ok():
            return {"type": "stopped", "status": self.status}
        if self.pending is not None:
            return deepcopy(self.pending)  # rerunning preview never generates/executes twice
        if self.call_state is not None:
            if self.config.task_reasoning_enabled and not self.call_state.planning_attempted:
                from lifemate.pipeline.task_reasoning import draft
                return draft(self)
            return actor.propose_action(self)
        if self.controller_state is not None:
            return adaptive.propose(self)
        if self.config.workflow_enabled:
            if self.tasks is None:
                return self._plan()
            if self.review_context is not None:
                return self._review()
        return actor.propose_action(self)

    def step(self):
        proposal = self.propose()
        if proposal["type"] == "call":
            return self.execute_pending()
        return proposal

    def run(self):
        while self.status == "running":
            self.step()
        return self.result()

    def _workflow_messages(self, instruction=None, payload=None):
        if self.call_state is not None:
            return protocol.actor_messages(self)
        return messages.build_workflow_messages(self, instruction, payload)

    def _control_reject(self, error, raw=None, phase=None):
        return controller.control_reject(self, error, raw, phase)

    def _plan(self):
        return controller.plan(self)

    def _review(self):
        return controller.review(self)

    def _reject(self, raw, message, call=None, lookup_hints=None, content="", issue=None, details=None):
        return actor.reject_reply(self, raw, message, call, lookup_hints, content, issue=issue, details=details)

    def execute_pending(self):
        return execution.execute_pending(self)

    def result(self):
        return results.session_result(self)


def run_agent(request, speech, available_tools, env, *, backend, config=None):
    """Integration entry point. Runner creates env once, then calls this once."""
    return AgentSession(request, speech, available_tools, env, backend, config).run()


def load_toolenv(data_root, split):
    """trả về hàm make_env(sample_id), chưa tạo env ở đây.

    Runner gọi make_env đúng một lần cho mỗi mẫu. Môi trường trả về cần có:
    validate(name, arguments), call(name, arguments), get_tool_calls(),
    get_state_hash(). Không gọi get_state(), không mở *_environment trực tiếp.
    """
    # Hook tích hợp BTC: chỉ nạp thư viện ở đây; runner tạo env theo từng mẫu.
    ToolEnv = get_tool_env_factory(data_root, split)

    def make_env(sample_id):
        return ToolEnv(split, sample_id, root=data_root)

    return make_env


def run_items(items, catalog, speech_by_id, env_factory, backend, config=None):
    return runner.run_items(items, catalog, speech_by_id, env_factory, backend, config,
                            session_factory=AgentSession)


def run_full_pipeline(data_root, split="train", *, sample_ids=None, limit=None,
                      output_dir=None, model_config=None, agent_config=None,
                      asr_loader=None, toolenv_loader=None, backend=None, asr_predictions_path=None):
    """Public runner; resolve integration hooks here so existing launchers keep working."""
    return runner.run_full_pipeline(
        data_root, split, sample_ids=sample_ids, limit=limit, output_dir=output_dir,
        model_config=model_config, agent_config=agent_config, backend=backend,
        asr_predictions_path=asr_predictions_path,
        asr_loader=asr_loader if asr_loader is not None else
            lambda root, selected_split: load_asr(root, selected_split, asr_predictions_path),
        toolenv_loader=toolenv_loader if toolenv_loader is not None else load_toolenv,
        session_factory=AgentSession, backend_loader=load_qwen_backend,
    )


def main():
    """Chạy toàn bộ split; sửa cấu hình trực tiếp hoặc dùng notebook launcher."""
    # DATA_ROOT chứa train/, public_test/ (KHÔNG thêm /train vào cuối).
    # Local: C:/zalo_ai/train+public_test
    # Kaggle: /kaggle/input/datasets/khangl123/zalo-ai/train+public_test
    # Có thể gán Path("...") trực tiếp nếu bộ dữ liệu nằm ở chỗ khác.
    DATA_ROOT = default_data_root()
    SPLIT = "train"

    # Model ID dưới đây chỉ dùng được offline nếu checkpoint/tokenizer đã cache.
    # Nếu dùng thư mục model: gán đường dẫn thật tới nơi có config.json + weights
    # và tokenizer; local dùng ổ đĩa máy, Kaggle dùng thư mục model đã Add Input.
    MODEL_SOURCE = "Qwen/Qwen3-4B-Instruct-2507"

    # File text ASR có sẵn, cùng SPLIT. Nếu lấy từ máy/notebook khác, sửa ASR_FILE.
    # Local: <repo>/output/lifemate/asr/<split>/asr_predictions.jsonl
    # Kaggle: /kaggle/working/lifemate/asr/<split>/asr_predictions.jsonl
    ASR_FILE = asr_prediction_file(SPLIT)
    # Local: <repo>/output/lifemate/agent; Kaggle: /kaggle/working/lifemate/agent.
    # Mỗi lần chạy có thư mục <run_id>/ riêng; không ghi kết quả vào data input.
    AGENT_OUTPUT = agent_output_dir()

    print("Dữ liệu:", DATA_ROOT)
    print("Input ASR:", ASR_FILE)
    print("Model agent:", MODEL_SOURCE)
    print("Output agent:", AGENT_OUTPUT)
    result = run_full_pipeline(
        data_root=DATA_ROOT,
        split=SPLIT,
        asr_predictions_path=ASR_FILE,
        output_dir=AGENT_OUTPUT,
        model_config=ModelLoadConfig(model_source=MODEL_SOURCE, local_files_only=True),
    )
    print("JSONL:", result["submission_path"])
    print("Trace:", result["trace_dir"])


if __name__ == "__main__":
    main()
