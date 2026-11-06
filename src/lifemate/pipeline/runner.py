"""Batch orchestration with explicit session/model/env factories; no model imports at import time."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import time

from lifemate.paths import agent_output_dir
from .config import AgentConfig, AgentInputError
from .inputs import load_asr, load_catalog, load_split_items, prepare_audio_input, prepare_request, validate_speech
from .results import write_submission
from .tools import select_available_tools


def run_items(items, catalog, speech_by_id, env_factory, backend, config=None, *, session_factory):
    """Small batch runner; exactly one factory call per unique sample.

    Caller owns a fresh independent inference run. No automatic reset/retry on
    a sample. No precomputed gold transcript fallback for missing ASR rows.
    """
    items = list(items)
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise AgentInputError("Batch có ID trùng")
    prepared = []
    for item in items:
        request = prepare_request(item)
        if item["id"] not in speech_by_id:
            raise AgentInputError(f"Thiếu output ASR: {item['id']}")
        speech = speech_by_id[item["id"]]
        validate_speech(request, speech)
        prepared.append((request, speech, select_available_tools(request, catalog)))
    results = []
    for request, speech, tools in prepared:
        env = env_factory(request["id"])
        result = session_factory(request, speech, tools, env, backend, config).run()
        results.append(result)
        if result["status"] in ("model_error", "execution_error"):
            raise RuntimeError(f"Dừng batch tại {request['id']}: {result['status']}; không ghi submission đầy đủ")
    return results


def run_full_pipeline(data_root, split="train", *, sample_ids=None, limit=None,
                      output_dir=None, model_config=None, agent_config=None,
                      asr_loader=None, toolenv_loader=None, backend=None, asr_predictions_path=None,
                      session_factory, backend_loader):
    """Đọc data/text ASR → nạp Qwen một lần → một env/mẫu → agent → JSONL.

    data_root: thư mục cha chứa các split; asr_predictions_path: file text đã có.
    output_dir: thư mục ghi được; runner tự tạo <run_id>/ và traces/ bên trong.
    """
    import uuid
    from datetime import datetime

    if sample_ids is not None:
        sample_ids = list(sample_ids)
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise AgentInputError("sample_ids phải khác rỗng và không trùng")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise AgentInputError("limit phải là số nguyên dương hoặc None để chạy toàn bộ")
    root = Path(data_root).resolve()
    items = load_split_items(root, split)
    catalog = load_catalog(root, split)
    if sample_ids is not None:
        indexed = {row["id"]: row for row in items}
        missing = set(sample_ids) - set(indexed)
        if missing:
            raise AgentInputError("Không tìm thấy ID: " + ", ".join(sorted(missing)))
        selected = [indexed[sample_id] for sample_id in sample_ids]
    else:
        selected = items
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise AgentInputError("Không có mẫu để chạy")
    # Kiểm tra metadata trước khi nạp model hoặc tạo ToolEnv.
    prepared = [(row, prepare_request(row), prepare_audio_input(row, root, split),
                 select_available_tools(prepare_request(row), catalog)) for row in selected]

    # Kiểm tra file text ASR và hook ToolEnv TRƯỚC Qwen: thiếu thì dừng sớm.
    transcribe = (asr_loader(root, split) if asr_loader is not None
                  else load_asr(root, split, asr_predictions_path))
    make_env = toolenv_loader(root, split)
    if not callable(transcribe) or not callable(make_env):
        raise AgentInputError("Hai loader phải trả về hàm transcribe và make_env")
    config = agent_config or AgentConfig()
    if backend is None:
        backend = backend_loader(model_config, config)  # QWEN NẠP MỘT LẦN

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = (Path(output_dir) if output_dir is not None else agent_output_dir()) / run_id
    trace_dir = run_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=False)
    records, statuses = [], {}
    for index, (row, request, audio_input, tools) in enumerate(prepared):
        sample_started = time.perf_counter()
        speech = transcribe(deepcopy(audio_input))  #  cung cấp text ASR thật
        validate_speech(request, speech)          # ASR sai/thiếu: dừng, không dùng gold
        env = make_env(row["id"])                #  đúng MỘT env cho mẫu này
        session = session_factory(request, speech, tools, env, backend, config)
        session.started = sample_started          # ngân sách gồm cả ASR + tạo env
        result = session.run()                    # TOÀN BỘ vòng lặp Qwen ↔ ToolEnv
        # Trace dùng chỉ số, không dùng ID làm path; giữ cả lỗi và calls đã chạy.
        trace_file = trace_dir / f"{index:06d}.json"
        trace_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        statuses[row["id"]] = result["status"]
        print(row["id"], result["status"], "calls:", len(result["submission"]["tool_calls"]))
        for event in result["trace"]:
            if event["type"] == "rejected":
                print("  rejected:", event["error"])
                if event.get("lookup_required"):
                    print("  lookup required:", json.dumps(event["lookup_required"], ensure_ascii=False))
            elif event["type"] == "control_rejected":
                print(f"  control rejected[{event.get('phase', '?')}]:", event["error"])
            elif event["type"] == "recovery":
                issue = event["issue"]
                print(f"  recovery[{event.get('phase', '?')}]:", issue["code"],
                      "step=", issue.get("step_id"), "field=", issue.get("field"),
                      "-", issue["message"])
            elif event["type"] == "recovery_exhausted":
                print("  recovery exhausted:", event.get("budget"), "-", event.get("reason"))
        if result["status"] != "finished":
            print("  trace:", trace_file)
        if result["status"] in ("model_error", "execution_error"):
            raise RuntimeError(f"Dừng tại {row['id']}: {result['status']}. Trace: {trace_dir}")
        records.append(result["submission"])
    complete = {r["id"] for r in records} == {row["id"] for row in items}
    filename = "submission.jsonl" if complete and split != "train" else f"debug_{split}.jsonl"
    submission_path = write_submission(run_dir / filename, records,
                                       expected_ids=[row["id"] for row in selected])
    return {"submission_path": str(submission_path), "trace_dir": str(trace_dir),
            "complete_split": complete, "statuses": statuses, "records": records}
