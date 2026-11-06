"""Build the self-contained Kaggle Agent -> ToolEnv -> evaluation notebook.

Only source/config is embedded. This command never downloads data/models, installs
packages, executes the notebook, or changes Git. Rebuild after source changes.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import textwrap
import zipfile


REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks/lifemate/test_lifemate.ipynb"


def _cell(kind, source, name):
    cell = {"cell_type": kind, "id": name, "metadata": {"tags": ["lifemate-" + name]},
            "source": textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind == "code":
        compile("".join(cell["source"]), name, "exec")
        cell.update(execution_count=None, outputs=[])
    return cell


def source_snapshot():
    paths = sorted((REPO / "src/lifemate").rglob("*.py"))
    paths.append(REPO / "configs/lifemate/agent-evaluation.json")
    files = {path.relative_to(REPO).as_posix(): path.read_bytes() for path in paths}
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, blob in files.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, blob)
    blob = content.getvalue()
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    return blob, hashes


def notebook():
    blob, hashes = source_snapshot()
    encoded = base64.b64encode(blob).decode("ascii")
    bundle_hash = hashlib.sha256(blob).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            check=True, capture_output=True, text=True).stdout.strip()
    bootstrap = f'''# The uploaded notebook carries this exact source/config snapshot.
import base64
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import zipfile

SOURCE_BASE_COMMIT = {commit!r}
SOURCE_BUNDLE_SHA256 = {bundle_hash!r}
SOURCE_FILES_SHA256 = {hashes!r}
SOURCE_ARCHIVE_BASE64 = {encoded!r}

def install_embedded_source(work_root):
    archive_bytes = base64.b64decode(SOURCE_ARCHIVE_BASE64, validate=True)
    if hashlib.sha256(archive_bytes).hexdigest() != SOURCE_BUNDLE_SHA256:
        raise RuntimeError("Embedded source archive checksum mismatch")
    target = Path(work_root).resolve() / SOURCE_BUNDLE_SHA256
    for name, loaded in list(sys.modules.items()):
        if loaded is None or not (name == "lifemate" or name.startswith("lifemate.")):
            continue
        origin = getattr(loaded, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(target / "src/lifemate"):
            raise RuntimeError("Kernel already imported another LifeMate source. Restart Session, then Run All.")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        if (len(archive.namelist()) != len(SOURCE_FILES_SHA256)
                or set(archive.namelist()) != set(SOURCE_FILES_SHA256)):
            raise RuntimeError("Unexpected files in embedded source archive")
        verified = []
        for name, expected in SOURCE_FILES_SHA256.items():
            path = (target / name).resolve()
            data = archive.read(name)
            if not path.is_relative_to(target) or hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError("Embedded source file checksum mismatch: " + name)
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError("Source file was changed; refusing to overwrite: " + str(path))
            verified.append((path, data))
        for path, data in verified:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.open("xb") as stream:
                    stream.write(data)
    source_path = str(target / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    importlib.invalidate_caches()
    return target

PROJECT_ROOT = install_embedded_source(SOURCE_WORK_ROOT)
print("Source root:", PROJECT_ROOT)
print("Source bundle SHA256:", SOURCE_BUNDLE_SHA256)
print("Base Git commit (source may include local edits):", SOURCE_BASE_COMMIT)
'''
    cells = [
        _cell("markdown", """
            # LifeMate: Agent + ToolEnv + evaluation trên Kaggle

            Upload **chính notebook này**, Add Input dataset BTC, bật GPU rồi chạy từ trên xuống.
            Notebook mang theo source Agent/evaluator cùng một snapshot; không cần clone GitHub
            hoặc upload riêng `agent.py`. Không nhúng dataset, checkpoint hay credentials.

            Mặc định: **10 mẫu validation lấy từ train**, transcript chuẩn current/history
            -> Qwen thật -> SDK ToolEnv thật -> validate/replay -> điểm ước lượng và diagnostics.
            `SAMPLE_COUNT=1` để smoke test; `None` để chạy toàn validation. Đây không phải public test.
            Gold calls/final state chỉ dùng sau inference. Oracle chỉ là chế độ kiểm tra riêng Agent;
            notebook không chạy ASR model.
            """, "intro"),
        _cell("markdown", """
            ## 1. Cấu hình

            `DATA_ROOT=None` tự tìm dataset đã Add Input. `MODEL_SOURCE` có thể là thư mục
            checkpoint có `config.json` hoặc model ID Hugging Face. Model ID cần cache hoặc
            Internet khi `ALLOW_MODEL_DOWNLOADS=True`. Để tái lập model, điền `MODEL_REVISION`.
            """, "config-help"),
        _cell("code", """
            from pathlib import Path

            DATA_ROOT = None
            SAMPLE_COUNT = 10       # 1: smoke; 10: thử ngắn; None: toàn validation
            SAMPLE_IDS = None       # Tuỳ chọn: IDs thuộc validation; không giả định IDs liên tục
            ASR_MODE = "oracle_text" # "oracle_text" hoặc "asr_text"
            SPEECH_FILE = None      # asr_text: file asr_predictions.jsonl gồm current + history

            MODEL_SOURCE = "Qwen/Qwen3-4B-Instruct-2507"
            MODEL_REVISION = None   # Commit checkpoint HF; None phải được ghi nhận là chưa pin
            DEVICE = "cuda:0"
            ALLOW_MODEL_DOWNLOADS = True
            INSTALL_DEPENDENCIES = True  # Chỉ cài khi thiếu/sai phiên bản; giữ PyTorch/CUDA Kaggle
            AGENT_OPTIONS = {}      # Mặc định 48 model steps, 12 tool calls; không đổi thuật toán

            WORK_ROOT = Path("/kaggle/working/lifemate-unified-evaluation")
            SOURCE_WORK_ROOT = Path("/kaggle/working/lifemate-unified-source")
            LAST_RESULT = None
            """, "config"),
        _cell("markdown", """
            ## 2. Cài source đi cùng notebook

            Cell này chỉ giải nén source/config, kiểm tra SHA256 và chọn đúng đường import.
            Nếu kernel đã import LifeMate từ notebook/source khác, **Restart Session** trước.
            Hash in ra là danh tính source thực chạy, không chỉ là Git commit.
            """, "source-help"),
        _cell("code", bootstrap, "source-bootstrap"),
        _cell("markdown", """
            ## 3. Chuẩn bị selection và kiểm tra cấu hình trước model

            Chia tập train theo grouping/seed cố định, chọn subset validation và dùng cùng selection
            cho inference lẫn evaluator. Chuẩn bị lần đầu có thể mất thời gian vì kiểm tra audio.
            Không tải/nạp Qwen tại bước này. Mỗi lần chạy cell tạo một thư mục thí nghiệm mới;
            bundle chia dữ liệu đã kiểm tra được dùng lại dưới cùng WORK_ROOT.
            """, "preflight-help"),
        _cell("code", """
            plan = None
            LAST_RESULT = None
            import json

            from lifemate.agent import discover_data_root
            from lifemate.evaluation.experiment import prepare_experiment

            if ASR_MODE not in {"oracle_text", "asr_text"}:
                raise ValueError("ASR_MODE phải là oracle_text hoặc asr_text")
            if ASR_MODE == "asr_text" and SPEECH_FILE is None:
                raise ValueError("Đặt SPEECH_FILE tới file ASR current/history của đội")
            dataset_root = discover_data_root(explicit=DATA_ROOT) if DATA_ROOT else discover_data_root()
            plan = prepare_experiment(
                dataset_root, WORK_ROOT, count=SAMPLE_COUNT, sample_ids=SAMPLE_IDS,
                speech_path=SPEECH_FILE, oracle=ASR_MODE == "oracle_text",
                agent_options=AGENT_OPTIONS, model_source=MODEL_SOURCE,
                model_revision=MODEL_REVISION, device=DEVICE,
            )
            effective_settings = json.loads(Path(plan["config"]).read_text(encoding="utf-8"))
            print("Dataset root:", dataset_root)
            print("ASR source:", ASR_MODE)
            print("Selected validation samples:", len(plan["selected_ids"]))
            print("Selected IDs:", plan["selected_ids"])
            print("Agent options:", effective_settings["generation"])
            print("Selection manifest:", plan["selection"])
            print("Experiment:", plan["output_dir"])
            """, "preflight"),
        _cell("markdown", """
            ## 4. Dependencies của kernel Kaggle

            Chỉ cài Transformers/Accelerate/HF Hub khi cần. Không cài lại Torch/CUDA hoặc ASR SDK.
            Nếu thư viện cũ đã được import trước khi thay phiên bản, restart session rồi Run All.
            """, "dependencies-help"),
        _cell("code", """
            import importlib.metadata
            import subprocess
            import sys

            def installed_version(name):
                try:
                    return importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    return None

            def compatible_major_minor(value, minimum):
                if value is None:
                    return False
                try:
                    parts = tuple(int(part) for part in value.split(".")[:2])
                    return minimum <= parts < (2, 0)
                except ValueError:
                    return False

            versions = {name: installed_version(name) for name in
                        ("torch", "transformers", "accelerate", "huggingface-hub")}
            needs_install = (versions["transformers"] != "5.17.0"
                             or not compatible_major_minor(versions["accelerate"], (1, 12))
                             or not compatible_major_minor(versions["huggingface-hub"], (1, 5)))
            if needs_install and INSTALL_DEPENDENCIES:
                if globals().get("qwen_backend") is not None or any(
                        name in sys.modules for name in ("transformers", "accelerate", "huggingface_hub")):
                    raise RuntimeError("Dependencies need updating after import. Restart Session, then Run All.")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                                       "transformers==5.17.0", "accelerate>=1.12,<2", "huggingface-hub>=1.5,<2"])
            elif needs_install:
                raise RuntimeError("Dependencies chưa phù hợp; đặt INSTALL_DEPENDENCIES=True và chạy lại cell này")
            print("Dependencies:", {name: installed_version(name) for name in versions})
            """, "dependencies"),
        _cell("markdown", """
            ## 5. Nạp Qwen một lần

            Cell này cần GPU. Nếu cùng source/revision/device đã được nạp, dùng lại `qwen_backend`.
            Khi thay model/device, restart session để tránh giữ hai bộ weights trong GPU.
            """, "model-help"),
        _cell("code", """
            import torch
            import transformers
            from transformers import GenerationConfig
            from lifemate.agent import AgentConfig, ModelLoadConfig, load_qwen_backend

            if not torch.cuda.is_available():
                raise RuntimeError("Bật GPU trong Settings của Kaggle trước khi nạp Qwen")
            if not hasattr(GenerationConfig(), "prefill_chunk_size"):
                raise RuntimeError("Transformers thiếu prefill_chunk_size; kiểm tra cell dependencies và restart")
            model_settings = json.loads(Path(plan["config"]).read_text(encoding="utf-8"))
            model_source = model_settings["model_source"]
            model_revision = model_settings.get("model_revision")
            model_device = model_settings.get("device", "cuda:0")
            backend_key = (model_source, model_revision, model_device)
            if globals().get("qwen_backend") is not None:
                if globals().get("loaded_qwen_key") != backend_key:
                    raise RuntimeError("Đã có Qwen khác trong kernel; restart trước khi đổi model/revision/device")
                print("Reusing Qwen backend:", loaded_qwen_key)
            else:
                qwen_model_config = ModelLoadConfig(
                    model_source=model_source, device=model_device, revision=model_revision,
                    local_files_only=not ALLOW_MODEL_DOWNLOADS,
                )
                qwen_backend = load_qwen_backend(
                    qwen_model_config, AgentConfig(**model_settings["generation"]))
                loaded_qwen_key = backend_key
            print("GPU:", torch.cuda.get_device_name(int(model_device.split(":")[1])))
            print("Torch:", torch.__version__, "Transformers:", transformers.__version__)
            print("Requested model revision:", model_revision)
            print("Loaded model commit:", getattr(getattr(getattr(qwen_backend, "model", None), "config", None), "_commit_hash", None))
            """, "model"),
        _cell("markdown", """
            ## 6. Chạy Agent, ToolEnv và evaluation

            Đây là cell chạy model. Cùng selection được dùng cho cả Agent và scorer; không chấm
            10 predictions trên mẫu số của toàn validation. Mỗi mẫu có một inference ToolEnv.
            Evaluator replay trong env riêng, không reset hoặc thay state của Agent.
            Chạy lại bằng một `plan` mới (cell 3); không ghi đè/chọn lại kết quả của run cũ.
            """, "run-help"),
        _cell("code", """
            LAST_RESULT = None
            if globals().get("qwen_backend") is None:
                raise RuntimeError("Chạy cell 5 để nạp Qwen trước")
            if globals().get("plan") is None:
                raise RuntimeError("Chạy cell 3 để tạo selection/config trước")
            import json
            from pathlib import Path
            planned = json.loads(Path(plan["config"]).read_text(encoding="utf-8"))
            expected_backend = (planned["model_source"], planned.get("model_revision"),
                                planned.get("device", "cuda:0"))
            if globals().get("loaded_qwen_key") != expected_backend:
                raise RuntimeError("Model đang nạp không khớp plan mới; chạy cell 5 hoặc restart trước khi đổi model")
            from lifemate.evaluation.experiment import run_experiment

            LAST_RESULT = run_experiment(plan, backend=qwen_backend)
            print("Summary:", LAST_RESULT["summary_path"])
            """, "run"),
        _cell("markdown", """
            ## 7. Báo cáo tổng hợp và tải kết quả

            Điểm local dựa trên calls/hash và convention được ghi trong report. Status runtime
            (`finished`, `waiting_for_user`, `step_limit`...) được báo riêng; `finished` không chứng minh đúng.
            `submitted_hash == replay_hash` kiểm tra pipeline, còn `replay_hash == gold_hash` đo trạng thái đích.
            """, "summary-help"),
        _cell("code", """
            from IPython.display import FileLink, display

            if LAST_RESULT is None:
                raise RuntimeError("Chưa có run hoàn tất; xem lỗi ở cell 6")
            summary = json.loads(Path(LAST_RESULT["summary_path"]).read_text(encoding="utf-8"))
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            artifacts = [Path(LAST_RESULT["summary_path"]), Path(LAST_RESULT["predictions"]),
                         Path(LAST_RESULT["traces"])]
            evaluation_dir = Path(LAST_RESULT["evaluation_dir"])
            artifacts += [evaluation_dir / name for name in ("metrics.json", "samples.jsonl", "report.md")]
            artifacts += [Path(plan[name]) for name in ("selection", "config")]
            if LAST_RESULT.get("validation_dir"):
                artifacts.append(Path(LAST_RESULT["validation_dir"]) / "validation_summary.json")
            elif LAST_RESULT.get("output_dir"):
                artifacts.append(Path(LAST_RESULT["output_dir"]) / "validation/validation_summary.json")
            for path in artifacts:
                if path.is_file():
                    display(FileLink(str(path)))
            """, "summary"),
        _cell("markdown", """
            ## 7b. In đầy đủ ba file evaluation

            Chạy cell dưới sau mục 6 để in `metrics.json`, toàn bộ `samples.jsonl` và
            `report.md` của đúng run. Cell chỉ đọc file, không chạy lại Qwen hay chấm lại.
            Nếu mở kết quả đã lưu sau restart, đặt `EVALUATION_DIR` tới thư mục có ba file.
            """, "evaluation-files-help"),
        _cell("code", """
            from pathlib import Path
            import json

            EVALUATION_DIR = None  # Hoặc "/kaggle/working/.../evaluation" cho run đã lưu
            latest_result = globals().get("LAST_RESULT")
            if EVALUATION_DIR is not None:
                evaluation_path = Path(EVALUATION_DIR)
            elif latest_result and latest_result.get("evaluation_dir"):
                evaluation_path = Path(latest_result["evaluation_dir"])
            else:
                raise RuntimeError("Chạy mục 6 trước, hoặc đặt EVALUATION_DIR tới thư mục evaluation đã lưu")

            filenames = ("metrics.json", "samples.jsonl", "report.md")
            for filename in filenames:
                if not (evaluation_path / filename).is_file():
                    raise FileNotFoundError(evaluation_path / filename)
            print("Evaluation directory:", evaluation_path)
            for filename in filenames:
                content = (evaluation_path / filename).read_text(encoding="utf-8")
                print("\\n==========", filename, "==========")
                if filename.endswith(".json"):
                    print(json.dumps(json.loads(content), ensure_ascii=False, indent=2))
                elif filename.endswith(".jsonl"):
                    for line in content.splitlines():
                        if line.strip():
                            print(json.dumps(json.loads(line), ensure_ascii=False, indent=2))
                else:
                    print(content)
            """, "evaluation-files"),
        _cell("markdown", """
            ## 8. Mẫu lỗi và phân biệt các chỉ số

            Xem `score`, `H` (hash khớp gold), `M/K/G` (matched/submitted/gold calls), status và lỗi.
            Một mẫu runtime dừng sớm vẫn được chấm từ submission thật nếu hợp lệ/replay được.
            Mẫu thiếu prediction hoặc hash không tái lập được không được giả thành danh sách rỗng đúng.
            """, "diagnostics-help"),
        _cell("code", """
            from collections import Counter

            if LAST_RESULT is None:
                raise RuntimeError("Chạy cell 6 trước")
            sample_reports = [json.loads(line) for line in
                              (Path(LAST_RESULT["evaluation_dir"]) / "samples.jsonl").read_text(
                                  encoding="utf-8").splitlines() if line.strip()]
            print("Statuses:", dict(Counter(row.get("runtime_status") for row in sample_reports)))
            print("Evaluated samples:", len(sample_reports))
            for row in [r for r in sample_reports if r.get("score", 0) < 1][:10]:
                print(row["id"], "score=", row["score"], "H=", row["H"],
                      "M/K/G=", (row["M"], row["K"], row["G"]),
                      "status=", row.get("runtime_status"), "errors=", row.get("observed_errors"))
            """, "diagnostics"),
        _cell("markdown", """
            ## 8b. Mở rộng thử nghiệm

            - Smoke: đặt `SAMPLE_COUNT=1`, chạy cell 3 rồi 5 -> 6 -> 7 -> 8. Cell 5 dùng lại model đã nạp.
            - Thử ngắn: `SAMPLE_COUNT=10`; toàn validation: `SAMPLE_COUNT=None`.
            - Đo sau ASR: `ASR_MODE="asr_text"`, đặt `SPEECH_FILE`, giữ cùng model/config/IDs.
            - Khi sửa code local, chạy `python scripts/build_lifemate_test_notebook.py`, upload lại notebook
              mới và restart session. Sửa file local không tự cập nhật notebook đã upload.

            Phép đo này không bao gồm thời gian nhận dạng ASR và chưa được đối chiếu scorer BTC chính thức.
            Giữ nguyên convention local `token-F1 > 0.6`, F1 calls `2M/(K+G)` theo hình đã cung cấp;
            tài liệu text trong repo có mâu thuẫn được ghi rõ trong báo cáo.
            """, "next-help"),
        _cell("code", """
            # Chỉ đọc đường dẫn, không chạy model/evaluator lại.
            if globals().get("plan"):
                print("Selected IDs:", plan["selected_ids"])
                print("Config:", plan["config"])
                print("Selection:", plan["selection"])
            print("Source bundle:", SOURCE_BUNDLE_SHA256)
            """, "provenance"),
        _cell("markdown", """
            ## 9. Trace chi tiết — tuỳ chọn

            Điền `TRACE_SAMPLE_ID` rồi chạy cell này để đọc các event của đúng run.
            Trace chứa request/tool arguments của dataset; giữ kết quả private khi chia sẻ notebook.
            Cell này chỉ đọc log, không gọi model hoặc thực thi lại tool.
            """, "trace-help"),
        _cell("code", """
            # Có thể đặt hai biến này trong cell trước để đọc kết quả đã lưu sau restart.
            TRACE_SAMPLE_ID = globals().get("TRACE_SAMPLE_ID")
            TRACE_RESULTS_FILE = globals().get("TRACE_RESULTS_FILE")
            if TRACE_SAMPLE_ID is None:
                print("Điền TRACE_SAMPLE_ID để xem chi tiết một mẫu")
            else:
                from pathlib import Path
                import json
                latest = globals().get("LAST_RESULT")
                trace_path = TRACE_RESULTS_FILE or (latest.get("traces") if latest else None)
                if trace_path is None:
                    raise RuntimeError("Đặt TRACE_RESULTS_FILE tới results.jsonl đã lưu, hoặc chạy cell 6")
                print("Trace file:", trace_path)
                rows = [json.loads(line) for line in Path(trace_path).read_text(
                    encoding="utf-8").splitlines() if line.strip()]
                selected_trace = next((r for r in rows if r.get("id") == TRACE_SAMPLE_ID), None)
                if selected_trace is None:
                    raise ValueError("ID không thuộc run hiện tại: " + str(TRACE_SAMPLE_ID))
                print("ID:", TRACE_SAMPLE_ID, "STATUS:", selected_trace.get("status"))
                for event in selected_trace.get("trace", []):
                    print(json.dumps(event, ensure_ascii=False, indent=2))
            """, "trace"),
    ]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "lifemate": {"format": "unified-agent-evaluation-v1", "source_bundle_sha256": bundle_hash},
    }, "nbformat": 4, "nbformat_minor": 5}


def build(output=NOTEBOOK, *, check=False):
    output = Path(output)
    content = (json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    if check:
        def comparison(document):
            # The base commit is provenance from build time, not the source
            # fingerprint. Committing the notebook changes HEAD without changing
            # its embedded source; JSON line endings may also differ by checkout.
            for cell in document["cells"]:
                if cell.get("id") == "source-bootstrap":
                    source = re.sub(r"(?m)^SOURCE_BASE_COMMIT = '[0-9a-f]{40}'$",
                                    "SOURCE_BASE_COMMIT = '<build-base>'", "".join(cell["source"]))
                    cell["source"] = source.splitlines(keepends=True)
            return document
        if not output.is_file() or comparison(json.loads(output.read_text(encoding="utf-8"))) != comparison(json.loads(content)):
            raise ValueError("Notebook source snapshot is stale; rebuild it")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
    return {"path": str(output), "bytes": len(content), "check_only": check}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=NOTEBOOK)
    parser.add_argument("--check", action="store_true")
    options = parser.parse_args()
    print(json.dumps(build(options.output, check=options.check)))
