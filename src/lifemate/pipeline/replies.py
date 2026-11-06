"""Strict single-call/native-final parser; malformed structured output never executes."""
from __future__ import annotations

import json
import re

from .config import ModelOutputError


def _unique_json_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ModelOutputError(f"JSON có khóa trùng: {key}")
        obj[key] = value
    return obj


def _bad_json_constant(value):
    raise ModelOutputError(f"JSON không hợp lệ: {value}")


def _parse_call(source):
    try:
        call = json.loads(source, object_pairs_hook=_unique_json_object,
                          parse_constant=_bad_json_constant)
    except json.JSONDecodeError as error:
        raise ModelOutputError(f"JSON tool call sai: {error.msg}") from error
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        raise ModelOutputError("Tool call phải có đúng name và arguments; không có action")
    if not isinstance(call["name"], str) or not call["name"].strip() or not isinstance(call["arguments"], dict):
        raise ModelOutputError("name phải là chuỗi không rỗng và arguments phải là object")
    try:
        json.dumps(call, allow_nan=False)
    except (ValueError, OverflowError) as error:
        raise ModelOutputError("Tool call phải chứa giá trị JSON hữu hạn") from error
    return {"type": "call", "call": call}


def parse_model_reply(raw):
    """Read one native Qwen call or a final response; never repair call JSON.

    Qwen's chat template permits assistant content before a function call and
    plain assistant text when no tool is called. Keep the single-call policy,
    but do not require the non-native <final> wrapper for a normal response.
    Malformed structured output must still be rejected, never treated as final.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ModelOutputError("Model trả rỗng")
    raw = raw.strip()
    if raw.startswith("{"):
        return _parse_call(raw)
    if "<tool_call" in raw or "</tool_call" in raw:
        if raw.count("<tool_call>") != 1 or raw.count("</tool_call>") != 1:
            raise ModelOutputError("Mỗi lượt cần đúng một tool call hoàn chỉnh")
        match = re.fullmatch(r"(?P<prefix>.*?)<tool_call>\s*(?P<call>.*?)\s*</tool_call>", raw, re.S)
        if not match or any(marker in match.group("prefix") for marker in ("<", "```", "{", "[")):
            raise ModelOutputError("Tag tool call sai hoặc có nội dung có cấu trúc ngoài tool call")
        parsed = _parse_call(match.group("call"))
        if match.group("prefix").strip():
            parsed["content"] = match.group("prefix").strip()
        return parsed
    final = re.fullmatch(r"<final>\s*(.*?)\s*</final>", raw, re.S)
    if final:
        return {"type": "finish", "text": final.group(1)}
    # Native prose followed by an identical legacy final is one unambiguous
    # final, not a damaged application call. Never extract arbitrary JSON/text.
    repeated = re.fullmatch(r"([^<>`{}\[\]]+?)\s*<final>\s*([^<>]+?)\s*</final>", raw, re.S)
    if repeated and " ".join(repeated.group(1).split()) == " ".join(repeated.group(2).split()):
        if not re.search(r'"(?:name|arguments|action|tool_calls)"\s*:', repeated.group(1)):
            return {"type": "finish", "text": repeated.group(2).strip()}
    if (raw.startswith(("{", "[", "<", "```"))
            or any(marker in raw for marker in ("<final", "</final", "<think", "</think", "```"))
            or re.search(r'"(?:name|arguments|action|tool_calls)"\s*:', raw)):
        raise ModelOutputError("Output có cấu trúc sai; cần một <tool_call> JSON hoàn chỉnh hoặc câu trả lời cuối")
    return {"type": "finish", "text": raw}
