"""Generate and judge ASR corrections with a strict total-token budget."""
from __future__ import annotations

import json

MAX_CONTEXT_TOKENS = 16_384
MAX_PROMPT_TOKENS = 16_000
CORRECTION_TOKENS = 256
JUDGE_TOKENS = 128

CORRECT_SYSTEM = (
    "Bạn sửa lỗi nhận dạng tiếng Việt. Chỉ sửa từ có căn cứ rõ từ câu và ngữ cảnh "
    "hội thoại trước đó. Giữ nguyên nội dung, tên, số, thời gian, phủ định và các "
    "vế lệnh khi không chắc. Không diễn giải hoặc thêm thông tin. Nếu câu đã đúng, "
    "giữ nguyên. Chỉ trả JSON: {\"corrected_text\": \"...\"}."
)
JUDGE_SYSTEM = (
    "Bạn kiểm tra một đề xuất sửa transcript ASR. Không sửa lại câu và không coi "
    "câu trôi chảy là bằng chứng âm thanh. Chỉ chấp nhận nếu từng thay đổi được "
    "ngữ cảnh hỗ trợ và không làm mất ý. Nếu không chắc, trả uncertain. "
    "Chỉ trả JSON: {\"decision\": \"accept|reject|uncertain\", "
    "\"reason\": \"một câu ngắn\"}."
)


class PromptTooLong(ValueError):
    pass


class TransformersLM:
    """One model load, two chat calls; never implicitly download model weights."""

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float16, device_map="auto", local_files_only=True,
            attn_implementation="sdpa",
        ).eval()
        self.eos_ids = self.model.generation_config.eos_token_id
        if isinstance(self.eos_ids, int):
            self.eos_ids = {self.eos_ids}
        else:
            self.eos_ids = set(self.eos_ids or [])

    def count_prompt(self, messages: list[dict]) -> int:
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        return ids.shape[-1]

    def chat(self, messages: list[dict], max_new_tokens: int) -> str:
        import torch

        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        prompt_tokens = ids.shape[-1]
        if prompt_tokens > MAX_PROMPT_TOKENS or prompt_tokens + max_new_tokens > MAX_CONTEXT_TOKENS:
            raise PromptTooLong(f"prompt={prompt_tokens}, output_reserve={max_new_tokens}")
        ids = ids.to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output[0, prompt_tokens:]
        if len(generated) >= max_new_tokens and int(generated[-1]) not in self.eos_ids:
            raise ValueError("LM output reached max_new_tokens without EOS")
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def _messages(system: str, payload: dict) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def fit_prompt(lm, system: str, payload: dict) -> list[dict]:
    """Keep current text and fixed examples; remove oldest history first."""
    payload = dict(payload)
    payload["prior_user_asr"] = list(payload.get("prior_user_asr", []))
    while True:
        messages = _messages(system, payload)
        if lm.count_prompt(messages) <= MAX_PROMPT_TOKENS:
            return messages
        if not payload["prior_user_asr"]:
            raise PromptTooLong("current text and required instructions exceed prompt budget")
        payload["prior_user_asr"].pop(0)


def _parse_json_object(raw: str) -> dict:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("LM output is not a JSON object")
    return value


def generate_correction(lm, original: str, history: list[dict], examples: list[dict]) -> tuple[str, str]:
    payload = {"prior_user_asr": history, "asr_text": original}
    if examples:
        payload["train_examples"] = examples
    try:
        messages = fit_prompt(lm, CORRECT_SYSTEM, payload)
        result = _parse_json_object(lm.chat(messages, CORRECTION_TOKENS))
        candidate = result.get("corrected_text")
        if not isinstance(candidate, str) or not candidate.strip():
            return original, "invalid_output"
        return candidate.strip(), "generated"
    except PromptTooLong:
        return original, "input_over_budget"
    except (ValueError, TypeError):
        return original, "invalid_output"


def judge_correction(lm, original: str, candidate: str, history: list[dict]) -> tuple[str, str]:
    payload = {"prior_user_asr": history, "asr_text": original, "proposed_text": candidate}
    try:
        messages = fit_prompt(lm, JUDGE_SYSTEM, payload)
        result = _parse_json_object(lm.chat(messages, JUDGE_TOKENS))
        decision = result.get("decision")
        reason = result.get("reason")
        if decision not in {"accept", "reject", "uncertain"} or not isinstance(reason, str):
            return "uncertain", "invalid_judge_output"
        return decision, reason[:240]
    except PromptTooLong:
        return "uncertain", "input_over_budget"
    except (ValueError, TypeError):
        return "uncertain", "invalid_judge_output"
