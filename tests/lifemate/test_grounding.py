"""Source-grounding regressions across synthetic targets; no model inference."""
from copy import deepcopy
import json
import os
from pathlib import Path
import unittest

from lifemate import agent
from lifemate.pipeline import grounding, protocol
from test_workflow import Backend, Env, call


def make_session(text, *replies, results=None, domain="inventory", context=None, history=None, **options):
    tools = [
        {"name": domain + ".inspect", "kind": "read", "parameters": {"type": "object", "properties": {
            "room": {"type": "string"}, "query": {"type": "string"}}}},
        {"name": domain + ".change", "kind": "write", "parameters": {"type": "object", "properties": {
            "object_id": {"type": "string", "x-ref": domain + ".objects", "description": "ID from " + domain + ".inspect"},
            "value": {"type": "integer"}}}},
        {"name": "payments.top_up", "kind": "write", "parameters": {"type": "object", "properties": {
            "phone": {"type": "string"}, "amount": {"type": "integer"}}}},
        {"name": "sys.ask_missing_info", "kind": "meta", "parameters": {"type": "object", "properties": {
            "tool_name": {"type": "string"}, "params": {"type": "array"}}}},
    ]
    request = {"id": "invented", "context": {**deepcopy(context or {}), "installed_apps": {
        domain: {"tools": [t["name"] for t in tools[:2]]}, "payments": {"tools": ["payments.top_up"]}}},
        "history": deepcopy(history or [])}
    speech = {"id": "invented", "current_text": text, "history_user_texts": [
        {"history_index": i, "text": turn["audio_text"]} for i, turn in enumerate(history or []) if turn["role"] == "user"]}
    options.setdefault("terminal_check_enabled", False)
    options.setdefault("task_reasoning_enabled", False)
    return agent.AgentSession(request, speech, tools, Env(results), Backend(*replies), agent.AgentConfig(**options))


class DigitSourceTests(unittest.TestCase):
    def test_spoken_digits_preserve_each_zero_and_source_offsets(self):
        text = "Nạp cho số không bảy không hai không ba không bốn không năm sáu nhé."
        spans = grounding.digit_spans(text)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["value"], "07020304056")
        self.assertEqual(text[spans[0]["start"]:spans[0]["end"]], spans[0]["text"])

    def test_case_punctuation_and_mixed_asr_digits_are_supported(self):
        for text in ("KHÔNG, bảy, không, hai, không, ba, không, bốn, không, năm, sáu",
                     "không 7 không 2 không 3 không 4 không 5 sáu"):
            with self.subTest(text=text):
                self.assertEqual(grounding.digit_spans(text)[0]["value"], "07020304056")

    def test_cardinal_units_dates_and_money_do_not_become_spoken_phone_digits(self):
        for text in ("mười một hai ba bốn năm sáu bảy", "một hai ba bốn năm sáu triệu",
                     "50 nghìn ngày 2028-04-13", "một trăm hai mươi nghìn"):
            with self.subTest(text=text):
                self.assertEqual(grounding.digit_spans(text), [])

    def test_direct_numbers_and_multiple_mentions_remain_distinct_sources(self):
        text = "Số 07123456789, à không, số 08234567890."
        self.assertEqual([s["value"] for s in grounding.digit_spans(text)], ["07123456789", "08234567890"])

    def test_formatted_digit_string_is_normalized_without_changing_its_length(self):
        span = grounding.digit_spans("Số 0812 345 6789")[0]
        self.assertEqual(span["value"], "08123456789")
        self.assertEqual(span["text"], "0812 345 6789")


class InputViewTests(unittest.TestCase):
    def test_current_request_history_and_original_messages_are_preserved(self):
        text = "Tạo lúc 6 giờ, à không, 6 rưỡi ngày mai."
        run = make_session(text, history=[{"role": "user", "audio": "old.wav", "audio_text": "Earlier request"},
                                        {"role": "assistant", "text": "Earlier answer"}])
        before = deepcopy(run.messages)
        messages = protocol.actor_messages(run)
        view = json.loads(messages[-1]["content"].split("\n", 1)[1])
        self.assertEqual(view["current_request"], text)
        self.assertEqual(messages[-2]["content"], text)
        self.assertEqual(run.messages, before)
        self.assertEqual(run.speech["history_user_texts"], [{"history_index": 0, "text": "Earlier request"}])
        self.assertIn("Earlier request", [m["content"] for m in messages])

    def test_first_effective_model_input_is_captured_without_new_control_generation(self):
        run = make_session("Inspect the small office", call("inventory.inspect"), "Done", results=[{"ok": True}])
        result = run.run()
        saved = result["diagnostics"]["first_model_input"]
        self.assertEqual(saved["messages"], run.backend.seen[0][0])
        self.assertEqual(saved["tools"], run.backend.seen[0][1])
        self.assertEqual(len(saved["sha256"]), 64)
        self.assertEqual(result["model_calls"], 2)
        self.assertFalse(any(e["type"] == "control_reply" for e in result["trace"]))
        saved["messages"][0]["content"] = "mutated"
        self.assertNotEqual(run.first_model_input["messages"][0]["content"], "mutated")

    def test_grounding_can_be_disabled_for_a_controlled_ablation(self):
        run = make_session("Inspect the small office", grounding_enabled=False)
        self.assertEqual(len(protocol.actor_messages(run)), len(run.messages))
        self.assertIsNone(run.call_state.grounding_view_json)

    def test_only_successful_source_records_are_presented_and_latest_record_wins(self):
        request = {"context": {}, "history": []}
        observations = {
            "a": {"outcome": "success", "call": {"name": "inventory.inspect"}, "result": {"objects": [{"id": "x", "name": "old label"}]}},
            "b": {"outcome": "error", "call": {"name": "inventory.inspect"}, "result": {"objects": [{"id": "bad", "name": "invented"}]}},
            "c": {"outcome": "success", "call": {"name": "inventory.change"}, "result": {"item": {"id": "x", "name": "new label"}}},
        }
        before = deepcopy(observations)
        view = grounding.input_view("Use the new label", request, observations)
        self.assertEqual(view["observed_entities"][0]["labels"]["name"], "new label")
        self.assertEqual(view["observed_entity_count"], 1)
        self.assertEqual(observations, before)


class PhoneGroundingTests(unittest.TestCase):
    def test_dropped_digit_is_rejected_then_correct_call_executes_unchanged(self):
        text = "Nạp số không bảy không hai không ba không bốn không năm sáu."
        bad = call("payments.top_up", phone="0720304056", amount=60000)
        good = call("payments.top_up", phone="07020304056", amount=60000)
        run = make_session(text, bad, good, "Done", results=[{"ok": True}])
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(run.env.calls, [{"name": "payments.top_up", "arguments": {"phone": "07020304056", "amount": 60000}}])
        self.assertEqual(result["invalid_replies"], 1)
        rejection = next(e for e in run.trace if e["type"] == "rejected")
        self.assertEqual(rejection["grounding"][0]["code"], "phone_grounding")
        self.assertEqual(rejection["grounding"][0]["digit_sources"][0]["value"], "07020304056")

    def test_direct_short_service_number_is_not_forced_to_a_phone_length(self):
        run = make_session("Gọi số 115", call("payments.top_up", phone="115", amount=1), "Done", results=[{"ok": True}])
        self.assertEqual(run.run()["status"], "finished")
        self.assertEqual(run.env.calls[0]["arguments"]["phone"], "115")

    def test_real_lookup_number_is_usable_even_with_another_explicit_number_in_request(self):
        text = "Nạp số 07123456789 rồi nạp cho người khác trong danh bạ."
        run = make_session(text, call("inventory.inspect"), call("payments.top_up", phone="08234567890", amount=60000), "Done",
            results=[{"ok": True, "people": [{"id": "p", "name": "Other person", "phone": "08234567890"}]}, {"ok": True}])
        self.assertEqual(run.run()["status"], "finished")
        self.assertEqual(run.invalid_replies, 0)

    def test_rejected_number_never_becomes_an_observed_source(self):
        bad = call("payments.top_up", phone="0720304056", amount=60000)
        run = make_session("Số đúng là 07020304056", bad, bad)
        self.assertEqual(run.run()["status"], "invalid_output")
        self.assertEqual(run.env.calls, [])
        self.assertNotIn("0720304056", run.known_strings)

    def test_phone_from_context_or_historical_user_speech_is_grounded(self):
        for context, history in (({"user_profile": {"phone": "08123456789"}}, []),
                                 ({}, [{"role": "user", "audio": "old.wav", "audio_text": "Số 08123456789"}])):
            run = make_session("Nạp số đã cung cấp", call("payments.top_up", phone="08123456789", amount=1), "Done",
                               context=context, history=history, results=[{"ok": True}])
            self.assertEqual(run.run()["status"], "finished")

    def test_number_embedded_in_real_tool_text_is_a_source(self):
        run = make_session("Gọi số mới nhất trong thông báo", call("inventory.inspect"),
            call("payments.top_up", phone="08123456789", amount=1), "Done",
            results=[{"ok": True, "message": "Số liên hệ mới: 0812 345 6789"}, {"ok": True}])
        self.assertEqual(run.run()["status"], "finished")

    def test_explicit_suffix_edit_preserves_the_rest_of_the_known_number(self):
        history = [{"role": "user", "audio": "old.wav", "audio_text": "Số liên hệ là 08123456789"}]
        run = make_session("Đuôi là sáu bảy chứ không phải tám chín", call("payments.top_up", phone="08123456767", amount=1), "Done",
                           history=history, results=[{"ok": True}])
        self.assertEqual(run.run()["status"], "finished")
        self.assertIn("08123456767", [e["value"] for e in run.call_state.snapshot()["grounding_context"]["explicit_digit_edits"]])
        self.assertEqual(grounding._digit_edits("Đuôi không phải sáu bảy", {"08123456789"}), [])

    def test_partial_digit_edit_does_not_justify_changing_unmentioned_digits(self):
        history = [{"role": "user", "audio": "old.wav", "audio_text": "Số liên hệ là 08123456789"}]
        bad = call("payments.top_up", phone="09123456767", amount=1)
        run = make_session("Đuôi là sáu bảy", bad, bad, history=history)
        self.assertEqual(run.run()["status"], "invalid_output")
        self.assertEqual(run.env.calls, [])

    def test_typed_prefix_and_suffix_edits_are_supported_without_changing_other_digits(self):
        numbers = {"08123456789"}
        self.assertEqual(grounding._digit_edits("Đuôi là 67", numbers)[0]["value"], "08123456767")
        self.assertEqual(grounding._digit_edits("Đầu số thành 09", numbers)[0]["value"], "09123456789")
        self.assertEqual(grounding._digit_edits("Đuôi là 67", {"500000"}), [])

    def test_person_name_cannot_be_used_as_a_phone_even_when_the_name_is_in_the_request(self):
        bad = call("payments.top_up", phone="Dì Hoa", amount=1)
        run = make_session("Nạp cho Dì Hoa", bad, bad)
        self.assertEqual(run.run()["status"], "invalid_output")
        self.assertEqual(run.env.calls, [])

    def test_amount_is_not_rewritten_or_validated_as_phone_digits(self):
        run = make_session("Nạp 75 nghìn cho số 08123456789", call("payments.top_up", phone="08123456789", amount=75000), "Done",
                           results=[{"ok": True}])
        self.assertEqual(run.run()["status"], "finished")
        self.assertEqual(run.env.calls[0]["arguments"]["amount"], 75000)


class EntityGroundingTests(unittest.TestCase):
    def test_lossy_lookup_filter_is_rejected_then_broad_lookup_can_run(self):
        run = make_session("Giảm đèn phòng làm việc nhỏ xuống một nửa", call("inventory.inspect", room="phòng làm việc"),
                           call("inventory.inspect"), "Done", results=[{"ok": True, "objects": []}])
        self.assertEqual(run.run()["status"], "finished")
        self.assertEqual(run.env.calls, [{"name": "inventory.inspect", "arguments": {}}])
        violation = next(e for e in run.trace if e["type"] == "rejected")["grounding"][0]
        self.assertEqual(violation["source_phrases"], ["phòng làm việc nhỏ"])

    def test_observed_id_must_not_lose_an_explicit_qualifier_across_domains(self):
        for domain, text, generic, specific in (
            ("inventory", "Đổi thiết bị phòng làm việc nhỏ", "thiết bị phòng làm việc", "thiết bị phòng làm việc nhỏ"),
            ("documents", "Update the small office", "office", "small office"),
            ("storage", "Đổi tủ kho số 3", "tủ kho", "tủ kho số 3"),
        ):
            with self.subTest(domain=domain):
                run = make_session(text, call(domain + ".inspect"), call(domain + ".change", object_id="generic", value=5),
                    call(domain + ".change", object_id="specific", value=5), "Done", domain=domain,
                    results=[{"ok": True, "objects": [{"id": "generic", "name": generic}, {"id": "specific", "name": specific}]}, {"ok": True}])
                self.assertEqual(run.run()["status"], "finished")
                self.assertEqual(run.env.calls[-1]["arguments"]["object_id"], "specific")
                self.assertEqual(len(run.env.calls), 2)

    def test_context_room_label_can_supply_a_qualifier_missing_from_short_display_name(self):
        run = make_session("Đổi thiết bị phòng làm việc nhỏ", call("inventory.inspect"),
            call("inventory.change", object_id="right", value=5), "Done",
            results=[{"ok": True, "objects": [{"id": "right", "name": "thiết bị phòng làm việc", "room": "phòng làm việc nhỏ"}]}, {"ok": True}])
        self.assertEqual(run.run()["status"], "finished")
        self.assertEqual(run.invalid_replies, 0)

    def test_ambiguous_multiple_targets_do_not_get_assigned_to_a_call_by_lexical_rules(self):
        for text in ("Đổi phòng làm việc và phòng làm việc nhỏ", "Đổi phòng làm việc nhỏ và phòng làm việc lớn"):
            self.assertEqual(grounding._qualifier_needs(text, "phòng làm việc"), [])

    def test_time_modal_and_polite_request_words_are_not_entity_qualifiers(self):
        for text in ("Tắt đèn phòng làm việc 2 giờ nữa", "Đèn phòng làm việc phải bật", "Đèn phòng làm việc phải xanh", "Đèn phòng làm việc nhờ bạn tắt",
                     "Giảm đèn phòng làm việc nhỏ hơn một chút", "Bật điều hòa phòng làm việc 24 độ",
                     "Bật đèn phòng làm việc 35 phần trăm"):
            with self.subTest(text=text):
                self.assertEqual(grounding._qualifier_needs(text, "phòng làm việc"), [])

    def test_model_and_argument_objects_are_not_mutated_by_grounding(self):
        value = {"name": "inventory.inspect", "arguments": {"room": "phòng làm việc"}}
        schema = {"properties": {"room": {"type": "string"}}}
        before = deepcopy(value)
        issues = grounding.check_call(value, schema, current_text="Dùng phòng làm việc nhỏ", request={"context": {}, "history": []},
            observations={}, known_strings=set(), history_user_texts=[], tool_kind="read")
        self.assertTrue(issues)
        self.assertEqual(value, before)

    def test_free_text_write_title_is_not_treated_as_a_lookup_filter(self):
        value = {"name": "notes.create", "arguments": {"title": "phòng làm việc"}}
        issues = grounding.check_call(value, {"properties": {"title": {"type": "string"}}},
            current_text="Ghi chú cho phòng làm việc nhỏ", request={"context": {}, "history": []},
            observations={}, known_strings=set(), history_user_texts=[], tool_kind="write")
        self.assertEqual(issues, [])

    def test_explicit_side_phrase_is_preserved_without_interpreting_modal_verbs(self):
        self.assertEqual(grounding._qualifier_needs("Dùng tủ bên phải", "tủ")[0]["qualifier"], "phai")

    def test_nested_object_array_reference_gets_the_exact_argument_path(self):
        value = {"name": "inventory.change", "arguments": {"targets": [{"object_id": "generic"}]}}
        schema = {"properties": {"targets": {"type": "array", "items": {"type": "object", "properties": {
            "object_id": {"type": "string", "x-ref": "inventory.objects"}}}}}}
        observations = {"obs": {"outcome": "success", "call": {"name": "inventory.inspect"}, "result": {
            "objects": [{"id": "generic", "name": "office"}]}}}
        issues = grounding.check_call(value, schema, current_text="Update the small office", request={"context": {}, "history": []},
            observations=observations, known_strings={"generic"}, history_user_texts=[])
        self.assertEqual(issues[0]["field"], "arguments.targets[0].object_id")

    def test_same_id_from_a_different_namespace_is_not_used_as_target_evidence(self):
        value = {"name": "inventory.change", "arguments": {"object_id": "shared"}}
        schema = {"properties": {"object_id": {"type": "string", "x-ref": "inventory.objects"}}}
        observations = {"obs": {"outcome": "success", "call": {"name": "contacts.search"}, "result": {
            "contacts": [{"id": "shared", "name": "office"}]}}}
        issues = grounding.check_call(value, schema, current_text="Update the small office", request={"context": {}, "history": []},
            observations=observations, known_strings={"shared"}, history_user_texts=[])
        self.assertEqual(issues, [])


class OfficialGroundingAuditTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("LIFEMATE_DATA_ROOT"), "Optional offline source-grounding corpus check")
    def test_official_label_calls_are_not_rejected_by_lexical_grounding_checks(self):
        # Labels are used only in this offline audit, never by the runtime guard.
        from lifemate.pipeline.inputs import prepare_request
        from lifemate.pipeline.tools import _strings
        root = Path(os.environ["LIFEMATE_DATA_ROOT"]) / "train"
        catalog = json.loads((root / "train_tools_catalog.json").read_text(encoding="utf-8"))
        tools = {t["name"]: t for app in catalog["apps"] for t in app["tools"]}
        count = 0
        failures = []
        with (root / "train.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip(): continue
                row = json.loads(line); request = prepare_request(row)
                known = set(_strings(request)); observations = {}
                history = [{"text": t["audio_text"]} for t in row.get("history", []) if t["role"] == "user"]
                for i, (value, result) in enumerate(zip(row["label"]["tool_calls"], row["label"]["tool_results"])):
                    issues = grounding.check_call(value, tools[value["name"]]["parameters"], current_text=row["audio_text"],
                        request=request, observations=observations, known_strings=known, history_user_texts=history,
                        tool_kind=tools[value["name"]].get("kind"))
                    if issues:
                        failures.append((row["id"], row["audio_text"], value, issues))
                    count += 1
                    outcome = "success" if result.get("ok") is True and result.get("done") is not False else "error"
                    observations[str(i)] = {"call": value, "result": result, "outcome": outcome}
                    if outcome == "success": known.update(_strings(result))
        self.assertGreater(count, 0)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
