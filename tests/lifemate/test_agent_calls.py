"""Regression tests for native Qwen response formats; no model-quality claims."""
from contextlib import redirect_stdout
from copy import deepcopy
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from lifemate import agent
from lifemate.asr.predictions import write_speech_file
from test_predictions import fixture


def native_call(name="calendar.list_events", arguments=None):
    return "<tool_call>" + json.dumps({
        "name": name, "arguments": arguments or {},
    }) + "</tool_call>"


class Backend:
    def __init__(self, *replies):
        self.replies = iter(replies)
        self.messages = []

    def generate(self, messages, tools, config, remaining_seconds=None):
        self.messages.append(deepcopy(messages))
        reply = next(self.replies)
        return reply if isinstance(reply, agent.ModelReply) else agent.ModelReply(reply)


class Env:
    def __init__(self, verdict=None):
        self.calls = []
        self.verdict = verdict or {"ok": True}

    def validate(self, name, arguments):
        return self.verdict

    def call(self, name, arguments):
        self.calls.append({"name": name, "arguments": deepcopy(arguments)})
        return {"ok": True, "events": [{"event_id": "observed_event_1"}]}

    def get_tool_calls(self):
        return deepcopy(self.calls)

    def get_state_hash(self):
        return "unit-test-state-hash"


def session(backend, env=None):
    tools = [
        {"name": "calendar.list_events", "parameters": {"type": "object"}},
        {"name": "calendar.update_event", "parameters": {
            "type": "object", "properties": {"event_id": {"type": "string", "x-ref": "event"}},
        }},
        {"name": "sys.ask_missing_info", "parameters": {"type": "object"}},
    ]
    request = {"id": "sample", "history": [], "context": {"installed_apps": {
        "calendar": {"tools": [tool["name"] for tool in tools[:2]]},
    }}}
    speech = {"id": "sample", "current_text": "List my events", "history_user_texts": []}
    return agent.AgentSession(request, speech, tools, env or Env(), backend, agent.AgentConfig(workflow_enabled=False))


class NativeReplyTests(unittest.TestCase):
    def test_identical_prose_and_final_wrapper_normalize_to_one_final(self):
        self.assertEqual(agent.parse_model_reply("Unable to proceed.\n<final>Unable to proceed.</final>"),
                         {"type": "finish", "text": "Unable to proceed."})
        for raw in ("Unable to proceed.<final>Completed</final>",
                    "Done<final>Done</final><final>Done</final>",
                    '["Done"]<final>Done</final>'):
            with self.subTest(raw=raw), self.assertRaises(agent.ModelOutputError):
                agent.parse_model_reply(raw)

    def test_plain_native_final_is_accepted_and_legacy_wrapper_still_works(self):
        for raw in ("Your alarm is set.", "<final>Your alarm is set.</final>"):
            with self.subTest(raw=raw):
                self.assertEqual(agent.parse_model_reply(raw), {
                    "type": "finish", "text": "Your alarm is set.",
                })

    def test_native_assistant_preamble_can_precede_one_complete_call(self):
        parsed = agent.parse_model_reply("I will check your events first.\n" + native_call())
        self.assertEqual(parsed, {"type": "call", "call": {
            "name": "calendar.list_events", "arguments": {},
        }, "content": "I will check your events first."})

    def test_invalid_structured_output_is_never_reclassified_as_plain_final(self):
        invalid = [
            "", " ", None,
            '{"name":"calendar.list_events","arguments":{},"status":"done"}',
            'Here is my call: {"name":"calendar.list_events","arguments":{}}',
            '```json\n{"name":"calendar.list_events","arguments":{}}\n```',
            "<tool_call>", "</tool_call>", "<tool_call >{}</tool_call>",
            "<final>Done", "Done</final>", "<think>thinking</think>Done",
            native_call() + native_call(),
            native_call() + "<final>Done</final>",
            "```\n" + native_call() + "\n```",
            "<final>Done</final>" + native_call(),
            '<tool_call>{"name":"calendar.list_events","name":"other","arguments":{}}</tool_call>',
            '<tool_call>{"name":"calendar.list_events","arguments":{"x":NaN}}</tool_call>',
            '<tool_call>{"name":"calendar.list_events","arguments":"{}"}</tool_call>',
            '<tool_call>{"action":"call","name":"calendar.list_events","arguments":{}}</tool_call>',
            '<tool_call>{"name":"calendar.list_events","arguments":{}</tool_call>',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(agent.ModelOutputError):
                agent.parse_model_reply(raw)


class AgentCallFlowTests(unittest.TestCase):
    def test_call_result_then_plain_final_preserves_canonical_log_without_retry(self):
        backend = Backend("I will check first.\n" + native_call(), "There is one event.")
        run = session(backend)
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["invalid_replies"], 0)
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(len(result["submission"]["tool_calls"]), 1)
        self.assertEqual(result["submission"]["tool_calls"], run.env.get_tool_calls())
        self.assertEqual(result["submission"]["state_hash"], run.env.get_state_hash())
        tool_message = backend.messages[1][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(json.loads(tool_message["content"])["events"][0]["event_id"], "observed_event_1")

    def test_plain_final_does_not_create_a_tool_call_or_prove_task_correctness(self):
        result = session(Backend("I cannot do that with the available tools.")).run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["submission"]["tool_calls"], [])

    def test_schema_errors_still_stop_without_execution_and_keep_reason(self):
        run = session(Backend(native_call(), native_call()), Env({"ok": False, "error": "Missing date"}))
        result = run.run()
        self.assertEqual(result["status"], "invalid_output")
        self.assertEqual(result["invalid_replies"], 2)
        self.assertEqual(result["submission"]["tool_calls"], [])
        self.assertIn("Missing date", result["trace"][-1]["error"])

    def test_disallowed_tools_still_rejected_without_execution(self):
        result = session(Backend(native_call("not_installed.send"), native_call("not_installed.send"))).run()
        self.assertEqual(result["status"], "invalid_output")
        self.assertEqual(result["submission"]["tool_calls"], [])

    def test_truncated_prose_and_calls_are_rejected_before_parsing(self):
        for text in ("Done", native_call()):
            with self.subTest(text=text):
                backend = Backend(agent.ModelReply(text, truncated=True), "I will stop.")
                result = session(backend).run()
                self.assertEqual(result["status"], "finished")
                self.assertEqual(result["invalid_replies"], 1)
                self.assertTrue(result["trace"][0]["truncated"])
                self.assertFalse(result["trace"][1]["executed"])
                self.assertEqual(result["submission"]["tool_calls"], [])

    def test_unknown_references_are_rejected_but_observed_tool_ids_can_be_used(self):
        update = native_call("calendar.update_event", {"event_id": "observed_event_1"})
        rejected = session(Backend(update, update)).run()
        self.assertEqual(rejected["status"], "invalid_output")
        self.assertEqual(rejected["submission"]["tool_calls"], [])
        accepted = session(Backend(native_call(), update, "Done")).run()
        self.assertEqual(accepted["status"], "finished")
        self.assertEqual(len(accepted["submission"]["tool_calls"]), 2)

    def test_multiple_calls_in_one_reply_never_partially_execute(self):
        result = session(Backend(native_call() * 2, "I will stop.")).run()
        self.assertEqual(result["submission"]["tool_calls"], [])
        self.assertEqual(result["invalid_replies"], 1)

    def test_pipeline_prints_rejection_reason_and_path_to_saved_trace(self):
        with fixture() as (root, _, __, ___):
            speech = write_speech_file(root / "speech.jsonl", [
                {"id": "a", "current_text": "List events", "history_user_texts": [
                    {"history_index": 1, "text": "Previous request"},
                ]},
                {"id": "b", "current_text": "List events", "history_user_texts": []},
            ])
            backend = Backend(agent.ModelReply(native_call(), truncated=True),
                              agent.ModelReply(native_call(), truncated=True))
            output = io.StringIO()
            with redirect_stdout(output):
                result = agent.run_full_pipeline(root, sample_ids=["b"], output_dir=root / "run",
                    asr_predictions_path=speech, backend=backend, agent_config=agent.AgentConfig(workflow_enabled=False),
                    toolenv_loader=lambda *_: lambda cid: Env())
            self.assertIn("rejected:", output.getvalue())
            self.assertIn("max_new_tokens", output.getvalue())
            trace_path = root / "run"
            traces = list(trace_path.glob("*/traces/*.json"))
            self.assertEqual(len(traces), 1)
            self.assertIn(str(traces[0]), output.getvalue())
            self.assertEqual(json.loads(traces[0].read_text(encoding="utf-8"))["invalid_replies"], 2)
            self.assertEqual(result["statuses"], {"b": "invalid_output"})

    def test_pipeline_prints_v3_initial_plan_issue_even_with_zero_tool_calls(self):
        with fixture() as (root, _, __, ___):
            speech = write_speech_file(root / "speech.jsonl", [
                {"id": "a", "current_text": "List events", "history_user_texts": [
                    {"history_index": 1, "text": "Previous request"}]},
                {"id": "b", "current_text": "List events", "history_user_texts": []},
            ])
            output = io.StringIO()
            env = Env()
            with redirect_stdout(output):
                report = agent.run_full_pipeline(root, sample_ids=["b"], output_dir=root / "run",
                    asr_predictions_path=speech, backend=Backend("{}", "{}"),
                    agent_config=agent.AgentConfig(unified_protocol=False),
                    toolenv_loader=lambda *_: lambda cid: env)
            self.assertEqual(report["statuses"], {"b": "control_error"})
            self.assertIn("calls: 0", output.getvalue())
            self.assertIn("recovery[plan]: plan_contract", output.getvalue())
            self.assertIn("Proposal needs reason, goals, steps", output.getvalue())
            self.assertEqual(env.calls, [])


def lookup_session(backend, env=None):
    tools = [
        {"name": "alarm.list", "kind": "read", "parameters": {"type": "object"}},
        {"name": "alarm.update", "kind": "write", "parameters": {
            "type": "object", "properties": {"alarm_id": {
                "type": "string", "x-ref": "alarm.alarms", "description": "ID from alarm.list",
            }},
        }},
        {"name": "contacts.search", "kind": "read", "parameters": {"type": "object"}},
        {"name": "sms.send", "kind": "write", "parameters": {
            "type": "object", "properties": {"contact_id": {
                "type": "string", "x-ref": "contacts.contacts", "description": "ID from contacts.search",
            }},
        }},
        {"name": "sys.ask_missing_info", "kind": "meta", "parameters": {"type": "object"}},
    ]
    installed = {}
    for tool in tools[:-1]:
        installed.setdefault(tool["name"].split(".")[0], {"tools": []})["tools"].append(tool["name"])
    request = {"id": "lookup", "history": [], "context": {"installed_apps": installed}}
    speech = {"id": "lookup", "current_text": "Update the alarm and text my contact", "history_user_texts": []}
    return agent.AgentSession(request, speech, tools, env or LookupEnv(), backend, agent.AgentConfig(workflow_enabled=False))


class LookupEnv(Env):
    def call(self, name, arguments):
        self.calls.append({"name": name, "arguments": deepcopy(arguments)})
        if name == "alarm.list":
            return {"ok": True, "alarms": [{"id": "observed_alarm", "label": "Morning"}]}
        if name == "contacts.search":
            return {"ok": True, "contacts": [{"id": "observed_contact", "name": "A contact"}]}
        return {"ok": True, "done": True}


class ReferenceLookupTests(unittest.TestCase):
    def test_unknown_alarm_gets_only_the_available_alarm_lookup_in_feedback(self):
        backend = Backend(native_call("alarm.update", {"alarm_id": "guessed_alarm"}))
        run = lookup_session(backend)
        event = run.step()
        self.assertEqual(event["type"], "rejected")
        self.assertEqual(run.env.get_tool_calls(), [])
        feedback = json.loads(run.messages[-1]["content"])
        self.assertFalse(feedback["executed"])
        self.assertEqual(feedback["lookup_required"], [{
            "parameter": "arguments.alarm_id", "reference": "alarm.alarms", "lookup_tools": ["alarm.list"],
        }])
        self.assertEqual(event["lookup_required"], feedback["lookup_required"])
        self.assertNotIn("guessed_alarm", run.known_strings)

    def test_lookup_hints_appear_in_native_tool_schema_before_the_first_reply(self):
        run = lookup_session(Backend())
        definitions = {tool["function"]["name"]: tool["function"] for tool in run.native_tools}
        self.assertIn('"lookup_tools": ["alarm.list"]', definitions["alarm.update"]["description"])
        self.assertIn('"lookup_tools": ["contacts.search"]', definitions["sms.send"]["description"])
        self.assertNotIn("contacts.search", definitions["alarm.update"]["description"])

    def test_two_independent_missing_ids_can_be_recovered_after_successful_lookups(self):
        backend = Backend(
            native_call("alarm.update", {"alarm_id": "guessed_alarm"}),
            native_call("alarm.list"),
            native_call("alarm.update", {"alarm_id": "observed_alarm"}),
            native_call("sms.send", {"contact_id": "guessed_contact"}),
            native_call("contacts.search", {"query": "A contact"}),
            native_call("sms.send", {"contact_id": "observed_contact"}),
            "Done",
        )
        run = lookup_session(backend)
        result = run.run()
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["invalid_replies"], 2)
        self.assertEqual(result["consecutive_invalid_replies"], 0)
        self.assertEqual([call["name"] for call in result["submission"]["tool_calls"]], [
            "alarm.list", "alarm.update", "contacts.search", "sms.send",
        ])
        self.assertNotIn("guessed_alarm", json.dumps(result["submission"]))
        self.assertNotIn("guessed_contact", json.dumps(result["submission"]))
        self.assertEqual(result["submission"]["tool_calls"], run.env.get_tool_calls())

    def test_two_consecutive_guesses_still_stop_without_executing_either(self):
        result = lookup_session(Backend(
            native_call("alarm.update", {"alarm_id": "guess_one"}),
            native_call("alarm.update", {"alarm_id": "guess_two"}),
        )).run()
        self.assertEqual(result["status"], "invalid_output")
        self.assertEqual(result["consecutive_invalid_replies"], 2)
        self.assertEqual(result["submission"]["tool_calls"], [])

    def test_lookup_does_not_make_the_rejected_guess_an_observed_id(self):
        bad = native_call("alarm.update", {"alarm_id": "guessed_alarm"})
        result = lookup_session(Backend(bad, native_call("alarm.list"), bad, bad)).run()
        self.assertEqual(result["status"], "invalid_output")
        self.assertEqual(result["invalid_replies"], 3)
        self.assertEqual([call["name"] for call in result["submission"]["tool_calls"]], ["alarm.list"])

    def test_unsuccessful_tool_result_does_not_reset_rejection_budget_or_verify_ids(self):
        class FailedLookup(LookupEnv):
            def call(self, name, arguments):
                super().call(name, arguments)
                return {"ok": True, "done": False, "alarm_id": "guessed_alarm"}
        bad = native_call("alarm.update", {"alarm_id": "guessed_alarm"})
        result = lookup_session(Backend(bad, native_call("alarm.list"), bad), FailedLookup()).run()
        self.assertEqual(result["status"], "invalid_output")
        self.assertEqual(result["consecutive_invalid_replies"], 2)

    def test_unavailable_lookup_names_from_descriptions_are_not_suggested(self):
        schema = {"properties": {"contact_id": {
            "x-ref": "contacts.contacts", "description": "ID from contacts.search or fake.lookup",
        }}}
        hints = agent.reference_lookup_hints([
            {"name": "contacts.get", "kind": "read"},
            {"name": "fake.lookup", "kind": "write"},
        ], schema)
        self.assertEqual(hints[0]["lookup_tools"], [])

    def test_nested_array_reference_feedback_points_to_the_unknown_field(self):
        schema = {"properties": {"targets": {"type": "array", "items": {
            "type": "object", "properties": {"contact_id": {
                "type": "string", "x-ref": "contacts.contacts", "description": "ID from contacts.search",
            }},
        }}}}
        hints = agent.reference_lookup_hints([{"name": "contacts.search", "kind": "read"}], schema)
        self.assertEqual(hints[0]["parameter"], "arguments.targets[].contact_id")
        self.assertEqual(hints[0]["lookup_tools"], ["contacts.search"])


class PromptIntegrityTests(unittest.TestCase):
    def test_schema_examples_are_omitted_but_types_refs_and_enums_are_preserved(self):
        original = {"type": "object", "required": ["phone"], "properties": {
            "phone": {"type": "string", "description": "Keep the string, for example 0123456789",
                      "examples": ["0123456789"]},
            "mode": {"type": "string", "enum": ["cash", "wallet"]},
            "contact_id": {"type": "string", "x-ref": "contacts.contacts",
                           "description": "ID from contacts.search"},
        }}
        before = deepcopy(original)
        clean = agent.prompt_schema(original)
        self.assertNotIn("0123456789", json.dumps(clean))
        self.assertEqual(clean["required"], ["phone"])
        self.assertEqual(clean["properties"]["phone"]["type"], "string")
        self.assertEqual(clean["properties"]["mode"]["enum"], ["cash", "wallet"])
        self.assertEqual(clean["properties"]["contact_id"], original["properties"]["contact_id"])
        self.assertEqual(original, before)

    def test_example_metadata_is_removed_without_removing_a_property_named_examples(self):
        original = {"examples": [{"examples": "fake"}], "properties": {
            "examples": {"type": "string", "description": "A real parameter"},
            "entries": {"type": "array", "items": {"type": "string", "example": "fake"}},
        }, "additionalProperties": False, "allOf": [True, {"description": "Text, e.g. fake"}]}
        clean = agent.prompt_schema(original)
        self.assertIn("examples", clean["properties"])
        self.assertNotIn("examples", clean)
        self.assertNotIn("fake", json.dumps(clean))
        self.assertIs(clean["additionalProperties"], False)

    def test_accepted_assistant_preamble_is_preserved_for_the_next_generation(self):
        backend = Backend("Inspect the event first; then update it.\n" + native_call(), "Done")
        session(backend).run()
        assistant = next(m for m in backend.messages[1] if m.get("tool_calls"))
        self.assertEqual(assistant["content"], "Inspect the event first; then update it.")

    def test_rejected_assistant_preamble_is_preserved_alongside_validation_feedback(self):
        backend = Backend("I will inspect it.\n" + native_call(), "Done")
        session(backend, Env({"ok": False, "error": "Missing date"})).run()
        assistant = next(m for m in backend.messages[1] if m.get("tool_calls"))
        self.assertEqual(assistant["content"], "I will inspect it.")
        self.assertFalse(json.loads(backend.messages[1][-1]["content"])["executed"])

    def test_history_text_and_calls_share_one_assistant_turn_and_are_not_replayed(self):
        run = session(Backend(native_call(), "Done"))
        run.request["history"] = [{"role": "assistant", "text": "Already checked", "tool_calls": [
            {"name": "calendar.list_events", "arguments": {}},
            {"name": "calendar.list_events", "arguments": {"date": "past"}},
        ], "tool_results": [{"ok": True}, {"ok": True}]}]
        run.messages = agent.build_messages(run.request, run.speech)
        historical = [m for m in run.messages if m["role"] == "assistant"]
        self.assertEqual(len(historical), 1)
        self.assertEqual(historical[0]["content"], "Already checked")
        self.assertEqual(len(historical[0]["tool_calls"]), 2)
        self.assertEqual([m["role"] for m in run.messages[-4:]], ["assistant", "tool", "tool", "user"])
        result = run.run()
        self.assertEqual(len(result["submission"]["tool_calls"]), 1)

    def test_trace_records_exact_initial_inputs_and_runtime_config_without_later_mutation(self):
        backend = Backend(native_call(), "Done")
        run = session(backend)
        result = run.run()
        diagnostics = result["diagnostics"]
        self.assertEqual(diagnostics["initial_messages"], backend.messages[0])
        self.assertEqual(diagnostics["tools"], run.native_tools)
        self.assertEqual(diagnostics["agent_config"]["max_new_tokens"], 256)
        self.assertEqual(diagnostics["initial_messages"][-1]["content"], run.speech["current_text"])
        self.assertNotIn("submission", diagnostics)
        run.messages.append({"role": "user", "content": "later mutation"})
        self.assertNotEqual(diagnostics["initial_messages"], run.messages)


def notebook_cell(role):
    repo = Path(__file__).resolve().parents[2]
    notebook = json.loads((repo / "notebooks/lifemate/test_lifemate.ipynb").read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == role
                or "lifemate-" + role in cell.get("metadata", {}).get("tags", []))
    return "".join(cell["source"])


class KaggleSourcePatchTests(unittest.TestCase):
    """The current notebook carries one immutable source archive, not Git patches."""

    def bootstrap_source(self):
        return notebook_cell("source-bootstrap")

    def unloaded_modules(self):
        return {name: None for name in sys.modules if name == "lifemate" or name.startswith("lifemate.")}

    def definitions(self):
        source = self.bootstrap_source()
        tree = ast.parse(source)
        install = next(node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "PROJECT_ROOT" for t in node.targets))
        namespace = {}
        exec(compile("\n".join(source.splitlines()[:install.lineno - 1]), "source-definitions", "exec"), namespace)
        return namespace

    def install(self, root, namespace=None):
        namespace = {"SOURCE_WORK_ROOT": root} if namespace is None else namespace
        with patch.dict(sys.modules, self.unloaded_modules()), patch.object(sys, "path", list(sys.path)), \
             redirect_stdout(io.StringIO()):
            exec(compile(self.bootstrap_source(), "source-bootstrap", "exec"), namespace)
        return namespace

    def test_archive_contains_exact_current_agent_pipeline_and_evaluation_source(self):
        repo = Path(__file__).resolve().parents[2]
        namespace = self.definitions()
        blob = base64.b64decode(namespace["SOURCE_ARCHIVE_BASE64"])
        self.assertEqual(hashlib.sha256(blob).hexdigest(), namespace["SOURCE_BUNDLE_SHA256"])
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            names = set(archive.namelist())
            expected = {p.relative_to(repo).as_posix() for p in (repo / "src/lifemate").rglob("*.py")}
            expected.add("configs/lifemate/agent-evaluation.json")
            self.assertEqual(names, expected)
            self.assertIn("src/lifemate/evaluation/runner.py", names)
            self.assertIn("src/lifemate/evaluation/experiment.py", names)
            for name in names:
                self.assertEqual(archive.read(name), (repo / name).read_bytes(), name)
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(),
                                 namespace["SOURCE_FILES_SHA256"][name])
            self.assertFalse(any("state.json" in name or name.endswith(".safetensors") for name in names))

    def test_same_source_setup_is_idempotent_and_preserves_preloaded_backend(self):
        with fixture() as (root, _, __, ___):
            backend = object()
            namespace = {"SOURCE_WORK_ROOT": root, "qwen_backend": backend}
            self.install(root, namespace)
            installed = namespace["PROJECT_ROOT"]
            self.install(root, namespace)
            self.assertEqual(namespace["PROJECT_ROOT"], installed)
            self.assertIs(namespace["qwen_backend"], backend)
            self.assertEqual((installed / "src/lifemate/agent.py").read_bytes(),
                             (Path(__file__).resolve().parents[2] / "src/lifemate/agent.py").read_bytes())

    def test_changed_agent_source_is_preserved_and_not_overwritten(self):
        with fixture() as (root, _, __, ___):
            namespace = self.install(root)
            source = namespace["PROJECT_ROOT"] / "src/lifemate/agent.py"
            source.write_text("# user changes\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                self.install(root)
            self.assertEqual(source.read_text(encoding="utf-8"), "# user changes\n")

    def test_changed_helper_blocks_all_missing_source_writes(self):
        with fixture() as (root, _, __, ___):
            namespace = self.install(root)
            source = namespace["PROJECT_ROOT"] / "src/lifemate/agent.py"
            helper = source.parent / "pipeline/actor.py"
            helper.write_text("# user actor changes\n", encoding="utf-8")
            source.unlink()
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                self.install(root)
            self.assertFalse(source.exists())
            self.assertEqual(helper.read_text(encoding="utf-8"), "# user actor changes\n")

    def test_corrupted_or_escaping_archive_is_rejected_before_writing(self):
        for escape in (False, True):
            with self.subTest(escape=escape), fixture() as (root, _, __, ___):
                namespace = self.definitions()
                if escape:
                    content = io.BytesIO()
                    with zipfile.ZipFile(content, "w") as archive:
                        archive.writestr("../outside.py", b"test")
                    blob = content.getvalue()
                    namespace["SOURCE_ARCHIVE_BASE64"] = base64.b64encode(blob).decode()
                    namespace["SOURCE_BUNDLE_SHA256"] = hashlib.sha256(blob).hexdigest()
                    namespace["SOURCE_FILES_SHA256"] = {"../outside.py": hashlib.sha256(b"test").hexdigest()}
                else:
                    namespace["SOURCE_ARCHIVE_BASE64"] = base64.b64encode(b"corrupted").decode()
                with patch.dict(sys.modules, self.unloaded_modules()), self.assertRaisesRegex(RuntimeError, "checksum"):
                    namespace["install_embedded_source"](root / "source")
                self.assertFalse((root / "source").exists())
                self.assertFalse((root / "outside.py").exists())

    def test_mixed_imported_source_requires_restart_without_changing_files(self):
        with fixture() as (root, _, __, ___):
            namespace = self.definitions()
            other = SimpleNamespace(__file__=str(root / "other/lifemate/__init__.py"))
            with patch.dict(sys.modules, {**self.unloaded_modules(), "lifemate": other}), self.assertRaisesRegex(RuntimeError, "Restart Session"):
                namespace["install_embedded_source"](root / "source")
            self.assertFalse((root / "source").exists())


class EmbeddedTraceReaderTests(unittest.TestCase):
    def source(self):
        return notebook_cell("trace")

    def write_results(self, path, sample_id):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"id": sample_id, "status": "finished", "trace": [
            {"type": "model_reply", "raw": "MODEL_REPLY_SENTINEL"},
            {"type": "tool_result", "result": {"ok": True}},
            {"type": "terminal_check", "candidate": {"type": "finish"}},
        ]}) + "\n", encoding="utf-8")

    def test_trace_reader_can_open_saved_results_without_model_or_run_namespace(self):
        with fixture() as (root, _, __, ___):
            saved = root / "saved/results.jsonl"
            self.write_results(saved, "train_demo")
            namespace = {"TRACE_RESULTS_FILE": saved, "TRACE_SAMPLE_ID": "train_demo"}
            output = io.StringIO()
            with redirect_stdout(output):
                exec(compile(self.source(), "trace-reader", "exec"), namespace)
            self.assertNotIn("qwen_backend", namespace)
            self.assertIn("MODEL_REPLY_SENTINEL", output.getvalue())
            self.assertIn("terminal_check", output.getvalue())
            self.assertIn("train_demo", output.getvalue())

    def test_current_result_selects_exact_run_and_does_not_search_other_runs(self):
        with fixture() as (root, _, __, ___):
            chosen = root / "current/results.jsonl"
            self.write_results(chosen, "train_demo")
            self.write_results(root / "other/results.jsonl", "UNRELATED_RUN")
            output = io.StringIO()
            with redirect_stdout(output):
                exec(compile(self.source(), "trace-reader", "exec"), {
                    "LAST_RESULT": {"traces": str(chosen)}, "TRACE_SAMPLE_ID": "train_demo"})
            self.assertIn(str(chosen), output.getvalue())
            self.assertNotIn("UNRELATED_RUN", output.getvalue())

    def test_missing_trace_configuration_prints_guidance_without_name_error(self):
        output = io.StringIO()
        with redirect_stdout(output):
            exec(compile(self.source(), "trace-reader", "exec"), {})
        self.assertIn("TRACE_SAMPLE_ID", output.getvalue())
        with self.assertRaisesRegex(RuntimeError, "TRACE_RESULTS_FILE"):
            exec(compile(self.source(), "trace-reader", "exec"), {"TRACE_SAMPLE_ID": "train_demo"})


if __name__ == "__main__":
    unittest.main()
