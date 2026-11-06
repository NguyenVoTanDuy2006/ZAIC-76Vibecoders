"""Explicit units are source facts; ambiguous action/value binding stays model-owned."""
import unittest

from lifemate.pipeline import quantities
from test_grounding import make_session
from test_workflow import call


class MoneyQuantityTests(unittest.TestCase):
    def test_numeric_spoken_grouped_and_decimal_vnd_keep_original_source(self):
        cases = {"Nạp 75 ngàn": 75000, "Gửi 2,5 triệu đồng": 2500000,
                 "Chuyển bảy mươi lăm nghìn": 75000, "Trả một trăm linh năm ngàn": 105000,
                 "Nạp 125.000 đồng": 125000, "Nạp 125,000 VND": 125000,
                 "Nạp 80k": 80000, "Nạp 90000₫": 90000, "Nạp mười lăm nghìn": 15000}
        for text, expected in cases.items():
            with self.subTest(text=text):
                spans = quantities.money_spans(text)
                self.assertEqual(len(spans), 1)
                self.assertEqual(spans[0]["value"], expected)
                self.assertEqual(text[spans[0]["start"]:spans[0]["end"]], spans[0]["text"])

    def test_ambiguous_threshold_relative_amount_and_correction_are_hints_only(self):
        for text in ("Nếu giá dưới 125 triệu thì mua", "Chuyển thêm 70 ngàn", "Gửi khoảng 20 ngàn",
                     "Chuyển 20 ngàn và nạp 30 ngàn", "Nạp 20 nghìn à không 30 nghìn",
                     "Nạp 75 ngàn rồi chuyển tiền điện cho người nhà",
                     "Trả phí 30 nghìn", "Nạp nửa triệu", "Nạp từ 10 đến 20 nghìn",
                     "Chuyển 20 ngàn USD", "Chuyển một nửa của 70 ngàn"):
            with self.subTest(text=text):
                self.assertIsNone(quantities.direct_amount(text))

    def test_compound_or_colloquial_amount_never_produces_partial_normalization(self):
        for text in ("Chuyển hai triệu ba", "Gửi hai triệu rưỡi", "Nạp một triệu hai trăm nghìn"):
            with self.subTest(text=text):
                self.assertEqual(quantities.money_spans(text), [])

    def test_counts_and_proper_names_are_not_money(self):
        for text in ("Đặt mục tiêu sáu nghìn bước", "Tờ sau Đồng Tháp", "Chạy 2 nghìn mét"):
            with self.subTest(text=text):
                self.assertEqual(quantities.money_spans(text), [])

    def test_unambiguous_amount_mismatch_rejected_before_tool_and_corrected_by_model(self):
        run = make_session("Nạp 75 ngàn", call("payments.top_up", amount=75),
                           call("payments.top_up", amount=75000), "Done", results=[{"ok": True}])
        self.assertEqual(run.run()["status"], "finished")
        self.assertEqual(run.env.calls[0]["arguments"]["amount"], 75000)
        self.assertEqual(len(run.env.calls), 1)
        rejection = next(e for e in run.trace if e["type"] == "rejected")
        self.assertEqual(rejection["repair"]["code"], "money_grounding")
        self.assertEqual(rejection["repair"]["source_constraints"][0]["money_source"]["value"], 75000)

    def test_multiple_amounts_are_not_auto_bound_to_one_action(self):
        for text, amount in (("Nạp 75 ngàn và chuyển 20 nghìn", 75000),
                             ("Nạp 75 ngàn với chuyển tiền điện", 845000)):
            run = make_session(text, call("payments.top_up", amount=amount), "Done", results=[{"ok": True}])
            self.assertEqual(run.run()["status"], "finished")
            self.assertEqual(run.invalid_replies, 0)

    def test_guard_does_not_rewrite_or_reject_unknown_units_or_foreign_currency(self):
        for text, args in (("Nạp vài chục", {"amount": 75000}),
                           ("Chuyển 20 ngàn", {"amount": 20000, "currency": "USD"})):
            with self.subTest(text=text):
                run = make_session(text, call("payments.top_up", **args), "Done", results=[{"ok": True}])
                self.assertEqual(run.run()["status"], "finished")
                self.assertEqual(run.env.calls[0]["arguments"], args)


if __name__ == "__main__":
    unittest.main()
