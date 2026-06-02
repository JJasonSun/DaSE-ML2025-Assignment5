import pathlib
import unittest

from agents.tool_augmented_agent import ToolAugmentedAgent


class StubPlannerAgent(ToolAugmentedAgent):
    def __init__(self, plan, repaired_plan=None):
        super().__init__(api_key="dummy", base_url="http://localhost/v1")
        self.plan = plan
        self.repaired_plan = repaired_plan
        self.repair_calls = 0

    async def _create_operation_plan(self, question, context):
        return self._normalize_plan(self.plan, question)

    async def _repair_operation_plan(self, question, context, plan, failure_reason):
        self.repair_calls += 1
        return self._normalize_plan(self.repaired_plan or {}, question) if self.repaired_plan else {}

    async def _fallback(self, prompt, reason, task_type, extraction=None):
        raise AssertionError(f"unexpected fallback: {reason}")


class ToolAugmentedAgentToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_caesar_decode_from_planner_shift(self):
        agent = StubPlannerAgent(
            {
                "task_type": "encoding",
                "evidence_items": [
                    {"entity": "encoded_signal", "value": "YBSYX857", "unit_or_type": "payload"},
                    {"entity": "shift_amount", "value": 10, "unit_or_type": "shift"},
                    {"entity": "method", "value": "Caesar shift", "unit_or_type": "method"},
                ],
                "operation": "caesar_decode(encoded_signal, shift=shift_amount)",
            }
        )

        answer = await agent.evaluate_model({"question": "Decode the encoded signal.", "context": ""})

        self.assertEqual(answer, "ORION857")
        self.assertEqual(agent.last_trace["execution_mode"], "planned_tool")
        self.assertEqual(agent.last_trace["planner_attempts"], 1)

    async def test_reverse_hex_and_base64_tools(self):
        cases = [
            (
                {
                    "task_type": "string_analysis",
                    "evidence_items": [{"entity": "confirmation_code", "value": "388NOIRO"}],
                    "operation": "confirmation_code[::-1]",
                },
                "Read backwards.",
                "ORION883",
            ),
            (
                {
                    "task_type": "encoding",
                    "evidence_items": [
                        {"entity": "payload", "value": "544954414E313339", "unit_or_type": "payload"},
                        {"entity": "method", "value": "ASCII hex", "unit_or_type": "method"},
                    ],
                },
                "Decode the ASCII hex payload.",
                "TITAN139",
            ),
            (
                {
                    "task_type": "encoding",
                    "evidence_items": [
                        {"entity": "payload", "value": "UEhPRU5JWDY0NA==", "unit_or_type": "payload"},
                        {"entity": "method", "value": "base64", "unit_or_type": "method"},
                    ],
                },
                "Decode this base64 payload.",
                "PHOENIX644",
            ),
        ]

        for plan, question, expected in cases:
            with self.subTest(expected=expected):
                answer = await StubPlannerAgent(plan).evaluate_model({"question": question, "context": ""})
                self.assertEqual(answer, expected)

    async def test_string_count_difference_operation(self):
        agent = StubPlannerAgent(
            {
                "task_type": "string_analysis",
                "evidence_items": [
                    {"entity": "verification_hash", "value": "BfA850C69Aa3Ed82714C1Dac2BD3a8Fc94eFFEcfEe25481c836FFF"}
                ],
                "operation": 'abs(verification_hash.count("F") - verification_hash.count("3"))',
            }
        )

        answer = await agent.evaluate_model({"question": "Count F and 3, then return the absolute difference.", "context": ""})

        self.assertEqual(answer, "3")

    async def test_md5_prefix_allows_import_statement_prefix(self):
        value = "feABEcA1bBd2dE5C769Be8bDCCCDF3AEBAaAbad064DF8df69de5d5EA1e3454a2d0E4ab1a7CFE50ae8bcccc23dECC96C06DEaf6bea5A1E01F6EAcfF7b91Cbf1F74D1e7B4Df6Dd9b1A07b1DfC2Efe5dDBA0B75AC6E2eeD5"
        agent = StubPlannerAgent(
            {
                "task_type": "computation",
                "evidence_items": [{"entity": "master_token_value", "value": value, "unit_or_type": "string"}],
                "operation": "import hashlib; hashlib.md5(master_token_value.encode()).hexdigest()[:8]",
            }
        )

        answer = await agent.evaluate_model({"question": "Calculate the first 8 characters of its MD5 hash.", "context": ""})

        self.assertEqual(answer, "6c2bb819")

    async def test_date_difference_and_weekday_tools(self):
        day_count = await StubPlannerAgent(
            {
                "task_type": "date_time",
                "evidence_items": [
                    {"entity": "start_date", "value": "2033-7-18", "unit_or_type": "date"},
                    {"entity": "end_date", "value": "2033-10-16", "unit_or_type": "date"},
                ],
                "operation": "(end_date - start_date).days",
                "expected_answer_format": "integer",
            }
        ).evaluate_model({"question": "How many days elapsed between the two dates?", "context": ""})
        weekday = await StubPlannerAgent(
            {
                "task_type": "date_time",
                "evidence_items": [{"entity": "activation_date", "value": "December 23, 2047", "unit_or_type": "date"}],
                "operation": "weekday(activation_date)",
                "expected_answer_format": "weekday",
            }
        ).evaluate_model({"question": "What day of the week is the activation?", "context": ""})

        self.assertEqual(day_count, "90")
        self.assertEqual(weekday, "Monday")

    async def test_big_integer_arithmetic_uses_generic_variable_names(self):
        agent = StubPlannerAgent(
            {
                "task_type": "computation",
                "evidence_items": [
                    {"entity": "first_value", "value": "5383420629290754"},
                    {"entity": "second_value", "value": "1536037696995403"},
                    {"entity": "multiplier", "value": "775"},
                    {"entity": "divisor", "value": "48"},
                ],
                "operation": "(first_value - second_value) * multiplier / divisor",
                "expected_answer_format": "integer",
            }
        )

        answer = await agent.evaluate_model({"question": "Compute the precise metric.", "context": ""})

        self.assertEqual(answer, "62119203594352021")

    async def test_integer_square_root_expression(self):
        agent = StubPlannerAgent(
            {
                "task_type": "computation",
                "evidence_items": [{"entity": "guard_shift_pattern", "value": "152803987454881"}],
                "operation": "int(guard_shift_pattern ** 0.5)",
            }
        )

        answer = await agent.evaluate_model({"question": "Calculate the integer square root.", "context": ""})

        self.assertEqual(answer, "12361391")

    async def test_missing_evidence_triggers_replanner_once(self):
        agent = StubPlannerAgent(
            {
                "task_type": "encoding",
                "evidence_items": [{"entity": "encoded_signal", "value": "YBSYX857", "unit_or_type": "payload"}],
                "operation": "caesar_decode(encoded_signal, shift=shift_amount)",
            },
            repaired_plan={
                "task_type": "encoding",
                "evidence_items": [
                    {"entity": "encoded_signal", "value": "YBSYX857", "unit_or_type": "payload"},
                    {"entity": "shift_amount", "value": 10, "unit_or_type": "shift"},
                    {"entity": "method", "value": "Caesar shift", "unit_or_type": "method"},
                ],
                "operation": "caesar_decode(encoded_signal, shift=shift_amount)",
            },
        )

        answer = await agent.evaluate_model({"question": "Decode the encoded signal.", "context": ""})

        self.assertEqual(answer, "ORION857")
        self.assertEqual(agent.repair_calls, 1)
        self.assertEqual(agent.last_trace["planner_attempts"], 2)
        self.assertEqual(agent.last_trace["execution_mode"], "repaired_planned_tool")

    async def test_invalid_decoded_output_returns_failed_trace(self):
        agent = StubPlannerAgent(
            {
                "task_type": "encoding",
                "evidence_items": [
                    {"entity": "payload", "value": "AAAA", "unit_or_type": "payload"},
                    {"entity": "method", "value": "base64", "unit_or_type": "method"},
                ],
            }
        )

        answer = await agent.evaluate_model({"question": "Decode this base64 payload.", "context": ""})

        self.assertEqual(answer, "Unknown")
        self.assertEqual(agent.last_trace["path"], "tool_augmented_failed")
        self.assertEqual(agent.last_trace["tool_failure_reason"], "invalid_decoded_output")
        self.assertEqual(agent.last_trace["execution_mode"], "failed")

    async def test_empty_plan_returns_unknown_with_trace_not_hybrid_guess(self):
        agent = StubPlannerAgent({})

        answer = await agent.evaluate_model({"question": "What is the answer?", "context": ""})

        self.assertEqual(answer, "Unknown")
        self.assertEqual(agent.last_trace["path"], "tool_augmented_failed")
        self.assertEqual(agent.last_trace["tool_failure_reason"], "unsupported_operation")

    def test_no_case_specific_alias_maps_in_tool_agent(self):
        source = pathlib.Path("agents/tool_augmented_agent.py").read_text(encoding="utf-8").lower()
        forbidden_fragments = [
            "inventory id alpha",
            "hyperdrive calibration constant",
            "nexus stabilization factor",
            "primary antenna frequency coefficient",
            "secondary communication wavelength constant",
            "master access code part a",
        ]

        for fragment in forbidden_fragments:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
