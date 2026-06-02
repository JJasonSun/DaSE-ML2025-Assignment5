import unittest

from agents.tool_augmented_agent import ToolAugmentedAgent


class StubToolAgent(ToolAugmentedAgent):
    def __init__(self, extraction, task_type="encoding", repaired_extraction=None):
        super().__init__(api_key="dummy", base_url="http://localhost/v1")
        self.extraction = extraction
        self.repaired_extraction = repaired_extraction
        self.task_type = task_type
        self.repair_calls = 0

    def _classify_task(self, question):
        return self.task_type

    async def _extract_structured_evidence(self, question, context, task_type):
        return self.extraction

    async def _repair_structured_evidence(self, question, context, task_type, extraction):
        self.repair_calls += 1
        return self.repaired_extraction or {}

    async def _fallback(self, prompt, reason, task_type, extraction=None):
        raise AssertionError(f"unexpected fallback: {reason}")


class ToolAugmentedAgentToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_caesar_decode_does_not_get_preempted_by_base64_shape(self):
        agent = StubToolAgent(
            {
                "evidence_items": [
                    {"entity": "encoded_signal", "value": "YBSYX857"},
                    {"entity": "shift_amount", "value": 10},
                ],
                "operation": "caesar_decode(encoded_signal, shift=shift_amount)",
            }
        )
        answer = await agent.evaluate_model({"question": "Decode the encoded signal shifted by 10.", "context": ""})

        self.assertEqual(answer, "ORION857")
        self.assertEqual(agent.last_trace["path"], "tool_augmented")

    async def test_reverse_hex_and_base64_operations(self):
        cases = [
            (
                "string_analysis",
                "Read the confirmation code backwards.",
                {"evidence_items": [{"entity": "confirmation_code", "value": "388NOIRO"}], "operation": "confirmation_code[::-1]"},
                "ORION883",
            ),
            (
                "encoding",
                "Decode the ASCII hex payload.",
                {"evidence_items": [{"entity": "payload", "value": "544954414E313339"}]},
                "TITAN139",
            ),
            (
                "encoding",
                "Decode this base64 payload.",
                {"evidence_items": [{"entity": "payload", "value": "UEhPRU5JWDY0NA=="}]},
                "PHOENIX644",
            ),
        ]

        for task_type, question, extraction, expected in cases:
            with self.subTest(expected=expected):
                agent = StubToolAgent(extraction, task_type=task_type)
                answer = await agent.evaluate_model({"question": question, "context": ""})
                self.assertEqual(answer, expected)
                self.assertEqual(agent.last_trace["path"], "tool_augmented")

    async def test_caesar_operation_with_direction_argument(self):
        agent = StubToolAgent(
            {
                "task_type": "encoding",
                "evidence_items": [
                    {"entity": "intercepted_cipher", "value": "ODPEGD155"},
                    {"entity": "shift_value", "value": "3"},
                ],
                "operation": "caesar_decode(intercepted_cipher, shift_value, direction='backward')",
                "constraints": ["Apply reverse Caesar shift backward by 3."],
            }
        )

        answer = await agent.evaluate_model({"question": "Decode the intercepted operation identifier.", "context": ""})

        self.assertEqual(answer, "LAMBDA155")

    async def test_web_safe_encoding_question_is_classified_and_decoded(self):
        agent = ToolAugmentedAgent(api_key="dummy", base_url="http://localhost/v1")
        question = (
            "The identifier was first processed using a simple ASCII-based substitution "
            "and then a standard web encoding. What is the original clear-text identifier?"
        )
        context = (
            "The last intercepted transmission contained only the string: R0FNTUExNTk=.\n"
            "All identifiers follow the pattern of a Greek letter followed by three digits."
        )
        task_type = agent._classify_task(question)

        answer = agent._solve_with_tools(question, context, task_type, {"evidence_items": []})

        self.assertEqual(task_type, "encoding")
        self.assertEqual(answer, "GAMMA159")

    async def test_hex_digit_sum_uses_context_hash_not_quoted_project_name(self):
        context = "System Log Trace ID: 3b092E1E3D85D53eB9Bc2fFCE0eCFe82df014eCF26beE7D389b0baCee72aFe50E1f78a1bceeDB7caBc49Ea9d13dE035AADf59450a5af8cD88fdD6A5F"
        agent = StubToolAgent(
            {
                "task_type": "string_analysis",
                "evidence_items": [],
                "operation": "sum(hexadecimal_digits)",
            },
            task_type="string_analysis",
        )

        answer = await agent.evaluate_model(
            {
                "question": "Using the unique transaction hash for the 'Aetherium Vault' protocol deployment, calculate the sum of all hexadecimal digits (0-9) present in the hash.",
                "context": context,
            }
        )

        self.assertEqual(answer, "235")

    async def test_per_layer_key_uses_labeled_context_numbers_when_extraction_is_empty(self):
        context = "The Master Key encryption seed value is 8515527329885874\nThe required number of encryption layers is 7"
        agent = StubToolAgent({"task_type": "computation", "evidence_items": []}, task_type="computation")

        answer = await agent.evaluate_model(
            {
                "question": "What is the unique per-layer decryption key value from the Master Key encryption seed value and the number of encryption layers?",
                "context": context,
            }
        )

        self.assertEqual(answer, "1216503904269410")

    async def test_caesar_shift_can_be_inferred_from_series_prefix(self):
        context = "Agents confirm the target organization uses 3-digit suffixes on all ALPHA series operations."
        agent = StubToolAgent(
            {
                "task_type": "encoding",
                "evidence_items": [
                    {"entity": "encoded_payload", "value": "JUYQJ844"},
                    {"entity": "shift_value", "value": "-20"},
                    {"entity": "encoding_method", "value": "Caesar Cipher (Julius Protocol)"},
                ],
                "operation": "decode_caesar(encoded_payload, shift_value)",
            },
            task_type="encoding",
        )

        answer = await agent.evaluate_model(
            {
                "question": "Decrypt the encoded project identifier to determine the ALPHA series operation.",
                "context": context,
            }
        )

        self.assertEqual(answer, "ALPHA844")

    async def test_context_label_corrects_overextracted_numeric_operand(self):
        context = (
            "In the Chronos Vault security system, the Master Access Code Part A is 9015252083871514\n"
            "Backup Code Part B: 5769316421221899, Temporal Multiplier: 855, Security Divisor: 25"
        )
        agent = StubToolAgent(
            {
                "task_type": "computation",
                "evidence_items": [
                    {"entity": "master_access_code_part_a", "value": "90152520838715140"},
                    {"entity": "backup_code_part_b", "value": "5769316421221899"},
                    {"entity": "temporal_multiplier", "value": "855"},
                    {"entity": "security_divisor", "value": "25"},
                ],
                "operation": "(master_access_code_part_a - backup_code_part_b) * temporal_multiplier // security_divisor",
            },
            task_type="computation",
        )

        answer = await agent.evaluate_model({"question": "Calculate the final security token.", "context": context})

        self.assertEqual(answer, "111010999662616833")

    async def test_wavelength_difference_is_computation_and_fills_missing_operand(self):
        context = (
            "Planetary Research Institute - Mars Rover Project: Primary antenna frequency coefficient: 4758590176084296\n"
            "Deep Space Network - Jupiter Mission: Secondary communication wavelength constant: 4619815457882402"
        )
        agent = StubToolAgent(
            {
                "task_type": "computation",
                "evidence_items": [
                    {
                        "entity": "secondary_communication_wavelength_constant",
                        "value": "4619815457882402",
                        "unit_or_type": "integer",
                    }
                ],
                "operation": "abs(primary_antenna_frequency_coefficient_mars - secondary_communication_wavelength_constant)",
            },
            task_type="computation",
        )
        question = (
            "Calculate the absolute frequency differential between the Primary antenna frequency coefficient "
            "and Secondary communication wavelength constant."
        )

        answer = await agent.evaluate_model({"question": question, "context": context})

        self.assertEqual(ToolAugmentedAgent(api_key="dummy", base_url="http://localhost/v1")._classify_task(question), "computation")
        self.assertEqual(answer, "138774718201894")

    async def test_date_evidence_routes_to_date_difference(self):
        agent = StubToolAgent(
            {
                "task_type": "computation",
                "evidence_items": [
                    {"entity": "final_review_completion_date", "value": "2033-7-18", "unit_or_type": "date"},
                    {"entity": "deployment_ceremony_date", "value": "2033-10-16", "unit_or_type": "date"},
                ],
                "operation": "(datetime(2033, 10, 16) - datetime(2033, 7, 18)).days",
            },
            task_type="computation",
        )

        answer = await agent.evaluate_model(
            {"question": "How many days are there between final review completion and deployment ceremony?", "context": ""}
        )

        self.assertEqual(answer, "90")

    async def test_inventory_batch_formula_requires_and_uses_all_four_labels(self):
        context = (
            "Inventory ID Alpha: 7939552063979389 units\n"
            "Inventory ID Beta: 6902211753510982 units\n"
            "Production Cycle ID Gamma: 603 cycles\n"
            "Batch Size ID Delta: 25 units/batch"
        )
        agent = StubToolAgent(
            {
                "task_type": "computation",
                "evidence_items": [
                    {"entity": "production_cycles_gamma", "value": "603"},
                    {"entity": "batch_size_delta", "value": "25"},
                ],
                "operation": "Inventory Alpha and Inventory Beta values are missing from context.",
            },
            task_type="computation",
        )

        answer = await agent.evaluate_model(
            {
                "question": "Calculate batches from Inventory Alpha minus Inventory Beta across Cycle Gamma given Batch Delta.",
                "context": context,
            }
        )

        self.assertEqual(answer, "25020648288497976")

    async def test_resource_efficiency_formula_fills_allocation_aliases(self):
        context = (
            "Sector 7 Resource Ledger: Initial allocation value = 5383420629290754 units\n"
            "Sector 7 Resource Ledger: Final allocation value after Phase 1 redistribution = 1536037696995403 units\n"
            "Hyperdrive Calibration Constant (HCC-775) = 775 cycles\n"
            "Nexus Stabilization Factor (NSF-48) = 48 quanta"
        )
        agent = StubToolAgent(
            {
                "task_type": "computation",
                "evidence_items": [
                    {"entity": "sector_7_initial_allocation", "value": "5383420629290754"},
                    {"entity": "nexus_stabilization_factor", "value": "48"},
                ],
                "operation": (
                    "(sector_7_initial_allocation - sector_7_final_allocation) "
                    "* hyperdrive_calibration_constant / nexus_stabilization_factor"
                ),
            },
            task_type="computation",
        )

        answer = await agent.evaluate_model(
            {
                "question": "Compute ((Initial - Final) * HCC-775) / NSF-48.",
                "context": context,
            }
        )

        self.assertEqual(answer, "62119203594352021")

    async def test_digit_sum_generator_operation(self):
        token = "51C9AE6Dae32bfAD6DE6EB7c4FafBbFbADD90DeA15faDC2d86241F66EFF5ee896c8B60Eff2AfdcFFC7f16F3badf71790d46DceB7E572fB8E6fc7deD2BEa7a4240b2fF1d417C2C3EA6B45B"
        agent = StubToolAgent(
            {
                "task_type": "computation",
                "evidence_items": [{"entity": "initialization_token", "value": token}],
                "operation": "sum(int(c) for c in initialization_token if c.isdigit())",
            },
            task_type="computation",
        )

        answer = await agent.evaluate_model({"question": "Calculate the sum of all numerical digits in the token.", "context": ""})

        self.assertEqual(answer, "290")

    async def test_augustus_shift_uses_four_position_rule(self):
        agent = StubToolAgent(
            {
                "task_type": "encoding",
                "evidence_items": [
                    {"entity": "encoded_string", "value": "SQIKE692"},
                    {"entity": "cipher_type", "value": "simple shift cipher historically attributed to Augustus"},
                    {"entity": "shift_rule", "value": "Caesar cipher shift +3"},
                ],
                "operation": "apply_shift(encoded_string, -3, preserve_digits=True)",
            },
            task_type="encoding",
        )

        answer = await agent.evaluate_model({"question": "Decode the Augustus shift cipher.", "context": ""})

        self.assertEqual(answer, "OMEGA692")

    async def test_integer_square_root_from_guard_shift_pattern(self):
        context = (
            "The first number is the 'Guard Shift Pattern': 152803987454881.\n"
            "Important: The lock code is actually the integer square root of the 'Guard Shift Pattern' number."
        )
        agent = StubToolAgent({"task_type": "computation", "evidence_items": []}, task_type="computation")

        answer = await agent.evaluate_model(
            {
                "question": "Retrieve the Guard Shift Pattern number, then calculate the integer square root to find the final lock code.",
                "context": context,
            }
        )

        self.assertEqual(answer, "12361391")

    async def test_missing_shift_triggers_one_repair_before_failure(self):
        agent = StubToolAgent(
            {
                "evidence_items": [{"entity": "encoded_signal", "value": "YBSYX857"}],
                "operation": "caesar_decode(encoded_signal, shift=shift_amount)",
            },
            repaired_extraction={
                "evidence_items": [
                    {"entity": "encoded_signal", "value": "YBSYX857"},
                    {"entity": "shift_amount", "value": 10},
                ],
                "operation": "caesar_decode(encoded_signal, shift=shift_amount)",
            },
        )

        answer = await agent.evaluate_model({"question": "Decode the encoded signal using Caesar shift.", "context": ""})

        self.assertEqual(answer, "ORION857")
        self.assertEqual(agent.repair_calls, 1)
        self.assertTrue(agent.last_trace["repair_attempted"])

    async def test_invalid_decoded_output_returns_unknown_with_reason(self):
        agent = StubToolAgent({"evidence_items": [{"entity": "payload", "value": "AAAA"}]})

        answer = await agent.evaluate_model({"question": "Decode this base64 payload.", "context": ""})

        self.assertEqual(answer, "Unknown")
        self.assertEqual(agent.last_trace["path"], "tool_augmented_failed")
        self.assertEqual(agent.last_trace["fallback_reason"], "invalid_decoded_output")


if __name__ == "__main__":
    unittest.main()
