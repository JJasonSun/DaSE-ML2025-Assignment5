import unittest

from agents.hybrid_retrieval_agent import HybridRetrievalAgent


class HybridRetrievalAgentTests(unittest.TestCase):
    def test_retrieval_queries_include_identifiers_quotes_and_operation_terms(self):
        agent = HybridRetrievalAgent(api_key="dummy", base_url="http://localhost/v1")
        question = (
            "For Project Veridian [CONFIG_CRYPTO_HASH_VALIDATOR_SIGNATURE_2024] (GIC-779-Alpha), subtract the processing fee, "
            "multiply by Regulation Code 394, then divide by Statute 50."
        )

        queries = agent._build_retrieval_queries(question)

        self.assertTrue(queries[0].startswith("For Project Veridian"))
        self.assertIn("GIC-779-Alpha", queries)
        self.assertIn("CONFIG_CRYPTO_HASH_VALIDATOR_SIGNATURE_2024", queries)
        self.assertIn("Project Veridian", queries)
        self.assertIn("difference subtract minus", queries)
        self.assertIn("multiplier product", queries)
        self.assertIn("divisor quotient", queries)

    def test_multi_evidence_questions_expand_window(self):
        agent = HybridRetrievalAgent(api_key="dummy", base_url="http://localhost/v1")
        multi_question = "Find value A and value B, then multiply by C and finally divide by D."
        simple_question = "What is the project launch date?"

        self.assertGreater(agent._dynamic_rerank_top_n(multi_question), agent.rerank_top_n)
        self.assertEqual(agent._dynamic_neighbor_radius(multi_question), 2)
        self.assertEqual(agent._dynamic_rerank_top_n(simple_question), agent.rerank_top_n)
        self.assertEqual(agent._dynamic_neighbor_radius(simple_question), 1)

    def test_full_context_trace_is_recorded(self):
        agent = HybridRetrievalAgent(api_key="dummy", base_url="http://localhost/v1")
        prompt = {
            "question": "What is the value?",
            "context_data": {
                "files": [
                    {"filename": "a.txt", "modified_content": "The value is 42."},
                    {"filename": "b.txt", "modified_content": "Other context."},
                ]
            },
        }

        context = agent._select_context(prompt, prompt["question"])

        self.assertIn("The value is 42", context)
        self.assertEqual(agent._last_retrieval_trace["retrieval_mode"], "full_context")
        self.assertEqual(agent._last_retrieval_trace["evidence_block_count"], 2)
        self.assertEqual(agent._last_retrieval_trace["retrieved_files"], ["a.txt", "b.txt"])
        self.assertEqual(agent._last_retrieval_trace["embedding_model"], agent.embedding_model)
        self.assertEqual(agent._last_retrieval_trace["rerank_model"], agent.rerank_model)


if __name__ == "__main__":
    unittest.main()
