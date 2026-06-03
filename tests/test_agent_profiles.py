import unittest

from core.agent_profiles import agent_profile_for


class AgentProfilesTests(unittest.TestCase):
    def test_reads_profile_from_agent_class(self):
        profile = agent_profile_for("agents.tool_augmented_agent:ToolAugmentedAgent")

        self.assertEqual(profile["name"], "ToolAugmentedAgent")
        self.assertIn("Planner-first", profile["positioning"])
        self.assertIn("operation_plan", profile["analysis_focus"])

    def test_unknown_agent_uses_default_profile(self):
        profile = agent_profile_for("agents.missing:MissingAgent")

        self.assertEqual(profile["name"], "MissingAgent")
        self.assertIn("Custom or unknown agent", profile["positioning"])


if __name__ == "__main__":
    unittest.main()
