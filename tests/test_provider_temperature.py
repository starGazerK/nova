from __future__ import annotations

import unittest


class ProviderTemperatureTests(unittest.TestCase):
    def test_gpt5_temperature_is_forced_to_one(self) -> None:
        from nova.providers.litellm_provider import _temperature_for_model

        self.assertEqual(_temperature_for_model("openai/gpt-5.4-mini", 0.7), 1.0)
        self.assertEqual(_temperature_for_model("gpt-5-codex", 0.3), 1.0)
        self.assertEqual(_temperature_for_model("gpt-5.1", 1.0), 1.0)

    def test_non_gpt5_temperature_is_preserved(self) -> None:
        from nova.providers.litellm_provider import _temperature_for_model

        self.assertEqual(_temperature_for_model("deepseek/deepseek-chat", 0.7), 0.7)
        self.assertEqual(_temperature_for_model("openai/gpt-4o", 0.3), 0.3)
        self.assertIsNone(_temperature_for_model("openai/gpt-4o", None))

    def test_provider_uses_configured_default_temperature(self) -> None:
        from nova.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(
            api_key="test",
            model="deepseek/deepseek-chat",
            temperature=0.2,
        )

        self.assertEqual(provider.generation.temperature, 0.2)


if __name__ == "__main__":
    unittest.main()
