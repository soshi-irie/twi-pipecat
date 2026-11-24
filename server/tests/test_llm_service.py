import unittest
from unittest.mock import patch

from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService

from src.services.llm_service import SYSTEM_INSTRUCTION, create_llm


class LLMSelectionTests(unittest.TestCase):
    def test_gemini_requires_no_openai_key(self):
        with patch.object(GoogleLLMService, "__init__", return_value=None) as init:
            llm = create_llm({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-only"})
        self.assertIsInstance(llm, GoogleLLMService)
        settings = init.call_args.kwargs["settings"]
        self.assertEqual(settings.model, "gemini-4-neon")
        self.assertEqual(settings.thinking.thinking_level, "low")
        self.assertEqual(settings.system_instruction, SYSTEM_INSTRUCTION)

    def test_openai_preserves_existing_defaults(self):
        with patch.object(OpenAILLMService, "__init__", return_value=None) as init:
            llm = create_llm({"OPENAPI_KEY": "test-only"})
        self.assertIsInstance(llm, OpenAILLMService)
        settings = init.call_args.kwargs["settings"]
        self.assertEqual(settings.model, "gpt-4.1")
        self.assertEqual(settings.max_completion_tokens, 150)
        self.assertEqual(settings.system_instruction, SYSTEM_INSTRUCTION)

    def test_missing_selected_key_does_not_fall_back(self):
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            create_llm({"LLM_PROVIDER": "gemini", "OPENAPI_KEY": "test-only"})
        with self.assertRaisesRegex(ValueError, "OPENAPI_KEY"):
            create_llm({"LLM_PROVIDER": "openai", "GEMINI_API_KEY": "test-only"})

    def test_invalid_provider_and_thinking_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "LLM_PROVIDER"):
            create_llm({"LLM_PROVIDER": "gmeini"})
        with self.assertRaisesRegex(ValueError, "GEMINI_THINKING_LEVEL"):
            create_llm(
                {
                    "LLM_PROVIDER": "gemini",
                    "GEMINI_API_KEY": "test-only",
                    "GEMINI_THINKING_LEVEL": "invalid",
                }
            )


if __name__ == "__main__":
    unittest.main()
