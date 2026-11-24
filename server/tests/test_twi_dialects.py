import unittest
from unittest.mock import patch

from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.google.tts import GeminiTTSService
from pipecat.services.openai.llm import OpenAILLMService

from src.services.llm_service import SYSTEM_INSTRUCTION, create_llm
from src.services.tts_service import VOICE_PROFILES, create_tts
from src.services.twi_dialects import TWI_DIALECTS


class TwiDialectTests(unittest.TestCase):
    def test_both_llm_providers_receive_dialect_and_keep_voice_safe_rules(self):
        for provider, service, key in (
            ("gemini", GoogleLLMService, "GEMINI_API_KEY"),
            ("openai", OpenAILLMService, "OPENAPI_KEY"),
        ):
            for dialect, name in TWI_DIALECTS.items():
                with self.subTest(provider=provider, dialect=dialect):
                    with patch.object(service, "__init__", return_value=None) as init:
                        create_llm({"LLM_PROVIDER": provider, key: "test-only"}, dialect=dialect)
                    instruction = init.call_args.kwargs["settings"].system_instruction
                    self.assertIn(name, instruction)
                    self.assertIn(SYSTEM_INSTRUCTION, instruction)
                    self.assertIn("ɛ", instruction)
                    self.assertIn("ɔ", instruction)

    def test_dialect_style_is_separate_from_voice_and_sessions_are_isolated(self):
        env = {"TTS_PROVIDER": "gemini", "GEMINI_API_KEY": "test-only"}
        for dialect, name in TWI_DIALECTS.items():
            for voice in VOICE_PROFILES:
                with self.subTest(dialect=dialect, voice=voice):
                    with patch.object(GeminiTTSService, "__init__", return_value=None) as init:
                        create_tts(env, dialect=dialect, voice_profile=voice)
                        settings = init.call_args.kwargs["settings"]
                        self.assertEqual(settings.voice, voice)
                        self.assertIn(name, settings.prompt)
                        self.assertIsNone(settings.language)
                        create_tts(env)
                        self.assertIsNone(init.call_args.kwargs["settings"].prompt)
        self.assertEqual(env, {"TTS_PROVIDER": "gemini", "GEMINI_API_KEY": "test-only"})

    def test_untrusted_dialects_are_rejected_by_both_factories(self):
        for value in ("invalid", "", [], {}, 12):
            for factory in (create_llm, create_tts):
                with self.subTest(value=value, factory=factory.__name__):
                    with self.assertRaisesRegex(ValueError, "Choose Asante Twi"):
                        factory({}, dialect=value)

    def test_notevibes_does_not_silently_ignore_selected_dialect(self):
        with self.assertRaisesRegex(ValueError, "TTS_PROVIDER=gemini"):
            create_tts({"TTS_PROVIDER": "notevibes"}, dialect="asante")
