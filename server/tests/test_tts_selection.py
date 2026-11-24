import unittest
from unittest.mock import patch

from pipecat.services.google.tts import GeminiTTSService

from src.services.notevibes_tts_service import NotevibesTTSService
from src.services.tts_service import create_tts


class TTSSelectionTests(unittest.TestCase):
    def test_session_voice_overrides_default_without_changing_other_sessions(self):
        env = {
            "TTS_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-only",
            "GEMINI_TTS_VOICE": "Callirrhoe",
        }
        for voice in ("Sulafat", "Callirrhoe", "Achird"):
            with self.subTest(voice=voice):
                with patch.object(GeminiTTSService, "__init__", return_value=None) as init:
                    create_tts(env, voice_profile=voice)
                    self.assertEqual(init.call_args.kwargs["settings"].voice, voice)
                    create_tts(env)
                    self.assertEqual(init.call_args.kwargs["settings"].voice, "Callirrhoe")
        self.assertEqual(env["GEMINI_TTS_VOICE"], "Callirrhoe")

    def test_invalid_session_voice_is_rejected_before_provider_call(self):
        for voice in ("unknown", "", [], {}):
            with self.subTest(voice=voice):
                with self.assertRaisesRegex(ValueError, "Choose Sulafat"):
                    create_tts({"TTS_PROVIDER": "gemini"}, voice_profile=voice)

    def test_session_voice_requires_gemini(self):
        with self.assertRaisesRegex(ValueError, "TTS_PROVIDER=gemini"):
            create_tts({"TTS_PROVIDER": "notevibes"}, voice_profile="Achird")

    def test_gemini_uses_streaming_backend_and_detects_text_language(self):
        with patch.object(GeminiTTSService, "__init__", return_value=None) as init:
            result = create_tts({"TTS_PROVIDER": "gemini", "GEMINI_API_KEY": "test-only"})
        self.assertIsInstance(result, GeminiTTSService)
        self.assertTrue(init.call_args.kwargs["use_genai"])
        settings = init.call_args.kwargs["settings"]
        self.assertIsNone(settings.language)
        self.assertEqual(settings.model, "gemini-3.8-flash-lite-tts")
        self.assertEqual(settings.voice, "Callirrhoe")

    def test_explicit_notevibes_rollback(self):
        result = create_tts({"TTS_PROVIDER": "notevibes"})
        self.assertIsInstance(result, NotevibesTTSService)
        self.assertIsNone(result._settings.language)

    def test_invalid_or_missing_provider_settings_do_not_fall_back(self):
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            create_tts({"TTS_PROVIDER": "gemini"})
        with self.assertRaisesRegex(ValueError, "TTS_PROVIDER"):
            create_tts({"TTS_PROVIDER": "invalid"})
