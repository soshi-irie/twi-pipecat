import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pipecat.frames.frames import TranscriptionFrame
from pipecat.services.whisper.stt import WhisperSTTServiceMLX

from src.services.stt_service import SUNFLOWER_MODEL, create_stt


class SunflowerSTTTests(unittest.IsolatedAsyncioTestCase):
    def test_existing_simba_backend_remains_selectable(self):
        legacy = SimpleNamespace(SimbaSTTService=Mock(), SimbaSTTSettings=Mock())
        with patch.dict("sys.modules", {"src.services.simba_stt_service": legacy}):
            with patch.dict("os.environ", {"SIMBA_STT_URL": "http://localhost:9000"}):
                result = create_stt({"STT_ENGINE": "simba"})
        legacy.SimbaSTTSettings.assert_called_once_with(model="simba-s", language="twi")
        self.assertIs(result, legacy.SimbaSTTService.return_value)

    async def test_mlx_uses_converted_checkpoint_and_akan_token(self):
        with TemporaryDirectory() as directory:
            for name in ("config.json", "weights.safetensors", "conversion.json"):
                (Path(directory) / name).touch()
            service = create_stt({"STT_ENGINE": "mlx", "STT_MLX_MODEL_PATH": directory})
            self.assertIsInstance(service, WhisperSTTServiceMLX)
            with patch(
                "mlx_whisper.transcribe",
                return_value={"segments": [{"text": "Akwaaba", "no_speech_prob": 0.01}]},
            ) as transcribe:
                frames = [f async for f in service.run_stt(b"\x00\x00" * 16000)]
            self.assertEqual(transcribe.call_args.kwargs["language"], "jw")
            self.assertEqual(transcribe.call_args.kwargs["path_or_hf_repo"], directory)
            self.assertEqual(frames[0].text.strip(), "Akwaaba")

    def test_missing_mlx_weights_fail_without_cpu_fallback(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "convert_sunflower_mlx"):
                create_stt({"STT_ENGINE": "mlx", "STT_MLX_MODEL_PATH": directory})
        with self.assertRaisesRegex(ValueError, "STT_ENGINE"):
            create_stt({"STT_ENGINE": "invalid"})

    async def test_twi_uses_akan_token_in_real_service_transcription(self):
        model = Mock(supported_languages=["jw", "en"])
        model.transcribe.return_value = (
            iter([SimpleNamespace(text="Akwaaba", no_speech_prob=0.01)]),
            SimpleNamespace(),
        )
        with patch("pipecat.services.whisper.stt.WhisperModel", return_value=model) as loader:
            service = create_stt({})
        loader.assert_called_once_with(SUNFLOWER_MODEL, device="cpu", compute_type="int8")
        frames = [frame async for frame in service.run_stt(b"\x00\x00" * 16000)]
        self.assertEqual(model.transcribe.call_args.kwargs["language"], "jw")
        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], TranscriptionFrame)
        self.assertEqual(frames[0].text.strip(), "Akwaaba")

    def test_english_is_an_explicit_choice(self):
        model = Mock(supported_languages=["jw", "en"])
        with patch("pipecat.services.whisper.stt.WhisperModel", return_value=model):
            service = create_stt({"STT_LANGUAGE": "eng"})
        self.assertEqual(service._settings.language, "en")

    def test_auto_and_unknown_languages_fail_before_loading(self):
        for language in ["auto", "", "twi", "invalid"]:
            with self.subTest(language=language):
                with patch("pipecat.services.whisper.stt.WhisperModel") as loader:
                    with self.assertRaisesRegex(ValueError, "STT_LANGUAGE"):
                        create_stt({"STT_LANGUAGE": language})
                    loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
