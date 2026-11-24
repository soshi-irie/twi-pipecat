import asyncio
import io
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from pipecat.frames.frames import ErrorFrame

from src.services.notevibes_tts_service import NotevibesTTSService, decode_audio
from src.tools.transaction import find_transaction


class TransactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_reports_lookup_results_to_pipeline(self):
        cases = [
            ("75 cedis to 0203456789", "found"),
            ("999999 cedis to 0203456789", "not_found"),
            ("Please check my transfer", "missing_information"),
        ]
        for transcript, expected in cases:
            with self.subTest(expected=expected):
                callback = AsyncMock()
                await find_transaction(SimpleNamespace(result_callback=callback), transcript)
                callback.assert_awaited_once()
                result = callback.call_args.args[0]
                self.assertEqual(result["status"], expected)
                if expected == "found":
                    self.assertEqual(result["transfer_status"], "pending")


class TTSServiceTests(unittest.IsolatedAsyncioTestCase):
    def audio_fixture(self):
        pcm = b"\x01\x00" * 2400
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(pcm)
        return buffer.getvalue(), pcm

    async def test_submit_poll_download_without_leaking_key(self):
        wav, pcm = self.audio_fixture()
        requests = []
        polls = 0

        def respond(request):
            nonlocal polls
            requests.append(request)
            if request.url.host == "audio.example":
                self.assertNotIn("authorization", request.headers)
                return httpx.Response(200, content=wav)
            self.assertEqual(request.headers["Authorization"], "Bearer test-only")
            if request.method == "POST":
                import json

                self.assertEqual(json.loads(request.content)["voice"], "test-voice")
                return httpx.Response(202, json={"jobId": "job-1"})
            polls += 1
            if polls == 1:
                return httpx.Response(200, json={"status": "processing"})
            return httpx.Response(
                200,
                json={"status": "completed", "result": {"audioUrl": "https://audio.example/reply"}},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        service = NotevibesTTSService(api_key="test-only", voice_id="test-voice", poll_interval=0)
        with patch("src.services.notevibes_tts_service.httpx.AsyncClient", return_value=client):
            result = await service.synthesize("Akwaaba")
        self.assertEqual(result, pcm)
        self.assertEqual(len(requests), 4)

    async def test_failed_job_returns_error_frame(self):
        def respond(request):
            return httpx.Response(
                200, json=({"jobId": "job-1"} if request.method == "POST" else {"status": "failed"})
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        service = NotevibesTTSService(api_key="test-only", voice_id="test-voice")
        with patch("src.services.notevibes_tts_service.httpx.AsyncClient", return_value=client):
            frames = [frame async for frame in service.run_tts("Hello", "context")]
        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], ErrorFrame)
        self.assertIn("job failed", frames[0].error)

    async def test_polling_has_deadline(self):
        def respond(request):
            return httpx.Response(
                200, json=({"jobId": "job-1"} if request.method == "POST" else {"status": "queued"})
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        service = NotevibesTTSService(
            api_key="test-only", voice_id="test-voice", job_timeout=0.01, poll_interval=1
        )
        with patch("src.services.notevibes_tts_service.httpx.AsyncClient", return_value=client):
            with self.assertRaises(TimeoutError):
                await service.synthesize("Hello")

    async def test_missing_configuration_is_actionable(self):
        service = NotevibesTTSService(api_key="", voice_id="")
        with self.assertRaisesRegex(ValueError, "NOTEVIBES_API_KEY"):
            await service.synthesize("Hello")
        service = NotevibesTTSService(api_key="test-only", voice_id="")
        with self.assertRaisesRegex(ValueError, "NOTEVIBES_VOICE_ID"):
            await service.synthesize("Hello")

    async def test_interruption_propagates_cancellation(self):
        service = NotevibesTTSService(api_key="test-only", voice_id="test-voice")
        with patch.object(service, "synthesize", side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await anext(service.run_tts("Hello", "context"))

    def test_decodes_and_resamples_audio(self):
        wav, _ = self.audio_fixture()
        pcm = decode_audio(wav, 16000)
        self.assertEqual(len(pcm), 1600 * 2)


if __name__ == "__main__":
    unittest.main()
