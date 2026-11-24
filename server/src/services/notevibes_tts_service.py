"""Notevibes' asynchronous TTS API adapted to Pipecat audio frames.

API contract: https://notevibes.com/developer/docs/tts
"""

import asyncio
import io
from collections.abc import AsyncGenerator
from urllib.parse import quote

import av
import httpx
from av.error import FFmpegError
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


def decode_audio(data: bytes, sample_rate: int) -> bytes:
    """Decode the downloaded MP3/WAV into mono, signed 16-bit PCM."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    chunks = []
    with av.open(io.BytesIO(data), mode="r") as container:
        for frame in container.decode(audio=0):
            chunks.extend(f.to_ndarray().tobytes() for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray().tobytes() for f in resampler.resample(None))
    pcm = b"".join(chunks)
    if not pcm:
        raise ValueError("Notevibes returned empty audio")
    return pcm


class NotevibesTTSService(TTSService):
    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        job_timeout: float = 90,
        poll_interval: float = 3,
        **kwargs,
    ):
        super().__init__(
            # Notevibes selects the language through its voice ID, not a separate parameter.
            settings=TTSSettings(model="notevibes", voice=voice_id, language=None),
            sample_rate=24000,
            push_start_frame=True,
            push_stop_frames=True,
            # Job generation can exceed Pipecat's default 3-second audio idle limit.
            stop_frame_timeout_s=job_timeout + 5,
            **kwargs,
        )
        self._api_key = api_key
        self._job_timeout = job_timeout
        self._poll_interval = poll_interval

    async def synthesize(self, text: str) -> bytes:
        # Check when speech is requested: text-mode evals bypass this service.
        if not self._api_key or self._api_key.startswith("your-"):
            raise ValueError("Configure NOTEVIBES_API_KEY in server/.env")
        voice = self._settings.voice
        if not isinstance(voice, str) or not voice or voice.startswith("your-"):
            raise ValueError("Configure NOTEVIBES_VOICE_ID in server/.env")
        headers = {"Authorization": f"Bearer {self._api_key}"}
        api_url = "https://notevibes.com/api/tts"
        async with asyncio.timeout(self._job_timeout), httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{api_url}/create",
                headers=headers,
                json={"title": "Voice assistant reply", "content": text, "voice": voice},
            )
            response.raise_for_status()
            job_id = response.json().get("jobId")
            if not isinstance(job_id, str) or not job_id:
                raise ValueError("Notevibes did not return a job ID")
            while True:
                response = await client.get(
                    f"{api_url}/jobs/{quote(job_id, safe='')}", headers=headers
                )
                response.raise_for_status()
                job = response.json()
                status = job.get("status")
                if status == "completed":
                    audio_url = job.get("result", {}).get("audioUrl")
                    if not isinstance(audio_url, str) or not audio_url.startswith("https://"):
                        raise ValueError("Notevibes did not return an HTTPS audio URL")
                    # Signed storage URLs do not receive the Notevibes API credential.
                    audio = await client.get(audio_url, follow_redirects=True)
                    audio.raise_for_status()
                    return await asyncio.to_thread(decode_audio, audio.content, 24000)
                if status == "failed":
                    raise ValueError("Notevibes synthesis job failed")
                if status not in ("queued", "processing"):
                    raise ValueError("Notevibes returned an unknown job status")
                await asyncio.sleep(self._poll_interval)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            pcm = await self.synthesize(text)
        except httpx.HTTPStatusError as exc:
            yield ErrorFrame(error=f"Notevibes request failed: HTTP {exc.response.status_code}")
            return
        except TimeoutError:
            yield ErrorFrame(error="Notevibes synthesis timed out")
            return
        except (httpx.RequestError, ValueError, FFmpegError) as exc:
            # Never log signed download URLs, authorization headers, or response bodies.
            detail = str(exc) if type(exc) is ValueError else type(exc).__name__
            yield ErrorFrame(error=f"Notevibes synthesis failed: {detail}")
            return
        for offset in range(0, len(pcm), self.chunk_size):
            yield TTSAudioRawFrame(
                audio=pcm[offset : offset + self.chunk_size],
                sample_rate=24000,
                num_channels=1,
                context_id=context_id,
            )
