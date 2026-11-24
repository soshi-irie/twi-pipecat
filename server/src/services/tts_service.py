"""Select streaming Gemini speech or the original Notevibes job API."""

import os
from collections.abc import Mapping

from pipecat.services.google.tts import GeminiTTSService
from pipecat.services.tts_service import TTSService

from src.services.notevibes_tts_service import NotevibesTTSService
from src.services.twi_dialects import dialect_speech_style

VOICE_PROFILES = ("Sulafat", "Callirrhoe", "Achird")


def create_tts(
    env: Mapping[str, str] | None = None,
    *,
    voice_profile: str | None = None,
    dialect: str | None = None,
) -> TTSService:
    env = os.environ if env is None else env
    provider = env.get("TTS_PROVIDER", "notevibes").strip().lower()
    speech_style = dialect_speech_style(dialect)
    if dialect is not None and provider != "gemini":
        raise ValueError("Dialect speech guidance requires TTS_PROVIDER=gemini")
    if voice_profile is not None:
        if voice_profile not in VOICE_PROFILES:
            raise ValueError("Choose Sulafat, Callirrhoe, or Achird")
        if provider != "gemini":
            raise ValueError("Voice profiles require TTS_PROVIDER=gemini")
    if provider == "gemini":
        key = env.get("GEMINI_API_KEY", "").strip()
        if not key or key.startswith("your-"):
            raise ValueError("Configure GEMINI_API_KEY in server/.env")
        return GeminiTTSService(
            api_key=key,
            use_genai=True,
            sample_rate=24000,
            settings=GeminiTTSService.Settings(
                model=env.get("GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts"),
                voice=voice_profile or env.get("GEMINI_TTS_VOICE", "Callirrhoe"),
                prompt=speech_style,
                # Detect the text language, rather than forcing English on Twi.
                language=None,
            ),
        )
    if provider == "notevibes":
        return NotevibesTTSService(
            api_key=env.get("NOTEVIBES_API_KEY", ""),
            voice_id=env.get("NOTEVIBES_VOICE_ID", ""),
        )
    raise ValueError("TTS_PROVIDER must be gemini or notevibes")
