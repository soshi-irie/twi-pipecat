"""Sunflower v2 ASR through Sunbird's official faster-whisper conversion."""

import os
from collections.abc import Mapping
from pathlib import Path

from pipecat.services.stt_service import STTService
from pipecat.services.whisper.stt import WhisperSTTService, WhisperSTTServiceMLX
from pipecat.transcriptions.language import Language

SUNFLOWER_MODEL = "Sunbird/faster-whisper-51-african-languages"
# CPU transcription is queued per speech segment. The first measured segment
# took 9s, longer than the aggregator's 5s idle watchdog. This conservative
# budget is not a measured p99; final transcripts still end turns immediately.
SUNFLOWER_TRANSCRIPT_WAIT_SECS = 60.0

# Sunflower reuses Whisper's language token slots. JW means Akan for this
# checkpoint, not Javanese. Automatic language detection is not supported.
# https://huggingface.co/Sunbird/faster-whisper-51-african-languages
LANGUAGE_TOKENS = {"aka": Language.JW, "eng": Language.EN}


def _mlx_model_path(env: Mapping[str, str]) -> Path:
    model_path = Path(
        env.get("STT_MLX_MODEL_PATH", str(Path.home() / ".cache/twi-pipecat/sunflower-v2-mlx"))
    ).expanduser()
    if not all(
        (model_path / name).is_file()
        for name in ("config.json", "weights.safetensors", "conversion.json")
    ):
        raise ValueError(
            "Run `uv run python scripts/convert_sunflower_mlx.py` first, "
            "or configure STT_MLX_MODEL_PATH"
        )
    return model_path


def warm_stt_model() -> None:
    """Load MLX weights and compile GPU kernels before accepting the first call."""
    if os.environ.get("STT_ENGINE", "faster-whisper").strip().lower() != "mlx":
        return
    import mlx_whisper
    import numpy as np

    mlx_whisper.transcribe(
        np.zeros(16000, dtype=np.float32),
        path_or_hf_repo=str(_mlx_model_path(os.environ)),
        language=LANGUAGE_TOKENS[os.environ.get("STT_LANGUAGE", "aka").strip().lower()],
        temperature=0.0,
    )


def create_stt(env: Mapping[str, str] | None = None) -> STTService:
    env = os.environ if env is None else env
    engine = env.get("STT_ENGINE", "faster-whisper").strip().lower()
    if engine == "simba":
        from src.services.simba_stt_service import SimbaSTTService, SimbaSTTSettings

        if not os.environ.get("SIMBA_STT_URL", "").strip():
            raise ValueError("Configure SIMBA_STT_URL in server/.env")
        return SimbaSTTService(settings=SimbaSTTSettings(model="simba-s", language="twi"))
    language = env.get("STT_LANGUAGE", "aka").strip().lower()
    if language not in LANGUAGE_TOKENS:
        raise ValueError(
            "STT_LANGUAGE must be aka (Akan/Twi) or eng (English); auto is unsupported"
        )
    engine = env.get("STT_ENGINE", "faster-whisper").strip().lower()
    if engine == "mlx":
        model_path = _mlx_model_path(env)
        return WhisperSTTServiceMLX(
            ttfs_p99_latency=SUNFLOWER_TRANSCRIPT_WAIT_SECS,
            settings=WhisperSTTServiceMLX.Settings(
                model=str(model_path),
                language=LANGUAGE_TOKENS[language],
                temperature=0.0,
            ),
        )
    if engine != "faster-whisper":
        raise ValueError("STT_ENGINE must be mlx, faster-whisper, or simba")
    return WhisperSTTService(
        # CTranslate2 runs on CPU on this Apple Silicon machine. INT8 avoids
        # attempting the checkpoint's CUDA/float16 example on unsupported hardware.
        device="cpu",
        compute_type="int8",
        ttfs_p99_latency=SUNFLOWER_TRANSCRIPT_WAIT_SECS,
        settings=WhisperSTTService.Settings(
            model=SUNFLOWER_MODEL,
            language=LANGUAGE_TOKENS[language],
        ),
    )
