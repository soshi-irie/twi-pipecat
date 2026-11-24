"""Validated session dialects shared by text generation and speech synthesis."""

TWI_DIALECTS = {"asante": "Asante Twi", "akuapem": "Akuapem Twi"}


def dialect_name(dialect: str | None) -> str | None:
    if dialect is None:
        return None
    if not isinstance(dialect, str) or dialect not in TWI_DIALECTS:
        raise ValueError("Choose Asante Twi or Akuapem Twi")
    return TWI_DIALECTS[dialect]


def dialect_text_instruction(dialect: str | None) -> str:
    name = dialect_name(dialect)
    if name is None:
        return ""
    return (
        f" When responding in Twi, use {name} vocabulary and spelling consistently. "
        "Preserve the letters ɛ and ɔ where appropriate; do not replace them with e and o. "
        "Follow the user's requested language, including English when requested."
    )


def dialect_speech_style(dialect: str | None) -> str | None:
    name = dialect_name(dialect)
    if name is None:
        return None
    return (
        f"For Twi passages, speak natural {name} with clear articulation, "
        "natural tone distinctions, and a conversational pace. "
        "Read English passages naturally in English."
    )
