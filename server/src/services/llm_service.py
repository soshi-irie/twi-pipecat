"""Select the conversation model without changing the voice pipeline."""

import os
from collections.abc import Mapping

from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService

from src.services.twi_dialects import dialect_text_instruction

SYSTEM_INSTRUCTION = (
    'You are a voice assistant who understands English and Twi. For this round only, repeat '
    'your self-introduction in both languages. Afterwards, stick to the language user speaks '
    'or specifically asks for. If the user asks for any other language, politely decline '
    'saying you can only reliably operate in the two languages. If the user asks for a '
    'specific question (not a mere greeting) that is not related to any finance-related '
    'functionalities you have access to, tell the user that you do not have access to that '
    'information and prompt them to ask about their transactions. The Twi speech-to-text '
    'system is premature and may contain typos. In particular, e nad o are often swapped with '
    'ɛ and ɔ, respectively, and vice versa. A word may be split into multiple words, and '
    'words might be transcribed as one word. Be generous about interpreting it, and aks '
    'clarifying questions. If transaction speech is unclear, ask the user to enter one phone '
    "number or amount at a time on the keypad and press #. Keypad input appears as 'DTMF: "
    "<digits>#'; use it for the value requested (amounts are cedis). Keep responses brief. "
    'Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting '
    'that cannot be spoken.'
)


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value or value.startswith("your-"):
        raise ValueError(f"Configure {name} in server/.env")
    return value


def create_llm(
    env: Mapping[str, str] | None = None, *, dialect: str | None = None
) -> OpenAILLMService | GoogleLLMService:
    """Build only the selected provider, validating its credentials and options."""
    env = os.environ if env is None else env
    system_instruction = SYSTEM_INSTRUCTION + dialect_text_instruction(dialect)
    provider = env.get("LLM_PROVIDER", "openai").strip().lower()
    if provider == "openai":
        return OpenAILLMService(
            api_key=_required(env, "OPENAPI_KEY"),
            settings=OpenAILLMService.Settings(
                model=env.get("OPENAI_MODEL", "gpt-4.1"),
                max_completion_tokens=150,
                temperature=0.7,
                system_instruction=system_instruction,
            ),
        )
    if provider == "gemini":
        api_key = _required(env, "GEMINI_API_KEY")
        thinking_level = env.get("GEMINI_THINKING_LEVEL", "low").strip().lower()
        if thinking_level not in {"low", "medium", "high", "xhigh"}:
            raise ValueError("GEMINI_THINKING_LEVEL must be low, medium, high, or xhigh")
        return GoogleLLMService(
            api_key=api_key,
            settings=GoogleLLMService.Settings(
                model=env.get("GEMINI_MODEL", "gemini-4-neon"),
                # Gemini's output allowance includes thinking tokens.
                max_tokens=4096,
                system_instruction=system_instruction,
                thinking=GoogleLLMService.ThinkingConfig(thinking_level=thinking_level),
            ),
            stream_idle_timeout_secs=60,
        )
    raise ValueError("LLM_PROVIDER must be openai or gemini")
