import os

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.dtmf_aggregator import DTMFAggregator
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import DailyDialinRequest, RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.workers.runner import WorkerRunner

from src.services.llm_service import create_llm
from src.services.stt_service import SUNFLOWER_TRANSCRIPT_WAIT_SECS, create_stt, warm_stt_model
from src.services.tts_service import create_tts
from src.tools.accounts import find_user_id, get_account_balance, get_account_type
from src.tools.transaction import find_transaction, get_sum

tools = [
    find_transaction,
    get_sum,
    get_account_balance,
    get_account_type,
    find_user_id,
]

# Keep the original method
_original_handle_dialin_ready = DailyTransport._handle_dialin_ready


async def patched_handle_dialin_ready(self, sip_endpoint):
    # Convert the dictionary into the SIP URI string
    if isinstance(sip_endpoint, dict):
        sip_endpoint = sip_endpoint["sipEndpoint"]

    # Call Pipecat's original implementation
    await _original_handle_dialin_ready(self, sip_endpoint)


# Replace the class method at runtime
DailyTransport._handle_dialin_ready = patched_handle_dialin_ready


# Bot logic
async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    session_body = {} if runner_args.body is None else runner_args.body
    if not isinstance(session_body, dict):
        raise ValueError("Session body must be an object")
    dialect = session_body.get("dialect")
    llm = create_llm(dialect=dialect)
    logger.info("LLM provider: {}", os.environ.get("LLM_PROVIDER", "openai"))

    # Frame processors
    stt = create_stt()
    tts = create_tts(voice_profile=session_body.get("voice"), dialect=dialect)
    logger.info("Session Twi dialect: {}", dialect or "unspecified")
    logger.info(
        "Session voice: {}",
        session_body.get("voice") or os.environ.get("GEMINI_TTS_VOICE", "Callirrhoe"),
    )
    logger.info("Speech services: STT={}, TTS={}", type(stt).__name__, type(tts).__name__)

    # LLm context manager
    context = LLMContext(messages=[], tools=tools)
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_stop_timeout=SUNFLOWER_TRANSCRIPT_WAIT_SECS,
        ),
    )
    dtmf_aggregator = DTMFAggregator(timeout=10.0)

    pipeline = Pipeline(
        [
            transport.input(),
            dtmf_aggregator,
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    # worker (pipeline runner)
    worker = PipelineWorker(
        pipeline,
        name="assistant",
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    if isinstance(transport, DailyTransport):
        # Start speaking when caller joins
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            logger.debug(f"First participant joined: {participant['id']}")
            await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    if isinstance(transport, DailyTransport):
        @transport.event_handler("on_dialin_ready")
        async def on_dialin_ready(transport, sip_endpoint):
            logger.info("Dial-in ready: {}", sip_endpoint)

        @transport.event_handler("on_dialin_connected")
        async def on_dialin_connected(transport, data):
            logger.info(f"Dial-in connected: {data}")

        @transport.event_handler("on_dialin_stopped")
        async def on_dialin_stopped(transport, data):
            logger.info(f"Dial-in stopped: {data}")
            await runner.cancel()

        @transport.event_handler("on_dialin_warning")
        async def on_dialin_warning(transport, data):
            logger.warning(f"Dial-in warning: {data}")

        @transport.event_handler("on_dialin_error")
        async def on_dialin_error(transport, data):
            logger.error(f"Dial-in error: {data}")
            await runner.cancel()

        @transport.event_handler("on_dtmf_event")
        async def on_dtmf_event(transport, data):
            logger.info(f"DTMF event: {data}")

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        # Kick off the conversation
        context.add_message(
            {
                "role": "developer",
                "content": """
                Start by concisely identifying yourself as a Twi-English voice assistent at the Bank of Minervapolis, and asking for the user's preferred language.
                """,
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    await runner.run()


# Session Initialization (Listens to users' connection requests)
async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""
    transport_params = {
        "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    # Optional: personalize using the dial-in request (which number called, which
    # number was dialed).
    if isinstance(runner_args.body, dict) and "dialin_settings" in runner_args.body:
        request = DailyDialinRequest.model_validate(runner_args.body)
        if request.dialin_settings.From:
            logger.info(f"Handling call from: {request.dialin_settings.From}")
        if request.dialin_settings.To:
            logger.info(f"Handling call to: {request.dialin_settings.To}")

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    warm_stt_model()
    main()
