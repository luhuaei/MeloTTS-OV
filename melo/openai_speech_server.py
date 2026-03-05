import argparse
import io
import os
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from pydub import AudioSegment

from .api import TTS

OPENAI_VOICE_TO_EN_SPK = {
    "alloy": "EN-Default",
    "echo": "EN-US",
    "fable": "EN-BR",
    "onyx": "EN_INDIA",
    "nova": "EN-AU",
    "shimmer": "EN-Default",
    "ash": "EN-US",
    "ballad": "EN-BR",
    "coral": "EN-Default",
    "sage": "EN_INDIA",
    "verse": "EN-AU",
}

SUPPORTED_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "application/octet-stream",
}


class SpeechRequest(BaseModel):
    model: str = Field(..., description="OpenAI-compatible TTS model name")
    input: str = Field(..., min_length=1, description="Text to synthesize")
    voice: str = Field(default="alloy")
    response_format: str = Field(default="mp3")
    speed: float = Field(default=1.0)
    instructions: Optional[str] = Field(default=None)
    language: Optional[str] = Field(default=None, description="Non-OpenAI extension: force language")


class MeloSpeechService:
    def __init__(self):
        self.default_language = os.getenv("MELO_DEFAULT_LANGUAGE", "ZH").upper()
        self.onnx_dir = os.getenv("MELO_ONNX_DIR", "onnx_models")
        self.torch_device = os.getenv("MELO_TORCH_DEVICE", "auto")
        self.tts_device = os.getenv("MELO_TTS_DEVICE", "auto")
        self.bert_device = os.getenv("MELO_BERT_DEVICE", "auto")
        self.num_workers = int(os.getenv("MELO_WORKERS", "4"))
        self.auto_export_onnx = os.getenv("MELO_AUTO_EXPORT_ONNX", "1") not in {"0", "false", "False"}

        self._models: Dict[str, TTS] = {}
        self._lock = threading.Lock()

    def _build_model(self, language: str) -> TTS:
        return TTS(
            language=language,
            torch_device=self.torch_device,
            tts_device=self.tts_device,
            bert_device=self.bert_device,
            num_workers=self.num_workers,
            onnx_dir=self.onnx_dir,
            auto_export_onnx=self.auto_export_onnx,
        )

    def get_model(self, language: str) -> TTS:
        language = language.upper()
        model = self._models.get(language)
        if model is not None:
            return model

        with self._lock:
            model = self._models.get(language)
            if model is None:
                model = self._build_model(language)
                self._models[language] = model
        return model

    def resolve_language_and_speaker(self, voice: str, forced_language: Optional[str]) -> Tuple[str, str]:
        canonical_voice = voice.upper()
        if forced_language:
            language = forced_language.upper()
            model = self.get_model(language)
            if voice in model.hps.data.spk2id:
                return language, voice
            first = list(model.hps.data.spk2id.keys())[0]
            return language, first

        # Voice can directly be a language id, e.g. ZH_MIX_EN / ZH / EN / JP.
        if canonical_voice in {"ZH_MIX_EN", "ZH", "EN", "JP", "KR", "ES", "FR"}:
            language = canonical_voice
            model = self.get_model(language)
            first = list(model.hps.data.spk2id.keys())[0]
            return language, first

        if ":" in voice:
            language, speaker = voice.split(":", 1)
            language = language.upper()
            model = self.get_model(language)
            if speaker in model.hps.data.spk2id:
                return language, speaker
            raise ValueError(f"Unknown speaker '{speaker}' for language '{language}'")

        if voice in OPENAI_VOICE_TO_EN_SPK:
            language = "EN"
            speaker = OPENAI_VOICE_TO_EN_SPK[voice]
            model = self.get_model(language)
            if speaker in model.hps.data.spk2id:
                return language, speaker
            first = list(model.hps.data.spk2id.keys())[0]
            return language, first

        # Native speaker names, e.g. EN-US / ZH / JP
        for language in ["EN", "ZH", "JP", "KR", "ES", "FR"]:
            model = self.get_model(language)
            if voice in model.hps.data.spk2id:
                return language, voice

        # Fallback to default language and first speaker
        language = self.default_language
        model = self.get_model(language)
        first = list(model.hps.data.spk2id.keys())[0]
        return language, first


service = MeloSpeechService()
app = FastAPI(title="MeloTTS OpenAI-Compatible Speech API", version="1.0.0")


def _openai_error(message: str, param: Optional[str] = None, status_code: int = 400) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": None,
            }
        },
    )


def _require_api_key(request: Request):
    expected = os.getenv("MELO_API_KEY")
    if not expected:
        return

    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Missing Bearer token",
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )

    token = auth[len("Bearer ") :].strip()
    if token != expected:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )


@app.exception_handler(HTTPException)
async def _http_exception_handler(_: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )


def _float_to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def _encode_audio(audio: np.ndarray, sampling_rate: int, response_format: str) -> bytes:
    fmt = response_format.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise _openai_error(
            f"Unsupported response_format '{response_format}'. Supported: {sorted(SUPPORTED_FORMATS)}",
            param="response_format",
        )

    if fmt == "pcm":
        return _float_to_int16(audio).tobytes()

    buf = io.BytesIO()
    if fmt == "wav":
        sf.write(buf, audio, sampling_rate, format="WAV")
    elif fmt == "flac":
        sf.write(buf, audio, sampling_rate, format="FLAC")
    elif fmt == "mp3":
        sf.write(buf, audio, sampling_rate, format="MP3")
    elif fmt == "opus":
        sf.write(buf, audio, sampling_rate, format="OGG", subtype="OPUS")
    elif fmt == "aac":
        pcm16 = _float_to_int16(audio)
        segment = AudioSegment(
            data=pcm16.tobytes(),
            sample_width=2,
            frame_rate=sampling_rate,
            channels=1,
        )
        segment.export(buf, format="adts")

    return buf.getvalue()


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/v1/audio/speech")
def create_speech(payload: SpeechRequest, _: None = Depends(_require_api_key)):
    if not payload.input.strip():
        raise _openai_error("'input' must be non-empty", param="input")

    if not (0.25 <= payload.speed <= 4.0):
        raise _openai_error("'speed' must be in [0.25, 4.0]", param="speed")

    language, speaker = service.resolve_language_and_speaker(payload.voice, payload.language)
    tts = service.get_model(language)
    speaker_map = tts.hps.data.spk2id
    if speaker not in speaker_map:
        raise _openai_error(f"Unknown speaker '{speaker}' for language '{language}'", param="voice")

    audio = tts.tts_to_file(
        payload.input,
        speaker_id=speaker_map[speaker],
        output_path=None,
        speed=payload.speed,
        quiet=True,
        concurrency=service.num_workers,
    )

    audio_bytes = _encode_audio(audio, tts.hps.data.sampling_rate, payload.response_format)
    fmt = payload.response_format.lower()
    return Response(
        content=audio_bytes,
        media_type=CONTENT_TYPES[fmt],
        headers={
            "x-melo-language": language,
            "x-melo-speaker": speaker,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible /v1/audio/speech server for MeloTTS")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("melo.openai_speech_server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
