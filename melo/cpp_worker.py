import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

from .api import TTS, _normalize_language_for_model
from .openai_speech_server import CONTENT_TYPES, OPENAI_VOICE_TO_EN_SPK, SUPPORTED_FORMATS, _encode_audio


class RequestError(Exception):
    def __init__(
        self,
        message: str,
        *,
        param: Optional[str] = None,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        error_code: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.param = param
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code


class CppSpeechService:
    def __init__(
        self,
        *,
        default_language: str,
        onnx_dir: str,
        onnx_path: Optional[str],
        torch_device: str,
        tts_device: str,
        bert_device: str,
        num_workers: int,
    ):
        self.default_language = default_language.upper()
        self.onnx_dir = Path(onnx_dir)
        self.onnx_path = Path(onnx_path).expanduser().resolve() if onnx_path else None
        self.torch_device = torch_device
        self.tts_device = tts_device
        self.bert_device = bert_device
        self.num_workers = max(int(num_workers), 1)

        self._models: Dict[str, TTS] = {}
        self._tmp_root = Path(tempfile.gettempdir()) / f"melo_cpp_worker_{os.getpid()}"
        self._tmp_root.mkdir(parents=True, exist_ok=True)

    def _detect_model_root(self, language: str) -> Path:
        base_language = _normalize_language_for_model(language)

        if self.onnx_path is not None:
            p = self.onnx_path
            if p.is_file() and p.suffix == ".onnx":
                return p.parent

            if not p.is_dir():
                raise RequestError(f"Invalid --onnx path: {p}", param="onnx", status_code=500)

            candidates = [
                p,
                p / f"tts_onnx_{language}",
                p / f"tts_onnx_{base_language}",
            ]

            for candidate in candidates:
                if not candidate.exists() or not candidate.is_dir():
                    continue
                tts_candidates = [
                    candidate / f"tts_{language}.onnx",
                    candidate / f"tts_{base_language}.onnx",
                    candidate / "tts.onnx",
                ]
                has_tts = any(x.exists() for x in tts_candidates)
                has_bert = (candidate / "bert_multilingual.onnx").exists() or (
                    candidate.parent / "shared_bert" / "bert_multilingual.onnx"
                ).exists()
                if has_tts and has_bert:
                    return candidate

            raise RequestError(
                f"Cannot locate ONNX model files for language '{language}' under {p}",
                param="onnx",
                status_code=500,
            )

        candidates = [
            self.onnx_dir / f"tts_onnx_{language}",
            self.onnx_dir / f"tts_onnx_{base_language}",
        ]

        for candidate in candidates:
            has_bert = (candidate / "bert_multilingual.onnx").exists() or (
                candidate.parent / "shared_bert" / "bert_multilingual.onnx"
            ).exists()
            if candidate.is_dir() and has_bert:
                return candidate

        raise RequestError(
            f"Cannot locate ONNX model directory for language '{language}' under {self.onnx_dir}",
            param="language",
            status_code=500,
        )

    def _resolve_effective_onnx_dir(self, language: str, model_root: Path) -> Path:
        if self.onnx_path is None:
            return self.onnx_dir

        wrapper_root = self._tmp_root / "onnx_root"
        wrapper_root.mkdir(parents=True, exist_ok=True)
        target_dir = wrapper_root / f"tts_onnx_{language}"

        if not target_dir.exists():
            try:
                os.symlink(model_root, target_dir)
            except OSError:
                # If symlink is unavailable, copytree fallback keeps behavior deterministic.
                import shutil
                shutil.copytree(model_root, target_dir, dirs_exist_ok=True)

        return wrapper_root

    def _build_model(self, language: str) -> TTS:
        language = language.upper()
        model_root = self._detect_model_root(language)
        onnx_dir = self._resolve_effective_onnx_dir(language, model_root)

        os.environ.setdefault("MELO_ZH_MIX_TOKENIZER_DIR", str(model_root))
        os.environ.setdefault("MELO_EN_TOKENIZER_DIR", str(model_root))

        return TTS(
            language=language,
            torch_device=self.torch_device,
            tts_device=self.tts_device,
            bert_device=self.bert_device,
            num_workers=self.num_workers,
            onnx_dir=str(onnx_dir),
            auto_export_onnx=False,
        )

    def get_model(self, language: str) -> TTS:
        language = language.upper()
        model = self._models.get(language)
        if model is None:
            model = self._build_model(language)
            self._models[language] = model
        return model

    def resolve_language_and_speaker(self, voice: str, forced_language: Optional[str]) -> Tuple[str, str]:
        voice = (voice or "alloy").strip()
        canonical_voice = voice.upper()

        if forced_language:
            language = forced_language.upper()
            model = self.get_model(language)
            if voice in model.hps.data.spk2id:
                return language, voice
            first = list(model.hps.data.spk2id.keys())[0]
            return language, first

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
            raise RequestError(
                f"Unknown speaker '{speaker}' for language '{language}'",
                param="voice",
            )

        if voice in OPENAI_VOICE_TO_EN_SPK:
            language = "EN"
            model = self.get_model(language)
            speaker = OPENAI_VOICE_TO_EN_SPK[voice]
            if speaker in model.hps.data.spk2id:
                return language, speaker
            first = list(model.hps.data.spk2id.keys())[0]
            return language, first

        for language in ["ZH_MIX_EN", "ZH", "EN", "JP", "KR", "ES", "FR"]:
            try:
                model = self.get_model(language)
            except Exception:
                continue
            if voice in model.hps.data.spk2id:
                return language, voice

        language = self.default_language
        model = self.get_model(language)
        first = list(model.hps.data.spk2id.keys())[0]
        return language, first

    def synthesize_to_file(self, payload: dict) -> dict:
        text = str(payload.get("input", ""))
        if not text.strip():
            raise RequestError("'input' must be non-empty", param="input")

        speed = float(payload.get("speed", 1.0))
        if not (0.25 <= speed <= 4.0):
            raise RequestError("'speed' must be in [0.25, 4.0]", param="speed")

        response_format = str(payload.get("response_format", "mp3")).lower()
        if response_format not in SUPPORTED_FORMATS:
            raise RequestError(
                f"Unsupported response_format '{response_format}'. Supported: {sorted(SUPPORTED_FORMATS)}",
                param="response_format",
            )

        output_path = payload.get("output_path")
        if not output_path:
            raise RequestError("'output_path' is required", param="output_path", status_code=500)

        voice = str(payload.get("voice", "alloy"))
        forced_language = payload.get("language")
        if forced_language is not None:
            forced_language = str(forced_language)

        language, speaker = self.resolve_language_and_speaker(voice, forced_language)
        tts = self.get_model(language)
        speaker_map = tts.hps.data.spk2id
        if speaker not in speaker_map:
            raise RequestError(
                f"Unknown speaker '{speaker}' for language '{language}'",
                param="voice",
            )

        audio = tts.tts_to_file(
            text,
            speaker_id=speaker_map[speaker],
            output_path=None,
            speed=speed,
            quiet=True,
            concurrency=self.num_workers,
        )

        audio_bytes = _encode_audio(audio, tts.hps.data.sampling_rate, response_format)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(audio_bytes)

        return {
            "content_type": CONTENT_TYPES[response_format],
            "language": language,
            "speaker": speaker,
            "bytes": len(audio_bytes),
        }


def _response_base(req_id: Optional[str]) -> dict:
    return {"id": req_id}


def _emit(resp: dict) -> bool:
    try:
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return True
    except BrokenPipeError:
        return False


def main():
    parser = argparse.ArgumentParser(description="MeloTTS C++ bridge worker")
    parser.add_argument("--default-language", default="ZH_MIX_EN")
    parser.add_argument("--onnx-dir", default="onnx_models")
    parser.add_argument("--onnx", dest="onnx_path", default=None)
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--tts-device", default="cuda")
    parser.add_argument("--bert-device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    service = CppSpeechService(
        default_language=args.default_language,
        onnx_dir=args.onnx_dir,
        onnx_path=args.onnx_path,
        torch_device=args.torch_device,
        tts_device=args.tts_device,
        bert_device=args.bert_device,
        num_workers=args.workers,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        req_id: Optional[str] = None
        try:
            payload = json.loads(line)
            req_id = str(payload.get("id")) if payload.get("id") is not None else None

            action = payload.get("action", "synthesize")
            if action == "quit":
                resp = _response_base(req_id)
                resp.update({"ok": True})
                _emit(resp)
                break

            if action != "synthesize":
                raise RequestError(f"Unsupported action '{action}'", param="action", status_code=400)

            meta = service.synthesize_to_file(payload)
            resp = _response_base(req_id)
            resp.update({"ok": True, **meta})
            if not _emit(resp):
                break
        except RequestError as exc:
            resp = _response_base(req_id)
            resp.update(
                {
                    "ok": False,
                    "status": exc.status_code,
                    "error_type": exc.error_type,
                    "error_message": exc.message,
                    "error_param": exc.param,
                    "error_code": exc.error_code,
                }
            )
            if not _emit(resp):
                break
        except Exception as exc:  # pragma: no cover
            traceback.print_exc(file=sys.stderr)
            resp = _response_base(req_id)
            resp.update(
                {
                    "ok": False,
                    "status": 500,
                    "error_type": "server_error",
                    "error_message": str(exc),
                    "error_param": None,
                    "error_code": None,
                }
            )
            if not _emit(resp):
                break


if __name__ == "__main__":
    main()
