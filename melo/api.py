import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import soundfile
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from . import utils
from .download_utils import load_or_download_config, load_or_download_model
from .models import SynthesizerTrn
from .split_utils import split_sentence

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def _require_onnxruntime():
    if ort is None:
        raise ImportError("onnxruntime is required for ONNX backend. Please install onnxruntime-gpu/onnxruntime.")


def _resolve_torch_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if "cuda" in device:
        assert torch.cuda.is_available(), "CUDA device requested but torch.cuda.is_available() is False"
    return device


def _normalize_language_for_model(language: str) -> str:
    language = language.upper()
    if language == "ZH_MIX_EN":
        return "ZH"
    return language


def _resolve_ort_providers(device: str) -> List[str]:
    if ort is None:
        return ["CPUExecutionProvider"]
    normalized = (device or "auto").lower()
    if normalized in {"auto", "gpu", "cuda", "cuda:0", "cuda:1"}:
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    if normalized in {"cpu"}:
        return ["CPUExecutionProvider"]
    if device in ort.get_available_providers():
        return [device, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _to_numpy(data, dtype=None) -> np.ndarray:
    if isinstance(data, np.ndarray):
        arr = data
    elif isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype)
    return arr


class _BertExportModel(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.model = base_model

    def forward(self, input_ids, token_type_ids, attention_mask):
        out = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return torch.cat(out["hidden_states"][-3:-2], -1)[0]


class _TTSExportModel(nn.Module):
    def __init__(self, base_model: SynthesizerTrn):
        super().__init__()
        self.model = base_model

    def forward(
        self,
        phones,
        phones_length,
        speakers,
        tones,
        lang_ids,
        bert,
        ja_bert,
        noise_scale,
        length_scale,
        noise_scale_w,
        sdp_ratio,
    ):
        audio, *_ = self.model(
            phones,
            phones_length,
            speakers,
            tones,
            lang_ids,
            bert,
            ja_bert,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sdp_ratio=sdp_ratio,
        )
        return audio


class Bert:
    def __init__(self, device: str = "auto", providers: Optional[Sequence[str]] = None, use_threads: int = 4):
        self.device = device
        self.providers = list(providers) if providers else _resolve_ort_providers(device)
        self.use_threads = max(int(use_threads), 1)

        self.bert_tokenizer = None
        self.bert_config = None
        self.bert_session = None
        self.bert_input_names = {}
        self.bert_output_name = None

    def save_tokenizer(self, tokenizer, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(out_dir)

    def bert_convert_to_onnx(
        self,
        onnx_dir: str,
        model_id: str = "bert-base-multilingual-uncased",
        opset_version: int = 17,
    ):
        _require_onnxruntime()
        onnx_dir = Path(onnx_dir)
        onnx_dir.mkdir(parents=True, exist_ok=True)

        base_model = AutoModelForMaskedLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)

        export_model = _BertExportModel(base_model).eval()

        sample_text = "当需要把翻译对象表示, 可以使用这个方法。A buffer is a container for data."
        encoded = tokenizer(sample_text, return_tensors="pt")

        onnx_model_path = onnx_dir / "bert_multilingual.onnx"
        torch.onnx.export(
            export_model,
            (
                encoded["input_ids"],
                encoded["token_type_ids"],
                encoded["attention_mask"],
            ),
            str(onnx_model_path),
            input_names=["input_ids", "token_type_ids", "attention_mask"],
            output_names=["hidden_states"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "token_type_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "hidden_states": {0: "seq"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )

        self.save_tokenizer(tokenizer, onnx_dir)
        config.save_pretrained(onnx_dir)

        return str(onnx_model_path)

    def onnx_bert_model_init(self, onnx_path: str, providers: Optional[Sequence[str]] = None):
        _require_onnxruntime()
        path = Path(onnx_path)
        if path.is_dir():
            candidate = path / "bert_multilingual.onnx"
            if not candidate.exists():
                all_onnx = sorted(path.glob("*.onnx"))
                if not all_onnx:
                    raise FileNotFoundError(f"No ONNX model found in {path}")
                candidate = all_onnx[0]
            model_path = candidate
            tokenizer_path = path
        else:
            model_path = path
            tokenizer_path = path.parent

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.use_threads
        sess_options.inter_op_num_threads = 1
        active_providers = list(providers) if providers else self.providers

        self.bert_session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=active_providers,
        )

        self.bert_input_names = {inp.name for inp in self.bert_session.get_inputs()}
        self.bert_output_name = self.bert_session.get_outputs()[0].name

        self.bert_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.bert_config = AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)

    def onnx_bert_infer(self, input_ids=None, token_type_ids=None, attention_mask=None):
        _require_onnxruntime()
        if self.bert_session is None:
            raise RuntimeError("BERT ONNX session is not initialized. Call onnx_bert_model_init first.")

        feed = {
            "input_ids": _to_numpy(input_ids, np.int64),
            "token_type_ids": _to_numpy(token_type_ids, np.int64),
            "attention_mask": _to_numpy(attention_mask, np.int64),
        }
        return self.bert_session.run([self.bert_output_name], feed)[0]

    def get_onnx_bert_feature(self, text: str, word2ph: List[int]) -> torch.Tensor:
        if self.bert_session is None or self.bert_tokenizer is None:
            raise RuntimeError("BERT ONNX model is not ready. Call onnx_bert_model_init first.")

        encoded = self.bert_tokenizer(text, return_tensors="np")
        res = self.onnx_bert_infer(
            input_ids=encoded["input_ids"],
            token_type_ids=encoded["token_type_ids"],
            attention_mask=encoded["attention_mask"],
        )
        res = torch.from_numpy(_to_numpy(res, np.float32))

        if len(word2ph) > res.shape[0]:
            raise ValueError(
                f"BERT token len({res.shape[0]}) is smaller than word2ph len({len(word2ph)})."
            )

        phone_level_feature = []
        for i, repeat_count in enumerate(word2ph):
            phone_level_feature.append(res[i].repeat(repeat_count, 1))

        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

class TTS(nn.Module):
    def __init__(
        self,
        language: Optional[str] = None,
        device: str = "auto",
        use_hf: bool = True,
        config_path: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        torch_device: Optional[str] = None,
        bert_device: str = "auto",
        tts_device: str = "auto",
        num_workers: int = 4,
        onnx_dir: str = "onnx_models",
        auto_export_onnx: bool = True,
        onnx_providers: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        requested_torch_device = torch_device if torch_device is not None else device
        self.device = _resolve_torch_device(requested_torch_device)
        self.bert_device = bert_device
        self.tts_device = tts_device
        self.num_workers = max(int(num_workers), 1)
        self.onnx_dir = Path(onnx_dir)
        self.auto_export_onnx = auto_export_onnx

        self.model = None
        self.tts_session = None
        self.tts_output_name = None
        self.tts_input_name_map = {}

        self.bert_model = Bert(
            device=bert_device,
            providers=onnx_providers if onnx_providers else _resolve_ort_providers(bert_device),
            use_threads=self.num_workers,
        )

        if language:
            requested_language = language.upper()
            onnx_root = self.onnx_dir / f"tts_onnx_{requested_language}"
            tts_onnx = onnx_root / f"tts_{requested_language}.onnx"
            bert_onnx = onnx_root / "bert_multilingual.onnx"

            if not (tts_onnx.exists() and bert_onnx.exists()):
                if not self.auto_export_onnx:
                    raise FileNotFoundError(
                        f"Missing ONNX files under {onnx_root}. "
                        "Set auto_export_onnx=True or export models first."
                    )
                self.torch_model_init(
                    language=requested_language,
                    torch_device=self.device,
                    use_hf=use_hf,
                    config_path=config_path,
                    ckpt_path=ckpt_path,
                )
                self.tts_convert_to_onnx(str(onnx_root), language=requested_language)

            self.onnx_model_init(
                onnx_path=str(onnx_root),
                language=requested_language,
            )

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.0):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        return np.asarray(audio_segments, dtype=np.float32)

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print("\n".join(texts))
            print(" > ===========================")
        return texts

    def torch_model_init(
        self,
        language,
        torch_device="cpu",
        use_hf=True,
        config_path=None,
        ckpt_path=None,
    ):
        torch_device = _resolve_torch_device(torch_device)

        requested_language = language.upper()
        base_language = _normalize_language_for_model(requested_language)
        hps = load_or_download_config(base_language, use_hf=use_hf, config_path=config_path)
        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            **hps.model,
        ).to(torch_device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = torch_device

        checkpoint_dict = load_or_download_model(base_language, torch_device, use_hf=use_hf, ckpt_path=ckpt_path)
        self.model.load_state_dict(checkpoint_dict["model"], strict=True)

        self.language = requested_language
        if self.language == "EN":
            try:
                import nltk

                nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            except Exception:
                pass

    def tts_convert_to_onnx(
        self,
        onnx_path: str,
        language: str = "ZH",
        sdp_ratio: float = 0.2,
        noise_scale: float = 0.6,
        noise_scale_w: float = 0.8,
        speed: float = 1.0,
        opset_version: int = 17,
        export_bert: bool = True,
    ):
        _require_onnxruntime()
        if self.model is None:
            raise RuntimeError("Torch model is not initialized. Call torch_model_init first.")

        output_dir = Path(onnx_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        if export_bert:
            self.bert_model.bert_convert_to_onnx(str(output_dir))

        export_model = _TTSExportModel(self.model).eval()
        x_tst = torch.tensor(
            [[0, 0, 0, 97, 0, 65, 0, 100, 0, 89, 0, 55, 0, 49, 0, 100, 0, 13, 0, 98, 0, 95, 0, 98]],
            dtype=torch.int64,
            device=self.device,
        )
        x_tst_lengths = torch.tensor([x_tst.shape[1]], dtype=torch.int64, device=self.device)
        speakers = torch.tensor([1], dtype=torch.int64, device=self.device)
        tones = torch.zeros_like(x_tst)
        lang_ids = torch.zeros_like(x_tst)
        bert = torch.zeros((1, 1024, x_tst.shape[1]), dtype=torch.float32, device=self.device)
        ja_bert = torch.zeros((1, 768, x_tst.shape[1]), dtype=torch.float32, device=self.device)
        noise_scale_t = torch.tensor([noise_scale], dtype=torch.float32, device=self.device)
        length_scale_t = torch.tensor([1.0 / speed], dtype=torch.float32, device=self.device)
        noise_scale_w_t = torch.tensor([noise_scale_w], dtype=torch.float32, device=self.device)
        sdp_ratio_t = torch.tensor([sdp_ratio], dtype=torch.float32, device=self.device)

        onnx_file = output_dir / f"tts_{language}.onnx"
        torch.onnx.export(
            export_model,
            (
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                noise_scale_t,
                length_scale_t,
                noise_scale_w_t,
                sdp_ratio_t,
            ),
            str(onnx_file),
            input_names=[
                "phones",
                "phones_length",
                "speakers",
                "tones",
                "lang_ids",
                "bert",
                "ja_bert",
                "noise_scale",
                "length_scale",
                "noise_scale_w",
                "sdp_ratio",
            ],
            output_names=["audio"],
            dynamic_axes={
                "phones": {0: "batch", 1: "seq"},
                "phones_length": {0: "batch"},
                "tones": {0: "batch", 1: "seq"},
                "lang_ids": {0: "batch", 1: "seq"},
                "bert": {0: "batch", 2: "seq"},
                "ja_bert": {0: "batch", 2: "seq"},
                "audio": {0: "batch", 2: "audio_len"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )

        return str(onnx_file)

    def _resolve_tts_input_name_map(self):
        aliases = {
            "phones": ["phones", "x"],
            "phones_length": ["phones_length", "x_lengths"],
            "speakers": ["speakers", "sid"],
            "tones": ["tones", "tone"],
            "lang_ids": ["lang_ids", "language"],
            "bert": ["bert"],
            "ja_bert": ["ja_bert"],
            "noise_scale": ["noise_scale"],
            "length_scale": ["length_scale"],
            "noise_scale_w": ["noise_scale_w"],
            "sdp_ratio": ["sdp_ratio"],
        }

        real_names = {inp.name for inp in self.tts_session.get_inputs()}
        mapping: Dict[str, str] = {}
        for canonical, candidates in aliases.items():
            for candidate in candidates:
                if candidate in real_names:
                    mapping[canonical] = candidate
                    break
            if canonical not in mapping:
                raise KeyError(f"Input '{canonical}' is missing in ONNX model inputs: {sorted(real_names)}")
        return mapping

    def onnx_model_init(
        self,
        onnx_path: str,
        hps_config_path: Optional[str] = None,
        language: str = "ZH",
        tts_providers: Optional[Sequence[str]] = None,
        bert_providers: Optional[Sequence[str]] = None,
    ):
        _require_onnxruntime()
        requested_language = language.upper()
        base_language = _normalize_language_for_model(requested_language)
        if hps_config_path:
            cfg = f"{hps_config_path}/{base_language}/config.json"
            hps = load_or_download_config(base_language, use_hf=False, config_path=cfg)
        else:
            hps = load_or_download_config(base_language, use_hf=True)

        self.symbol_to_id = {s: i for i, s in enumerate(hps.symbols)}
        self.hps = hps
        self.language = requested_language

        onnx_root = Path(onnx_path)
        if onnx_root.is_file():
            tts_model_file = onnx_root
            onnx_root = onnx_root.parent
        else:
            candidate = onnx_root / f"tts_{requested_language}.onnx"
            if not candidate.exists():
                fallback = onnx_root / "tts.onnx"
                if fallback.exists():
                    candidate = fallback
                else:
                    all_onnx = [x for x in sorted(onnx_root.glob("*.onnx")) if "bert" not in x.name]
                    if not all_onnx:
                        raise FileNotFoundError(f"No TTS ONNX model found in {onnx_root}")
                    candidate = all_onnx[0]
            tts_model_file = candidate

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.num_workers
        sess_options.inter_op_num_threads = 1

        active_tts_providers = list(tts_providers) if tts_providers else _resolve_ort_providers(self.tts_device)
        self.tts_session = ort.InferenceSession(
            str(tts_model_file),
            sess_options=sess_options,
            providers=active_tts_providers,
        )
        self.tts_output_name = self.tts_session.get_outputs()[0].name
        self.tts_input_name_map = self._resolve_tts_input_name_map()

        bert_onnx = onnx_root / "bert_multilingual.onnx"
        if not bert_onnx.exists():
            raise FileNotFoundError(f"Required BERT ONNX file not found: {bert_onnx}")
        self.bert_model.onnx_bert_model_init(str(onnx_root), providers=bert_providers)

    def onnx_infer(
        self,
        x_tst=None,
        x_tst_lengths=None,
        speakers=None,
        tones=None,
        lang_ids=None,
        bert=None,
        ja_bert=None,
        sdp_ratio=0.2,
        noise_scale=0.6,
        noise_scale_w=0.8,
        speed=1.0,
    ) -> np.ndarray:
        _require_onnxruntime()
        if self.tts_session is None:
            raise RuntimeError("TTS ONNX session is not initialized. Call onnx_model_init first.")

        feed = {
            self.tts_input_name_map["phones"]: _to_numpy(x_tst, np.int64),
            self.tts_input_name_map["phones_length"]: _to_numpy(x_tst_lengths, np.int64),
            self.tts_input_name_map["speakers"]: _to_numpy(speakers, np.int64),
            self.tts_input_name_map["tones"]: _to_numpy(tones, np.int64),
            self.tts_input_name_map["lang_ids"]: _to_numpy(lang_ids, np.int64),
            self.tts_input_name_map["bert"]: _to_numpy(bert, np.float32),
            self.tts_input_name_map["ja_bert"]: _to_numpy(ja_bert, np.float32),
            self.tts_input_name_map["noise_scale"]: np.asarray([noise_scale], dtype=np.float32),
            self.tts_input_name_map["length_scale"]: np.asarray([1.0 / speed], dtype=np.float32),
            self.tts_input_name_map["noise_scale_w"]: np.asarray([noise_scale_w], dtype=np.float32),
            self.tts_input_name_map["sdp_ratio"]: np.asarray([sdp_ratio], dtype=np.float32),
        }

        audio = self.tts_session.run([self.tts_output_name], feed)[0]
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim >= 3:
            return audio[0, 0]
        if audio.ndim == 2:
            return audio[0]
        return audio.reshape(-1)

    def _infer_segment(
        self,
        t: str,
        speaker_id: int,
        language: str,
        sdp_ratio: float,
        noise_scale: float,
        noise_scale_w: float,
        speed: float,
    ) -> np.ndarray:
        if language in ["EN", "ZH_MIX_EN"]:
            t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)

        bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
            t,
            language,
            self.hps,
            self.device,
            symbol_to_id=self.symbol_to_id,
            bert_model=self.bert_model,
        )

        x_tst = phones.unsqueeze(0)
        tones = tones.unsqueeze(0)
        lang_ids = lang_ids.unsqueeze(0)
        bert = bert.unsqueeze(0)
        ja_bert = ja_bert.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)])
        speakers = torch.LongTensor([speaker_id])

        audio = self.onnx_infer(
            x_tst=x_tst,
            x_tst_lengths=x_tst_lengths,
            speakers=speakers,
            tones=tones,
            lang_ids=lang_ids,
            bert=bert,
            ja_bert=ja_bert,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            speed=speed,
        )

        return utils.fix_loudness(audio, self.hps.data.sampling_rate)

    def tts_to_file(
        self,
        text,
        speaker_id,
        output_path=None,
        sdp_ratio=0.2,
        noise_scale=0.6,
        noise_scale_w=0.8,
        speed=1.0,
        pbar=None,
        format=None,
        position=None,
        quiet=False,
        concurrency: Optional[int] = None,
    ):
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet)

        if self.tts_session is None:
            raise RuntimeError("ONNX session is not initialized. Call onnx_model_init first.")

        worker_count = 1 if pbar else (concurrency if concurrency is not None else self.num_workers)
        worker_count = max(int(worker_count), 1)

        audio_list = [None] * len(texts)

        if worker_count == 1 or len(texts) <= 1:
            if pbar:
                iterator = pbar(texts)
            else:
                if position:
                    iterator = tqdm(texts, position=position)
                elif quiet:
                    iterator = texts
                else:
                    iterator = tqdm(texts)

            for idx, segment_text in enumerate(iterator):
                audio_list[idx] = self._infer_segment(
                    t=segment_text,
                    speaker_id=speaker_id,
                    language=language,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    speed=speed,
                )
        else:
            progress = tqdm(total=len(texts), disable=quiet, position=position)
            with ThreadPoolExecutor(max_workers=min(worker_count, len(texts))) as executor:
                futures = {
                    executor.submit(
                        self._infer_segment,
                        segment_text,
                        speaker_id,
                        language,
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                        speed,
                    ): idx
                    for idx, segment_text in enumerate(texts)
                }

                for future in as_completed(futures):
                    idx = futures[future]
                    audio_list[idx] = future.result()
                    progress.update(1)
            progress.close()

        if "cuda" in self.device:
            torch.cuda.empty_cache()

        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio

        if format:
            soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
        else:
            soundfile.write(output_path, audio, self.hps.data.sampling_rate)
