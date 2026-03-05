<div align="center">
  <div>&nbsp;</div>
  <img src="logo.png" width="300"/> 
</div>

<details>
  <summary>Click here to expand/collapse content</summary>
  <ul>
## Introduction
MeloTTS is a **high-quality multi-lingual** text-to-speech library by [MIT](https://www.mit.edu/) and [MyShell.ai](https://myshell.ai). Supported languages include:

| Language | Example |
| --- | --- |
| English (American)    | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-US/speed_1.0/sent_000.wav) |
| English (British)     | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-BR/speed_1.0/sent_000.wav) |
| English (Indian)      | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN_INDIA/speed_1.0/sent_000.wav) |
| English (Australian)  | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-AU/speed_1.0/sent_000.wav) |
| English (Default)     | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-Default/speed_1.0/sent_000.wav) |
| Spanish               | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/es/ES/speed_1.0/sent_000.wav) |
| French                | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/fr/FR/speed_1.0/sent_000.wav) |
| Chinese (mix EN)      | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/zh/ZH/speed_1.0/sent_008.wav) |
| Japanese              | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/jp/JP/speed_1.0/sent_000.wav) |
| Korean                | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/kr/KR/speed_1.0/sent_000.wav) |

Some other features include:
- The Chinese speaker supports `mixed Chinese and English`.
- Fast enough for `CPU real-time inference`.

## Usage
- [Use without Installation](docs/quick_use.md)
- [Install and Use Locally](docs/install.md)
- [Training on Custom Dataset](docs/training.md)

The Python API and model cards can be found in [this repo](https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md#python-api) or on [HuggingFace](https://huggingface.co/myshell-ai).

## Join the Community

**Discord**

Join our [Discord community](https://discord.gg/myshell) and select the `Developer` role upon joining to gain exclusive access to our developer-only channel! Don't miss out on valuable discussions and collaboration opportunities.

**Contributing**

If you find this work useful, please consider contributing to this repo.

- Many thanks to [@fakerybakery](https://github.com/fakerybakery) for adding the Web UI and CLI part.

## Authors

- [Wenliang Zhao](https://wl-zhao.github.io) at Tsinghua University
- [Xumin Yu](https://yuxumin.github.io) at Tsinghua University
- [Zengyi Qin](https://www.qinzy.tech) at MIT and MyShell

**Citation**
```
@software{zhao2024melo,
  author={Zhao, Wenliang and Yu, Xumin and Qin, Zengyi},
  title = {MeloTTS: High-quality Multi-lingual Multi-accent Text-to-Speech},
  url = {https://github.com/myshell-ai/MeloTTS},
  year = {2023}
}
```

## License

This library is under MIT License, which means it is free for both commercial and non-commercial use.

## Acknowledgements

This implementation is based on [TTS](https://github.com/coqui-ai/TTS), [VITS](https://github.com/jaywalnut310/vits), [VITS2](https://github.com/daniilrobnikov/vits2) and [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2). We appreciate their awesome work.

  </ul>
</details>

## Update Notes
### 2024/08/21
* Added ONNX Runtime inference backend.
### 2024/08/28
* TTS and BERT can be exported to ONNX models.

### 2024/09/22
* Added sentence-level concurrent inference support.


## Install MeloTTS with ONNX Runtime (CUDA)

```
uv sync
uv run python -m unidic download
uv run python -m nltk.downloader averaged_perceptron_tagger_eng
uv add deepfilternet # optional for enhancing speech
```

Jetson AGX Orin (CUDA 12.8) recommendation:
```bash
# install onnxruntime-gpu wheel compatible with your JetPack/CUDA first
uv add onnxruntime-gpu
```

## Export ONNX and run inference
```shell
uv run python test_tts.py --language ZH --torch_device cuda --tts_device cuda --bert_device cuda --workers 4 --export_onnx
```

## Demo
`test_tts.py` now supports:
- ONNX Runtime CUDA inference (`--tts_device cuda --bert_device cuda`)
- Sentence-level concurrent inference (`--workers N`)
- Optional ONNX export (`--export_onnx`)

## OpenAI-Compatible Speech API
Start server:
```bash
uv run melo-openai-speech --host 0.0.0.0 --port 8000
```

Request example:
```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \\
  -H \"Content-Type: application/json\" \\
  -d '{
    \"model\": \"tts-1\",
    \"input\": \"你好，欢迎使用 MeloTTS\",
    \"voice\": \"ZH\",
    \"response_format\": \"mp3\",
    \"speed\": 1.0
  }' --output speech.mp3
```
