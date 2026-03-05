from pathlib import Path
import argparse
import time

from melo.api import TTS


parser = argparse.ArgumentParser(description="MeloTTS ONNX Runtime inference")
parser.add_argument("--language", type=str, default="ZH", help="Language model, e.g. ZH/EN/JP/KR/ES/FR")
parser.add_argument("--torch_device", type=str, default="auto", help="auto/cpu/cuda/cuda:0")
parser.add_argument("--tts_device", type=str, default="auto", help="auto/cpu/cuda")
parser.add_argument("--bert_device", type=str, default="auto", help="auto/cpu/cuda")
parser.add_argument("--onnx_dir", type=str, default="onnx_models", help="Directory for ONNX files")
parser.add_argument("--hps_config_path", type=str, default=None, help="Optional local hps config root")
parser.add_argument("--workers", type=int, default=4, help="Concurrent workers for sentence inference")
parser.add_argument("--export_onnx", action="store_true", help="Force export ONNX from torch checkpoint")
parser.add_argument("--speech_enhance", action="store_true", help="Enable DeepFilterNet post-processing")
args = parser.parse_args()

speed = 1.0
lang = args.language.upper()

if args.speech_enhance:
    from df.enhance import enhance, init_df, save_audio
    import torchaudio

    def process_audio(input_file: str, output_file: str, new_sample_rate: int = 48000):
        model, df_state, _ = init_df()
        audio, sr = torchaudio.load(input_file)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=new_sample_rate)
        resampled_audio = resampler(audio)
        enhanced = enhance(model, df_state, resampled_audio)
        save_audio(output_file, enhanced, df_state.sr())


text = """知名爆料人 Moore’s Law is Dead 在近期的视频中表示，PlayStation 5 Pro 不带光驱的型号价格有望低至 500 美元，因为它的生产成本不会比 PlayStation 5 高出多少。

据 SteamDB 数据显示，截至发稿，《HELLDIVERS 2》Steam 同时在线人数的峰值已经来到了 255189 人，即时玩家人数也超过了 16 万人，预计在本周末热度会进一步上涨。

每个人的电脑都会保存着大量的文档文件，就算你很花心思去组织和管理它们，时间一长东西就容易乱起来，经常要花大半天时间才找到需要的文档，急用时可谓相当尴尬。然而，Windows 自带的「搜索」功能实在是太慢了，我们都需要更快更强大的搜索工具来提高工作效率！

Everthing 正是当之无愧的 Windows 强悍文件搜索「神器」！没有之一！它能在闪电般瞬间从海量的硬盘中找到你需要的文件！速度快到难以置信！首次接触到 Everything 可真让我惊讶和兴奋了许久！而且它还是一款完全免费的软件，界面简洁高效，体积很小巧，但功能却非常丰富……"""

model = TTS(
    torch_device=args.torch_device,
    bert_device=args.bert_device,
    tts_device=args.tts_device,
    num_workers=args.workers,
)

onnx_root = Path(args.onnx_dir) / f"tts_onnx_{lang}"
tts_onnx = onnx_root / f"tts_{lang}.onnx"
bert_onnx = onnx_root / "bert_multilingual.onnx"

if args.export_onnx or not (tts_onnx.exists() and bert_onnx.exists()):
    model.torch_model_init(language=lang, torch_device=args.torch_device)
    model.tts_convert_to_onnx(str(onnx_root), language=lang)

model.onnx_model_init(
    onnx_path=str(onnx_root),
    hps_config_path=args.hps_config_path,
    language=lang,
)

speaker_ids = model.hps.data.spk2id
speakers = list(speaker_ids.keys())

dur_time_list = []
loop_num = 1

for _ in range(loop_num):
    for speaker in speakers:
        output_path = f"{lang}_onnx_{speaker}.wav"
        start = time.perf_counter()
        model.tts_to_file(
            text,
            speaker_ids[speaker],
            output_path,
            speed=speed,
            concurrency=args.workers,
        )
        if args.speech_enhance:
            process_audio(output_path, output_path)
        end = time.perf_counter()

    dur_time_ms = (end - start) * 1000
    dur_time_list.append(dur_time_ms)

avg_latency = sum(dur_time_list) / len(dur_time_list)
print(f"MeloTTS (onnx) e2e avg latency: {avg_latency:.2f} ms")
print(f"text length: {len(text)}")
print(f"workers: {args.workers}")
