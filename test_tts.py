from melo.api import TTS
from pathlib import Path
import time
import argparse
# Speed is adjustable
speed = 1.0


use_ov = True  ## Used to control whether to use torch or openvino
speech_enhance = True
lang = "ZH" # or ZH


# Parse args for ov device
parser = argparse.ArgumentParser(description="Select inference devices for TTS and BERT")

parser.add_argument("--tts_device", type=str, choices=["CPU", "GPU"], default="CPU",
                    help="Select inference device for TTS: CPU or GPU")
parser.add_argument("--bert_device", type=str, choices=["CPU", "GPU", "NPU"], default="CPU",
                    help="Select inference device for BERT: CPU GPU or NPU")

# Parse command-line arguments
args = parser.parse_args()
# ov device
tts_device = args.tts_device
bert_device = args.bert_device

if speech_enhance:
    from df.enhance import enhance, init_df, load_audio, save_audio
    import torchaudio
    def process_audio(input_file: str, output_file: str, new_sample_rate: int = 48000):
        """
        Load an audio file, enhance it using a DeepFilterNet, and save the result.

        Parameters:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the enhanced audio file.
        new_sample_rate (int): Desired sample rate for the output audio file (default is 48000 Hz).
        """

        model, df_state, _ = init_df()
        audio, sr = torchaudio.load(input_file)

        # Resample the WAV file to meet the requirements of DeepFilterNet
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=new_sample_rate)
        resampled_audio = resampler(audio)

        enhanced = enhance(model, df_state, resampled_audio)

        # Save the enhanced audio
        save_audio(output_file, enhanced, df_state.sr())

text = '''知名爆料人 Moore’s Law is Dead 在近期的视频中表示，PlayStation 5 Pro 不带光驱的型号价格有望低至 500 美元，因为它的生产成本不会比 PlayStation 5 高出多少。

据 SteamDB 数据显示，截至发稿，《HELLDIVERS 2》Steam 同时在线人数的峰值已经来到了 255189 人，即时玩家人数也超过了 16 万人，预计在本周末热度会进一步上涨。

每个人的电脑都会保存着大量的文档文件，就算你很花心思去组织和管理它们，时间一长东西就容易乱起来，经常要花大半天时间才找到需要的文档，急用时可谓相当尴尬。然而，Windows 自带的「搜索」功能实在是太慢了，我们都需要更快更强大的搜索工具来提高工作效率！

Everthing 正是当之无愧的 Windows 强悍文件搜索「神器」！没有之一！它能在闪电般瞬间从海量的硬盘中找到你需要的文件！速度快到难以置信！首次接触到 Everything 可真让我惊讶和兴奋了许久！而且它还是一款完全免费的软件，界面简洁高效，体积很小巧，但功能却非常丰富……'''

model = TTS()


dur_time_list = []
loop_num = 1

hps_config_path = "ov_models/hps/"
if use_ov:
    bert_path = f"ov_models/bert_multilingual"

    if not Path(bert_path).exists():
        model.bert_model.bert_convert_to_ov(bert_path, "multilingual")

    ov_path = f"ov_models/tts_ov_{lang}"
    if not Path(ov_path).exists():
        model.torch_model_init(language=lang)
        model.tts_convert_to_ov(ov_path, language= lang)

    model.bert_model.ov_bert_model_init(bert_path, bert_device = "CPU", language="multilingual")
    model.ov_model_init(ov_path, hps_config_path, language = lang)

speaker_ids = model.hps.data.spk2id
speakers = list(speaker_ids.keys())
for i in range(loop_num):
    for speaker in speakers:
        output_path = '{}_ov_{}.wav'.format(lang, speaker)
        start = time.perf_counter()
        model.tts_to_file(text, speaker_ids[speaker], output_path, speed=speed, use_ov=use_ov)
        if speech_enhance:
            print("Use speech enhance")
            process_audio(output_path,output_path)
        end = time.perf_counter()

    dur_time = (end - start) * 1000
    dur_time_list.append(dur_time)

avg_lantecy = sum(dur_time_list) / (len(dur_time_list))
print(f"MeloTTS model e2e avg latency: {avg_lantecy:.2f} ms")
print(f"text length: {len(text)}")
