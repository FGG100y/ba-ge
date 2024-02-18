"""
Optimizing Bark Using Transformers
"""

import torch
from transformers import AutoProcessor, AutoModel, BarkModel
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
import simpleaudio

device = "cuda" if torch.cuda.is_available() else "cpu"
bark_model_path = "./models/hfLLMs/bark"


def load_bark_model(use_flash_attn=False):
    """Load bark and optimizing it"""
    # repo name: "suno/bark-small", or local file path:
    if use_flash_attn:
        # Using half precision and,
        # Using Flash Attention 2 as well:
        # `pip install -U flash-attn --no-build-isolation`
        model = BarkModel.from_pretrained(
            bark_model_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        ).to(device)
    else:
        model = AutoModel.from_pretrained(
            bark_model_path, torch_dtype=torch.float16
        ).to(device)
        # Or: Using Better Transformer
        # `python -m pip install optimum`
        model = model.to_bettertransformer()
    processor = AutoProcessor.from_pretrained(bark_model_path)

    #  # Using CPU offload
    model.enable_cpu_offload()

    return model, processor


def speaker(
    model, processor, prompt, voice_preset="v2/zh_speaker_4", save_wav=False
):
    """Bark TTS (it even can sing :) NOTE 但表现不稳定，需要fine-tuning"""
    #  inputs = processor(prompt, return_tensors="pt").to(device)
    inputs = processor(
        prompt, return_tensors="pt", voice_preset=voice_preset
    ).to(device)

    # generation:
    audio_array = model.generate(**inputs, do_sample=True)
    #  audio_array = model.generate(**inputs, do_sample=True)
    # send array to CPU:
    audio_array = audio_array.cpu().numpy().squeeze()

    # take the sample rate from the model config
    sample_rate = model.generation_config.sample_rate
    if save_wav:
        # save audio to disk
        write_wav("data/tmp/bark_generation.wav", sample_rate, audio_array)
        print("Save wav to: data/tmp/bark_generation.wav")

    # audio can be played in notebook
    audio = Audio(audio_array, rate=sample_rate)

    # simpleaudio to speak out loud
    play_obj = simpleaudio.play_buffer(audio.data, 1, 2, sample_rate)
    play_obj.wait_done()


if __name__ == "__main__":

    model, processor = load_bark_model()
    # model inputs:
    speak_zh = True
    if speak_zh:
        voice_preset = "v2/zh_speaker_4"
        prompt = "惊人的！我会说中文"
        prompt_long = """这是我知道的，凡我所编辑的期刊，大概是因为往往有始无终之故
        罢，销行一向就甚为寥落，然而在这样的生活艰难中，毅然预定了《莽原》全年的就
        有她。我也早觉得有写一点东西的必要了，这虽然于死者毫不相干，但在生者，却大
        抵只能如此而已。倘使我能够相信真有所谓“在天之灵”，那自然可以得到更大的安慰，
        但是，现在，却只能如此而已。可是我实在无话可说。我只觉得所住的并非人间。四
        十多个青年的血，洋溢在我的周围，使我艰于呼吸视听，那里还能有什么言语？长歌
        当哭，是必须在痛定之后的。而此后几个所谓学者文人的阴险的论调，尤使我觉得悲
        哀。我已经出离愤怒了。我将深味这非人间的浓黑的悲凉；以我的最大哀痛显示于非
        人间，使它们快意于我的苦痛，就将这作为后死者的菲薄的祭品，奉献于逝者的灵前。
        真的猛士，敢于直面惨淡的人生，敢于正视淋漓的鲜血。"""
        prompt_sing = "♪ 黄四娘家花满溪，千朵万朵压枝低 ♪"
    else:
        voice_preset = "v2/en_speaker_6"
        prompt = "Hello, my dog is cute"
        prompt_sing = "♪Hello, my dog is cute♪"
        #  inputs = processor(prompt, voice_preset=voice_preset)

    speaker(model, processor, prompt, voice_preset)
    #  speaker(model, processor, prompt_long, voice_preset)
    speaker(model, processor, prompt_sing, voice_preset)
