"""Text-To-Speech

- coqui_xtts-v2 as base model (offline supported)

- gtts-cli (online needed)

"""
#  import os
import tempfile

import simpleaudio
import torch
from gtts import gTTS  # text to speech via google
from IPython.display import Audio
from pygame import mixer
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

device = "cuda:0" if torch.cuda.is_available() else "cpu"

use_deepspeed = True  # GPU: `nvcc --version` must match `torch.__version__`
if use_deepspeed:
    device = "cpu"  # CPU only; works ok

# for voice clone:
speaker_wav = "./data/wavs/LJ001-0001.wav"


def load_xtts_model():
    checkpoint_dir = "models/hfLLMs/coqui_xtts-v2/"
    config = XttsConfig()
    config.load_json(checkpoint_dir+"config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=checkpoint_dir,
        use_deepspeed=use_deepspeed,
        eval=True,
    )
    if device != "cpu":
        model.cuda()

    print("Loading XTTS from: ", checkpoint_dir)
    return model, config


def coquitts_speaker(
    model,
    config,
    itext=None,
    sr=16000,
    language="en",
    save_wav=False,
):
    outputs = model.synthesize(  # FIXME GPU not work
        itext,
        config,
        speaker_wav=speaker_wav,  # 声音克隆的音频文件
        gpt_cond_len=5,  # Length of the audio used for cloning.
        language=language,
    )
    wav_arr = outputs["wav"]

    breakpoint()

    # IPython.display.Audio object:
    audio = Audio(wav_arr, rate=sr)

    if save_wav:
        with open("./tmp/test.wav", "wb") as f:
            f.write(audio.data)
        print("save wav file to ./tmp/test.wav")

    # simpleaudio to speak out loud
    play_obj = simpleaudio.play_buffer(audio.data, 1, 2, sr)
    play_obj.wait_done()


def googletts_speaker(texts, lang="zh-cn"):
    mixer.init()  # pygame mixer as audio player
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts = gTTS(text=texts, lang=lang)
        tts.save("{}.mp3".format(fp.name))
        mixer.music.load("{}.mp3".format(fp.name))
        mixer.music.play()
    print(texts)


if __name__ == "__main__":
    input_text = """It took me quite a long time to develop a voice and now
    that I have it I am not going to be silent."""
    coquitts_speaker(itext=input_text)
