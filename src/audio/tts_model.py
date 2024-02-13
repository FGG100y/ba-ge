"""Text-To-Speech

- coqui_xtts-v2 as base model (offline supported)

- gtts-cli (online needed)

"""
import os
import tempfile

import simpleaudio
import torch
from gtts import gTTS  # text to speech via google
from IPython.display import Audio
from pygame import mixer
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

device = "cuda:0" if torch.cuda.is_available() else "cpu"
config = XttsConfig()
config.load_json("/home/ds01/hfLLMs/coqui_xtts-v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_dir="/home/ds01/hfLLMs/coqui_xtts-v2/",
    eval=True,
    use_deepspeed=True,
)
if device != "cpu":
    model.cuda()

print("Loading XTTS")


speaker_wav = "./data/wavs/LJ001-0001.wav"
input_text = """It took me quite a long time to develop a voice and now that I
have it I am not going to be silent."""


def coquitts_speaker(
    model=model,
    config=config,
    itext=input_text,
    sr=16000,
    language="en",
    save_wav=False,
):
    outputs = model.synthesize(
        itext,
        config,
        speaker_wav=speaker_wav,  # 声音克隆
        gpt_cond_len=3,  # what's this?
        language=language,
    )
    wav_arr = outputs["wav"]

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
    coquitts_speaker(itext=input_text)
