"""Text-To-Speech

- coqui_xtts-v2 as offline model (supported)

- gtts-cli (online needed, network blocking issue)

"""

from io import BytesIO

#  import tempfile

import simpleaudio
import torch
from gtts import gTTS  # text to speech via google
from IPython.display import Audio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
from pygame import mixer  # noqa

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# GPU: `nvcc --version` must match `torch.__version__`
use_deepspeed = False
if not use_deepspeed:
    device = "cpu"  # CPU only; works ok

# for voice clone:
speaker_wav = "./data/wavs/LJ001-0001.wav"


def load_xtts_model():
    checkpoint_dir = "models/hfLLMs/coqui_xtts-v2/"
    config = XttsConfig()
    config.load_json(checkpoint_dir + "config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config=config,
        checkpoint_dir=checkpoint_dir,
        use_deepspeed=use_deepspeed,
        eval=True,
    )
    if device != "cpu":
        model.cuda()

    print("Loading XTTS from: ", checkpoint_dir)
    return model, config


# NOTE max chars of 82 limited in language "zh":
def coquitts_speaker(
    model,
    config,
    intext=None,
    sr=16000,
    language="en",
    save_wav=False,
):
    outputs = model.synthesize(
        text=intext,
        config=config,
        speaker_wav=speaker_wav,  # 声音克隆的音频文件
        gpt_cond_len=5,  # Length of the audio used for cloning.
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


# NOTE network blocking issue
# NOTE GOOGLE_TTS_MAX_CHARS = 100  # Max characters the Google TTS API takes at a time # noqa
def googletts_speaker(texts, lang="en", tld="com.hk"):
    mixer.init()  # pygame mixer as audio player
    mp3_fp = BytesIO()
    tts = gTTS(text=texts, lang=lang, tld=tld)
    tts.write_to_fp(mp3_fp)
    mixer.music.load(mp3_fp)
    mixer.music.play()
    #  with tempfile.NamedTemporaryFile(delete_on_close=True) as fp:
    #      tts = gTTS(text=texts, lang=lang, tld=tld)
    #      tts.save("{}.mp3".format(fp.name))
    #      mixer.music.load("{}.mp3".format(fp.name))
    #      mixer.music.play()
    print(texts)


if __name__ == "__main__":
    input_text = """It took me quite a long time to develop a voice and now
    that I have it I am not going to be silent."""
    input_text_zh = "黄四娘家花满蹊，千朵万朵压枝低"
    use_gtts = False

    if use_gtts:  # network blocking
        googletts_speaker(input_text, lang="en", tld="com")
    else:
        xtts_model, config = load_xtts_model()
        #  coquitts_speaker(xtts_model, config, intext=input_text)
        coquitts_speaker(
            xtts_model, config, intext=input_text_zh, language="zh-cn"
        )
