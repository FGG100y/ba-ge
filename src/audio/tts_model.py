"""Text-To-Speech

- coqui_xtts-v2 as offline model

- gtts-cli (online needed, network blocking issue)

"""
from io import BytesIO

import nltk
import simpleaudio
import numpy as np
import torch
from gtts import gTTS  # text to speech via google
from IPython.display import Audio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from sentence_spliter.logic_graph import (
    long_short_cuter,
    simple_cuter,
)  # V1.2.4; 依赖包 attrdict 过时，它的import需修改 collections -> collections.abc
from sentence_spliter.automata.state_machine import StateMachine
from sentence_spliter.automata.sequence import StrSequence
from tqdm import tqdm

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
#  speaker_wav = "./data/wavs/voice_东北狠人范德彪.ogg"


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
    simple_cut=False,
    save_wav=False,
):
    # Long sentences need to split first
    if len(intext) > 30:  # FIXME magic number
        if "en" in language:
            sentences = nltk.sent_tokenize(intext)
        elif "zh" in language:
            if not simple_cut:
                cuter = StateMachine(
                    #  long_short_cuter()  # not so good
                    long_short_cuter(hard_max=80, max_len=80, min_len=15)
                )
            else:  # use simple_cuter
                cuter = StateMachine(simple_cuter())
            sequence = cuter.run(StrSequence(intext))
            sentences = sequence.sentence_list()
        pieces = []
        for sentence in tqdm(sentences):
            outputs = model.synthesize(
                text=sentence,
                config=config,
                speaker_wav=speaker_wav,
                gpt_cond_len=10,
                language=language,
            )
            pieces += [outputs["wav"]]
        audio_array = np.concatenate(pieces)
    else:
        outputs = model.synthesize(
            text=intext,
            config=config,
            speaker_wav=speaker_wav,  # 声音克隆的音频文件 NOT SO GOOD
            gpt_cond_len=15,  # Length of the audio used for cloning.
            language=language,
        )
        audio_array = outputs["wav"]

    # IPython.display.Audio object:
    audio = Audio(audio_array, rate=sr)

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
    #  import tempfile
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
    text_zh_long = """这是我知道的，凡我所编辑的期刊，大概是因为往往有
    始无终之故罢，销行一向就甚为寥落，然而在这样的生活艰难中，毅然预定了
    《莽原》全年的就有她。我也早觉得有写一点东西的必要了，这虽然于死者毫不
    相干，但在生者，却大抵只能如此而已。倘使我能够相信真有所谓“在天之灵”，
    那自然可以得到更大的安慰，但是，现在，却只能如此而已。可是我实在无话可
    说。我只觉得所住的并非人间。四十多个青年的血，洋溢在我的周围，使我艰于
    呼吸视听，那里还能有什么言语？长歌当哭，是必须在痛定之后的。而此后几个
    所谓学者文人的阴险的论调，尤使我觉得悲哀。我已经出离愤怒了。我将深味这
    非人间的浓黑的悲凉；以我的最大哀痛显示于非人间，使它们快意于我的苦痛，
    就将这作为后死者的菲薄的祭品，奉献于逝者的灵前。真的猛士，敢于直面惨淡
    的人生，敢于正视淋漓的鲜血。"""

    use_gtts = False
    if use_gtts:  # network blocking
        googletts_speaker(input_text, lang="en", tld="com")
    else:
        xtts_model, config = load_xtts_model()
        coquitts_speaker(
            model=xtts_model,
            config=config,
            intext=text_zh_long,
            sr=24000,
            language="zh-cn",
        )
