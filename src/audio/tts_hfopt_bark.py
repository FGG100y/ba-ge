"""
Optimizing Bark Using Transformers
"""

import nltk
import torch
from transformers import AutoProcessor, AutoModel, BarkModel
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
import simpleaudio
import numpy as np
from sentence_spliter.logic_graph import (
    long_short_cuter,
    simple_cuter,
)  # V1.2.4; 依赖包 attrdict 过时，它的import需修改 collections -> collections.abc
from sentence_spliter.automata.state_machine import StateMachine
from sentence_spliter.automata.sequence import StrSequence
from tqdm import tqdm

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
    model,
    processor,
    text_prompt,
    voice_preset="v2/zh_speaker_4",  # [0,9]; 4,6,7,9 -> female, others -> male
    simple_cut=True,
    save_wav=False,
):
    """Bark TTS (it even can sing :) BUT 表现不稳定，需要fine-tuning"""

    sample_rate = model.generation_config.sample_rate
    silence = np.zeros(int(0.25 * sample_rate))  # quarter second of silence
    if len(text_prompt) > 80:  # Long sentences need to split first
        if "en_speaker" in voice_preset:
            sentences = nltk.sent_tokenize(text_prompt)
        elif "zh_speaker" in voice_preset:
            if not simple_cut:
                cuter = StateMachine(
                    long_short_cuter(hard_max=16, max_len=64, min_len=8)
                )
            else:  # use simple_cuter
                cuter = StateMachine(simple_cuter())
            sequence = cuter.run(StrSequence(text_prompt))
            sentences = sequence.sentence_list()

        pieces = []
        for sentence in tqdm(sentences):
            inputs = processor(
                sentence, return_tensors="pt", voice_preset=voice_preset
            ).to(device)
            audio_array = model.generate(**inputs, do_sample=True)
            pieces += [audio_array.cpu().numpy().squeeze(), silence.copy()]
        audio_array = np.concatenate(pieces)
    else:
        #  inputs = processor(text_prompt, return_tensors="pt").to(device)
        inputs = processor(
            text_prompt, return_tensors="pt", voice_preset=voice_preset
        ).to(device)
        # generation:
        audio_array = model.generate(**inputs, do_sample=True)
        # send array to CPU:
        audio_array = audio_array.cpu().numpy().squeeze()

    # take the sample rate from the model config
    if save_wav:
        sample_rate = model.generation_config.sample_rate
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
        text_prompt = "难以置信！我会说中文"
        text_prompt_long = """这是我知道的，凡我所编辑的期刊，大概是因为往往有
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
        text_prompt_sing = "♪黄四娘家花满溪，千朵万朵压枝低 ♪"
    else:
        voice_preset = "v2/en_speaker_6"
        text_prompt = "Hello, my dog is cute"
        # FIXME long paragraph not work well in bark tts
        text_prompt_long = """
        There was nothing so very remarkable in that; nor did Alice think it so
        very much out of the way to hear the Rabbit say to itself, `Oh dear! Oh
        dear! I shall be late!' (when she thought it over afterwards, it
        occurred to her that she ought to have wondered at this, but at the
        time it all seemed quite natural); but when the Rabbit actually took a
        watch out of its waistcoat-pocket, and looked at it, and then hurried
        on, Alice started to her feet, for it flashed across her mind that she
        had never before seen a rabbit with either a waistcoat-pocket, or a
        watch to take out of it, and burning with curiosity, she ran across the
        field after it, and fortunately was just in time to see it pop down a
        large rabbit-hole under the hedge. In another moment down went Alice
        after it, never once considering how in the world she was to get out
        again. The rabbit-hole went straight on like a tunnel for some way, and then
        dipped suddenly down, so suddenly that Alice had not a moment to think
        about stopping herself before she found herself falling down a very
        deep well. Either the well was very deep, or she fell very slowly, for she had
        plenty of time as she went down to look about her and to wonder what
        was going to happen next. First, she tried to look down and make out
        what she was coming to, but it was too dark to see anything; then she
        looked at the sides of the well, and noticed that they were filled with
        cupboards and book-shelves; here and there she saw maps and pictures
        hung upon pegs. She took down a jar from one of the shelves as she
        passed; it was labelled `ORANGE MARMALADE', but to her great
        disappointment it was empty: she did not like to drop the jar for fear
        of killing somebody, so managed to put it into one of the cupboards as
        she fell past it. `Well!' thought Alice to herself, `after such a fall
        as this, I shall think nothing of tumbling down stairs! How brave
        they'll all think me at home! Why, I wouldn't say anything about it,
        even if I fell off the top of the house!' (Which was very likely true.)
        """
        text_prompt_sing = "♪Hello, my dog is cute♪"

    speaker(model, processor, text_prompt, voice_preset)
    speaker(model, processor, text_prompt_long, voice_preset)
    speaker(model, processor, text_prompt_sing, voice_preset)
