import openai

from audio import stt_model, tts_hfopt_bark, tts_elevenlabs, tts_model
from nlp import llm_model
from wake_word.wake_gamgin_stream import wake_gamgin
from utils import load_yaml_cfg as yamlcfg


init = 1
wake_again = False

# TODO test configuration 2024-02-18 Sun
# preset tts greetings, goodbyes, exit-words
preset_shorts = yamlcfg.load_config()
EXIT_WORDS_L = preset_shorts["tts_greetings"]["exit_words"]
HELLOS_D = preset_shorts["tts_greetings"]["hellos"]
GOODBYES_D = preset_shorts["tts_greetings"]["goodbyes"]
LLM_BACKUPS_D = preset_shorts["llm_backups"]

make_bot_polite = True
use_xtts = False
if use_xtts:
    # use coqui-ai xtts
    XTTS_MODEL, CONFIG = tts_model.load_xtts_model()
else:
    # use suno bark from huggingface transformers
    hf_bark, bark_processor = tts_hfopt_bark.load_bark_model()


#  use_faster_whisper = True
faster_whisper = stt_model.load_faster_whisper()


def tts_greeting(
    greeting,
    use_bark=True,
    use_hf_bark=True,
    use_11labs=False,
    use_gtts=False,
    xtts_sr=16000,
):
    if use_bark:
        if use_hf_bark:  # from transformers 4.31 onward
            tts_hfopt_bark.speaker(
                model=hf_bark,
                processor=bark_processor,
                text_prompt=greeting,
                save_wav=False,
            )
        else:
            raise NotImplementedError  # more details see tts_suno_bark.py
    elif use_11labs:  # only free for 10000 chars
        tts_elevenlabs.lllabs_speaker(
            intext=greeting, voice="Rachel", model="eleven_multilingual_v2"
        )
    elif use_gtts:  # FIXME gtts not work (mainly) due to network issues
        tts_model.googletts_speaker(greeting, lang="zh-CN")
    else:  # only 82 chars for 'zh'
        tts_model.coquitts_speaker(
            model=XTTS_MODEL,
            config=CONFIG,
            intext=greeting,
            sr=xtts_sr,
            language="zh-cn",
            save_wav=False,
        )


while True:
    # PART01: wake word detection

    if not init and wake_again:
        hello_again = HELLOS_D[
            "hello_again"
        ]
        tts_greeting(hello_again, xtts_sr=24000)

    if init and wake_gamgin():
        # responding the calling:
        hello = HELLOS_D["say_hello"]
        tts_greeting(hello, xtts_sr=24000)
        init = 0

    # PART02: speech2text (faster-whisper-large), input and transcribe
    speech2text = stt_model.transcribe_fast(
        model=faster_whisper, duration=50, verbose=1
    )

    say_goodbye = [w for w in EXIT_WORDS_L if w in speech2text]
    if len(say_goodbye) > 0:
        if make_bot_polite:
            intext = GOODBYES_D["say_goodbye"]
            tts_greeting(intext, xtts_sr=24000)

        #  break  # 结束程序

        # 如果用户说再见，则进入等待唤醒状态
        if wake_gamgin():
            wake_again = True
            continue

    # PART03: query the LLMs
    try:
        #  # limit response lenght < 72 chars (coqui-tts limit):
        #  prompt = speech2text + " 请简答，并且字数少于72个。"
        # NOTE 使用 Bark model 部分解决了这个问题（但速度成了另一个问题）
        prompt = speech2text
        llm_response = llm_model.query_llm(query=prompt, verbose=True)
        no_llm = False
    except (ConnectionRefusedError, openai.APIConnectionError):
        llm_response = LLM_BACKUPS_D[
            "no_service"
        ]  # "无法连接到大模型服务，请稍后再试"
        no_llm = True

    # PART04: TTS of LLM result
    if no_llm:
        tts_greeting(llm_response, use_bark=False, xtts_sr=24000)
    else:
        # Bark model not work well on long text prompts,
        # TODO solution: split long text into sentences:
        # see https://github.com/suno-ai/bark/blob/main/notebooks/long_form_generation.ipynb
        tts_greeting(llm_response, xtts_sr=24000)
