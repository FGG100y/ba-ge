#  import os

import openai

from audio import stt_model, tts_model
from nlp import llm_model

#  from wake_word.wake_detection import wakeBot
from wake_word.wake_gamgin_stream import wake_gamgin

GOODBYES = [
    "再见",
    "退下吧",
    "byebye",
]

init = 1
use_gtts = False
if init and not use_gtts:
    XTTS_MODEL, CONFIG = tts_model.load_xtts_model()

make_it_polite = False
use_faster_whisper = True
if use_faster_whisper:
    faster_whisper = stt_model.load_faster_whisper()

while True:
    # PART01: wake word detection
    if init and wake_gamgin():
        # responding the calling:
        greeting = "你好，金坚愿为您效劳"
        if use_gtts:
            tts_model.googletts_speaker(greeting, lang="zh-CN")
        else:
            tts_model.coquitts_speaker(
                model=XTTS_MODEL,
                config=CONFIG,
                intext=greeting,
                sr=24000,
                language="zh-cn",
                save_wav=False,
            )

        init = 0

    # PART02: speech2text (whisper-large), input and transcribe
    #  text = stt_model.transcribe(duration=3)
    #  breakpoint()

    text = stt_model.transcribe_fast(
        model=faster_whisper, duration=30, verbose=2
    )

    say_goodbye = [w for w in GOODBYES if w in text]
    if len(say_goodbye) > 0:
        if make_it_polite:
            tts_model.coquitts_speaker(
                model=XTTS_MODEL,
                config=CONFIG,
                intext="回聊，再见",
                sr=24000,
                language="zh-cn",
                save_wav=False,
            )
        break

    # PART03: query the LLMs
    try:
        #  query = "Hello, how are you today?" if text is None else text
        llm_response = llm_model.query_llm(query=text, verbose=True)
    except (ConnectionRefusedError, openai.APIConnectionError):
        llm_response = "无法连接到大模型服务，请稍后再试"

    # PART04: TTS using coqui-tts/xtts-v2
    tts_model.coquitts_speaker(
        model=XTTS_MODEL,
        config=CONFIG,
        intext=llm_response,
        sr=24000,
        language="zh-cn",
        save_wav=False,
    )
