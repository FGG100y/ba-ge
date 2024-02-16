#  import os

import openai

from audio import stt_model, tts_model, tts_elevenlabs
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
use_11labs = True
if init and not use_gtts and not use_11labs:
    XTTS_MODEL, CONFIG = tts_model.load_xtts_model()

make_it_polite = True
use_faster_whisper = True
if use_faster_whisper:
    faster_whisper = stt_model.load_faster_whisper()

while True:
    # PART01: wake word detection
    if init and wake_gamgin():
        # responding the calling:
        greeting = "你好，金坚愿为您效劳"
        if use_11labs:
            tts_elevenlabs.tts_11labs(intext=greeting)
        elif use_gtts:
            tts_model.googletts_speaker(greeting, lang="zh-CN")
        else:
            tts_model.coquitts_speaker(
                model=XTTS_MODEL,
                config=CONFIG,
                intext=greeting,
                sr=16000,
                language="zh-cn",
                save_wav=False,
            )

        init = 0

    # PART02: speech2text (whisper-large), input and transcribe
    #  speech2text = stt_model.transcribe(duration=3)
    #  breakpoint()

    speech2text = stt_model.transcribe_fast(
        model=faster_whisper, duration=50, verbose=1
    )

    say_goodbye = [w for w in GOODBYES if w in speech2text]
    if len(say_goodbye) > 0:
        if make_it_polite:
            intext = "那就回聊啦，盆友再见",
            if use_11labs:
                tts_elevenlabs.tts_11labs(intext=intext)
            else:
                tts_model.coquitts_speaker(
                    model=XTTS_MODEL,
                    config=CONFIG,
                    intext="回聊，再见",
                    sr=16000,
                    language="zh-cn",
                    save_wav=False,
                )
        break

    # PART03: query the LLMs
    try:
        # (FIXME later): limit response lenght < 72 chars (coqui-tts limit):
        if use_11labs:
            prompt = speech2text
        else:
            prompt = speech2text + " 请简答，并且字数少于72个。"
        llm_response = llm_model.query_llm(query=prompt, verbose=True)
    except (ConnectionRefusedError, openai.APIConnectionError):
        llm_response = "无法连接到大模型服务，请稍后再试"
    # PART04: TTS of LLM result
    if use_11labs:
        tts_elevenlabs.tts_11labs(intext=llm_response)
    else:
        tts_model.coquitts_speaker(
            model=XTTS_MODEL,
            config=CONFIG,
            intext=llm_response,
            sr=16000,
            language="zh-cn",
            save_wav=False,
        )
