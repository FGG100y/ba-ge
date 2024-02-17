import openai

from audio import stt_model, tts_model, tts_elevenlabs
from nlp import llm_model

from wake_word.wake_gamgin_stream import wake_gamgin

GOODBYES = [
    # 都是朋友
    "再见",
    "再見",
    "先这样",
    "byebye",
    # 惹人生气
    "滚蛋",
    "跪安吧",
    "退下吧",
]

init = 1
wake_again = False

make_bot_polite = True
XTTS_MODEL, CONFIG = tts_model.load_xtts_model()

#  use_faster_whisper = True
faster_whisper = stt_model.load_faster_whisper()


def tts_greeting(greeting, use_11labs=False, use_gtts=False, xtts_sr=16000):
    if use_11labs:
        tts_elevenlabs.lllabs_speaker(
            intext=greeting, voice="Rachel", model="eleven_multilingual_v2"
        )
    elif use_gtts:  # FIXME gtts not work (mainly) due to network issues
        tts_model.googletts_speaker(greeting, lang="zh-CN")
    else:
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
        hello_again = "我在呢，请问有什么可以帮忙的？"
        tts_greeting(hello_again, xtts_sr=24000)

    if init and wake_gamgin():
        # responding the calling:
        hello = "盆友，你好"
        tts_greeting(hello, xtts_sr=24000)
        init = 0

    # PART02: speech2text (faster-whisper-large), input and transcribe
    speech2text = stt_model.transcribe_fast(
        model=faster_whisper, duration=50, verbose=1
    )

    say_goodbye = [w for w in GOODBYES if w in speech2text]
    if len(say_goodbye) > 0:
        if make_bot_polite:
            intext = "盆友再见, 下次聊"
            tts_greeting(intext, xtts_sr=24000)

        #  break  # 结束程序

        # 如果用户说再见，则进入等待唤醒状态
        if wake_gamgin():
            wake_again = True
            continue

    # PART03: query the LLMs
    try:
        # (FIXME later): limit response lenght < 72 chars (coqui-tts limit):
        prompt = speech2text + " 请简答，并且字数少于72个。"
        llm_response = llm_model.query_llm(query=prompt, verbose=True)
        no_llm = False
    except (ConnectionRefusedError, openai.APIConnectionError):
        llm_response = "无法连接到大模型服务，请稍后再试"
        no_llm = True

    # PART04: TTS of LLM result
    if no_llm:
        tts_greeting(llm_response)
    else:
        tts_greeting(llm_response, use_11labs=False, xtts_sr=24000)
