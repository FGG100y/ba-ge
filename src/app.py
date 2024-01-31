import openai
from audio import listen_microphone as stt_model
from audio import tts_model
from nlp import llm_model
#  from wake_word.wake_detection import wakeBot
from wake_word.wake_gamgin_stream import wake_gamgin

init = 1
goodbyes = ["bye", "再见", ]
while True:
    # PART01: wake word detection
    if (init and wake_gamgin()):
        # responding the calling:
        tts_model.speak_out_loud(
            itext="你好，金坚愿为您效劳",
            sr=24000,
            language="zh-cn",
            save_wav=False,
        )
        init = 0

    # PART02: speech2text (whisper-large), input and transcribe
    # FIXME: 每次都加载一次模型，太慢了
    text = stt_model.transcribe(duration=3)
    say_goodbye = [w for w in goodbyes if w in text]
    if len(say_goodbye) > 0:
        tts_model.speak_out_loud(
            itext="拜了个拜, 回见.",
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
    tts_model.speak_out_loud(
        itext=llm_response,
        sr=24000,
        language="zh-cn",
        save_wav=False,
    )
