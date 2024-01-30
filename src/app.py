import openai
from audio import listen_microphone as stt_model
from audio import tts_model
from nlp import llm_model
from wake_word.wake_detection import wakeBot
from wake_word.wake_gamgin_stream import wake_gamgin

init = 1
while True:
    # PART01: wake word detection  # FIXME update using custom wake-word
    if (init and wake_gamgin()):
        # responding the calling:
        tts_model.speak_out_loud(
            itext="你好，金坚愿为您效劳",
            sr=24000,
            language="zh-cn",
            save_wav=False,
        )

    #  while True:
    #      # PART02: speech2text, input and transcribe
    #      text = stt_model.transcribe(duration=10)
    #      #  text = stt_model.transcribe(  # NOT work well
    #      #      chunk_length_s=5.0, stream_chunk_s=1.0, max_new_tokens=28
    #      #  )
    #      print("If Whisper thinks not correct, say 'backward' to do it again, say 'foreward' to proceed.")  # noqa
    #      if wakeBot().listening(debug=True) == "backward":
    #          continue
    #      print("If Whisper thinks correct, say 'forward' to proceed")
    #      if wakeBot(prob_threshold=0.8).listening() == "foreward":
    #          break
    text = stt_model.transcribe(duration=5)

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
    tts_model.speak_out_loud(
        itext="请问是否继续对话？",
        sr=24000,
        language="zh-cn",
        save_wav=False,
    )
    if wakeBot(prob_threshold=0.8).listening() == "yes":
        init = 0
    else:
        tts_model.speak_out_loud(
            itext="拜了个拜, 回见.",
            sr=24000,
            language="zh-cn",
            save_wav=False,
        )
        break
