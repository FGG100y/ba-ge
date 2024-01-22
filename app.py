from src import listen_microphone as stt_model
from src.wake_detection import wakeBot
from src import llm_model, tts_model

init = 1
while True:
    # PART01: wake word detection
    if (init and wakeBot().launch_fn(wake_word="marvin", debug=True)):
        # responding the calling:
        tts_model.speak_out_loud(
            itext="I'm here at your service.",
            sr=24000,
            language="en",
            save_wav=False,
        )

    #  while True:
    #      # PART02: speek2text, input and transcribe
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
    text = stt_model.transcribe(duration=10)

    # PART03: query the LLMs
    #  query = "Hello, how are you today?" if text is None else text
    llm_response = llm_model.query_llm(query=text, verbose=True)

    # PART04: TTS using coqui-tts/xtts-v2
    tts_model.speak_out_loud(
        itext=llm_response,
        sr=24000,
        language="zh-cn",
        save_wav=False,
    )
    tts_model.speak_out_loud(
        itext="Would you like to go on?",
        sr=24000,
        language="en",
        save_wav=False,
    )
    if wakeBot().listening() == "yes":
        init = 0
    else:
        tts_model.speak_out_loud(
            itext="Okay, see you.",
            sr=24000,
            language="en",
            save_wav=False,
        )
        break
