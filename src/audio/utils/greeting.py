"""TTS speaking out loud
"""
from audio import tts_hfopt_bark, tts_elevenlabs, tts_model

XTTS_MODEL, CONFIG = tts_model.load_xtts_model()  # CPU is ok
#  #  use suno-bark from hf-transformers, No length limited (but not so good for zh)
#  hf_bark, bark_processor = tts_hfopt_bark.load_bark_model()



# TODO language auto detection
def tts_greeting(
    greeting,
    use_bark=False,
    use_hf_bark=True,
    use_11labs=False,  # backup
    use_gtts=False,  # backup
    xtts_sr=16000,
    language="zh-cn",
):
    if use_bark:
        if use_hf_bark:  # from transformers 4.31 onward
            tts_hfopt_bark.speaker(
                model=hf_bark,
                processor=bark_processor,
                text_prompt=greeting,
                voice_preset="v2/zh_speaker_1",
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
    else:  # only 82 chars for 'zh' at a time
        tts_model.coquitts_speaker(
            model=XTTS_MODEL,
            config=CONFIG,
            intext=greeting,
            sr=xtts_sr,
            language=language,
            save_wav=False,
        )
