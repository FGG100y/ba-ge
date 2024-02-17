"""Text-To-Speech

- elevenlabs tts

NOTE 由于依赖包处于开发状态的缘故，此模块的函数可能随时会过时:

`pip install git+https://github.com/elevenlabs/elevenlabs-python.git@v3`

elevenlabs==0.3.0b0

"""

import os
from dotenv import load_dotenv
#  from elevenlabs import voices, generate, play
from elevenlabs import generate, play
from elevenlabs.client import ElevenLabs


load_dotenv()
api_key = os.getenv("elevenlabs_api_key")
client = ElevenLabs(api_key=api_key)


# elevenlabs 0.3.a
def lllabs_speaker(intext):
    audio = generate(
        text=intext,
        voice="Alice",
        model="eleven_turbo_v2",
    )

    play(audio)


if __name__ == "__main__":
    #  intext = "Hello! 你好! Hola! नमस्ते! Bonjour! こ Привіт!"
    #  lllabs_speaker(intext)

    all_voices = client.voices.get_all()
    all_voices_d = all_voices.dict()['voices']

    v_ids = [v.get("voice_id", "no_id") for v in all_voices_d]
    names = [v.get("name", "no_name") for v in all_voices_d]
    labels_d = [v.get("labels", "no_labels") for v in all_voices_d]
    infos = [
        (
            v_id,
            name,
            label.get("gender", "no-gender"),
            label.get("accent", "no-accent"),
            label.get("use case", "no-use_case"),
        )
        for v_id, name, label in zip(v_ids, names, labels_d)
    ]
    print(infos)

    # VoiceResponse(voice_id='Xb7hH8MSUJpSbSDYk0k2', name='Alice',
    # category='premade', labels={'accent': 'british', 'description':
    # 'confident', 'age': 'middle aged', 'gender': 'female', 'featured': 'new',
    # 'use case': 'news'}, available_for_tiers=[],
    # high_quality_base_model_ids=['eleven_turbo_v2'], samples=None,
    # fine_tuning=FineTuningResponseModel(is_allowed_to_fine_tune=True,
    # fine_tuning_requested=None, finetuning_state=<FinetunigState.FINE_TUNED:
    # 'fine_tuned'>, verification_failures=[], verification_attempts_count=0,
    # manual_verification_requested=False, verification_attempts=None,
    # slice_ids=None, manual_verification=None, language='en')
