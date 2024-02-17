import os

import pvporcupine
from dotenv import load_dotenv

from wake_word.set_mic import MicrophoneStream

#  import pyaudio
#  import sounddevice as sd
#  from porcupine import Porcupine


load_dotenv()
# for porcupine:
access_key = os.getenv("porcupine_key")
KEYWORD_PATH = "models/porcupine-wakeword-model/gamgin_zh_linux.ppn"
MODEL_PATH = "models/porcupine-wakeword-model/porcupine_params_zh.pv"
SENSITIVITY = 0.5

# Create a Porcupine object and start listening for keywords
porcupine = pvporcupine.create(
    access_key=access_key,
    keyword_paths=[KEYWORD_PATH],
    model_path=MODEL_PATH,
    sensitivities=[SENSITIVITY],
)


def wake_gamgin():
    with MicrophoneStream() as stream:
        audio_generator = stream.generator()
        while True:
            frame_data = next(audio_generator)
            keyword_index = porcupine.process(frame_data)
            if keyword_index >= 0:
                wake_word_detected = "\n-*-金坚-*- 听到您的呼唤了:)"
                print(wake_word_detected)
                return True
            else:
                standy = " 🐶金坚🐶  等待呼唤 📞 ...."
                print(f"\r{standy}", end="")
                #  # alternative:
                #  sys.stdout.write("\r" + standy)
                #  sys.stdout.flush()


if __name__ == "__main__":
    wake_gamgin()
