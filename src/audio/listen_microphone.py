import pyttsx3
import speech_recognition as sr

#  import whisper


# Initialize the recognizer and text-to-speech engines
engine = pyttsx3.init()


# speech_recognition using `whisper` (which load .cache/whisper/model.pt)
def transcribe(language="chinese", duration=5):
    """fixed interval stt"""
    # obtain audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("I'm whisper, I'm listening, say something:")
        audio = r.listen(source, phrase_time_limit=duration)

    # recognize speech using whisper
    try:
        text = r.recognize_whisper(audio, model="large", language=language)
        print("Whisper thinks you said:", text)
    except sr.UnknownValueError:
        print("Whisper could not understand audio")
    except sr.RequestError as e:  # noqa
        print("Could not request results from Whisper")

    return text


def listen_and_respond(prompt, language="zh-cn"):
    """Listen for user input, transcribe it, and respond accordingly."""

    # Prepare the text-to-speech prompt
    engine.say(prompt)
    engine.runAndWait()

    r = sr.Recognizer()
    # Continuously listen for user input until they confirm or cancel
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = None

        while True:
            try:
                # Listen for user input and transcribe it
                audio = r.listen(source)
                text = r.recognize_whisper(
                    audio, model="large", language=language
                )
                break
            except sr.WaitTimeoutError:
                print("Timed out waiting for user input.")
            except sr.UnknownValueError:
                print("Unable to recognize speech.")
            except sr.RequestError as e:
                print(f"Speech recognition service error: {e}")

    # Check if the user confirmed or canceled the interaction
    if text.lower() in {"yes", "yeah", "sure", "ok", "okay"}:
        print("User confirmed.")
        return True
    elif text.lower() in {"no", "nope", "nevermind", "cancel", "quit"}:
        print("User canceled.")
        return False
    else:
        print(f"Unrecognized response: {text}")
        return


if __name__ == "__main__":
    # Example usage
    listen_and_respond("Are you ready to begin? Say yes or no.")

    #  # create a speech recognition object
    #  recognizer = sr.Recognizer()
    #
    #  with sr.Microphone() as source:
    #      print("Say something!")
    #      # read the audio data from the default microphone
    #      audio_data = recognizer.record(source, duration=5)
