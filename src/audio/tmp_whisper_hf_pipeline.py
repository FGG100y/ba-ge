import torch
import torchaudio
from transformers import pipeline
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# initialize the pipeline (huggingface model)
pipe = pipeline(
    "automatic-speech-recognition", model="", device=device
)


def get_long_transcription_whisper(
    audio_path,
    pipe,
    return_timestamps=True,
    chunk_length_s=10,
    stride_length_s=2,
):
    """Get the transcription of a long audio file using the Whisper model"""
    return pipe(
        load_audio(audio_path).numpy(),
        return_timestamps=return_timestamps,
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
    )


def load_audio(audio_path):
    """Load the audio file & convert to 16,000 sampling rate"""
    # load our wav file
    speech, sr = torchaudio.load(audio_path)
    resampler = torchaudio.transforms.Resample(sr, 16000)
    speech = resampler(speech)
    return speech.squeeze()


# get the transcription of a sample long audio file
output = get_long_transcription_whisper(
    "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0060_8kwav",
    pipe,
    chunk_length_s=10,
    stride_length_s=1,
)

for chunk in output["chunks"]:
    # print the timestamp and the text
    print(chunk["timestamp"], ":", chunk["text"])
