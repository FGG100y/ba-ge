import sys

import torch
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

device = "cuda:0" if torch.cuda.is_available() else "cpu"

localmodel = "/home/ds01/hfLLMs/whisper-large-v3"
transcriber = pipeline(
    "automatic-speech-recognition", model=localmodel, device=device
)


def transcribe(chunk_length_s=5.0, stream_chunk_s=1.0, max_new_tokens=128):
    """Speak to text using pretrained model [default: whisper-large-v3]

    Args:

    chunk_length_s: the maximum audio length (increase it if speak slowly)

    stream_chunk_s: the real-time factor
    """
    sampling_rate = transcriber.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Start speaking...")
    for item in transcriber(
        mic, generate_kwargs={"max_new_tokens": max_new_tokens}
    ):
        sys.stdout.write("\033[K")
        print(item["text"], end="\r")
        if not item["partial"][0]:
            break
    print("Transcribe to:", item["text"])

    return item["text"]


if __name__ == "__main__":
    transcribe(chunk_length_s=3, max_new_tokens=28)
