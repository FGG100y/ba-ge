import torch
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

device = "cuda:0" if torch.cuda.is_available() else "cpu"
localmodel = "/home/ds01/hfLLMs/ast-finetuned-speech-commands-v2"

classifier = pipeline("audio-classification", model=localmodel, device=device)


class wakeBot:
    def __init__(self, prob_threshold=0.5):
        #  self.wake_word = wake_word
        self.chunk_length_s = 2.0
        self.stream_chunk_s = 0.25
        self.prob_threshold = prob_threshold
        self.sampling_rate = classifier.feature_extractor.sampling_rate

    def listening(self, debug=False):
        mic = ffmpeg_microphone_live(
            sampling_rate=self.sampling_rate,
            chunk_length_s=self.chunk_length_s,
            stream_chunk_s=self.stream_chunk_s,
        )

        print("Listening for wake word...")
        for prediction in classifier(mic):
            prediction = prediction[0]
            if debug:
                print(prediction)
            if prediction["score"] > self.prob_threshold:
                return prediction["label"]

    def launch_fn(self, wake_word, debug=False):
        if wake_word not in classifier.model.config.label2id.keys():
            if debug:
                print(classifier.model.config.id2label)

            raise ValueError(
                f"Wake word '{wake_word}' not in set of valid class labels, pick a wake word in the set {classifier.model.config.label2id.keys()}."  # noqa
            )

        mic = ffmpeg_microphone_live(
            sampling_rate=self.sampling_rate,
            chunk_length_s=self.chunk_length_s,
            stream_chunk_s=self.stream_chunk_s,
        )

        print("Listening for wake word...")
        for prediction in classifier(mic):
            prediction = prediction[0]
            if debug:
                print(prediction)
            if prediction["label"] == wake_word:
                if prediction["score"] > self.prob_threshold:
                    return True


if __name__ == "__main__":
    bot = wakeBot(prob_threshold=0.8)
    #  bot.launch_fn(wake_word="awelw")  # list wake-words
    # ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
    # 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
    # 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila',
    # 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes',
    # 'zero']
    if bot.listening(debug=True) == "backward":
        print("Yes, I will follow the instruction.")
    else:
        print("you are too late")
    #  elif bot.listening(debug=True) == "backward":
    #      print("you are too late")
