import torch
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
#  from transformers import Wav2Vec2ForPreTraining



device = "cuda:0" if torch.cuda.is_available() else "cpu"
localmodel = "/home/ds01/hfLLMs/ast-finetuned-speech-commands-v2"

classifier = pipeline("audio-classification", model=localmodel, device=device)



def tmpfunc():
    # Initialize the ffmpeg_microphone_live function with the desired parameters
    input_stream = pipeline.ffmpeg_microphone_live(device_index=0, sample_rate=16000, num_channels=1)

    # Set the batch size to 4
    batch_size = 4

    while True:
        # Read chunks of audio input from the microphone and store them in a list
        audio_chunks = []
        for i in range(batch_size):
            chunk = next(input_stream).numpy()
            audio_chunks.append(chunk)

        # Concatenate the chunks into a single array
        audio_array = np.concatenate(audio_chunks)

        # Process the audio array using the pipeline
        result = pipeline(audio_array)[0]

        # Do something with the result, such as print it to the console
        print(result)



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
    wake_word = "backward"
    if bot.listening(debug=True) == wake_word:
        print("Yes, I will follow the instruction.")
    else:
        print("you are too late")
    #  elif bot.listening(debug=True) == "backward":
    #      print("you are too late")
