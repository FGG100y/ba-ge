import queue

import numpy as np
import sounddevice as sd

# for sounddevice:
RATE = 16000
CHUNK = 512


class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate=RATE, chunk=CHUNK):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = sd.InputStream(
            callback=self._callback,
            channels=1,
            samplerate=self._rate,
            blocksize=self._chunk,
            dtype='int16',  # NOTE not work as my expectation
            #  frames_per_buffer=self._chunk,
        )
        self._audio_interface.start()
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_interface.stop()
        self.closed = True

    def _callback(self, in_data, frame_count, time_info, status):
        """This is called (from a separate thread) for each audio block."""
        self._buff.put(in_data)

    def generator_raw(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least some data,
            # otherwise the generator will sleep indefinitely if the queue is
            # empty.
            chunk = self._buff.get()
            yield chunk  # array of floats

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            chunk_int16_data = (
                np.iinfo(np.int16).max / np.amax(np.abs(chunk)) * chunk
            ).astype(np.int16).flatten()
            yield chunk_int16_data
