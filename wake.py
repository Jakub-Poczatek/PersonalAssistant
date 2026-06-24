import numpy as numpy
import sounddevice as sd
import time as time
import logging
from openwakeword.model import Model

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
FLUSH_SIZE = 32000
THRESHOLD = 0.75

logger = logging.getLogger(__name__)

def wait_for_wake_word(model):
    logger.info("Listening for wake word... Press Ctrl+C to stop.")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
        while True:
            chunk, _ = stream.read(CHUNK_SIZE)

            # (CHUNK_SIZE, 1) -> (CHUNK_SIZE,)
            chunk = chunk.squeeze()

            # openWakeWord expects int16 PCM at 16kHz
            predictions = model.predict(chunk)

            #predictions is usually a dict: {model_name: score}
            for name, score in predictions.items():
                if score >= THRESHOLD:
                    logger.debug(f"Wake word detected: {name}, with score {score:.3f}")
                    _ = model.predict(numpy.zeros(FLUSH_SIZE, dtype=numpy.int16))
                    return True

def main():
    model = Model(inference_framework="onnx", wakeword_models=["hey_jarvis"])
    logging.basicConfig(level=logging.DEBUG)
    while (True):
        wait_for_wake_word(model)
        time.sleep(1)

if __name__ == "__main__":
    main()