import numpy
import time
import wave
import logging
import sys
import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice
from openwakeword.model import Model as WakeWordModel
from wake import wait_for_wake_word

# Needs to be 16khz for faster_whisper
SAMPLE_RATE = 16000
CHANNELS = 1
WHISPER_MODEL = "large-v3"
WAV_PATH = "test.wav"
KNOWN_COMMANDS = ["weather", "time"]

sys.stdout.reconfigure(encoding="utf-8")
logger = logging.getLogger()
logging.getLogger("piper").setLevel(logging.WARNING)

def main():
        
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler("log.txt")
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    try:
        whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        wake_word_model = WakeWordModel(inference_framework="onnx", wakeword_models=["hey_jarvis"])
        piper_voice = PiperVoice.load(r"voices\en_US-lessac-medium.onnx")
    except Exception as e:
        logger.critical(f"Failed to load models, exiting... \n{e}")
        quit()

    while True:
        try:
            wait_for_wake_word(wake_word_model)
            time.sleep(0.25)
            listen(whisper_model, piper_voice)
        except Exception as e:
            logger.error(f"Something went wrong, restarting... \n{e}")
    

def listen(model, piper_voice):
    logger.info("Recording...")

    recording = None
    while (recording is None):
        recording = record_until_silence()
    
    logger.info("Recording Stopped")

    segments, _ = model.transcribe(recording, beam_size=5, language="en")
    
    command = "".join(s.text for s in segments)
    command = command.lower()

    commands = [c.strip() for c in command.split(" and ")]
    commands = [c for c in commands if any(kc in c for kc in KNOWN_COMMANDS)]

    say(decide(commands), piper_voice)

def record_until_silence():
    CHUNK_SIZE = 512
    RMS_MIN = 0.075
    MAX_RECORD_SECS = 15
    MAX_RECORD_SAMPLES = MAX_RECORD_SECS * SAMPLE_RATE
    SILENCE_MIN = 1
    SILENT_CHUNKS_CUTOFF = (SILENCE_MIN * SAMPLE_RATE) / CHUNK_SIZE
    started_talking = False
    silent_chunk_count = 0
    chunks = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
        while True:
            chunk, _ = stream.read(CHUNK_SIZE)
            chunk = chunk.squeeze()
            chunks.append(chunk)

            rms = numpy.sqrt(numpy.mean(chunk ** 2))

            if rms >= RMS_MIN:
                started_talking = True
                silent_chunk_count = 0

            if started_talking and rms < RMS_MIN:
                silent_chunk_count += 1

            if silent_chunk_count >= SILENT_CHUNKS_CUTOFF:
                break

            if len(chunks) * CHUNK_SIZE >= MAX_RECORD_SAMPLES:
                break;

    recording = numpy.concatenate(chunks)
    return recording if started_talking else None 

def decide(commands):
    for c in commands:
        if "weather" in c:
            return "The weather is nice"
        elif "time" in c:
            return "It's the right time"
        
    return "I didn't get that, please repeat"

def say(text, voice):
    logger.debug(text)

    with wave.open(WAV_PATH, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    with wave.open(WAV_PATH, "rb") as wav_in:
        rate = wav_in.getframerate()
        n_frames = wav_in.getnframes()
        frames = wav_in.readframes(n_frames)

    audio = numpy.frombuffer(frames, dtype=numpy.int16)

    sd.play(audio, rate)
    sd.wait()

if __name__ == "__main__":
    main()