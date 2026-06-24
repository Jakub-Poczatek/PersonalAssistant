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

#skills
import skills.weather as skill_weather
import skills.time as skill_time

# Needs to be 16khz for faster_whisper
SAMPLE_RATE = 16000
CHANNELS = 1
WHISPER_MODEL = "large-v3"
WAV_PATH = "test.wav"
KNOWN_COMMANDS = ["weather", "time", "date"]

def default_decision():
    return "I didn't get that, please repeat"

skill_reg = {
    "default": default_decision, 
    "weather": skill_weather.get_weather, 
    "time": skill_time.get_time,
    "date": skill_time.get_date
}

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
            command_recognised = False
            while not command_recognised:
                command_recognised = listen(whisper_model, piper_voice)
        except Exception as e:
            logger.error(f"Something went wrong, restarting... \n{e}")
        except KeyboardInterrupt:
            logger.info(f"Shutting down...")
            quit()
    

def listen(model, piper_voice):
    recording = None
    while (recording is None):
        logger.info("Recording...")
        recording = record_until_silence()
    
    logger.info("Recording Stopped")

    segments, _ = model.transcribe(recording, beam_size=5, language="en")
    
    command = "".join(s.text for s in segments)
    command = command.lower()

    commands = [c.strip() for c in command.split(" and ")]
    commands = [c for c in commands if any(kc in c for kc in KNOWN_COMMANDS)]

    decision = decide(commands)
    
    say(skill_reg[decision](), piper_voice)

    if decision == "default":
        return False
    
    return True

def record_until_silence():
    CHUNK_SIZE = 512
    RMS_MIN = 0.05
    MAX_RECORD_SECS = 5
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
            return "weather"
        elif "time" in c:
            return "time"
        elif "date" in c:
            return "date"
        
    return "default"

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