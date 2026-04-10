import numpy
import time
import wave
import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice
from openwakeword.model import Model as WakeWordModel
from wake import wait_for_wake_word

# Needs to be 16khz for faster_whisper
SAMPLE_RATE = 16000
DURATION_SEC = 5
CHANNELS = 1
WHISPER_MODEL = "large-v3"
WAV_PATH = "test.wav"
KNOWN_COMMANDS = ["weather", "time"]

def main():
    whisperModel = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    wakeWordModel = WakeWordModel(inference_framework="onnx", wakeword_models=["hey_jarvis"])
    piperVoice = PiperVoice.load(r"voices\en_US-lessac-medium.onnx")

    while True:
        wait_for_wake_word(wakeWordModel)
        time.sleep(0.25)
        listen(whisperModel, piperVoice)
    

def listen(model, piperVoice):
    print("Recording for {} seconds".format(DURATION_SEC))
    recording = sd.rec(
        frames=int(DURATION_SEC * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32"
    )
    sd.wait()

    print("Recording Stopped")

    recording = recording.squeeze()

    segments, _ = model.transcribe(recording, beam_size=5, language="en")

    {
        # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        # for segment in segments:
        #     print("[{} -> {}] {}".format(segment.start, segment.end, segment.text))
    }

    command = "".join(s.text for s in segments)
    command = command.lower()

    commands = [c.strip() for c in command.split(" and ")]
    commands = [c for c in commands if any(kc in c for kc in KNOWN_COMMANDS)]

    say(decide(commands), piperVoice)

def decide(commands):
    for c in commands:
        if "weather" in c:
            return "The weather is nice"
        elif "time" in c:
            return "It's the right time"
        
    return "I didn't get that, please repeat"

def say(text, voice):
    print(text)

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