import wave
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from piper import PiperVoice

# Needs to be 16khz for faster_whisper
SAMPLE_RATE = 16000
DURATION_SEC = 5
CHANNELS = 1
WHISPER_MODEL = "large-v3"
WAV_PATH = "test.wav"

def main():
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    piperVoice = PiperVoice.load("voices\en_US-lessac-medium.onnx")

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

    # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    # for segment in segments:
    #     print("[{} -> {}] {}".format(segment.start, segment.end, segment.text))

    command = "".join(s.text for s in segments)
    command = command.lower()

    commands = [c.strip() for c in command.split(" and ")]

    for c in commands:
        if c.find("weather") != -1:
            say("The weather is nice", piperVoice)
        elif c.find("time") != -1:
            say("It's the right time", piperVoice)
        else:
            say("I didn't get that, please repeat", piperVoice)


def say(text, voice):
    print(text)

    with wave.open(WAV_PATH, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    with wave.open(WAV_PATH, "rb") as wav_in:
        rate = wav_in.getframerate()
        n_frames = wav_in.getnframes()
        frames = wav_in.readframes(n_frames)

    audio = np.frombuffer(frames, dtype=np.int16)

    sd.play(audio, rate)
    sd.wait()

if __name__ == "__main__":
    main()