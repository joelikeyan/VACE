"""Simple any-to-any agent with voice *or* text input.

The script can capture audio from the microphone or accept typed text,
send the prompt to an OpenAI language model, and return both a textual
and vocal reply. The assistant's text is printed to the terminal and the
spoken audio is saved to a temporary WAV file while also being played
back through the speakers.

Usage examples::

    # Record five seconds of audio and respond with speech + text
    python any2any_voice_agent.py

    # Type your message instead of recording audio
    python any2any_voice_agent.py --mode text --text "How's the weather?"

Environment variables ``OPENAI_API_KEY`` and ``HF_API_KEY`` must be set with valid
API keys.
"""

from __future__ import annotations

import argparse
import io
import tempfile
import wave

import os

import numpy as np
import sounddevice as sd
from huggingface_hub import InferenceClient
from openai import OpenAI


client = OpenAI()
hf_client = InferenceClient(
    model="cognitivecomputations/dolphin-2.9.2-qwen2-7b",
    token=os.environ.get("HF_API_KEY"),
)


def record_audio(seconds: int = 5, samplerate: int = 44_100) -> str:
    """Record ``seconds`` of mono audio and return path to a WAV file."""
    print("Recording...")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1)
    sd.wait()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_file.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(samplerate)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    return temp_file.name


def transcribe_audio(path: str) -> str:
    """Transcribe the WAV file located at ``path`` using OpenAI."""
    with open(path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
        )
    return transcription.text


def generate_response(prompt: str) -> str:
    """Generate an assistant reply for ``prompt`` using Hugging Face Inference."""
    if hf_client.token is None:
        raise RuntimeError(
            "HF_API_KEY environment variable is required for dolphin-2.9.2-qwen2-7b."
        )
    completion = hf_client.text_generation(
        prompt,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        return_full_text=False,
    )
    return completion.strip()


def synthesize_speech(text: str) -> tuple[str, np.ndarray, int]:
    """Create speech audio for ``text`` and return (path, data, rate)."""
    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        format="wav",
    )

    audio_bytes = speech.audio
    buffer = io.BytesIO(audio_bytes)
    with wave.open(buffer, "rb") as wav_reader:
        framerate = wav_reader.getframerate()
        frames = wav_reader.readframes(wav_reader.getnframes())
    audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    speech_path = tempfile.mkstemp(suffix=".wav")[1]
    with open(speech_path, "wb") as f:
        f.write(audio_bytes)

    return speech_path, audio_np, framerate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Any-to-any voice/text agent")
    parser.add_argument(
        "--mode",
        choices=("voice", "text"),
        default="voice",
        help="Input mode: record from microphone or use typed text.",
    )
    parser.add_argument(
        "--text",
        help="Text to send when --mode text is selected. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Length of the microphone recording in seconds (voice mode only).",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=44_100,
        help="Sample rate for microphone recording (voice mode only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "voice":
        audio_path = record_audio(seconds=int(args.seconds), samplerate=args.samplerate)
        user_text = transcribe_audio(audio_path)
    else:
        if args.text:
            user_text = args.text
        else:
            user_text = input("Type your message: ").strip()
        if not user_text:
            print("No text provided. Exiting.")
            return

    print(f"User: {user_text}")

    reply_text = generate_response(user_text)
    print(f"Assistant: {reply_text}")

    speech_path, audio_np, samplerate = synthesize_speech(reply_text)
    print(f"Vocal response saved to {speech_path}")

    sd.play(audio_np, samplerate)
    sd.wait()


if __name__ == "__main__":
    main()

