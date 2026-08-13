"""Offline text-to-speech, for people who'd rather listen than read.

Uses pyttsx3 (OS-native voices via SAPI5/NSSpeechSynthesizer/espeak) so it needs
no API key and no network call — it just has to work.
"""
from __future__ import annotations
import os
import tempfile


def synthesize_to_wav(text: str) -> bytes:
    import pyttsx3

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, path)
        engine.runAndWait()
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)
