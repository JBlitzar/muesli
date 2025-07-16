"""
Transcription module for the Muesli application.

This module provides functionality for transcribing audio files and streams
using whisper.cpp. It handles both batch transcription of audio files and
real-time streaming transcription from microphone input.
"""

# Re-export key classes for easier imports
from muesli.transcription.whisper_wrapper import WhisperTranscriber
from muesli.transcription.stream_processor import TranscriptionStreamProcessor

__all__ = [
    "WhisperTranscriber",
    "TranscriptionStreamProcessor",
]
