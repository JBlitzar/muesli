"""
Muesli - Offline-first, privacy-centric voice transcription and summarization desktop application.

This package provides a desktop application for transcribing audio files and generating
summaries using local models, with a focus on privacy and offline operation.
"""

__version__ = "0.1.0"

# Import main components to make them available at package level
from main import MuesliApp, main
from models import (
    AudioFile,
    AudioFormat,
    Summary,
    SummaryType,
    Transcript,
    TranscriptSegment,
)
from ollama_client import OllamaClient
from stream_processor import TranscriptionStreamProcessor
from summarizer import TranscriptSummarizer
from whisper_wrapper import WhisperModelSize, WhisperTranscriber

# Export public API
__all__ = [
    # Main application
    "MuesliApp",
    "main",
    # Data models
    "AudioFile",
    "Transcript",
    "TranscriptSegment",
    "Summary",
    "SummaryType",
    "AudioFormat",
    # Components
    "WhisperTranscriber",
    "WhisperModelSize",
    "OllamaClient",
    "TranscriptSummarizer",
    "TranscriptionStreamProcessor",
]
