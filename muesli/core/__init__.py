"""
Core module for the Muesli application.

This module serves as the central orchestrator for the application, providing:
- Application initialization and lifecycle management
- Configuration handling and validation
- Data models for transcripts and metadata
  (Previously included a background job-queue, now removed for simplicity)

All other modules (transcription, llm, ui) communicate with the application
through the interfaces provided by this module.
"""

# Re-export key classes for easier imports
from muesli.core.app import MuesliApp
from muesli.core.config import AppConfig, load_config
from muesli.core.models import Transcript, AudioFile, Summary

__all__ = [
    "MuesliApp",
    "AppConfig", 
    "load_config",
    "Transcript", 
    "AudioFile", 
    "Summary",
]
