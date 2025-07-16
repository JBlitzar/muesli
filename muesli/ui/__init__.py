"""
UI module for the Muesli application.

This module provides the user interface components for the application,
built with PySide6 and QML. It includes the main application window,
transcript display, and controls for starting and stopping transcription.

The UI is designed to be simple and intuitive, with a focus on displaying
transcription results in real-time and providing basic controls for the
transcription process.
"""

# Re-export key classes for easier imports
from muesli.ui.main_window import MainWindow

__all__ = [
    "MainWindow",
]
