"""
Main window UI for the Muesli application.

This module provides the main window UI for the Muesli application,
including controls for transcription, summarization, and displaying results.
"""

import logging
import math
import os
import re
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QDateTime, QRunnable, Qt, QThreadPool, QTimer, QUrl, Slot
from PySide6.QtGui import QAction, QFont, QFontDatabase, QIcon, QTextCursor, QTextOption

# NOTE: QML engine kept for future; spinner now text-based so we drop SVG deps
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from models import AudioFile, Summary, Transcript

logger = logging.getLogger(__name__)

# Color scheme constants
ACCENT_COLOR = "#e8e8e8"  # (232, 232, 232) - Main accent color
BACKGROUND_COLOR = "#fcfbf6"  # Main background
BORDER_COLOR = "#e3e6d6"  # Border color
TEXT_COLOR = "#333333"  # Primary text color
SECONDARY_TEXT_COLOR = "#7f8c8d"  # Secondary text color
SCROLL_BAR_BACKGROUND = "#ebeced"  # Scroll bar track background
BUTTON_HOVER_COLOR = "#ebeced"  # Button hover background color


class MainWindow(QMainWindow):
    """
    Main window for the Muesli application.

    This class provides the main UI for the application, including controls
    for transcription, summarization, and displaying results.
    """

    def __init__(self, app):
        """
        Initialize the main window.

        Args:
            app: MuesliApp instance
        """
        super().__init__()

        # Store reference to the application
        self.app = app

        # Recording state flag
        self._recording = False
        # Thread-pool for background work
        self._thread_pool: QThreadPool = QThreadPool.globalInstance()

        # Set window properties
        self.setWindowTitle(f"{app.config.app_name} v{app.config.app_version}")
        self.resize(1024, 768)

        # Set window background color
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background-color: {BACKGROUND_COLOR};
            }}
        """
        )

        # Active transcripts and summaries
        self._active_transcript_id = None
        self._active_summary_id = None

        # Set up UI components
        self._setup_ui()

        # Connect signals from application
        self._connect_signals()

        # Ensure graceful cleanup when the Qt application is about to quit
        if getattr(self.app, "_app", None):  # QApplication instance exists
            self.app._app.aboutToQuit.connect(self._on_app_quit)

        logger.info("Main window initialized")

    def _setup_ui(self):
        """Set up the UI components."""
        # Create central widget
        central_widget = QWidget()
        central_widget.setStyleSheet(
            f"""
            QWidget {{
                background-color: {BACKGROUND_COLOR};
            }}
        """
        )
        self.setCentralWidget(central_widget)

        # Create main layout with padding
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(
            20, 20, 20, 20
        )  # Add padding around main content
        main_layout.setSpacing(16)  # Add spacing between elements

        # Create menu bar
        self._create_menu_bar()

        # Create toolbar
        self._create_toolbar()

        # ----- Combined transcript & summary area -----
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(
            20, 0, 20, 0
        )  # Add horizontal padding to header

        header_label = QLabel("Summary & Transcript")
        header_font = QFont("Geist", 14, QFont.Bold)
        header_font.setStyleHint(QFont.SansSerif)
        header_label.setFont(header_font)
        header_label.setStyleSheet(f"color: {TEXT_COLOR};")
        header_layout.addWidget(header_label)

        self.transcript_status = QLabel("No transcript loaded")
        self.transcript_status.setStyleSheet(f"color: {SECONDARY_TEXT_COLOR};")
        header_layout.addWidget(self.transcript_status)

        self.summary_status = QLabel("No summary generated")
        self.summary_status.setStyleSheet(f"color: {SECONDARY_TEXT_COLOR};")
        header_layout.addWidget(self.summary_status)
        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        # Set up the combined text area with improved font and styling
        self.combined_text = QTextEdit()
        self.combined_text.setReadOnly(True)

        # Use a nice sans-serif font with larger size
        content_font = QFont("Geist", 18)
        content_font.setStyleHint(
            QFont.SansSerif
        )  # Fallback to system sans-serif if Inter not available
        self.combined_text.setFont(content_font)

        # Set line width for word wrap (65 characters)
        self.combined_text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.combined_text.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)

        # Add styling with padding, rounded corners, and background
        self.combined_text.setStyleSheet(
            f"""
            QTextEdit {{
                background-color: {BACKGROUND_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 12px;
                padding: 24px;
                color: {TEXT_COLOR};
                selection-background-color: #3498db;
                selection-color: white;
            }}
            QScrollBar:vertical {{
                background-color: {SCROLL_BAR_BACKGROUND};
                border: none;
                border-radius: 6px;
                width: 12px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background-color: {ACCENT_COLOR};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: #c7c2b4;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
        """
        )

        self.combined_text.setPlaceholderText(
            "Summary (Markdown) and transcript will appear here"
        )

        # Create a container widget for the text area to add outer padding/styling
        text_container = QWidget()
        text_container.setStyleSheet(
            f"""
            QWidget {{
                background-color: {BACKGROUND_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 12px;
            }}
        """
        )

        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(2, 2, 2, 2)  # Small margin for the border
        text_layout.addWidget(self.combined_text)

        main_layout.addWidget(text_container)

        # ----- Notes input area -----
        self.notes_container = QWidget()
        self.notes_container.setStyleSheet(
            f"""
            QWidget {{
                background-color: {BACKGROUND_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 12px;
            }}
        """
        )

        notes_layout = QVBoxLayout(self.notes_container)
        notes_layout.setContentsMargins(2, 2, 2, 2)

        # Notes header
        notes_header = QLabel("Notes (Available during recording)")
        notes_header.setFont(QFont("Geist", 14, QFont.Bold))
        notes_header.setStyleSheet(f"color: {TEXT_COLOR}; padding: 8px;")
        notes_layout.addWidget(notes_header)

        # Notes text area
        self.notes_text = QTextEdit()
        self.notes_text.setFont(QFont("Geist", 16))
        self.notes_text.setMaximumHeight(150)  # Limit height to keep it compact
        self.notes_text.setStyleSheet(
            f"""
            QTextEdit {{
                background-color: {BACKGROUND_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 12px;
                color: {TEXT_COLOR};
                selection-background-color: #3498db;
                selection-color: white;
            }}
            QScrollBar:vertical {{
                background-color: {SCROLL_BAR_BACKGROUND};
                border: none;
                border-radius: 6px;
                width: 12px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background-color: {ACCENT_COLOR};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: #c7c2b4;
            }}
        """
        )
        self.notes_text.setPlaceholderText("Type your notes here during recording...")
        self.notes_text.setEnabled(False)  # Disabled by default
        notes_layout.addWidget(self.notes_text)

        # Connect notes text changes to update transcript
        self.notes_text.textChanged.connect(self._on_notes_changed)

        main_layout.addWidget(self.notes_container)

        # Holds latest markdown so we can save exact content
        self._combined_markdown: str = ""

        # Create status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(
            f"""
            QStatusBar {{
                background-color: {ACCENT_COLOR};
                color: white;
                border: none;
                padding: 12px 16px;
                margin: 0px;
                min-height: 20px;
            }}
            QStatusBar QLabel {{
                color: black;
                padding-left: 8px;
                background: transparent;
            }}
            QStatusBar QProgressBar {{
                background-color: rgba(211, 219, 230, 0.3);
                border: none;
                height: 20px;
            }}
        """
        )
        self.setStatusBar(self.status_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(
            "color: black; font-family: 'Geist', sans-serif;"
        )
        self.status_bar.addWidget(self.status_label, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                background-color: rgba(211, 219, 230, 0.3);
                height: 20px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 10px;
            }
        """
        )
        self.status_bar.addWidget(self.progress_bar)

        # ------------------------------------------------------------------#
        # Time-based progress estimation                                    #
        # ------------------------------------------------------------------#
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(200)  # update ~5× per second
        self._progress_timer.timeout.connect(self._on_progress_tick)
        self._progress_total_secs: float = 0.0
        self._progress_elapsed_secs: float = 0.0
        self._progress_start_time = None

        # ------------------------------------------------------------------#
        # Text-based loading indicator (ASCII spinner)                      #
        # ------------------------------------------------------------------#
        self._spinner_chars = list("⣾⣽⣻⢿⡿⣟⣯⣷")[::-1]
        self._spinner_idx = 0
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(150)
        self._spinner_timer.timeout.connect(self._update_spinner)

        self.loading_label = QLabel("Loading " + self._spinner_chars[0])
        loading_font = QFont("Geist", 12, QFont.Bold)
        loading_font.setStyleHint(QFont.SansSerif)
        self.loading_label.setFont(loading_font)
        self.loading_label.setStyleSheet(
            """
            QLabel {
                color: black;
                background-color: rgba(211, 219, 230, 0.2);
                border-radius: 0px;
                padding: 4px 8px;
                margin: 2px;
            }
        """
        )
        self.loading_label.setVisible(False)
        self.status_bar.addPermanentWidget(self.loading_label)

    def _create_menu_bar(self):
        """Create the menu bar."""
        menu_bar = self.menuBar()
        menu_bar.setStyleSheet(
            f"""
            QMenuBar {{
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
                border: none;
                padding: 4px;
            }}
            QMenuBar::item {{
                background-color: transparent;
                padding: 8px 12px;
            }}
            QMenuBar::item:selected {{
                background-color: {BORDER_COLOR};
            }}
            QMenu {{
                background-color: {BACKGROUND_COLOR};
                border: 1px solid {BORDER_COLOR};
                padding: 4px;
            }}
            QMenu::item {{
                padding: 8px 16px;
            }}
            QMenu::item:selected {{
                background-color: {BORDER_COLOR};
            }}
        """
        )

        # File menu
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open Audio File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_file)
        file_menu.addAction(open_action)

        record_action = QAction("&Record from Microphone", self)
        record_action.setShortcut("Ctrl+R")
        record_action.triggered.connect(self._on_record_audio)
        file_menu.addAction(record_action)

        file_menu.addSeparator()

        # Combined save action replaces separate transcript and summary save actions
        save_combined_action = QAction("&Save Content...", self)
        save_combined_action.setShortcut("Ctrl+S")
        save_combined_action.triggered.connect(self._on_save_combined)
        file_menu.addAction(save_combined_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Tools menu
        tools_menu = menu_bar.addMenu("&Tools")

        transcribe_action = QAction("&Transcribe", self)
        transcribe_action.setShortcut("F5")
        transcribe_action.triggered.connect(self._on_transcribe)
        tools_menu.addAction(transcribe_action)

        summarize_action = QAction("&Summarize", self)
        summarize_action.setShortcut("F6")
        summarize_action.triggered.connect(self._on_summarize_clicked)
        tools_menu.addAction(summarize_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _create_toolbar(self):
        """Create the toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setStyleSheet(
            f"""
            QToolBar {{
                background-color: {ACCENT_COLOR};
                border: none;
                padding: 12px 16px;
                spacing: 8px;
                margin: 0px 0px 8px 0px;
                min-height: 44px;
            }}
            QToolBar QToolButton {{
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                padding: 8px 16px;
                margin: 2px;
                font-family: 'Geist', sans-serif;
                font-size: 14px;
                min-height: 20px;
                border-radius: 6px;
            }}
            QToolBar QToolButton:hover {{
                background-color: {BUTTON_HOVER_COLOR};
                border-color: #d4d0c4;
            }}
            QToolBar QToolButton:pressed {{
                background-color: {BORDER_COLOR};
            }}
            QToolBar QToolButton:disabled {{
                background-color: #f5f4f1;
                color: #a0a0a0;
                border-color: {BORDER_COLOR};
            }}
            QToolBar::separator {{
                background-color: rgba(255, 255, 255, 0.2);
                width: 1px;
                margin: 4px 8px;
            }}
        """
        )

        # Set toolbar to be movable and set proper size policy
        toolbar.setMovable(False)
        toolbar.setFloatable(False)

        self.addToolBar(toolbar)

        # Open file button
        open_action = QAction("Open", self)
        open_action.triggered.connect(self._on_open_file)
        toolbar.addAction(open_action)

        # Record button
        # Keep reference so we can change its label later
        self.record_action = QAction("Record", self)
        self.record_action.triggered.connect(self._on_record_audio)
        toolbar.addAction(self.record_action)

        toolbar.addSeparator()

        # Transcribe button
        transcribe_action = QAction("Transcribe", self)
        transcribe_action.triggered.connect(self._on_transcribe)
        toolbar.addAction(transcribe_action)

        # Summarize button
        summarize_action = QAction("Summarize", self)
        summarize_action.triggered.connect(self._on_summarize_clicked)
        toolbar.addAction(summarize_action)

    def _connect_signals(self):
        """Connect signals from the application."""
        # Transcription signals
        self.app.transcription_started.connect(self._on_transcription_started)
        self.app.transcription_progress.connect(self._on_transcription_progress)
        self.app.transcription_complete.connect(self._on_transcription_complete)
        self.app.transcription_failed.connect(self._on_transcription_failed)

        # Summarization signals
        self.app.summarization_started.connect(self._on_summarization_started)
        self.app.summarization_complete.connect(self._on_summarization_complete)
        self.app.summarization_failed.connect(self._on_summarization_failed)

    # ------------------------------------------------------------------#
    # Thread helpers                                                    #
    # ------------------------------------------------------------------#

    class _FunctionWorker(QRunnable):
        """Run an arbitrary callable in a background Qt thread."""

        def __init__(self, fn, *args, **kwargs):
            super().__init__()
            self.fn = fn
            self.args = args
            self.kwargs = kwargs

        def run(self):
            try:
                self.fn(*self.args, **self.kwargs)
            except Exception:  # pragma: no cover
                logger.exception("Background task failed")

    # ------------------------------------------------------------------#
    # Graceful-shutdown helpers                                         #
    # ------------------------------------------------------------------#

    def _cleanup_resources(self) -> None:
        """
        Stop any timers / recordings and propagate shutdown to the core
        application.  This is called from both the window close event and the
        global ``aboutToQuit`` signal to guarantee cleanup.
        """
        # Stop active recording stream
        if getattr(self, "_recording", False):
            try:
                self.app.stop_streaming_transcription()
            except Exception:  # pragma: no cover
                logger.exception("Error stopping recording during shutdown")

        # Stop timers
        self._stop_progress_timer()
        if self._spinner_timer.isActive():
            self._spinner_timer.stop()

        # Call application-level shutdown to release subprocesses / temp files
        try:
            self.app.shutdown()
        except Exception:  # pragma: no cover
            logger.exception("Error during app.shutdown()")

    def _on_app_quit(self) -> None:
        """
        Handle application quit signal.

        This is connected to the QApplication.aboutToQuit signal to ensure
        resources are properly cleaned up even if the window close event
        is bypassed.
        """
        logger.info("Application quit signal received, cleaning up resources")
        self._cleanup_resources()

    # ------------------------------------------------------------------#
    # Progress estimation helpers                                       #
    # ------------------------------------------------------------------#

    def _get_audio_duration(self, audio_path: str) -> float:
        """
        Get the duration of an audio file in seconds.

        Args:
            audio_path: Path to the audio file

        Returns:
            Duration in seconds or a default value if duration can't be determined
        """
        try:
            # Try to use ffprobe to get duration
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            logger.info(f"Audio duration from ffprobe: {duration:.2f} seconds")
            return duration
        except (subprocess.SubprocessError, ValueError) as e:
            logger.warning(f"Could not determine audio duration with ffprobe: {e}")

            # Fallback: use file size as a rough estimate (1MB ≈ 1 minute)
            try:
                file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                estimated_duration = (
                    file_size_mb * 60
                )  # 1MB ≈ 1 minute as rough estimate
                logger.info(
                    f"Estimated audio duration from file size: {estimated_duration:.2f} seconds"
                )
                return max(estimated_duration, 30)  # At least 30 seconds
            except Exception as e2:
                logger.warning(f"Could not estimate duration from file size: {e2}")

                # Default fallback: assume 3 minutes
                logger.info("Using default duration of 180 seconds")
                return 180.0

    def _start_progress_timer(self, audio_duration: float):
        """
        Start the progress timer for time-based estimation.

        Args:
            audio_duration: Duration of the audio in seconds
        """
        # Set up timer parameters
        self._progress_total_secs = audio_duration / 20  # Divide by 20 as requested
        self._progress_elapsed_secs = 0.0
        self._progress_start_time = QDateTime.currentDateTime()

        # Start the timer
        if not self._progress_timer.isActive():
            self._progress_timer.start()

        logger.info(
            f"Started progress timer: estimating {self._progress_total_secs:.2f} seconds for transcription"
        )

    def _stop_progress_timer(self):
        """Stop the progress timer."""
        if self._progress_timer.isActive():
            self._progress_timer.stop()
            logger.info("Stopped progress timer")

    def _on_progress_tick(self):
        """Update progress based on elapsed time."""
        if self._progress_start_time is None or self._progress_total_secs <= 0:
            return

        # Calculate elapsed time
        now = QDateTime.currentDateTime()
        self._progress_elapsed_secs = self._progress_start_time.msecsTo(now) / 1000.0

        # Calculate progress percentage (capped at 95% to avoid appearing complete)
        progress_pct = min(
            self._progress_elapsed_secs / self._progress_total_secs, 0.95
        )

        # Update progress bar
        self.progress_bar.setValue(int(progress_pct * 100))

        # Update status with time estimate
        remaining_secs = max(0, self._progress_total_secs - self._progress_elapsed_secs)
        if remaining_secs > 0:
            self.status_label.setText(
                f"Transcribing... (approx. {int(remaining_secs)} seconds remaining)"
            )

    # ------------------------------------------------------------------#
    # Loading / spinner helpers                                         #
    # ------------------------------------------------------------------#

    def _update_spinner(self) -> None:
        """Advance spinner frame."""
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_chars)
        char = self._spinner_chars[self._spinner_idx]
        self.loading_label.setText(f"Loading {char}")

    def _show_loading(self, show: bool = True) -> None:
        """Toggle spinner visibility."""
        if show:
            if not self._spinner_timer.isActive():
                self._spinner_idx = 0
                self._spinner_timer.start()
            self.loading_label.setVisible(True)
        else:
            if self._spinner_timer.isActive():
                self._spinner_timer.stop()
            self.loading_label.setVisible(False)

    def _wrap_text(self, text, width=65):
        """Wrap text to specified width."""
        wrapped_lines = []
        for line in text.split("\n"):
            if line.strip():
                wrapped_lines.extend(textwrap.wrap(line, width=width))
            else:
                wrapped_lines.append("")  # Preserve empty lines
        return "\n".join(wrapped_lines)

    def _update_combined_content(
        self, transcript_text="", summary_text="", notes_text=""
    ):
        """
        Update the combined content with summary, transcript, and notes.

        Args:
            transcript_text: The transcript text
            summary_text: The summary text (markdown)
            notes_text: The notes text
        """
        # Apply 65 character line wrapping to transcript
        wrapped_transcript = self._wrap_text(transcript_text, width=65)

        # Include notes in the raw content for saving
        content_parts = []
        if summary_text:
            content_parts.append(summary_text)
        if notes_text:
            content_parts.append(f"## Notes\n\n{notes_text}")
        if transcript_text:
            content_parts.append(f"## Transcript\n\n{transcript_text}")

        # Store raw content for saving (use unwrapped text for saving)
        self._combined_markdown = "\n\n---\n\n".join(content_parts)

        # Convert markdown to plain text for better font rendering
        display_parts = []

        if summary_text:
            # Simple markdown to plain text conversion
            plain_summary = self._markdown_to_plain_text(summary_text)
            wrapped_summary = self._wrap_text(plain_summary, width=65)
            display_parts.append(wrapped_summary)

        if notes_text:
            wrapped_notes = self._wrap_text(f"NOTES:\n{notes_text}", width=65)
            display_parts.append(wrapped_notes)

        if transcript_text:
            display_parts.append(f"TRANSCRIPT:\n{wrapped_transcript}")

        # Create the combined plain text content
        combined_content = f"\n\n{'─' * 65}\n\n".join(display_parts)

        # Set as plain text to avoid font rendering issues
        self.combined_text.setPlainText(combined_content)

    def _markdown_to_plain_text(self, markdown_text: str) -> str:
        """
        Convert markdown to plain text with simple formatting.

        Args:
            markdown_text: The markdown text to convert

        Returns:
            Plain text representation
        """
        text = markdown_text

        # Convert headers to uppercase with spacing
        import re

        # H1 headers (# )
        text = re.sub(
            r"^# (.+)$",
            lambda m: f"\n{m.group(1).upper()}\n{'=' * len(m.group(1))}",
            text,
            flags=re.MULTILINE,
        )

        # H2 headers (## )
        text = re.sub(
            r"^## (.+)$",
            lambda m: f"\n{m.group(1).upper()}\n{'-' * len(m.group(1))}",
            text,
            flags=re.MULTILINE,
        )

        # H3+ headers (### )
        text = re.sub(
            r"^#{3,} (.+)$",
            lambda m: f"\n{m.group(1).upper()}",
            text,
            flags=re.MULTILINE,
        )

        # Remove bold/italic markers (keep the text)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # **bold**
        text = re.sub(r"__(.+?)__", r"\1", text)  # __bold__
        text = re.sub(r"\*(.+?)\*", r"\1", text)  # *italic*
        text = re.sub(r"_(.+?)_", r"\1", text)  # _italic_

        # Convert bullet points
        text = re.sub(r"^[-*+] (.+)$", r"• \1", text, flags=re.MULTILINE)

        # Convert numbered lists
        text = re.sub(r"^\d+\. (.+)$", r"• \1", text, flags=re.MULTILINE)

        # Clean up extra whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return text

    @Slot()
    def _on_open_file(self):
        """Handle opening an audio file."""
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Open Audio File")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter(
            "Audio Files (*.wav *.mp3 *.m4a *.flac *.ogg);;All Files (*.*)"
        )

        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                audio_path = file_paths[0]
                self._open_audio_file(audio_path)

    def _open_audio_file(self, audio_path: str):
        """
        Open an audio file for transcription.

        Args:
            audio_path: Path to the audio file
        """
        try:
            # Create audio file object
            audio_file = AudioFile.from_path(Path(audio_path))

            # Update UI
            self.status_label.setText(f"Loaded: {audio_file.path.name}")
            self.transcript_status.setText(f"File: {audio_file.path.name}")
            self.combined_text.clear()
            self._combined_markdown = ""
            self.summary_status.setText("No summary generated")

            # Auto-transcribe if configured
            if self.app.config.auto_transcribe:
                self._transcribe_file(audio_path)

        except Exception as e:
            logger.error(f"Error opening audio file: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open audio file: {e}")

    @Slot()
    def _on_record_audio(self):
        """Handle recording from microphone."""
        # Check if already recording
        if self._recording:
            # Stop recording
            self.app.stop_streaming_transcription()
            self._recording = False
            # Restore record button text
            if hasattr(self, "record_action"):
                self.record_action.setText("Record")

            # Disable notes input
            self.notes_text.setEnabled(False)

            # Attempt to fetch finalised transcript and update UI
            transcript = (
                self.app.get_transcript(self._active_transcript_id)
                if self._active_transcript_id
                else None
            )
            if transcript:
                self.transcript_status.setText("Transcription complete")
                self.status_label.setText("Transcription complete")

                # Update the combined text with transcript
                self._update_combined_content(
                    transcript_text=transcript.text,
                    notes_text=getattr(transcript, "notes", ""),
                )

                # Auto-summarize if LLM is available
                if hasattr(self.app, "summarizer") and self.app.summarizer:
                    self._start_summarization(transcript)
            else:
                # Fallback UI update
                self.status_label.setText("Recording stopped")
                self.transcript_status.setText("Processing recording...")
            return

        try:
            # Start streaming transcription
            transcript = self.app.start_streaming_transcription()
            self._active_transcript_id = transcript.id

            # Update UI
            self.status_label.setText("Recording from microphone...")
            self.transcript_status.setText("Recording in progress...")
            self.combined_text.clear()
            self._combined_markdown = ""
            self.summary_status.setText("No summary generated")

            # Enable notes input for recording
            self.notes_text.setEnabled(True)
            self.notes_text.clear()

            # Set recording flag
            self._recording = True
            # Update record button text
            if hasattr(self, "record_action"):
                self.record_action.setText("Stop Recording")

        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start recording: {e}")

    @Slot()
    def _on_transcribe(self):
        """Handle manual transcription request."""
        # If we have an active transcript, use that
        if self._active_transcript_id:
            transcript = self.app.get_transcript(self._active_transcript_id)
            if transcript and transcript.audio_file.path != Path("stream"):
                self._transcribe_file(str(transcript.audio_file.path))
                return

        # Otherwise, prompt for a file
        self._on_open_file()

    def _transcribe_file(self, audio_path: str):
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file
        """
        try:
            # Update UI
            self.status_label.setText(f"Transcribing: {Path(audio_path).name}")
            self.transcript_status.setText("Transcription in progress...")
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            self._show_loading(True)  # Show loading animation

            # Get audio duration for progress estimation
            audio_duration = self._get_audio_duration(audio_path)

            # Start progress timer
            self._start_progress_timer(audio_duration)

            # Start transcription in a separate thread
            def on_progress(progress, message):
                # We'll ignore the actual progress since we're using time-based estimation
                # But we'll still use the messages
                if message:
                    self.status_label.setText(message)

            # Use app's transcribe_file method
            threading_timer = QTimer.singleShot(
                0,  # Execute as soon as possible
                lambda: self._start_transcription(audio_path, on_progress),
            )

        except Exception as e:
            logger.error(f"Error starting transcription: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start transcription: {e}")
            self.progress_bar.setVisible(False)
            self._show_loading(False)  # Hide loading animation
            self._stop_progress_timer()  # Stop the progress timer

    def _start_transcription(self, audio_path, on_progress):
        """Start transcription in the app."""
        try:
            # Use app's transcribe_file method
            transcript = self.app.transcribe_file(
                audio_path=audio_path,
                on_progress=on_progress,
            )
            self._active_transcript_id = transcript.id
        except Exception as e:
            logger.error(f"Error in transcription thread: {e}")
            # We'll handle this via the transcription_failed signal
            self._stop_progress_timer()  # Stop the progress timer

    @Slot()
    def _on_summarize_clicked(self):
        """Handle summarize button click."""
        if not self._active_transcript_id:
            QMessageBox.warning(self, "Warning", "No transcript available to summarize")
            return

        try:
            # Get the transcript
            transcript = self.app.get_transcript(self._active_transcript_id)
            if not transcript:
                QMessageBox.warning(self, "Warning", "Transcript not found")
                return

            # Check if LLM is enabled
            if not hasattr(self.app, "summarizer") or not self.app.summarizer:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Summarization is disabled. Please enable an LLM provider in the configuration.",
                )
                return

            # Update UI
            self.status_label.setText("Generating summary...")
            self.summary_status.setText("Summarization in progress...")
            self._show_loading(True)  # Show loading animation

            # Start summarization in a separate thread
            threading_timer = QTimer.singleShot(
                0,  # Execute as soon as possible
                lambda: self._start_summarization(transcript),
            )

        except Exception as e:
            logger.error(f"Error starting summarization: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start summarization: {e}")
            self._show_loading(False)  # Hide loading animation

    def _start_summarization(self, transcript):
        """Start summarization in the app."""
        try:
            # Make sure notes are saved to the transcript before summarization
            if self._active_transcript_id and hasattr(self, "notes_text"):
                current_notes = self.notes_text.toPlainText()
                self.app.update_transcript_notes(
                    self._active_transcript_id, current_notes
                )

            # Use app's summarize_transcript method
            summary = self.app.summarize_transcript(transcript)
            if summary:
                self._active_summary_id = summary.id
        except Exception as e:
            logger.error(f"Error in summarization thread: {e}")
            # We'll handle this via the summarization_failed signal

    @Slot()
    def _on_save_combined(self):
        """Handle saving the combined content (summary + transcript)."""
        if not self._active_transcript_id:
            QMessageBox.warning(self, "Warning", "No content available to save")
            return

        transcript = self.app.get_transcript(self._active_transcript_id)
        if not transcript:
            QMessageBox.warning(self, "Warning", "Transcript not found")
            return

        # Show save dialog
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Save Content")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter(
            "Text Files (*.txt);;Markdown Files (*.md);;All Files (*.*)"
        )

        # Suggest filename based on audio file
        suggested_name = transcript.audio_file.path.stem + "_transcript.md"
        file_dialog.selectFile(suggested_name)

        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                save_path = file_paths[0]

                try:
                    # Save the combined content
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(self._combined_markdown)

                    self.status_label.setText(f"Content saved to {save_path}")

                except Exception as e:
                    logger.error(f"Error saving content: {e}")
                    QMessageBox.critical(self, "Error", f"Failed to save content: {e}")

    @Slot()
    def _on_about(self):
        """Show about dialog."""
        about_text = (
            f"<h2>{self.app.config.app_name} v{self.app.config.app_version}</h2>"
            "<p>An offline-first, privacy-centric voice transcription "
            "and summarization desktop application.</p>"
            "<p>Built with:</p>"
            "<ul>"
            "<li>Python and PySide6</li>"
            "<li>whisper.cpp for transcription</li>"
            "<li>Ollama for LLM capabilities</li>"
            "</ul>"
        )

        QMessageBox.about(self, f"About {self.app.config.app_name}", about_text)

    @Slot(str)
    def _on_transcription_started(self, transcript_id):
        """Handle transcription started signal."""
        self._active_transcript_id = transcript_id
        self.transcript_status.setText("Transcription in progress...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self._show_loading(True)  # Show loading animation

    @Slot(str, float, str)
    def _on_transcription_progress(self, transcript_id, progress, message):
        """Handle transcription progress signal."""
        if transcript_id != self._active_transcript_id:
            return

        # We don't update the progress bar here since we're using time-based estimation
        # But we'll still update the status message
        if message:
            # Don't override the time remaining message from the timer
            if not message.startswith("Transcribing... (approx"):
                self.status_label.setText(message)

        # Update transcript text if available
        transcript = self.app.get_transcript(transcript_id)
        if transcript:
            # Update the combined text with transcript only during progress
            self._update_combined_content(
                transcript_text=transcript.text,
                notes_text=getattr(transcript, "notes", ""),
            )

            # Auto-scroll to bottom
            if self.app.config.ui.auto_scroll:
                cursor = self.combined_text.textCursor()
                cursor.movePosition(QTextCursor.End)
                self.combined_text.setTextCursor(cursor)

    @Slot(str)
    def _on_transcription_complete(self, transcript_id):
        """Handle transcription complete signal."""
        if transcript_id != self._active_transcript_id:
            return

        # Stop the progress timer
        self._stop_progress_timer()

        # Get the transcript
        transcript = self.app.get_transcript(transcript_id)
        if not transcript:
            return

        # Update UI
        self.transcript_status.setText(f"Transcription complete")
        self.status_label.setText("Transcription complete")
        self.progress_bar.setVisible(False)
        self._show_loading(False)  # Hide loading animation

        # Update the combined text with transcript
        self._update_combined_content(
            transcript_text=transcript.text, notes_text=getattr(transcript, "notes", "")
        )

        # Auto-summarize if configured and LLM is available
        if (
            self.app.config.auto_summarize
            and hasattr(self.app, "summarizer")
            and self.app.summarizer
        ):
            self._start_summarization(transcript)

    @Slot(str, str)
    def _on_transcription_failed(self, transcript_id, error_message):
        """Handle transcription failed signal."""
        if transcript_id != self._active_transcript_id:
            return

        # Stop the progress timer
        self._stop_progress_timer()

        # Update UI
        self.transcript_status.setText("Transcription failed")
        self.status_label.setText(f"Error: {error_message}")
        self.progress_bar.setVisible(False)
        self._show_loading(False)  # Hide loading animation

        # Show error message
        QMessageBox.critical(self, "Transcription Error", error_message)

    @Slot(str)
    def _on_summarization_started(self, summary_id):
        """Handle summarization started signal."""
        self._active_summary_id = summary_id
        self.summary_status.setText("Summarization in progress...")
        self.status_label.setText("Generating summary...")
        self._show_loading(True)  # Show loading animation

    @Slot(str)
    def _on_summarization_complete(self, summary_id):
        """Handle summarization complete signal."""
        if summary_id != self._active_summary_id:
            return

        # Get the summary
        summary = self.app.get_summary(summary_id)
        if not summary:
            return

        # Get the transcript
        transcript = self.app.get_transcript(summary.transcript_id)
        if not transcript:
            return

        # Update UI
        self.summary_status.setText("Summary generated")
        self.status_label.setText("Summary generated successfully")
        self._show_loading(False)  # Hide loading animation

        # Update the combined text with summary and transcript
        self._update_combined_content(
            transcript_text=transcript.text,
            summary_text=summary.text,
            notes_text=getattr(transcript, "notes", ""),
        )

    @Slot(str, str)
    def _on_summarization_failed(self, summary_id, error_message):
        """Handle summarization failed signal."""
        if summary_id != self._active_summary_id:
            return

        # Update UI
        self.summary_status.setText("Summarization failed")
        self.status_label.setText(f"Error: {error_message}")
        self._show_loading(False)  # Hide loading animation

        # Show error message
        QMessageBox.critical(self, "Summarization Error", error_message)

    def closeEvent(self, event):
        """Handle window close event."""
        # Run cleanup to ensure all resources are properly released
        self._cleanup_resources()

        # Accept the close event
        event.accept()

    def _on_notes_changed(self):
        """Handle notes text changes."""
        if self._active_transcript_id and self._recording:
            notes_text = self.notes_text.toPlainText()
            self.app.update_transcript_notes(self._active_transcript_id, notes_text)
