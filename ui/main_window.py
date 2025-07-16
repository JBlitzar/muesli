"""
Main window UI for the Muesli application.

This module provides the main window UI for the Muesli application,
including controls for transcription, summarization, and displaying results.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QUrl, Slot, QTimer
from PySide6.QtGui import QAction, QFont, QIcon, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, 
    QMenu, QMenuBar, QMessageBox, QProgressBar, QPushButton, 
    QStatusBar, QTextEdit, QToolBar, QVBoxLayout, QWidget
)
from PySide6.QtQml import QQmlApplicationEngine

from models import Transcript, Summary, AudioFile

logger = logging.getLogger(__name__)


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

        # Set window properties
        self.setWindowTitle(f"{app.config.app_name} v{app.config.app_version}")
        self.resize(1024, 768)
        
        # Active transcripts and summaries
        self._active_transcript_id = None
        self._active_summary_id = None
        
        # Set up UI components
        self._setup_ui()
        
        # Connect signals from application
        self._connect_signals()
        
        logger.info("Main window initialized")
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create toolbar
        self._create_toolbar()
        
        # ----- Combined transcript & summary area -----
        header_layout = QHBoxLayout()
        header_label = QLabel("Summary & Transcript")
        header_label.setFont(QFont("Arial", 12, QFont.Bold))
        header_layout.addWidget(header_label)

        self.transcript_status = QLabel("No transcript loaded")
        header_layout.addWidget(self.transcript_status)

        self.summary_status = QLabel("No summary generated")
        header_layout.addWidget(self.summary_status)
        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        self.combined_text = QTextEdit()
        self.combined_text.setReadOnly(True)
        self.combined_text.setPlaceholderText(
            "Summary (Markdown) and transcript will appear here"
        )
        main_layout.addWidget(self.combined_text)

        # Holds latest markdown so we can save exact content
        self._combined_markdown: str = ""
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label, 1)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.status_bar.addWidget(self.progress_bar)
    
    def _create_menu_bar(self):
        """Create the menu bar."""
        menu_bar = self.menuBar()
        
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
    
    def _update_combined_content(self, transcript_text="", summary_text=""):
        """
        Update the combined content with summary and transcript.
        
        Args:
            transcript_text: The transcript text
            summary_text: The summary text (markdown)
        """
        # Store raw content for saving
        if summary_text:
            self._combined_markdown = f"{summary_text}\n\n---\n\n{transcript_text}"
        else:
            self._combined_markdown = transcript_text
        
        # Render markdown for the summary
        if summary_text:
            # Simple markdown to HTML conversion for the summary
            # This is a basic implementation - in a real app, you'd use a proper markdown library
            html_summary = summary_text
            # Convert headers
            for i in range(6, 0, -1):
                h_tag = f"h{i}"
                html_summary = html_summary.replace(f"{'#' * i} ", f"<{h_tag}>") + f"</{h_tag}>"
            
            # Convert bold and italic
            html_summary = html_summary.replace("**", "<strong>").replace("__", "<strong>")
            html_summary = html_summary.replace("*", "<em>").replace("_", "<em>")
            
            # Convert bullet lists
            html_summary = html_summary.replace("\n- ", "\n<li>").replace("\n* ", "\n<li>")
            if "<li>" in html_summary:
                html_summary = "<ul>" + html_summary + "</ul>"
                html_summary = html_summary.replace("\n<li>", "</li>\n<li>")
            
            # Convert line breaks
            html_summary = html_summary.replace("\n\n", "<br><br>")
            
            # Create the combined HTML content
            html_content = f"""
            <html>
            <body>
            {html_summary}
            <hr>
            <pre>{transcript_text}</pre>
            </body>
            </html>
            """
            
            # Set the HTML content
            self.combined_text.setHtml(html_content)
        else:
            # Just plain text for transcript only
            self.combined_text.setPlainText(transcript_text)
    
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

            # Attempt to fetch finalised transcript and update UI
            transcript = self.app.get_transcript(self._active_transcript_id) if self._active_transcript_id else None
            if transcript:
                self.transcript_status.setText("Transcription complete")
                self.status_label.setText("Transcription complete")
                
                # Update the combined text with transcript
                self._update_combined_content(transcript_text=transcript.text)
                
                # Auto-summarize if LLM is available
                if hasattr(self.app, 'summarizer') and self.app.summarizer:
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
            
            # Start transcription in a separate thread
            def on_progress(progress, message):
                # Update progress bar
                self.progress_bar.setValue(int(progress * 100))
            
            # Use app's transcribe_file method
            threading_timer = QTimer.singleShot(
                0,  # Execute as soon as possible
                lambda: self._start_transcription(audio_path, on_progress)
            )
            
        except Exception as e:
            logger.error(f"Error starting transcription: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start transcription: {e}")
            self.progress_bar.setVisible(False)
    
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
            if not hasattr(self.app, 'summarizer') or not self.app.summarizer:
                QMessageBox.warning(
                    self, 
                    "Warning", 
                    "Summarization is disabled. Please enable an LLM provider in the configuration."
                )
                return
            
            # Update UI
            self.status_label.setText("Generating summary...")
            self.summary_status.setText("Summarization in progress...")
            
            # Start summarization in a separate thread
            threading_timer = QTimer.singleShot(
                0,  # Execute as soon as possible
                lambda: self._start_summarization(transcript)
            )
            
        except Exception as e:
            logger.error(f"Error starting summarization: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start summarization: {e}")
    
    def _start_summarization(self, transcript):
        """Start summarization in the app."""
        try:
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
        file_dialog.setNameFilter("Text Files (*.txt);;Markdown Files (*.md);;All Files (*.*)")
        
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
    
    @Slot(str, float, str)
    def _on_transcription_progress(self, transcript_id, progress, message):
        """Handle transcription progress signal."""
        if transcript_id != self._active_transcript_id:
            return
        
        # Update progress bar
        self.progress_bar.setValue(int(progress * 100))
        
        # Update status
        if message:
            self.status_label.setText(message)
        
        # Update transcript text if available
        transcript = self.app.get_transcript(transcript_id)
        if transcript:
            # Update the combined text with transcript only during progress
            self._update_combined_content(transcript_text=transcript.text)
            
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
        
        # Get the transcript
        transcript = self.app.get_transcript(transcript_id)
        if not transcript:
            return
        
        # Update UI
        self.transcript_status.setText(f"Transcription complete")
        self.status_label.setText("Transcription complete")
        self.progress_bar.setVisible(False)
        
        # Update the combined text with transcript
        self._update_combined_content(transcript_text=transcript.text)
        
        # Auto-summarize if configured and LLM is available
        if self.app.config.auto_summarize and hasattr(self.app, 'summarizer') and self.app.summarizer:
            self._start_summarization(transcript)
    
    @Slot(str, str)
    def _on_transcription_failed(self, transcript_id, error_message):
        """Handle transcription failed signal."""
        if transcript_id != self._active_transcript_id:
            return
        
        # Update UI
        self.transcript_status.setText("Transcription failed")
        self.status_label.setText(f"Error: {error_message}")
        self.progress_bar.setVisible(False)
        
        # Show error message
        QMessageBox.critical(self, "Transcription Error", error_message)
    
    @Slot(str)
    def _on_summarization_started(self, summary_id):
        """Handle summarization started signal."""
        self._active_summary_id = summary_id
        self.summary_status.setText("Summarization in progress...")
        self.status_label.setText("Generating summary...")
    
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
        
        # Update the combined text with summary and transcript
        self._update_combined_content(
            transcript_text=transcript.text,
            summary_text=summary.text
        )
    
    @Slot(str, str)
    def _on_summarization_failed(self, summary_id, error_message):
        """Handle summarization failed signal."""
        if summary_id != self._active_summary_id:
            return
        
        # Update UI
        self.summary_status.setText("Summarization failed")
        self.status_label.setText(f"Error: {error_message}")
        
        # Show error message
        QMessageBox.critical(self, "Summarization Error", error_message)
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop any active recording
        if hasattr(self, '_recording') and self._recording:
            self.app.stop_streaming_transcription()
        
        # Accept the close event
        event.accept()
