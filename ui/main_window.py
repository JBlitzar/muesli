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
    QSplitter, QStatusBar, QTextEdit, QToolBar, QVBoxLayout, QWidget
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
        
        # Create splitter for transcript and summary
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # Create transcript area
        transcript_widget = QWidget()
        transcript_layout = QVBoxLayout(transcript_widget)
        
        transcript_header = QHBoxLayout()
        transcript_label = QLabel("Transcript")
        transcript_label.setFont(QFont("Arial", 12, QFont.Bold))
        transcript_header.addWidget(transcript_label)
        
        self.transcript_status = QLabel("No transcript loaded")
        transcript_header.addWidget(self.transcript_status)
        transcript_header.addStretch()
        
        self.transcript_text = QTextEdit()
        self.transcript_text.setReadOnly(True)
        self.transcript_text.setPlaceholderText("Transcript will appear here")
        
        transcript_layout.addLayout(transcript_header)
        transcript_layout.addWidget(self.transcript_text)
        
        # Create summary area
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        
        summary_header = QHBoxLayout()
        summary_label = QLabel("Summary")
        summary_label.setFont(QFont("Arial", 12, QFont.Bold))
        summary_header.addWidget(summary_label)
        
        self.summary_status = QLabel("No summary generated")
        summary_header.addWidget(self.summary_status)
        summary_header.addStretch()
        
        self.summarize_button = QPushButton("Generate Summary")
        self.summarize_button.setEnabled(False)
        self.summarize_button.clicked.connect(self._on_summarize_clicked)
        summary_header.addWidget(self.summarize_button)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText("Summary will appear here")
        
        summary_layout.addLayout(summary_header)
        summary_layout.addWidget(self.summary_text)
        
        # Add widgets to splitter
        splitter.addWidget(transcript_widget)
        splitter.addWidget(summary_widget)
        splitter.setSizes([600, 400])  # Initial sizes
        
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
        
        save_transcript_action = QAction("Save &Transcript...", self)
        save_transcript_action.setShortcut("Ctrl+S")
        save_transcript_action.triggered.connect(self._on_save_transcript)
        file_menu.addAction(save_transcript_action)
        
        save_summary_action = QAction("Save S&ummary...", self)
        save_summary_action.setShortcut("Ctrl+Shift+S")
        save_summary_action.triggered.connect(self._on_save_summary)
        file_menu.addAction(save_summary_action)
        
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
        record_action = QAction("Record", self)
        record_action.triggered.connect(self._on_record_audio)
        toolbar.addAction(record_action)
        
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
            self.transcript_text.clear()
            self.summary_text.clear()
            self.summary_status.setText("No summary generated")
            self.summarize_button.setEnabled(False)
            
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
        if hasattr(self, '_recording') and self._recording:
            # Stop recording
            self.app.stop_streaming_transcription()
            self._recording = False
            self.status_label.setText("Recording stopped")
            return
        
        try:
            # Start streaming transcription
            transcript = self.app.start_streaming_transcription()
            self._active_transcript_id = transcript.id
            
            # Update UI
            self.status_label.setText("Recording from microphone...")
            self.transcript_status.setText("Recording in progress...")
            self.transcript_text.clear()
            self.summary_text.clear()
            self.summary_status.setText("No summary generated")
            self.summarize_button.setEnabled(False)
            
            # Set recording flag
            self._recording = True
            
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
    def _on_save_transcript(self):
        """Handle saving the transcript."""
        if not self._active_transcript_id:
            QMessageBox.warning(self, "Warning", "No transcript available to save")
            return
        
        transcript = self.app.get_transcript(self._active_transcript_id)
        if not transcript:
            QMessageBox.warning(self, "Warning", "Transcript not found")
            return
        
        # Show save dialog
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Save Transcript")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("Text Files (*.txt);;SRT Files (*.srt);;All Files (*.*)")
        
        # Suggest filename based on audio file
        suggested_name = transcript.audio_file.path.stem + ".txt"
        file_dialog.selectFile(suggested_name)
        
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                save_path = file_paths[0]
                
                try:
                    # Determine format based on extension
                    if save_path.lower().endswith(".srt"):
                        # Save as SRT
                        with open(save_path, "w", encoding="utf-8") as f:
                            f.write(transcript.to_srt())
                    else:
                        # Save as plain text
                        with open(save_path, "w", encoding="utf-8") as f:
                            f.write(transcript.text)
                    
                    self.status_label.setText(f"Transcript saved to {save_path}")
                    
                except Exception as e:
                    logger.error(f"Error saving transcript: {e}")
                    QMessageBox.critical(self, "Error", f"Failed to save transcript: {e}")
    
    @Slot()
    def _on_save_summary(self):
        """Handle saving the summary."""
        if not self._active_summary_id:
            QMessageBox.warning(self, "Warning", "No summary available to save")
            return
        
        summary = self.app.get_summary(self._active_summary_id)
        if not summary:
            QMessageBox.warning(self, "Warning", "Summary not found")
            return
        
        # Show save dialog
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Save Summary")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("Text Files (*.txt);;All Files (*.*)")
        
        # Get the transcript to suggest a filename
        transcript = self.app.get_transcript(summary.transcript_id)
        if transcript:
            suggested_name = transcript.audio_file.path.stem + "_summary.txt"
        else:
            suggested_name = "summary.txt"
        
        file_dialog.selectFile(suggested_name)
        
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                save_path = file_paths[0]
                
                try:
                    # Save as plain text
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(summary.text)
                    
                    self.status_label.setText(f"Summary saved to {save_path}")
                    
                except Exception as e:
                    logger.error(f"Error saving summary: {e}")
                    QMessageBox.critical(self, "Error", f"Failed to save summary: {e}")
    
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
            self.transcript_text.setText(transcript.text)
            
            # Auto-scroll to bottom
            if self.app.config.ui.auto_scroll:
                cursor = self.transcript_text.textCursor()
                cursor.movePosition(QTextCursor.End)
                self.transcript_text.setTextCursor(cursor)
    
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
        
        # Update transcript text
        self.transcript_text.setText(transcript.text)
        
        # Enable summarize button if LLM is available
        if hasattr(self.app, 'summarizer') and self.app.summarizer:
            self.summarize_button.setEnabled(True)
    
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
        
        # Update UI
        self.summary_status.setText("Summary generated")
        self.status_label.setText("Summary generated successfully")
        
        # Update summary text
        self.summary_text.setText(summary.text)
    
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
