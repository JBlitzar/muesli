"""
Main window for the Muesli application.

This module provides the main application window for the Muesli application,
built with PySide6 and QML. It handles the integration between the Python
backend and QML frontend, and provides the main user interface for the
application.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import (
    QObject, 
    Signal, 
    Slot, 
    QUrl, 
    Property, 
    QStringListModel,
    QTimer,
)
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine, QQmlContext
from PySide6.QtWidgets import QMainWindow, QApplication

from muesli.core.models import Transcript, Summary

logger = logging.getLogger(__name__)


class TranscriptModel(QObject):
    """
    Model for transcript data to be used in QML.
    
    This class provides a bridge between the Python transcript data
    and the QML interface, exposing properties and signals for
    binding in QML.
    """
    
    # Signals for property changes
    textChanged = Signal()
    segmentsChanged = Signal()
    isRecordingChanged = Signal()
    
    def __init__(self, parent: Optional[QObject] = None):
        """Initialize the transcript model."""
        super().__init__(parent)
        self._text = ""
        self._segments = []
        self._is_recording = False
    
    @Property(str, notify=textChanged)
    def text(self) -> str:
        """Get the full transcript text."""
        return self._text
    
    @text.setter
    def text(self, value: str) -> None:
        """Set the transcript text."""
        if self._text != value:
            self._text = value
            self.textChanged.emit()
    
    @Property(list, notify=segmentsChanged)
    def segments(self) -> List[Dict[str, Any]]:
        """Get the transcript segments."""
        return self._segments
    
    @segments.setter
    def segments(self, value: List[Dict[str, Any]]) -> None:
        """Set the transcript segments."""
        self._segments = value
        self.segmentsChanged.emit()
    
    @Property(bool, notify=isRecordingChanged)
    def isRecording(self) -> bool:
        """Get whether recording is active."""
        return self._is_recording
    
    @isRecording.setter
    def isRecording(self, value: bool) -> None:
        """Set whether recording is active."""
        if self._is_recording != value:
            self._is_recording = value
            self.isRecordingChanged.emit()
    
    @Slot()
    def clear(self) -> None:
        """Clear the transcript data."""
        self._text = ""
        self._segments = []
        self.textChanged.emit()
        self.segmentsChanged.emit()
    
    def update_from_transcript(self, transcript: Transcript) -> None:
        """
        Update the model from a Transcript object.
        
        Args:
            transcript: Transcript object to update from
        """
        # Update text
        self.text = transcript.text
        
        # Update segments
        segments = []
        for segment in transcript.segments:
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "confidence": segment.confidence,
                "speaker": segment.speaker or 0,
            })
        
        self.segments = segments


class SummaryModel(QObject):
    """
    Model for summary data to be used in QML.
    
    This class provides a bridge between the Python summary data
    and the QML interface, exposing properties and signals for
    binding in QML.
    """
    
    # Signals for property changes
    textChanged = Signal()
    isGeneratingChanged = Signal()
    
    def __init__(self, parent: Optional[QObject] = None):
        """Initialize the summary model."""
        super().__init__(parent)
        self._text = ""
        self._is_generating = False
    
    @Property(str, notify=textChanged)
    def text(self) -> str:
        """Get the summary text."""
        return self._text
    
    @text.setter
    def text(self, value: str) -> None:
        """Set the summary text."""
        if self._text != value:
            self._text = value
            self.textChanged.emit()
    
    @Property(bool, notify=isGeneratingChanged)
    def isGenerating(self) -> bool:
        """Get whether summary generation is active."""
        return self._is_generating
    
    @isGenerating.setter
    def isGenerating(self, value: bool) -> None:
        """Set whether summary generation is active."""
        if self._is_generating != value:
            self._is_generating = value
            self.isGeneratingChanged.emit()
    
    @Slot()
    def clear(self) -> None:
        """Clear the summary data."""
        self._text = ""
        self.textChanged.emit()
    
    def update_from_summary(self, summary: Summary) -> None:
        """
        Update the model from a Summary object.
        
        Args:
            summary: Summary object to update from
        """
        self.text = summary.text


class MainWindow(QObject):
    """
    Main window for the Muesli application.
    
    This class provides the main application window for the Muesli
    application, integrating the Python backend with the QML frontend.
    """
    
    # Signals for QML
    transcriptionStarted = Signal()
    transcriptionStopped = Signal()
    transcriptionError = Signal(str)
    
    summaryStarted = Signal()
    summaryCompleted = Signal()
    summaryError = Signal(str)
    
    def __init__(self, app_controller):
        """
        Initialize the main window.
        
        Args:
            app_controller: MuesliApp instance for controlling the application
        """
        super().__init__()
        
        # Store app controller
        self.app_controller = app_controller
        
        # Create models for QML
        self.transcript_model = TranscriptModel(self)
        self.summary_model = SummaryModel(self)
        
        # Initialize QML engine
        self.engine = QQmlApplicationEngine()
        
        # Set up context properties
        root_context = self.engine.rootContext()
        root_context.setContextProperty("transcriptModel", self.transcript_model)
        root_context.setContextProperty("summaryModel", self.summary_model)
        root_context.setContextProperty("mainWindow", self)
        
        # Connect signals from app controller
        self._connect_signals()
        
        # Current transcript and summary
        self._current_transcript_id = None
        self._current_summary_id = None
        
        # Load QML
        self._load_qml()
        
        logger.info("Main window initialized")
    
    def _connect_signals(self) -> None:
        """Connect signals from the app controller."""
        # Transcription signals
        self.app_controller.transcription_started.connect(self._on_transcription_started)
        self.app_controller.transcription_progress.connect(self._on_transcription_progress)
        self.app_controller.transcription_complete.connect(self._on_transcription_complete)
        self.app_controller.transcription_failed.connect(self._on_transcription_failed)
        
        # Summarization signals
        self.app_controller.summarization_started.connect(self._on_summarization_started)
        self.app_controller.summarization_complete.connect(self._on_summarization_complete)
        self.app_controller.summarization_failed.connect(self._on_summarization_failed)
    
    def _load_qml(self) -> None:
        """Load the QML interface."""
        # Get the path to the QML files
        qml_dir = Path(__file__).parent / "qml"
        main_qml = qml_dir / "main.qml"
        
        # Check if QML file exists
        if not main_qml.exists():
            logger.error(f"QML file not found: {main_qml}")
            # Create a placeholder QML file for development
            self._create_placeholder_qml(qml_dir)
            main_qml = qml_dir / "main.qml"
        
        # Load the QML file
        self.engine.load(QUrl.fromLocalFile(str(main_qml)))
        
        # Check if QML loaded successfully
        if not self.engine.rootObjects():
            logger.error("Failed to load QML")
            return
        
        logger.info("QML interface loaded")
    
    def _create_placeholder_qml(self, qml_dir: Path) -> None:
        """
        Create placeholder QML files for development.
        
        Args:
            qml_dir: Directory to create QML files in
        """
        logger.warning("Creating placeholder QML files for development")
        
        # Create QML directory if it doesn't exist
        qml_dir.mkdir(parents=True, exist_ok=True)
        
        # Create main.qml
        main_qml = qml_dir / "main.qml"
        with open(main_qml, "w") as f:
            f.write("""
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

ApplicationWindow {
    visible: true
    width: 800
    height: 600
    title: "Muesli - Private Transcription"
    
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 10
        
        RowLayout {
            Layout.fillWidth: true
            
            Button {
                text: transcriptModel.isRecording ? "Stop Recording" : "Start Recording"
                onClicked: {
                    if (transcriptModel.isRecording) {
                        mainWindow.stopTranscription();
                    } else {
                        mainWindow.startTranscription();
                    }
                }
            }
            
            Button {
                text: "Generate Summary"
                enabled: !summaryModel.isGenerating && transcriptModel.text.length > 0
                onClicked: mainWindow.generateSummary()
            }
            
            Item { Layout.fillWidth: true }
            
            Label {
                text: transcriptModel.isRecording ? "Recording..." : "Ready"
                color: transcriptModel.isRecording ? "red" : "green"
            }
        }
        
        TabBar {
            id: tabBar
            Layout.fillWidth: true
            
            TabButton {
                text: "Transcript"
            }
            TabButton {
                text: "Summary"
            }
        }
        
        StackLayout {
            currentIndex: tabBar.currentIndex
            Layout.fillWidth: true
            Layout.fillHeight: true
            
            ScrollView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                
                TextArea {
                    text: transcriptModel.text
                    readOnly: true
                    wrapMode: TextEdit.Wrap
                    selectByMouse: true
                    background: Rectangle {
                        color: "#f0f0f0"
                    }
                }
            }
            
            ScrollView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                
                TextArea {
                    text: summaryModel.text
                    readOnly: true
                    wrapMode: TextEdit.Wrap
                    selectByMouse: true
                    background: Rectangle {
                        color: "#f0f0f0"
                    }
                }
            }
        }
    }
}
            """)
        
        logger.info(f"Created placeholder QML file: {main_qml}")
    
    def show(self) -> None:
        """Show the main window."""
        # The QML engine will show the window automatically
        pass
    
    # Slots for QML
    
    @Slot()
    def startTranscription(self) -> None:
        """Start streaming transcription from microphone."""
        try:
            # Clear current transcript
            self.transcript_model.clear()
            
            # Start transcription
            transcript = self.app_controller.start_streaming_transcription()
            self._current_transcript_id = transcript.id
            
            # Update UI state
            self.transcript_model.isRecording = True
            self.transcriptionStarted.emit()
            
            logger.info("Started streaming transcription")
            
        except Exception as e:
            logger.error(f"Error starting transcription: {e}")
            self.transcriptionError.emit(str(e))
    
    @Slot()
    def stopTranscription(self) -> None:
        """Stop streaming transcription."""
        try:
            # Stop transcription
            self.app_controller.stop_streaming_transcription()
            
            # Update UI state
            self.transcript_model.isRecording = False
            self.transcriptionStopped.emit()
            
            logger.info("Stopped streaming transcription")
            
        except Exception as e:
            logger.error(f"Error stopping transcription: {e}")
            self.transcriptionError.emit(str(e))
    
    @Slot()
    def generateSummary(self) -> None:
        """Generate a summary of the current transcript."""
        try:
            # Check if we have a transcript
            if not self._current_transcript_id:
                logger.warning("No transcript to summarize")
                return
            
            # Clear current summary
            self.summary_model.clear()
            
            # Set UI state
            self.summary_model.isGenerating = True
            
            # Generate summary
            summary = self.app_controller.summarize_transcript(self._current_transcript_id)
            if summary:
                self._current_summary_id = summary.id
                logger.info(f"Started summary generation (id={summary.id})")
            else:
                logger.warning("Summary generation not available (LLM disabled?)")
                self.summary_model.isGenerating = False
                self.summaryError.emit("Summary generation not available")
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            self.summary_model.isGenerating = False
            self.summaryError.emit(str(e))
    
    @Slot(str, result=str)
    def formatTime(self, seconds: str) -> str:
        """
        Format time in seconds as MM:SS.
        
        Args:
            seconds: Time in seconds as string
            
        Returns:
            Formatted time string
        """
        try:
            secs = float(seconds)
            minutes = int(secs // 60)
            secs = int(secs % 60)
            return f"{minutes:02d}:{secs:02d}"
        except ValueError:
            return "00:00"
    
    # Signal handlers
    
    def _on_transcription_started(self, transcript_id: str) -> None:
        """
        Handle transcription started signal.
        
        Args:
            transcript_id: ID of the transcript
        """
        logger.debug(f"Transcription started: {transcript_id}")
        self._current_transcript_id = transcript_id
        self.transcript_model.isRecording = True
        self.transcriptionStarted.emit()
    
    def _on_transcription_progress(self, transcript_id: str, progress: float, message: str) -> None:
        """
        Handle transcription progress signal.
        
        Args:
            transcript_id: ID of the transcript
            progress: Progress value (0.0 to 1.0)
            message: Progress message
        """
        # Update transcript if it's the current one
        if transcript_id == self._current_transcript_id:
            transcript = self.app_controller.get_transcript(transcript_id)
            if transcript:
                self.transcript_model.update_from_transcript(transcript)
    
    def _on_transcription_complete(self, transcript_id: str) -> None:
        """
        Handle transcription complete signal.
        
        Args:
            transcript_id: ID of the completed transcript
        """
        logger.debug(f"Transcription complete: {transcript_id}")
        
        # Update transcript if it's the current one
        if transcript_id == self._current_transcript_id:
            transcript = self.app_controller.get_transcript(transcript_id)
            if transcript:
                self.transcript_model.update_from_transcript(transcript)
            
            # Update UI state
            self.transcript_model.isRecording = False
            self.transcriptionStopped.emit()
    
    def _on_transcription_failed(self, transcript_id: str, error_message: str) -> None:
        """
        Handle transcription failed signal.
        
        Args:
            transcript_id: ID of the failed transcript
            error_message: Error message
        """
        logger.error(f"Transcription failed: {transcript_id} - {error_message}")
        
        # Update UI state if it's the current transcript
        if transcript_id == self._current_transcript_id:
            self.transcript_model.isRecording = False
            self.transcriptionError.emit(error_message)
    
    def _on_summarization_started(self, summary_id: str) -> None:
        """
        Handle summarization started signal.
        
        Args:
            summary_id: ID of the summary
        """
        logger.debug(f"Summarization started: {summary_id}")
        self._current_summary_id = summary_id
        self.summary_model.isGenerating = True
        self.summaryStarted.emit()
    
    def _on_summarization_complete(self, summary_id: str) -> None:
        """
        Handle summarization complete signal.
        
        Args:
            summary_id: ID of the completed summary
        """
        logger.debug(f"Summarization complete: {summary_id}")
        
        # Update summary if it's the current one
        if summary_id == self._current_summary_id:
            summary = self.app_controller.get_summary(summary_id)
            if summary:
                self.summary_model.update_from_summary(summary)
            
            # Update UI state
            self.summary_model.isGenerating = False
            self.summaryCompleted.emit()
    
    def _on_summarization_failed(self, summary_id: str, error_message: str) -> None:
        """
        Handle summarization failed signal.
        
        Args:
            summary_id: ID of the failed summary
            error_message: Error message
        """
        logger.error(f"Summarization failed: {summary_id} - {error_message}")
        
        # Update UI state if it's the current summary
        if summary_id == self._current_summary_id:
            self.summary_model.isGenerating = False
            self.summaryError.emit(error_message)
