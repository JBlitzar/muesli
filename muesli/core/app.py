"""
Core application class for the Muesli application.

This module provides the MuesliApp class, which serves as the main orchestrator
for the application. It initializes all components, handles configuration,
and provides the main API for the application.
"""

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QApplication

from muesli.core.config import AppConfig, load_config
from muesli.core.job_queue import Job, JobQueue, JobStatus, JobType
from muesli.core.models import AudioFile, Summary, Transcript, TranscriptSegment
from muesli.llm.ollama_client import OllamaClient
from muesli.llm.summarizer import TranscriptSummarizer
from muesli.transcription.stream_processor import TranscriptionStreamProcessor
from muesli.transcription.whisper_wrapper import WhisperTranscriber
from muesli.ui.main_window import MainWindow


logger = logging.getLogger(__name__)


class MuesliApp(QObject):
    """
    Main application class for Muesli.
    
    This class serves as the central orchestrator for the application,
    initializing all components and providing the main API.
    """
    
    # Signals for UI updates
    transcription_started = Signal(str)  # transcript_id
    transcription_progress = Signal(str, float, str)  # transcript_id, progress, message
    transcription_complete = Signal(str)  # transcript_id
    transcription_failed = Signal(str, str)  # transcript_id, error_message
    
    summarization_started = Signal(str)  # summary_id
    summarization_complete = Signal(str)  # summary_id
    summarization_failed = Signal(str, str)  # summary_id, error_message
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the application.
        
        Args:
            config: Application configuration
        """
        super().__init__()
        
        # Load configuration if not provided
        self.config = config or load_config()
        
        # Set up data directories
        self._setup_directories()
        
        # Initialize components
        self._init_components()
        
        # Initialize job queue
        self.job_queue = self._create_job_queue()
        
        # UI components (initialized lazily)
        self._app: Optional[QApplication] = None
        self._main_window: Optional[MainWindow] = None
        
        # Active transcripts and summaries
        self._active_transcripts: Dict[str, Transcript] = {}
        self._active_summaries: Dict[str, Summary] = {}
        
        # Thread synchronization
        self._lock = threading.RLock()
        
        logger.info(f"Muesli application initialized (version {self.config.app_version})")
    
    def _setup_directories(self) -> None:
        """Set up the necessary directories for the application."""
        # Ensure data directories exist
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure model directories exist
        self.config.transcription.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create database directory if needed
        self.config.database.path.parent.mkdir(parents=True, exist_ok=True)
    
    def _init_components(self) -> None:
        """Initialize application components based on configuration."""
        # Initialize transcription component
        self.transcriber = WhisperTranscriber(
            model_name=self.config.transcription.model.value,
            model_dir=self.config.transcription.model_dir,
            device=self.config.transcription.device,
            beam_size=self.config.transcription.beam_size,
        )
        
        # Initialize stream processor for real-time transcription
        self.stream_processor = TranscriptionStreamProcessor(
            transcriber=self.transcriber,
            vad_filter=self.config.transcription.vad_filter,
        )
        
        # Initialize LLM client if enabled
        if self.config.llm.provider == "ollama":
            self.llm_client = OllamaClient(
                base_url=self.config.llm.ollama_base_url,
                model_name=self.config.llm.model_name,
                offline_mode=self.config.llm.offline_mode,
            )
            
            # Initialize summarizer
            self.summarizer = TranscriptSummarizer(
                llm_client=self.llm_client,
                prompt_template=self.config.llm.summary_prompt_template,
            )
        else:
            self.llm_client = None
            self.summarizer = None
            logger.info("LLM features disabled (provider set to 'none')")
    
    def _create_job_queue(self) -> JobQueue:
        """Create and configure the job queue."""
        # Create a job queue with handlers for our job types
        job_queue = _MuesliJobQueue(max_workers=4)
        
        # Register handlers for job types
        job_queue.register_handler(JobType.TRANSCRIPTION, self._handle_transcription_job)
        job_queue.register_handler(JobType.SUMMARIZATION, self._handle_summarization_job)
        
        return job_queue
    
    def start_ui(self) -> int:
        """
        Start the UI application.
        
        Returns:
            Exit code from the application
        """
        # Create QApplication if not already created
        if self._app is None:
            self._app = QApplication(sys.argv)
            self._app.setApplicationName(self.config.app_name)
            self._app.setApplicationVersion(self.config.app_version)
        
        # Create main window if not already created
        if self._main_window is None:
            self._main_window = MainWindow(self)
        
        # Show the main window
        self._main_window.show()
        
        # Run the application
        return self._app.exec()
    
    def shutdown(self) -> None:
        """Shut down the application and clean up resources."""
        logger.info("Shutting down Muesli application")
        
        # Shut down job queue
        if hasattr(self, 'job_queue'):
            self.job_queue.shutdown(wait=True)
        
        # Close any open resources
        if hasattr(self, 'transcriber'):
            self.transcriber.close()
        
        if hasattr(self, 'llm_client') and self.llm_client:
            self.llm_client.close()
    
    # Transcription methods
    
    def transcribe_file(
        self, 
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        auto_detect_language: Optional[bool] = None,
        on_progress: Optional[callable] = None,
    ) -> Transcript:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (ISO 639-1) or None for auto-detection
            auto_detect_language: Whether to auto-detect language (overrides config)
            on_progress: Callback for progress updates
            
        Returns:
            Transcript object
        """
        # Convert to Path if string
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)
        
        # Create AudioFile object
        audio_file = AudioFile.from_path(audio_path)
        
        # Create empty transcript
        transcript = Transcript(audio_file=audio_file)
        
        # Determine language settings
        if auto_detect_language is None:
            auto_detect_language = self.config.transcription.auto_language_detection
        
        if not auto_detect_language and language is None:
            language = self.config.transcription.default_language
        
        # Add to active transcripts
        with self._lock:
            self._active_transcripts[transcript.id] = transcript
        
        # Create job parameters
        params = {
            "transcript_id": transcript.id,
            "audio_path": str(audio_path),
            "language": language,
            "auto_detect_language": auto_detect_language,
        }
        
        # Define callbacks
        def on_complete(result: Transcript) -> None:
            self.transcription_complete.emit(result.id)
            
            # Auto-summarize if configured
            if self.config.auto_summarize and self.summarizer:
                self.summarize_transcript(result)
        
        def on_error(error: Exception) -> None:
            self.transcription_failed.emit(transcript.id, str(error))
        
        def on_progress_update(progress: float, message: Optional[str]) -> None:
            self.transcription_progress.emit(transcript.id, progress, message or "")
            if on_progress:
                on_progress(progress, message)
        
        # Add job to queue
        self.job_queue.add_job(
            job_type=JobType.TRANSCRIPTION,
            params=params,
            on_complete=on_complete,
            on_error=on_error,
            on_progress=on_progress_update,
        )
        
        # Emit signal
        self.transcription_started.emit(transcript.id)
        
        return transcript
    
    def start_streaming_transcription(
        self,
        device_index: Optional[int] = None,
        language: Optional[str] = None,
    ) -> Transcript:
        """
        Start streaming transcription from microphone.
        
        Args:
            device_index: Index of audio input device or None for default
            language: Language code (ISO 639-1) or None for auto-detection
            
        Returns:
            Transcript object for the stream
        """
        # Create a dummy AudioFile for the stream
        audio_file = AudioFile(
            path=Path("stream"),
            format="wav",
            metadata={"device_index": device_index}
        )
        
        # Create empty transcript
        transcript = Transcript(audio_file=audio_file)
        
        # Add to active transcripts
        with self._lock:
            self._active_transcripts[transcript.id] = transcript
        
        # Start streaming
        self.stream_processor.start(
            transcript_id=transcript.id,
            device_index=device_index,
            language=language or self.config.transcription.default_language,
            on_segment=self._on_stream_segment,
            on_error=lambda e: self.transcription_failed.emit(transcript.id, str(e)),
        )
        
        # Emit signal
        self.transcription_started.emit(transcript.id)
        
        return transcript
    
    def stop_streaming_transcription(self) -> None:
        """Stop streaming transcription."""
        self.stream_processor.stop()
    
    def _on_stream_segment(self, transcript_id: str, segment: TranscriptSegment) -> None:
        """
        Handle a new segment from streaming transcription.
        
        Args:
            transcript_id: ID of the transcript
            segment: New transcript segment
        """
        with self._lock:
            transcript = self._active_transcripts.get(transcript_id)
            if transcript:
                transcript.add_segment(segment)
                # Signal could be added here to update UI with new segment
    
    def get_transcript(self, transcript_id: str) -> Optional[Transcript]:
        """
        Get a transcript by ID.
        
        Args:
            transcript_id: ID of the transcript
            
        Returns:
            Transcript object or None if not found
        """
        with self._lock:
            return self._active_transcripts.get(transcript_id)
    
    # Summarization methods
    
    def summarize_transcript(
        self, 
        transcript: Union[Transcript, str],
        prompt_template: Optional[str] = None,
    ) -> Optional[Summary]:
        """
        Generate a summary for a transcript.
        
        Args:
            transcript: Transcript object or transcript ID
            prompt_template: Custom prompt template or None to use default
            
        Returns:
            Summary object or None if summarization is disabled
        """
        # Check if LLM is enabled
        if not self.summarizer:
            logger.warning("Summarization requested but LLM is disabled")
            return None
        
        # Get transcript object if ID was provided
        if isinstance(transcript, str):
            transcript_obj = self.get_transcript(transcript)
            if not transcript_obj:
                logger.error(f"Transcript not found: {transcript}")
                return None
        else:
            transcript_obj = transcript
        
        # Create empty summary
        summary = Summary(
            transcript_id=transcript_obj.id,
            text="",
            prompt_template=prompt_template or self.config.llm.summary_prompt_template,
        )
        
        # Add to active summaries
        with self._lock:
            self._active_summaries[summary.id] = summary
        
        # Create job parameters
        params = {
            "summary_id": summary.id,
            "transcript_id": transcript_obj.id,
            "prompt_template": prompt_template,
        }
        
        # Define callbacks
        def on_complete(result: Summary) -> None:
            self.summarization_complete.emit(result.id)
        
        def on_error(error: Exception) -> None:
            self.summarization_failed.emit(summary.id, str(error))
        
        # Add job to queue
        self.job_queue.add_job(
            job_type=JobType.SUMMARIZATION,
            params=params,
            on_complete=on_complete,
            on_error=on_error,
        )
        
        # Emit signal
        self.summarization_started.emit(summary.id)
        
        return summary
    
    def get_summary(self, summary_id: str) -> Optional[Summary]:
        """
        Get a summary by ID.
        
        Args:
            summary_id: ID of the summary
            
        Returns:
            Summary object or None if not found
        """
        with self._lock:
            return self._active_summaries.get(summary_id)
    
    # Job handlers
    
    def _handle_transcription_job(self, job: Job) -> Transcript:
        """
        Handle a transcription job.
        
        Args:
            job: Transcription job
            
        Returns:
            Updated transcript
        """
        params = job.params
        transcript_id = params["transcript_id"]
        audio_path = params["audio_path"]
        language = params.get("language")
        auto_detect_language = params.get("auto_detect_language", self.config.transcription.auto_language_detection)
        
        # Get transcript
        with self._lock:
            transcript = self._active_transcripts.get(transcript_id)
            if not transcript:
                raise ValueError(f"Transcript not found: {transcript_id}")
        
        # Progress callback
        def progress_callback(progress: float, message: Optional[str] = None) -> None:
            job.update_progress(progress, message)
        
        # Perform transcription
        result = self.transcriber.transcribe(
            audio_path=audio_path,
            language=language,
            auto_detect_language=auto_detect_language,
            progress_callback=progress_callback,
        )
        
        # Update transcript
        with self._lock:
            transcript.segments = result.segments
            transcript.language = result.language
            transcript.model_name = result.model_name
            transcript.is_complete = True
            transcript.metadata.update(result.metadata)
        
        return transcript
    
    def _handle_summarization_job(self, job: Job) -> Summary:
        """
        Handle a summarization job.
        
        Args:
            job: Summarization job
            
        Returns:
            Updated summary
        """
        params = job.params
        summary_id = params["summary_id"]
        transcript_id = params["transcript_id"]
        prompt_template = params.get("prompt_template")
        
        # Get summary and transcript
        with self._lock:
            summary = self._active_summaries.get(summary_id)
            if not summary:
                raise ValueError(f"Summary not found: {summary_id}")
            
            transcript = self._active_transcripts.get(transcript_id)
            if not transcript:
                raise ValueError(f"Transcript not found: {transcript_id}")
        
        # Generate summary
        summary_text = self.summarizer.summarize(
            transcript=transcript,
            prompt_template=prompt_template,
        )
        
        # Update summary
        with self._lock:
            summary.text = summary_text
            summary.model_name = self.config.llm.model_name
        
        return summary


class _MuesliJobQueue(JobQueue):
    """
    Extended JobQueue with handlers for Muesli-specific job types.
    """
    
    def __init__(self, max_workers: int = 4):
        """Initialize the job queue."""
        super().__init__(max_workers=max_workers)
        self._handlers: Dict[JobType, callable] = {}
    
    def register_handler(self, job_type: JobType, handler: callable) -> None:
        """
        Register a handler for a job type.
        
        Args:
            job_type: Type of job
            handler: Handler function
        """
        self._handlers[job_type] = handler
    
    def _get_job_handler(self, job_type: JobType) -> Optional[callable]:
        """
        Get the handler for a job type.
        
        Args:
            job_type: Type of job
            
        Returns:
            Handler function or None if not registered
        """
        return self._handlers.get(job_type)
