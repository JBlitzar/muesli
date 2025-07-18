#!/usr/bin/env python3
"""
Muesli - Offline-first, privacy-centric voice transcription and summarization desktop application.

This is the main entry point for the application, handling argument parsing,
configuration loading, and application initialization.
"""

import argparse
import datetime
import enum
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import QApplication

# Global application instance for signal handlers
_app_instance = None
logger = logging.getLogger(__name__)


# ======== Configuration Models ========


class WhisperModel(str, enum.Enum):
    """Available Whisper model sizes."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V3 = "large-v3"


class LLMProvider(str, enum.Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    NONE = "none"  # Disable LLM features


class ThemeMode(str, enum.Enum):
    """UI theme options."""

    DARK = "dark"
    LIGHT = "light"
    SYSTEM = "system"


class TranscriptionConfig(BaseModel):
    """Configuration for the transcription module."""

    model: WhisperModel = Field(
        default=WhisperModel.MEDIUM,
        description="Whisper model size to use for transcription",
    )

    model_dir: Path = Field(
        default=Path.home() / ".muesli" / "models" / "whisper",
        description="Directory where Whisper models are stored",
    )

    auto_language_detection: bool = Field(
        default=True, description="Automatically detect language of audio"
    )

    default_language: str = Field(
        default="en",
        description="Default language code (ISO 639-1) when auto-detection is disabled",
    )

    device: str = Field(
        default="auto",
        description="Device to use for inference: 'cpu', 'cuda', 'mps', or 'auto' to detect",
    )

    beam_size: int = Field(
        default=5,
        description="Beam size for decoding (higher = more accurate but slower)",
    )

    vad_filter: bool = Field(
        default=True,
        description="Use voice activity detection to filter out non-speech",
    )

    @field_validator("model_dir")
    @classmethod
    def ensure_model_dir(cls, v: Path) -> Path:
        """Ensure the model directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class LLMConfig(BaseModel):
    """Configuration for the LLM module."""

    provider: LLMProvider = Field(
        default=LLMProvider.OLLAMA, description="LLM provider to use"
    )

    ollama_host: str = Field(
        default="localhost", description="Hostname for Ollama server"
    )

    ollama_port: int = Field(default=11434, description="Port for Ollama server")

    model_name: str = Field(
        default="llama3.1:8b", description="Model name to use with the provider"
    )

    offline_mode: bool = Field(
        default=True, description="Run LLM in offline mode (no network access)"
    )

    summary_prompt_template: str = Field(
        default=(
            "Below is a transcript of an audio recording. "
            "Please provide a concise summary of the key points discussed:\n\n"
            "{transcript}\n\n"
            "Summary:"
        ),
        description="Prompt template for generating summaries",
    )

    @property
    def ollama_base_url(self) -> str:
        """Get the base URL for Ollama API."""
        return f"http://{self.ollama_host}:{self.ollama_port}"


class UIConfig(BaseModel):
    """Configuration for the UI module."""

    theme: ThemeMode = Field(default=ThemeMode.SYSTEM, description="UI theme mode")

    font_size: int = Field(default=12, description="Base font size in points")

    recent_files_limit: int = Field(
        default=10, description="Number of recent files to display"
    )

    waveform_visible: bool = Field(
        default=True, description="Show audio waveform in transcript view"
    )

    auto_scroll: bool = Field(
        default=True, description="Auto-scroll transcript during playback"
    )


class PrivacyConfig(BaseModel):
    """Configuration for privacy settings."""

    telemetry_enabled: bool = Field(
        default=False, description="Enable anonymous usage telemetry"
    )

    allow_network: bool = Field(
        default=False,
        description="Allow network access for model downloads and updates",
    )

    model_hash_verification: bool = Field(
        default=True, description="Verify hash of downloaded models"
    )

    log_redaction: bool = Field(
        default=True, description="Redact potentially sensitive information from logs"
    )


class AppConfig(BaseModel):
    """Main application configuration."""

    model_config = ConfigDict(
        extra="forbid",  # Forbid extra fields not defined in the model
    )

    # Application metadata
    app_name: str = Field(default="Muesli", description="Application name")

    app_version: str = Field(default="0.1.0", description="Application version")

    # Module configurations
    transcription: TranscriptionConfig = Field(
        default_factory=TranscriptionConfig,
        description="Transcription module configuration",
    )

    llm: LLMConfig = Field(
        default_factory=LLMConfig, description="LLM module configuration"
    )

    ui: UIConfig = Field(
        default_factory=UIConfig, description="UI module configuration"
    )

    privacy: PrivacyConfig = Field(
        default_factory=PrivacyConfig, description="Privacy settings"
    )

    # Application behavior
    auto_transcribe: bool = Field(
        default=True,
        description="Automatically start transcription when audio is loaded",
    )

    auto_summarize: bool = Field(
        default=True,
        description="Automatically generate summary when transcription completes",
    )

    data_dir: Path = Field(
        default=Path.home() / ".muesli" / "data",
        description="Directory for application data",
    )

    cache_dir: Path = Field(
        default=Path.home() / ".muesli" / "cache",
        description="Directory for application cache",
    )

    @model_validator(mode="after")
    def ensure_directories(self) -> "AppConfig":
        """Ensure all required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self


# ======== Data Models ========


class AudioFormat(str, enum.Enum):
    """Supported audio file formats."""

    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    FLAC = "flac"
    OGG = "ogg"

    @classmethod
    def from_path(cls, path: Path) -> "AudioFormat":
        """Determine audio format from file extension."""
        ext = path.suffix.lower().lstrip(".")
        try:
            return cls(ext)
        except ValueError:
            raise ValueError(f"Unsupported audio format: {ext}")


class AudioFile(BaseModel):
    """Represents an audio file that can be transcribed."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    path: Path = Field(..., description="Path to the audio file")

    format: AudioFormat = Field(..., description="Audio file format")

    duration: Optional[float] = Field(
        default=None, description="Duration of audio in seconds"
    )

    sample_rate: Optional[int] = Field(default=None, description="Sample rate in Hz")

    channels: Optional[int] = Field(
        default=None, description="Number of audio channels"
    )

    bit_depth: Optional[int] = Field(default=None, description="Bit depth of audio")

    file_size: Optional[int] = Field(default=None, description="File size in bytes")

    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="When this record was created",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the audio file"
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        """Validate that the audio file exists."""
        if not v.exists() and str(v) != "stream":
            raise ValueError(f"Audio file does not exist: {v}")
        if v.exists() and not v.is_file():
            raise ValueError(f"Path is not a file: {v}")
        return v

    @classmethod
    def from_path(cls, path: Path) -> "AudioFile":
        """Create an AudioFile instance from a path."""
        path = Path(path).absolute()
        format = AudioFormat.from_path(path)

        # Get basic file info
        file_size = path.stat().st_size if path.exists() else 0

        return cls(
            path=path,
            format=format,
            file_size=file_size,
        )

    def exists(self) -> bool:
        """Check if the audio file still exists."""
        return self.path.exists() or str(self.path) == "stream"


class Transcript(BaseModel):
    """Represents a transcript of an audio file."""

    id: str = Field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
        description="Unique identifier for this transcript",
    )

    audio_file: AudioFile = Field(
        ..., description="The audio file this transcript is for"
    )

    text: str = Field(default_factory=str, description="Full transcript text")

    notes: str = Field(default_factory=str, description="User notes entered during recording")

    language: str = Field(
        default="en", description="Detected or specified language code (ISO 639-1)"
    )

    model_name: str = Field(
        default="", description="Name of the model used for transcription"
    )

    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="When this transcript was created",
    )

    updated_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="When this transcript was last updated",
    )

    is_complete: bool = Field(
        default=False, description="Whether the transcription is complete"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the transcript"
    )

    # No segment-level helpers needed for plain-text transcripts.


class SummaryType(str, enum.Enum):
    """Types of summaries that can be generated."""

    BULLET_POINTS = "bullet_points"
    PARAGRAPH = "paragraph"
    EXECUTIVE = "executive"
    DETAILED = "detailed"


class Summary(BaseModel):
    """Represents a summary of a transcript."""

    id: str = Field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
        description="Unique identifier for this summary",
    )

    transcript_id: str = Field(
        ..., description="ID of the transcript this summary is for"
    )

    text: str = Field(..., description="The summary text")

    summary_type: SummaryType = Field(
        default=SummaryType.PARAGRAPH, description="Type of summary generated"
    )

    model_name: str = Field(
        default="", description="Name of the LLM model used for summarization"
    )

    created_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="When this summary was created",
    )

    prompt_template: Optional[str] = Field(
        default=None, description="The prompt template used to generate this summary"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the summary"
    )


# ======== Configuration Functions ========


def load_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """
    Load application configuration from a YAML file with environment variable overrides.

    Args:
        config_path: Path to configuration file (optional)

    Returns:
        Validated AppConfig instance
    """
    logger = logging.getLogger(__name__)

    # Default configuration
    config_data: Dict[str, Any] = {}

    # Standard config locations
    standard_locations = [
        Path.cwd() / "muesli.yaml",
        Path.cwd() / "muesli.yml",
        Path.home() / ".muesli" / "config.yaml",
        Path.home() / ".muesli" / "config.yml",
    ]

    # If config_path is provided, try to load it
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}")
        else:
            try:
                with open(config_file, "r") as f:
                    config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading config from {config_file}: {e}")
    else:
        # Try standard locations
        for loc in standard_locations:
            if loc.exists():
                try:
                    with open(loc, "r") as f:
                        config_data = yaml.safe_load(f) or {}
                    logger.info(f"Loaded configuration from {loc}")
                    break
                except Exception as e:
                    logger.error(f"Error loading config from {loc}: {e}")

    # Override with environment variables
    # Format: MUESLI_SECTION_KEY=value (e.g., MUESLI_TRANSCRIPTION_MODEL=small)
    env_prefix = "MUESLI_"
    for env_var, value in os.environ.items():
        if env_var.startswith(env_prefix):
            parts = env_var[len(env_prefix) :].lower().split("_", 1)
            if len(parts) == 2:
                section, key = parts

                # Handle nested sections
                if section not in config_data:
                    config_data[section] = {}

                # Convert value types
                if value.lower() in ("true", "yes", "1"):
                    value = True
                elif value.lower() in ("false", "no", "0"):
                    value = False
                elif value.isdigit():
                    value = int(value)

                config_data[section][key] = value
                logger.debug(
                    f"Config override from environment: {section}.{key}={value}"
                )

    # Create and validate config
    try:
        return AppConfig(**config_data)
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        logger.warning("Falling back to default configuration")
        return AppConfig()


# ======== Main Application Class ========


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

    class _BackgroundWorker(QRunnable):
        """Worker class for running tasks in background threads."""

        def __init__(self, fn, *args, **kwargs):
            super().__init__()
            self.fn = fn
            self.args = args
            self.kwargs = kwargs

        def run(self):
            """Execute the function in a background thread."""
            try:
                self.fn(*self.args, **self.kwargs)
            except Exception as e:
                logger.exception(f"Background task failed: {e}")

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

        # UI components (initialized lazily)
        self._app: Optional[QApplication] = None
        self._main_window = None

        # Active transcripts and summaries
        self._active_transcripts: Dict[str, Transcript] = {}
        self._active_summaries: Dict[str, Summary] = {}

        # Thread synchronization
        self._lock = threading.RLock()

        # Thread pool for background tasks
        self._thread_pool = QThreadPool.globalInstance()

        # Initialize components
        self._init_components()

        logger.info(
            f"Muesli application initialized (version {self.config.app_version})"
        )

    def _setup_directories(self) -> None:
        """Set up the necessary directories for the application."""
        # Ensure data directories exist
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        # Ensure model directories exist
        self.config.transcription.model_dir.mkdir(parents=True, exist_ok=True)

    def _init_components(self) -> None:
        """Initialize application components based on configuration."""
        # Import here to avoid circular imports
        from stream_processor import TranscriptionStreamProcessor
        from whisper_wrapper import WhisperTranscriber

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
            from ollama_client import OllamaClient
            from summarizer import TranscriptSummarizer

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

    def start_ui(self) -> int:
        """
        Start the UI application.

        Returns:
            Exit code from the application
        """
        # Import UI components here to avoid circular imports
        from ui.main_window import MainWindow

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

        # Close any open resources
        if hasattr(self, "transcriber"):
            self.transcriber.close()

        import glob
        import os

        for file in glob.glob(os.path.expanduser("~/.muesli/recordings/*")):
            os.remove(file)

        for file in glob.glob(os.path.expanduser("~/.muesli/logs/*")):
            os.remove(file)

        for file in glob.glob(os.path.expanduser("~/.muesli/output/*")):
            os.remove(file)

        if hasattr(self, "llm_client") and self.llm_client:
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

        # Emit signal that transcription has started
        self.transcription_started.emit(transcript.id)

        # Define progress callback
        def progress_callback(progress: float, message: Optional[str] = None) -> None:
            self.transcription_progress.emit(transcript.id, progress, message or "")
            if on_progress:
                on_progress(progress, message)

        # Define the transcription task to run in background
        def transcription_task():
            try:
                # Perform transcription
                result = self.transcriber.transcribe(
                    audio_path=str(audio_path),
                    language=language,
                    auto_detect_language=auto_detect_language,
                    progress_callback=progress_callback,
                )

                # Update transcript
                with self._lock:
                    transcript.text = result.text
                    transcript.language = result.language
                    transcript.model_name = result.model_name
                    transcript.is_complete = True
                    transcript.metadata.update(result.metadata)

                # Emit completion signal
                self.transcription_complete.emit(transcript.id)

                # Auto-summarize if configured
                if self.config.auto_summarize and self.summarizer:
                    self.summarize_transcript(transcript)

            except Exception as e:
                logger.error(f"Transcription failed: {e}")
                self.transcription_failed.emit(transcript.id, str(e))

        # Start the transcription task in a background thread
        worker = self._BackgroundWorker(transcription_task)
        self._thread_pool.start(worker)

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
            path=Path("stream"), format="wav", metadata={"device_index": device_index}
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
            on_transcription_update=self._on_transcription_update,
            on_error=lambda e: self.transcription_failed.emit(transcript.id, str(e)),
        )

        # Emit signal
        self.transcription_started.emit(transcript.id)

        return transcript

    def stop_streaming_transcription(self) -> None:
        """Stop streaming transcription."""
        self.stream_processor.stop()

    def _on_transcription_update(self, transcript_id: str, full_text: str) -> None:
        """
        Handle transcript text updates from real-time transcription.

        Args:
            transcript_id: ID of the transcript being updated
            full_text: The current full transcript text
        """
        with self._lock:
            transcript = self._active_transcripts.get(transcript_id)
            if transcript:
                transcript.text = full_text
                # Emit progress as unknown (use 0-1 dummy value); here set to 1.0 when update received.
                self.transcription_progress.emit(
                    transcript_id, 1.0, f"Transcript updated ({len(full_text)} chars)"
                )

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

    def update_transcript_notes(self, transcript_id: str, notes: str) -> bool:
        """
        Update notes for a transcript.

        Args:
            transcript_id: ID of the transcript
            notes: Notes text to update

        Returns:
            True if successful, False if transcript not found
        """
        with self._lock:
            transcript = self._active_transcripts.get(transcript_id)
            if transcript:
                transcript.notes = notes
                return True
            return False

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

        # Emit signal
        self.summarization_started.emit(summary.id)

        # Define the summarization task to run in background
        def summarization_task():
            try:
                # Generate summary
                summary_text = self.summarizer.summarize(
                    transcript=transcript_obj,
                    prompt_template=prompt_template,
                )

                # Update summary
                with self._lock:
                    summary.text = summary_text
                    summary.model_name = self.config.llm.model_name

                # Emit completion signal
                self.summarization_complete.emit(summary.id)

            except Exception as e:
                logger.error(f"Summarization failed: {e}")
                self.summarization_failed.emit(summary.id, str(e))

        # Start the summarization task in a background thread
        worker = self._BackgroundWorker(summarization_task)
        self._thread_pool.start(worker)

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


# ======== Signal Handlers ========


def signal_handler(sig, frame):
    """
    Handle signals for graceful shutdown.

    Args:
        sig: Signal number
        frame: Current stack frame
    """
    logger = logging.getLogger(__name__)
    signal_name = signal.Signals(sig).name

    logger.info(f"Received {signal_name}, shutting down gracefully...")

    # Clean up resources
    if _app_instance is not None:
        _app_instance.shutdown()

    sys.exit(0)


def setup_signal_handlers():
    """Register signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request

    # Handle SIGBREAK on Windows (Ctrl+Break)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal_handler)


# ======== Logging Setup ========


def setup_logging(verbose: bool = False) -> None:
    """
    Configure the logging system for the application.

    Args:
        verbose: Whether to enable verbose logging (DEBUG level)
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Ensure logs directory exists
    log_dir = Path.home() / ".muesli" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "muesli.log"

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    # Reduce verbosity of third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("PySide6").setLevel(logging.WARNING)


# ======== Command Line Interface ========


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments.

    Args:
        args: Command line arguments (uses sys.argv if None)

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Muesli - Offline-first, privacy-centric voice transcription and summarization"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "-c", "--config", type=str, help="Path to custom config file", default=None
    )

    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Run in headless mode (for CLI operations only)",
    )

    parser.add_argument(
        "--transcribe", type=str, help="Path to audio file to transcribe directly"
    )

    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the application.

    Args:
        args: Command line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    global _app_instance

    parsed_args = parse_args(args)

    # Set up logging
    setup_logging(parsed_args.verbose)
    logger = logging.getLogger(__name__)
    logger.info("Starting Muesli application")

    # Set up signal handlers
    setup_signal_handlers()

    # Initialize the application
    app = None

    try:
        # Load configuration
        config_path = parsed_args.config
        config = load_config(config_path)

        # Initialize the application
        app = MuesliApp(config)
        _app_instance = app  # Store globally for signal handlers

        if parsed_args.transcribe:
            # CLI mode: transcribe a single file
            audio_path = Path(parsed_args.transcribe)
            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_path}")
                return 1

            logger.info(f"Transcribing file: {audio_path}")
            result = app.transcribe_file(audio_path)

            # Wait for transcription to complete
            while not result.is_complete:
                time.sleep(0.5)

            print(result.text)

            if config.auto_summarize:
                summary = app.summarize_transcript(result)
                if summary:
                    # Wait for summary to be generated
                    while not summary.text:
                        time.sleep(0.5)
                    print("\nSummary:")
                    print(summary.text)

            return 0

        elif not parsed_args.no_ui:
            # Start the UI
            logger.info("Launching UI")
            return app.start_ui()
        else:
            logger.error("No action specified in headless mode")
            return 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down...")
        return 0
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        return 1
    finally:
        # Clean up resources
        if app is not None and app is not _app_instance:
            # Only clean up if the app wasn't already cleaned up by signal handler
            logger.info("Cleaning up resources...")
            app.shutdown()


# ======== Entry Point ========

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    sys.exit(main())
