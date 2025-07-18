"""
Stream processor for audio recording and transcription.

This module provides functionality for capturing audio from a microphone
and processing it to generate transcriptions using the WhisperTranscriber class.
"""

import audioop
import logging
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Optional, Union

# Note: In a real implementation, we would use PyAudio or a similar library
# for microphone input. This is a placeholder implementation.
try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logging.warning("PyAudio not available. Microphone input will not work.")

import numpy as np

from whisper_wrapper import WhisperTranscriber

logger = logging.getLogger(__name__)


class TranscriptionStreamProcessor:
    """
    Processor for audio recording and transcription.

    This class captures audio from a microphone, saves it to a file when stopped,
    and then transcribes the entire recording at once.
    """

    # Audio settings
    SAMPLE_RATE = 16000  # Hz
    CHUNK_SIZE = 1024  # samples
    FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else 8  # 16-bit audio
    CHANNELS = 1  # mono

    # VAD (Voice Activity Detection) settings
    VAD_THRESHOLD = 300  # RMS threshold for voice activity

    def __init__(
        self,
        transcriber: WhisperTranscriber,
        vad_filter: bool = True,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the stream processor.

        Args:
            transcriber: WhisperTranscriber instance for transcription
            vad_filter: Whether to use voice activity detection to filter silence
            output_dir: Directory for output files (uses ~/.muesli/recordings if None)
        """
        self.transcriber = transcriber
        self.vad_filter = vad_filter
        self.output_dir = output_dir or Path.home() / ".muesli" / "recordings"

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Internal state
        self._pyaudio = None
        self._stream = None
        self._is_recording = False
        self._stop_event = threading.Event()

        # Audio buffer
        self._audio_frames = []
        self._buffer_lock = threading.Lock()

        # Recording thread
        self._recording_thread = None

        # Callbacks
        self._on_transcription_complete = None
        # New callback for plain-text transcript updates
        self._on_transcription_update = None
        self._on_error = None

        # Current transcript
        self._transcript_id = None
        self._language = None

        # Initialize PyAudio if available
        if PYAUDIO_AVAILABLE:
            self._pyaudio = pyaudio.PyAudio()

    def start(
        self,
        transcript_id: str,
        device_index: Optional[int] = None,
        language: Optional[str] = None,
        on_transcription_update: Optional[Callable[[str, str], None]] = None,
        on_transcription_complete: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> bool:
        """
        Start recording audio from microphone.

        Args:
            transcript_id: ID of the transcript to update
            device_index: Index of audio input device or None for default
            language: Language code (ISO 639-1) or None for auto-detection
            on_transcription_update: Callback when transcript text is available/
                updated. Called with (transcript_id, full_text).
            on_transcription_complete: Callback when transcription is complete
            on_error: Callback for errors

        Returns:
            True if started successfully, False otherwise
        """
        if not PYAUDIO_AVAILABLE:
            error = RuntimeError("PyAudio not available. Cannot start recording.")
            if on_error:
                on_error(error)
            else:
                logger.error(str(error))
            return False

        if self._is_recording:
            logger.warning("Stream processor already recording. Stop it first.")
            return False

        # Store parameters
        self._transcript_id = transcript_id
        self._on_transcription_complete = on_transcription_complete
        self._on_error = on_error
        self._on_transcription_update = on_transcription_update
        self._language = language

        # Reset state
        self._stop_event.clear()
        self._audio_frames = []

        try:
            # Open audio stream
            self._stream = self._pyaudio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK_SIZE,
            )

            # Start recording thread
            self._is_recording = True
            self._recording_thread = threading.Thread(
                target=self._record_audio_thread,
                daemon=True,
            )
            self._recording_thread.start()

            logger.info(f"Started recording (transcript_id={transcript_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            if self._on_error:
                self._on_error(e)
            return False

    def stop(self) -> None:
        """
        Stop recording and start transcription.

        This method stops the recording, saves the audio to a file,
        and starts the transcription process.
        """
        if not self._is_recording:
            return

        logger.info("Stopping recording")
        self._stop_event.set()

        # Wait for recording thread to finish
        if self._recording_thread and self._recording_thread.is_alive():
            self._recording_thread.join(timeout=2.0)

        # Stop audio stream
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        self._is_recording = False

        # Process the recorded audio
        self._process_recording()

        logger.info("Recording stopped and processing complete")

    def is_recording(self) -> bool:
        """Check if recording is currently active."""
        return self._is_recording

    def _record_audio_thread(self):
        """Thread function for recording audio."""
        try:
            logger.info("Recording thread started")

            while not self._stop_event.is_set():
                # Read audio data
                data = self._stream.read(self.CHUNK_SIZE, exception_on_overflow=False)

                # Check for voice activity if VAD is enabled
                if self.vad_filter:
                    rms = audioop.rms(data, 2)  # 2 bytes per sample for paInt16
                    if rms < self.VAD_THRESHOLD:
                        # Skip silence
                        continue

                # Add to buffer
                with self._buffer_lock:
                    self._audio_frames.append(data)

        except Exception as e:
            logger.error(f"Error in recording thread: {e}")
            if self._on_error:
                self._on_error(e)
            self._stop_event.set()

    def _process_recording(self) -> None:
        """
        Process the recorded audio.

        This method saves the recorded audio to a file and transcribes it.
        """
        with self._buffer_lock:
            if not self._audio_frames:
                logger.warning("No audio recorded")
                return

            # Create a unique filename
            timestamp = int(time.time())
            audio_path = self.output_dir / f"recording_{timestamp}.wav"

            # Save audio to file
            self._save_wav_file(audio_path, self._audio_frames)

            logger.info(f"Saved recording to {audio_path}")

        try:
            # Transcribe the audio
            logger.info(f"Transcribing recording {audio_path}")
            transcript = self.transcriber.transcribe(
                audio_path=audio_path,
                language=self._language,
                auto_detect_language=self._language is None,
                progress_callback=lambda progress, message: logger.debug(
                    f"Transcription progress: {progress:.1%} - {message}"
                ),
            )

            logger.info(f"Transcription complete: {len(transcript.text)} characters")

            # Update callback with full transcript text
            if self._on_transcription_update:
                try:
                    self._on_transcription_update(self._transcript_id, transcript.text)
                except Exception as cb_err:
                    logger.error(f"on_transcription_update callback error: {cb_err}")

            # Call completion callback
            if self._on_transcription_complete:
                self._on_transcription_complete(self._transcript_id)

        except Exception as e:
            logger.error(f"Error transcribing recording: {e}")
            if self._on_error:
                self._on_error(e)

    def _save_wav_file(self, path: Path, frames) -> None:
        """
        Save audio frames to a WAV file.

        Args:
            path: Path to save the WAV file
            frames: List of audio frames
        """
        try:
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self._pyaudio.get_sample_size(self.FORMAT))
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(b"".join(frames))

            logger.debug(f"Saved WAV file: {path}")
        except Exception as e:
            logger.error(f"Error saving WAV file: {e}")
            if self._on_error:
                self._on_error(e)

    def __del__(self):
        """Clean up resources."""
        self.stop()

        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
