"""
Stream processor for real-time transcription from microphone input.

This module provides functionality for capturing audio from a microphone
and processing it in real-time to generate transcriptions using the
WhisperTranscriber class.
"""

import audioop
import collections
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Deque

# Note: In a real implementation, we would use PyAudio or a similar library
# for microphone input. This is a placeholder implementation.
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logging.warning("PyAudio not available. Microphone input will not work.")

import numpy as np

from muesli.core.models import TranscriptSegment
from muesli.transcription.whisper_wrapper import WhisperTranscriber

logger = logging.getLogger(__name__)


class TranscriptionStreamProcessor:
    """
    Processor for real-time transcription from microphone input.
    
    This class captures audio from a microphone, processes it in chunks,
    and generates transcriptions in real-time using the WhisperTranscriber.
    """
    
    # Audio settings
    SAMPLE_RATE = 16000  # Hz
    CHUNK_SIZE = 1024  # samples
    FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else 8  # 16-bit audio
    CHANNELS = 1  # mono
    
    # VAD (Voice Activity Detection) settings
    VAD_THRESHOLD = 300  # RMS threshold for voice activity
    VAD_SILENCE_DURATION = 1.0  # seconds of silence to consider end of speech
    VAD_SPEECH_DURATION = 0.3  # seconds of speech to consider start of speech
    
    # Processing settings
    INFERENCE_CHUNK_DURATION = 5.0  # seconds of audio to process at once
    MAX_BUFFER_DURATION = 60.0  # maximum seconds of audio to keep in buffer
    
    def __init__(
        self,
        transcriber: WhisperTranscriber,
        vad_filter: bool = True,
        temp_dir: Optional[Path] = None,
    ):
        """
        Initialize the stream processor.
        
        Args:
            transcriber: WhisperTranscriber instance for transcription
            vad_filter: Whether to use voice activity detection
            temp_dir: Directory for temporary files (uses system temp if None)
        """
        self.transcriber = transcriber
        self.vad_filter = vad_filter
        self.temp_dir = temp_dir
        
        # Internal state
        self._pyaudio = None
        self._stream = None
        self._is_running = False
        self._stop_event = threading.Event()
        
        # Audio buffer
        self._audio_buffer: Deque[bytes] = collections.deque()
        self._buffer_lock = threading.Lock()
        
        # Processing threads
        self._capture_thread = None
        self._processing_thread = None
        
        # Callbacks
        self._on_segment = None
        self._on_error = None
        
        # VAD state
        self._is_speech = False
        self._silence_frames = 0
        self._speech_frames = 0
        
        # Current transcript
        self._transcript_id = None
        self._current_segment_start = 0.0
        self._total_audio_duration = 0.0
        
        # Initialize PyAudio if available
        if PYAUDIO_AVAILABLE:
            self._pyaudio = pyaudio.PyAudio()
    
    def start(
        self,
        transcript_id: str,
        device_index: Optional[int] = None,
        language: Optional[str] = None,
        on_segment: Optional[Callable[[str, TranscriptSegment], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> bool:
        """
        Start streaming transcription from microphone.
        
        Args:
            transcript_id: ID of the transcript to update
            device_index: Index of audio input device or None for default
            language: Language code (ISO 639-1) or None for auto-detection
            on_segment: Callback for new transcript segments
            on_error: Callback for errors
            
        Returns:
            True if started successfully, False otherwise
        """
        if not PYAUDIO_AVAILABLE:
            error = RuntimeError("PyAudio not available. Cannot start streaming.")
            if on_error:
                on_error(error)
            else:
                logger.error(str(error))
            return False
        
        if self._is_running:
            logger.warning("Stream processor already running. Stop it first.")
            return False
        
        # Store parameters
        self._transcript_id = transcript_id
        self._on_segment = on_segment
        self._on_error = on_error
        self._language = language
        
        # Reset state
        self._stop_event.clear()
        self._audio_buffer.clear()
        self._is_speech = False
        self._silence_frames = 0
        self._speech_frames = 0
        self._current_segment_start = 0.0
        self._total_audio_duration = 0.0
        
        try:
            # Open audio stream
            self._stream = self._pyaudio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK_SIZE,
                stream_callback=self._audio_callback if self.vad_filter else None,
            )
            
            # Start threads
            self._is_running = True
            
            if self.vad_filter:
                # If using VAD, the audio callback handles capturing
                self._processing_thread = threading.Thread(
                    target=self._process_audio_thread,
                    daemon=True,
                )
                self._processing_thread.start()
            else:
                # If not using VAD, we need separate capture and processing threads
                self._capture_thread = threading.Thread(
                    target=self._capture_audio_thread,
                    daemon=True,
                )
                self._processing_thread = threading.Thread(
                    target=self._process_audio_thread,
                    daemon=True,
                )
                self._capture_thread.start()
                self._processing_thread.start()
            
            logger.info(f"Started streaming transcription (transcript_id={transcript_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            if self._on_error:
                self._on_error(e)
            return False
    
    def stop(self) -> None:
        """Stop streaming transcription."""
        if not self._is_running:
            return
        
        logger.info("Stopping streaming transcription")
        self._stop_event.set()
        
        # Stop audio stream
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        
        # Wait for threads to finish
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)
        
        # Process any remaining audio
        self._process_final_audio()
        
        self._is_running = False
        logger.info("Streaming transcription stopped")
    
    def is_running(self) -> bool:
        """Check if streaming is currently active."""
        return self._is_running
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback for audio data from PyAudio.
        
        This is used when VAD is enabled to filter out silence.
        """
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        try:
            # Check for voice activity
            rms = audioop.rms(in_data, 2)  # 2 bytes per sample for paInt16
            
            # Update VAD state
            if rms > self.VAD_THRESHOLD:
                # Speech detected
                self._silence_frames = 0
                self._speech_frames += 1
                
                # If we have enough speech frames, consider it speech
                if not self._is_speech and self._speech_frames > (self.VAD_SPEECH_DURATION * self.SAMPLE_RATE / self.CHUNK_SIZE):
                    self._is_speech = True
                    logger.debug("Speech started")
            else:
                # Silence detected
                self._silence_frames += 1
                self._speech_frames = 0
                
                # If we have enough silence frames, consider it silence
                if self._is_speech and self._silence_frames > (self.VAD_SILENCE_DURATION * self.SAMPLE_RATE / self.CHUNK_SIZE):
                    self._is_speech = False
                    logger.debug("Speech ended")
            
            # If speech is detected, add to buffer
            if self._is_speech:
                with self._buffer_lock:
                    self._audio_buffer.append(in_data)
                    
                    # Update total duration
                    self._total_audio_duration += frame_count / self.SAMPLE_RATE
                    
                    # Limit buffer size
                    max_frames = int(self.MAX_BUFFER_DURATION * self.SAMPLE_RATE / self.CHUNK_SIZE)
                    while len(self._audio_buffer) > max_frames:
                        self._audio_buffer.popleft()
                        self._total_audio_duration -= self.CHUNK_SIZE / self.SAMPLE_RATE
            
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
            if self._on_error:
                self._on_error(e)
        
        # Continue streaming
        return (in_data, pyaudio.paContinue)
    
    def _capture_audio_thread(self):
        """
        Thread function for capturing audio when VAD is disabled.
        """
        try:
            while not self._stop_event.is_set():
                # Read audio data
                data = self._stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                
                # Add to buffer
                with self._buffer_lock:
                    self._audio_buffer.append(data)
                    
                    # Update total duration
                    self._total_audio_duration += self.CHUNK_SIZE / self.SAMPLE_RATE
                    
                    # Limit buffer size
                    max_frames = int(self.MAX_BUFFER_DURATION * self.SAMPLE_RATE / self.CHUNK_SIZE)
                    while len(self._audio_buffer) > max_frames:
                        self._audio_buffer.popleft()
                        self._total_audio_duration -= self.CHUNK_SIZE / self.SAMPLE_RATE
        
        except Exception as e:
            logger.error(f"Error in capture thread: {e}")
            if self._on_error:
                self._on_error(e)
            self._stop_event.set()
    
    def _process_audio_thread(self):
        """
        Thread function for processing audio and generating transcriptions.
        """
        try:
            # Wait for initial audio
            time.sleep(0.5)
            
            while not self._stop_event.is_set():
                # Check if we have enough audio to process
                chunk_frames = int(self.INFERENCE_CHUNK_DURATION * self.SAMPLE_RATE / self.CHUNK_SIZE)
                
                with self._buffer_lock:
                    if len(self._audio_buffer) >= chunk_frames:
                        # Get audio data for processing
                        audio_data = b''.join(list(self._audio_buffer)[-chunk_frames:])
                        
                        # Clear processed frames if using VAD
                        if self.vad_filter and not self._is_speech:
                            self._audio_buffer.clear()
                            self._current_segment_start = self._total_audio_duration
                    else:
                        audio_data = None
                
                if audio_data:
                    # Process audio chunk
                    self._process_audio_chunk(audio_data)
                
                # Sleep to avoid tight loop
                time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Error in processing thread: {e}")
            if self._on_error:
                self._on_error(e)
            self._stop_event.set()
    
    def _process_audio_chunk(self, audio_data: bytes) -> None:
        """
        Process an audio chunk and generate transcription.
        
        Args:
            audio_data: Raw audio data to process
        """
        try:
            # Save audio to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=self.temp_dir) as temp_file:
                temp_path = Path(temp_file.name)
                
                # Write WAV header and data
                self._write_wav_file(temp_file, audio_data)
            
            # Transcribe the audio
            transcript = self.transcriber.transcribe(
                audio_path=temp_path,
                language=self._language,
                auto_detect_language=False,
            )
            
            # Process segments
            for segment in transcript.segments:
                # Adjust segment times based on current position
                adjusted_segment = TranscriptSegment(
                    start=self._current_segment_start + segment.start,
                    end=self._current_segment_start + segment.end,
                    text=segment.text,
                    confidence=segment.confidence,
                    speaker=segment.speaker,
                )
                
                # Call segment callback
                if self._on_segment:
                    self._on_segment(self._transcript_id, adjusted_segment)
            
            # Clean up temporary file
            temp_path.unlink()
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            if self._on_error:
                self._on_error(e)
    
    def _process_final_audio(self) -> None:
        """Process any remaining audio after stopping."""
        with self._buffer_lock:
            if not self._audio_buffer:
                return
            
            # Get all remaining audio
            audio_data = b''.join(self._audio_buffer)
            self._audio_buffer.clear()
        
        # Process the audio
        if audio_data:
            self._process_audio_chunk(audio_data)
    
    def _write_wav_file(self, file, audio_data: bytes) -> None:
        """
        Write audio data to a WAV file.
        
        Args:
            file: File object to write to
            audio_data: Raw audio data
        """
        # Write WAV header
        file.write(b'RIFF')
        file.write((36 + len(audio_data)).to_bytes(4, 'little'))  # File size
        file.write(b'WAVE')
        
        # Format chunk
        file.write(b'fmt ')
        file.write((16).to_bytes(4, 'little'))  # Chunk size
        file.write((1).to_bytes(2, 'little'))  # Audio format (PCM)
        file.write((self.CHANNELS).to_bytes(2, 'little'))  # Channels
        file.write((self.SAMPLE_RATE).to_bytes(4, 'little'))  # Sample rate
        file.write((self.SAMPLE_RATE * self.CHANNELS * 2).to_bytes(4, 'little'))  # Byte rate
        file.write((self.CHANNELS * 2).to_bytes(2, 'little'))  # Block align
        file.write((16).to_bytes(2, 'little'))  # Bits per sample
        
        # Data chunk
        file.write(b'data')
        file.write(len(audio_data).to_bytes(4, 'little'))  # Chunk size
        file.write(audio_data)
        
        file.flush()
    
    def __del__(self):
        """Clean up resources."""
        self.stop()
        
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
