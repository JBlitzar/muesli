"""
Whisper.cpp wrapper for the Muesli application.

This module provides a Python wrapper around the whisper.cpp library for
transcribing audio files. It handles model loading, inference, and resource
management.
"""

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from models import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)


class WhisperModelSize(str, Enum):
    """Available Whisper model sizes."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V3 = "large-v3"


class WhisperTranscriber:
    """
    Wrapper for whisper.cpp library for transcribing audio files.
    
    This class provides an interface to the whisper.cpp library, handling
    model loading, inference, and resource management.
    """
    
    # Model file URLs and checksums for verification
    MODEL_URLS = {
        WhisperModelSize.TINY: {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
            "sha256": "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
            "size_mb": 75,
        },
        WhisperModelSize.BASE: {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
            "sha256": "137c40403d40b2faf25f4c8659fb2fb5d179d7591264e6f09c9aaa3c3fa80de3",
            "size_mb": 142,
        },
        WhisperModelSize.SMALL: {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
            "sha256": "55d9d544f5438ce8b7c8b0a0334a1d7d7c386864bbe8fb425f68b1e2d5b9ac0c",
            "size_mb": 466,
        },
        WhisperModelSize.MEDIUM: {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
            "sha256": "d58e08a866e91a6eb38c679a7dc7b56a55a041c893a532233d3a8b8e91f0e5af",
            "size_mb": 1500,
        },
        WhisperModelSize.LARGE: {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large.bin",
            "sha256": "9a423fe4d40c82774b6af34115b7f49e59e5f5541e25a6b26411c5a7d1ab02c5",
            "size_mb": 2900,
        },
        WhisperModelSize.LARGE_V3: {
            "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
            "sha256": "bfd3cb2fca3ef32b0db82d0c2ef3d19b329e9c7a6e5b7c978fbcf2e11e7f1e54",
            "size_mb": 2900,
        },
    }
    
    def __init__(
        self,
        model_name: str = "medium",
        model_dir: Optional[Path] = None,
        device: str = "auto",
        beam_size: int = 5,
        download_if_missing: bool = True,
        whisper_binary: Optional[str] = None,
    ):
        """
        Initialize the transcriber.
        
        Args:
            model_name: Name of the Whisper model to use
            model_dir: Directory where models are stored
            device: Device to use for inference ('cpu', 'cuda', 'mps', or 'auto')
            beam_size: Beam size for decoding
            download_if_missing: Whether to download the model if not found
            whisper_binary: Path to whisper.cpp binary (defaults to 'whisper-cpp' or 'whisper.exe')
        """
        self.model_name = model_name
        self.model_dir = model_dir or Path.home() / ".muesli" / "models" / "whisper"
        self.device = self._resolve_device(device)
        self.beam_size = beam_size
        self.download_if_missing = download_if_missing
        
        # Find whisper binary
        self.whisper_binary = self._find_whisper_binary(whisper_binary)
        
        # Create model directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Internal state
        self._model_path = None
        self._is_initialized = False
        
        # Try to initialize the model
        self._initialize()
    
    def _find_whisper_binary(self, whisper_binary: Optional[str]) -> str:
        """
        Find the whisper.cpp binary.
        
        Args:
            whisper_binary: Path to whisper.cpp binary or None to auto-detect
            
        Returns:
            Path to whisper.cpp binary
            
        Raises:
            RuntimeError: If whisper.cpp binary is not found
        """
        if whisper_binary:
            # Use provided binary path
            if not os.path.exists(whisper_binary):
                raise RuntimeError(f"Whisper binary not found at {whisper_binary}")
            return whisper_binary
        
        # Try to find binary in PATH
        binary_names = ["whisper-cpp", "whisper", "main"]  # 'main' is often used in whisper.cpp repo
        if platform.system() == "Windows":
            binary_names = [f"{name}.exe" for name in binary_names]
        
        for name in binary_names:
            binary_path = shutil.which(name)
            if binary_path:
                logger.info(f"Found whisper.cpp binary at {binary_path}")
                return binary_path
        
        # Check if we can run the whisper command directly
        try:
            result = subprocess.run(
                ["whisper", "--help"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            # Check for whisper.cpp specific output
            output = result.stdout + result.stderr
            if "whisper.cpp" in output or "ggml" in output or "model path" in output:
                logger.info("Found whisper.cpp binary as 'whisper'")
                return "whisper"
        except Exception:
            pass
        
        # Not found, provide instructions
        raise RuntimeError(
            "whisper.cpp binary not found. Please install whisper.cpp and ensure "
            "it's in your PATH, or provide the path using the whisper_binary parameter. "
            "Installation instructions: https://github.com/ggerganov/whisper.cpp"
        )
    
    def _resolve_device(self, device: str) -> str:
        """
        Resolve the device to use for inference.
        
        Args:
            device: Device specification ('cpu', 'cuda', 'mps', or 'auto')
            
        Returns:
            Resolved device name
        """
        if device != "auto":
            return device
        
        # Auto-detect device
        system = platform.system()
        
        # Check for CUDA
        try:
            # Check if nvidia-smi is available
            result = subprocess.run(
                ["nvidia-smi"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=False
            )
            if result.returncode == 0:
                logger.info("CUDA detected via nvidia-smi, using GPU for transcription")
                return "cuda"
        except Exception:
            pass
        
        # Check for MPS (Apple Silicon)
        if system == "Darwin" and platform.processor() == "arm":
            try:
                # Check if we're on macOS 12.3+ with Apple Silicon
                mac_version = tuple(map(int, platform.mac_ver()[0].split('.')))
                if mac_version >= (12, 3):
                    logger.info("Apple Silicon detected, using MPS for transcription")
                    return "mps"
            except Exception:
                pass
        
        # Fall back to CPU
        logger.info("No GPU detected, using CPU for transcription")
        return "cpu"
    
    def _initialize(self) -> None:
        """
        Initialize the transcriber by loading the model.
        
        This method loads the Whisper model and prepares it for inference.
        """
        try:
            model_path = self._get_model_path()
            self._model_path = model_path
            
            # Verify the model file exists
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            # Verify we can run the whisper binary
            self._verify_whisper_binary()
            
            self._is_initialized = True
            logger.info(f"Whisper model initialized successfully: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            self._is_initialized = False
    
    def _verify_whisper_binary(self) -> None:
        """
        Verify that the whisper.cpp binary works.
        
        Raises:
            RuntimeError: If the whisper.cpp binary doesn't work
        """
        try:
            # Try to run whisper with --help
            result = subprocess.run(
                [self.whisper_binary, "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            # Log the output for debugging
            logger.debug(f"whisper.cpp help output:\n{result.stdout}\n{result.stderr}")
            
            # Check if it ran successfully
            if result.returncode != 0:
                raise RuntimeError(
                    f"whisper.cpp binary test failed with exit code {result.returncode}: {result.stderr}"
                )
            
            logger.debug("whisper.cpp binary verified successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to verify whisper.cpp binary: {e}")
    
    def _get_model_path(self) -> Path:
        """
        Get the path to the model file, downloading it if necessary.
        
        Returns:
            Path to the model file
        
        Raises:
            FileNotFoundError: If the model is not found and download_if_missing is False
            RuntimeError: If the model download fails
        """
        # Normalize model name
        model_size = WhisperModelSize(self.model_name.lower())
        
        # ------------------------------------------------------------------
        # Accept multiple naming conventions (with or without language tag)
        # Some users (and older docs) still download plain
        #   ggml-medium.bin
        # whereas newer whisper.cpp releases ship
        #   ggml-medium.en.bin
        # Probe both before deciding to download.
        # ------------------------------------------------------------------
        candidate_filenames = [
            f"ggml-{model_size.value}.bin",
            f"ggml-{model_size.value}.en.bin",
        ]

        for filename in candidate_filenames:
            path = self.model_dir / filename
            if path.exists():
                logger.debug(f"Found existing model at {path}")
                return path

        # Default target path for (potential) download – keep modern .en.bin
        model_filename = candidate_filenames[-1]
        model_path = self.model_dir / model_filename
        
        # Model not found, download if allowed
        if not self.download_if_missing:
            raise FileNotFoundError(
                f"Model file not found at {model_path} and download_if_missing is False"
            )
        
        # Download the model
        logger.info(f"Downloading Whisper model {model_size.value} to {model_path}")
        self._download_model(model_size, model_path)
        
        return model_path
    
    def _download_model(self, model_size: WhisperModelSize, model_path: Path) -> None:
        """
        Download a Whisper model.
        
        Args:
            model_size: Size of the model to download
            model_path: Path to save the model to
            
        Raises:
            RuntimeError: If the download fails
        """
        # Get model info
        model_info = self.MODEL_URLS.get(model_size)
        if not model_info:
            raise ValueError(f"Unknown model size: {model_size}")
        
        url = model_info["url"]
        expected_sha256 = model_info["sha256"]
        size_mb = model_info["size_mb"]
        
        logger.info(f"Downloading {size_mb} MB model from {url}")
        
        if not HTTPX_AVAILABLE:
            raise RuntimeError(
                "httpx library not available for downloading models. "
                "Please install it with 'pip install httpx' or manually download "
                f"the model from {url} to {model_path}"
            )
        
        # Download with progress tracking
        try:
            with httpx.stream("GET", url, follow_redirects=True) as response:
                response.raise_for_status()
                
                # Get content length if available
                total_size = int(response.headers.get("content-length", 0))
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                    
                    # Download chunks
                    downloaded_size = 0
                    for chunk in response.iter_bytes(chunk_size=8192):
                        temp_file.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Calculate progress
                        if total_size > 0:
                            progress = downloaded_size / total_size
                            logger.debug(f"Download progress: {progress:.1%}")
            
            # Verify checksum
            logger.info("Verifying model checksum...")
            sha256_hash = hashlib.sha256()
            with open(temp_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            
            actual_sha256 = sha256_hash.hexdigest()
            if actual_sha256 != expected_sha256:
                os.unlink(temp_path)
                raise RuntimeError(
                    f"Model checksum verification failed. Expected {expected_sha256}, got {actual_sha256}"
                )
            
            # Move to final location
            model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(temp_path, model_path)
            logger.info(f"Model downloaded and verified successfully to {model_path}")
            
        except Exception as e:
            # Clean up temporary file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            
            raise RuntimeError(f"Failed to download model: {e}")
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        auto_detect_language: bool = True,
        progress_callback: Optional[Callable[[float, Optional[str]], None]] = None,
    ) -> Transcript:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (ISO 639-1) or None for auto-detection
            auto_detect_language: Whether to auto-detect language
            progress_callback: Callback for progress updates
            
        Returns:
            Transcript object
            
        Raises:
            RuntimeError: If transcription fails
            ValueError: If the audio file is invalid
        """
        # Convert to Path if string
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)
        
        # Check if the file exists
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check if the model is initialized
        if not self._is_initialized:
            logger.warning("Model not initialized, attempting to initialize")
            self._initialize()
            if not self._is_initialized:
                raise RuntimeError("Failed to initialize model")
        
        # Create audio file object
        from models import AudioFile
        audio_file = AudioFile.from_path(audio_path)
        
        # Create empty transcript
        transcript = Transcript(audio_file=audio_file)
        transcript.model_name = self.model_name
        
        # Detect language if needed
        detected_language = language
        if auto_detect_language:
            if progress_callback:
                progress_callback(0.1, "Detecting language...")
            
            detected_language = self._detect_language(audio_path)
            logger.info(f"Detected language: {detected_language}")
        
        transcript.language = detected_language or "en"
        
        # Perform transcription
        if progress_callback:
            progress_callback(0.2, "Starting transcription...")
        
        try:
            # Run whisper.cpp on the audio file
            segments = self._run_whisper_transcription(
                audio_path, 
                transcript.language, 
                progress_callback
            )
            
            # Add segments to transcript
            transcript.segments = segments
            transcript.is_complete = True
            
            if progress_callback:
                progress_callback(1.0, "Transcription complete")
            
            return transcript
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}")
    
    def _detect_language(self, audio_path: Path) -> str:
        """
        Detect the language of an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Detected language code (ISO 639-1)
        """
        logger.info(f"Detecting language for {audio_path}")
        
        try:
            # Create a temporary directory for output files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                
                # Run whisper.cpp with language detection only
                cmd = [
                    self.whisper_binary,
                    "--model", str(self._model_path),
                    "--language", "auto",
                    "--detect-language",
                    "--output-txt", "true",
                    "--output-dir", str(temp_dir_path),
                    str(audio_path)
                ]
                
                # Add device-specific arguments
                if self.device == "cuda":
                    cmd.extend(["--gpu", "0"])
                elif self.device == "mps":
                    cmd.append("--use-mmap")
                
                logger.debug(f"Running language detection command: {' '.join(cmd)}")
                
                # Run the command
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                
                # Log the output for debugging
                logger.debug(f"Language detection stdout: {result.stdout}")
                logger.debug(f"Language detection stderr: {result.stderr}")
                
                # Check for errors
                if result.returncode != 0:
                    logger.error(f"Language detection failed with code {result.returncode}: {result.stderr}")
                    return "en"  # Default to English on failure
                
                # First try to extract language from stdout
                stdout = result.stdout.strip()
                stderr = result.stderr.strip()
                
                # Look for language in output
                language = self._extract_language_from_output(stdout + "\n" + stderr)
                if language:
                    return language
                
                # Try to find language in text output file
                txt_files = list(temp_dir_path.glob("*.txt"))
                if txt_files:
                    with open(txt_files[0], "r", encoding="utf-8") as f:
                        content = f.read()
                        language = self._extract_language_from_output(content)
                        if language:
                            return language
                
                # Default to English if we couldn't find the language
                return "en"
                
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "en"  # Default to English on failure
    
    def _extract_language_from_output(self, output: str) -> Optional[str]:
        """
        Extract language code from whisper.cpp output.
        
        Args:
            output: Output text from whisper.cpp
            
        Returns:
            Language code or None if not found
        """
        # Try different patterns to find language code
        patterns = [
            r"detected language: ([a-z]{2})",  # Format: "detected language: en"
            r"language: ([a-z]{2})",           # Format: "language: en"
            r"\[([a-z]{2})\]",                 # Format: "[en]"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, output.lower())
            if matches:
                return matches[0]
        
        return None
    
    def _run_whisper_transcription(
        self,
        audio_path: Path,
        language: str,
        progress_callback: Optional[Callable[[float, Optional[str]], None]] = None,
    ) -> List[TranscriptSegment]:
        """
        Run whisper.cpp on an audio file and parse the results.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (ISO 639-1)
            progress_callback: Callback for progress updates
            
        Returns:
            List of transcript segments
        """
        logger.info(f"Transcribing {audio_path} with language {language}")
        
        # Create a temporary directory for output files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            json_output_path = temp_dir_path / "transcript.json"
            txt_output_path = temp_dir_path / "transcript.txt"
            
            try:
                # Build command
                cmd = [
                    self.whisper_binary,
                    "--model", str(self._model_path),
                    "--language", language,
                    "--output-json", str(json_output_path),
                    "--output-txt", "true",
                    "--output-dir", str(temp_dir_path),
                    "--output-srt", "false",
                    "--print-progress", "true",
                    "--beam-size", str(self.beam_size),
                    str(audio_path)
                ]
                
                # Add device-specific arguments
                if self.device == "cuda":
                    cmd.extend(["--gpu", "0"])
                elif self.device == "mps":
                    cmd.append("--use-mmap")
                
                logger.debug(f"Running transcription command: {' '.join(cmd)}")
                
                # Run the command with real-time progress monitoring
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Track progress
                stdout_lines = []
                for line in iter(process.stdout.readline, ''):
                    stdout_lines.append(line)
                    
                    # Try to parse progress information
                    if "progress" in line.lower():
                        try:
                            # Extract progress percentage
                            progress_str = line.split(':')[1].strip().rstrip('%')
                            progress = float(progress_str) / 100.0
                            
                            # Update progress (scale from 20% to 90%)
                            if progress_callback:
                                scaled_progress = 0.2 + (0.7 * progress)
                                progress_callback(scaled_progress, f"Transcribing: {int(progress * 100)}%")
                        except Exception as e:
                            logger.debug(f"Failed to parse progress: {e}")
                
                # Wait for process to complete
                stdout, stderr = process.communicate()
                stdout_lines.append(stdout)
                
                # Log the complete output for debugging
                full_stdout = "".join(stdout_lines)
                logger.debug(f"Transcription stdout: {full_stdout}")
                logger.debug(f"Transcription stderr: {stderr}")
                
                # Check for errors
                if process.returncode != 0:
                    raise RuntimeError(
                        f"Transcription failed with exit code {process.returncode}: {stderr}"
                    )
                
                # Parse the JSON output
                if progress_callback:
                    progress_callback(0.9, "Processing results...")
                
                # Try to parse JSON output
                segments = self._parse_json_output(json_output_path)
                
                # If JSON parsing failed, try to parse text output as fallback
                if not segments and txt_output_path.exists():
                    logger.info("JSON parsing failed, falling back to text output")
                    segments = self._parse_text_output(txt_output_path)
                
                # If we still don't have segments, try to parse stdout directly
                if not segments:
                    logger.info("Falling back to stdout parsing")
                    segments = self._parse_stdout(full_stdout)
                
                logger.info(f"Transcription complete: {len(segments)} segments")
                return segments
                
            except Exception as e:
                logger.error(f"Error running whisper.cpp: {e}")
                raise
    
    def _parse_json_output(self, json_path: Path) -> List[TranscriptSegment]:
        """
        Parse JSON output from whisper.cpp.
        
        Args:
            json_path: Path to JSON output file
            
        Returns:
            List of transcript segments
        """
        if not json_path.exists():
            logger.warning(f"JSON output file not found: {json_path}")
            return []
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    # Read a short preview of the raw file to aid debugging
                    f.seek(0)
                    preview = f.read(500)
                    logger.error(
                        "JSON decoding failed on whisper.cpp output: %s\nPreview:\n%s",
                        e, preview
                    )
                    return []
            
            # Convert to transcript segments
            segments = []
            
            if 'segments' in data:
                for segment_data in data['segments']:
                    segment = TranscriptSegment(
                        start=segment_data.get('start', 0.0),
                        end=segment_data.get('end', 0.0),
                        text=segment_data.get('text', '').strip(),
                        confidence=segment_data.get('confidence', 1.0),
                    )
                    segments.append(segment)
            
            return segments
            
        except Exception as e:
            logger.error(f"Error parsing JSON output: {e}")
            return []
    
    def _parse_text_output(self, txt_path: Path) -> List[TranscriptSegment]:
        """
        Parse text output from whisper.cpp as a fallback.
        
        Args:
            txt_path: Path to text output file
            
        Returns:
            List of transcript segments
        """
        if not txt_path.exists():
            logger.warning(f"Text output file not found: {txt_path}")
            return []
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple fallback: create a single segment with the entire text
            if content.strip():
                segment = TranscriptSegment(
                    start=0.0,
                    end=0.0,  # We don't have timing information
                    text=content.strip(),
                    confidence=1.0,
                )
                return [segment]
            
            return []
            
        except Exception as e:
            logger.error(f"Error parsing text output: {e}")
            return []
    
    def _parse_stdout(self, stdout: str) -> List[TranscriptSegment]:
        """
        Parse stdout from whisper.cpp as a last resort fallback.
        
        Args:
            stdout: Standard output from whisper.cpp
            
        Returns:
            List of transcript segments
        """
        try:
            # Try to extract timestamps and text
            # Format: [00:00:00.000 --> 00:00:05.000]  Text content
            segments = []
            pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]\s+(.*?)(?=\[\d{2}:\d{2}:\d{2}\.\d{3} -->|\Z)'
            
            matches = re.findall(pattern, stdout, re.DOTALL)
            
            for start_str, end_str, text in matches:
                try:
                    start = self._timestamp_to_seconds(start_str)
                    end = self._timestamp_to_seconds(end_str)
                    
                    segment = TranscriptSegment(
                        start=start,
                        end=end,
                        text=text.strip(),
                        confidence=1.0,
                    )
                    segments.append(segment)
                except Exception as e:
                    logger.debug(f"Error parsing timestamp: {e}")
            
            # If no segments with timestamps found, create a single segment with all text
            if not segments:
                # Remove progress lines and other non-transcript content
                lines = [line for line in stdout.split('\n') 
                         if line.strip() and 'progress' not in line.lower() 
                         and 'whisper' not in line.lower()]
                
                if lines:
                    text = '\n'.join(lines)
                    segment = TranscriptSegment(
                        start=0.0,
                        end=0.0,
                        text=text.strip(),
                        confidence=1.0,
                    )
                    segments.append(segment)
            
            return segments
            
        except Exception as e:
            logger.error(f"Error parsing stdout: {e}")
            return []
    
    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """
        Convert a timestamp string to seconds.
        
        Args:
            timestamp: Timestamp string in format HH:MM:SS.mmm
            
        Returns:
            Time in seconds
        """
        h, m, s = timestamp.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    
    def close(self) -> None:
        """
        Release resources used by the transcriber.
        
        This method should be called when the transcriber is no longer needed
        to free up memory and other resources.
        """
        # No need to explicitly free resources when using subprocess approach
        self._is_initialized = False
        logger.info("Transcriber resources released")
