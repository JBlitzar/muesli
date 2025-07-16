"""
Whisper.cpp wrapper for the Muesli application.

This module provides a simple Python wrapper around the whisper.cpp CLI tool
for transcribing audio files.
"""

import json
import logging
import os
import platform
import shutil
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Union

from models import Transcript, AudioFile

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
    """Simple wrapper for whisper.cpp CLI tool for transcribing audio files."""
    
    def __init__(
        self,
        model_name: str = "medium",
        model_dir: Optional[Path] = None,
        device: str = "auto",
        beam_size: int = 5,
        whisper_binary: Optional[str] = None,
    ):
        """
        Initialize the transcriber.
        
        Args:
            model_name: Name of the Whisper model to use
            model_dir: Directory where models are stored
            device: Device to use for inference ('cpu', 'cuda', 'mps', or 'auto')
            beam_size: Beam size for decoding
            whisper_binary: Path to whisper.cpp binary
        """
        self.model_name = model_name
        self.model_dir = model_dir or Path.home() / ".muesli" / "models" / "whisper"
        self.device = self._resolve_device(device)
        self.beam_size = beam_size
        
        # Find whisper binary
        self.whisper_binary = whisper_binary or self._find_whisper_binary()
        
        # Create model directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Directory where intermediate whisper output is stored
        self.output_dir = Path.home() / ".muesli" / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _find_whisper_binary(self) -> str:
        """Find the whisper.cpp binary in PATH."""
        binary_names = ["whisper", "whisper-cpp", "main"]
        if platform.system() == "Windows":
            binary_names = [f"{name}.exe" for name in binary_names]
        
        for name in binary_names:
            binary_path = shutil.which(name)
            if binary_path:
                logger.info(f"Found whisper.cpp binary at {binary_path}")
                return binary_path
        
        # Not found
        raise RuntimeError(
            "whisper.cpp binary not found. Please install it and ensure "
            "it's in your PATH, or provide the path using the whisper_binary parameter."
        )
    
    def _resolve_device(self, device: str) -> str:
        """Auto-detect device if set to 'auto'."""
        if device != "auto":
            return device
            
        # Check for CUDA
        try:
            result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if result.returncode == 0:
                return "cuda"
        except Exception:
            pass
        
        # Check for MPS (Apple Silicon)
        if platform.system() == "Darwin" and platform.processor() == "arm":
            return "mps"
        
        # Default to CPU
        return "cpu"
    
    def _get_model_path(self) -> Path:
        """Get the path to the model file."""
        # Check for both naming conventions
        candidate_filenames = [
            f"ggml-{self.model_name}.bin",
            f"ggml-{self.model_name}.en.bin",
        ]

        for filename in candidate_filenames:
            path = self.model_dir / filename
            if path.exists():
                logger.info(f"Found model at {path}")
                return path

        # Default to the newer naming convention
        model_path = self.model_dir / f"ggml-{self.model_name}.en.bin"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Please download it manually."
            )
        
        return model_path
    
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
        """
        # Convert to Path if string
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)
        
        # Create audio file object
        audio_file = AudioFile.from_path(audio_path)
        
        # Create empty transcript
        transcript = Transcript(audio_file=audio_file)
        transcript.model_name = self.model_name
        
        # Get model path
        model_path = self._get_model_path()
        
        # Detect or set language
        lang = language or "auto" if auto_detect_language else "en"
        
        # Create a unique output filename inside ~/.muesli/output
        timestamp = int(time.time() * 1000)
        output_file = self.output_dir / f"{audio_path.stem}_{timestamp}"
        
        # Build command
        cmd = [
            self.whisper_binary,
            "--model", str(model_path),
            "--language", lang,
            "--output-json", "true",
            "--output-file", str(output_file),
            "--print-progress", "true",
            "--beam-size", str(self.beam_size),
            str(audio_path)
        ]

       


        
        logger.info(f"Running command: {' '.join(cmd)}")

        start = time.time()
        
        # Run the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Process output and track progress
        for line in iter(process.stdout.readline, ''):
            logger.debug(line.strip())
            if "progress" in line.lower() and progress_callback:
                try:
                    progress_str = line.split(':')[1].strip().rstrip('%')
                    progress = float(progress_str) / 100.0
                    progress_callback(progress, f"Transcribing: {int(progress * 100)}%")
                except Exception:
                    pass
            
            # Check for language detection
            if "detected language:" in line.lower() and auto_detect_language:
                try:
                    lang_code = line.split(":")[1].strip().split()[0]
                    transcript.language = lang_code
                    logger.info(f"Detected language: {lang_code}")
                except Exception:
                    pass
        
        # Wait for process to complete
        stdout, stderr = process.communicate()

        end = time.time()
        elapsed = end - start

        # ------------------------------------------------------------------+
        # Log more informative timing information                           +
        # ------------------------------------------------------------------+
        audio_duration: Optional[float] = None
        try:
            # Use ffprobe (if available) to get the audio duration
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=5
            )
            audio_duration = float(result.stdout.strip())
        except Exception:
            # Silently ignore if ffprobe is not available or fails
            pass

        if audio_duration and audio_duration > 0:
            realtime_factor = audio_duration / elapsed if elapsed > 0 else 0.0
            print(
                f"Transcribed {audio_duration:.1f}s of audio in "
                f"{elapsed:.1f}s (≈{realtime_factor:.2f}× realtime)"
            )
        else:
            print(f"Transcription wall time: {elapsed:.2f}s")
        
        if process.returncode != 0:
            logger.error(f"Transcription failed: {stderr}")
            raise RuntimeError(f"Transcription failed: {stderr}")
        
        # Parse JSON output
        json_path = output_file.with_suffix(".json")
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                print(data)
                # whisper.cpp (>=1.7) puts full text under transcription.text
                # older builds may put plain "text" at root level
                full_text: Optional[str] = None
                if isinstance(data, dict):
                    if "transcription" in data and isinstance(data["transcription"], list):
                        full_text = "\n".join(data["transcription"][i]["text"] for i in range(len(data["transcription"])))
                    elif "text" in data:  # fallback
                        full_text = data.get("text")

                if not full_text:
                    raise ValueError("Could not find transcription text in JSON")

                transcript.text = full_text.strip()

                # Update language if present
                if "language" in data and auto_detect_language:
                    transcript.language = data["language"]

            except Exception as e:
                logger.error(f"Error parsing JSON output: {e}")
                raise RuntimeError(f"Error parsing transcription output: {e}") from e
        else:
            logger.error(f"JSON output file not found: {json_path}")
            raise RuntimeError("No output generated by whisper.cpp")
        
        # Mark as complete
        transcript.is_complete = True

        # Cleanup generated files
        try:
            json_path.unlink(missing_ok=True)  # requires Python 3.8+
        except Exception:
            pass

        return transcript
    
    def close(self) -> None:
        """Release resources (no-op for subprocess approach)."""
        pass
