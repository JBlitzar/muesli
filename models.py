"""
Data models for the Muesli application.

This module defines the core data models used throughout the application,
including audio files, transcripts, and summaries.
"""

import datetime
import enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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

    # No segment-level operations are needed for plain-text transcripts.

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def to_srt(self) -> str:
        """Return the transcript in SubRip (SRT) format.

        The old implementation produced multiple timed segments.  Since the new
        model stores only full-text, we emit a single subtitle block that spans
        an hour (00:00:00,000 → 01:00:00,000).  This keeps the existing “Save
        as SRT” menu action functional without re-introducing segment logic.
        """
        srt_lines = [
            "1",
            "00:00:00,000 --> 01:00:00,000",
            self.text.strip(),
            "",  # trailing blank line required by SRT spec
        ]
        return "\n".join(srt_lines)


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
