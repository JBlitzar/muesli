"""
Configuration module for the Muesli application.

This module provides Pydantic models for configuration validation and
utilities for loading and saving configuration from various sources.
"""

import os
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import yaml
from pydantic import (
    BaseModel, 
    Field, 
    field_validator, 
    model_validator, 
    ConfigDict,
)


logger = logging.getLogger(__name__)


class ThemeMode(str, Enum):
    """UI theme options."""
    DARK = "dark"
    LIGHT = "light"
    SYSTEM = "system"


class WhisperModel(str, Enum):
    """Available Whisper model sizes."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V3 = "large-v3"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    NONE = "none"  # Disable LLM features


class TranscriptionConfig(BaseModel):
    """Configuration for the transcription module."""
    
    model: WhisperModel = Field(
        default=WhisperModel.MEDIUM,
        description="Whisper model size to use for transcription"
    )
    
    model_dir: Path = Field(
        default=Path.home() / ".muesli" / "models" / "whisper",
        description="Directory where Whisper models are stored"
    )
    
    auto_language_detection: bool = Field(
        default=True,
        description="Automatically detect language of audio"
    )
    
    default_language: str = Field(
        default="en",
        description="Default language code (ISO 639-1) when auto-detection is disabled"
    )
    
    device: str = Field(
        default="auto",
        description="Device to use for inference: 'cpu', 'cuda', 'mps', or 'auto' to detect"
    )
    
    beam_size: int = Field(
        default=5,
        description="Beam size for decoding (higher = more accurate but slower)"
    )
    
    vad_filter: bool = Field(
        default=True,
        description="Use voice activity detection to filter out non-speech"
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
        default=LLMProvider.OLLAMA,
        description="LLM provider to use"
    )
    
    ollama_host: str = Field(
        default="localhost",
        description="Hostname for Ollama server"
    )
    
    ollama_port: int = Field(
        default=11434,
        description="Port for Ollama server"
    )
    
    model_name: str = Field(
        default="llama3:8b-instruct",
        description="Model name to use with the provider"
    )
    
    offline_mode: bool = Field(
        default=True,
        description="Run LLM in offline mode (no network access)"
    )
    
    summary_prompt_template: str = Field(
        default=(
            "Below is a transcript of an audio recording. "
            "Please provide a concise summary of the key points discussed:\n\n"
            "{transcript}\n\n"
            "Summary:"
        ),
        description="Prompt template for generating summaries"
    )
    
    @property
    def ollama_base_url(self) -> str:
        """Get the base URL for Ollama API."""
        return f"http://{self.ollama_host}:{self.ollama_port}"


class UIConfig(BaseModel):
    """Configuration for the UI module."""
    
    theme: ThemeMode = Field(
        default=ThemeMode.SYSTEM,
        description="UI theme mode"
    )
    
    font_size: int = Field(
        default=12,
        description="Base font size in points"
    )
    
    recent_files_limit: int = Field(
        default=10,
        description="Number of recent files to display"
    )
    
    waveform_visible: bool = Field(
        default=True,
        description="Show audio waveform in transcript view"
    )
    
    auto_scroll: bool = Field(
        default=True,
        description="Auto-scroll transcript during playback"
    )


class PrivacyConfig(BaseModel):
    """Configuration for privacy settings."""
    
    telemetry_enabled: bool = Field(
        default=False,
        description="Enable anonymous usage telemetry"
    )
    
    allow_network: bool = Field(
        default=False,
        description="Allow network access for model downloads and updates"
    )
    
    model_hash_verification: bool = Field(
        default=True,
        description="Verify hash of downloaded models"
    )
    
    log_redaction: bool = Field(
        default=True,
        description="Redact potentially sensitive information from logs"
    )


class AppConfig(BaseModel):
    """Main application configuration."""
    
    model_config = ConfigDict(
        extra="forbid",  # Forbid extra fields not defined in the model
    )
    
    # Application metadata
    app_name: str = Field(
        default="Muesli",
        description="Application name"
    )
    
    app_version: str = Field(
        default="0.1.0",
        description="Application version"
    )
    
    # Module configurations
    transcription: TranscriptionConfig = Field(
        default_factory=TranscriptionConfig,
        description="Transcription module configuration"
    )
    
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM module configuration"
    )
    
    ui: UIConfig = Field(
        default_factory=UIConfig,
        description="UI module configuration"
    )
    
    privacy: PrivacyConfig = Field(
        default_factory=PrivacyConfig,
        description="Privacy settings"
    )
    
    # Application behavior
    auto_transcribe: bool = Field(
        default=True,
        description="Automatically start transcription when audio is loaded"
    )
    
    auto_summarize: bool = Field(
        default=False,
        description="Automatically generate summary when transcription completes"
    )
    
    data_dir: Path = Field(
        default=Path.home() / ".muesli" / "data",
        description="Directory for application data"
    )
    
    cache_dir: Path = Field(
        default=Path.home() / ".muesli" / "cache",
        description="Directory for application cache"
    )
    
    @model_validator(mode="after")
    def ensure_directories(self) -> "AppConfig":
        """Ensure all required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self


def load_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """
    Load application configuration from a YAML file with environment variable overrides.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Validated AppConfig instance
        
    If config_path is None, looks for config in standard locations:
    1. ./muesli.yaml
    2. ~/.muesli/config.yaml
    3. If not found, returns default configuration
    """
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
            parts = env_var[len(env_prefix):].lower().split("_", 1)
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
                logger.debug(f"Config override from environment: {section}.{key}={value}")
    
    # Create and validate config
    try:
        return AppConfig(**config_data)
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        logger.warning("Falling back to default configuration")
        return AppConfig()


def save_config(config: AppConfig, config_path: Optional[Union[str, Path]] = None) -> bool:
    """
    Save configuration to a YAML file.
    
    Args:
        config: AppConfig instance to save
        config_path: Path to save configuration to
        
    Returns:
        True if successful, False otherwise
        
    If config_path is None, saves to ~/.muesli/config.yaml
    """
    if config_path is None:
        config_dir = Path.home() / ".muesli"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"
    else:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert to dict and save as YAML
        config_dict = config.model_dump()
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {e}")
        return False
