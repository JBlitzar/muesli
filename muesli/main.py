#!/usr/bin/env python3
"""
Main entry point for the Muesli application.

This module initializes the application, sets up logging, and handles command line arguments.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from muesli.core.app import MuesliApp
from muesli.core.config import AppConfig, load_config


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
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce verbosity of third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("PySide6").setLevel(logging.WARNING)


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
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        help="Path to custom config file",
        default=None
    )
    
    parser.add_argument(
        "--no-ui", 
        action="store_true", 
        help="Run in headless mode (for CLI operations only)"
    )
    
    parser.add_argument(
        "--transcribe", 
        type=str, 
        help="Path to audio file to transcribe directly"
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
    parsed_args = parse_args(args)
    
    # Set up logging
    setup_logging(parsed_args.verbose)
    logger = logging.getLogger(__name__)
    logger.info("Starting Muesli application")
    
    try:
        # Load configuration
        config_path = parsed_args.config
        config = load_config(config_path)
        
        # Initialize the application
        app = MuesliApp(config)
        
        if parsed_args.transcribe:
            # CLI mode: transcribe a single file
            audio_path = Path(parsed_args.transcribe)
            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_path}")
                return 1
                
            logger.info(f"Transcribing file: {audio_path}")
            result = app.transcribe_file(audio_path)
            print(result.text)
            
            if config.auto_summarize:
                summary = app.summarize_transcript(result)
                print("\nSummary:")
                print(summary)
            
            return 0
            
        elif not parsed_args.no_ui:
            # Start the UI
            logger.info("Launching UI")
            return app.start_ui()
        else:
            logger.error("No action specified in headless mode")
            return 1
            
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
