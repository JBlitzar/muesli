#!/usr/bin/env python3
"""
Main entry point for the Muesli application.

This module initializes the application, sets up logging, and handles command line arguments.
"""

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import List, Optional

from muesli.core.app import MuesliApp
from muesli.core.config import AppConfig, load_config


# Global application instance for signal handlers
_app_instance = None


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
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request
    
    # Handle SIGBREAK on Windows (Ctrl+Break)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)


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
            print(result.text)
            
            if config.auto_summarize:
                summary = app.summarize_transcript(result)
                if summary:
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


if __name__ == "__main__":
    sys.exit(main())
