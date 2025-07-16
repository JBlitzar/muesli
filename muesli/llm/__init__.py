"""
LLM module for the Muesli application.

This module provides functionality for interacting with local LLMs via Ollama
and generating summaries of transcripts. It handles prompt templating,
LLM inference, and result processing.
"""

# Re-export key classes for easier imports
from muesli.llm.ollama_client import OllamaClient
from muesli.llm.summarizer import TranscriptSummarizer
from muesli.llm.prompt_templates import get_summary_prompt

__all__ = [
    "OllamaClient",
    "TranscriptSummarizer",
    "get_summary_prompt",
]
