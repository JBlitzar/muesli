"""
Transcript summarizer for the Muesli application.

This module provides functionality for generating summaries of transcripts
using local LLMs via the Ollama client. It handles prompt formatting,
LLM inference, and result processing.
"""

import logging
import time
from typing import Dict, List, Optional, Union

from muesli.core.models import Summary, SummaryType, Transcript
from muesli.llm.ollama_client import OllamaClient, OllamaError
from muesli.llm.prompt_templates import get_summary_prompt

logger = logging.getLogger(__name__)


class SummarizerError(Exception):
    """Base exception for summarizer-related errors."""
    pass


class TranscriptSummarizer:
    """
    Summarizer for generating summaries of transcripts using LLMs.
    
    This class uses the OllamaClient to generate summaries of transcripts
    based on configurable prompt templates.
    """
    
    def __init__(
        self,
        llm_client: OllamaClient,
        prompt_template: Optional[str] = None,
        max_transcript_length: int = 12000,
        temperature: float = 0.7,
    ):
        """
        Initialize the transcript summarizer.
        
        Args:
            llm_client: OllamaClient instance for LLM inference
            prompt_template: Default prompt template for summarization
            max_transcript_length: Maximum length of transcript to include in prompt
            temperature: Temperature for LLM generation (0.0 to 1.0)
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.max_transcript_length = max_transcript_length
        self.temperature = temperature
        
        logger.info("Transcript summarizer initialized")
    
    def summarize(
        self,
        transcript: Union[Transcript, str],
        summary_type: SummaryType = SummaryType.PARAGRAPH,
        prompt_template: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate a summary of a transcript.
        
        Args:
            transcript: Transcript object or transcript text
            summary_type: Type of summary to generate
            prompt_template: Custom prompt template (overrides default)
            model_name: Custom model name (overrides client default)
            temperature: Custom temperature (overrides default)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated summary text
            
        Raises:
            SummarizerError: If summarization fails
        """
        # Validate transcript
        if isinstance(transcript, Transcript) and not transcript.segments:
            raise SummarizerError("Cannot summarize empty transcript")
        
        # Get transcript text
        if isinstance(transcript, Transcript):
            transcript_text = transcript.text
            if not transcript_text.strip():
                raise SummarizerError("Transcript text is empty")
        else:
            transcript_text = str(transcript)
            if not transcript_text.strip():
                raise SummarizerError("Transcript text is empty")
        
        # Format prompt
        template = prompt_template or self.prompt_template
        prompt = get_summary_prompt(
            transcript=transcript_text,
            summary_type=summary_type,
            custom_template=template,
            max_length=self.max_transcript_length,
        )
        
        # Set generation parameters
        temp = temperature if temperature is not None else self.temperature
        
        # Generate summary
        try:
            logger.info(f"Generating summary for transcript (type={summary_type.value})")
            start_time = time.time()
            
            summary_text = self.llm_client.generate(
                prompt=prompt,
                model_name=model_name,
                temperature=temp,
                max_tokens=max_tokens,
            )
            
            duration = time.time() - start_time
            logger.info(f"Summary generated in {duration:.2f} seconds")
            
            return summary_text.strip()
            
        except OllamaError as e:
            logger.error(f"Error generating summary: {e}")
            raise SummarizerError(f"Failed to generate summary: {e}")
        
        except Exception as e:
            logger.error(f"Unexpected error in summarization: {e}")
            raise SummarizerError(f"Unexpected error: {e}")
    
    def summarize_streaming(
        self,
        transcript: Union[Transcript, str],
        summary_type: SummaryType = SummaryType.PARAGRAPH,
        prompt_template: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
        callback: Optional[callable] = None,
    ) -> str:
        """
        Generate a summary of a transcript with streaming output.
        
        Args:
            transcript: Transcript object or transcript text
            summary_type: Type of summary to generate
            prompt_template: Custom prompt template (overrides default)
            model_name: Custom model name (overrides client default)
            temperature: Custom temperature (overrides default)
            max_tokens: Maximum tokens to generate
            callback: Callback function for streaming chunks
            
        Returns:
            Complete generated summary text
            
        Raises:
            SummarizerError: If summarization fails
        """
        # Validate transcript
        if isinstance(transcript, Transcript) and not transcript.segments:
            raise SummarizerError("Cannot summarize empty transcript")
        
        # Get transcript text
        if isinstance(transcript, Transcript):
            transcript_text = transcript.text
            if not transcript_text.strip():
                raise SummarizerError("Transcript text is empty")
        else:
            transcript_text = str(transcript)
            if not transcript_text.strip():
                raise SummarizerError("Transcript text is empty")
        
        # Format prompt
        template = prompt_template or self.prompt_template
        prompt = get_summary_prompt(
            transcript=transcript_text,
            summary_type=summary_type,
            custom_template=template,
            max_length=self.max_transcript_length,
        )
        
        # Set generation parameters
        temp = temperature if temperature is not None else self.temperature
        
        # Generate summary with streaming
        try:
            logger.info(f"Generating streaming summary for transcript (type={summary_type.value})")
            start_time = time.time()
            
            full_text = ""
            for chunk in self.llm_client.generate_streaming(
                prompt=prompt,
                model_name=model_name,
                temperature=temp,
                max_tokens=max_tokens,
            ):
                full_text += chunk
                if callback:
                    callback(chunk)
            
            duration = time.time() - start_time
            logger.info(f"Streaming summary generated in {duration:.2f} seconds")
            
            return full_text.strip()
            
        except OllamaError as e:
            logger.error(f"Error generating streaming summary: {e}")
            raise SummarizerError(f"Failed to generate summary: {e}")
        
        except Exception as e:
            logger.error(f"Unexpected error in streaming summarization: {e}")
            raise SummarizerError(f"Unexpected error: {e}")
    
    def create_summary_object(
        self,
        transcript: Transcript,
        summary_text: str,
        summary_type: SummaryType = SummaryType.PARAGRAPH,
        model_name: Optional[str] = None,
        prompt_template: Optional[str] = None,
    ) -> Summary:
        """
        Create a Summary object from generated summary text.
        
        Args:
            transcript: Transcript object
            summary_text: Generated summary text
            summary_type: Type of summary
            model_name: Name of the model used
            prompt_template: Prompt template used
            
        Returns:
            Summary object
        """
        return Summary(
            transcript_id=transcript.id,
            text=summary_text,
            summary_type=summary_type,
            model_name=model_name or self.llm_client.model_name,
            prompt_template=prompt_template or self.prompt_template,
        )
