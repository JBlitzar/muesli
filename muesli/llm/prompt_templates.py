"""
Prompt templates for LLM interactions in the Muesli application.

This module provides templates and formatting functions for LLM prompts,
particularly focused on transcript summarization. It includes default 
templates for different summary types and functions to format prompts
with transcript content.
"""

import logging
from enum import Enum
from typing import Dict, Optional, Union

from muesli.core.models import SummaryType, Transcript

logger = logging.getLogger(__name__)


# Default prompt templates for different summary types
DEFAULT_TEMPLATES = {
    SummaryType.PARAGRAPH: (
        "Below is a transcript of an audio recording. "
        "Please provide a concise summary of the key points discussed in paragraph form:\n\n"
        "{transcript}\n\n"
        "Summary:"
    ),
    
    SummaryType.BULLET_POINTS: (
        "Below is a transcript of an audio recording. "
        "Please summarize the key points in a bullet point format:\n\n"
        "{transcript}\n\n"
        "Key points:"
    ),
    
    SummaryType.EXECUTIVE: (
        "Below is a transcript of an audio recording. "
        "Please provide a brief executive summary (3-5 sentences maximum) "
        "highlighting only the most critical information:\n\n"
        "{transcript}\n\n"
        "Executive summary:"
    ),
    
    SummaryType.DETAILED: (
        "Below is a transcript of an audio recording. "
        "Please provide a detailed summary that captures all significant "
        "points, arguments, and decisions discussed. Organize by topics "
        "where appropriate:\n\n"
        "{transcript}\n\n"
        "Detailed summary:"
    ),
}


def get_summary_prompt(
    transcript: Union[str, Transcript],
    summary_type: SummaryType = SummaryType.PARAGRAPH,
    custom_template: Optional[str] = None,
    max_length: Optional[int] = None,
) -> str:
    """
    Get a formatted prompt for summarizing a transcript.
    
    Args:
        transcript: Transcript object or transcript text
        summary_type: Type of summary to generate
        custom_template: Custom prompt template (overrides default)
        max_length: Maximum length of transcript to include in prompt
        
    Returns:
        Formatted prompt string
    """
    # Get transcript text
    if isinstance(transcript, Transcript):
        transcript_text = transcript.text
    else:
        transcript_text = str(transcript)
    
    # Truncate if needed
    if max_length and len(transcript_text) > max_length:
        logger.debug(f"Truncating transcript from {len(transcript_text)} to {max_length} characters")
        transcript_text = transcript_text[:max_length] + "...[truncated]"
    
    # Get template
    template = custom_template or DEFAULT_TEMPLATES.get(summary_type, DEFAULT_TEMPLATES[SummaryType.PARAGRAPH])
    
    # Format template
    prompt = template.format(transcript=transcript_text)
    
    return prompt


def get_all_templates() -> Dict[SummaryType, str]:
    """
    Get all default templates.
    
    Returns:
        Dictionary mapping summary types to their default templates
    """
    return DEFAULT_TEMPLATES.copy()


def register_custom_template(summary_type: SummaryType, template: str) -> None:
    """
    Register a custom template for a summary type.
    
    This updates the default templates dictionary with a custom template.
    
    Args:
        summary_type: Type of summary
        template: Custom prompt template
    """
    DEFAULT_TEMPLATES[summary_type] = template
    logger.debug(f"Registered custom template for summary type: {summary_type}")


def format_prompt_with_context(
    template: str,
    context: Dict[str, str],
) -> str:
    """
    Format a prompt template with context variables.
    
    Args:
        template: Prompt template with {variable} placeholders
        context: Dictionary mapping variable names to values
        
    Returns:
        Formatted prompt string
    """
    try:
        return template.format(**context)
    except KeyError as e:
        logger.warning(f"Missing context variable in template: {e}")
        # Return partially formatted template
        for key, value in context.items():
            template = template.replace(f"{{{key}}}", value)
        return template
