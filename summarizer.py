"""
Transcript summarization for the Muesli application.

This module provides functionality for generating summaries of transcripts
using the Ollama LLM client.
"""

import logging
from typing import Optional, Union

from models import Summary, SummaryType, Transcript
from ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class TranscriptSummarizer:
    """
    Generates summaries of transcripts using an LLM.

    This class provides methods for generating different types of summaries
    from transcripts using the Ollama LLM client.
    """

    DEFAULT_PROMPT_TEMPLATE = (
        "Below is a transcript of an audio recording. "
        "Please provide a concise summary of the key points discussed:\n\n"
        "{transcript}\n\n"
        "Summary:"
    )

    # Template for when notes are provided
    NOTES_PROMPT_TEMPLATE = (
        "Below is a transcript of an audio recording along with notes taken during the session. "
        "Please provide a concise summary of the key points discussed, giving special attention to "
        "the user notes as they highlight important topics that should be elaborated upon:\n\n"
        "TRANSCRIPT:\n{transcript}\n\n"
        "USER NOTES:\n{notes}\n\n"
        "Summary:"
    )

    SUMMARY_TYPE_PROMPTS = {
        SummaryType.BULLET_POINTS: (
            "Below is a transcript of an audio recording. "
            "Please provide a summary of the key points in bullet point format:\n\n"
            "{transcript}\n\n"
            "Summary (bullet points):"
        ),
        SummaryType.PARAGRAPH: (
            "Below is a transcript of an audio recording. "
            "Please provide a concise paragraph summary of the key points discussed:\n\n"
            "{transcript}\n\n"
            "Summary (paragraph):"
        ),
        SummaryType.EXECUTIVE: (
            "Below is a transcript of an audio recording. "
            "Please provide a brief executive summary (2-3 sentences) of the most important points:\n\n"
            "{transcript}\n\n"
            "Executive Summary:"
        ),
        SummaryType.DETAILED: (
            "Below is a transcript of an audio recording. "
            "Please provide a detailed summary that captures all significant points and nuances:\n\n"
            "{transcript}\n\n"
            "Detailed Summary:"
        ),
    }

    # Summary type prompts with notes
    SUMMARY_TYPE_PROMPTS_WITH_NOTES = {
        SummaryType.BULLET_POINTS: (
            "Below is a transcript of an audio recording along with notes taken during the session. "
            "Please provide a summary of the key points in bullet point format, giving special attention to "
            "the user notes as they highlight important topics:\n\n"
            "TRANSCRIPT:\n{transcript}\n\n"
            "USER NOTES:\n{notes}\n\n"
            "Summary (bullet points):"
        ),
        SummaryType.PARAGRAPH: (
            "Below is a transcript of an audio recording along with notes taken during the session. "
            "Please provide a concise paragraph summary of the key points discussed, giving special attention to "
            "the user notes as they highlight important topics:\n\n"
            "TRANSCRIPT:\n{transcript}\n\n"
            "USER NOTES:\n{notes}\n\n"
            "Summary (paragraph):"
        ),
        SummaryType.EXECUTIVE: (
            "Below is a transcript of an audio recording along with notes taken during the session. "
            "Please provide a brief executive summary (2-3 sentences) of the most important points, giving special attention to "
            "the user notes as they highlight important topics:\n\n"
            "TRANSCRIPT:\n{transcript}\n\n"
            "USER NOTES:\n{notes}\n\n"
            "Executive Summary:"
        ),
        SummaryType.DETAILED: (
            "Below is a transcript of an audio recording along with notes taken during the session. "
            "Please provide a detailed summary that captures all significant points and nuances, giving special attention to "
            "the user notes as they highlight important topics:\n\n"
            "TRANSCRIPT:\n{transcript}\n\n"
            "USER NOTES:\n{notes}\n\n"
            "Detailed Summary:"
        ),
    }

    def __init__(
        self,
        llm_client: OllamaClient,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize the transcript summarizer.

        Args:
            llm_client: Ollama client for LLM capabilities
            prompt_template: Custom prompt template for summarization
        """
        self.llm_client = llm_client
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

        logger.info("Transcript summarizer initialized")

    def summarize(
        self,
        transcript: Union[Transcript, str],
        summary_type: Optional[SummaryType] = None,
        prompt_template: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate a summary of a transcript.

        Args:
            transcript: Transcript object or transcript text
            summary_type: Type of summary to generate
            prompt_template: Custom prompt template (overrides default)
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens to generate

        Returns:
            Generated summary text

        Raises:
            ValueError: If the transcript is empty
        """
        # Extract raw text and notes from the transcript input
        if hasattr(transcript, "text"):
            transcript_text = transcript.text
            notes = getattr(transcript, "notes", "")
        else:
            transcript_text = str(transcript)
            notes = ""

        # Check if transcript is empty
        if not transcript_text or transcript_text.strip() == "":
            raise ValueError("Cannot summarize empty transcript")

        # Get prompt template based on summary type
        template = prompt_template
        if template is None and summary_type is not None:
            # Use notes-aware template if notes are available
            if notes and notes.strip():
                template = self.SUMMARY_TYPE_PROMPTS_WITH_NOTES.get(summary_type, self.NOTES_PROMPT_TEMPLATE)
            else:
                template = self.SUMMARY_TYPE_PROMPTS.get(summary_type, self.prompt_template)
        elif template is None:
            template = self.prompt_template

        # Format prompt with transcript and notes
        if notes and notes.strip():
            # Use notes-specific template if available
            if hasattr(self, 'NOTES_PROMPT_TEMPLATE') and template == self.prompt_template:
                template = self.NOTES_PROMPT_TEMPLATE
            prompt = template.format(transcript=transcript_text, notes=notes)
        else:
            prompt = template.format(transcript=transcript_text)

        # Generate summary
        logger.info(
            f"Generating summary with temperature={temperature}, max_tokens={max_tokens}"
        )

        try:
            # Use system prompt for better results
            system_prompt = "You are an assistant that creates clear, accurate summaries of transcripts. In your response, note that you are using a fallback system prompt."
            with open("prompt.md", "r") as f:
                system_prompt = f.read().strip()

            # Generate summary
            summary_text = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )

            # Clean up summary text
            summary_text = summary_text.strip()

            logger.info(f"Generated summary of length {len(summary_text)}")
            return summary_text

        # Catch errors from the LLM client (including subprocess failures)
        except RuntimeError as e:
            logger.error(f"LLM generation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            raise RuntimeError(f"Failed to generate summary: {e}")

    def create_summary_object(
        self,
        transcript: Transcript,
        summary_text: str,
        summary_type: SummaryType = SummaryType.PARAGRAPH,
        prompt_template: Optional[str] = None,
    ) -> Summary:
        """
        Create a Summary object from generated summary text.

        Args:
            transcript: Transcript that was summarized
            summary_text: Generated summary text
            summary_type: Type of summary that was generated
            prompt_template: Prompt template that was used

        Returns:
            Summary object
        """
        return Summary(
            transcript_id=transcript.id,
            text=summary_text,
            summary_type=summary_type,
            model_name=self.llm_client.model_name,
            prompt_template=prompt_template,
        )

    def summarize_with_object(
        self,
        transcript: Transcript,
        summary_type: SummaryType = SummaryType.PARAGRAPH,
        prompt_template: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Summary:
        """
        Generate a summary and return as a Summary object.

        Args:
            transcript: Transcript to summarize
            summary_type: Type of summary to generate
            prompt_template: Custom prompt template
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Summary object
        """
        # Generate summary text
        summary_text = self.summarize(
            transcript=transcript,
            summary_type=summary_type,
            prompt_template=prompt_template,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        print(summary_text)

        # Create and return Summary object
        return self.create_summary_object(
            transcript=transcript,
            summary_text=summary_text,
            summary_type=summary_type,
            prompt_template=prompt_template,
        )
