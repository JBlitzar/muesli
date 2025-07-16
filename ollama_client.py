"""
Ollama client for the Muesli application.

This module provides a client for interacting with a local Ollama installation
for LLM capabilities, such as generating text completions for summarization.
"""

import logging
import subprocess
import shlex
from typing import Dict, Iterator, Optional, Union

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with a local Ollama installation.
    
    This class provides methods for generating text completions using
    the Ollama CLI directly via subprocess.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",  # Kept for API compatibility
        model_name: str = "llama3:8b-instruct",
        offline_mode: bool = True,
    ):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Ignored (kept for API compatibility)
            model_name: Name of the model to use
            offline_mode: Whether to run in offline mode (no network access)
        """
        self.model_name = model_name
        self.offline_mode = offline_mode
        
        # Verify Ollama is installed
        self._verify_installation()
        
        logger.info(f"Initialized Ollama client for model {model_name}")
    
    def _verify_installation(self) -> None:
        """
        Verify Ollama is installed and accessible.
        
        Raises:
            RuntimeError: If Ollama is not installed or accessible
        """
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Ollama command failed: {result.stderr}")
                
            logger.info("Ollama installation verified")
            
        except FileNotFoundError:
            logger.error("Ollama not found. Please install Ollama.")
            raise RuntimeError("Ollama not found. Please install Ollama.")
        except Exception as e:
            logger.error(f"Failed to verify Ollama installation: {e}")
            raise RuntimeError(f"Failed to verify Ollama installation: {e}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        """
        Generate text completion using Ollama.
        
        Args:
            prompt: The prompt to generate text from
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Generated text or iterator of text chunks if streaming
            
        Raises:
            RuntimeError: If generation fails
        """
        # Build the command
        cmd = ["ollama", "run"]
        
        # Add model name
        cmd.append(self.model_name)
        
        # Add parameters
        if system_prompt:
            cmd.extend(["--system", system_prompt])
        
        cmd.extend(["--temperature", str(temperature)])
        cmd.extend(["--num-predict", str(max_tokens)])
        
        # Add offline mode if enabled
        if self.offline_mode:
            cmd.append("--nowordwrap")  # Prevent word wrapping in output
        
        # Add the prompt (must be last argument)
        cmd.append(prompt)
        
        logger.debug(f"Executing command: {' '.join(cmd)}")
        
        try:
            if stream:
                return self._stream_generate(cmd)
            else:
                # For non-streaming, capture all output at once
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                return result.stdout.strip()
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Ollama command failed (code {e.returncode}): {e.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            raise RuntimeError(f"Failed to generate text: {e}")
    
    def _stream_generate(self, cmd: list) -> Iterator[str]:
        """
        Stream text generation from Ollama.
        
        Args:
            cmd: Command list to execute
            
        Yields:
            Text chunks as they are generated
            
        Raises:
            RuntimeError: If streaming fails
        """
        try:
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Read output line by line
            for line in process.stdout:
                yield line.rstrip('\n')
            
            # Check for any errors after process completes
            return_code = process.wait()
            if return_code != 0:
                stderr = process.stderr.read()
                logger.error(f"Ollama command failed (code {return_code}): {stderr}")
                raise RuntimeError(f"Ollama command failed: {stderr}")
                
        except Exception as e:
            logger.error(f"Error during streaming generation: {e}")
            raise RuntimeError(f"Error during streaming generation: {e}")
    
    def check_model_availability(self, model_name: Optional[str] = None) -> bool:
        """
        Check if a model is available on the local Ollama installation.
        
        Args:
            model_name: Name of the model to check (defaults to the client's model)
            
        Returns:
            True if the model is available, False otherwise
        """
        model = model_name or self.model_name
        
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Check if model exists in the output
            return model in result.stdout
            
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            return False
    
    def close(self) -> None:
        """Close the client and release resources (no-op for subprocess approach)."""
        logger.debug("Ollama client closed")
