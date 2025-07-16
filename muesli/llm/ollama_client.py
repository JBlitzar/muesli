"""
Ollama client for the Muesli application.

This module provides a client for interacting with the Ollama API,
which serves local LLMs. It handles communication with the Ollama server,
model management, and inference requests.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Base exception for Ollama-related errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Exception raised when connection to Ollama server fails."""
    pass


class OllamaModelError(OllamaError):
    """Exception raised when there's an issue with a model."""
    pass


class OllamaClient:
    """
    Client for interacting with the Ollama API.
    
    This class provides methods for generating text, checking model
    availability, and managing models with the Ollama API.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3:8b-instruct",
        offline_mode: bool = True,
        request_timeout: float = 60.0,
    ):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Base URL for the Ollama API
            model_name: Default model to use for generation
            offline_mode: Whether to run in offline mode (no network access for the model)
            request_timeout: Timeout for API requests in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.offline_mode = offline_mode
        self.request_timeout = request_timeout
        
        # Create HTTP client
        self.client = httpx.Client(timeout=request_timeout)
        
        # Check if Ollama is running
        self._check_server_connection()
        
        logger.info(f"Ollama client initialized (model={model_name}, offline={offline_mode})")
    
    def _check_server_connection(self) -> None:
        """
        Check if the Ollama server is running.
        
        Raises:
            OllamaConnectionError: If connection to Ollama server fails
        """
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            logger.debug("Successfully connected to Ollama server")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server at {self.base_url}: {e}")
            raise OllamaConnectionError(
                f"Failed to connect to Ollama server at {self.base_url}. "
                f"Please ensure Ollama is installed and running: {e}"
            )
    
    def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text using the Ollama API.
        
        Args:
            prompt: The prompt to generate text from
            model_name: Model to use (defaults to self.model_name)
            system_prompt: Optional system prompt for instruction models
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences that will stop generation
            
        Returns:
            Generated text
            
        Raises:
            OllamaConnectionError: If connection to Ollama server fails
            OllamaModelError: If there's an issue with the model
        """
        model = model_name or self.model_name
        
        # Check if model is available
        if not self.is_model_available(model):
            raise OllamaModelError(
                f"Model '{model}' is not available. Please pull it with 'ollama pull {model}'"
            )
        
        # Prepare request data
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system_prompt:
            data["system"] = system_prompt
            
        if stop_sequences:
            data["options"]["stop"] = stop_sequences
            
        if self.offline_mode:
            data["options"]["num_ctx"] = 0  # Disable external API calls
        
        # Make request
        try:
            logger.debug(f"Generating text with model '{model}'")
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json=data,
            )
            response.raise_for_status()
            result = response.json()
            
            return result.get("response", "")
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during generation: {e.response.status_code} - {e.response.text}")
            raise OllamaModelError(f"Error generating text: {e}")
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise OllamaConnectionError(f"Failed to communicate with Ollama server: {e}")
    
    def generate_streaming(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
        stop_sequences: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate text using the Ollama API with streaming.
        
        Args:
            prompt: The prompt to generate text from
            model_name: Model to use (defaults to self.model_name)
            system_prompt: Optional system prompt for instruction models
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences that will stop generation
            
        Yields:
            Generated text chunks
            
        Raises:
            OllamaConnectionError: If connection to Ollama server fails
            OllamaModelError: If there's an issue with the model
        """
        model = model_name or self.model_name
        
        # Check if model is available
        if not self.is_model_available(model):
            raise OllamaModelError(
                f"Model '{model}' is not available. Please pull it with 'ollama pull {model}'"
            )
        
        # Prepare request data
        data = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system_prompt:
            data["system"] = system_prompt
            
        if stop_sequences:
            data["options"]["stop"] = stop_sequences
            
        if self.offline_mode:
            data["options"]["num_ctx"] = 0  # Disable external API calls
        
        # Make request
        try:
            logger.debug(f"Generating text with streaming using model '{model}'")
            with self.client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=data,
                timeout=self.request_timeout,
            ) as response:
                response.raise_for_status()
                
                chunks = []
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    
                    try:
                        chunk_data = json.loads(line)
                        chunk = chunk_data.get("response", "")
                        chunks.append(chunk)
                        yield chunk
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse streaming response: {line}")
                
                return chunks
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during streaming generation: {e.response.status_code} - {e.response.text}")
            raise OllamaModelError(f"Error generating text: {e}")
            
        except Exception as e:
            logger.error(f"Error generating text with streaming: {e}")
            raise OllamaConnectionError(f"Failed to communicate with Ollama server: {e}")
    
    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a model is available locally.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if the model is available, False otherwise
        """
        try:
            models = self.list_models()
            return model_name in models
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """
        List available models.
        
        Returns:
            List of available model names
            
        Raises:
            OllamaConnectionError: If connection to Ollama server fails
        """
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            models = []
            for model in response.json().get("models", []):
                models.append(model.get("name"))
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise OllamaConnectionError(f"Failed to list models: {e}")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
            
        Raises:
            OllamaConnectionError: If connection to Ollama server fails
            OllamaModelError: If the model is not available
        """
        try:
            response = self.client.get(f"{self.base_url}/api/show", params={"name": model_name})
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise OllamaModelError(f"Model '{model_name}' not found")
            logger.error(f"HTTP error getting model info: {e.response.status_code} - {e.response.text}")
            raise OllamaModelError(f"Error getting model info: {e}")
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            raise OllamaConnectionError(f"Failed to get model info: {e}")
    
    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from the Ollama library.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            OllamaConnectionError: If connection to Ollama server fails
        """
        if self.offline_mode:
            logger.warning("Cannot pull model in offline mode")
            return False
        
        try:
            logger.info(f"Pulling model '{model_name}'")
            response = self.client.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
            )
            response.raise_for_status()
            
            # Wait for model to be available
            max_attempts = 10
            for attempt in range(max_attempts):
                if self.is_model_available(model_name):
                    logger.info(f"Model '{model_name}' pulled successfully")
                    return True
                
                logger.debug(f"Waiting for model to be available (attempt {attempt+1}/{max_attempts})")
                time.sleep(2)
            
            logger.warning(f"Model '{model_name}' not available after pulling")
            return False
            
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            raise OllamaConnectionError(f"Failed to pull model: {e}")
    
    def close(self) -> None:
        """Close the client and release resources."""
        if hasattr(self, 'client'):
            self.client.close()
