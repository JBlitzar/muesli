"""
Ollama client for the Muesli application.

This module provides a client for interacting with a local Ollama server
for LLM capabilities, such as generating text completions for summarization.
"""

import json
import logging
from typing import Dict, Iterator, Optional, Union

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logging.warning("httpx not available. Ollama client will not work.")

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with a local Ollama server.
    
    This class provides methods for generating text completions using
    a local Ollama server.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3:8b-instruct",
        offline_mode: bool = True,
    ):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Base URL of the Ollama server
            model_name: Name of the model to use
            offline_mode: Whether to run in offline mode (no network access)
        """
        self.base_url = base_url
        self.model_name = model_name
        self.offline_mode = offline_mode
        
        # Check if httpx is available
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx library not available. Please install it with 'pip install httpx'"
            )
        
        # Create HTTP client
        self._client = httpx.Client(timeout=60.0)
        
        logger.info(f"Initialized Ollama client for model {model_name}")
        
        # Verify connection to Ollama server
        self._verify_connection()
    
    def _verify_connection(self) -> None:
        """
        Verify connection to the Ollama server.
        
        Raises:
            ConnectionError: If the connection to the Ollama server fails
        """
        try:
            response = self._client.get(f"{self.base_url}/api/version")
            response.raise_for_status()
            version_info = response.json()
            logger.info(f"Connected to Ollama server version {version_info.get('version', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server at {self.base_url}: {e}")
            raise ConnectionError(f"Failed to connect to Ollama server: {e}")
    
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
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": stream,
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        # Add offline mode if enabled
        if self.offline_mode:
            payload["options"] = {"num_ctx": 2048, "offline": True}
        
        try:
            # Send request to Ollama server
            url = f"{self.base_url}/api/generate"
            
            if stream:
                # Return streaming iterator
                return self._stream_generate(url, payload)
            else:
                # Send non-streaming request
                response = self._client.post(url, json=payload)
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                return result.get("response", "")
                
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            raise RuntimeError(f"Failed to generate text: {e}")
    
    def _stream_generate(self, url: str, payload: Dict) -> Iterator[str]:
        """
        Stream text generation from Ollama.
        
        Args:
            url: API URL
            payload: Request payload
            
        Yields:
            Text chunks as they are generated
            
        Raises:
            RuntimeError: If streaming fails
        """
        try:
            with self._client.stream("POST", url, json=payload, timeout=120.0) as response:
                response.raise_for_status()
                
                # Process streaming response
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON from Ollama: {line}")
                
        except Exception as e:
            logger.error(f"Error during streaming generation: {e}")
            raise RuntimeError(f"Error during streaming generation: {e}")
    
    def check_model_availability(self, model_name: Optional[str] = None) -> bool:
        """
        Check if a model is available on the Ollama server.
        
        Args:
            model_name: Name of the model to check (defaults to the client's model)
            
        Returns:
            True if the model is available, False otherwise
        """
        model = model_name or self.model_name
        
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            models = result.get("models", [])
            
            # Check if model exists
            for model_info in models:
                if model_info.get("name") == model:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            return False
    
    def close(self) -> None:
        """Close the client and release resources."""
        if hasattr(self, '_client'):
            self._client.close()
            logger.debug("Ollama client closed")
