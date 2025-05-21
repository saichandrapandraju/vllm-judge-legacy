import json
from typing import Dict, Any, Optional, List

import requests


class SyncVLLMClient:
    """A simple synchronous client for vLLM API."""
    
    def __init__(self, 
                 api_base: str, 
                 api_key: Optional[str] = None, 
                 timeout: int = 60):
        """
        Initialize the synchronous vLLM client.
        
        Args:
            api_base: Base URL for the vLLM API
            api_key: API key (optional)
            timeout: Timeout for requests in seconds
        """
        self.api_base = api_base.rstrip("/")
        if not self.api_base.endswith("/v1"):
            self.api_base = f"{self.api_base}/v1"
        self.api_key = api_key
        self.timeout = timeout
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def generate_completion(self, 
                           model: str, 
                           messages: List[Dict[str, str]], 
                           sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a completion using vLLM's chat completion API.
        
        This is a synchronous version that replaces the async one.
        
        Args:
            model: Model ID to use
            messages: List of message dictionaries
            sampling_params: Parameters for sampling
            
        Returns:
            Completion response
        """
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self._get_headers(),
                json={
                    "model": model,
                    "messages": messages,
                    **sampling_params
                },
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_message = f"Failed to generate completion: {response.status_code} - {response.text}"
                raise ValueError(error_message)
            
            return response.json()
        except requests.RequestException as e:
            raise ValueError(f"Network error when contacting vLLM server: {str(e)}")
        except json.JSONDecodeError:
            raise ValueError("Failed to parse response from vLLM server")