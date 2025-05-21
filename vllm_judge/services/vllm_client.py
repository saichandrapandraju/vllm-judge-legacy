import json
import httpx
import backoff
from typing import Dict, Any, Optional

from vllm_judge.core.config import settings
from vllm_judge.core.errors import VLLMServerError


class VLLMClient:
    """Client for communicating with vLLM server."""
    
    def __init__(self, api_base: Optional[str] = None, api_key: Optional[str] = None, timeout: Optional[int] = None):
        self.api_base = api_base or settings.VLLM_API_BASE
        self.api_key = api_key or settings.VLLM_API_KEY
        self.timeout = timeout or settings.DEFAULT_TIMEOUT
        
        # Ensure the API base URL ends with /v1
        if not self.api_base.endswith("/v1"):
            self.api_base = f"{self.api_base}/v1"
            
    def _get_headers(self) -> Dict[str, str]:
        """Get the HTTP headers for requests to vLLM."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    @backoff.on_exception(
        backoff.expo,
        (httpx.RequestError, httpx.TimeoutException),
        max_tries=settings.MAX_RETRY_ATTEMPTS,
    )
    async def generate_completion(self, model: str, messages: list, sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a completion using the vLLM server's chat completions API.
        
        Args:
            model: The model ID to use for generation
            messages: The messages to send to the model
            sampling_params: Parameters for the generation
            
        Returns:
            The response from the vLLM server
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=self._get_headers(),
                    json={
                        "model": model,
                        "messages": messages,
                        **sampling_params,
                    },
                )
                
                if response.status_code != 200:
                    raise VLLMServerError(
                        f"Failed to generate completion: {response.status_code} - {response.text}"
                    )
                
                return response.json()
        except (httpx.RequestError, httpx.TimeoutException) as e:
            raise VLLMServerError(f"Network error when contacting vLLM server: {str(e)}")
        except json.JSONDecodeError:
            raise VLLMServerError("Failed to parse response from vLLM server")
