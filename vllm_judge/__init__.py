"""
vLLM Judge - A model-agnostic adapter for enabling LLM-as-a-judge capabilities for vLLM.
"""

__version__ = "0.1.0"

# Export important classes and functions
from vllm_judge.client import (
    VLLMJudgeClient,
    VLLMJudgeError,
    detect_toxicity,
    evaluate_factual_accuracy,
    detect_hallucinations,
    compare_responses,
    evaluate_code,
)

# Optional direct imports for advanced usage
from vllm_judge.services.vllm_client import VLLMClient
from vllm_judge.services.prompt_manager import PromptManager
from vllm_judge.services.output_parser import OutputParser

__all__ = [
    "VLLMJudgeClient",
    "VLLMJudgeError",
    "detect_toxicity",
    "evaluate_factual_accuracy",
    "detect_hallucinations",
    "compare_responses",
    "evaluate_code",
    "VLLMClient",
    "PromptManager",
    "OutputParser",
]