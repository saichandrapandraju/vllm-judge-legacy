from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Status of an evaluation task."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class VLLMSamplingParams(BaseModel):
    """Parameters for vLLM sampling."""
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    stop: Optional[List[str]] = None
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0


class CustomPromptSegments(BaseModel):
    """Custom segments for building a prompt."""
    system_message: Optional[str] = None
    user_instruction_prefix: Optional[str] = None
    user_instruction_suffix: Optional[str] = None


class SingleEvaluationRequest(BaseModel):
    """Request model for evaluating a single text."""
    judge_model_id: str = Field(..., description="ID of the vLLM-hosted model to use as judge")
    text_to_evaluate: str = Field(..., description="The content to be judged")
    evaluation_criteria: str = Field(..., description="Detailed rubric, instructions, or description of the evaluation task")
    prompt_template_id: Optional[str] = Field(None, description="Reference to a pre-configured prompt template")
    custom_prompt_segments: Optional[CustomPromptSegments] = Field(None, description="Optional segments to override parts of a template")
    output_format_instruction: Optional[str] = Field(None, description="Instructions for the judge LLM on how to format its output")
    vllm_sampling_params: Optional[VLLMSamplingParams] = Field(None, description="Parameters for vLLM inference")
    provide_reasoning: Optional[bool] = Field(False, description="Whether to instruct the judge to provide reasoning")


class PairwiseComparisonRequest(BaseModel):
    """Request model for comparing two texts."""
    judge_model_id: str = Field(..., description="ID of the vLLM-hosted model to use as judge")
    text_A: str = Field(..., description="First text to compare")
    text_B: str = Field(..., description="Second text to compare")
    comparison_criteria: str = Field(..., description="Criteria for comparing the two texts")
    prompt_template_id: Optional[str] = Field(None, description="Reference to a pre-configured prompt template")
    custom_prompt_segments: Optional[CustomPromptSegments] = Field(None, description="Optional segments to override parts of a template")
    output_format_instruction: Optional[str] = Field(None, description="Instructions for the judge LLM on how to format its output")
    vllm_sampling_params: Optional[VLLMSamplingParams] = Field(None, description="Parameters for vLLM inference")
    provide_reasoning: Optional[bool] = Field(False, description="Whether to instruct the judge to provide reasoning")


class EvaluationResult(BaseModel):
    """Result of an evaluation."""
    judgment: Any = Field(..., description="The parsed judgment from the judge LLM")
    raw_judge_output: str = Field(..., description="The raw text response from the judge LLM")
    reasoning: Optional[str] = Field(None, description="Optional reasoning provided by the judge")


class EvaluationResponse(BaseModel):
    """Response model for evaluation requests."""
    evaluation_id: str = Field(..., description="Unique ID for this evaluation task")
    status: TaskStatus = Field(..., description="Current status of the evaluation task")
    result: Optional[EvaluationResult] = Field(None, description="Result of the evaluation, if completed")
    error_message: Optional[str] = Field(None, description="Error message, if the evaluation failed")


class TemplateCreateRequest(BaseModel):
    """Request model for creating a new prompt template."""
    template_name: str = Field(..., description="Name of the template")
    target_judge_model_family: Optional[str] = Field(None, description="Model family this template is optimized for")
    prompt_structure: Dict[str, Any] = Field(..., description="Structure of the prompt template")
    output_parser_rules: Optional[Dict[str, Any]] = Field(None, description="Rules for parsing the output")
    description: Optional[str] = Field(None, description="Description of the template")


class TemplateResponse(BaseModel):
    """Response model for template operations."""
    template_id: str = Field(..., description="Unique ID of the template")
    template_name: str = Field(..., description="Name of the template")
    target_judge_model_family: Optional[str] = Field(None, description="Model family this template is optimized for")
    prompt_structure: Dict[str, Any] = Field(..., description="Structure of the prompt template")
    output_parser_rules: Optional[Dict[str, Any]] = Field(None, description="Rules for parsing the output")
    description: Optional[str] = Field(None, description="Description of the template")
