import uuid
import asyncio
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, BackgroundTasks

from vllm_judge.core.models import (
    SingleEvaluationRequest,
    PairwiseComparisonRequest,
    EvaluationResponse,
    TaskStatus,
    EvaluationResult,
)
from vllm_judge.core.errors import TaskNotFoundError, TemplateNotFoundError
from vllm_judge.services.vllm_client import VLLMClient
from vllm_judge.services.prompt_manager import PromptManager
from vllm_judge.services.output_parser import OutputParser


router = APIRouter(prefix="/v1/evaluate", tags=["evaluate"])

#TODO: persistent store..?
evaluation_tasks: Dict[str, Dict[str, Any]] = {}

# Service instances
vllm_client = VLLMClient()
prompt_manager = PromptManager()
output_parser = OutputParser()


async def process_single_evaluation(
    evaluation_id: str,
    request: SingleEvaluationRequest,
    vllm_client: VLLMClient,
    prompt_manager: PromptManager,
    output_parser: OutputParser,
) -> None:
    """
    Process a single evaluation task in the background.
    
    Args:
        evaluation_id: ID of the evaluation task
        request: The evaluation request
        vllm_client: Instance of VLLMClient
        prompt_manager: Instance of PromptManager
        output_parser: Instance of OutputParser
    """
    try:
        # Update task status
        evaluation_tasks[evaluation_id]["status"] = TaskStatus.RUNNING
        
        # Generate prompt
        messages = prompt_manager.generate_single_evaluation_prompt(
            text_to_evaluate=request.text_to_evaluate,
            evaluation_criteria=request.evaluation_criteria,
            prompt_template_id=request.prompt_template_id,
            custom_prompt_segments=request.custom_prompt_segments,
            output_format_instruction=request.output_format_instruction,
            provide_reasoning=request.provide_reasoning,
        )
        
        # Get sampling parameters
        sampling_params = request.vllm_sampling_params.dict() if request.vllm_sampling_params else {}
        
        # Generate completion
        response = await vllm_client.generate_completion(
            model=request.judge_model_id,
            messages=messages,
            sampling_params=sampling_params,
        )
        
        # Extract the response text
        raw_output = response["choices"][0]["message"]["content"]
        
        # Get parser rules if a template was used
        parser_rules = None
        if request.prompt_template_id:
            try:
                template = prompt_manager.get_template(request.prompt_template_id)
                parser_rules = template.get("output_parser_rules")
            except TemplateNotFoundError:
                pass
        
        # Parse the output
        parsed_result = output_parser.parse_single_evaluation(
            raw_output=raw_output,
            template_id=request.prompt_template_id,
            parser_rules=parser_rules,
            provide_reasoning=request.provide_reasoning,
        )
        
        # Update task with result
        evaluation_tasks[evaluation_id].update({
            "status": TaskStatus.COMPLETED,
            "result": EvaluationResult(
                judgment=parsed_result["judgment"],
                raw_judge_output=raw_output,
                reasoning=parsed_result["reasoning"],
            ),
        })
    except Exception as e:
        # Update task with error
        evaluation_tasks[evaluation_id].update({
            "status": TaskStatus.FAILED,
            "error_message": str(e),
        })


async def process_pairwise_comparison(
    evaluation_id: str,
    request: PairwiseComparisonRequest,
    vllm_client: VLLMClient,
    prompt_manager: PromptManager,
    output_parser: OutputParser,
) -> None:
    """
    Process a pairwise comparison task in the background.
    
    Args:
        evaluation_id: ID of the evaluation task
        request: The comparison request
        vllm_client: Instance of VLLMClient
        prompt_manager: Instance of PromptManager
        output_parser: Instance of OutputParser
    """
    try:
        # Update task status
        evaluation_tasks[evaluation_id]["status"] = TaskStatus.RUNNING
        
        # Generate prompt
        messages = prompt_manager.generate_pairwise_comparison_prompt(
            text_A=request.text_A,
            text_B=request.text_B,
            comparison_criteria=request.comparison_criteria,
            prompt_template_id=request.prompt_template_id,
            custom_prompt_segments=request.custom_prompt_segments,
            output_format_instruction=request.output_format_instruction,
            provide_reasoning=request.provide_reasoning,
        )
        
        # Get sampling parameters
        sampling_params = request.vllm_sampling_params.dict() if request.vllm_sampling_params else {}
        
        # Generate completion
        response = await vllm_client.generate_completion(
            model=request.judge_model_id,
            messages=messages,
            sampling_params=sampling_params,
        )
        
        # Extract the response text
        raw_output = response["choices"][0]["message"]["content"]
        
        # Get parser rules if a template was used
        parser_rules = None
        if request.prompt_template_id:
            try:
                template = prompt_manager.get_template(request.prompt_template_id)
                parser_rules = template.get("output_parser_rules")
            except TemplateNotFoundError:
                pass
        
        # Parse the output
        parsed_result = output_parser.parse_pairwise_comparison(
            raw_output=raw_output,
            template_id=request.prompt_template_id,
            parser_rules=parser_rules,
            provide_reasoning=request.provide_reasoning,
        )
        
        # Update task with result
        evaluation_tasks[evaluation_id].update({
            "status": TaskStatus.COMPLETED,
            "result": EvaluationResult(
                judgment=parsed_result["judgment"],
                raw_judge_output=raw_output,
                reasoning=parsed_result["reasoning"],
            ),
        })
    except Exception as e:
        # Update task with error
        evaluation_tasks[evaluation_id].update({
            "status": TaskStatus.FAILED,
            "error_message": str(e),
        })


@router.post("/single_response", response_model=EvaluationResponse)
async def evaluate_single_response(
    request: SingleEvaluationRequest,
    background_tasks: BackgroundTasks,
) -> EvaluationResponse:
    """
    Evaluate a single piece of text based on specified criteria.
    
    Args:
        request: The evaluation request
        background_tasks: FastAPI BackgroundTasks
        
    Returns:
        Evaluation response with task ID
    """
    # Generate a unique ID for this evaluation
    evaluation_id = str(uuid.uuid4())
    
    # Create a new task
    evaluation_tasks[evaluation_id] = {
        "evaluation_id": evaluation_id,
        "status": TaskStatus.PENDING,
        "result": None,
        "error_message": None,
    }
    
    # Process the evaluation in the background
    background_tasks.add_task(
        process_single_evaluation,
        evaluation_id,
        request,
        vllm_client,
        prompt_manager,
        output_parser,
    )
    
    # Return the response
    return EvaluationResponse(
        evaluation_id=evaluation_id,
        status=TaskStatus.PENDING,
    )


@router.post("/pairwise_comparison", response_model=EvaluationResponse)
async def pairwise_comparison(
    request: PairwiseComparisonRequest,
    background_tasks: BackgroundTasks,
) -> EvaluationResponse:
    """
    Compare two pieces of text using a judge LLM.
    
    Args:
        request: The comparison request
        background_tasks: FastAPI BackgroundTasks
        
    Returns:
        Evaluation response with task ID
    """
    # Generate a unique ID for this evaluation
    evaluation_id = str(uuid.uuid4())
    
    # Create a new task
    evaluation_tasks[evaluation_id] = {
        "evaluation_id": evaluation_id,
        "status": TaskStatus.PENDING,
        "result": None,
        "error_message": None,
    }
    
    # Process the evaluation in the background
    background_tasks.add_task(
        process_pairwise_comparison,
        evaluation_id,
        request,
        vllm_client,
        prompt_manager,
        output_parser,
    )
    
    # Return the response
    return EvaluationResponse(
        evaluation_id=evaluation_id,
        status=TaskStatus.PENDING,
    )


@router.get("/status/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation_status(evaluation_id: str) -> EvaluationResponse:
    """
    Get the status of an evaluation task.
    
    Args:
        evaluation_id: ID of the evaluation task
        
    Returns:
        Evaluation response with current status and result if completed
        
    Raises:
        TaskNotFoundError: If the task is not found
    """
    # Check if the task exists
    if evaluation_id not in evaluation_tasks:
        raise TaskNotFoundError(evaluation_id)
    
    # Get the task
    task = evaluation_tasks[evaluation_id]
    
    # Return the response
    return EvaluationResponse(
        evaluation_id=task["evaluation_id"],
        status=task["status"],
        result=task["result"],
        error_message=task["error_message"],
    )