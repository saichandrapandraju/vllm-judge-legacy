from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status

from vllm_judge.core.models import TemplateCreateRequest, TemplateResponse
from vllm_judge.core.errors import TemplateNotFoundError
from vllm_judge.services.prompt_manager import PromptManager


router = APIRouter(prefix="/v1/config", tags=["config"])

# Service instances
prompt_manager = PromptManager()


@router.get("/judge_templates", response_model=List[TemplateResponse])
async def list_judge_templates() -> List[TemplateResponse]:
    """
    List all available judge prompt templates.
    
    Returns:
        List of templates
    """
    templates = prompt_manager.list_templates()
    return [TemplateResponse(**template) for template in templates]


@router.get("/judge_templates/{template_id}", response_model=TemplateResponse)
async def get_judge_template(template_id: str) -> TemplateResponse:
    """
    Get a judge prompt template by ID.
    
    Args:
        template_id: ID of the template to get
        
    Returns:
        The template
        
    Raises:
        TemplateNotFoundError: If the template is not found
    """
    try:
        template = prompt_manager.get_template(template_id)
        return TemplateResponse(**template)
    except TemplateNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/judge_templates", response_model=TemplateResponse)
async def create_judge_template(template_data: TemplateCreateRequest) -> TemplateResponse:
    """
    Create a new judge prompt template.
    
    Args:
        template_data: Data for the new template
        
    Returns:
        The created template
    """
    template = prompt_manager.create_template(template_data.dict())
    return TemplateResponse(**template)


@router.put("/judge_templates/{template_id}", response_model=TemplateResponse)
async def update_judge_template(template_id: str, template_data: TemplateCreateRequest) -> TemplateResponse:
    """
    Update an existing judge prompt template.
    
    Args:
        template_id: ID of the template to update
        template_data: New data for the template
        
    Returns:
        The updated template
        
    Raises:
        TemplateNotFoundError: If the template is not found
    """
    try:
        template = prompt_manager.update_template(template_id, template_data.dict())
        return TemplateResponse(**template)
    except TemplateNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete("/judge_templates/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_judge_template(template_id: str) -> None:
    """
    Delete a judge prompt template.
    
    Args:
        template_id: ID of the template to delete
        
    Raises:
        TemplateNotFoundError: If the template is not found
    """
    try:
        prompt_manager.delete_template(template_id)
    except TemplateNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
