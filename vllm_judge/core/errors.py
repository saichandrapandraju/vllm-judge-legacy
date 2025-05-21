from fastapi import HTTPException


class AdapterError(HTTPException):
    """Base exception for adapter-specific errors."""
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)


class VLLMServerError(AdapterError):
    """Exception raised when there is an error communicating with the vLLM server."""
    def __init__(self, detail: str):
        super().__init__(status_code=502, detail=f"vLLM server error: {detail}")


class PromptTemplateError(AdapterError):
    """Exception raised when there is an error with a prompt template."""
    def __init__(self, detail: str):
        super().__init__(status_code=400, detail=f"Prompt template error: {detail}")


class OutputParsingError(AdapterError):
    """Exception raised when there is an error parsing the output from a judge LLM."""
    def __init__(self, detail: str):
        super().__init__(status_code=422, detail=f"Output parsing error: {detail}")


class TaskNotFoundError(AdapterError):
    """Exception raised when a requested task is not found."""
    def __init__(self, task_id: str):
        super().__init__(status_code=404, detail=f"Task not found: {task_id}")


class TemplateNotFoundError(AdapterError):
    """Exception raised when a requested template is not found."""
    def __init__(self, template_id: str):
        super().__init__(status_code=404, detail=f"Template not found: {template_id}")
