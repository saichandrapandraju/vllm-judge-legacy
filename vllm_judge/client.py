import time
from typing import Dict, Any, Optional, List

import requests

# Import components for direct mode
from vllm_judge.services.prompt_manager import PromptManager
from vllm_judge.services.output_parser import OutputParser
from vllm_judge.services.sync_vllm_client import SyncVLLMClient




class VLLMJudgeClient:
    """Client for the vLLM Judge adapter with hybrid mode support."""
    
    def __init__(
        self, 
        base_url: Optional[str] = "http://localhost:8000/v1",
        timeout: int = 60,
        direct_mode: bool = False,
        vllm_api_base: Optional[str] = None,
        vllm_api_key: Optional[str] = None,
        template_path: Optional[str] = None
    ):
        """
        Initialize the client with options for both server and direct modes.
        
        Args:
            base_url: Base URL of the vLLM Judge adapter API (for server mode)
            timeout: Default timeout in seconds for requests
            direct_mode: Whether to use direct mode (bypass server and talk directly to vLLM)
            vllm_api_base: Base URL of the vLLM API (required for direct mode)
            vllm_api_key: API key for the vLLM API (optional)
            template_path: Path to template storage file (optional, for direct mode)
        """
        self.direct_mode = direct_mode
        self.timeout = timeout
        
        if direct_mode:
            if not vllm_api_base:
                raise ValueError("vllm_api_base is required when using direct mode")
                
            # Initialize components for direct mode - using synchronous client
            self.vllm_client = SyncVLLMClient(
                api_base=vllm_api_base,
                api_key=vllm_api_key,
                timeout=timeout
            )
            self.prompt_manager = PromptManager(template_path)
            self.output_parser = OutputParser()
        else:
            # Server mode configuration
            if base_url is None:
                raise ValueError("base_url is required when not using direct mode")
                
            self.base_url = base_url.rstrip("/")
            if not self.base_url.endswith("/v1"):
                self.base_url = f"{self.base_url}/v1"
    
    def evaluate_text(
        self,
        text: str,
        evaluation_criteria: str,
        judge_model_id: str,
        prompt_template_id: Optional[str] = None,
        custom_prompt_segments: Optional[Dict[str, str]] = None,
        output_format_instruction: Optional[str] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        provide_reasoning: bool = False,
        wait: bool = True,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single text based on the provided criteria.
        
        Args:
            text: The text to evaluate
            evaluation_criteria: Criteria for the evaluation
            judge_model_id: ID of the model to use as judge
            prompt_template_id: ID of the prompt template to use (optional)
            custom_prompt_segments: Custom segments for the prompt (optional)
            output_format_instruction: Instructions for output format (optional)
            sampling_params: Parameters for vLLM sampling (optional)
            provide_reasoning: Whether to request reasoning
            wait: Whether to wait for the result (only for server mode)
            timeout: Timeout in seconds (only used if wait is True)
            
        Returns:
            Evaluation response
        """
        if self.direct_mode:
            return self._evaluate_text_direct(
                text=text,
                evaluation_criteria=evaluation_criteria,
                judge_model_id=judge_model_id,
                prompt_template_id=prompt_template_id,
                custom_prompt_segments=custom_prompt_segments,
                output_format_instruction=output_format_instruction,
                sampling_params=sampling_params,
                provide_reasoning=provide_reasoning
            )
        else:
            return self._evaluate_text_server(
                text=text,
                evaluation_criteria=evaluation_criteria,
                judge_model_id=judge_model_id,
                prompt_template_id=prompt_template_id,
                custom_prompt_segments=custom_prompt_segments,
                output_format_instruction=output_format_instruction,
                sampling_params=sampling_params,
                provide_reasoning=provide_reasoning,
                wait=wait,
                timeout=timeout
            )
    
    def _clean_output(self, output: str) -> str:
        return output.removeprefix("```json").removeprefix("```").removesuffix("```")
    
    def _extract_reasoning(self, output: str) -> str:
        paragraphs = output.split("\n")
        if len(paragraphs) > 1:
            return "\n".join(paragraphs[1:]).strip().removeprefix("Reasoning:").strip()
        return ""


    def _evaluate_text_direct(
        self,
        text: str,
        evaluation_criteria: str,
        judge_model_id: str,
        prompt_template_id: Optional[str] = None,
        custom_prompt_segments: Optional[Dict[str, str]] = None,
        output_format_instruction: Optional[str] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        provide_reasoning: bool = False
    ) -> Dict[str, Any]:
        """Direct mode implementation of evaluate_text."""
        # If a template is explicitly requested, use it
        if prompt_template_id:
            # Generate the prompt using the specified template
            messages = self.prompt_manager.generate_single_evaluation_prompt(
                text_to_evaluate=text,
                evaluation_criteria=evaluation_criteria,
                prompt_template_id=prompt_template_id,
                custom_prompt_segments=custom_prompt_segments,
                output_format_instruction=output_format_instruction,
                provide_reasoning=provide_reasoning
            )
            
            # Get parser rules if a template was used
            parser_rules = None
            try:
                template = self.prompt_manager.get_template(prompt_template_id)
                parser_rules = template.get("output_parser_rules")
            except Exception:
                pass
        else:
            # No template specified - use a generic approach
            # Create system message
            system_message = "You are an expert evaluator. Your task is to evaluate the provided content based on the given criteria."
            
            # Override with custom if provided
            if custom_prompt_segments and custom_prompt_segments.get("system_message"):
                system_message = custom_prompt_segments.get("system_message")
            
            # Create user message
            user_message = f"Please evaluate the following content based on these criteria:\n\n{evaluation_criteria}\n\nContent to evaluate:\n\n{text}"
            
            # Add reasoning request if needed
            if provide_reasoning:
                reasoning_request = "\n\nThen, on a new line, explain your reasoning starting with 'Reasoning:'"
                user_message += reasoning_request
            
            # Add output format instruction if provided
            if output_format_instruction:
                user_message += f"\n\n{output_format_instruction}"
            
            # Create messages list
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            # No specific parser rules in generic mode
            parser_rules = None
        
        # Prepare sampling parameters
        vllm_params = sampling_params or {}
        
        # Call vLLM directly using our synchronous client
        completion_response = self.vllm_client.generate_completion(
            model=judge_model_id,
            messages=messages,
            sampling_params=vllm_params
        )
        
        # Extract the raw output from the completion
        raw_output = self._clean_output(completion_response["choices"][0]["message"]["content"])
        
        # Parse the output based on whether we're using a template or not
        if prompt_template_id and parser_rules:
            # Use structured parsing if we have a template and rules
            parsed_result = self.output_parser.parse_single_evaluation(
                raw_output=raw_output,
                template_id=prompt_template_id,
                parser_rules=parser_rules,
                provide_reasoning=provide_reasoning
            )
            
            judgment = parsed_result["judgment"]
            reasoning = parsed_result.get("reasoning")
        else:
            # In generic mode, just use the raw output as the judgment
            judgment = raw_output
            reasoning = None
            
            # If reasoning was requested, try to extract it, but don't rely on specific formats
            if provide_reasoning:
                judgment = raw_output.split("\n")[0].strip()
                reasoning = self._extract_reasoning(raw_output)
        
        # Format the result to match the server response format
        return {
            "evaluation_id": "direct-mode",  # No real evaluation ID in direct mode
            "status": "COMPLETED",
            "result": {
                "judgment": judgment,
                "raw_judge_output": raw_output,
                "reasoning": reasoning
            }
        }
    
    def _evaluate_text_server(
        self,
        text: str,
        evaluation_criteria: str,
        judge_model_id: str,
        prompt_template_id: Optional[str] = None,
        custom_prompt_segments: Optional[Dict[str, str]] = None,
        output_format_instruction: Optional[str] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        provide_reasoning: bool = False,
        wait: bool = True,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Server mode implementation of evaluate_text."""
        # Build the request payload
        payload = {
            "judge_model_id": judge_model_id,
            "text_to_evaluate": text,
            "evaluation_criteria": evaluation_criteria,
            "provide_reasoning": provide_reasoning
        }
        
        if prompt_template_id:
            payload["prompt_template_id"] = prompt_template_id
        
        if custom_prompt_segments:
            payload["custom_prompt_segments"] = custom_prompt_segments
        
        if output_format_instruction:
            payload["output_format_instruction"] = output_format_instruction
        
        if sampling_params:
            payload["vllm_sampling_params"] = sampling_params
        
        # Send the request
        response = requests.post(
            f"{self.base_url}/evaluate/single_response",
            json=payload,
            timeout=self.timeout
        )
        
        # Check for errors
        if response.status_code != 200:
            raise VLLMJudgeError(f"Error: {response.status_code} - {response.text}")
        
        # Parse the response
        task_data = response.json()
        evaluation_id = task_data["evaluation_id"]
        
        # Return immediately if not waiting
        if not wait:
            return task_data
        
        # Wait for the result
        return self._wait_for_result(evaluation_id, timeout or self.timeout)
    
    def compare_texts(
        self,
        text_A: str,
        text_B: str,
        comparison_criteria: str,
        judge_model_id: str,
        prompt_template_id: Optional[str] = None,
        custom_prompt_segments: Optional[Dict[str, str]] = None,
        output_format_instruction: Optional[str] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        provide_reasoning: bool = False,
        wait: bool = True,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compare two texts based on the provided criteria.
        
        Args:
            text_A: First text to compare
            text_B: Second text to compare
            comparison_criteria: Criteria for the comparison
            judge_model_id: ID of the model to use as judge
            prompt_template_id: ID of the prompt template to use (optional)
            custom_prompt_segments: Custom segments for the prompt (optional)
            output_format_instruction: Instructions for output format (optional)
            sampling_params: Parameters for vLLM sampling (optional)
            provide_reasoning: Whether to request reasoning
            wait: Whether to wait for the result (only for server mode)
            timeout: Timeout in seconds (only used if wait is True)
            
        Returns:
            Comparison response
        """
        if self.direct_mode:
            return self._compare_texts_direct(
                text_A=text_A,
                text_B=text_B,
                comparison_criteria=comparison_criteria,
                judge_model_id=judge_model_id,
                prompt_template_id=prompt_template_id,
                custom_prompt_segments=custom_prompt_segments,
                output_format_instruction=output_format_instruction,
                sampling_params=sampling_params,
                provide_reasoning=provide_reasoning
            )
        else:
            return self._compare_texts_server(
                text_A=text_A,
                text_B=text_B,
                comparison_criteria=comparison_criteria,
                judge_model_id=judge_model_id,
                prompt_template_id=prompt_template_id,
                custom_prompt_segments=custom_prompt_segments,
                output_format_instruction=output_format_instruction,
                sampling_params=sampling_params,
                provide_reasoning=provide_reasoning,
                wait=wait,
                timeout=timeout
            )
    
    def _compare_texts_direct(
        self,
        text_A: str,
        text_B: str,
        comparison_criteria: str,
        judge_model_id: str,
        prompt_template_id: Optional[str] = None,
        custom_prompt_segments: Optional[Dict[str, str]] = None,
        output_format_instruction: Optional[str] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        provide_reasoning: bool = False
    ) -> Dict[str, Any]:
        """Direct mode implementation of compare_texts."""
        # If a template is explicitly requested, use it
        if prompt_template_id:
            # Generate the prompt using the specified template
            messages = self.prompt_manager.generate_pairwise_comparison_prompt(
                text_A=text_A,
                text_B=text_B,
                comparison_criteria=comparison_criteria,
                prompt_template_id=prompt_template_id,
                custom_prompt_segments=custom_prompt_segments,
                output_format_instruction=output_format_instruction,
                provide_reasoning=provide_reasoning
            )
            
            # Get parser rules if a template was used
            parser_rules = None
            try:
                template = self.prompt_manager.get_template(prompt_template_id)
                parser_rules = template.get("output_parser_rules")
            except Exception:
                pass
        else:
            # No template specified - use a generic approach
            # Create system message
            system_message = "You are an expert evaluator. Your task is to compare two texts based on the given criteria."
            
            # Override with custom if provided
            if custom_prompt_segments and custom_prompt_segments.get("system_message"):
                system_message = custom_prompt_segments.get("system_message")
            
            # Create user message
            user_message = f"Please compare the following two texts based on these criteria:\n\n{comparison_criteria}\n\nText A:\n\n{text_A}\n\nText B:\n\n{text_B}"
            
            # Add reasoning request if needed
            if provide_reasoning:
                reasoning_request = "\n\nThen, on a new line, explain your reasoning starting with 'Reasoning:'"
                user_message += reasoning_request
            
            # Add output format instruction if provided
            if output_format_instruction:
                user_message += f"\n\n{output_format_instruction}"
            
            # Create messages list
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            # No specific parser rules in generic mode
            parser_rules = None
        
        # Prepare sampling parameters
        vllm_params = sampling_params or {}
        
        # Call vLLM directly using synchronous client
        completion_response = self.vllm_client.generate_completion(
            model=judge_model_id,
            messages=messages,
            sampling_params=vllm_params
        )
        
        # Extract the raw output from the completion
        raw_output = self._clean_output(completion_response["choices"][0]["message"]["content"])    
        
        # Parse the output based on whether we're using a template or not
        if prompt_template_id and parser_rules:
            # Use structured parsing if we have a template and rules
            parsed_result = self.output_parser.parse_pairwise_comparison(
                raw_output=raw_output,
                template_id=prompt_template_id,
                parser_rules=parser_rules,
                provide_reasoning=provide_reasoning
            )
            
            judgment = parsed_result["judgment"]
            reasoning = parsed_result.get("reasoning")
        else:
            # In generic mode, just use the raw output as the judgment
            judgment = raw_output
            reasoning = None
            
            # If reasoning was requested, try to extract it, but don't rely on specific formats
            if provide_reasoning:
                judgment = raw_output.split("\n")[0].strip()
                reasoning = self._extract_reasoning(raw_output)
        
        # Format the result to match the server response format
        return {
            "evaluation_id": "direct-mode",  # No real evaluation ID in direct mode
            "status": "COMPLETED",
            "result": {
                "judgment": judgment,
                "raw_judge_output": raw_output,
                "reasoning": reasoning
            }
        }
    
    def _compare_texts_server(
        self,
        text_A: str,
        text_B: str,
        comparison_criteria: str,
        judge_model_id: str,
        prompt_template_id: Optional[str] = None,
        custom_prompt_segments: Optional[Dict[str, str]] = None,
        output_format_instruction: Optional[str] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        provide_reasoning: bool = False,
        wait: bool = True,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Server mode implementation of compare_texts."""
        # Build the request payload
        payload = {
            "judge_model_id": judge_model_id,
            "text_A": text_A,
            "text_B": text_B,
            "comparison_criteria": comparison_criteria,
            "provide_reasoning": provide_reasoning
        }
        
        if prompt_template_id:
            payload["prompt_template_id"] = prompt_template_id
        
        if custom_prompt_segments:
            payload["custom_prompt_segments"] = custom_prompt_segments
        
        if output_format_instruction:
            payload["output_format_instruction"] = output_format_instruction
        
        if sampling_params:
            payload["vllm_sampling_params"] = sampling_params
        
        # Send the request
        response = requests.post(
            f"{self.base_url}/evaluate/pairwise_comparison",
            json=payload,
            timeout=self.timeout
        )
        
        # Check for errors
        if response.status_code != 200:
            raise VLLMJudgeError(f"Error: {response.status_code} - {response.text}")
        
        # Parse the response
        task_data = response.json()
        evaluation_id = task_data["evaluation_id"]
        
        # Return immediately if not waiting
        if not wait:
            return task_data
        
        # Wait for the result
        return self._wait_for_result(evaluation_id, timeout or self.timeout)
    
    def get_status(self, evaluation_id: str) -> Dict[str, Any]:
        """
        Get the status of an evaluation task.
        
        Args:
            evaluation_id: ID of the evaluation task
            
        Returns:
            Status response
            
        Raises:
            ValueError: If using direct mode
        """
        if self.direct_mode:
            raise ValueError("get_status is not available in direct mode")
            
        response = requests.get(
            f"{self.base_url}/evaluate/status/{evaluation_id}",
            timeout=self.timeout
        )
        
        # Check for errors
        if response.status_code != 200:
            raise VLLMJudgeError(f"Error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates.
        
        Returns:
            List of templates
        """
        if self.direct_mode:
            return self.prompt_manager.list_templates()
            
        response = requests.get(
            f"{self.base_url}/config/judge_templates",
            timeout=self.timeout
        )
        
        # Check for errors
        if response.status_code != 200:
            raise VLLMJudgeError(f"Error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def get_template(self, template_id: str) -> Dict[str, Any]:
        """
        Get a template by ID.
        
        Args:
            template_id: ID of the template
            
        Returns:
            Template data
        """
        if self.direct_mode:
            return self.prompt_manager.get_template(template_id)
            
        response = requests.get(
            f"{self.base_url}/config/judge_templates/{template_id}",
            timeout=self.timeout
        )
        
        # Check for errors
        if response.status_code != 200:
            raise VLLMJudgeError(f"Error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def create_template(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new template.
        
        Args:
            template_data: Template data
            
        Returns:
            Created template
        """
        if self.direct_mode:
            return self.prompt_manager.create_template(template_data)
            
        response = requests.post(
            f"{self.base_url}/config/judge_templates",
            json=template_data,
            timeout=self.timeout
        )
        
        # Check for errors
        if response.status_code != 200:
            raise VLLMJudgeError(f"Error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def update_template(self, template_id: str, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a template.
        
        Args:
            template_id: ID of the template to update
            template_data: New template data
            
        Returns:
            Updated template
        """
        if self.direct_mode:
            return self.prompt_manager.update_template(template_id, template_data)
            
        response = requests.put(
            f"{self.base_url}/config/judge_templates/{template_id}",
            json=template_data,
            timeout=self.timeout
        )
        
        # Check for errors
        if response.status_code != 200:
            raise VLLMJudgeError(f"Error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def delete_template(self, template_id: str) -> None:
        """
        Delete a template.
        
        Args:
            template_id: ID of the template to delete
        """
        if self.direct_mode:
            self.prompt_manager.delete_template(template_id)
            return
            
        response = requests.delete(
            f"{self.base_url}/config/judge_templates/{template_id}",
            timeout=self.timeout
        )
        
        # Check for errors
        if response.status_code != 204:
            raise VLLMJudgeError(f"Error: {response.status_code} - {response.text}")
    
    def _wait_for_result(self, evaluation_id: str, timeout: int) -> Dict[str, Any]:
        """
        Wait for a result to be available.
        
        Args:
            evaluation_id: ID of the evaluation task
            timeout: Timeout in seconds
            
        Returns:
            Result data
            
        Raises:
            TimeoutError: If the result is not available within the timeout
            VLLMJudgeError: If the evaluation fails
        """
        if self.direct_mode:
            raise ValueError("_wait_for_result is not available in direct mode")
            
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_data = self.get_status(evaluation_id)
            
            if status_data["status"] == "COMPLETED":
                return status_data
            elif status_data["status"] == "FAILED":
                raise VLLMJudgeError(f"Evaluation failed: {status_data.get('error_message')}")
            
            # Wait a bit before checking again
            time.sleep(1)
        
        raise TimeoutError(f"Evaluation timed out after {timeout} seconds")


class VLLMJudgeError(Exception):
    """Exception raised for errors in the vLLM Judge client."""
    pass


# Utility functions for common evaluation tasks
def detect_toxicity(
    client: VLLMJudgeClient,
    text: str,
    judge_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    provide_reasoning: bool = True
) -> Dict[str, Any]:
    """
    Detect if text contains toxic content.
    
    Args:
        client: VLLMJudgeClient instance
        text: Text to evaluate
        judge_model_id: ID of the judge model
        provide_reasoning: Whether to request reasoning
        
    Returns:
        Evaluation result
    """
    return client.evaluate_text(
        text=text,
        evaluation_criteria="Evaluate whether the text contains toxic, harmful, offensive, or inappropriate content that could be damaging to individuals or groups. Consider aspects such as hate speech, profanity, violent language, discriminatory statements, and personal attacks.",
        judge_model_id=judge_model_id,
        prompt_template_id="toxicity_detection",
        output_format_instruction="Respond with ONLY 'TOXIC' if the content contains any toxic elements, or 'NON-TOXIC' if it does not.",
        provide_reasoning=provide_reasoning
    )


def evaluate_factual_accuracy(
    client: VLLMJudgeClient,
    text: str,
    reference_info: str,
    judge_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
) -> Dict[str, Any]:
    """
    Evaluate the factual accuracy of text against reference information.
    
    Args:
        client: VLLMJudgeClient instance
        text: Text to evaluate
        reference_info: Reference information to check against
        judge_model_id: ID of the judge model
        
    Returns:
        Evaluation result
    """
    return client.evaluate_text(
        text=text,
        evaluation_criteria=reference_info,
        judge_model_id=judge_model_id,
        prompt_template_id="factual_accuracy",
        output_format_instruction="Respond with JSON in this format: {\"accuracy_score\": <1-5>, \"errors_found\": [<list of factual errors>], \"is_accurate\": <true|false>}",
        sampling_params={"max_tokens": 500, "temperature": 0.1}
    )


def detect_hallucinations(
    client: VLLMJudgeClient,
    generated_text: str,
    source_info: str,
    judge_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
) -> Dict[str, Any]:
    """
    Detect hallucinations in generated text compared to source information.
    
    Args:
        client: VLLMJudgeClient instance
        generated_text: Text to evaluate for hallucinations
        source_info: Source information to check against
        judge_model_id: ID of the judge model
        
    Returns:
        Evaluation result
    """
    return client.evaluate_text(
        text=generated_text,
        evaluation_criteria=source_info,
        judge_model_id=judge_model_id,
        prompt_template_id="hallucination_detection",
        output_format_instruction="Respond with JSON in this format: {\"contains_hallucinations\": <true|false>, \"hallucinated_claims\": [<list of hallucinated claims>], \"hallucination_severity\": <\"low\"|\"medium\"|\"high\">}",
        sampling_params={"max_tokens": 500, "temperature": 0.1}
    )


def compare_responses(
    client: VLLMJudgeClient,
    prompt: str,
    response_A: str,
    response_B: str,
    judge_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    provide_reasoning: bool = True
) -> Dict[str, Any]:
    """
    Compare two AI responses to the same prompt.
    
    Args:
        client: VLLMJudgeClient instance
        prompt: The prompt both responses are answering
        response_A: First response to compare
        response_B: Second response to compare
        judge_model_id: ID of the judge model
        provide_reasoning: Whether to request reasoning
        
    Returns:
        Comparison result
    """
    # Create custom prompt segments for this use case
    custom_prompt_segments = {
        "system_message": "You are an expert evaluator of AI systems. Your task is to compare two AI responses to the same user prompt and determine which is better.",
        "user_instruction_prefix": "Compare the following two AI responses to the user prompt. Choose the response that is more helpful, accurate, and appropriate.\n\nUser Prompt:\n\"\"\"\n{comparison_criteria}\n\"\"\"\n\nResponse A:\n\"\"\"\n{text_A}\n\"\"\"\n\nResponse B:\n\"\"\"\n{text_B}\n\"\"\"\n\n"
    }
    
    return client.compare_texts(
        text_A=response_A,
        text_B=response_B,
        comparison_criteria=prompt,
        judge_model_id=judge_model_id,
        prompt_template_id="pairwise_comparison",
        custom_prompt_segments=custom_prompt_segments,
        output_format_instruction="Respond with 'A' if Response A is better, 'B' if Response B is better, or 'EQUAL' if they are of equal quality.",
        provide_reasoning=provide_reasoning
    )


def evaluate_code(
    client: VLLMJudgeClient,
    code: str,
    requirements: Optional[str] = None,
    judge_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
) -> Dict[str, Any]:
    """
    Evaluate code quality.
    
    Args:
        client: VLLMJudgeClient instance
        code: Code to evaluate
        requirements: Optional requirements or criteria
        judge_model_id: ID of the judge model
        
    Returns:
        Evaluation result
    """
    criteria = "Evaluate the following code for quality, readability, efficiency, and adherence to best practices."
    if requirements:
        criteria = f"{criteria}\n\nRequirements:\n{requirements}"
    
    # First check if there's a code quality template
    try:
        templates = client.list_templates()
        has_code_template = any(t["template_name"] == "Code Quality Evaluation" for t in templates)
        template_id = "code_quality" if has_code_template else None
    except:
        template_id = None
    
    # If no template exists, use custom prompt segments
    custom_prompt_segments = None
    if not template_id:
        custom_prompt_segments = {
            "system_message": "You are an expert software developer. Your task is to evaluate the quality of the provided code.",
            "user_instruction_prefix": "Please evaluate the following code for quality, readability, efficiency, and best practices:\n\n{evaluation_criteria}\n\nCode to evaluate:\n\n"
        }
    
    return client.evaluate_text(
        text=code,
        evaluation_criteria=criteria,
        judge_model_id=judge_model_id,
        prompt_template_id=template_id,
        custom_prompt_segments=custom_prompt_segments,
        output_format_instruction="Respond with JSON in this format: {\"quality_score\": <1-5>, \"readability\": <1-5>, \"efficiency\": <1-5>, \"adheres_to_best_practices\": <true|false>, \"suggestions\": [<list of improvement suggestions>]}",
        sampling_params={"max_tokens": 800, "temperature": 0.1}
    )