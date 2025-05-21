import json
import os
import uuid
from typing import Dict, List, Any, Optional

from vllm_judge.core.config import settings
from vllm_judge.core.errors import PromptTemplateError, TemplateNotFoundError
from vllm_judge.core.models import CustomPromptSegments


class PromptManager:
    """Manages prompt templates and their generation."""
    
    def __init__(self, template_path: Optional[str] = None):
        self.template_path = template_path or settings.TEMPLATE_STORAGE_PATH
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, Any]:
        """Load templates from the template file."""
        if not os.path.exists(self.template_path):
            # Create template file with default templates if it doesn't exist
            os.makedirs(os.path.dirname(self.template_path), exist_ok=True)
            default_templates = self._get_default_templates()
            with open(self.template_path, "w") as f:
                json.dump(default_templates, f, indent=2)
            return default_templates
        
        try:
            with open(self.template_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise PromptTemplateError("Failed to parse template file")
            
    def _save_templates(self) -> None:
        """Save templates to the template file."""
        os.makedirs(os.path.dirname(self.template_path), exist_ok=True)
        with open(self.template_path, "w") as f:
            json.dump(self.templates, f, indent=2)
    
    def _get_default_templates(self) -> Dict[str, Any]:
        """Get default prompt templates."""
        return {
            "templates": {
                # Binary classification template
                "binary_classification": {
                    "template_id": "binary_classification",
                    "template_name": "Binary Classification",
                    "target_judge_model_family": None,
                    "description": "Template for binary classification tasks (yes/no, safe/unsafe, etc.)",
                    "prompt_structure": {
                        "system_message": "You are an expert evaluator. Your task is to analyze the given content and make a binary classification decision based on the provided criteria.",
                        "user_instruction_prefix": "Please evaluate the following content based on these criteria:\n\n{evaluation_criteria}\n\nContent to evaluate:\n\n",
                        "user_instruction_suffix": "\n\n{output_format_instruction}"
                    },
                    "output_parser_rules": {
                        "type": "binary",
                        "positive_patterns": ["yes", "true", "safe", "positive", "acceptable"],
                        "negative_patterns": ["no", "false", "unsafe", "negative", "unacceptable"]
                    }
                },
                # Likert scale template
                "likert_scale": {
                    "template_id": "likert_scale",
                    "template_name": "Likert Scale Evaluation",
                    "target_judge_model_family": None,
                    "description": "Template for Likert scale evaluations (rating on a scale, e.g., 1-5)",
                    "prompt_structure": {
                        "system_message": "You are an expert evaluator. Your task is to rate the given content on a scale based on the provided criteria.",
                        "user_instruction_prefix": "Please evaluate the following content based on these criteria:\n\n{evaluation_criteria}\n\nContent to evaluate:\n\n",
                        "user_instruction_suffix": "\n\n{output_format_instruction}"
                    },
                    "output_parser_rules": {
                        "type": "numeric",
                        "pattern": r"\b([1-5])\b"
                    }
                },
                # Pairwise comparison template
                "pairwise_comparison": {
                    "template_id": "pairwise_comparison",
                    "template_name": "Pairwise Comparison",
                    "target_judge_model_family": None,
                    "description": "Template for comparing two texts and selecting the better one",
                    "prompt_structure": {
                        "system_message": "You are an expert evaluator. Your task is to compare two texts and select the better one based on the provided criteria.",
                        "user_instruction_prefix": "Please compare the following two texts based on these criteria:\n\n{comparison_criteria}\n\nText A:\n\n{text_A}\n\nText B:\n\n{text_B}\n\n",
                        "user_instruction_suffix": "\n\n{output_format_instruction}"
                    },
                    "output_parser_rules": {
                        "type": "preference",
                        "pattern": r"(?:(?:Text|Option|Response)\s*)?([AB])"
                    }
                }
            }
        }
            
    def get_template(self, template_id: str) -> Dict[str, Any]:
        """
        Get a prompt template by ID.
        
        Args:
            template_id: ID of the template to get
            
        Returns:
            The prompt template
            
        Raises:
            TemplateNotFoundError: If the template is not found
        """
        if template_id not in self.templates["templates"]:
            raise TemplateNotFoundError(template_id)
        
        return self.templates["templates"][template_id]
        
    def create_template(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new prompt template.
        
        Args:
            template_data: Data for the new template
            
        Returns:
            The created template
        """
        # Generate a unique ID for the template
        template_id = str(uuid.uuid4())
        
        # Add the template to the templates dictionary
        self.templates["templates"][template_id] = {
            "template_id": template_id,
            **template_data
        }
        
        # Save the templates
        self._save_templates()
        
        return self.templates["templates"][template_id]
        
    def update_template(self, template_id: str, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing prompt template.
        
        Args:
            template_id: ID of the template to update
            template_data: New data for the template
            
        Returns:
            The updated template
            
        Raises:
            TemplateNotFoundError: If the template is not found
        """
        if template_id not in self.templates["templates"]:
            raise TemplateNotFoundError(template_id)
        
        # Update the template
        self.templates["templates"][template_id].update(template_data)
        
        # Save the templates
        self._save_templates()
        
        return self.templates["templates"][template_id]
        
    def delete_template(self, template_id: str) -> None:
        """
        Delete a prompt template.
        
        Args:
            template_id: ID of the template to delete
            
        Raises:
            TemplateNotFoundError: If the template is not found
        """
        if template_id not in self.templates["templates"]:
            raise TemplateNotFoundError(template_id)
        
        # Delete the template
        del self.templates["templates"][template_id]
        
        # Save the templates
        self._save_templates()
        
    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates.
        
        Returns:
            List of templates
        """
        return list(self.templates["templates"].values())
        
    def generate_single_evaluation_prompt(
        self,
        text_to_evaluate: str,
        evaluation_criteria: str,
        prompt_template_id: Optional[str] = None,
        custom_prompt_segments: Optional[CustomPromptSegments] = None,
        output_format_instruction: Optional[str] = None,
        provide_reasoning: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Generate a prompt for evaluating a single text.
        
        Args:
            text_to_evaluate: The text to evaluate
            evaluation_criteria: Criteria for the evaluation
            prompt_template_id: ID of the prompt template to use (optional)
            custom_prompt_segments: Custom segments to override parts of the template (optional)
            output_format_instruction: Instructions for the judge LLM on output format (optional)
            provide_reasoning: Whether to instruct the judge to provide reasoning
            
        Returns:
            A list of messages in the format expected by the vLLM server
        """
        # Default output format instruction if not provided
        if not output_format_instruction:
            output_format_instruction = (
                "Respond with ONLY a single word: 'POSITIVE' or 'NEGATIVE'."
            )
            
        # Add reasoning request if needed
        if provide_reasoning:
            output_format_instruction += (
                " Then, on a new line, explain your reasoning starting with 'Reasoning:'"
            )
            
        # Use the provided template or default to binary classification
        template = None
        if prompt_template_id:
            try:
                template = self.get_template(prompt_template_id)["prompt_structure"]
            except TemplateNotFoundError:
                raise PromptTemplateError(f"Template not found: {prompt_template_id}")
        else:
            # Use the binary classification template
            template = self.get_template("binary_classification")["prompt_structure"]
            
        # Override template segments with custom segments if provided
        if custom_prompt_segments:
            if custom_prompt_segments.system_message:
                template["system_message"] = custom_prompt_segments.system_message
            if custom_prompt_segments.user_instruction_prefix:
                template["user_instruction_prefix"] = custom_prompt_segments.user_instruction_prefix
            if custom_prompt_segments.user_instruction_suffix:
                template["user_instruction_suffix"] = custom_prompt_segments.user_instruction_suffix
                
        # Format the prompt
        user_content = template["user_instruction_prefix"].format(
            evaluation_criteria=evaluation_criteria,
            output_format_instruction=output_format_instruction,
        ) + text_to_evaluate
        
        if "user_instruction_suffix" in template:
            user_content += template["user_instruction_suffix"].format(
                output_format_instruction=output_format_instruction,
            )
            
        # Create the chat messages
        messages = [
            {
                "role": "system",
                "content": template["system_message"],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        
        return messages
        
    def generate_pairwise_comparison_prompt(
        self,
        text_A: str,
        text_B: str,
        comparison_criteria: str,
        prompt_template_id: Optional[str] = None,
        custom_prompt_segments: Optional[Dict[str, str]] = None,
        output_format_instruction: Optional[str] = None,
        provide_reasoning: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Generate a prompt for comparing two texts.
        
        Args:
            text_A: First text to compare
            text_B: Second text to compare
            comparison_criteria: Criteria for the comparison
            prompt_template_id: ID of the prompt template to use (optional)
            custom_prompt_segments: Custom segments to override parts of the template (optional)
            output_format_instruction: Instructions for the judge LLM on output format (optional)
            provide_reasoning: Whether to instruct the judge to provide reasoning
            
        Returns:
            A list of messages in the format expected by the vLLM server
        """
        # Default output format instruction if not provided
        if not output_format_instruction:
            output_format_instruction = (
                "Respond with ONLY: 'A' if Text A is better, 'B' if Text B is better, or 'EQUAL' if they are of equal quality."
            )
            
        # Add reasoning request if needed
        if provide_reasoning:
            output_format_instruction += (
                " Then, on a new line, explain your reasoning starting with 'Reasoning:'"
            )
            
        # Use the provided template or default to pairwise comparison
        template = None
        if prompt_template_id:
            try:
                template = self.get_template(prompt_template_id)["prompt_structure"]
            except TemplateNotFoundError:
                raise PromptTemplateError(f"Template not found: {prompt_template_id}")
        else:
            # Use the pairwise comparison template
            template = self.get_template("pairwise_comparison")["prompt_structure"]
            
        # Override template segments with custom segments if provided
        if custom_prompt_segments:
            if custom_prompt_segments.get("system_message"):
                template["system_message"] = custom_prompt_segments.get("system_message")
            if custom_prompt_segments.get("user_instruction_prefix"):
                template["user_instruction_prefix"] = custom_prompt_segments.get("user_instruction_prefix")
            if custom_prompt_segments.get("user_instruction_suffix"):
                template["user_instruction_suffix"] = custom_prompt_segments.get("user_instruction_suffix")
                
        # Format the prompt
        user_content = template["user_instruction_prefix"].format(
            comparison_criteria=comparison_criteria,
            text_A=text_A,
            text_B=text_B,
            output_format_instruction=output_format_instruction,
        )
        
        if "user_instruction_suffix" in template:
            user_content += template["user_instruction_suffix"].format(
                output_format_instruction=output_format_instruction,
            )
            
        # Create the chat messages
        messages = [
            {
                "role": "system",
                "content": template["system_message"],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        
        return messages
