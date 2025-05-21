import json
import os
from typing import Dict, Any, List, Optional


def load_templates(template_path: str) -> Dict[str, Any]:
    """
    Load templates from a JSON file.
    
    Args:
        template_path: Path to the template file
        
    Returns:
        Dictionary of templates
        
    Raises:
        FileNotFoundError: If the template file doesn't exist
        json.JSONDecodeError: If the template file is not valid JSON
    """
    with open(template_path, "r") as f:
        return json.load(f)


def save_templates(templates: Dict[str, Any], template_path: str) -> None:
    """
    Save templates to a JSON file.
    
    Args:
        templates: Dictionary of templates
        template_path: Path to the template file
    """
    os.makedirs(os.path.dirname(template_path), exist_ok=True)
    with open(template_path, "w") as f:
        json.dump(templates, f, indent=2)


def format_prompt(
    template: Dict[str, Any],
    variables: Dict[str, str],
    custom_segments: Optional[Dict[str, str]] = None
) -> List[Dict[str, str]]:
    """
    Format a prompt template with variables.
    
    Args:
        template: The prompt template
        variables: Dictionary of variables to substitute in the template
        custom_segments: Optional custom segments to override parts of the template
        
    Returns:
        A list of messages in the format expected by the vLLM server
    """
    prompt_structure = template["prompt_structure"].copy()
    
    # Override template segments with custom segments if provided
    if custom_segments:
        for key, value in custom_segments.items():
            if key in prompt_structure:
                prompt_structure[key] = value
    
    # Format the system message
    system_message = prompt_structure["system_message"]
    
    # Format the user content
    user_content = prompt_structure["user_instruction_prefix"].format(**variables)
    
    # Add text to evaluate or comparison texts (these are handled specially)
    if "text_to_evaluate" in variables:
        user_content += variables["text_to_evaluate"]
    elif "text_A" in variables and "text_B" in variables:
        # For pairwise comparison, the texts are already included in the prefix
        pass
    
    # Add suffix if present
    if "user_instruction_suffix" in prompt_structure:
        user_content += prompt_structure["user_instruction_suffix"].format(**variables)
    
    # Create the chat messages
    messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    
    return messages


def get_model_specific_template(
    template_data: Dict[str, Any],
    model_id: str
) -> Dict[str, Any]:
    """
    Get the most appropriate template version for a specific model.
    
    This function checks if there are model-specific adaptations of a template
    and returns the most appropriate one for the given model.
    
    Args:
        template_data: The template data
        model_id: The ID of the model
        
    Returns:
        The most appropriate template for the model
    """
    # If there are no model adaptations, return the base template
    if "model_adaptations" not in template_data:
        return template_data
    
    # Get the model family from the model ID
    model_family = get_model_family(model_id)
    
    # Check if there's an adaptation for this model family
    if model_family in template_data["model_adaptations"]:
        # Create a copy of the base template
        adapted_template = dict(template_data)
        
        # Update with the model-specific adaptations
        for key, value in template_data["model_adaptations"][model_family].items():
            if isinstance(value, dict) and isinstance(adapted_template.get(key), dict):
                # Merge dictionaries recursively
                adapted_template[key] = {**adapted_template[key], **value}
            else:
                # Replace value
                adapted_template[key] = value
        
        return adapted_template
    
    # No specific adaptation found, return the base template
    return template_data


def get_model_family(model_id: str) -> str:
    """
    Extract the model family from a model ID.
    
    Args:
        model_id: The ID of the model
        
    Returns:
        The model family
    """
    # Extract model family based on common naming patterns
    lower_id = model_id.lower()
    
    if "llama" in lower_id:
        return "llama"
    elif "mistral" in lower_id:
        return "mistral"
    elif "falcon" in lower_id:
        return "falcon"
    elif "claude" in lower_id:
        return "claude"
    elif "gpt" in lower_id:
        return "gpt"
    elif "palm" in lower_id or "gemini" in lower_id:
        return "google"
    elif "bloom" in lower_id:
        return "bloom"
    else:
        # Return the organization part of the model ID if available
        parts = model_id.split("/")
        if len(parts) > 1:
            return parts[0].lower()
        
        # If no clear family can be determined, return unknown
        return "unknown"


def get_default_output_format_instruction(task_type: str) -> str:
    """
    Get a default output format instruction for a task type.
    
    Args:
        task_type: The type of task
        
    Returns:
        A default output format instruction
    """
    if task_type == "binary_classification":
        return "Respond with ONLY 'POSITIVE' or 'NEGATIVE'."
    elif task_type == "likert_scale":
        return "Respond with ONLY a single number from 1 to 5, where 1 is the worst and 5 is the best."
    elif task_type == "pairwise_comparison":
        return "Respond with ONLY 'A' if Text A is better, 'B' if Text B is better, or 'EQUAL' if they are of equal quality."
    elif task_type == "toxicity_detection":
        return "Respond with ONLY 'TOXIC' or 'NON-TOXIC'."
    elif task_type == "factual_accuracy":
        return "Respond with JSON in this format: {\"accuracy_score\": <1-5>, \"errors_found\": [<list of factual errors>], \"is_accurate\": <true|false>}"
    elif task_type == "reasoning_evaluation":
        return "Respond with ONLY a single number from 1 to 5, where 1 indicates poor reasoning and 5 indicates excellent reasoning."
    elif task_type == "summarization_quality":
        return "Respond with JSON in this format: {\"completeness_score\": <1-5>, \"conciseness_score\": <1-5>, \"accuracy_score\": <1-5>, \"overall_score\": <1-5>}"
    elif task_type == "hallucination_detection":
        return "Respond with JSON in this format: {\"contains_hallucinations\": <true|false>, \"hallucinated_claims\": [<list of hallucinated claims>], \"hallucination_severity\": <\"low\"|\"medium\"|\"high\">}"
    else:
        return "Provide a clear and concise evaluation."


def get_chain_of_thought_prompt(task_type: str) -> str:
    """
    Get a chain-of-thought prompt for a task type.
    
    Args:
        task_type: The type of task
        
    Returns:
        A chain-of-thought prompt
    """
    if task_type == "binary_classification":
        return "Think step by step about your evaluation. First, identify the key aspects of the content relevant to the criteria. Then, assess how well the content meets or fails to meet each criterion. Finally, make your determination. After your analysis, "
    elif task_type == "likert_scale":
        return "Think step by step about your evaluation. First, identify the key aspects of the content relevant to the criteria. Then, assess how well the content meets or fails to meet each criterion. Consider the strengths and weaknesses to determine the appropriate score. After your analysis, "
    elif task_type == "pairwise_comparison":
        return "Think step by step about your comparison. First, identify the key aspects of both texts relevant to the criteria. Then, compare how well each text addresses these aspects. Note the strengths and weaknesses of each. Finally, determine which text is better overall. After your analysis, "
    elif task_type == "toxicity_detection":
        return "Think step by step about your evaluation. First, identify any potentially harmful or inappropriate content. Then, assess the severity and intent of this content. Consider the context and the potential impact on different audiences. After your analysis, "
    elif task_type == "factual_accuracy":
        return "Think step by step about your evaluation. First, identify the key factual claims in the text. Then, check each claim against the reference information. Note any inconsistencies, omissions, or additions that are not supported by the reference. After your analysis, "
    elif task_type == "reasoning_evaluation":
        return "Think step by step about your evaluation. First, identify the main arguments and logical structure of the text. Then, assess the clarity, coherence, and soundness of the reasoning. Look for logical fallacies, unsupported assumptions, or gaps in the argument. After your analysis, "
    elif task_type == "summarization_quality":
        return "Think step by step about your evaluation. First, identify the key points in the original text. Then, check if the summary captures these key points. Assess the completeness, accuracy, and conciseness of the summary. Consider if any important information is missing or if any extraneous details are included. After your analysis, "
    elif task_type == "hallucination_detection":
        return "Think step by step about your evaluation. First, identify the key factual claims in the generated text. Then, check each claim against the source information. Note any claims that are not supported by or contradict the source. Assess the severity of any hallucinations found. After your analysis, "
    else:
        return "Think step by step about your evaluation. Consider all relevant aspects of the content in relation to the criteria. Carefully weigh the evidence before making your determination. After your analysis, "