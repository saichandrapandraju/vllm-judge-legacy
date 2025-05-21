"""
Examples for creating, managing, and using templates with vLLM Judge.
"""

from vllm_judge import VLLMJudgeClient
import json

vllm_api_base = "http://localhost:8000/v1"  # TODO: Pass as a CLI argument
judge_model_id = "qwen2"  # TODO: Pass as a CLI argument


# Initialize the client in direct mode
client = VLLMJudgeClient(
    direct_mode=True,
    vllm_api_base=vllm_api_base
)


print("TEMPLATE MANAGEMENT EXAMPLES")
print("=" * 80)


# List existing templates
print("\nListing available templates:")
print("-" * 50)
templates = client.list_templates()
for template in templates:
    print(f"- {template['template_name']} (ID: {template['template_id']})")


# Example 1: Creating a Basic Template
print("\n\nEXAMPLE 1: Creating a Basic Template")
print("-" * 50)

basic_template = client.create_template({
    "template_name": "Sentiment Analysis",
    "description": "Template for analyzing sentiment in text",
    "prompt_structure": {
        "system_message": "You are an expert in sentiment analysis. Your task is to determine the emotional tone of the given text.",
        "user_instruction_prefix": "Please analyze the sentiment of the following text based on these criteria:\n\n{evaluation_criteria}\n\nText to analyze:\n\n",
        "user_instruction_suffix": "\n\n{output_format_instruction}"
    },
    "output_parser_rules": {
        "type": "text"  # Simple text parsing - no special rules
    }
})

print(f"Created template: {basic_template['template_name']} (ID: {basic_template['template_id']})")

# Using the basic template
sentiment_result = client.evaluate_text(
    text="I absolutely love this product! It's the best purchase I've made all year.",
    evaluation_criteria="Consider the emotional tone, word choice, and intensity of feelings expressed.",
    judge_model_id=judge_model_id,
    prompt_template_id=basic_template["template_id"],
    output_format_instruction="Respond with POSITIVE, NEGATIVE, or NEUTRAL."
)

print(f"\nSentiment Analysis Result:")
print(f"Judgment: {sentiment_result['result']['judgment']}")


# Example 2: Creating a Template with Structured Output Parsing
print("\n\nEXAMPLE 2: Creating a Template with Structured Output Parsing")
print("-" * 50)

json_template = client.create_template({
    "template_name": "Product Review Analysis",
    "description": "Template for analyzing product reviews with structured output",
    "prompt_structure": {
        "system_message": "You are an expert in analyzing product reviews. Your task is to extract key insights from the given review.",
        "user_instruction_prefix": "Please analyze the following product review based on these criteria:\n\n{evaluation_criteria}\n\nReview to analyze:\n\n",
        "user_instruction_suffix": "\n\n{output_format_instruction}"
    },
    "output_parser_rules": {
        "type": "json"  # This will attempt to parse the output as JSON
    }
})

print(f"Created template: {json_template['template_name']} (ID: {json_template['template_id']})")

# Using the JSON template
review_result = client.evaluate_text(
    text="I've been using this laptop for about 3 months now. The battery life is excellent, lasting me all day. The screen is bright and crisp, and the keyboard feels great to type on. The only downside is that it runs a bit hot when doing intensive tasks. Overall though, I'm very satisfied with this purchase.",
    evaluation_criteria="Extract the key positive and negative points, and determine the overall sentiment.",
    judge_model_id=judge_model_id,
    prompt_template_id=json_template["template_id"],
    output_format_instruction="Respond with JSON in this format: {\"positives\": [\"list of positive points\"], \"negatives\": [\"list of negative points\"], \"overall_sentiment\": \"positive|negative|neutral\", \"sentiment_score\": <1-5>}"
)

print(f"\nProduct Review Analysis Result:")
# Pretty print the JSON result if it parsed correctly
if isinstance(review_result['result']['judgment'], dict):
    print(json.dumps(review_result['result']['judgment'], indent=2))
else:
    print(f"Raw result: {review_result['result']['judgment']}")


# Example 3: Creating a Template with Specific Parser Rules
print("\n\nEXAMPLE 3: Creating a Template with Specific Parser Rules")
print("-" * 50)

binary_template = client.create_template({
    "template_name": "Factual Verification",
    "description": "Template for verifying factual statements",
    "prompt_structure": {
        "system_message": "You are an expert fact-checker. Your task is to verify whether the given statement is factually correct.",
        "user_instruction_prefix": "Please verify the following statement based on established facts:\n\n{evaluation_criteria}\n\nStatement to verify:\n\n",
        "user_instruction_suffix": "\n\n{output_format_instruction}"
    },
    "output_parser_rules": {
        "type": "binary",
        "positive_patterns": ["true", "correct", "accurate", "factual", "yes"],
        "negative_patterns": ["false", "incorrect", "inaccurate", "wrong", "no"]
    }
})

print(f"Created template: {binary_template['template_name']} (ID: {binary_template['template_id']})")

# Using the binary template
fact_check_result = client.evaluate_text(
    text="The Great Wall of China is visible from the Moon.",
    evaluation_criteria="Verify if this statement is scientifically accurate based on astronomical facts and human vision capabilities.",
    judge_model_id=judge_model_id,
    prompt_template_id=binary_template["template_id"],
    output_format_instruction="Respond with TRUE if the statement is factually correct, or FALSE if it is incorrect."
)

print(f"\nFactual Verification Result:")
print(f"Judgment: {fact_check_result['result']['judgment']}")
print(f"Raw output: {fact_check_result['result']['raw_judge_output']}")


# Example 4: Creating a template for numeric scoring
print("\n\nEXAMPLE 4: Creating a Template for Numeric Scoring")
print("-" * 50)

scoring_template = client.create_template({
    "template_name": "Essay Quality Scoring",
    "description": "Template for scoring essays on multiple dimensions",
    "prompt_structure": {
        "system_message": "You are an expert essay evaluator. Your task is to evaluate the quality of the given essay across multiple dimensions.",
        "user_instruction_prefix": "Please evaluate the following essay based on these criteria:\n\n{evaluation_criteria}\n\nEssay to evaluate:\n\n",
        "user_instruction_suffix": "\n\n{output_format_instruction}"
    },
    "output_parser_rules": {
        "type": "json"
    }
})

print(f"Created template: {scoring_template['template_name']} (ID: {scoring_template['template_id']})")

# Using the scoring template
essay_result = client.evaluate_text(
    text="The Impact of Climate Change\n\nClimate change is one of the most pressing issues of our time. Rising global temperatures have led to more frequent extreme weather events, rising sea levels, and disruptions to ecosystems worldwide. Human activities, particularly the burning of fossil fuels, have significantly contributed to this problem through greenhouse gas emissions. To address climate change, we need a multi-faceted approach involving renewable energy transition, policy changes, and individual actions. Without prompt and coordinated global efforts, the consequences could be catastrophic for future generations.",
    evaluation_criteria="Evaluate this essay for content quality, organization, language use, and overall effectiveness.",
    judge_model_id=judge_model_id,
    prompt_template_id=scoring_template["template_id"],
    output_format_instruction="Respond with JSON in this format: {\"content_score\": <1-10>, \"organization_score\": <1-10>, \"language_score\": <1-10>, \"overall_score\": <1-10>, \"strengths\": [\"list of strengths\"], \"areas_for_improvement\": [\"list of areas for improvement\"]}"
)

print(f"\nEssay Evaluation Result:")
# Pretty print the JSON result if it parsed correctly
if isinstance(essay_result['result']['judgment'], dict):
    print(json.dumps(essay_result['result']['judgment'], indent=2))
else:
    print(f"Raw result: {essay_result['result']['judgment']}")


# Example 5: Retrieving, Updating and Deleting Templates
print("\n\nEXAMPLE 5: Template Management Operations")
print("-" * 50)

# Get a template by ID
retrieved_template = client.get_template(basic_template["template_id"])
print(f"Retrieved template: {retrieved_template['template_name']}")

# Update a template
updated_template = client.update_template(
    basic_template["template_id"],
    {
        "template_name": "Enhanced Sentiment Analysis",
        "description": "Improved template for analyzing sentiment with more detailed output"
    }
)
print(f"Updated template name: {updated_template['template_name']}")
print(f"Updated template description: {updated_template['description']}")

# List templates again to see changes
print("\nUpdated template list:")
templates = client.list_templates()
for template in templates:
    print(f"- {template['template_name']} (ID: {template['template_id']})")

# Delete a template (uncomment to execute)
# print("\nDeleting template:", json_template["template_id"])
# client.delete_template(json_template["template_id"])
# 
# # Verify deletion
# print("\nTemplate list after deletion:")
# templates = client.list_templates()
# for template in templates:
#     print(f"- {template['template_name']} (ID: {template['template_id']})")