# vLLM Judge [DEPRECATED]

A flexible Python library for enabling LLM-as-a-judge capabilities with vLLM-hosted models. Use it directly in your application or as a standalone service.

## Overview

vLLM Judge provides a complete framework for leveraging any LLM hosted on vLLM as a judge or evaluator. The library features a hybrid approach that gives you maximum flexibility:

1. **Direct Mode**: Communicate directly with a vLLM server without running a separate service
2. **Server Mode**: Run a dedicated vLLM Judge server for advanced features and multi-client support

## Installation

```bash
git clone git@github.com:saichandrapandraju/vllm-judge.git
cd vllm-judge
pip install -e .
```

## Progressive Usage Guide: From Beginner to Expert

vLLM Judge is designed to grow with your needs, from simple evaluations to complex, structured assessments. This guide will walk you through increasingly sophisticated ways to use the library.

### Level 1: Basic Evaluation (Raw LLM Responses)

At its simplest, vLLM Judge lets you ask an LLM to evaluate content with minimal setup:

```python
from vllm_judge import VLLMJudgeClient

# Initialize client in direct mode (talking directly to vLLM)
client = VLLMJudgeClient(
    direct_mode=True,
    vllm_api_base="http://your-vllm-server:8000/v1"
)

# Get a basic evaluation
result = client.evaluate_text(
    text="This product is fantastic and exceeded my expectations!",
    evaluation_criteria="Determine if this review expresses a positive or negative sentiment.",
    judge_model_id="Qwen/Qwen2.5-7B-Instruct"
)

# The LLM's raw response
print(result['result']['judgment'])
```

This approach:
- Uses no templates or special formatting
- Returns the raw LLM output
- Is perfect for exploration and quick evaluations

### Level 2: Adding Output Guidance in Your Prompt

For more control over the response format, simply include format instructions in your criteria:

```python
result = client.evaluate_text(
    text="The capital of France is Paris.",
    evaluation_criteria="Determine if this statement is factually correct. Answer with CORRECT or INCORRECT.",
    judge_model_id="Qwen/Qwen2.5-7B-Instruct"
)
```

### Level 3: Using the Output Format Instruction Parameter

For a cleaner separation between criteria and format instructions:

```python
result = client.evaluate_text(
    text="The capital of France is Paris.",
    evaluation_criteria="Determine if this statement is factually correct.",
    output_format_instruction="Answer with only the word CORRECT or INCORRECT.",
    judge_model_id="Qwen/Qwen2.5-7B-Instruct"
)
```

### Level 4: Adding Reasoning to Evaluations

Request explanations along with judgments:

```python
result = client.evaluate_text(
    text="The capital of France is Paris.",
    evaluation_criteria="Determine if this statement is factually correct.",
    output_format_instruction="Answer with only the word CORRECT or INCORRECT.",
    judge_model_id="Qwen/Qwen2.5-7B-Instruct",
    provide_reasoning=True
)

print(f"Judgment: {result['result']['judgment']}")
print(f"Reasoning: {result['result']['reasoning']}")
```

### Level 5: Using Custom Prompt Segments

For more control over how the prompt is constructed:

```python
result = client.evaluate_text(
    text="def fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]",
    evaluation_criteria="Evaluate if this function correctly implements the Fibonacci sequence.",
    judge_model_id="Qwen/Qwen2.5-7B-Instruct",
    custom_prompt_segments={
        "system_message": "You are an expert Python developer with deep knowledge of algorithms.",
        "user_instruction_prefix": "Please review this code for correctness:\n\n{evaluation_criteria}\n\nCode to review:\n\n"
    },
    output_format_instruction="Provide your assessment as JSON: {\"correct\": true|false, \"issues\": [\"list of issues\"]}"
)
```

### Level 6: Comparing Texts

Compare two pieces of text:

```python
result = client.compare_texts(
    text_A="Python is a programming language with clean syntax.",
    text_B="Python is a high-level programming language known for its readability and simple syntax.",
    comparison_criteria="Which text provides a more complete description of Python?",
    judge_model_id="Qwen/Qwen2.5-7B-Instruct",
    output_format_instruction="Answer with A, B, or EQUAL, followed by a brief explanation."
)
```

### Level 7: Creating Basic Templates

For repeated evaluations with similar criteria, create reusable templates:

```python
# Create a template for sentiment analysis
template = client.create_template({
    "template_name": "Sentiment Analysis",
    "description": "Template for analyzing sentiment in text",
    "prompt_structure": {
        "system_message": "You are an expert in sentiment analysis.",
        "user_instruction_prefix": "Analyze the sentiment of the following text:\n\n{evaluation_criteria}\n\nText to analyze:\n\n",
        "user_instruction_suffix": "\n\n{output_format_instruction}"
    },
    "output_parser_rules": {
        "type": "text"
    }
})

# Use the template
result = client.evaluate_text(
    text="This product is fantastic!",
    evaluation_criteria="Consider the emotional tone and word choice.",
    judge_model_id="Qwen/Qwen2.5-7B-Instruct",
    prompt_template_id=template["template_id"],
    output_format_instruction="Respond with POSITIVE, NEGATIVE, or NEUTRAL."
)
```

### Level 8: Advanced Templates with Structured Output Parsing

Create templates with specific output formats that get automatically parsed:

```python
# Create a template with structured parsing
template = client.create_template({
    "template_name": "Code Quality",
    "description": "Template for evaluating code quality",
    "prompt_structure": {
        "system_message": "You are an expert software developer.",
        "user_instruction_prefix": "Evaluate this code for quality:\n\n{evaluation_criteria}\n\nCode to evaluate:\n\n",
        "user_instruction_suffix": "\n\n{output_format_instruction}"
    },
    "output_parser_rules": {
        "type": "json",
        "format": {
            "quality_score": "number (1-5)",
            "readability": "number (1-5)",
            "efficiency": "number (1-5)",
            "best_practices": "boolean"
        }
    }
})

# Use with JSON response parsing
result = client.evaluate_text(
    text="def add(a, b):\n    return a + b",
    evaluation_criteria="Evaluate this simple function.",
    judge_model_id="Qwen/Qwen2.5-7B-Instruct",
    prompt_template_id=template["template_id"],
    output_format_instruction="Respond with JSON in format: {\"quality_score\": 1-5, \"readability\": 1-5, \"efficiency\": 1-5, \"best_practices\": true|false}"
)

# The judgment is automatically parsed into a dictionary
print(f"Quality score: {result['result']['judgment']['quality_score']}")
print(f"Readability: {result['result']['judgment']['readability']}")
```

### Level 9: Using Built-in Utility Functions

For common evaluation tasks, use the built-in utility functions:

```python
from vllm_judge import detect_toxicity, evaluate_factual_accuracy, detect_hallucinations, compare_responses

# Toxicity detection
toxicity_result = detect_toxicity(
    client=client,
    text="This product is terrible and I want my money back!",
    judge_model_id="Qwen/Qwen2.5-7B-Instruct",
    provide_reasoning=True
)

# Factual accuracy checking
reference = "The Eiffel Tower is located in Paris, France. It was completed in 1889."
accuracy_result = evaluate_factual_accuracy(
    client=client,
    text="The Eiffel Tower is in Paris and was built in 1890.",
    reference_info=reference,
    judge_model_id="Qwen/Qwen2.5-7B-Instruct"
)

# Hallucination detection
source = "The Earth orbits the Sun at an average distance of 93 million miles."
hallucination_result = detect_hallucinations(
    client=client,
    generated_text="The Earth orbits the Sun at an average distance of 93 million miles, taking exactly 365.24 days to complete one orbit. Mars is the fourth planet from the Sun and takes 687 Earth days to orbit.",
    source_info=source,
    judge_model_id="Qwen/Qwen2.5-7B-Instruct"
)

# Response comparison
comparison_result = compare_responses(
    client=client,
    prompt="Explain how photosynthesis works",
    response_A="Photosynthesis is how plants make food.",
    response_B="Photosynthesis is the process by which plants convert light energy into chemical energy. Plants use sunlight, water, and carbon dioxide to create glucose and oxygen.",
    judge_model_id="Qwen/Qwen2.5-7B-Instruct"
)
```

### Level 10: Production Deployment with Server Mode

For production environments, run vLLM Judge as a dedicated service:

```bash
# Start the server
vllm-judge --vllm-api-base http://your-vllm-server:8000/v1 --host 0.0.0.0 --port 8000
```

```python
# Connect to the server
client = VLLMJudgeClient(base_url="http://your-vllm-judge-server:8000/v1")

# Use exactly the same API as in direct mode
result = client.evaluate_text(
    text="Content to evaluate",
    evaluation_criteria="Evaluation criteria",
    judge_model_id="model-id"
)
```

Server mode benefits:
- Task management and persistence
- Multiple clients can share templates
- Ideal for production workloads

## Code Examples

The repository includes several example scripts demonstrating different usage levels:

- `examples/basic_usage.py` - Simple direct mode examples
- `examples/generic_approach.py` - Using the library without templates
- `examples/template_management.py` - Creating and using templates
- `examples/utility_functions.py` - Using built-in evaluation utilities
- `examples/jupyter_example.ipynb` - Using vLLM Judge in Jupyter notebooks

## Server Configuration

The vLLM Judge server can be configured using environment variables or command-line arguments:

| Variable/Argument | Description | Default |
| --- | --- | --- |
| `--vllm-api-base` | Base URL of the vLLM server | `http://vllm-server:8000/v1` |
| `--vllm-api-key` | API key for the vLLM server (if needed) | `None` |
| `--template-storage-path` | Path to the template storage file | `vllm_judge/templates/default_templates.json` |
| `--host` | Host to bind the server to | `0.0.0.0` |
| `--port` | Port to bind the server to | `8000` |
| `--reload` | Enable auto-reload for development | `False` |

## API Reference

### Client Initialization

```python
# Direct mode
client = VLLMJudgeClient(
    direct_mode=True,
    vllm_api_base="http://your-vllm-server:8000/v1",
    vllm_api_key="your-api-key",  # Optional
    timeout=60  # Optional
)

# Server mode
client = VLLMJudgeClient(
    base_url="http://your-vllm-judge-server:8000/v1",
    timeout=60  # Optional
)
```

### Client Methods

- `evaluate_text()` - Evaluate a single text
- `compare_texts()` - Compare two texts
- `get_status()` - Check the status of an evaluation (server mode only)
- `list_templates()` - List available templates
- `get_template()` - Get a template by ID
- `create_template()` - Create a new template
- `update_template()` - Update an existing template
- `delete_template()` - Delete a template

### Utility Functions

- `detect_toxicity()` - Check if text contains toxic content
- `evaluate_factual_accuracy()` - Check factual accuracy against reference
- `detect_hallucinations()` - Identify hallucinations in generated content
- `compare_responses()` - Compare two AI responses to the same prompt
- `evaluate_code()` - Evaluate code quality
