"""
Basic usage examples for vLLM Judge - from simplest to more advanced.
"""

# Import the library
from vllm_judge import VLLMJudgeClient

vllm_api_base = "http://localhost:8080/v1"  # TODO: Pass as a CLI argument
judge_model_id = "qwen2"  # TODO: Pass as a CLI argument

# Initialize the client in direct mode
client = VLLMJudgeClient(
    direct_mode=True,
    vllm_api_base=vllm_api_base
)

print("LEVEL 1: Basic Evaluation (Raw LLM Response)")
print("=" * 70)

# LEVEL 1: Basic Evaluation - Just ask the model to evaluate with minimal setup
result = client.evaluate_text(
    text="This product is fantastic and exceeded my expectations!",
    evaluation_criteria="Determine if this review expresses a positive or negative sentiment.",
    judge_model_id=judge_model_id
)

print(f"Raw LLM response: {result['result']['judgment']}")
print("\n" + "=" * 70)


print("LEVEL 2: Adding Format Guidance in Criteria")
print("=" * 70)

# LEVEL 2: Adding Format Guidance in Criteria
result = client.evaluate_text(
    text="The capital of France is Paris.",
    evaluation_criteria="Determine if this statement is factually correct. Answer with CORRECT or INCORRECT.",
    judge_model_id=judge_model_id
)

print(f"Response with embedded format guide: {result['result']['judgment']}")
print("\n" + "=" * 70)


print("LEVEL 3: Using Output Format Parameter")
print("=" * 70)

# LEVEL 3: Using the Output Format Instruction Parameter
result = client.evaluate_text(
    text="The capital of France is Paris.",
    evaluation_criteria="Determine if this statement is factually correct.",
    output_format_instruction="Answer with only the word CORRECT or INCORRECT.",
    judge_model_id=judge_model_id
)

print(f"Response with format parameter: {result['result']['judgment']}")
print("\n" + "=" * 70)


print("LEVEL 4: Adding Reasoning")
print("=" * 70)

# LEVEL 4: Adding Reasoning to Evaluations
result = client.evaluate_text(
    text="The Earth orbits the Sun.",
    evaluation_criteria="Determine if this statement is factually correct.",
    output_format_instruction="Answer with only the word CORRECT or INCORRECT.",
    judge_model_id=judge_model_id,
    provide_reasoning=True
)

print(f"Judgment: {result['result']['judgment']}")
if result['result'].get('reasoning'):
    print(f"Reasoning: {result['result']['reasoning']}")
print("\n" + "=" * 70)


print("LEVEL 5: Using Custom Prompt Segments")
print("=" * 70)

# LEVEL 5: Using Custom Prompt Segments for more control
code = """def add(a, b):
    return a + b"""

result = client.evaluate_text(
    text=code,
    evaluation_criteria="Evaluate if this function correctly implements addition.",
    judge_model_id=judge_model_id,
    custom_prompt_segments={
        "system_message": "You are an expert Python developer with deep knowledge of algorithms.",
        "user_instruction_prefix": "Please review this code for correctness:\n\n{evaluation_criteria}\n\nCode to review:\n\n"
    },
    output_format_instruction="Provide your assessment in JSON: {\"correct\": true|false, \"issues\": [\"list of issues\"]}"
)

print(f"Code: {code}")
print(f"Assessment with custom prompt: {result['result']['judgment']}")
print("\n" + "=" * 70)


print("LEVEL 6: Comparing Texts")
print("=" * 70)

# LEVEL 6: Comparing Texts
result = client.compare_texts(
    text_A="Python is a programming language with clean syntax.",
    text_B="Python is a high-level programming language known for its readability and simple syntax.",
    comparison_criteria="Which text provides a more complete description of Python?",
    judge_model_id=judge_model_id,
    output_format_instruction="Answer with A, B, or EQUAL, followed by a brief explanation."
)

print(f"Text A: Python is a programming language with clean syntax.")
print(f"Text B: Python is a high-level programming language known for its readability and simple syntax.")
print(f"Comparison result: {result['result']['judgment']}")
print("\n" + "=" * 70)


print("Using Built-in Utility Functions")
print("=" * 70)

# Using a built-in utility function for a common task
from vllm_judge import detect_toxicity

toxicity_result = detect_toxicity(
    client=client,
    text="This customer service is excellent!",
    judge_model_id=judge_model_id,
    provide_reasoning=True
)

print(f"Toxicity detection: {toxicity_result['result']['judgment']}")
if toxicity_result['result'].get('reasoning'):
    print(f"Reasoning: {toxicity_result['result']['reasoning']}")