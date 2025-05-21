"""
Example of using vLLM Judge with the generic approach (no templates)
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

print("Using vLLM Judge in Direct Mode with Generic Approach")
print("=" * 70)

# Example 1: Generic question evaluation - no templates, just pure assessment
question = "What is the capital of France?"
answer = "The capital of France is Paris."

result = client.evaluate_text(
    text=answer,
    evaluation_criteria="Assess if this answer correctly identifies the capital of France",
    judge_model_id=judge_model_id
)

print("Example 1: Simple Factual Assessment")
print(f"Question: {question}")
print(f"Answer: {answer}")
print(f"Raw response: {result}")
print(f"Evaluation: {result['result']['judgment']}")
print("\n" + "=" * 70)

# Example 2: Comparison without enforced structure
text1 = "The solar system consists of the Sun and everything that orbits around it, including planets, moons, asteroids, and comets."
text2 = "Our solar system has 8 planets (Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune) plus dwarf planets like Pluto, all orbiting around the Sun."

result = client.compare_texts(
    text_A=text1,
    text_B=text2,
    comparison_criteria="Which text provides more specific details about the components of our solar system?",
    judge_model_id=judge_model_id,
    # No template specified - using generic approach
    output_format_instruction="Briefly explain which text (A or B) is more detailed and why."
)

print("Example 2: Comparison Without Templates")
print(f"Text A: {text1}")
print(f"Text B: {text2}")
print(f"Raw response: {result}")
print(f"Judgment: {result['result']['judgment']}")
print("\n" + "=" * 70)

# Example 3: Open-ended evaluation with reasoning
essay = """
Climate change is a pressing global issue that requires immediate attention. Rising temperatures, melting ice caps, and extreme weather events are just some of the observable effects. While there are multiple causes, human activities such as burning fossil fuels and deforestation are significant contributors. Solutions include transitioning to renewable energy, improving energy efficiency, and implementing policies to reduce carbon emissions.
"""

result = client.evaluate_text(
    text=essay,
    evaluation_criteria="Evaluate this short essay on climate change for accuracy of information, completeness, and clarity of explanation.",
    judge_model_id=judge_model_id,
    provide_reasoning=True
)

print("Example 3: Open-ended Essay Evaluation")
print(f"Essay: {essay}")
print(f"Raw response: {result}")
print(f"Evaluation: {result['result']['judgment']}")
if result['result'].get('reasoning'):
    print(f"Reasoning: {result['result']['reasoning']}")
print("\n" + "=" * 70)

# Example 4: Code review with custom format but no fixed template
code = """
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib
"""

result = client.evaluate_text(
    text=code,
    evaluation_criteria="Evaluate this Python function that generates Fibonacci sequences for correctness, efficiency, and readability.",
    judge_model_id=judge_model_id,
    # No template, but specifying a format instruction
    output_format_instruction="Please provide your evaluation in these categories: correctness (correct/incorrect), efficiency (1-5 scale), readability (1-5 scale), and suggestions for improvement."
)

print("Example 4: Code Evaluation with Custom Format")
print(f"Code: {code}")
print(f"Raw response: {result}")
print(f"Evaluation: {result['result']['judgment']}")
print("\n" + "=" * 70)

# Example 5: Using templates for specific tasks (still an option)
from vllm_judge import detect_toxicity

print("Example 5: Using Templates When Needed")
result = detect_toxicity(
    client=client,
    text="This product is absolutely terrible. I want my money back!",
    judge_model_id=judge_model_id,
    provide_reasoning=True
)
print(f"Raw response: {result}")
print(f"Toxicity judgment: {result['result']['judgment']}")
if result['result'].get('reasoning'):
    print(f"Reasoning: {result['result']['reasoning']}")