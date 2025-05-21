"""
Examples of using built-in utility functions for common evaluation tasks with vLLM Judge.
"""

import json
from vllm_judge import (
    VLLMJudgeClient,
    detect_toxicity,
    evaluate_factual_accuracy,
    detect_hallucinations,
    compare_responses,
    evaluate_code
)

vllm_api_base = "http://localhost:8000/v1"  # TODO: Pass as a CLI argument
judge_model_id = "qwen2"  # TODO: Pass as a CLI argument

# Initialize the client in direct mode
client = VLLMJudgeClient(
    direct_mode=True,
    vllm_api_base=vllm_api_base
)

print("UTILITY FUNCTION EXAMPLES")
print("=" * 80)


# Example 1: Toxicity Detection
print("\n\nEXAMPLE 1: Toxicity Detection")
print("-" * 50)

texts_to_check = [
    "This product is amazing! I'm very satisfied with my purchase.",
    "This is the worst service I've ever experienced. The staff was incredibly rude!",
    "I don't think this solution will work for our needs."
]

print("Checking multiple texts for toxicity:\n")

for i, text in enumerate(texts_to_check, 1):
    print(f"Text {i}: \"{text}\"")
    
    toxicity_result = detect_toxicity(
        client=client,
        text=text,
        judge_model_id=judge_model_id,
        provide_reasoning=True
    )
    
    print(f"Judgment: {toxicity_result['result']['judgment']}")
    if toxicity_result['result'].get('reasoning'):
        print(f"Reasoning: {toxicity_result['result']['reasoning']}")
    print()


# Example 2: Factual Accuracy Evaluation
print("\n\nEXAMPLE 2: Factual Accuracy Evaluation")
print("-" * 50)

reference_info = """
The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.
It was named after the engineer Gustave Eiffel, whose company designed and built the tower.
The tower was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair.
It is 330 meters (1,083 ft) tall, and was the tallest man-made structure in the world for 41 years 
until the completion of the Chrysler Building in New York City in 1930.
"""

statements_to_check = [
    "The Eiffel Tower is located in Paris, France and was built for the 1889 World's Fair.",
    "The Eiffel Tower was built in 1789 and is currently the tallest structure in Europe.",
    "The Eiffel Tower, designed by Gustave Eiffel, stands at 330 meters tall and was the world's tallest structure until 1930 when the Chrysler Building was completed."
]

print("Checking statements against reference information:\n")
print(f"Reference information: {reference_info.strip()}\n")

for i, statement in enumerate(statements_to_check, 1):
    print(f"Statement {i}: \"{statement}\"")
    
    accuracy_result = evaluate_factual_accuracy(
        client=client,
        text=statement,
        reference_info=reference_info,
        judge_model_id=judge_model_id
    )
    
    print(f"Result:")
    # Pretty print if it's a dictionary
    if isinstance(accuracy_result['result']['judgment'], dict):
        print(json.dumps(accuracy_result['result']['judgment'], indent=2))
    else:
        print(accuracy_result['result']['judgment'])
    print()


# Example 3: Hallucination Detection
print("\n\nEXAMPLE 3: Hallucination Detection")
print("-" * 50)

source_text = """
The solar system consists of the Sun and everything that orbits around it, including
planets, moons, asteroids, and comets. There are eight planets in our solar system:
Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Pluto was once
considered the ninth planet but was reclassified as a dwarf planet in 2006.
"""

generated_texts = [
    "Our solar system has eight planets that orbit the Sun: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Pluto was formerly classified as a planet but is now considered a dwarf planet.",
    "The solar system contains nine planets orbiting the Sun: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, and Pluto. Each planet has its own unique characteristics and composition.",
    "Our solar system consists of the Sun and eight planets. These planets are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. There are also dwarf planets like Pluto, which was reclassified in 2006. The largest planet is Jupiter, which has over 79 moons and a giant storm called the Great Red Spot that has been raging for hundreds of years."
]

print("Detecting hallucinations in generated texts:\n")
print(f"Source information: {source_text.strip()}\n")

for i, text in enumerate(generated_texts, 1):
    print(f"Generated text {i}: \"{text}\"")
    
    hallucination_result = detect_hallucinations(
        client=client,
        generated_text=text,
        source_info=source_text,
        judge_model_id=judge_model_id
    )
    
    print(f"Result:")
    # Pretty print if it's a dictionary
    if isinstance(hallucination_result['result']['judgment'], dict):
        print(json.dumps(hallucination_result['result']['judgment'], indent=2))
    else:
        print(hallucination_result['result']['judgment'])
    print()


# Example 4: Compare Responses
print("\n\nEXAMPLE 4: Compare Responses")
print("-" * 50)

questions = [
    "What is photosynthesis?",
    "How does a computer work?",
    "Why is the sky blue?"
]

response_pairs = [
    # Photosynthesis responses
    (
        "Photosynthesis is how plants make food.",
        "Photosynthesis is the process by which plants convert light energy into chemical energy. During this process, plants use sunlight, water, and carbon dioxide to create glucose and oxygen. The glucose serves as food for the plant, while oxygen is released as a byproduct."
    ),
    # Computer responses
    (
        "Computers work by processing data through a CPU and memory.",
        "Computers work by executing instructions stored in memory. The central processing unit (CPU) fetches instructions, decodes them, executes operations, and stores results. Modern computers use binary (0s and 1s) for all operations and data storage. Input devices collect information from users, while output devices display results. Data can be stored long-term in storage devices like hard drives or SSDs."
    ),
    # Sky responses
    (
        "The sky is blue because of how sunlight interacts with our atmosphere.",
        "The sky appears blue due to a phenomenon called Rayleigh scattering. As sunlight passes through the atmosphere, shorter wavelengths (blues) scatter more than longer wavelengths (reds). This scattered blue light comes to our eyes from all directions, making the sky appear blue during the day."
    )
]

print("Comparing pairs of responses to questions:\n")

for i, (question, (response_A, response_B)) in enumerate(zip(questions, response_pairs), 1):
    print(f"Question {i}: \"{question}\"")
    print(f"Response A: \"{response_A}\"")
    print(f"Response B: \"{response_B}\"")
    
    comparison_result = compare_responses(
        client=client,
        prompt=question,
        response_A=response_A,
        response_B=response_B,
        judge_model_id=judge_model_id,
        provide_reasoning=True
    )
    
    print(f"Better response: {comparison_result['result']['judgment']}")
    if comparison_result['result'].get('reasoning'):
        print(f"Reasoning: {comparison_result['result']['reasoning']}")
    print()


# Example 5: Code Evaluation
print("\n\nEXAMPLE 5: Code Evaluation")
print("-" * 50)

code_samples = [
    # Example 1: Basic function with an issue
    """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
""",
    # Example 2: More complex function with efficiency issues
    """
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
""",
    # Example 3: Recursive function with potential stack issues
    """
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        fib_list = fibonacci(n-1)
        fib_list.append(fib_list[-1] + fib_list[-2])
        return fib_list
"""
]

requirements = [
    "Create a function that calculates the average of a list of numbers.",
    "Create a function that checks if a number is prime.",
    "Create a function that generates a Fibonacci sequence of length n."
]

print("Evaluating code samples:\n")

for i, (code, req) in enumerate(zip(code_samples, requirements), 1):
    print(f"Code Sample {i}:")
    print(f"Requirement: {req}")
    print(f"Code:\n{code}")
    
    code_result = evaluate_code(
        client=client,
        code=code,
        requirements=req,
        judge_model_id=judge_model_id
    )
    
    print(f"Evaluation:")
    # Pretty print if it's a dictionary
    if isinstance(code_result['result']['judgment'], dict):
        print(json.dumps(code_result['result']['judgment'], indent=2))
    else:
        print(code_result['result']['judgment'])
    print()