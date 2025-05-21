from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setup(
    name="vllm_judge",
    version="0.1.0",
    author="Sai Chandra Pandraju (TrustyAI team)",
    author_email="saichandrapandraju@gmail.com",
    description="A model-agnostic adapter for enabling LLM-as-a-judge capabilities with vLLM-hosted models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saichandrapandraju/vllm-judge",
    project_urls={
        "Bug Tracker": "https://github.com/saichandrapandraju/vllm-judge/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(include=["vllm_judge", "vllm_judge.*"]),
    package_data={
        "vllm_judge": ["templates/default_templates.json"],
    },
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.110.0",
        "uvicorn>=0.27.0",
        "httpx>=0.26.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.1.0",
        "backoff>=2.2.1",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vllm-judge=vllm_judge.cli:main",
        ],
    },
)