import argparse
import os
import sys
import uvicorn
from importlib.metadata import version

from vllm_judge.core.config import settings


def get_version() -> str:
    """Get the version of the package."""
    try:
        return version("vllm-judge")
    except Exception:
        return "0.1.0"  # Default version


def main() -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(
        description="vLLM Judge: A model-agnostic adapter for enabling LLM-as-a-judge capabilities"
    )
    
    parser.add_argument(
        "--version", "-v", action="store_true", help="Show version and exit"
    )
    
    # Server options
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    
    # vLLM connection options
    parser.add_argument(
        "--vllm-api-base", 
        type=str, 
        help="Base URL for the vLLM API (e.g., http://vllm-server:8000/v1)"
    )
    parser.add_argument(
        "--vllm-api-key", 
        type=str, 
        help="API key for the vLLM API (if required)"
    )
    
    # Template options
    parser.add_argument(
        "--template-storage-path",
        type=str,
        help="Path to the template storage file"
    )
    
    args = parser.parse_args()
    
    # Show version and exit if requested
    if args.version:
        print(f"vLLM Judge v{get_version()}")
        sys.exit(0)
    
    # Set environment variables for configuration
    if args.vllm_api_base:
        os.environ["VLLM_API_BASE"] = args.vllm_api_base
    
    if args.vllm_api_key:
        os.environ["VLLM_API_KEY"] = args.vllm_api_key
    
    if args.template_storage_path:
        os.environ["TEMPLATE_STORAGE_PATH"] = args.template_storage_path
    
    # Print configuration info
    print(f"Starting vLLM Judge v{get_version()}")
    print(f"Server: {args.host}:{args.port}")
    print(f"vLLM API Base: {settings.VLLM_API_BASE}")
    print(f"Template Storage: {settings.TEMPLATE_STORAGE_PATH}")
    
    # Start the server
    uvicorn.run(
        "vllm_judge.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()