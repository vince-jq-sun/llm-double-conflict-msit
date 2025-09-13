#!/usr/bin/env python3
"""
Ollama Llama 3.1 8B Instruct Caller
A script to call Llama 3.1 8B Instruct model through Ollama with custom content.
"""

import argparse
import json
import requests
import sys
from typing import Optional, Dict, Any


class OllamaLlamaCaller:
    def __init__(self, model_name: str = "llama3.1:8b-instruct-q4_0", base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama Llama caller.
        
        Args:
            model_name: The Ollama model name (default: llama3.1:8b-instruct-q4_0)
            base_url: The Ollama API base URL (default: http://localhost:11434)
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def check_ollama_status(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def check_model_available(self) -> bool:
        """Check if the specified model is available in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                return self.model_name in available_models
            return False
        except requests.exceptions.RequestException:
            return False
    
    def generate_response(self, prompt: str, max_tokens: Optional[int] = None, 
                         temperature: float = 0.7, stream: bool = False) -> Dict[str, Any]:
        """
        Generate a response from the Llama model.
        
        Args:
            prompt: The input prompt/content
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing the response and metadata
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=300)
            response.raise_for_status()
            
            if stream:
                return {"status": "success", "response": response}
            else:
                result = response.json()
                return {
                    "status": "success",
                    "response": result.get("response", ""),
                    "model": result.get("model", ""),
                    "total_duration": result.get("total_duration", 0),
                    "load_duration": result.get("load_duration", 0),
                    "prompt_eval_count": result.get("prompt_eval_count", 0),
                    "eval_count": result.get("eval_count", 0)
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": f"Request failed: {str(e)}"
            }
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "error": f"JSON decode failed: {str(e)}"
            }
    
    def interactive_mode(self):
        """Run in interactive mode for continuous conversation."""
        print(f"🦙 Ollama Llama 3.1 8B Instruct Interactive Mode")
        print(f"Model: {self.model_name}")
        print("Type 'quit', 'exit', or 'q' to exit")
        print("-" * 50)
        
        while True:
            try:
                prompt = input("\n💬 You: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not prompt:
                    continue
                
                print("🤔 Thinking...")
                result = self.generate_response(prompt)
                
                if result["status"] == "success":
                    print(f"🦙 Llama: {result['response']}")
                    if result.get('eval_count'):
                        print(f"📊 Tokens: {result['eval_count']} | Duration: {result.get('total_duration', 0) / 1e9:.2f}s")
                else:
                    print(f"❌ Error: {result['error']}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Call Llama 3.1 8B Instruct through Ollama")
    parser.add_argument("--prompt", "-p", type=str, help="The prompt/content to send to the model")
    parser.add_argument("--file", "-f", type=str, help="Read prompt from file")
    parser.add_argument("--model", "-m", type=str, default="llama3.1:8b-instruct-q4_0", 
                       help="Ollama model name (default: llama3.1:8b-instruct-q4_0)")
    parser.add_argument("--max-tokens", "-t", type=int, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", "-temp", type=float, default=0.7, 
                       help="Sampling temperature (default: 0.7)")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--url", "-u", type=str, default="http://localhost:11434",
                       help="Ollama API base URL (default: http://localhost:11434)")
    parser.add_argument("--check", "-c", action="store_true", 
                       help="Check Ollama status and model availability")
    
    args = parser.parse_args()
    
    # Initialize the caller
    caller = OllamaLlamaCaller(model_name=args.model, base_url=args.url)
    
    # Check status if requested
    if args.check:
        print("🔍 Checking Ollama status...")
        if not caller.check_ollama_status():
            print("❌ Ollama is not running or not accessible")
            print("💡 Make sure Ollama is installed and running: ollama serve")
            sys.exit(1)
        
        print("✅ Ollama is running")
        
        if not caller.check_model_available():
            print(f"❌ Model '{args.model}' is not available")
            print("💡 Pull the model first: ollama pull llama3.1:8b-instruct-q4_0")
            sys.exit(1)
        
        print(f"✅ Model '{args.model}' is available")
        return
    
    # Interactive mode
    if args.interactive:
        if not caller.check_ollama_status():
            print("❌ Ollama is not running. Please start it with: ollama serve")
            sys.exit(1)
        caller.interactive_mode()
        return
    
    # Get prompt from arguments or file
    prompt = None
    if args.prompt:
        prompt = args.prompt
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
        except FileNotFoundError:
            print(f"❌ File not found: {args.file}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading file: {str(e)}")
            sys.exit(1)
    else:
        print("❌ Please provide a prompt with --prompt or --file, or use --interactive mode")
        sys.exit(1)
    
    # Check Ollama status
    if not caller.check_ollama_status():
        print("❌ Ollama is not running. Please start it with: ollama serve")
        sys.exit(1)
    
    # Generate response
    print("🤔 Generating response...")
    result = caller.generate_response(
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    if result["status"] == "success":
        print("\n" + "="*50)
        print("🦙 Llama Response:")
        print("="*50)
        print(result["response"])
        print("="*50)
        
        # Print statistics
        if result.get('eval_count'):
            duration = result.get('total_duration', 0) / 1e9
            tokens_per_sec = result['eval_count'] / duration if duration > 0 else 0
            print(f"📊 Stats: {result['eval_count']} tokens | {duration:.2f}s | {tokens_per_sec:.1f} tok/s")
    else:
        print(f"❌ Error: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
