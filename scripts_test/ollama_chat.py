#!/usr/bin/env python3
"""
Script to interact with Llama 3.1 model via Ollama API
Requires Ollama to be running locally and the llama3.1 model to be pulled
"""

import requests
import json
import sys
from typing import Dict, Any, Optional

class OllamaChat:
    def __init__(self, model: str = "llama3.1:8b-instruct-q4_0", base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama chat client
        
        Args:
            model: The model name (default: llama3.1:8b-instruct-q4_0)
            base_url: The Ollama API base URL (default: http://localhost:11434)
        """
        self.model = model
        self.base_url = base_url
        self.chat_url = f"{base_url}/api/chat"
        self.generate_url = f"{base_url}/api/generate"
        
    def check_ollama_status(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> Optional[Dict[str, Any]]:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error listing models: {e}")
            return None
    
    def send_message(self, message: str, stream: bool = False) -> Optional[str]:
        """
        Send a message to the model
        
        Args:
            message: The message to send
            stream: Whether to stream the response (default: False)
            
        Returns:
            The model's response or None if error
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ],
            "stream": stream
        }
        
        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=60,
                stream=stream
            )
            
            if response.status_code == 200:
                if stream:
                    # Handle streaming response
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                if 'message' in chunk and 'content' in chunk['message']:
                                    content = chunk['message']['content']
                                    print(content, end='', flush=True)
                                    full_response += content
                                if chunk.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                    print()  # New line after streaming
                    return full_response
                else:
                    # Handle non-streaming response
                    result = response.json()
                    if 'message' in result and 'content' in result['message']:
                        return result['message']['content']
            else:
                print(f"Error: HTTP {response.status_code}")
                print(response.text)
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error sending message: {e}")
            return None
    
    def interactive_chat(self):
        """Start an interactive chat session"""
        print(f"Starting interactive chat with {self.model}")
        print("Type 'quit', 'exit', or 'q' to end the session")
        print("Type 'clear' to clear the conversation history")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    print("Conversation cleared!")
                    continue
                
                if not user_input:
                    continue
                
                print(f"\n{self.model}: ", end='')
                response = self.send_message(user_input, stream=True)
                
                if response is None:
                    print("Failed to get response from model.")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break

def main():
    """Main function"""
    # Pre-defined message variable - set this to send a specific message
    # If empty string "", will use interactive mode
    PREDEFINED_MESSAGE = "MSIT Task Instruction: \n\n    Goal: in each row, there's a number (or an alphabetic letter) that only appears once; report its position.\n    - Counting from left to right, the candidate positions are 1, 2, 3. \n\n    Important:\n    - Return your answers for all rows in order, separated by spaces. \n    - Treat each row independently.\n    - respond with ONLY the answer (a single number). \n\nHere comes the stimuli.\n\n1 3 1"
    # PREDEFINED_MESSAGE = "which model are you?"

    # Initialize the chat client
    chat = OllamaChat()
    
    # Check if Ollama is running
    print("Checking Ollama status...")
    if not chat.check_ollama_status():
        print("❌ Ollama is not running or not accessible at http://localhost:11434")
        print("Please make sure Ollama is running with: ollama serve")
        sys.exit(1)
    
    print("✅ Ollama is running")
    
    # List available models
    print("\nChecking available models...")
    models = chat.list_models()
    if models and 'models' in models:
        model_names = [model['name'] for model in models['models']]
        print(f"Available models: {', '.join(model_names)}")
        
        # Check if our target model is available
        if chat.model not in model_names:
            print(f"⚠️  Model '{chat.model}' not found.")
            print(f"To pull the model, run: ollama pull {chat.model}")
            
            # Ask user if they want to continue with a different model
            if model_names:
                print(f"Available models: {', '.join(model_names)}")
                choice = input("Enter a model name to use instead, or press Enter to exit: ").strip()
                if choice and choice in model_names:
                    chat.model = choice
                else:
                    sys.exit(1)
            else:
                sys.exit(1)
    
    # Check if predefined message is set
    if PREDEFINED_MESSAGE:
        # Use predefined message mode
        print(f"Sending predefined message to {chat.model}...")
        print(f"Message: {PREDEFINED_MESSAGE}")
        print(f"\nResponse from {chat.model}:")
        print("-" * 50)
        
        response = chat.send_message(PREDEFINED_MESSAGE, stream=True)
        if response is None:
            print("Failed to get response from model.")
            sys.exit(1)
    elif len(sys.argv) > 1:
        # Single message mode from command line arguments
        message = ' '.join(sys.argv[1:])
        print(f"Sending message to {chat.model}...")
        print(f"Message: {message}")
        print(f"\nResponse from {chat.model}:")
        print("-" * 50)
        
        response = chat.send_message(message, stream=True)
        if response is None:
            print("Failed to get response from model.")
            sys.exit(1)
    else:
        # Interactive mode
        chat.interactive_chat()

if __name__ == "__main__":
    main()
