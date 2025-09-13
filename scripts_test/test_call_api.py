#!/usr/bin/env python3
"""
Simple test script for the call_api function with timeout handling
"""

import signal
import sys
from msit_api_test import call_api, load_api_keys

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("API call timed out")

def main():
    """Test the call_api function with a simple message."""
    
    # Test parameters
    model_name = "gemini-2.0-flash"  # You can change this to any model you want to test
    prompt = "what date is it?"
    max_tokens = 500
    temperature = 0
    timeout_seconds = 30  # 30 second timeout
    
    print(f"Testing call_api function...")
    print(f"Model: {model_name}")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}")
    print(f"Timeout: {timeout_seconds} seconds")
    print("-" * 50)
    
    # Load API keys
    print("Loading API keys...")
    api_keys = load_api_keys("API.json")
    
    if not api_keys:
        print("ERROR: No API keys loaded!")
        return
    
    print("API keys loaded successfully")
    
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    # Call the API
    try:
        print("Making API call...")
        response = call_api(
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            api_keys=api_keys
        )
        
        # Cancel the alarm
        signal.alarm(0)
        
        print("Response received:")
        print(response)
        
    except TimeoutError:
        print(f"ERROR: API call timed out after {timeout_seconds} seconds")
    except KeyboardInterrupt:
        print("\nERROR: Interrupted by user")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Make sure to cancel any remaining alarm
        signal.alarm(0)

if __name__ == "__main__":
    main()
