#!/usr/bin/env python3
"""
Setup script for MSIT-LLM framework
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Set up the development environment."""
    print("🚀 Setting up MSIT-LLM framework...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print("✅ Python version check passed")
    
    # Create necessary directories
    dirs_to_create = [
        "results/test_runs",
        "temp",
        "logs"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")
    
    # Copy environment file if it doesn't exist
    if not Path(".env").exists():
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("📋 Created .env file from template")
            print("⚠️  Please edit .env file with your API keys")
        else:
            print("⚠️  No .env.example found")
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. For local models: pip install -r requirements_local.txt")
    print("4. Run tests: python scripts_test/msit_api_test.py --help")

if __name__ == "__main__":
    setup_environment()
