#!/usr/bin/env python3
"""
Startup script for the viclab Real-time Video Analysis API
"""

import os
import sys
import subprocess
import logging

# Add the viclab package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "viclab"))

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import cv2
        import torch
        import transformers
        print("✅ All core dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def start_server():
    """Start the FastAPI server."""
    print("🚀 Starting viclab Real-time Video Analysis API...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📖 API docs will be available at: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop the server")
    
    try:
        # Change to the correct directory
        os.chdir(os.path.dirname(__file__))
        
        # Start uvicorn server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api.main:app", 
            "--reload", 
            "--host", "0.0.0.0",
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

if __name__ == "__main__":
    if check_dependencies():
        start_server()
    else:
        sys.exit(1) 