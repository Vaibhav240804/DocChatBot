#!/usr/bin/env python3
"""
Simple run script for the RAG Documentation Chatbot
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if requirements are installed"""
    try:
        import streamlit
        import langchain
        import chromadb
        from langchain_groq import ChatGroq
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("❌ .env file not found")
        print("Please copy .env.template to .env and add your Groq API key")
        return False
    
    # Read .env file
    with open(env_file) as f:
        content = f.read()
    
    if 'GROQ_API_KEY=' not in content or 'your_groq_api_key_here' in content:
        print("❌ GROQ_API_KEY not configured in .env file")
        print("Please add your actual Groq API key to the .env file")
        print("Get your API key from: https://console.groq.com/keys")
        return False
    
    print("✅ Environment configuration looks good")
    return True

def main():
    """Main function to run the application"""
    print("🚀 Starting RAG Documentation Chatbot...")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment
    if not check_env_file():
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("vector_db", exist_ok=True)
    os.makedirs("graph_db", exist_ok=True)
    
    print("✅ All checks passed!")
    print("\n🌟 Starting Streamlit application...")
    print("📱 The app will open in your browser at http://localhost:8501")
    print("🔄 To stop the application, press Ctrl+C")
    print("=" * 50)
    
    # Run streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()