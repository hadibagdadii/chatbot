#!/usr/bin/env python3
"""
Quick test to verify Ollama is working before starting the full app
Run this from the chatbot directory: python test_ollama.py
"""

import requests
import sys

def test_ollama():
    print("🔍 Testing Ollama connection...\n")
    
    # Test 1: Check if Ollama is running
    print("1. Checking if Ollama is running...")
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("   ✓ Ollama is running")
            models = response.json().get("models", [])
            print(f"   ✓ Available models: {[m['name'] for m in models]}")
        else:
            print(f"   ✗ Ollama returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Cannot connect to Ollama: {e}")
        print("   → Start Ollama with: ollama serve")
        return False
    
    # Test 2: Check if llama3.2 is available
    print("\n2. Checking for llama3.2 model...")
    model_names = [m['name'] for m in models]
    if any('llama3.2' in name for name in model_names):
        print("   ✓ llama3.2 model found")
    else:
        print("   ✗ llama3.2 model not found")
        print("   → Pull model with: ollama pull llama3.2")
        return False
    
    # Test 3: Try generating an embedding
    print("\n3. Testing embedding generation...")
    try:
        embed_response = requests.post(
            "http://127.0.0.1:11434/api/embeddings",
            json={"model": "llama3.2", "prompt": "test"},
            timeout=30
        )
        if embed_response.status_code == 200:
            print("   ✓ Embedding generation works")
        else:
            print(f"   ✗ Embedding failed: {embed_response.text}")
            return False
    except Exception as e:
        print(f"   ✗ Embedding test failed: {e}")
        return False
    
    print("\n✅ All tests passed! Ollama is ready.")
    print("\nYou can now start the server with:")
    print("   uvicorn backend.main:app --reload --port 8000")
    return True

if __name__ == "__main__":
    success = test_ollama()
    sys.exit(0 if success else 1)