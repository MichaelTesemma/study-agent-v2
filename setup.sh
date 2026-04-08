#!/usr/bin/env bash
set -e

echo "=== Study Agent Setup ==="
echo ""

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt --quiet
echo "Dependencies installed."

# Check if Ollama is running
echo ""
echo "Checking Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama is running."
else
    echo ""
    echo "WARNING: Ollama does not appear to be running."
    echo ""
    echo "To fix this:"
    echo "  1. Install Ollama from https://ollama.com"
    echo "  2. Start Ollama (run 'ollama serve' in a terminal, or open the Ollama app)"
    echo "  3. Pull the default model: ollama pull llama3.2"
    echo ""
    echo "You can still run the setup, but the app needs Ollama to work."
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run the app:"
echo "  python3 server.py"
echo ""
echo "Then open http://localhost:8000 in your browser."
