#!/bin/bash

# Set Ollama to use more GPU memory
export OLLAMA_GPU_LAYERS=35

# Increase system limits for better performance
ulimit -n 1000000

# Run the RAG test
python rag_ollama.py "$@" 