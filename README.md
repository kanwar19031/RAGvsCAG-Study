# RAG vs CAG: A Cost-Optimization Case Study

This repository presents a comparative analysis between Retrieval-Augmented Generation (RAG) and Cache-Augmented Generation (CAG) architectures, with a focus on API cost optimization and local deployment capabilities. This implementation is inspired by the paper [Don't Do RAG](https://github.com/hhhuang/CAG) by Chan et al.

## Project Overview

This case study explores:
1. **Cost Reduction**: Comparing API costs between RAG and CAG approaches
2. **Local Deployment**: CPU-optimized implementation using Ollama
3. **Multi-Model Analysis**: Implementation across different models and platforms

### Implemented Models
- **Google Gemini**: For cloud-based API implementation
- **Llama 3.2 3B (via Ollama)**: For local CPU-optimized deployment

## Key Features

- **CAG Implementation**:
  - Efficient caching mechanism
  - Reduced API calls
  - Cost-effective knowledge retrieval

- **Local Optimization**:
  - CPU-optimized Ollama integration
  - Efficient batch processing
  - Thread optimization
  - Memory-efficient operations

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Install Ollama** (for local deployment):
```bash
# For Windows, download from https://ollama.ai/download
# For Linux:
curl https://ollama.ai/install.sh | sh
```

3. **Setup Environment**:
```bash
cp .env.template .env
# Add your API keys if using cloud services (Gemini)
```

4. **Pull Llama Model**:
```bash
ollama pull llama3.2
```

## Usage

### CAG with Ollama
```bash
python kvcache_ollama.py --dataset hotpotqa-train \
                        --similarity bertscore \
                        --maxKnowledge 27 \
                        --maxQuestion 27 \
                        --model "llama3.2" \
                        --output "./results/cag_ollama_results.txt"
```

### RAG with Gemini
```bash
python rag_gemini.py --dataset hotpotqa-train \
                     --similarity bertscore \
                     --maxKnowledge 120 \
                     --maxQuestion 120 \
                     --output "./results/rag_gemini_results.txt"
```

## Results

Results from different implementations can be found in:
- CAG Results: `CAG/New_Results/CAG/`
- RAG Results: `CAG/New_Results/RAG/`

## Project Structure
```
CAG/
├── kvcache_gemini.py    # CAG implementation with Gemini
├── kvcache_ollama.py    # CAG implementation with Ollama
├── rag_gemini.py        # RAG implementation with Gemini
├── rag_groq.py         # RAG implementation with Groq
├── rag_ollama.py       # RAG implementation with Ollama
└── New_Results/        # Performance results and comparisons
```

## Acknowledgments

This implementation builds upon the work presented in:
```bibtex
@misc{chan2024dontragcacheaugmentedgeneration,
      title={Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks}, 
      author={Brian J Chan and Chao-Ting Chen and Jui-Hung Cheng and Hen-Hsen Huang},
      year={2024},
      eprint={2412.15605},
      archivePrefix={arXiv}
}
```

## License

This project is licensed under the MIT License.


