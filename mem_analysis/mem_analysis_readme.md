# Memory Analysis - Table 6 Generation Guide

This directory contains scripts for generating memorization analysis results (Table 6 in the paper). 

## 🚀 Quick Start (Recommended)
**Use pre-computed results:** Simply run `result_analysis.ipynb` to analyze the existing results in the `results/` directory.

## 🔧 Advanced: Regenerate Raw Results

### Environment Setup
```bash
# Create & activate environment (Python 3.10 recommended)
conda create -n llmbot_mem_analysis python=3.10 -y
conda activate llmbot_mem_analysis

# Install dependencies from project root
pip install -r requirements.txt
```

### Hardware Requirements

| Model Type | GPU Memory | Runtime |
|------------|------------|----------|
| **GPT variants** | ≥8GB VRAM | ~12 hours |
| **LLaMA & Gemma variants** | ≥40GB VRAM | ~1 week |

Need storage at least 200GB for saving models

### API Configuration (Closed-Source LLMs)
For closed-source LLMs, configure API keys in respective vendor websites:
- OpenAI (gpt-4o-mini)
- Anthropic (claude-3-haiku-20240307)
- Google (gemini-1.5-flash)
- Cohere (command-r-08-2024)

Then run:
```bash
python closedsource_next_sent_gen.py
```

### Running Open-Source Models

#### GPT2-XL (8GB GPU)
```bash
python opensource_next_sent_gen.py ./dataset/dataset_2019.json gpt2-xl ./result/result_gpt.json
```

#### Gemma-2-9B (40GB GPU + HuggingFace Token)
```bash
python opensource_next_sent_gen.py ./dataset/dataset_2023.json google/gemma-2-9b YOUR_HF_TOKEN ./result/result_gemma.json
```

#### Llama-3-8B (40GB GPU + HuggingFace Token)
```bash
python opensource_next_sent_gen.py ./dataset/dataset_2023.json meta-llama/Meta-Llama-3-8B YOUR_HF_TOKEN ./result/result_llama.json
```

### Output
- Results are saved to the `result/` directory
- **Note:** Results may vary slightly due to model updates (closed-source), temperature settings and hardware differences
- Use `result_analysis.ipynb` to analyze the generated results

