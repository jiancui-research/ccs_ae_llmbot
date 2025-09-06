# Table 6 README

## Environment Setup
We recommend using **conda**, then installing Python dependencies via the requirements file.

```bash
# Create & activate environment (Python 3.9+ recommended)
conda create -n myenv python=3.10 -y
conda activate myenv

# Install dependencies (choose the file name you actually have)
pip install -r requirements.txt
```

## Hardware Requirements
- **GPU for GPT variants:** ≥ **8 GB VRAM**
- **GPU for LLaMA & Gemma variants:** ≥ **40 GB VRAM**
- **Disk space:** ≥ **200 GB** free (models, caches, intermediate artifacts)

## Running
For faster processing, please run scripts on GPUs. A full run typically takes about one week under the recommended hardware configuration.

- **Full run (GPT + LLaMA + Gemma):** requires a 40 GB GPU for the LLaMA/Gemma stages.
- **GPT-only validation:** if you do not have a ≥40 GB GPU, you can run GPT-only on an ≥8 GB GPU; expected runtime is ~12 hours.



### For GPT2-XL
```bash
python beam_search.py ./dataset/dataset_2019.json gpt2-xl ./result/result_gpt.json
```

### For Gemma-2-9B
```bash
python beam_search.py ./dataset/dataset_2023.json google/gemma-2-9b YOUR_HF_TOKEN ./result/result_gemma.json
```

### For Llama-3-8B
```bash
python beam_search.py ./dataset/dataset_2023.json meta-llama/Meta-Llama-3-8B YOUR_HF_TOKEN ./result/result_llama.json
```

Results will be saved to the `result` directory.

