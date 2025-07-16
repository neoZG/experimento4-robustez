# Robustness Evaluation of Language Models

This repository contains code for evaluating the robustness of various language models on natural language understanding tasks. The evaluation focuses on two key aspects:

1. Zero-shot performance on ANLI (Adversarial Natural Language Inference)
2. Robustness to character-level noise on SST-2 (sentiment analysis) and XNLI (cross-lingual NLI)

## Models Evaluated

- **BitNet-b1.58** (microsoft/bitnet-b1.58-2B-4T-bf16): A 2B parameter causal LM using binary weights
- **DistilBERT-NLI** (kweinmeister/distilbert-mnli): DistilBERT fine-tuned on MNLI
- **DistilBERT-SST2** (distilbert-base-uncased-finetuned-sst-2-english): DistilBERT fine-tuned on SST-2
- **Mistral-7B** (mistralai/Mistral-7B-v0.1): 7B parameter causal LM
- **Phi-2** (microsoft/phi-2): 2.7B parameter causal LM
- **Gemma-2B** (google/gemma-2b-it): 2B parameter instruction-tuned causal LM

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure experiment parameters in `config.yaml`:
- Model selection and aliases
- Dataset sizes
- Batch size and device settings

## Running the Evaluation

```bash
python eval.py
```

The script will:
1. Load each model and its tokenizer
2. Evaluate on ANLI test sets (R1, R2, R3)
3. Evaluate on clean and noisy versions of SST-2 and XNLI
4. Save results to CSV and JSON files in the `output/` directory

## Results Format

The evaluation produces three output files:

1. `output/anli_results.csv`: ANLI zero-shot performance per round
2. `output/robustness_results.csv`: Clean vs. noisy performance on SST-2 and XNLI
3. `output/results.json`: Complete results in JSON format

## Code Structure

- `eval.py`: Main evaluation script with comprehensive error handling
- `eval_utils.py`: Core evaluation functions for both sequence classification and causal LMs
- `utils.py`: Helper functions for dataset processing, noise generation, and environment validation  
- `config.yaml`: Configuration file for models and experiment parameters
- `output/`: Directory containing evaluation results (CSV and JSON files)

## Metrics

- **Accuracy**: Percentage of correct predictions
- **F1 Score**: Macro-averaged F1 score
- **Accuracy Drop**: Difference in accuracy between clean and noisy inputs (in percentage points)

## Notes

- **Reproducibility**: Random seeds are fixed (42) for consistent results across runs
- **Character-level noise**: Includes random character swaps, deletions, and insertions
- **Zero-shot evaluation**: All evaluations done without fine-tuning
- **Robust prompt parsing**: Uses regex patterns to handle various model response formats
- **Dynamic generation parameters**: Different max_new_tokens for NLI vs sentiment tasks
- **Error handling**: Comprehensive validation for datasets, models, and file operations
- **Memory management**: Automatic cleanup between model evaluations
- **Environment validation**: Checks for required packages and versions

## Improvements Made

✅ **Reproducibility**: Fixed random seeds across all libraries (random, numpy, torch)  
✅ **Code organization**: Modularized evaluation functions into `eval_utils.py`  
✅ **Robust parsing**: Improved answer extraction with regex patterns  
✅ **Dynamic parameters**: Task-specific `max_new_tokens` configuration  
✅ **Error handling**: Comprehensive validation and graceful error recovery  
✅ **Environment checks**: Package version reporting and CUDA detection  
✅ **Memory optimization**: Better cleanup between model evaluations