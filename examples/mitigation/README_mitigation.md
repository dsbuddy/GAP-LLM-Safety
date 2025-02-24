# Mitigation Experiments

This repository contains scripts for evaluating different mitigation techniques against jailbreak attempts using Prompt Guard, Perplexity, and LlamaGuard.

## Scripts Overview

### 1. run_prompt_guard_experiment.py
- Evaluates text for jailbreaks and indirect injections using the PromptGuard model
- Processes texts in batches and handles arbitrary length inputs
- Generates scores for different jailbreak methods and outputs results to CSV

### 2. run_perplexity_experiment.py 
- Computes perplexity scores using Llama-2 model
- Generates visualizations of perplexity distributions
- Outputs perplexity scores to CSV and creates violin/density plots

### 3. run_llamaguard_experiment.py / run_llamaguard2_experiment.py
- Evaluates safety of prompts using LlamaGuard model
- Processes prompts through safety classification
- Outputs safety scores and classifications to CSV

### 4. perplexity_get_counts.py
- Analyzes perplexity results 
- Performs statistical tests between different methods
- Provides counts and statistical significance of differences

### 5. analyze_prompt_guard_scores.py
- Analyzes the Prompt Guard classification results
- Provides counts of safe/unsafe classifications by method

## Required Data Files

1. Dataset Files:
- `perplexity_dataset_advbench_clean_seeds.txt`: Clean benchmark prompts
- `perplexity_dataset_gcg_paper_suffixes.csv`: GCG attack suffixes
- `perplexity_dataset_gptfuzzer_paper_templates.csv`: GPTFuzzer templates

2. Results Files:
- `results_advbench_gpt35_tap.json`: TAP jailbreak results
- `results_advbench_gpt35_gap.json`: GAP jailbreak results

3. Model Files:
- `prompt_guard_finetuned_5e7_2.pth`: Finetuned Prompt Guard model weights

## Order of Operations

1. First, run the Prompt Guard evaluation:
```bash
python run_prompt_guard_experiment.py
```
This generates `prompt_guard_scores_advbench.csv`

2. Next, run the perplexity evaluation:
```bash
python run_perplexity_experiment.py
```
This generates `perplexity_dataset_results.csv` and visualization plots

3. Run the LlamaGuard evaluation (choose one):
```bash
python run_llamaguard_experiment.py  # For LlamaGuard 1
# or
python run_llamaguard2_experiment.py # For LlamaGuard 2
```
This generates `llama_guard_results.csv`

4. Analyze the perplexity results:
```bash
python perplexity_get_counts.py
```

5. Finally, analyze the Prompt Guard results:
```bash
python analyze_prompt_guard_scores.py
```

## Environment Setup

Required packages:
- torch
- transformers
- pandas
- numpy
- seaborn
- matplotlib
- scipy
- boto3 (for LlamaGuard 1)
- sagemaker (for LlamaGuard 1)

Models required:
- meta-llama/Prompt-Guard-86M
- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Meta-Llama-Guard-2-8B