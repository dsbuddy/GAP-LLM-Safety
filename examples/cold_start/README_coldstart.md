# Prompt Generation and Evaluation Scripts

This repository contains a collection of scripts for generating and evaluating harmful and benign prompts, as well as measuring the diversity of the generated prompts.

## Scripts Overview

### 1. cold_start_generate_seeds.py
- Generates harmful goal prompts for given categories using a large language model
- Uses the JailbreakBench dataset to extract unique categories
- Generates fine-grained behaviors for each category
- Creates harmful goal prompts for each behavior
- Evaluates prompts for harmfulness and relevance
- Outputs: cold_start_generated_seeds.json

### 2. cold_start_generate_target_responses.py
- Generates target responses for harmful goal prompts
- Takes input from cold_start_generated_seeds.json
- Creates brief, affirmative responses for each prompt
- Outputs: cold_start_generated_seeds_target_response_category_*.json (one file per category)

### 3. cold_start_generate_seeds_benign.py
- Generates benign versions of goal prompts
- Takes input from cold_start_generated_seeds.json
- Creates non-harmful alternatives for each prompt
- Evaluates prompts for harmfulness and relevance
- Outputs: cold_start_generated_seeds_benign_category_*.json (one file per category)

### 4. cold_start_measure_diversity.py
- Measures diversity of generated prompts
- Compares diversity to original JailbreakBench dataset
- Uses TF-IDF vectorization and cosine distance
- Takes input from cold_start_generated_seeds.json and JailbreakBench dataset

### 5. evaluate_diversity_prompts.py
- Computes diversity metrics for prompts from different models
- Calculates Self-BLEU, n-gram entropy, Zipf coefficient, and unique n-gram ratio
- Takes input from various results files including:
  - results_advbench_gpt35_tap.json
  - results_advbench_gpt35_gap.json
  - results_advbench_llama2_tap.json
  - results_advbench_llama2_gap.json
  - cold_start_generated_seeds_harmful_all.json

### 6. format_cold_start_dataset.py
- Combines and formats the generated data into a unified dataset
- Takes input from:
  - cold_start_generated_seeds.json
  - cold_start_generated_seeds_harmful_all.json
  - cold_start_generated_seeds_benign_all.json
- Outputs:
  - cold_start_dataset_finetune.json
  - cold_start_dataset_finetune.csv

### 7. HarmOnTopic.py
- Utility class for evaluating prompt harmfulness and relevance
- Used by other scripts for prompt evaluation

## Order of Operations

1. **Initial Setup**
   - Install required dependencies:
     ```bash
     pip install numpy pandas scikit-learn transformers regex datasets easyjailbreak
     ```
   - Ensure access to the JailbreakBench dataset
   - Set up API access for Claude and Mistral models

2. **Generate Harmful Seeds**
   ```bash
   python cold_start_generate_seeds.py
   ```
   - Creates cold_start_generated_seeds.json

3. **Generate Target Responses**
   ```bash
   python cold_start_generate_target_responses.py
   ```
   - Creates category-specific JSON files with target responses

4. **Generate Benign Seeds**
   ```bash
   python cold_start_generate_seeds_benign.py
   ```
   - Creates category-specific JSON files with benign alternatives

5. **Measure Diversity**
   ```bash
   python cold_start_measure_diversity.py
   ```
   - Analyzes diversity of generated prompts

6. **Evaluate Prompt Diversity**
   ```bash
   python evaluate_diversity_prompts.py
   ```
   - Computes detailed diversity metrics

7. **Format Final Dataset**
   ```bash
   python format_cold_start_dataset.py
   ```
   - Creates final formatted dataset files

## Required Data Files

1. **Input Datasets**
   - JailbreakBench dataset (automatically downloaded via HuggingFace datasets)

2. **Generated Files**
   - cold_start_generated_seeds.json
   - cold_start_generated_seeds_target_response_category_*.json
   - cold_start_generated_seeds_benign_category_*.json
   - cold_start_generated_seeds_harmful_all.json
   - cold_start_generated_seeds_benign_all.json

3. **Model Results Files** (for diversity evaluation)
   - results_advbench_gpt35_tap.json
   - results_advbench_gpt35_gap.json
   - results_advbench_llama2_tap.json
   - results_advbench_llama2_gap.json

## Dependencies

- Python 3.x
- NumPy
- Pandas
- scikit-learn
- transformers
- regex
- datasets
- easyjailbreak
- nltk
