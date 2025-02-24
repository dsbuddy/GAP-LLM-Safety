# Attack Driver Scripts

This repository contains three driver scripts for running LLM jailbreak attacks using different approaches and configurations.

## Overview

1. `run_GAP_coldstart.py` - Novel cold-start approach using GAP with custom generated seeds
2. `run_GAP.py` - Standard GAP implementation using benchmark datasets
3. `run_TAP_debug.py` - Enhanced TAP implementation with detailed logging and metrics

## Scripts

### GAP Cold Start (`run_GAP_coldstart.py`)
Implements our novel approach for running GAP without requiring benchmark datasets:

```python
dataset = JailbreakDataset(
    local_file_type='json', 
    dataset='../easyjailbreak/datasets/data/cold_start_generated_seeds_target_response_all_shuffled_ordered.json'
)
```

Key features:
- Uses custom generated seed dataset
- Supports Mistral, Bedrock, and other LLM backends
- Configurable attack parameters
- No dependency on existing benchmark datasets

### Standard GAP (`run_GAP.py`)
Standard implementation of GAP using benchmark datasets:

```python
dataset = JailbreakDataset(
    local_file_type='json', 
    dataset='../easyjailbreak/datasets/data/advbench_subset.json'
)
```

Key features:
- Uses established benchmark datasets (e.g., AdvBench)
- Supports multiple model backends
- Configurable tree width and depth
- Graph-based attack pattern learning

### Enhanced TAP Debug (`run_TAP_debug.py`)
Extended version of TAP with additional logging and metrics:

```python
attacker = TAP(
    ...
    logging_filename=dataset_name
)
```

Key features:
- Detailed metrics collection
- Performance statistics
- Call counting for model interactions
- Enhanced debugging capabilities

## Usage

### Environment Setup
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OPENAI_KEY="your-key-here"
```

### Running Attacks

Cold Start GAP:
```bash
python run_GAP_coldstart.py
```

Standard GAP:
```bash
python run_GAP.py
```

Debug TAP:
```bash
python run_TAP_debug.py
```

## Configuration Options

Common parameters across all scripts:
```python
attacker = GAP/TAP(
    tree_width=10,          # Width of attack tree/graph
    tree_depth=10,          # Maximum depth of attacks
    root_num=1,            # Number of parallel trees/graphs
    branching_factor=4,    # Children per node
    keep_last_n=3,        # Conversation history length
    max_n_attack_attempts=5 # Max attempts per branch
)
```

## Model Support

- Attack Models:
  - Vicuna
  - Mistral
  - GPT-4
  - Custom HuggingFace models

- Target Models:
  - GPT-3.5
  - Llama 2/3
  - Custom HuggingFace models

- Evaluation Models:
  - GPT-4
  - Custom evaluators

## Output

All scripts generate:
- JSON results file
- Detailed logs
- Performance metrics
- Attack success statistics

## File Naming Convention

Results are saved using the pattern:
```python
"{MODEL_CONFIG}_result.jsonl"
```

Example: `GAP_Atk_Vic_Evl_GPT4_Tgt_GPT35_D_ColdStart_result.jsonl`

## Requirements

- Python 3.7+
- CUDA-capable GPU(s)
- OpenAI API key
- Required packages in `requirements.txt`