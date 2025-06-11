# Graph of Attacks with Pruning (GAP) - EasyJailbreak Extension

## Overview

This repository contains the implementation of GAP (Graph of Attacks with Pruning), a novel extension of the Tree of Attacks with Pruning (TAP) method for automated jailbreaking of Large Language Models (LLMs). GAP is integrated into the EasyJailbreak framework, providing enhanced capabilities for LLM security research.

**Paper:** [Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation](https://arxiv.org/abs/2501.18638) (arXiv:2501.18638)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/dsbuddy/GAP-LLM-Safety.git
cd GAP-LLM-Safety

# Create and activate a virtual environment (recommended)
conda create -n gap-env python=3.9
conda activate gap-env

# Install dependencies
pip install -e .

# Set environment variables
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OPENAI_KEY="your-key-here"  # Only if using OpenAI models
```

## Key Features

### 1. Core Innovations vs TAP (Tree of Attacks with Pruning [TAP](https://arxiv.org/abs/2312.02119))

- **Global Conversation History**
  - Maintains comprehensive attack pattern history across all branches, so graph of thought is built
  - Enables learning from past successful attempts
  - Doubles message history with global conversation patterns (`2*keep_last_n`)

- **Enhanced Pruning**
  - Sophisticated pruning based on historical success patterns
  - DeleteOffTopic constraint for irrelevant attempts
  - Pattern-based filtering of conversation history

- **Pattern-based Refinement**
  - Uses successful patterns to guide future attacks
  - Advanced mutation with `IntrospectGenerationAggregateMaxConcatenate`
  - Aggregate scoring across the entire attack graph

### 2. Usage Examples

#### Standard GAP Attack API (Recommended)

```python
import easyjailbreak
import gap_easyjailbreak
from gap_easyjailbreak.attacker.GAP_Schwartz_2024 import GAP
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.datasets.jailbreak_datasets import JailbreakDataset
from easyjailbreak.models.openai_model import OpenaiModel

# Initialize models
attack_model = from_pretrained(model_name_or_path='lmsys/vicuna-13b-v1.5', model_name='vicuna_v13b')
target_model = from_pretrained(model_name_or_path='meta-llama/Llama-2-7b-chat-hf', model_name='llama-2')
eval_model = OpenaiModel(model_name='gpt-4', api_keys='your-key-here')

# Create dataset
dataset = JailbreakDataset('AdvBench')

# Initialize and run GAP
attacker = GAP(
    attack_model=attack_model,
    target_model=target_model,
    eval_model=eval_model,
    jailbreak_datasets=dataset,
    tree_width=10,
    tree_depth=10,
    selection_strategy='max_score'
)
attacker.attack(save_path='gap_results.jsonl')
```

#### Cold Start Approach via API usage

```python
# Using cold start dataset
dataset = JailbreakDataset(
    local_file_type='json',
    dataset='../easyjailbreak/datasets/data/cold_start_generated_seeds_target_response_all_shuffled_ordered.json'
)

# Initialize GAP with cold start configuration      
attacker = GAP(
    attack_model=attack_model,
    target_model=target_model,
    eval_model=eval_model,
    jailbreak_datasets=dataset,
    tree_width=10,
    tree_depth=10,
    selection_strategy='baseline'
)
```

### Driver Scripts Usage Way Overview 

- Under the folder `scripts`, there are three driver scripts for running LLM jailbreak attacks using different approaches and configurations.

- `run_GAP_coldstart.py` - Novel cold-start approach using GAP with custom generated seeds
- `run_GAP.py` - Standard GAP implementation using benchmark datasets
- `run_TAP_debug.py` - Enhanced TAP implementation with detailed logging and metrics

- The detailed guide of these driver scripts is detailed in the `scripts/README_driver.md` file.


## Repository Structure

```
easyjailbreak-root-github-package/
├── easyjailbreak/
│   ├── attacker/
│   │   ├── GAP_Schwartz_2024.py          # Main GAP implementation
│   │   ├── TAP_Mehrotra_2023.py          # Original TAP implementation
│   │   ├── TAP_Mehrotra_2023_debug.py    # Debug version with extra logging
│   │   └── README_model.md               # Model documentation
│   └── mutation/
│       └── generation/
│           ├── IntrospectGeneration.py    # Original TAP mutation
│           ├── IntrospectGenerationAggregateMaxConcatenate.py  # GAP mutation
│           └── README_mutation.md         # Mutation documentation
├── scripts/
│   ├── cold_start/                       # Cold start implementation
│   │   ├── HarmOnTopic.py
│   │   ├── cold_start_generate_seeds.py
│   │   ├── cold_start_generate_seeds_benign.py
│   │   ├── cold_start_generate_target_responses.py
│   │   ├── cold_start_measure_diversity.py
│   │   ├── evaluate_diversity_prompts.py
│   │   ├── format_cold_start_dataset.py
│   │   └── README_coldstart.md
│   ├── fine_tune/
│   │   ├── fine_tune_prompt_guard_experiment_paper.py
│   │   └── README_finetune.md
│   ├── mitigation/
│   │   ├── analyze_prompt_guard_scores.py
│   │   ├── run_llamaguard_experiment.py
│   │   ├── run_perplexity_experiment.py
│   │   ├── run_prompt_guard_experiment.py
│   │   └── README_mitigation.md
│   ├── run_GAP.py                        # Standard GAP runner
│   ├── run_GAP_coldstart.py             # Cold start GAP runner
│   ├── run_TAP_debug.py                 # Debug TAP runner
│   └── README_driver.md                 # Runner documentation
└── README.md                            # Main framework documentation
```

## Technical Details

### 1. Core Components

#### GAP Implementation

- gap_easyjailbreak/attacker/GAP_Schwartz_2024.py includes the Main GAP implementation

```python
class GAP:
    def __init__(self,
                 attack_model,
                 target_model,
                 eval_model,
                 jailbreak_datasets,
                 tree_width=10,
                 tree_depth=10,
                 root_num=1,
                 branching_factor=4,
                 keep_last_n=3,
                 selection_strategy='max_score')
```

Key parameters:
- `tree_width`: Maximum width of conversation nodes
- `tree_depth`: Maximum iteration depth
- `root_num`: Number of parallel attack graphs
- `branching_factor`: Children nodes per parent
- `keep_last_n`: Conversation history length
- `selection_strategy`: Strategy for selecting conversation patterns

#### Mutation Strategies

```python
class IntrospectGenerationAggregateMaxConcatenate(MutationBase):
    def _get_mutated_instance(self, instance, conversation_history, *args, **kwargs):
        # Sample high-performing conversations
        sampled_conversations = self.sample_max_score(conversation_history)
        
        # Combine with current context
        conv.messages = sampled_conversations + conv.messages[-self.keep_last_n*2:]
```

### 2. System Requirements

- Python 3.9+
- CUDA-capable GPU (recommended)
- OpenAI API key (for certain models)

### 3. Required Dependencies

```
torch
transformers
pandas
numpy
seaborn
matplotlib
scipy
boto3 (for LlamaGuard)
sagemaker (for LlamaGuard)
easyjailbreak
```

## Additional Features

### 1. Cold Start Generation

Generate initial attack datasets without relying on existing jailbreak examples:

#### Harmful Seed Generation
```bash
python scripts/cold_start/cold_start_generate_seeds.py \
    --model "mistral.mistral-large-2407-v1:0" \
    --eval_model "anthropic.claude-3-sonnet-20240229-v1:0" \
    --min_behaviors 100 \
    --max_tries 5
```

#### Target Response Generation
```bash
python scripts/cold_start/cold_start_generate_target_responses.py \
    --model "mistral.mistral-large-2407-v1:0" \
    --max_tries 5
```

#### Benign Alternative Generation
```bash
python scripts/cold_start/cold_start_generate_seeds_benign.py \
    --model "mistral.mistral-large-2407-v1:0" \
    --eval_model "anthropic.claude-3-sonnet-20240229-v1:0"
```

#### Diversity Analysis
```bash
python scripts/cold_start/cold_start_measure_diversity.py
```

Features:
- TF-IDF based diversity scoring
- Bootstrap validation
- Comparative analysis between original and generated datasets
- Statistical significance testing

### 2. Fine-tuning

Adapt the Prompt-Guard-86M model for enhanced jailbreak detection:

```bash
python scripts/fine_tune/fine_tune_prompt_guard_experiment_paper.py \
    --learning_rate 4e-6 \
    --batch_size 32 \
    --epochs 1 \
    --max_length 512 \
    --temperature 3.0
```

Features:
- **Base Model**: meta-llama/Prompt-Guard-86M
- **Binary Classification**: Modified output layer
- **Comprehensive Evaluation**: ROC curves, score distributions
- **External Validation**: LMSYS Toxic Chat, OpenAI Moderation datasets

Training Configuration:
```python
TRAINING_CONFIG = {
    'learning_rate': 4e-6,
    'batch_size': 32,
    'epochs': 1,
    'max_length': 512,
    'optimizer': 'AdamW',
    'temperature': 3.0,
    'classification_threshold': 0.5
}
```

### 3. Mitigation Strategies

Three sophisticated defense mechanisms:

#### Prompt Guard Defense
```bash
python scripts/mitigation/run_prompt_guard_experiment.py \
    --model_path "prompt_guard_finetuned_5e7_2.pth" \
    --batch_size 32 \
    --temperature 3.0
```

#### Perplexity-based Detection
```bash
python scripts/mitigation/run_perplexity_experiment.py \
    --threshold -1.0 \
    --window_size 10 \
    --model "meta-llama/Llama-2-7b-chat-hf"
```

#### LlamaGuard Integration
```bash
# LlamaGuard v1 (API)
python scripts/mitigation/run_llamaguard_experiment.py \
    --endpoint_name "llamaguard-endpoint" \
    --profile_name "your-profile"

# LlamaGuard v2 (Local)
python scripts/mitigation/run_llamaguard2_experiment.py \
    --model "meta-llama/Meta-Llama-Guard-2-8B"
```

## Analysis Tools

The framework includes comprehensive analysis tools:

### Defense Effectiveness Analysis
- **Prompt Guard Analysis**: `scripts/mitigation/analyze_prompt_guard_scores.py`
  - Reads Prompt Guard CSV results
  - Computes safe/unsafe classifications
  - Provides detailed scores

- **Perplexity Analysis**: `scripts/mitigation/perplexity_get_counts.py`
  - Reads perplexity-based defense CSV
  - Performs statistical tests:
    - Kolmogorov-Smirnov
    - Anderson-Darling
    - Mann-Whitney U
  - Generates visualizations:
    - Violin plots
    - KDE curves

### Evaluation Metrics

All mitigation strategies provide comprehensive evaluation:

```python
# Sample output format
{
    "method": "GAP Learned Jailbreaks",
    "total_samples": 100,
    "prompt_guard_score": 0.82,  # Higher is more malicious
    "perplexity_score": 4.3,     # Higher indicates more unusual patterns
    "llamaguard_safe_ratio": 0.15 # Lower indicates more unsafe content
}
```

## Contributing

We welcome contributions to improve GAP and extend its capabilities. Please follow the existing code structure and include appropriate tests and documentation.

## License

This project is licensed under the same terms as the EasyJailbreak framework. Please refer to the main repository for license details.

## Citation

If you use GAP in your research, please cite our paper:

```bibtex
@article{schwartz2025graph,
  title={Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation},
  author={Schwartz, Daniel and Bespalov, Dmitriy and Wang, Zhe and Kulkarni, Ninad and Qi, Yanjun},
  journal={arXiv preprint arXiv:2501.18638},
  year={2025}
}
```
