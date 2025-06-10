# Graph of Attacks with Pruning (GAP) - EasyJailbreak Extension

## Overview

This repository contains the implementation of GAP (Graph of Attacks with Pruning), a novel extension of the Tree of Attacks with Pruning (TAP) method for automated jailbreaking of Large Language Models (LLMs). GAP is integrated into the EasyJailbreak framework, providing enhanced capabilities for LLM security research.

**Paper:** [Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation](https://arxiv.org/abs/2501.18638) (arXiv:2501.18638)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/EasyJailbreak/EasyJailbreak.git

# Create and activate a virtual environment (recommended)
conda create -n gap-env python=3.9
conda activate gap-env

# Install dependencies
pip install -e .

# Set environment variables (only if using GPU)
export OPENAI_KEY="your-key-here"
```

## Key Features

### 1. Core Innovations vs TAP

- **Global Conversation History**
  - Maintains comprehensive attack pattern history
  - Enables learning from past successful attempts
  - Doubles message history with global conversation patterns

- **Enhanced Pruning**
  - Sophisticated pruning based on historical success
  - DeleteOffTopic constraint for irrelevant attempts
  - Pattern-based filtering of conversation history

- **Pattern-based Refinement**
  - Uses successful patterns to guide future attacks
  - Advanced mutation with `IntrospectGenerationAggregateMaxConcatenate`

### 2. Usage Examples

#### Standard GAP Attack

```python
from easyjailbreak.attacker.GAP_Schwartz_2024 import GAP
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.datasets.jailbreak_datasets import JailbreakDataset

# Initialize models
attack_model = from_pretrained('lmsys/vicuna-13b-v1.5')
target_model = from_pretrained('meta-llama/Llama-2-7b-chat-hf')
eval_model = from_pretrained('gpt-4')

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
attacker.attack()
attacker.save_results('gap_results.jsonl')
```

#### Cold Start Approach

```python
dataset = JailbreakDataset(
    local_file_type='json',
    # ... additional parameters
)
```

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
├── examples/
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
└── README-easyjailbreak.md              # Main framework documentation
```

## Technical Details

### 1. Core Components

#### GAP Implementation

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

#### Mutation Strategies

```python
class IntrospectGenerationAggregateMaxConcatenate(MutationBase):
    def _get_mutated_instance(self, instance, conversation_history, *args, **kwargs):
        sampled_conversations = self.sample_max_score(conversation_history)
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

## Analysis Tools

The framework includes tools for analyzing defense effectiveness:

- **Prompt Guard Analysis**: `examples/mitigation/analyze_prompt_guard_scores.py`
  - Reads Prompt Guard CSV results
  - Computes safe/unsafe classifications
  - Provides detailed scores

- **Perplexity Analysis**: `examples/mitigation/perplexity_get_counts.py`
  - Reads perplexity-based defense CSV
  - Performs statistical tests:
    - Kolmogorov-Smirnov
    - Anderson-Darling
    - Mann-Whitney U
  - Generates visualizations:
    - Violin plots
    - KDE curves

## Contributing

[To be added]

## License

[To be added]
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

## Additional Features


### 1. Cold Start Generation

The cold start module provides a sophisticated pipeline for generating and evaluating initial attack datasets without relying on existing jailbreak examples:

#### 1.1 Harmful Seed Generation
```python
python examples/cold_start/cold_start_generate_seeds.py \
    --model "mistral.mistral-large-2407-v1:0" \
    --eval_model "anthropic.claude-3-sonnet-20240229-v1:0" \
    --min_behaviors 100 \
    --max_tries 5
```

**Components:**
1. **Category Extraction**
   - Sources categories from JailbreakBench dataset
   - Uses HuggingFace datasets API
   - Ensures comprehensive coverage of attack types

2. **Behavior Generation**
```python
def generate_behaviors_from_category(category, attack_model, attacker_system_message, max_tries=5, min_behaviors=50):
    """
    Generates fine-grained behaviors for each category using:
    - Pattern-based generation
    - Category-specific constraints
    - Diversity enforcement
    """
```

3. **Goal Generation**
```python
def generate_goal_from_behavior(category, behavior, attack_model, attacker_system_message, max_tries=5):
    """
    Creates specific attack goals with:
    - Category alignment
    - Behavior specificity
    - Harmfulness validation
    """
```

4. **Evaluation Pipeline**
```python
def extract_goal(parsed_goal, finegrained_behavior, evaluator):
    """
    Evaluates generated content for:
    - Harmfulness (YES/NO)
    - Topic relevance (YES/NO)
    - Category alignment
    """
```

#### 1.2 Target Response Generation
```python
python examples/cold_start/cold_start_generate_target_responses.py \
    --model "mistral.mistral-large-2407-v1:0" \
    --max_tries 5
```

**Features:**
1. **Response Generation**
```python
def generate_target_response(goal, attack_model, attacker_system_message, max_tries=5):
    """
    Generates target responses with:
    - Affirmative framing
    - Concise format
    - Goal alignment
    - Consistent style
    """
```

2. **Output Format**
```python
{
    "category": "category_name",
    "behavior": "specific_behavior",
    "query": "generated_goal",
    "reference_responses": ["target_response"]
}
```

3. **Category-specific Files**
- Generates separate files for each category
- Enables focused evaluation and testing
- Maintains organizational clarity

#### 1.3 Benign Alternative Generation
```python
python examples/cold_start/cold_start_generate_seeds_benign.py \
    --model "mistral.mistral-large-2407-v1:0" \
    --eval_model "anthropic.claude-3-sonnet-20240229-v1:0"
```

**Process:**
1. **Benign Transformation**
```python
def generate_benign_goal_from_behavior(category, behavior, attack_model, max_tries=5):
    """
    Creates non-harmful alternatives with:
    - Same topic coverage
    - Ethical framing
    - Educational focus
    """
```

2. **Validation**
- Uses HarmOnTopic evaluator
- Ensures topic relevance
- Verifies non-harmful nature

#### 1.4 Diversity Analysis
```python
python examples/cold_start/cold_start_measure_diversity.py
```

**Metrics:**
1. **TF-IDF Based Diversity**
```python
def calculate_diversity(behaviors):
    """
    Computes diversity scores using:
    - TF-IDF vectorization
    - Cosine distance
    - Pairwise comparisons
    """
```

2. **Comparative Analysis**
```python
def compare_diversity(original_behaviors, generated_behaviors, category):
    """
    Analyzes diversity through:
    - Original vs. Generated comparison
    - Category-specific metrics
    - Statistical significance tests
    """
```

3. **Bootstrap Validation**
```python
def bootstrap_diversity(original_behaviors, generated_behaviors, num_resamples=10000):
    """
    Validates diversity metrics via:
    - Bootstrap resampling
    - Confidence intervals
    - Robustness checks
    """
```

#### 1.5 Dataset Formatting
```python
python examples/cold_start/format_cold_start_dataset.py
```

**Output Formats:**
1. **JSON Structure**
```python
{
    "query": "attack_goal",
    "prompt": "jailbreak_prompt",
    "category": "attack_category",
    "behavior": "specific_behavior",
    "harmful": binary_label
}
```

2. **CSV Format**
- Standardized columns
- Ready for model training
- Includes metadata

#### Usage Example

```python
# Complete cold start pipeline
def run_cold_start_pipeline():
    # 1. Generate harmful seeds
    generate_harmful_seeds()
    
    # 2. Create target responses
    generate_target_responses()
    
    # 3. Generate benign alternatives
    generate_benign_alternatives()
    
    # 4. Measure diversity
    analyze_diversity()
    
    # 5. Format final dataset
    format_dataset()
```

The cold start module provides a comprehensive solution for generating high-quality, diverse datasets for jailbreak research without relying on existing examples, while maintaining careful ethical considerations and validation throughout the process.



### 2. Fine-tuning

The fine-tuning module provides sophisticated model adaptation of the Prompt-Guard-86M model for enhanced jailbreak detection:

```python
python examples/fine_tune/fine_tune_prompt_guard_experiment_paper.py \
    --learning_rate 4e-6 \
    --batch_size 32 \
    --epochs 1 \
    --max_length 512 \
    --temperature 3.0
```

#### Model Architecture
- **Base Model**: meta-llama/Prompt-Guard-86M
- **Output Layer**: Modified for binary classification
  ```python
  model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
  model.num_labels = 2
  ```
- **Device Handling**: Automatic GPU/CPU detection and optimization

#### Training Configuration

1. **Hyperparameters**
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

2. **Dataset Processing**
- Primary Dataset: `cold_start_dataset_finetune.csv`
  - Features: query, prompt, category, behavior
  - Label: harmful (binary)
- Split Ratios:
  ```python
  train_split = 0.72  # Training set
  val_split = 0.08    # Validation set
  test_split = 0.20   # Test set
  ```

3. **External Validation**
- LMSYS Toxic Chat Dataset
- OpenAI Moderation API Evaluation Dataset
- Metrics computed across all datasets for robustness

#### Evaluation Metrics

The fine-tuning process tracks comprehensive metrics:

```python
def evaluate_metrics(labels, scores, threshold=0.5):
    """
    Computes classification metrics:
    - Accuracy
    - F1 Score
    - Precision
    - Recall
    - True Positive Rate (TPR)
    - False Positive Rate (FPR)
    - ROC AUC Score
    """
```

#### Visualization Tools

1. **ROC Curve Analysis**
```python
def plot_roc_curve(labels, scores, filename):
    """
    Generates ROC curves with:
    - True Positive Rate vs False Positive Rate
    - AUC Score calculation
    - Confidence intervals
    """
```

2. **Score Distribution**
```python
def plot_score_distribution(positive_scores, negative_scores, filename):
    """
    Creates distribution plots showing:
    - Positive/negative class separation
    - Decision boundary analysis
    - Probability calibration
    """
```

#### Performance Monitoring

1. **Training Metrics**
- Loss tracking per epoch
- Learning rate adjustment
- Gradient statistics
- Memory usage

2. **Validation Checks**
```python
def evaluate_and_report(model, dataset, dataset_name, temperature=3.0):
    """
    Comprehensive evaluation including:
    - Classification metrics
    - Error analysis
    - Performance comparison
    """
```

#### Model Artifacts

1. **Checkpoints**
- Best model weights saved
- Training state preservation
- Configuration logging

2. **Output Files**
```python
OUTPUTS = {
    'model_weights': 'prompt_guard_finetuned_4e6.pth',
    'roc_curves': {
        'initial': 'fig_test_initial_roc_curve.png',
        'final': 'fig_test_trained_roc_curve.png'
    },
    'distributions': {
        'initial': 'fig_test_initial_score_distribution.png',
        'final': 'fig_test_trained_score_distribution.png'
    }
}
```

#### Usage Example

```python
# Load and fine-tune model
model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Prompt-Guard-86M")
model.classifier = torch.nn.Linear(model.classifier.in_features, 2)

# Train model
train_model(
    train_dataset=train_dataset,
    model=model,
    tokenizer=tokenizer,
    batch_size=32,
    epochs=1,
    lr=4e-6
)

# Save fine-tuned model
torch.save(model.state_dict(), "prompt_guard_finetuned_4e6.pth")

# Evaluate on test set
evaluate_and_report(
    model=model,
    dataset=test_dataset,
    dataset_name="Test Dataset",
    temperature=3.0
)
```

The fine-tuning module provides a complete pipeline for adapting the Prompt-Guard model to specific jailbreak detection tasks, with comprehensive monitoring, evaluation, and visualization tools for tracking performance improvements.



### 3. Mitigation Strategies

The mitigation module implements three sophisticated defense mechanisms to detect and prevent jailbreak attempts:

1. **Prompt Guard Defense**
```python
python examples/mitigation/run_prompt_guard_experiment.py \
    --model_path "prompt_guard_finetuned_5e7_2.pth" \
    --batch_size 32 \
    --temperature 3.0
```

Features:
- Uses fine-tuned DeBERTa model (meta-llama/Prompt-Guard-86M)
- Implements efficient parallel batched evaluation
- Handles arbitrary length inputs through smart chunking
- Provides two scoring mechanisms:
  - Jailbreak detection score: Evaluates direct attack attempts
  - Indirect injection score: Detects embedded malicious instructions
- Preprocessing includes token-aware space normalization
- Temperature scaling for confidence calibration

Key Parameters:
- `batch_size`: Number of prompts to evaluate in parallel
- `temperature`: Controls prediction confidence (default: 3.0)
- `max_length`: Maximum input length (default: 512 tokens)

2. **Perplexity-based Detection**
```python
python examples/mitigation/run_perplexity_experiment.py \
    --threshold -1.0 \
    --window_size 10 \
    --model "meta-llama/Llama-2-7b-chat-hf"
```

Features:
- Uses Llama-2-7B model for perplexity scoring
- Implements sliding window analysis
- Generates visualization plots:
  - Violin plots showing score distributions
  - KDE curves for pattern analysis
- Statistical analysis including:
  - Kolmogorov-Smirnov tests
  - Anderson-Darling tests
  - Mann-Whitney U tests

Key Parameters:
- `threshold`: Perplexity threshold for classification
- `window_size`: Size of sliding window
- `device_map`: GPU allocation strategy

3. **LlamaGuard Integration**
```python
# For LlamaGuard v1
python examples/mitigation/run_llamaguard_experiment.py \
    --endpoint_name "llamaguard-endpoint" \
    --profile_name "your-profile"

# For LlamaGuard v2
python examples/mitigation/run_llamaguard2_experiment.py \
    --model "meta-llama/Meta-Llama-Guard-2-8B"
```

Features:
- Supports both LlamaGuard v1 (API) and v2 (local) implementations
- Comprehensive safety policy categories:
  - Violence and Hate
  - Sexual Content
  - Criminal Planning
  - Guns and Illegal Weapons
  - Regulated Substances
  - Self-Harm
- Provides detailed safety assessments
- Batch processing capabilities
- AWS SageMaker integration for v1

Configuration:
```python
SAFETY_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.001,
    "top_p": 0.9
}
```

### Evaluation Metrics

All three mitigation strategies provide comprehensive evaluation metrics:

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

### Results Analysis

The framework includes tools for analyzing defense effectiveness:

- **Analyze Prompt Guard results**: `examples/mitigation/analyze_prompt_guard_scores.py`
  - Reads in the CSV file generated by the Prompt Guard defense
  - Computes counts of safe/unsafe classifications
  - Provides detailed scores for each method
- **Analyze perplexity distributions**: `examples/mitigation/perplexity_get_counts.py`
  - Reads in the CSV file generated by the perplexity-based defense
  - Computes counts of perplexity scores above/below a specified threshold
  - Performs statistical significance tests:
    - Kolmogorov-Smirnov test
    - Anderson-Darling test
    - Mann-Whitney U test
  - Generates visualization plots:
    - Violin plots showing score distributions
    - KDE curves for pattern analysis
  - Provides method comparison reports

The mitigation strategies can be used individually or combined for enhanced protection against jailbreak attempts.