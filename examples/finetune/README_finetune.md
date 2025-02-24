# Prompt Guard Model Fine-Tuning

This repository contains code for fine-tuning the meta-llama/Prompt-Guard-86M model for detecting malicious prompts, jailbreak attempts, and prompt injections.

## Experimental Setup

### Model Architecture
- Base Model: meta-llama/Prompt-Guard-86M
- Output Layer: Modified to binary classification (2 output classes)
- Device: Automatically uses GPU (CUDA) if available, falls back to CPU

### Hyperparameters

#### Training Parameters
- Learning Rate: 4e-6
- Batch Size: 32
- Number of Epochs: 1
- Optimizer: AdamW
- Max Sequence Length: 512 tokens

#### Evaluation Parameters
- Temperature: 3.0 (for softmax scaling)
- Classification Threshold: 0.5
- Evaluation Batch Size: 32

## Datasets

### Primary Dataset
- File: `cold_start_dataset_finetune.csv`
- Format: CSV with columns:
  - query
  - prompt
  - category
  - behavior
  - harmful (binary label)
- Split Ratios:
  - Training: 72%
  - Validation: 8%
  - Test: 20%

### External Evaluation Datasets
1. LMSYS Toxic Chat Dataset
   - Source: `datasets/lmsys/toxic-chat`
   - Label: jailbreaking (binary)

2. OpenAI Moderation API Evaluation Dataset
   - Source: `mmathys/openai-moderation-api-evaluation`
   - Labels: Combined toxicity indicators (S, H, V, HR, SH, S3, H2, V2)

## Setup Instructions

1. Environment Setup
```bash
# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Unix
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install transformers torch pandas numpy matplotlib seaborn sklearn tqdm datasets
```

2. Data Preparation
```bash
# Place the primary dataset in the project root
cp path/to/cold_start_dataset_finetune.csv ./
```

3. Running the Fine-tuning Pipeline
```bash
python fine_tune_prompt_guard_experiment_paper.py
```

## Pipeline Stages

1. **Initial Setup**
   - GPU/CPU device detection
   - Model and tokenizer loading
   - Dataset loading and preprocessing

2. **Pre-training Evaluation**
   - Evaluates base model on all datasets
   - Generates initial ROC curves and score distributions
   - Records baseline metrics

3. **Model Fine-tuning**
   - Modifies classifier layer for binary classification
   - Trains on the primary dataset
   - Saves model checkpoints

4. **Post-training Evaluation**
   - Evaluates fine-tuned model on all datasets
   - Generates comparative ROC curves and score distributions
   - Records final metrics

## Output Files

### Model Artifacts
- `prompt_guard_finetuned_4e6.pth`: Fine-tuned model weights

### Evaluation Plots
- `fig_test_initial_roc_curve.png`: Initial ROC curve
- `fig_test_initial_score_distribution.png`: Initial score distribution
- `fig_test_trained_roc_curve.png`: Post-training ROC curve
- `fig_test_trained_score_distribution.png`: Post-training score distribution

## Evaluation Metrics

The following metrics are computed for each dataset:
- Accuracy
- F1 Score
- Precision
- Recall
- True Positive Rate (TPR)
- False Positive Rate (FPR)
- ROC AUC Score

## Model Usage

### Inference Functions

1. Jailbreak Detection:
```python
score = get_jailbreak_score(text, temperature=3.0)
```

2. Indirect Injection Detection:
```python
score = get_indirect_injection_score(text, temperature=3.0)
```

### Temperature Scaling
- Default temperature of 3.0 is used for prediction calibration
- Higher temperatures produce softer probability distributions
- Lower temperatures make predictions more decisive

## Performance Monitoring

The script automatically logs:
- Training loss per epoch
- Evaluation metrics for all datasets
- Comparative performance between base and fine-tuned models
- Score distributions and ROC curves

## Notes

- The model automatically handles GPU acceleration when available
- All evaluation metrics are computed with temperature scaling
- The training process includes automatic mixed precision for efficiency
- Model checkpoints are saved after training for future use