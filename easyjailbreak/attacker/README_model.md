# Graph of Attacks with Pruning (GAP)

This repository contains the implementation of GAP (Graph of Attacks with Pruning), a novel extension of the Tree of Attacks with Pruning (TAP) method for automated jailbreaking of Large Language Models (LLMs).

## Overview

GAP builds upon the TAP framework by introducing a graph-based approach that enables more effective exploration of the attack space through:

1. Maintaining a global conversation history of successful attack patterns
2. Using aggregate scoring across multiple conversation paths
3. Enhanced prompt refinement through pattern recognition
4. More sophisticated pruning mechanisms

The implementation includes three key files:

- `GAP_Schwartz_2024.py` - The main GAP implementation
- `TAP_Mehrotra_2023.py` - The original TAP implementation 
- `TAP_Mehrotra_2023_debug.py` - Debug version of TAP with additional logging

## Key Innovations

GAP introduces several improvements over TAP:

- **Graph-based Memory**: Instead of treating each attack attempt as an isolated tree, GAP maintains a graph of successful attack patterns and their relationships
- **Conversation History**: Tracks effective conversation patterns across multiple attempts
- **Aggregate Scoring**: Uses pattern recognition across the attack graph to identify promising directions
- **Enhanced Pruning**: More sophisticated pruning mechanisms that consider historical success patterns

## Implementation Details

### Core Components

Both TAP and GAP share four major components:

1. Seed Template Generation
2. Mutation/Generation Strategy  
3. Constraint Checking
4. Evaluation & Selection

GAP enhances these with:

- `IntrospectGenerationAggregateMaxConcatenate` - Advanced mutation strategy
- Global conversation history tracking
- Pattern-based refinement
- Enhanced scoring mechanisms

### Key Parameters

- `tree_width`: Maximum width of conversation nodes
- `tree_depth`: Maximum iteration depth
- `root_num`: Number of parallel attack trees/graphs
- `branching_factor`: Children nodes per parent
- `keep_last_n`: Conversation history length

## Usage

```python
from easyjailbreak.attacker.GAP_Schwartz_2024 import GAP
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.datasets.jailbreak_datasets import JailbreakDataset

# Initialize models
attack_model = from_pretrained(model_path_1)
target_model = from_pretrained(model_path_2) 
eval_model = from_pretrained(model_path_3)

# Create dataset
dataset = JailbreakDataset('AdvBench')

# Initialize attacker
attacker = GAP(
    attack_model=attack_model,
    target_model=target_model, 
    eval_model=eval_model,
    jailbreak_datasets=dataset,
    tree_width=10,
    tree_depth=10
)

# Run attack
attacker.attack()

# Save results
attacker.jailbreak_Dataset.save_to_jsonl("./GAP_results.jsonl")
```

## Evaluation Metrics

The implementation tracks:

- Attack Success Rate (ASR)
- Number of queries required
- Model call counts
- Rejection rates
- Conversation pattern effectiveness
