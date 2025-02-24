# Graph-of-Thought vs Tree-of-Thought Attack Generation

This repository contains implementations of two approaches for generating adversarial prompts in LLM jailbreaking attacks:

1. `IntrospectGenerationAggregateMaxConcatenate.py` - Novel graph-of-thought (GoT) approach
2. `IntrospectGeneration.py` - Original tree-of-thought (ToT) approach

## Overview

Our novel Graph-of-Thought (GoT) mutation strategy builds upon the Tree-of-Thought approach by maintaining a global graph of successful attack patterns and using them to guide future mutations. This allows for more effective exploration of the attack space through pattern recognition and historical success.

## Key Innovations in Graph-of-Thought

The GoT approach (`IntrospectGenerationAggregateMaxConcatenate.py`) introduces several improvements:

- **Global Conversation History**: Maintains a history of successful attack patterns across all branches
- **Score-based Sampling**: Uses `sample_max_score()` to select high-performing conversation patterns
- **Pattern Aggregation**: Combines successful patterns with current conversation context
- **Adaptive Message Filtering**: Dynamically filters and incorporates historical patterns
- **Enhanced JSON Validation**: Additional checks to ensure quality of generated prompts

## Implementation Details

### Core Components

Both implementations share basic functionality:

```python
class IntrospectGeneration(MutationBase):
    def __init__(self, model, system_prompt, branching_factor=5, 
                 keep_last_n=3, max_n_attack_attempts=5,
                 attr_name="jailbreak_prompt", prompt_format=None):
```

Key differences in GoT implementation:

```python
def _get_mutated_instance(self, instance, conversation_history, *args, **kwargs):
    # GoT-specific sampling and filtering
    sampled_conversations = self.sample_max_score(conversation_history, k=num_messages_to_filter)
    
    # Combine historical patterns with current context
    conv.messages = sampled_conversations_messages_flattened + conv.messages[-self.keep_last_n*2:]
```

### Key Features

#### Tree-of-Thought (Original)
- Local conversation history within each branch
- Fixed message retention
- Independent branches
- Basic JSON validation

#### Graph-of-Thought (Novel)
- Global conversation history across branches
- Score-based pattern selection
- Pattern aggregation and reuse
- Enhanced validation and filtering
- Adaptive message incorporation

## Usage

The mutation strategies are used internally by the GAP and TAP attackers. The correct way to use them is through the attacker classes:

```python
from easyjailbreak.attacker.GAP_Schwartz_2024 import GAP
from easyjailbreak.attacker.TAP_Mehrotra_2023 import TAP
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.datasets.jailbreak_datasets import JailbreakDataset

# Initialize models
attack_model = from_pretrained(model_path_1)
target_model = from_pretrained(model_path_2)
eval_model = from_pretrained(model_path_3)

# Create dataset
dataset = JailbreakDataset('AdvBench')

# Initialize GAP attacker (uses Graph-of-Thought mutation)
gap_attacker = GAP(
    attack_model=attack_model,
    target_model=target_model,
    eval_model=eval_model,
    jailbreak_datasets=dataset,
    tree_width=10,
    tree_depth=10,
    selection_strategy='max_score'
)

# Initialize TAP attacker (uses Tree-of-Thought mutation) 
tap_attacker = TAP(
    attack_model=attack_model,
    target_model=target_model,
    eval_model=eval_model,
    jailbreak_datasets=dataset,
    tree_width=10,
    tree_depth=10
)

# Run attacks
gap_attacker.attack()
tap_attacker.attack()
```

The mutation strategies (`IntrospectGenerationAggregateMaxConcatenate.py` and `IntrospectGeneration.py`) are components used by the respective attackers rather than being used directly.

## Key Parameters

- `branching_factor`: Number of children nodes per parent
- `keep_last_n`: Number of conversation rounds to retain
- `max_n_attack_attempts`: Maximum attempts for valid generation
- `selection_strategy`: Pattern selection method (GoT only)

## Advantages of Graph-of-Thought

1. **Pattern Recognition**: Learns from successful attacks across all branches
2. **Efficient Exploration**: Reuses proven attack patterns
3. **Adaptive Learning**: Incorporates historical successes into current attempts
4. **Quality Control**: Enhanced validation of generated prompts
5. **Memory Efficiency**: Smart filtering of conversation history