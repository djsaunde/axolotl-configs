# Streaming Dataset Mixing Configuration Examples

This directory contains example configurations that exercise the streaming dataset mixing functionality added to Axolotl.

## Features Tested

### Dataset Mixing Strategies

1. **concatenate_mix.yaml** - Tests concatenate strategy with streaming datasets
   - Uses `dataset_mixing_strategy: concatenate`
   - Should combine datasets sequentially

2. **round_robin_mix.yaml** - Tests round-robin mixing with streaming datasets  
   - Uses `dataset_mixing_strategy: round_robin`
   - Alternates samples between datasets until shortest is exhausted

3. **weighted_mix.yaml** - Tests weighted mixing with streaming datasets
   - Uses `dataset_mixing_strategy: weighted` with `mixing_weights: [0.7, 0.3]`
   - Samples 70% from first dataset, 30% from second

4. **random_mix.yaml** - Tests random sampling with equal probability
   - Uses `dataset_mixing_strategy: random` 
   - Equivalent to weighted with equal probabilities

5. **three_datasets_weighted.yaml** - Tests weighted mixing with 3 datasets
   - Uses 3 datasets with weights `[0.5, 0.3, 0.2]`
   - Tests multi-dataset weight validation

### Advanced Features

6. **eval_specific_mix.yaml** - Tests different mixing strategies for train vs eval
   - Training uses `dataset_mixing_strategy: round_robin`
   - Evaluation uses `eval_dataset_mixing_strategy: weighted` with `eval_mixing_weights: [0.6, 0.4]`
   - Tests the eval-specific overrides

### Validation Testing  

7. **validation_test.yaml** - Tests pydantic validation
   - Intentionally misconfigured with 2 datasets but 1 weight
   - Should fail validation with clear error message

## Key Changes Exercised

- **Unified dataset mixing**: All strategies work with both regular and streaming datasets
- **Pydantic validation**: Weight length validation at config time
- **Eval-specific mixing**: Separate mixing configs for training vs evaluation
- **Seed consistency**: Reproducible mixing with proper seed usage
- **Error handling**: Clear validation errors instead of silent fallbacks

## Usage

```bash
# This should work (valid config)
axolotl train configs/streaming/weighted_mix.yaml

# This should fail with validation error (invalid weights)
axolotl train configs/streaming/validation_test.yaml
```