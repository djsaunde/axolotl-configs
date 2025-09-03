# MoE Model Initialization and Testing Configs

This directory contains configurations for creating and testing smaller (~1B parameter) Mixture of Experts models for development and testing purposes.

## Quick Start

### 1. Initialize Models from Scratch

```bash
# Initialize a 1B parameter Mixtral-style MoE model
axolotl train /workspace/configs/moe/init-mixtral-1b-moe.yml

# Initialize a 1B parameter Qwen2-MoE style model  
axolotl train /workspace/configs/moe/init-qwen2-moe-1b.yml

# Initialize a 1B parameter DeepSeek-V3 style MoE model
axolotl train /workspace/configs/moe/init-deepseek-v3-1b-moe.yml
```

### 2. Test Existing Models

```bash
# Test with existing models using smaller configs
axolotl train /workspace/configs/moe/deepseek-v3-1b-test.yml
axolotl train /workspace/configs/moe/mixtral-8x7b-moe-test.yml
```

### 3. Upload Initialized Models

```bash
# Upload your initialized models to Hugging Face Hub
python /workspace/configs/moe/upload-models.py --username YOUR_USERNAME
```

## Configuration Types

### Initialization Configs (`init-*.yml`)
- **Purpose**: Create new MoE models from scratch with `random_init_weights: true`
- **Size**: ~1B parameters each
- **Features**: Custom architectures optimized for testing

### Testing Configs (`*-test.yml`) 
- **Purpose**: Fast testing with existing or initialized models
- **Features**: Small datasets (2-5%), short sequences, early stopping

### Production Configs (`*-optimized.yml`)
- **Purpose**: Full-scale training with existing large models
- **Features**: Full datasets, longer contexts, complete training

## Architecture Details

### Mixtral-1B-MoE (`init-mixtral-1b-moe.yml`)
- **Parameters**: ~1B total
- **Hidden Size**: 1024
- **Layers**: 16
- **Experts**: 8 experts, 2 active per token
- **Context**: 4096 tokens

### Qwen2-MoE-1B (`init-qwen2-moe-1b.yml`) 
- **Parameters**: ~1B total
- **Hidden Size**: 1024
- **Layers**: 16  
- **Experts**: 16 experts, 2 active per token
- **Context**: 8192 tokens
- **Features**: Shared experts + routed experts

### DeepSeek-V3-1B-MoE (`init-deepseek-v3-1b-moe.yml`)
- **Parameters**: ~1B total
- **Hidden Size**: 1024
- **Layers**: 16
- **Experts**: 32 routed + 2 shared experts, 4 active per token
- **Context**: 8192 tokens

## MoE Kernel Optimization

All configs include optimized MoE kernels:

```yaml
plugins:
  - axolotl.integrations.moe_kernels.plugin.MoeOptimizedPlugin

moe_kernels: true
moe_group_size: 128
moe_persistent_kernel: true
```

These provide significant speedups through:
- Contiguous grouped GEMM operations
- Token sorting for memory coalescence  
- Triton kernel auto-tuning

## Training Tips

### For Initialization (from scratch):
- Use higher learning rates (3e-4)
- Increase warmup ratio (0.1) 
- Higher weight decay (0.1)
- More gradient accumulation

### For Fine-tuning:
- Lower learning rates (1e-5 to 2e-5)
- Less warmup (0.05)
- Standard weight decay (0.01)

## Memory Requirements

Approximate GPU memory needed:
- **1B MoE models**: 8-16GB (depending on batch size)
- **Testing configs**: 4-8GB  
- **Large models (Mixtral-8x7B)**: 32GB+ or multi-GPU

## Supported Architectures

✅ **Working with MoE kernels**:
- Mixtral (8x7B, 8x22B)
- Qwen2-MoE  
- Qwen3-MoE
- DeepSeek-V3

❌ **Not yet supported**:
- DeepSeek-V2 (template provided for future support)

## Files

```
├── init-mixtral-1b-moe.yml      # Initialize Mixtral-1B from scratch
├── init-qwen2-moe-1b.yml        # Initialize Qwen2-MoE-1B from scratch  
├── init-deepseek-v3-1b-moe.yml  # Initialize DeepSeek-V3-1B from scratch
├── *-test.yml                   # Fast testing configs
├── *-optimized.yml              # Production training configs  
├── upload-models.py             # Script to upload models to HF Hub
└── README.md                    # This file
```

## Next Steps

1. **Train initialization configs** to create your 1B MoE models
2. **Test MoE kernel performance** with the optimized configs
3. **Upload models** for sharing and collaboration
4. **Scale up** to larger architectures once testing is complete

The initialized 1B models will be perfect for testing MoE kernel optimizations, architecture experiments, and rapid iteration without the cost of training large models from scratch.