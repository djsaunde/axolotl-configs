# Updated MoE Configs Using Our Initialized Models

All configs have been updated to use our newly initialized 1B parameter MoE models instead of the large existing models.

## **Updated Model Mappings:**

| **Config File** | **Old Model** | **New Model** | **Parameters** |
|-----------------|---------------|---------------|----------------|
| `mixtral-8x7b-moe-optimized.yml` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | `axolotl-ai-co/mixtral-1b-moe-test` | ~1.5B |
| `qwen2-moe-a14b-optimized.yml` | `Qwen/Qwen2.5-A14B-Instruct` | `axolotl-ai-co/qwen2-moe-1b-test` | ~1.3B |
| `deepseek-v3-1b-moe-optimized.yml` | `axolotl-ai-co/DeepSeek-V3-1B` | `axolotl-ai-co/deepseek-v3-1b-moe-test` | ~2.2B |

## **Updated Test Configs:**

| **Config File** | **Old Model** | **New Model** | **Parameters** |
|-----------------|---------------|---------------|----------------|
| `mixtral-8x7b-moe-test.yml` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | `axolotl-ai-co/mixtral-1b-moe-test` | ~1.5B |
| `qwen2-moe-7b-test.yml` | `Qwen/Qwen2.5-7B-Instruct` | `axolotl-ai-co/qwen2-moe-1b-test` | ~1.3B |
| `deepseek-v3-1b-test.yml` | `axolotl-ai-co/DeepSeek-V3-1B` | `axolotl-ai-co/deepseek-v3-1b-moe-test` | ~2.2B |

## **Benefits of Using Our Models:**

✅ **Much Faster Training**: 1-2B vs 7-45B parameters  
✅ **Lower Memory Usage**: Fits on smaller GPUs  
✅ **Rapid Iteration**: Test MoE kernel optimizations quickly  
✅ **Cost Effective**: Less compute required  
✅ **Same Architecture**: Same MoE structure as large models  
✅ **Private Models**: Under axolotl-ai-co control  

## **Quick Test Commands:**

```bash
# Test fastest model (DeepSeek V3 1B) 
axolotl train /workspace/configs/moe/deepseek-v3-1b-test.yml

# Test Mixtral 1B
axolotl train /workspace/configs/moe/mixtral-8x7b-moe-test.yml

# Test Qwen2-MoE 1B
axolotl train /workspace/configs/moe/qwen2-moe-7b-test.yml
```

## **Full Training Commands:**

```bash
# Full training with optimized models
axolotl train /workspace/configs/moe/deepseek-v3-1b-moe-optimized.yml
axolotl train /workspace/configs/moe/mixtral-8x7b-moe-optimized.yml  
axolotl train /workspace/configs/moe/qwen2-moe-a14b-optimized.yml
```

## **Model Architecture Details:**

### **Mixtral-1B MoE** (`axolotl-ai-co/mixtral-1b-moe-test`)
- **Hidden Size**: 1024
- **Layers**: 16  
- **Experts**: 8 total, 2 active per token
- **Context**: 4096 tokens
- **Parameters**: ~1.5B total

### **Qwen2-MoE-1B** (`axolotl-ai-co/qwen2-moe-1b-test`)
- **Hidden Size**: 1024
- **Layers**: 16
- **Experts**: 16 total, 2 active per token  
- **Context**: 8192 tokens
- **Parameters**: ~1.3B total

### **DeepSeek-V3-1B** (`axolotl-ai-co/deepseek-v3-1b-moe-test`)
- **Hidden Size**: 1024
- **Layers**: 16
- **Experts**: 34 total (32 routed + 2 shared), 4 active per token
- **Context**: 8192 tokens  
- **Parameters**: ~2.2B total

All models are **private repositories** under `axolotl-ai-co` and ready for MoE kernel optimization testing!