# Consistent MoE Config Naming

All configs now have **consistent naming** for easy comparison and use.

## **📁 Config Structure:**

```
/workspace/configs/moe/
├── mixtral-1b-baseline.yml          # Mixtral 1B WITHOUT MoE kernels  
├── mixtral-1b-optimized.yml         # Mixtral 1B WITH MoE kernels
├── mixtral-1b-test.yml              # Mixtral 1B fast test (WITH kernels)
├── 
├── qwen2-moe-1b-baseline.yml        # Qwen2-MoE 1B WITHOUT MoE kernels
├── qwen2-moe-1b-optimized.yml       # Qwen2-MoE 1B WITH MoE kernels  
├── qwen2-moe-1b-test.yml            # Qwen2-MoE 1B fast test (WITH kernels)
├──
├── deepseek-v3-1b-baseline.yml      # DeepSeek-V3 1B WITHOUT MoE kernels
├── deepseek-v3-1b-optimized.yml     # DeepSeek-V3 1B WITH MoE kernels
└── deepseek-v3-1b-test.yml          # DeepSeek-V3 1B fast test (WITH kernels)
```

## **🔄 Config Comparison Pairs:**

### **Mixtral 1B:**
| **Baseline (No Kernels)** | **Optimized (With Kernels)** | **Fast Test (With Kernels)** |
|---------------------------|-------------------------------|------------------------------|
| `mixtral-1b-baseline.yml` | `mixtral-1b-optimized.yml` | `mixtral-1b-test.yml` |

### **Qwen2-MoE 1B:**
| **Baseline (No Kernels)** | **Optimized (With Kernels)** | **Fast Test (With Kernels)** |
|---------------------------|-------------------------------|------------------------------|
| `qwen2-moe-1b-baseline.yml` | `qwen2-moe-1b-optimized.yml` | `qwen2-moe-1b-test.yml` |

### **DeepSeek-V3 1B:**
| **Baseline (No Kernels)** | **Optimized (With Kernels)** | **Fast Test (With Kernels)** |
|---------------------------|-------------------------------|------------------------------|
| `deepseek-v3-1b-baseline.yml` | `deepseek-v3-1b-optimized.yml` | `deepseek-v3-1b-test.yml` |

---

## **⚡ Quick Commands:**

### **Performance Comparison (A/B Test):**
```bash
# Test standard PyTorch MoE (baseline)
axolotl train /workspace/configs/moe/deepseek-v3-1b-baseline.yml

# Test optimized MoE kernels  
axolotl train /workspace/configs/moe/deepseek-v3-1b-optimized.yml
```

### **Fast Testing (50-100 steps):**
```bash
axolotl train /workspace/configs/moe/deepseek-v3-1b-test.yml
axolotl train /workspace/configs/moe/mixtral-1b-test.yml
axolotl train /workspace/configs/moe/qwen2-moe-1b-test.yml
```

### **Full Training:**
```bash
axolotl train /workspace/configs/moe/deepseek-v3-1b-optimized.yml
axolotl train /workspace/configs/moe/mixtral-1b-optimized.yml
axolotl train /workspace/configs/moe/qwen2-moe-1b-optimized.yml
```

---

## **📊 Config Types Explained:**

### **🔸 Baseline Configs:**
- ❌ **No MoE kernel optimizations**
- 📈 Standard PyTorch MoE implementation  
- 🎯 Used for performance comparison
- 📊 WandB project: `moe-baseline`

### **🔸 Optimized Configs:**
- ✅ **MoE kernel optimizations enabled**
- ⚡ Contiguous grouped GEMM operations
- 🚀 Token sorting and memory coalescence
- 📊 WandB project: `moe-kernels`

### **🔸 Test Configs:**
- ✅ **MoE kernel optimizations enabled**
- ⚡ Small datasets (5% data)
- 🏃 Fast iteration (50-100 max steps)
- 📊 WandB project: `moe-kernels-test`

---

## **🎯 Recommended Testing Workflow:**

1. **Start with fast test**: `deepseek-v3-1b-test.yml` (50 steps)
2. **Run baseline comparison**: `deepseek-v3-1b-baseline.yml` vs `deepseek-v3-1b-optimized.yml`
3. **Scale to other architectures**: Mixtral and Qwen2-MoE tests
4. **Full training**: Once satisfied with optimizations

All configs use your **private 1B models** (`axolotl-ai-co/*-test`) for consistent, fast testing!