#!/usr/bin/env python3
"""
Script to upload initialized MoE models to Hugging Face Hub for testing.

This script assumes the models have been initialized and trained using the 
init-*-moe.yml configs, and will upload them with appropriate model cards.
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, Repository, create_repo
from transformers import AutoTokenizer, AutoModelForCausalLM


def create_model_card(model_name: str, architecture: str, params: str, experts: int, experts_per_tok: int) -> str:
    """Create a model card for the uploaded MoE model."""
    
    card_content = f"""---
license: apache-2.0
tags:
  - mixtral-of-experts
  - moe
  - {architecture.lower()}
  - axolotl
  - testing
language:
  - en
library_name: transformers
pipeline_tag: text-generation
---

# {model_name}

## Model Description

This is a small **{architecture}**-style Mixture of Experts (MoE) model with approximately **{params}** parameters, created for testing and development purposes. The model was initialized from scratch using a custom configuration and trained using [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) with optimized MoE kernels.

## Architecture

- **Architecture**: {architecture}
- **Total Parameters**: ~{params}
- **Number of Experts**: {experts}
- **Experts per Token**: {experts_per_tok}
- **Hidden Size**: 1024
- **Context Length**: 2048-8192 tokens
- **Vocabulary Size**: Varies by architecture

## Training

- **Framework**: Axolotl with MoE kernel optimizations
- **Data**: FineTome-100k (10% subset)
- **Training Type**: Continued pretraining from random initialization
- **Optimization**: AdamW with cosine scheduling
- **Mixed Precision**: bfloat16

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

# Generate text
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

## Intended Use

This model is intended for:
- Testing MoE kernel optimizations
- Architecture experimentation  
- Educational purposes
- Development and debugging

**Note**: This is a small model trained on limited data for testing purposes. It is not intended for production use.

## Training Details

The model was initialized using `random_init_weights: true` in Axolotl and trained with:
- Optimized MoE kernels (`moe_kernels: true`)
- Flash Attention
- Gradient checkpointing
- Router auxiliary loss for load balancing

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{model_name.lower().replace("-", "_"),
  title={{model_name}},
  author={{Axolotl AI}},
  year={{2024}},
  howpublished={{\\url{{https://huggingface.co/{model_name}}}}}
}}
```
"""
    return card_content


def upload_model(model_path: str, hub_name: str, architecture: str, experts: int, experts_per_tok: int):
    """Upload a model to Hugging Face Hub with model card."""
    
    print(f"Uploading {model_path} to {hub_name}...")
    
    # Determine parameter count from architecture
    param_map = {
        "mixtral": "1B", 
        "qwen2_moe": "1B",
        "deepseek_v3": "1B"
    }
    params = param_map.get(architecture.lower(), "1B")
    
    try:
        # Create repository (private)
        create_repo(hub_name, exist_ok=True, private=True)
        print(f"Created repository: {hub_name}")
        
        # Initialize HF API
        api = HfApi()
        
        # Upload model files
        api.upload_folder(
            folder_path=model_path,
            repo_id=hub_name,
            repo_type="model"
        )
        
        # Create and upload model card
        model_card = create_model_card(
            model_name=hub_name,
            architecture=architecture,
            params=params,
            experts=experts,
            experts_per_tok=experts_per_tok
        )
        
        with open("README.md", "w") as f:
            f.write(model_card)
            
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md", 
            repo_id=hub_name,
            repo_type="model"
        )
        
        os.remove("README.md")  # Clean up
        
        print(f"✅ Successfully uploaded {hub_name}")
        
    except Exception as e:
        print(f"❌ Error uploading {hub_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Upload initialized MoE models to Hugging Face Hub")
    parser.add_argument("--output-dir", default="./outputs", help="Directory containing trained models")
    parser.add_argument("--username", required=True, help="Hugging Face username/organization")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without uploading")
    
    args = parser.parse_args()
    
    # Model configurations for upload
    models_to_upload = [
        {
            "local_path": "mixtral-1b-moe-init",
            "hub_name": f"{args.username}/mixtral-1b-moe-test", 
            "architecture": "Mixtral",
            "experts": 8,
            "experts_per_tok": 2
        },
        {
            "local_path": "qwen2-moe-1b-init", 
            "hub_name": f"{args.username}/qwen2-moe-1b-test",
            "architecture": "Qwen2-MoE", 
            "experts": 16,
            "experts_per_tok": 2
        },
        {
            "local_path": "deepseek-v3-1b-init",
            "hub_name": f"{args.username}/deepseek-v3-1b-moe-test",
            "architecture": "DeepSeek-V3",
            "experts": 32, 
            "experts_per_tok": 4
        }
    ]
    
    print("🚀 MoE Model Upload Script")
    print(f"Output directory: {args.output_dir}")
    print(f"Target username: {args.username}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    for model in models_to_upload:
        local_path = os.path.join(args.output_dir, model["local_path"])
        
        if not os.path.exists(local_path):
            print(f"⚠️  Model not found: {local_path}")
            continue
            
        if args.dry_run:
            print(f"Would upload: {local_path} -> {model['hub_name']}")
        else:
            upload_model(
                model_path=local_path,
                hub_name=model["hub_name"],
                architecture=model["architecture"], 
                experts=model["experts"],
                experts_per_tok=model["experts_per_tok"]
            )
    
    print("\n✅ Upload process completed!")


if __name__ == "__main__":
    main()