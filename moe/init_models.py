#!/usr/bin/env python3
"""
Direct model initialization script using transformers.
Creates small MoE models from scratch with random weights.
"""

import os
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    MixtralConfig, MixtralForCausalLM,
    Qwen2MoeConfig, Qwen2MoeForCausalLM
)
from pathlib import Path


def init_mixtral_1b():
    """Initialize a 1B parameter Mixtral-style MoE model."""
    print("🚀 Initializing Mixtral-1B MoE model...")
    
    # Configuration for ~1B parameters
    config = MixtralConfig(
        vocab_size=32000,
        hidden_size=1024,          # Reduced from 4096
        intermediate_size=3584,    # 3.5x hidden_size
        num_hidden_layers=16,      # Reduced from 32
        num_attention_heads=16,    # Reduced from 32
        num_key_value_heads=4,     # GQA ratio
        num_experts_per_tok=2,
        num_local_experts=8,
        max_position_embeddings=4096,
        sliding_window=2048,
        rope_theta=1000000.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        output_router_logits=True,
        router_aux_loss_coef=0.001,
    )
    
    # Initialize model with random weights
    model = MixtralForCausalLM(config)
    
    # Get tokenizer from existing model
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    
    # Save model and tokenizer
    output_dir = "./outputs/mixtral-1b-moe-init"
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Mixtral-1B MoE saved to {output_dir}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Experts: {config.num_local_experts}, Active per token: {config.num_experts_per_tok}")
    

def init_qwen2_moe_1b():
    """Initialize a 1B parameter Qwen2-MoE style model."""
    print("🚀 Initializing Qwen2-MoE-1B model...")
    
    # Configuration for ~1B parameters  
    config = Qwen2MoeConfig(
        vocab_size=151936,         # Qwen2 vocab size
        hidden_size=1024,          # Reduced from 2048
        intermediate_size=2816,    # 2.75x hidden_size
        max_position_embeddings=8192,
        num_hidden_layers=16,      # Reduced from 24
        num_attention_heads=16,
        num_key_value_heads=16,    # No GQA
        num_experts=16,            # Reduced from 60
        num_experts_per_tok=2,     # Reduced from 4
        moe_intermediate_size=1024, # Expert FFN size
        shared_expert_intermediate_size=2816,
        decoder_sparse_step=1,     # MoE every layer
        output_router_logits=True,
        router_aux_loss_coef=0.001,
        norm_topk_prob=True,
        attention_dropout=0.0,
        rope_theta=1000000.0,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        initializer_range=0.02,
    )
    
    # Initialize model with random weights
    model = Qwen2MoeForCausalLM(config)
    
    # Get tokenizer from existing model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    # Save model and tokenizer
    output_dir = "./outputs/qwen2-moe-1b-init"
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Qwen2-MoE-1B saved to {output_dir}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Experts: {config.num_experts}, Active per token: {config.num_experts_per_tok}")


def init_deepseek_v3_1b():
    """Initialize a 1B parameter DeepSeek-V3 style model."""
    print("🚀 Initializing DeepSeek-V3-1B MoE model...")
    
    # DeepSeek-V3 isn't in standard transformers yet, so we'll create a Qwen2-MoE 
    # config that mimics the DeepSeek-V3 architecture
    config = Qwen2MoeConfig(
        vocab_size=129024,         # DeepSeek vocab size
        hidden_size=1024,
        intermediate_size=2752,    # 2.69x hidden_size
        max_position_embeddings=8192,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_experts=34,            # 32 routed + 2 shared (approx)
        num_experts_per_tok=4,     # DeepSeek-V3 style
        moe_intermediate_size=1024,
        shared_expert_intermediate_size=2752,
        decoder_sparse_step=1,
        output_router_logits=True,
        router_aux_loss_coef=0.001,
        norm_topk_prob=True,
        attention_dropout=0.0,
        rope_theta=10000.0,        # DeepSeek default
        hidden_act="silu",
        rms_norm_eps=1e-6,
        initializer_range=0.02,
    )
    
    # Initialize model with random weights
    model = Qwen2MoeForCausalLM(config)
    
    # Get tokenizer from existing model
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite-Chat")
    
    # Save model and tokenizer
    output_dir = "./outputs/deepseek-v3-1b-init"
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ DeepSeek-V3-1B MoE saved to {output_dir}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Experts: {config.num_experts}, Active per token: {config.num_experts_per_tok}")


def main():
    print("🔧 Initializing 1B Parameter MoE Models")
    print("=" * 50)
    
    try:
        init_mixtral_1b()
        print()
        
        init_qwen2_moe_1b()
        print()
        
        init_deepseek_v3_1b()
        print()
        
        print("🎉 All models initialized successfully!")
        print("\nNext steps:")
        print("1. Test the models with inference")
        print("2. Upload to Hugging Face Hub")
        print("3. Use with axolotl MoE kernel optimizations")
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        raise


if __name__ == "__main__":
    main()