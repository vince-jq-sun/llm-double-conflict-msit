#!/usr/bin/env python3
"""
Standalone ACDC Circuit Discovery Utility
-----------------------------------------
A standalone script for running ACDC analysis on transformer models.
Can be used independently of the MSIT test infrastructure.
"""

import argparse
import json
import torch
from pathlib import Path
from typing import List, Dict, Any

# Import ACDC functionality
try:
    from acdc_circuit_discovery import (
        ACDCCircuitDiscovery, ACDCConfig, HAS_ACDC
    )
except ImportError:
    HAS_ACDC = False
    print("ERROR: ACDC dependencies not found. Install with: pip install -r requirements_acdc.txt")
    exit(1)


def load_prompts_from_file(filepath: str) -> List[str]:
    """Load prompts from a text file (one per line) or JSON file."""
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'prompts' in data:
                return data['prompts']
            else:
                raise ValueError("JSON file must contain a list of prompts or dict with 'prompts' key")
    
    else:
        # Assume text file with one prompt per line
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]


def create_custom_corrupted_data(clean_prompts: List[str], corruption_method: str) -> List[str]:
    """Create corrupted versions of prompts using various methods."""
    import random
    import re
    
    corrupt_prompts = []
    
    for prompt in clean_prompts:
        if corruption_method == "shuffle_tokens":
            # Shuffle tokens in the prompt
            tokens = prompt.split()
            if len(tokens) > 1:
                shuffled = tokens.copy()
                random.shuffle(shuffled)
                corrupt_prompts.append(' '.join(shuffled))
            else:
                corrupt_prompts.append(prompt)
        
        elif corruption_method == "shuffle_digits":
            # Shuffle only digits in the prompt
            digits = re.findall(r'\d', prompt)
            if len(digits) > 1:
                shuffled_digits = digits.copy()
                random.shuffle(shuffled_digits)
                
                corrupted = prompt
                digit_idx = 0
                for i, char in enumerate(prompt):
                    if char.isdigit():
                        corrupted = corrupted[:i] + shuffled_digits[digit_idx] + corrupted[i+1:]
                        digit_idx += 1
                
                corrupt_prompts.append(corrupted)
            else:
                corrupt_prompts.append(prompt)
        
        elif corruption_method == "random_chars":
            # Replace random characters
            chars = list(prompt)
            n_changes = max(1, len(chars) // 10)  # Change 10% of characters
            positions = random.sample(range(len(chars)), min(n_changes, len(chars)))
            
            for pos in positions:
                if chars[pos].isalpha():
                    chars[pos] = chr(ord('a') + random.randint(0, 25))
                elif chars[pos].isdigit():
                    chars[pos] = str(random.randint(0, 9))
            
            corrupt_prompts.append(''.join(chars))
        
        else:  # "none" or unknown
            corrupt_prompts.append(prompt)
    
    return corrupt_prompts


def main():
    parser = argparse.ArgumentParser(description="Standalone ACDC Circuit Discovery")
    
    # Input/Output
    parser.add_argument("--input", type=str, required=True,
                       help="Input file containing prompts (text or JSON)")
    parser.add_argument("--output_dir", type=str, default="acdc_results",
                       help="Output directory for results")
    
    # Model
    parser.add_argument("--model", type=str, default="gpt2-small",
                       help="Model name (e.g., gpt2-small, gpt2-large)")
    
    # Corruption
    parser.add_argument("--corruption_method", type=str, default="shuffle_digits",
                       choices=["shuffle_tokens", "shuffle_digits", "random_chars", "none"],
                       help="Method for creating corrupted prompts")
    
    # ACDC Parameters
    parser.add_argument("--tau", type=float, default=1e-3,
                       help="ACDC tolerance (default: 1e-3)")
    parser.add_argument("--k_edges", type=int, default=80,
                       help="Number of edges to keep (default: 80)")
    parser.add_argument("--faithfulness_target", type=str, default="kl_div",
                       choices=["kl_div", "logit_diff"],
                       help="Faithfulness target (default: kl_div)")
    parser.add_argument("--seq_len", type=int, default=64,
                       help="Sequence length (default: 64)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size (default: 8)")
    parser.add_argument("--train_samples", type=int, default=100,
                       help="Training samples (default: 100)")
    parser.add_argument("--test_samples", type=int, default=50,
                       help="Test samples (default: 50)")
    parser.add_argument("--ablation_type", type=str, default="tokenwise_mean_corrupt",
                       choices=["tokenwise_mean_corrupt", "zero", "mean_corrupt"],
                       help="Ablation type (default: tokenwise_mean_corrupt)")
    
    # Analysis options
    parser.add_argument("--save_viz", action="store_true",
                       help="Save visualizations")
    parser.add_argument("--head_ablation", action="store_true",
                       help="Run individual head ablation analysis")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if not HAS_ACDC:
        print("ERROR: ACDC dependencies not installed.")
        print("Install with: pip install -r requirements_acdc.txt")
        return 1
    
    # Load prompts
    print(f"Loading prompts from: {args.input}")
    try:
        clean_prompts = load_prompts_from_file(args.input)
        print(f"Loaded {len(clean_prompts)} prompts")
    except Exception as e:
        print(f"ERROR loading prompts: {e}")
        return 1
    
    if len(clean_prompts) < args.train_samples + args.test_samples:
        print(f"WARNING: Only {len(clean_prompts)} prompts, need {args.train_samples + args.test_samples}")
        args.train_samples = min(args.train_samples, len(clean_prompts) // 2)
        args.test_samples = min(args.test_samples, len(clean_prompts) - args.train_samples)
        print(f"Adjusted to {args.train_samples} train, {args.test_samples} test")
    
    # Create corrupted prompts
    print(f"Creating corrupted prompts using method: {args.corruption_method}")
    corrupt_prompts = create_custom_corrupted_data(clean_prompts, args.corruption_method)
    
    # Create ACDC configuration
    config = ACDCConfig(
        tau=args.tau,
        k_edges=args.k_edges,
        faithfulness_target=args.faithfulness_target,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        n_train_samples=args.train_samples,
        n_test_samples=args.test_samples,
        ablation_type=args.ablation_type,
        save_visualizations=args.save_viz
    )
    
    print(f"\nStarting ACDC analysis...")
    print(f"Model: {args.model}")
    print(f"Config: tau={args.tau}, k_edges={args.k_edges}, target={args.faithfulness_target}")
    
    try:
        # Initialize ACDC
        acdc = ACDCCircuitDiscovery(args.model, config)
        
        # Prepare data
        train_loader, test_loader = acdc.prepare_msit_data(clean_prompts, corrupt_prompts)
        
        # Discover circuit
        print("Discovering circuit...")
        edges = acdc.discover_circuit(train_loader)
        
        # Evaluate circuit
        print("Evaluating circuit...")
        eval_results = acdc.evaluate_circuit(test_loader)
        
        # Optional head ablation analysis
        ablation_results = {}
        if args.head_ablation and acdc.discovered_heads:
            print("Running head ablation analysis...")
            test_batch = next(iter(test_loader))
            ablation_results = acdc.run_head_ablation_analysis(test_batch.clean)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        acdc.save_circuit(output_dir, "standalone_circuit")
        
        # Create comprehensive results
        results = {
            'input_file': args.input,
            'model': args.model,
            'corruption_method': args.corruption_method,
            'config': config.to_dict(),
            'circuit_discovery': {
                'n_edges': len(edges),
                'n_heads': len(acdc.discovered_heads),
                'n_mlps': len(acdc.discovered_mlps),
                'heads': acdc.discovered_heads,
                'mlps': acdc.discovered_mlps
            },
            'evaluation': eval_results,
            'ablation_analysis': ablation_results
        }
        
        # Save comprehensive results
        results_file = output_dir / "standalone_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n=== ACDC Analysis Complete ===")
        print(f"Discovered {len(edges)} edges")
        print(f"Found {len(acdc.discovered_heads)} attention heads")
        print(f"Found {len(acdc.discovered_mlps)} MLP layers")
        print(f"Circuit faithfulness: {eval_results.get('circuit_faithfulness', 'N/A'):.3f}")
        print(f"Circuit KL divergence: {eval_results.get('circuit_kl_divergence', 'N/A'):.4f}")
        
        if acdc.discovered_heads:
            print(f"\nDiscovered Heads:")
            for layer, head in acdc.discovered_heads[:10]:  # Show first 10
                print(f"  L{layer}H{head}")
            if len(acdc.discovered_heads) > 10:
                print(f"  ... and {len(acdc.discovered_heads) - 10} more")
        
        if ablation_results and 'head_rankings' in ablation_results:
            print(f"\nTop Head Effects (by KL divergence):")
            for head, effect in ablation_results['head_rankings'][:5]:
                print(f"  {head}: {effect:.4f}")
        
        print(f"\nResults saved to: {output_dir}")
        
    except Exception as e:
        print(f"ERROR: ACDC analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
