#!/usr/bin/env python3
"""
Multi-Head Swap Research Tool

Standalone tool for testing custom combinations of attention head swaps between clean and corrupt prompts.
Allows researchers to specify exact head combinations and analyze their combined interference effects.

Usage:
    python multi_head_swap_tool.py --heads "L5H8,L3H2,L7H4" --samples 20
    python multi_head_swap_tool.py --heads "L0H0,L0H1,L1H0" --model gpt2-large --samples 50
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any
import sys
import os
import torch

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from head_ablation_sweep import HeadAblationSweep

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def parse_head_list(head_str: str) -> List[Tuple[int, int]]:
    """Parse comma-separated head list like 'L5H8,L3H2,L7H4' into [(5,8), (3,2), (7,4)]."""
    heads = []
    for head_spec in head_str.split(','):
        head_spec = head_spec.strip()
        if not head_spec.startswith('L') or 'H' not in head_spec:
            raise ValueError(f"Invalid head format: {head_spec}. Use format like 'L5H8'")
        
        parts = head_spec[1:].split('H')  # Remove 'L' and split by 'H'
        if len(parts) != 2:
            raise ValueError(f"Invalid head format: {head_spec}. Use format like 'L5H8'")
        
        try:
            layer = int(parts[0])
            head = int(parts[1])
            heads.append((layer, head))
        except ValueError:
            raise ValueError(f"Invalid head format: {head_spec}. Layer and head must be integers")
    
    return heads

def format_results(results: Dict[str, Any], baseline_results: Dict[str, Any] = None) -> str:
    """Format multi-head swap results for display."""
    head_names = [f"L{layer}H{head}" for layer, head in results['heads_swapped']]
    
    output = []
    output.append(f"\n🚀 Multi-Head Swap Results")
    output.append(f"📋 Heads: {', '.join(head_names)} ({results['num_heads']} total)")
    output.append(f"📊 Sample size: {results['summary']['clean_results_count']}")
    output.append("")
    
    # Performance metrics
    output.append(f"🎯 Performance Metrics:")
    output.append(f"  Clean accuracy: {results['summary']['clean_accuracy']:.3f}")
    output.append(f"  Corrupt accuracy: {results['summary']['corrupt_accuracy']:.3f}")
    output.append(f"  Clean avg rank: {results['summary']['clean_avg_correct_rank']:.3f}")
    output.append(f"  Corrupt avg rank: {results['summary']['corrupt_avg_correct_rank']:.3f}")
    
    # Baseline comparison if available
    if baseline_results:
        baseline_clean_acc = baseline_results['baseline']['clean']['accuracy']
        baseline_corrupt_acc = baseline_results['baseline']['corrupt']['accuracy']
        baseline_clean_rank = baseline_results['baseline']['clean']['avg_correct_rank']
        baseline_corrupt_rank = baseline_results['baseline']['corrupt']['avg_correct_rank']
        
        clean_acc_change = results['summary']['clean_accuracy'] - baseline_clean_acc
        corrupt_acc_change = results['summary']['corrupt_accuracy'] - baseline_corrupt_acc
        clean_rank_change = results['summary']['clean_avg_correct_rank'] - baseline_clean_rank
        corrupt_rank_change = results['summary']['corrupt_avg_correct_rank'] - baseline_corrupt_rank
        
        output.append("")
        output.append(f"📈 Changes from Baseline:")
        output.append(f"  Clean accuracy change: {clean_acc_change:+.3f}")
        output.append(f"  Corrupt accuracy change: {corrupt_acc_change:+.3f}")
        output.append(f"  Clean rank change: {clean_rank_change:+.3f}")
        output.append(f"  Corrupt rank change: {corrupt_rank_change:+.3f}")
        
        # Task discrimination analysis
        baseline_task_disc = baseline_clean_acc - baseline_corrupt_acc
        multi_task_disc = results['summary']['clean_accuracy'] - results['summary']['corrupt_accuracy']
        task_disc_change = multi_task_disc - baseline_task_disc
        
        output.append("")
        output.append(f"🔍 Task Discrimination Analysis:")
        output.append(f"  Baseline task discrimination: {baseline_task_disc:.3f}")
        output.append(f"  Multi-head task discrimination: {multi_task_disc:.3f}")
        output.append(f"  Task discrimination change: {task_disc_change:+.3f}")
    
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description='Multi-Head Swap Research Tool')
    parser.add_argument('--heads', required=True, 
                       help='Comma-separated list of heads to swap (e.g., "L5H8,L3H2,L7H4")')
    parser.add_argument('--model', default='gpt2',
                       help='Model name (default: gpt2)')
    parser.add_argument('--samples', type=int, default=20,
                       help='Number of test samples (default: 20)')
    parser.add_argument('--output', default='multi_head_swap_results.json',
                       help='Output JSON filename (default: multi_head_swap_results.json)')
    parser.add_argument('--output_dir', default='../data/msit_pilot_outputs_smallnrep',
                       help='Output directory (default: ../data/msit_pilot_outputs_smallnrep)')
    parser.add_argument('--no_examples', action='store_true',
                       help='Disable few-shot examples in prompts')
    parser.add_argument('--baseline_file', 
                       help='Optional baseline results JSON file for comparison')
    parser.add_argument('--save_baseline', action='store_true',
                       help='Save baseline results for future comparisons')
    
    args = parser.parse_args()
    
    # Parse head list
    try:
        head_list = parse_head_list(args.heads)
    except ValueError as e:
        logger.error(f"Error parsing head list: {e}")
        return 1
    
    logger.info(f"Multi-Head Swap Tool")
    logger.info(f"Model: {args.model}")
    logger.info(f"Heads to swap: {args.heads} ({len(head_list)} heads)")
    logger.info(f"Samples: {args.samples}")
    logger.info(f"Few-shot examples: {not args.no_examples}")
    
    # Initialize sweep class
    sweep = HeadAblationSweep(model_name=args.model)
    sweep.load_model()  # Load model before using it
    
    # Generate prompts
    logger.info("Generating MSIT prompts...")
    clean_prompts, corrupt_prompts = sweep.create_msit_test_data(
        n_samples=args.samples,
        with_examples=not args.no_examples
    )
    
    # Load baseline if provided
    baseline_results = None
    if args.baseline_file:
        try:
            with open(args.baseline_file, 'r') as f:
                baseline_results = json.load(f)
            logger.info(f"Loaded baseline from: {args.baseline_file}")
        except FileNotFoundError:
            logger.warning(f"Baseline file not found: {args.baseline_file}")
        except Exception as e:
            logger.error(f"Error loading baseline: {e}")
    
    # Run baseline if needed
    if args.save_baseline or not baseline_results:
        logger.info("Running baseline measurement...")
        baseline_clean_results = []
        baseline_corrupt_results = []
        
        # Get baseline performance without any modifications
        with torch.no_grad():  # Use no_grad context instead
            for i, (clean_prompt, corrupt_prompt) in enumerate(zip(clean_prompts, corrupt_prompts)):
                # Process clean prompt
                clean_tokens = sweep.model.to_tokens(clean_prompt, prepend_bos=True)
                clean_logits = sweep.model(clean_tokens)
                clean_last_logits = clean_logits[0, -1, :]
                clean_logits_info = sweep.get_digit_logits_info_transformer_lens(clean_prompt, clean_last_logits)
                
                if "error" not in clean_logits_info:
                    clean_correct_answer, clean_stimulus_digits = sweep.extract_msit_info(clean_prompt)
                    if clean_correct_answer and clean_stimulus_digits:
                        clean_stimulus_probs = {d: clean_logits_info["digit_softmax_probs"][d] 
                                              for d in clean_stimulus_digits if d in clean_logits_info["digit_softmax_probs"]}
                        if clean_stimulus_probs:
                            clean_sorted_stimulus = sorted(clean_stimulus_probs.items(), key=lambda x: x[1], reverse=True)
                            clean_stimulus_ranks = {digit: rank + 1 for rank, (digit, _) in enumerate(clean_sorted_stimulus)}
                            baseline_clean_results.append({
                                'correct_answer': clean_correct_answer,
                                'stimulus_ranks': clean_stimulus_ranks,
                                'is_correct_top_choice': clean_stimulus_ranks.get(clean_correct_answer, 999) == 1,
                                'correct_answer_stimulus_rank': clean_stimulus_ranks.get(clean_correct_answer, len(clean_stimulus_digits) + 1)
                            })
                
                # Process corrupt prompt
                corrupt_tokens = sweep.model.to_tokens(corrupt_prompt, prepend_bos=True)
                corrupt_logits = sweep.model(corrupt_tokens)
                corrupt_last_logits = corrupt_logits[0, -1, :]
                corrupt_logits_info = sweep.get_digit_logits_info_transformer_lens(corrupt_prompt, corrupt_last_logits)
                
                if "error" not in corrupt_logits_info:
                    corrupt_correct_answer, corrupt_stimulus_digits = sweep.extract_msit_info(corrupt_prompt)
                    if corrupt_correct_answer and corrupt_stimulus_digits:
                        corrupt_stimulus_probs = {d: corrupt_logits_info["digit_softmax_probs"][d] 
                                                for d in corrupt_stimulus_digits if d in corrupt_logits_info["digit_softmax_probs"]}
                        if corrupt_stimulus_probs:
                            corrupt_sorted_stimulus = sorted(corrupt_stimulus_probs.items(), key=lambda x: x[1], reverse=True)
                            corrupt_stimulus_ranks = {digit: rank + 1 for rank, (digit, _) in enumerate(corrupt_sorted_stimulus)}
                            baseline_corrupt_results.append({
                                'correct_answer': corrupt_correct_answer,
                                'stimulus_ranks': corrupt_stimulus_ranks,
                                'is_correct_top_choice': corrupt_stimulus_ranks.get(corrupt_correct_answer, 999) == 1,
                                'correct_answer_stimulus_rank': corrupt_stimulus_ranks.get(corrupt_correct_answer, len(corrupt_stimulus_digits) + 1)
                            })
        
        # Calculate baseline statistics
        if baseline_clean_results:
            clean_accuracy = sum(r['is_correct_top_choice'] for r in baseline_clean_results) / len(baseline_clean_results)
            clean_avg_rank = sum(r['correct_answer_stimulus_rank'] for r in baseline_clean_results) / len(baseline_clean_results)
        else:
            clean_accuracy = 0.0
            clean_avg_rank = 999.0
            
        if baseline_corrupt_results:
            corrupt_accuracy = sum(r['is_correct_top_choice'] for r in baseline_corrupt_results) / len(baseline_corrupt_results)
            corrupt_avg_rank = sum(r['correct_answer_stimulus_rank'] for r in baseline_corrupt_results) / len(baseline_corrupt_results)
        else:
            corrupt_accuracy = 0.0
            corrupt_avg_rank = 999.0
        
        baseline_results = {
            'baseline': {
                'clean': {
                    'accuracy': clean_accuracy,
                    'avg_correct_rank': clean_avg_rank,
                    'results_count': len(baseline_clean_results)
                },
                'corrupt': {
                    'accuracy': corrupt_accuracy,
                    'avg_correct_rank': corrupt_avg_rank,
                    'results_count': len(baseline_corrupt_results)
                },
                'task_discrimination': {
                    'accuracy_gap': clean_accuracy - corrupt_accuracy,
                    'rank_gap': clean_avg_rank - corrupt_avg_rank
                }
            }
        }
        
        logger.info(f"Baseline - Clean accuracy: {clean_accuracy:.3f}, Corrupt accuracy: {corrupt_accuracy:.3f}")
    
    # Run multi-head swap
    logger.info(f"Running multi-head swap for {len(head_list)} heads...")
    multi_results = sweep.multi_head_swap_ranking(head_list, clean_prompts, corrupt_prompts)
    
    # Combine results
    full_results = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'samples': args.samples,
        'heads_tested': args.heads,
        'multi_head_swap': multi_results
    }
    
    if baseline_results:
        full_results.update(baseline_results)
    
    # Save results
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        # Get repo root from current script location (llm_control is now repo root)
        script_dir = Path(__file__).parent  # scripts_test/
        repo_root = script_dir.parent        # llm_control/ (now repo root)
        # Handle relative paths correctly - ensure they stay within repo
        if args.output_dir.startswith("../../"):
            # Convert ../../data/... to ../data/... from repo root
            relative_part = args.output_dir[6:]  # Remove "../../"
            output_dir = repo_root / relative_part
        elif args.output_dir.startswith("../"):
            # Convert ../data/... to data/... from repo root  
            relative_part = args.output_dir[3:]  # Remove "../"
            output_dir = repo_root / relative_part
        else:
            output_dir = repo_root / args.output_dir
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"{timestamp}_{args.model}_multi-head-swap_heads-{len(head_list)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = run_dir / args.output
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")
    
    # Save baseline if requested
    if args.save_baseline and baseline_results:
        baseline_path = run_dir / "baseline_results.json"
        with open(baseline_path, 'w') as f:
            json.dump(baseline_results, f, indent=2)
        logger.info(f"Baseline saved to: {baseline_path}")
    
    # Display results
    print(format_results(multi_results, baseline_results))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
