#!/usr/bin/env python3
"""
Head Impact Analysis Tool for MSIT Results
Generates readable reports from head ablation/swap sweep results.
"""

import json
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

def load_sweep_results(results_path: str) -> Dict[str, Any]:
    """Load head ablation/swap results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def extract_single_head_impact(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract impact data from single head verification results."""
    mode = results.get('analysis_mode', 'unknown')
    target_head = results.get('target_head', {})
    layer = target_head.get('layer', 0)
    head = target_head.get('head', 0)
    
    # Get baseline performance
    baseline_clean = results.get('baseline', {}).get('clean_results', {})
    baseline_corrupt = results.get('baseline', {}).get('corrupt_results', {})
    
    clean_acc = baseline_clean.get('summary_stats', {}).get('accuracy', 0)
    corrupt_acc = baseline_corrupt.get('summary_stats', {}).get('accuracy', 0)
    
    # Get modified performance 
    modified_clean = results.get('modified', {}).get('clean_results', {})
    modified_corrupt = results.get('modified', {}).get('corrupt_results', {})
    
    mod_clean_acc = modified_clean.get('summary_stats', {}).get('accuracy', 0) if modified_clean else clean_acc
    mod_corrupt_acc = modified_corrupt.get('summary_stats', {}).get('accuracy', 0) if modified_corrupt else corrupt_acc
    
    # Calculate changes
    clean_acc_change = clean_acc - mod_clean_acc if mode == 'ablate' else mod_clean_acc - clean_acc
    corrupt_acc_change = corrupt_acc - mod_corrupt_acc if mode == 'ablate' else mod_corrupt_acc - corrupt_acc
    
    baseline_discrimination = clean_acc - corrupt_acc
    modified_discrimination = mod_clean_acc - mod_corrupt_acc
    discrimination_change = abs(baseline_discrimination - modified_discrimination)
    
    return [{
        'layer': layer,
        'head': head,
        'head_id': f"L{layer}H{head}",
        'mode': mode,
        'clean_accuracy_change': clean_acc_change,
        'corrupt_accuracy_change': corrupt_acc_change,
        'clean_rank_change': 0,  # Not available in single verification
        'corrupt_rank_change': 0,
        'clean_logit_change': 0,
        'corrupt_logit_change': 0,
        'task_discrimination_loss': discrimination_change,
        'importance_score': discrimination_change + abs(clean_acc_change) + abs(corrupt_acc_change),
        'abs_clean_acc_change': abs(clean_acc_change),
        'abs_corrupt_acc_change': abs(corrupt_acc_change),
        'max_acc_change': max(abs(clean_acc_change), abs(corrupt_acc_change)),
        'rank_impact': 0,
        'logit_impact': 0
    }]

def extract_head_impacts(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract head impact data for analysis."""
    head_impacts = []
    
    mode = results.get('analysis_mode', results.get('mode', 'ablate'))
    
    # Handle different result formats
    if results.get('mode') == 'single_head_verification':
        # Single head verification format
        return extract_single_head_impact(results)
    
    # Full sweep format
    if 'head_effects' not in results:
        print("Warning: No head_effects found in results")
        return []
    
    baseline_clean_acc = results['baseline_clean']['summary_stats']['accuracy']
    baseline_corrupt_acc = results['baseline_corrupt']['summary_stats']['accuracy']
    baseline_task_discrimination = baseline_clean_acc - baseline_corrupt_acc
    
    for head_key, head_data in results['head_effects'].items():
        layer, head = head_key.replace('L', '').replace('H', ':').split(':')
        layer, head = int(layer), int(head)
        
        # Extract performance metrics
        clean_acc_change = head_data.get('clean_accuracy_change', 0)
        corrupt_acc_change = head_data.get('corrupt_accuracy_change', 0)
        clean_rank_change = head_data.get('clean_rank_change', 0)
        corrupt_rank_change = head_data.get('corrupt_rank_change', 0)
        clean_logit_change = head_data.get('clean_logit_change', 0)
        corrupt_logit_change = head_data.get('corrupt_logit_change', 0)
        
        # Calculate task discrimination impact
        if mode == "ablate":
            # For ablation: how much task discrimination is lost
            modified_task_discrimination = (baseline_clean_acc - clean_acc_change) - (baseline_corrupt_acc - corrupt_acc_change)
            discrimination_loss = baseline_task_discrimination - modified_task_discrimination
        else:  # swap mode
            # For swap: how task discrimination changes
            modified_task_discrimination = head_data['modified_clean_results']['accuracy'] - head_data['modified_corrupt_results']['accuracy']
            discrimination_change = modified_task_discrimination - baseline_task_discrimination
            discrimination_loss = -discrimination_change  # Negative change = loss
        
        # Importance metrics
        importance_score = head_data.get('importance_score', 0)
        
        head_impacts.append({
            'layer': layer,
            'head': head,
            'head_id': f"L{layer}H{head}",
            'mode': mode,
            'clean_accuracy_change': clean_acc_change,
            'corrupt_accuracy_change': corrupt_acc_change,
            'clean_rank_change': clean_rank_change,
            'corrupt_rank_change': corrupt_rank_change,
            'clean_logit_change': clean_logit_change,
            'corrupt_logit_change': corrupt_logit_change,
            'task_discrimination_loss': discrimination_loss,
            'importance_score': importance_score,
            'abs_clean_acc_change': abs(clean_acc_change),
            'abs_corrupt_acc_change': abs(corrupt_acc_change),
            'max_acc_change': max(abs(clean_acc_change), abs(corrupt_acc_change)),
            'rank_impact': max(abs(clean_rank_change), abs(corrupt_rank_change)),
            'logit_impact': max(abs(clean_logit_change), abs(corrupt_logit_change))
        })
    
    return head_impacts

def generate_impact_report(results_path: str, output_path: str = None, top_n: int = 20):
    """Generate comprehensive head impact analysis report."""
    
    # Load results
    results = load_sweep_results(results_path)
    head_impacts = extract_head_impacts(results)
    
    if not head_impacts:
        print("No head impact data found in results.")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(head_impacts)
    
    # Sort by different criteria
    df_by_importance = df.sort_values('importance_score', ascending=False)
    df_by_discrimination = df.sort_values('task_discrimination_loss', ascending=False)
    df_by_accuracy = df.sort_values('max_acc_change', ascending=False)
    df_by_rank = df.sort_values('rank_impact', ascending=False)
    
    # Generate report
    mode = results.get('analysis_mode', results.get('mode', 'unknown'))
    model_name = results.get('model_name', 'gpt2')
    n_samples = results.get('n_samples', 'unknown')
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"HEAD IMPACT ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Model: {model_name}")
    report_lines.append(f"Analysis Mode: {mode.upper()}")
    report_lines.append(f"Test Samples: {n_samples}")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Handle baseline performance for different formats
    if results.get('mode') == 'single_head_verification':
        baseline_clean_acc = results.get('baseline', {}).get('clean_results', {}).get('summary_stats', {}).get('accuracy', 0)
        baseline_corrupt_acc = results.get('baseline', {}).get('corrupt_results', {}).get('summary_stats', {}).get('accuracy', 0)
    else:
        baseline_clean_acc = results.get('baseline_clean', {}).get('summary_stats', {}).get('accuracy', 0)
        baseline_corrupt_acc = results.get('baseline_corrupt', {}).get('summary_stats', {}).get('accuracy', 0)
    
    baseline_task_discrimination = baseline_clean_acc - baseline_corrupt_acc
    
    report_lines.append("BASELINE PERFORMANCE:")
    report_lines.append(f"  Clean Accuracy: {baseline_clean_acc:.3f}")
    report_lines.append(f"  Corrupt Accuracy: {baseline_corrupt_acc:.3f}")
    report_lines.append(f"  Task Discrimination: {baseline_task_discrimination:.3f}")
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("SUMMARY STATISTICS:")
    report_lines.append(f"  Total heads analyzed: {len(df)}")
    report_lines.append(f"  Heads with significant impact (importance > 0.05): {len(df[df['importance_score'] > 0.05])}")
    report_lines.append(f"  Max accuracy change: {df['max_acc_change'].max():.3f}")
    report_lines.append(f"  Max task discrimination loss: {df['task_discrimination_loss'].max():.3f}")
    report_lines.append(f"  Max rank impact: {df['rank_impact'].max():.1f}")
    report_lines.append("")
    
    # Top heads by different criteria
    report_lines.append("=" * 80)
    report_lines.append("TOP IMPACTFUL HEADS")
    report_lines.append("=" * 80)
    
    # 1. By overall importance score
    report_lines.append(f"\n1. TOP {top_n} HEADS BY IMPORTANCE SCORE:")
    report_lines.append("-" * 50)
    report_lines.append("Rank | Head   | Import | TaskDisc | CleanAcc | CorrptAcc | CleanRnk | CorrptRnk")
    report_lines.append("-" * 50)
    
    for i, (_, row) in enumerate(df_by_importance.head(top_n).iterrows(), 1):
        report_lines.append(f"{i:4d} | {row['head_id']:6s} | {row['importance_score']:6.3f} | "
                          f"{row['task_discrimination_loss']:+7.3f} | {row['clean_accuracy_change']:+8.3f} | "
                          f"{row['corrupt_accuracy_change']:+9.3f} | {row['clean_rank_change']:+8.1f} | "
                          f"{row['corrupt_rank_change']:+9.1f}")
    
    # 2. By task discrimination impact
    report_lines.append(f"\n2. TOP {top_n} HEADS BY TASK DISCRIMINATION IMPACT:")
    report_lines.append("-" * 50)
    report_lines.append("Rank | Head   | TaskDisc | Import | CleanAcc | CorrptAcc | Description")
    report_lines.append("-" * 50)
    
    for i, (_, row) in enumerate(df_by_discrimination.head(top_n).iterrows(), 1):
        if mode == "ablate":
            desc = "Ablation disrupts task discrimination"
        else:
            desc = "Swap changes task discrimination"
            
        report_lines.append(f"{i:4d} | {row['head_id']:6s} | {row['task_discrimination_loss']:+7.3f} | "
                          f"{row['importance_score']:6.3f} | {row['clean_accuracy_change']:+8.3f} | "
                          f"{row['corrupt_accuracy_change']:+9.3f} | {desc}")
    
    # 3. By accuracy impact
    report_lines.append(f"\n3. TOP {top_n} HEADS BY ACCURACY IMPACT:")
    report_lines.append("-" * 50)
    report_lines.append("Rank | Head   | MaxAccΔ | CleanAcc | CorrptAcc | Primary Effect")
    report_lines.append("-" * 50)
    
    for i, (_, row) in enumerate(df_by_accuracy.head(top_n).iterrows(), 1):
        if abs(row['clean_accuracy_change']) > abs(row['corrupt_accuracy_change']):
            primary = f"Clean ({row['clean_accuracy_change']:+.3f})"
        else:
            primary = f"Corrupt ({row['corrupt_accuracy_change']:+.3f})"
            
        report_lines.append(f"{i:4d} | {row['head_id']:6s} | {row['max_acc_change']:7.3f} | "
                          f"{row['clean_accuracy_change']:+8.3f} | {row['corrupt_accuracy_change']:+9.3f} | {primary}")
    
    # 4. By rank impact
    report_lines.append(f"\n4. TOP {top_n} HEADS BY RANK IMPACT:")
    report_lines.append("-" * 50)
    report_lines.append("Rank | Head   | MaxRnkΔ | CleanRnk | CorrptRnk | Rank Effect")
    report_lines.append("-" * 50)
    
    for i, (_, row) in enumerate(df_by_rank.head(top_n).iterrows(), 1):
        if abs(row['clean_rank_change']) > abs(row['corrupt_rank_change']):
            rank_effect = f"Clean rank {row['clean_rank_change']:+.1f}"
        else:
            rank_effect = f"Corrupt rank {row['corrupt_rank_change']:+.1f}"
            
        report_lines.append(f"{i:4d} | {row['head_id']:6s} | {row['rank_impact']:7.1f} | "
                          f"{row['clean_rank_change']:+8.1f} | {row['corrupt_rank_change']:+9.1f} | {rank_effect}")
    
    # Layer-wise analysis
    report_lines.append("\n" + "=" * 80)
    report_lines.append("LAYER-WISE ANALYSIS")
    report_lines.append("=" * 80)
    
    layer_stats = df.groupby('layer').agg({
        'importance_score': ['mean', 'max', 'std'],
        'task_discrimination_loss': ['mean', 'max'],
        'max_acc_change': ['mean', 'max'],
        'rank_impact': ['mean', 'max']
    }).round(3)
    
    report_lines.append("\nLayer | AvgImport | MaxImport | AvgTaskDisc | MaxTaskDisc | AvgAccΔ | MaxAccΔ")
    report_lines.append("-" * 70)
    
    for layer in sorted(df['layer'].unique()):
        layer_data = df[df['layer'] == layer]
        avg_imp = layer_data['importance_score'].mean()
        max_imp = layer_data['importance_score'].max()
        avg_disc = layer_data['task_discrimination_loss'].mean()
        max_disc = layer_data['task_discrimination_loss'].max()
        avg_acc = layer_data['max_acc_change'].mean()
        max_acc = layer_data['max_acc_change'].max()
        
        report_lines.append(f"  {layer:2d}  | {avg_imp:9.3f} | {max_imp:9.3f} | "
                          f"{avg_disc:+10.3f} | {max_disc:+10.3f} | {avg_acc:7.3f} | {max_acc:7.3f}")
    
    # Phenomenon insights
    report_lines.append("\n" + "=" * 80)
    report_lines.append("PHENOMENON INSIGHTS")
    report_lines.append("=" * 80)
    
    # Find patterns
    high_impact_heads = df[df['importance_score'] > 0.1]
    early_layers = df[df['layer'] <= 2]
    late_layers = df[df['layer'] >= df['layer'].max() - 2]
    
    report_lines.append(f"\nKEY OBSERVATIONS:")
    report_lines.append(f"• {len(high_impact_heads)} heads show high impact (importance > 0.1)")
    
    if len(high_impact_heads) > 0:
        most_critical = high_impact_heads.iloc[0]
        report_lines.append(f"• Most critical head: {most_critical['head_id']} "
                          f"(importance: {most_critical['importance_score']:.3f})")
    
    early_avg = early_layers['importance_score'].mean() if len(early_layers) > 0 else 0
    late_avg = late_layers['importance_score'].mean() if len(late_layers) > 0 else 0
    
    report_lines.append(f"• Early layers (0-2) avg importance: {early_avg:.3f}")
    report_lines.append(f"• Late layers avg importance: {late_avg:.3f}")
    
    if early_avg > late_avg * 1.5:
        report_lines.append("• Pattern: Early layers show higher impact (input processing critical)")
    elif late_avg > early_avg * 1.5:
        report_lines.append("• Pattern: Late layers show higher impact (output formation critical)")
    else:
        report_lines.append("• Pattern: Impact distributed across layers")
    
    # Mode-specific insights
    if mode == "ablate":
        report_lines.append(f"\nABLATION MODE INSIGHTS:")
        report_lines.append(f"• Heads with positive task discrimination loss are critical for MSIT")
        report_lines.append(f"• Negative clean accuracy change indicates interference processing disruption")
        report_lines.append(f"• Large rank changes suggest disrupted confidence ordering")
    else:
        report_lines.append(f"\nSWAP MODE INSIGHTS:")
        report_lines.append(f"• Heads with large discrimination changes encode task-specific information")
        report_lines.append(f"• Swapping reveals which heads process clean vs corrupt differently")
        report_lines.append(f"• Asymmetric effects suggest specialized interference handling")
    
    # Save report
    if output_path is None:
        results_path_obj = Path(results_path)
        output_path = results_path_obj.parent / f"{results_path_obj.stem}_impact_report.txt"
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Head impact report saved to: {output_path}")
    print(f"Report contains analysis of {len(df)} heads with {len(high_impact_heads)} high-impact heads identified.")
    
    return output_path

def main():
    """Main entry point for head impact analysis."""
    parser = argparse.ArgumentParser(description="Analyze head impact from ablation/swap results")
    parser.add_argument("results_file", help="Path to JSON results file from head ablation sweep")
    parser.add_argument("--output", "-o", help="Output path for impact report (default: auto-generated)")
    parser.add_argument("--top_n", "-n", type=int, default=20, help="Number of top heads to show (default: 20)")
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        return 1
    
    try:
        generate_impact_report(args.results_file, args.output, args.top_n)
        return 0
    except Exception as e:
        print(f"Error generating report: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
