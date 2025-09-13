#!/usr/bin/env python3
"""
ACDC Circuit Discovery Module
----------------------------
Implements ACDC-based circuit discovery for transformer models using AutoCircuit.
Integrates with existing MSIT test infrastructure.
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dependencies with error handling
try:
    from transformer_lens import HookedTransformer
    from auto_circuit.utils.graph_utils import patchable_model, patch_mode
    from auto_circuit.data import PromptDataset, PromptDataLoader
    from auto_circuit.prune import (
        src_ablations, 
        AblationType,
        desc_prune_scores,
        prune_scores_threshold
    )
    from auto_circuit.prune_algos.ACDC import acdc_prune_scores
    from auto_circuit.types import Edge
    
    def choose_top_k_edges(scores, k):
        # Convert scores dict to edge-like objects and return top k
        if isinstance(scores, dict):
            # Sort by score and return top k edge names
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [edge_name for edge_name, score in sorted_items[:k]]
        else:
            # If scores is already a list/collection, return top k
            return list(scores)[:k]
    
    HAS_ACDC = True
except ImportError as e:
    HAS_ACDC = False
    logger.warning(f"ACDC dependencies not available: {e}")
    logger.warning("Install with: pip install auto-circuit transformer_lens")
    # Create dummy classes to avoid NameError
    class PromptDataset: pass
    class PromptDataLoader: pass
    class Edge: pass
    class AblationType: 
        TOKENWISE_MEAN_CORRUPT = "tokenwise_mean_corrupt"
    def patchable_model(*args, **kwargs): return None
    def patch_mode(*args, **kwargs): return None
    def acdc_prune_scores(*args, **kwargs): return {}
    def choose_top_k_edges(*args, **kwargs): return []
    def src_ablations(*args, **kwargs): return {}

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logger.warning("Plotting libraries not available. Install with: pip install matplotlib seaborn")


@dataclass
class ACDCConfig:
    """Configuration for ACDC circuit discovery."""
    # Core ACDC parameters
    tau: float = 1e-3  # Tolerance for faithfulness degradation
    k_edges: int = 80  # Number of top edges to keep
    faithfulness_target: str = "kl_div"  # "kl_div" or "logit_diff"
    factorized: bool = True  # Use edge-level graph (separate Q/K/V)
    
    # Model parameters
    seq_len: int = 64  # Sequence length for patchable model
    batch_size: int = 8  # Batch size for ACDC computation
    
    # Data parameters
    n_train_samples: int = 500  # Number of training samples for ACDC
    n_test_samples: int = 100   # Number of test samples for evaluation
    
    # Ablation type
    ablation_type: str = "tokenwise_mean_corrupt"  # Type of ablation to use
    
    # Output parameters
    save_edges: bool = True  # Save discovered edges to file
    save_visualizations: bool = True  # Save circuit visualizations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'tau': self.tau,
            'k_edges': self.k_edges,
            'faithfulness_target': self.faithfulness_target,
            'factorized': self.factorized,
            'seq_len': self.seq_len,
            'batch_size': self.batch_size,
            'n_train_samples': self.n_train_samples,
            'n_test_samples': self.n_test_samples,
            'ablation_type': self.ablation_type,
            'save_edges': self.save_edges,
            'save_visualizations': self.save_visualizations
        }


class ACDCCircuitDiscovery:
    """Main class for ACDC circuit discovery."""
    
    def __init__(self, model_name: str, config: ACDCConfig = None):
        """
        Initialize ACDC circuit discovery.
        
        Args:
            model_name: Name of the transformer model (e.g., 'gpt2-small')
            config: ACDC configuration parameters
        """
        if not HAS_ACDC:
            raise ImportError("ACDC dependencies not installed. Run: pip install -r requirements_acdc.txt")
        
        self.model_name = model_name
        self.config = config or ACDCConfig()
        self.tl_model = None
        self.acdc_model = None
        self.discovered_edges = []
        self.discovered_heads = []
        self.discovered_mlps = []
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load and prepare the transformer model for ACDC."""
        logger.info(f"Loading {self.model_name} for ACDC analysis...")
        
        # Determine device
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        # Load TransformerLens model with ACDC-required config
        self.tl_model = HookedTransformer.from_pretrained(
            self.model_name, 
            device=device,
            dtype=torch.float16 if device != "cpu" else torch.float32
        )
        
        # Enable required config for ACDC
        self.tl_model.cfg.use_attn_result = True
        self.tl_model.cfg.use_hook_mlp_in = True
        self.tl_model.cfg.use_split_qkv_input = True
        
        # Create patchable model for ACDC
        self.acdc_model = patchable_model(
            model=self.tl_model,
            factorized=self.config.factorized,
            seq_len=self.config.seq_len,
            separate_qkv=True,  # Required for LLM models
            device=torch.device(device) if isinstance(device, str) else device
        )
        
        logger.info(f"Model loaded on {device}")
    
    def prepare_msit_data(self, clean_prompts: List[str], corrupt_prompts: List[str], 
                         labels: Optional[List[Any]] = None) -> Tuple[Any, Any]:
        """
        Prepare MSIT data for ACDC analysis.
        
        Args:
            clean_prompts: List of clean (correct) prompts
            corrupt_prompts: List of corrupted prompts
            labels: Optional labels for the prompts
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        # Use dummy labels if none provided
        if labels is None:
            labels = list(range(len(clean_prompts)))
        
        # Tokenize all prompts first
        clean_tokens = []
        corrupt_tokens = []
        answers = []
        wrong_answers = []
        
        for clean_prompt, corrupt_prompt in zip(clean_prompts, corrupt_prompts):
            # Tokenize prompts
            clean_tok = self.tl_model.to_tokens(clean_prompt, prepend_bos=True).squeeze(0)
            corrupt_tok = self.tl_model.to_tokens(corrupt_prompt, prepend_bos=True).squeeze(0)
            
            # Focus on last N tokens only (huge search space reduction)
            target_len = self.config.seq_len
            
            # Take last target_len tokens if sequence is longer
            if len(clean_tok) > target_len:
                clean_tok = clean_tok[-target_len:]  # Last N tokens only
            elif len(clean_tok) < target_len:
                # Pad at beginning to maintain "last tokens" focus
                pad_len = target_len - len(clean_tok)
                pad_token = self.tl_model.tokenizer.pad_token_id or self.tl_model.tokenizer.eos_token_id
                clean_tok = torch.cat([torch.full((pad_len,), pad_token, device=clean_tok.device), clean_tok])
            
            if len(corrupt_tok) > target_len:
                corrupt_tok = corrupt_tok[-target_len:]  # Last N tokens only
            elif len(corrupt_tok) < target_len:
                # Pad at beginning to maintain "last tokens" focus
                pad_len = target_len - len(corrupt_tok)
                pad_token = self.tl_model.tokenizer.pad_token_id or self.tl_model.tokenizer.eos_token_id
                corrupt_tok = torch.cat([torch.full((pad_len,), pad_token, device=corrupt_tok.device), corrupt_tok])
            
            # For MSIT, we expect the answer to be the last token (the response)
            # Create dummy answer tokens - in real use you'd extract from the prompt
            answer_tok = torch.tensor([50256])  # GPT2 end token as placeholder
            wrong_answer_tok = torch.tensor([0])  # Placeholder wrong answer
            
            clean_tokens.append(clean_tok)
            corrupt_tokens.append(corrupt_tok)
            answers.append(answer_tok)
            wrong_answers.append(wrong_answer_tok)
        
        # Split data into train and test
        n_train = min(self.config.n_train_samples, len(clean_prompts))
        n_test = min(self.config.n_test_samples, len(clean_prompts) - n_train)
        
        # Training data
        train_clean_tokens = clean_tokens[:n_train]
        train_corrupt_tokens = corrupt_tokens[:n_train]
        train_answers = answers[:n_train]
        train_wrong_answers = wrong_answers[:n_train]
        
        # Test data  
        test_clean_tokens = clean_tokens[n_train:n_train + n_test]
        test_corrupt_tokens = corrupt_tokens[n_train:n_train + n_test]
        test_answers = answers[n_train:n_train + n_test]
        test_wrong_answers = wrong_answers[n_train:n_train + n_test]
        
        # Create datasets
        train_dataset = PromptDataset(
            clean_prompts=train_clean_tokens,
            corrupt_prompts=train_corrupt_tokens, 
            answers=train_answers,
            wrong_answers=train_wrong_answers
        )
        
        test_dataset = PromptDataset(
            clean_prompts=test_clean_tokens,
            corrupt_prompts=test_corrupt_tokens,
            answers=test_answers, 
            wrong_answers=test_wrong_answers
        )
        
        # Create dataloaders
        train_loader = PromptDataLoader(
            train_dataset,
            seq_len=self.config.seq_len,
            diverge_idx=0,  # Assume prompts diverge at start for MSIT
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        test_loader = PromptDataLoader(
            test_dataset,
            seq_len=self.config.seq_len,
            diverge_idx=0,  # Assume prompts diverge at start for MSIT
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        logger.info(f"Prepared ACDC data: {n_train} train, {n_test} test samples")
        return train_loader, test_loader
    
    def discover_circuit(self, train_loader: Any) -> List[Any]:
        """
        Run ACDC to discover the circuit.
        
        Args:
            train_loader: Training data for circuit discovery
            
        Returns:
            List of discovered edges
        """
        logger.info("Starting ACDC circuit discovery...")
        logger.info(f"Parameters: tau={self.config.tau}, k_edges={self.config.k_edges}")
        
        # Compute ACDC prune scores  
        # Convert tau to tao_exps and tao_bases for ACDC
        import math
        tau_log = math.log10(self.config.tau)
        tao_exp = int(tau_log)
        tao_base = int(10**(tau_log - tao_exp))
        
        scores = acdc_prune_scores(
            model=self.acdc_model,
            dataloader=train_loader,
            official_edges=None,
            tao_exps=[tao_exp],
            tao_bases=[tao_base],
            faithfulness_target=self.config.faithfulness_target,
        )
        
        # Select top edges
        candidate_edges = choose_top_k_edges(scores, k=self.config.k_edges)
        
        self.discovered_edges = candidate_edges
        logger.info(f"Discovered {len(candidate_edges)} edges")
        
        # Extract heads and MLPs from edges
        self._extract_components_from_edges(candidate_edges)
        
        return candidate_edges
    
    def _extract_components_from_edges(self, edges: List[Any]):
        """Extract attention heads and MLPs from discovered edges."""
        heads = set()
        mlps = set()
        
        # Handle different edge formats
        if isinstance(edges, list) and edges:
            if isinstance(edges[0], str):
                # Edge names as strings, parse them
                for edge_name in edges:
                    # Parse edge names like "blocks.2.attn.hook_result -> blocks.3.mlp.hook_pre"
                    if 'attn' in edge_name and 'hook_result' in edge_name:
                        # Extract layer number from attention edges
                        import re
                        match = re.search(r'blocks\.(\d+)\.attn', edge_name)
                        if match:
                            layer = int(match.group(1))
                            # For now, assume all heads in the layer (we'll need better parsing later)
                            for head in range(12):  # GPT2 has 12 heads per layer
                                heads.add((layer, head))
                    
                    if 'mlp' in edge_name:
                        match = re.search(r'blocks\.(\d+)\.mlp', edge_name)
                        if match:
                            layer = int(match.group(1))
                            mlps.add(layer)
            else:
                # Try to handle edge objects
                for edge in edges:
                    if hasattr(edge, 'src') and hasattr(edge, 'dest'):
                        # Check source
                        if hasattr(edge.src, 'head_idx') and edge.src.head_idx is not None:
                            heads.add((edge.src.layer, edge.src.head_idx))
                        elif hasattr(edge.src, 'name') and 'MLP' in str(edge.src.name):
                            mlps.add(edge.src.layer)
                        
                        # Check destination  
                        if hasattr(edge.dest, 'head_idx') and edge.dest.head_idx is not None:
                            heads.add((edge.dest.layer, edge.dest.head_idx))
                        elif hasattr(edge.dest, 'name') and 'MLP' in str(edge.dest.name):
                            mlps.add(edge.dest.layer)
        
        self.discovered_heads = sorted(list(heads))
        self.discovered_mlps = sorted(list(mlps))
        
        logger.info(f"Extracted {len(self.discovered_heads)} heads, {len(self.discovered_mlps)} MLPs")
    
    def evaluate_circuit(self, test_loader: Any, 
                        metric_fn: Optional[callable] = None) -> Dict[str, float]:
        """
        Evaluate the discovered circuit.
        
        Args:
            test_loader: Test data for evaluation
            metric_fn: Optional custom metric function
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.discovered_edges:
            raise ValueError("No circuit discovered yet. Run discover_circuit() first.")
        
        logger.info("Evaluating discovered circuit...")
        
        # Prepare ablations
        ablation_type_map = {
            "tokenwise_mean_corrupt": AblationType.TOKENWISE_MEAN_CORRUPT,
            "zero": AblationType.ZERO,
            "mean_corrupt": AblationType.MEAN_CORRUPT
        }
        
        ablation_type = ablation_type_map.get(
            self.config.ablation_type, 
            AblationType.TOKENWISE_MEAN_CORRUPT
        )
        
        ablations = src_ablations(self.acdc_model, test_loader, ablation_type)
        
        # Evaluate with and without circuit
        results = {}
        
        # Get test tokens
        test_batch = next(iter(test_loader))
        clean_tokens = test_batch.clean
        corrupt_tokens = test_batch.corrupt
        
        # Baseline (no intervention)
        with torch.no_grad():
            baseline_logits = self.acdc_model(clean_tokens)
        
        # Circuit intervention (patch only discovered edges)
        with patch_mode(self.acdc_model, ablations, self.discovered_edges):
            with torch.no_grad():
                circuit_logits = self.acdc_model(clean_tokens)
        
        # Full ablation (for comparison)
        all_edges = list(ablations.keys())  # All available edges
        with patch_mode(self.acdc_model, ablations, all_edges):
            with torch.no_grad():
                full_ablation_logits = self.acdc_model(clean_tokens)
        
        # Compute metrics
        if metric_fn:
            results['baseline_metric'] = metric_fn(baseline_logits).item()
            results['circuit_metric'] = metric_fn(circuit_logits).item()
            results['full_ablation_metric'] = metric_fn(full_ablation_logits).item()
        
        # KL divergence metrics
        baseline_probs = torch.softmax(baseline_logits, dim=-1)
        circuit_probs = torch.softmax(circuit_logits, dim=-1)
        full_ablation_probs = torch.softmax(full_ablation_logits, dim=-1)
        
        circuit_kl = torch.nn.functional.kl_div(
            torch.log(circuit_probs + 1e-10), baseline_probs, reduction='batchmean'
        ).item()
        
        full_kl = torch.nn.functional.kl_div(
            torch.log(full_ablation_probs + 1e-10), baseline_probs, reduction='batchmean'
        ).item()
        
        results.update({
            'circuit_kl_divergence': circuit_kl,
            'full_ablation_kl_divergence': full_kl,
            'circuit_faithfulness': 1.0 - (circuit_kl / max(full_kl, 1e-10)),
            'n_edges': len(self.discovered_edges),
            'n_heads': len(self.discovered_heads),
            'n_mlps': len(self.discovered_mlps)
        })
        
        logger.info(f"Circuit evaluation: KL={circuit_kl:.4f}, Faithfulness={results['circuit_faithfulness']:.3f}")
        
        return results
    
    def save_circuit(self, output_dir: Path, filename_prefix: str = "circuit"):
        """Save discovered circuit to files."""
        if not self.discovered_edges:
            logger.warning("No circuit to save. Run discover_circuit() first.")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save circuit data
        circuit_data = {
            'model_name': self.model_name,
            'config': self.config.to_dict(),
            'n_edges': len(self.discovered_edges),
            'n_heads': len(self.discovered_heads),
            'n_mlps': len(self.discovered_mlps),
            'heads': self.discovered_heads,
            'mlps': self.discovered_mlps,
            'edges': [self._edge_to_dict(edge) for edge in self.discovered_edges]
        }
        
        circuit_file = output_dir / f"{filename_prefix}_data.json"
        with open(circuit_file, 'w') as f:
            json.dump(circuit_data, f, indent=2)
        
        logger.info(f"Circuit saved to {circuit_file}")
    
    def _edge_to_dict(self, edge: Any) -> Dict[str, Any]:
        """Convert edge object to dictionary for serialization."""
        return {
            'src_name': str(edge.src.name) if hasattr(edge.src, 'name') else str(edge.src),
            'dest_name': str(edge.dest.name) if hasattr(edge.dest, 'name') else str(edge.dest),
            'src_layer': getattr(edge.src, 'layer', None),
            'dest_layer': getattr(edge.dest, 'layer', None),
            'src_head_idx': getattr(edge.src, 'head_idx', None),
            'dest_head_idx': getattr(edge.dest, 'head_idx', None),
        }
    
    def run_head_ablation_analysis(self, test_tokens: torch.Tensor) -> Dict[str, Any]:
        """
        Run targeted head ablation analysis on discovered heads.
        
        Args:
            test_tokens: Test tokens for ablation
            
        Returns:
            Dictionary containing ablation results
        """
        if not self.discovered_heads:
            logger.warning("No heads discovered. Run discover_circuit() first.")
            return {}
        
        logger.info(f"Running head ablation analysis on {len(self.discovered_heads)} heads...")
        
        results = {
            'individual_head_effects': {},
            'collective_effect': None,
            'head_rankings': []
        }
        
        # Get baseline logits
        with torch.no_grad():
            baseline_logits = self.tl_model(test_tokens)
        
        # Individual head ablations
        for layer, head in self.discovered_heads:
            def ablate_head(act, hook):
                act[:, :, head, :] = 0.0
                return act
            
            hook_name = f"blocks.{layer}.attn.hook_result"
            with torch.no_grad():
                ablated_logits = self.tl_model.run_with_hooks(
                    test_tokens,
                    fwd_hooks=[(hook_name, ablate_head)]
                )
            
            # Compute effect size (KL divergence)
            baseline_probs = torch.softmax(baseline_logits, dim=-1)
            ablated_probs = torch.softmax(ablated_logits, dim=-1)
            
            kl_effect = torch.nn.functional.kl_div(
                torch.log(ablated_probs + 1e-10), baseline_probs, reduction='batchmean'
            ).item()
            
            results['individual_head_effects'][f"L{layer}H{head}"] = kl_effect
        
        # Collective ablation (all discovered heads)
        def ablate_all_heads(act, hook):
            layer = int(hook.name.split('.')[1])
            for _, head in self.discovered_heads:
                if _ == layer:  # Only ablate heads in this layer
                    act[:, :, head, :] = 0.0
            return act
        
        # Build hooks for all layers with discovered heads
        hook_layers = set(layer for layer, _ in self.discovered_heads)
        hooks = [(f"blocks.{layer}.attn.hook_result", ablate_all_heads) for layer in hook_layers]
        
        with torch.no_grad():
            collective_ablated_logits = self.tl_model.run_with_hooks(test_tokens, fwd_hooks=hooks)
        
        collective_probs = torch.softmax(collective_ablated_logits, dim=-1)
        collective_kl = torch.nn.functional.kl_div(
            torch.log(collective_probs + 1e-10), baseline_probs, reduction='batchmean'
        ).item()
        
        results['collective_effect'] = collective_kl
        
        # Rank heads by effect size
        head_effects = [(head, effect) for head, effect in results['individual_head_effects'].items()]
        head_effects.sort(key=lambda x: x[1], reverse=True)
        results['head_rankings'] = head_effects
        
        logger.info(f"Head ablation analysis complete. Collective effect: {collective_kl:.4f}")
        
        return results


def create_msit_corrupted_data(clean_prompts: List[str], 
                              corruption_type: str = "shuffle_positions") -> List[str]:
    """
    Create corrupted versions of MSIT prompts for ACDC analysis.
    
    Args:
        clean_prompts: List of clean MSIT prompts
        corruption_type: Type of corruption to apply
        
    Returns:
        List of corrupted prompts
    """
    corrupt_prompts = []
    
    for prompt in clean_prompts:
        if corruption_type == "shuffle_positions":
            # Simple corruption: shuffle digit positions in MSIT stimuli
            lines = prompt.split('\n')
            corrupted_lines = []
            
            for line in lines:
                if line.strip() and any(c.isdigit() for c in line):
                    # Extract digits and shuffle them
                    import re
                    digits = re.findall(r'\d', line)
                    if len(digits) > 1:
                        import random
                        shuffled_digits = digits.copy()
                        random.shuffle(shuffled_digits)
                        
                        # Replace digits with shuffled versions
                        corrupted_line = line
                        digit_idx = 0
                        for i, char in enumerate(line):
                            if char.isdigit():
                                corrupted_line = corrupted_line[:i] + shuffled_digits[digit_idx] + corrupted_line[i+1:]
                                digit_idx += 1
                        
                        corrupted_lines.append(corrupted_line)
                    else:
                        corrupted_lines.append(line)
                else:
                    corrupted_lines.append(line)
            
            corrupt_prompts.append('\n'.join(corrupted_lines))
        
        else:
            # Default: just return the same prompt (no corruption)
            corrupt_prompts.append(prompt)
    
    return corrupt_prompts


# Example usage function
def run_msit_acdc_analysis(clean_prompts: List[str], 
                          model_name: str = "gpt2-small",
                          config: ACDCConfig = None,
                          output_dir: str = "acdc_results") -> Dict[str, Any]:
    """
    Complete MSIT ACDC analysis pipeline.
    
    Args:
        clean_prompts: List of clean MSIT prompts
        model_name: Name of the model to analyze
        config: ACDC configuration
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing all analysis results
    """
    if not HAS_ACDC:
        raise ImportError("ACDC dependencies not installed. Run: pip install -r requirements_acdc.txt")
    
    # Initialize ACDC
    acdc = ACDCCircuitDiscovery(model_name, config)
    
    # Create corrupted data
    corrupt_prompts = create_msit_corrupted_data(clean_prompts)
    
    # Prepare data
    train_loader, test_loader = acdc.prepare_msit_data(clean_prompts, corrupt_prompts)
    
    # Discover circuit
    edges = acdc.discover_circuit(train_loader)
    
    # Evaluate circuit
    eval_results = acdc.evaluate_circuit(test_loader)
    
    # Run head ablation analysis
    test_batch = next(iter(test_loader))
    ablation_results = acdc.run_head_ablation_analysis(test_batch.clean)
    
    # Save results
    output_path = Path(output_dir)
    acdc.save_circuit(output_path)
    
    # Combine all results
    results = {
        'circuit_discovery': {
            'n_edges': len(edges),
            'heads': acdc.discovered_heads,
            'mlps': acdc.discovered_mlps
        },
        'evaluation': eval_results,
        'ablation_analysis': ablation_results,
        'config': config.to_dict() if config else ACDCConfig().to_dict()
    }
    
    # Save combined results
    results_file = output_path / "acdc_analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results
