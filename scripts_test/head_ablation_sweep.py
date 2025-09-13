#!/usr/bin/env python3
"""
Fast head ablation sweep for MSIT analysis.
Tests each attention head individually to find task-critical components.
"""

import torch
import logging
import numpy as np
from typing import List, Dict, Tuple, Any
from pathlib import Path
import json
from datetime import datetime
from transformer_lens import HookedTransformer
from local_model_handler import LocalModelHandler, get_local_model_digit_logits

logger = logging.getLogger(__name__)

class HeadAblationSweep:
    """Fast attention head ablation analysis for transformer models."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.model = None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    def load_model(self):
        """Load TransformerLens model for head ablation."""
        logger.info(f"Loading {self.model_name} for head ablation...")
        
        self.model = HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device,
            dtype=torch.float16 if self.device.type != "cpu" else torch.float32
        )
        
        # CRITICAL: Enable attention result hooks for ablation
        self.model.cfg.use_attn_result = True
        
        logger.info(f"Model loaded on {self.device}")
        
    def create_msit_test_data(self, n_samples: int = 10, with_examples: bool = True) -> Tuple[List[str], List[str]]:
        """Create MSIT test data with few-shot examples for better model understanding."""
        clean_prompts = []
        corrupt_prompts = []
        
        # Few-shot examples to help model understand task format
        examples_text = ""
        if with_examples:
            examples_text = """Here are some examples of the Multi-source Interference Task (MSIT):

Example 1:
Stimulus: 1 0 0
Target position: 1
Congruent condition
Answer: 1

Example 2:
Stimulus: 0 2 0  
Target position: 2
Congruent condition
Answer: 2

Example 3:
Stimulus: 0 0 3
Target position: 3
Congruent condition
Answer: 3

Now solve these:

"""
        
        # MSIT stimuli using proper stim_types classification
        # stim_type 0 = Congruent (clean/easy)
        # stim_type 2 = Incongruent (corrupted/interference)
        
        clean_stimuli = [
            # stim_type 0: Congruent - target digit in correct position (easy)
            ("1 0 0", 1, "1"),  # Target "1" in position 1
            ("0 2 0", 2, "2"),  # Target "2" in position 2  
            ("0 0 3", 3, "3"),  # Target "3" in position 3
        ]
        
        corrupt_stimuli = [
            # stim_type 2: Incongruent - target digit in wrong position (interference)
            ("2 0 0", 1, "2"),  # Target "2" but answer position 1 
            ("0 1 0", 2, "1"),  # Target "1" but answer position 2
            ("0 0 1", 3, "1"),  # Target "1" but answer position 3
        ]
        
        for i in range(n_samples):
            # Generate congruent (clean) stimulus
            clean_stimulus, clean_answer = self.generate_msit_stimulus(stim_type=0)
            clean_prompt = f"{examples_text}Stimulus: {clean_stimulus}\nAnswer:"
            clean_prompts.append(clean_prompt)
            
            # Generate incongruent (corrupt) stimulus  
            corrupt_stimulus, corrupt_answer = self.generate_msit_stimulus(stim_type=2)
            corrupt_prompt = f"{examples_text}Stimulus: {corrupt_stimulus}\nAnswer:"
            corrupt_prompts.append(corrupt_prompt)
            
            # Debug: Print first few stimuli to verify they're different
            if i < 2:
                logger.info(f"DEBUG Sample {i}: Clean stimulus='{clean_stimulus}' answer={clean_answer}")
                logger.info(f"DEBUG Sample {i}: Corrupt stimulus='{corrupt_stimulus}' answer={corrupt_answer}")
            
        return clean_prompts, corrupt_prompts
    
    def generate_msit_stimulus(self, stim_type: int) -> Tuple[str, str]:
        """Generate MSIT stimulus based on stim_type.
        stim_type 0: Congruent (target digit in correct position)
        stim_type 2: Incongruent (target digit in wrong position - interference)
        """
        import random
        
        if stim_type == 0:  # Congruent
            # Target digit appears in the position it should be identified from
            target_pos = random.randint(1, 3)  # Position 1, 2, or 3
            target_digit = random.choice([1, 2, 3])  # Use digits 1, 2, 3 for simplicity
            
            # Create stimulus with target digit in correct position, zeros elsewhere
            stimulus = ["0", "0", "0"]
            stimulus[target_pos - 1] = str(target_digit)  # Convert to 0-indexed
            stimulus_str = " ".join(stimulus)
            answer = str(target_digit)
            
        else:  # stim_type == 2, Incongruent
            # Target digit appears in wrong position, creating interference
            target_digit = random.choice([1, 2, 3])
            correct_pos = target_digit  # Where it should be (1-indexed)
            
            # Place target digit in a different position
            available_positions = [1, 2, 3]
            available_positions.remove(correct_pos)
            wrong_pos = random.choice(available_positions)
            
            # Create stimulus with target digit in wrong position
            stimulus = ["0", "0", "0"]
            stimulus[wrong_pos - 1] = str(target_digit)  # Convert to 0-indexed
            stimulus_str = " ".join(stimulus)
            
            # Answer is still the target digit, but position creates interference
            answer = str(target_digit)
            
        return stimulus_str, answer
        
    
    def extract_msit_info(self, prompt: str) -> Tuple[str, List[str]]:
        """Extract correct answer and candidate digits from MSIT prompt."""
        import re
        
        # Extract the correct answer from the stimulus (not from Answer: which doesn't exist yet)
        stimulus_match = re.search(r"Stimulus: ([\d\s]+)", prompt)
        correct_answer = None
        candidate_digits = ['1', '2', '3']  # Default MSIT candidates
        
        if stimulus_match:
            stimulus_str = stimulus_match.group(1).strip()
            all_digits = re.findall(r"\d", stimulus_str)
            
            # For MSIT: the correct answer is the non-zero digit in the stimulus
            non_zero_digits = [d for d in all_digits if d != '0']
            if non_zero_digits:
                # In MSIT, there should be exactly one non-zero digit, which is the target/answer
                correct_answer = non_zero_digits[0]
            
            # Ensure 1,2,3 are always included as candidates 
            candidate_digits = list(set(['1', '2', '3'] + non_zero_digits))
        
        return correct_answer, candidate_digits
    
    def get_digit_logits_info_transformer_lens(self, prompt: str, logits_tensor: torch.Tensor = None) -> Dict[str, Any]:
        """
        Get logits information for digits 0-9 compatible with TransformerLens models.
        
        Args:
            prompt: Input prompt
            logits_tensor: Optional pre-computed logits tensor (for ablation experiments)
            
        Returns:
            Dictionary containing logits, ranks, and probabilities for digits 0-9
        """
        try:
            if logits_tensor is None:
                # Generate logits using the model
                tokens = self.model.to_tokens(prompt, prepend_bos=True)
                with torch.no_grad():
                    model_logits = self.model(tokens)
                last_logits = model_logits[0, -1, :]
            else:
                # Use provided logits (for ablation experiments)
                last_logits = logits_tensor
                
            # Get token IDs for digits 0-9
            digit_tokens = {}
            digit_logits = {}
            
            for digit in range(10):
                digit_str = str(digit)
                # Try different tokenizations of the digit
                possible_tokens = [
                    self.model.to_tokens(digit_str, prepend_bos=False)[0, 0].item(),
                ]
                
                # Use the first valid token
                token_id = possible_tokens[0]
                if token_id is not None:
                    digit_tokens[digit] = token_id
                    digit_logits[str(digit)] = float(last_logits[token_id].cpu())
            
            # Calculate global ranks
            all_logits = last_logits.cpu().numpy()
            sorted_indices = np.argsort(-all_logits)  # Sort in descending order
            rank_map = {idx: rank for rank, idx in enumerate(sorted_indices)}
            
            global_ranks = {}
            for digit, token_id in digit_tokens.items():
                if token_id in rank_map:
                    global_ranks[str(digit)] = int(rank_map[token_id])  # 0-indexed rank
                else:
                    global_ranks[str(digit)] = -1
            
            # Calculate softmax probabilities among digits 0-9
            digit_logits_array = np.array([digit_logits.get(str(i), -np.inf) for i in range(10)])
            valid_mask = digit_logits_array != -np.inf
            if np.any(valid_mask):
                softmax_probs = np.full(10, 0.0)
                valid_logits = digit_logits_array[valid_mask]
                valid_softmax = np.exp(valid_logits - np.max(valid_logits))
                valid_softmax = valid_softmax / np.sum(valid_softmax)
                softmax_probs[valid_mask] = valid_softmax
            else:
                softmax_probs = np.full(10, 0.1)  # Uniform if no valid tokens
            
            return {
                "digit_logits": digit_logits,
                "digit_global_ranks": global_ranks,
                "digit_softmax_probs": {str(k): float(softmax_probs[k]) for k in range(10)},
                "digit_token_ids": {str(k): v for k, v in digit_tokens.items()}
            }
            
        except Exception as e:
            logger.error(f"Error getting digit logits: {e}")
            return {"error": f"Failed to get digit logits: {str(e)}"}
    
    def measure_digit_rankings(self, prompts: List[str], condition_name: str = "baseline") -> Dict[str, Any]:
        """Measure digit ranking performance using integrated logits function."""
        results = {
            'condition': condition_name,
            'prompt_results': [],
            'summary_stats': {}
        }
        
        for i, prompt in enumerate(prompts):
            # Use our TransformerLens-compatible logits function
            logits_info = self.get_digit_logits_info_transformer_lens(prompt)
            
            if "error" in logits_info:
                logger.warning(f"Error getting logits for prompt {i}: {logits_info['error']}")
                continue
                
            # Extract data from integrated function
            digit_logits = {k: float(v) for k, v in logits_info["digit_logits"].items()}
            digit_probs = {k: float(v) for k, v in logits_info["digit_softmax_probs"].items()}
            
            # Convert global ranks to local digit ranks (1-indexed like original code)
            global_ranks = logits_info["digit_global_ranks"]
            sorted_digits = sorted(digit_probs.items(), key=lambda x: x[1], reverse=True)
            digit_ranks = {digit: rank + 1 for rank, (digit, _) in enumerate(sorted_digits)}
            
            # Extract MSIT task information
            correct_answer, stimulus_digits = self.extract_msit_info(prompt)
            
            # Calculate candidate-specific metrics
            candidate_performance = {}
            if correct_answer and stimulus_digits:
                # Rank within stimulus digits only
                stimulus_probs = {d: digit_probs[d] for d in stimulus_digits if d in digit_probs}
                if stimulus_probs:
                    sorted_stimulus = sorted(stimulus_probs.items(), key=lambda x: x[1], reverse=True)
                    stimulus_ranks = {digit: rank + 1 for rank, (digit, _) in enumerate(sorted_stimulus)}
                    
                    candidate_performance = {
                        'correct_answer': correct_answer,
                        'stimulus_digits': stimulus_digits,
                        'correct_answer_global_rank': global_ranks.get(correct_answer, 50000) + 1,  # Convert to 1-indexed
                        'correct_answer_stimulus_rank': stimulus_ranks.get(correct_answer, len(stimulus_digits) + 1),
                        'stimulus_ranks': stimulus_ranks,
                        'is_correct_top_choice': stimulus_ranks.get(correct_answer, 999) == 1
                    }
            
            prompt_result = {
                'prompt_index': i,
                'prompt': prompt,
                'digit_logits': digit_logits,
                'digit_probabilities': digit_probs,
                'digit_ranks': digit_ranks,
                'digit_global_ranks': global_ranks,
                'candidate_performance': candidate_performance
            }
            
            results['prompt_results'].append(prompt_result)
        
        # Calculate summary statistics
        if len(results['prompt_results']) > 0:
            correct_top_count = sum(1 for r in results['prompt_results'] 
                                   if r['candidate_performance'].get('is_correct_top_choice', False))
            
            avg_correct_rank = sum(r['candidate_performance'].get('correct_answer_stimulus_rank', 999) 
                                  for r in results['prompt_results']) / len(results['prompt_results'])
            
            results['summary_stats'] = {
                'accuracy': correct_top_count / len(results['prompt_results']),
                'avg_correct_rank': avg_correct_rank,
                'total_prompts': len(results['prompt_results'])
            }
        else:
            results['summary_stats'] = {
                'accuracy': 0.0,
                'avg_correct_rank': 999.0,
                'total_prompts': 0
            }
        
        return results
    
    def swap_head_ranking(self, layer: int, head: int, clean_prompts: List[str], corrupt_prompts: List[str]) -> Dict[str, Any]:
        """Swap specific head activations between clean and corrupt prompts and measure performance changes."""
        if len(clean_prompts) != len(corrupt_prompts):
            raise ValueError("Clean and corrupt prompt lists must have the same length")
            
        hook_called = False
        swap_activations = {}  # Store activations to swap
        
        def capture_and_swap_hook(act, hook, prompt_type):
            nonlocal hook_called, swap_activations
            hook_called = True
            
            # Store or swap activations based on prompt type
            if prompt_type == 'clean_capture':
                # Store clean activations for later use in corrupt
                if head < act.shape[2]:
                    swap_activations['clean_to_corrupt'] = act[:, :, head, :].clone()
                    swap_activations['clean_seq_len'] = act.shape[1]
            elif prompt_type == 'corrupt_capture':
                # Store corrupt activations for later use in clean
                if head < act.shape[2]:
                    swap_activations['corrupt_to_clean'] = act[:, :, head, :].clone()
                    swap_activations['corrupt_seq_len'] = act.shape[1]
            elif prompt_type == 'clean_apply':
                # Apply corrupt activations to clean prompt (handle sequence length mismatch)
                if head < act.shape[2] and 'corrupt_to_clean' in swap_activations:
                    stored_activation = swap_activations['corrupt_to_clean']
                    current_seq_len = act.shape[1]
                    stored_seq_len = stored_activation.shape[1]
                    
                    if current_seq_len == stored_seq_len:
                        # Direct swap when lengths match
                        act[:, :, head, :] = stored_activation
                    else:
                        # Handle length mismatch by taking the minimum length
                        min_len = min(current_seq_len, stored_seq_len)
                        act[:, :min_len, head, :] = stored_activation[:, :min_len, :]
                        logger.warning(f"Sequence length mismatch in swap: clean={current_seq_len}, corrupt={stored_seq_len}, using min={min_len}")
            elif prompt_type == 'corrupt_apply':
                # Apply clean activations to corrupt prompt (handle sequence length mismatch)
                if head < act.shape[2] and 'clean_to_corrupt' in swap_activations:
                    stored_activation = swap_activations['clean_to_corrupt']
                    current_seq_len = act.shape[1]
                    stored_seq_len = stored_activation.shape[1]
                    
                    if current_seq_len == stored_seq_len:
                        # Direct swap when lengths match
                        act[:, :, head, :] = stored_activation
                    else:
                        # Handle length mismatch by taking the minimum length
                        min_len = min(current_seq_len, stored_seq_len)
                        act[:, :min_len, head, :] = stored_activation[:, :min_len, :]
                        logger.warning(f"Sequence length mismatch in swap: corrupt={current_seq_len}, clean={stored_seq_len}, using min={min_len}")
                    
            return act
            
        hook_name = f"blocks.{layer}.attn.hook_result"
        
        swapped_clean_results = []
        swapped_corrupt_results = []
        
        with torch.no_grad():
            # First pass: capture activations from both clean and corrupt
            for i, (clean_prompt, corrupt_prompt) in enumerate(zip(clean_prompts, corrupt_prompts)):
                # Reset swap storage for each pair
                swap_activations = {}
                
                # Create separate hook functions to avoid lambda closure issues
                def clean_capture_hook(act, hook):
                    return capture_and_swap_hook(act, hook, 'clean_capture')
                    
                def corrupt_capture_hook(act, hook):
                    return capture_and_swap_hook(act, hook, 'corrupt_capture')
                    
                def clean_apply_hook(act, hook):
                    return capture_and_swap_hook(act, hook, 'clean_apply')
                    
                def corrupt_apply_hook(act, hook):
                    return capture_and_swap_hook(act, hook, 'corrupt_apply')
                
                # Capture clean activation
                clean_tokens = self.model.to_tokens(clean_prompt, prepend_bos=True)
                _ = self.model.run_with_hooks(
                    clean_tokens,
                    fwd_hooks=[(hook_name, clean_capture_hook)]
                )
                
                # Capture corrupt activation
                corrupt_tokens = self.model.to_tokens(corrupt_prompt, prepend_bos=True)
                _ = self.model.run_with_hooks(
                    corrupt_tokens,
                    fwd_hooks=[(hook_name, corrupt_capture_hook)]
                )
                
                # Verify activations were captured before proceeding
                if 'clean_to_corrupt' not in swap_activations or 'corrupt_to_clean' not in swap_activations:
                    logger.error(f"Failed to capture activations for L{layer}H{head}, prompt pair {i}")
                    continue
                
                # Apply swapped activations and get results
                # Clean prompt with corrupt head activation
                clean_swapped_logits = self.model.run_with_hooks(
                    clean_tokens,
                    fwd_hooks=[(hook_name, clean_apply_hook)]
                )
                
                # Corrupt prompt with clean head activation  
                corrupt_swapped_logits = self.model.run_with_hooks(
                    corrupt_tokens,
                    fwd_hooks=[(hook_name, corrupt_apply_hook)]
                )
                
                # Process clean swapped results (primary metric)
                clean_last_logits = clean_swapped_logits[0, -1, :]
                clean_logits_info = self.get_digit_logits_info_transformer_lens(clean_prompt, clean_last_logits)
                
                if "error" not in clean_logits_info:
                    clean_digit_logits = {k: float(v) for k, v in clean_logits_info["digit_logits"].items()}
                    clean_digit_probs = {k: float(v) for k, v in clean_logits_info["digit_softmax_probs"].items()}
                    clean_global_ranks = clean_logits_info["digit_global_ranks"]
                    
                    clean_sorted_digits = sorted(clean_digit_probs.items(), key=lambda x: x[1], reverse=True)
                    clean_digit_ranks = {digit: rank + 1 for rank, (digit, _) in enumerate(clean_sorted_digits)}
                    
                    clean_correct_answer, clean_stimulus_digits = self.extract_msit_info(clean_prompt)
                    clean_candidate_performance = {}
                    if clean_correct_answer and clean_stimulus_digits:
                        clean_stimulus_probs = {d: clean_digit_probs[d] for d in clean_stimulus_digits if d in clean_digit_probs}
                        if clean_stimulus_probs:
                            clean_sorted_stimulus = sorted(clean_stimulus_probs.items(), key=lambda x: x[1], reverse=True)
                            clean_stimulus_ranks = {digit: rank + 1 for rank, (digit, _) in enumerate(clean_sorted_stimulus)}
                            
                            clean_candidate_performance = {
                                'correct_answer': clean_correct_answer,
                                'stimulus_digits': clean_stimulus_digits,
                                'correct_answer_global_rank': clean_global_ranks.get(clean_correct_answer, 50000) + 1,
                                'correct_answer_stimulus_rank': clean_stimulus_ranks.get(clean_correct_answer, len(clean_stimulus_digits) + 1),
                                'stimulus_ranks': clean_stimulus_ranks,
                                'is_correct_top_choice': clean_stimulus_ranks.get(clean_correct_answer, 999) == 1
                            }
                    
                    swapped_clean_results.append({
                        'prompt_index': i,
                        'digit_logits': clean_digit_logits,
                        'digit_probabilities': clean_digit_probs,
                        'digit_ranks': clean_digit_ranks,
                        'digit_global_ranks': clean_global_ranks,
                        'candidate_performance': clean_candidate_performance
                    })
                
                # Process corrupt swapped results (secondary analysis)
                corrupt_last_logits = corrupt_swapped_logits[0, -1, :]
                corrupt_logits_info = self.get_digit_logits_info_transformer_lens(corrupt_prompt, corrupt_last_logits)
                
                if "error" not in corrupt_logits_info:
                    corrupt_digit_logits = {k: float(v) for k, v in corrupt_logits_info["digit_logits"].items()}
                    corrupt_digit_probs = {k: float(v) for k, v in corrupt_logits_info["digit_softmax_probs"].items()}
                    corrupt_global_ranks = corrupt_logits_info["digit_global_ranks"]
                    
                    corrupt_sorted_digits = sorted(corrupt_digit_probs.items(), key=lambda x: x[1], reverse=True)
                    corrupt_digit_ranks = {digit: rank + 1 for rank, (digit, _) in enumerate(corrupt_sorted_digits)}
                    
                    corrupt_correct_answer, corrupt_stimulus_digits = self.extract_msit_info(corrupt_prompt)
                    corrupt_candidate_performance = {}
                    if corrupt_correct_answer and corrupt_stimulus_digits:
                        corrupt_stimulus_probs = {d: corrupt_digit_probs[d] for d in corrupt_stimulus_digits if d in corrupt_digit_probs}
                        if corrupt_stimulus_probs:
                            corrupt_sorted_stimulus = sorted(corrupt_stimulus_probs.items(), key=lambda x: x[1], reverse=True)
                            corrupt_stimulus_ranks = {digit: rank + 1 for rank, (digit, _) in enumerate(corrupt_sorted_stimulus)}
                            
                            corrupt_candidate_performance = {
                                'correct_answer': corrupt_correct_answer,
                                'stimulus_digits': corrupt_stimulus_digits,
                                'correct_answer_global_rank': corrupt_global_ranks.get(corrupt_correct_answer, 50000) + 1,
                                'correct_answer_stimulus_rank': corrupt_stimulus_ranks.get(corrupt_correct_answer, len(corrupt_stimulus_digits) + 1),
                                'stimulus_ranks': corrupt_stimulus_ranks,
                                'is_correct_top_choice': corrupt_stimulus_ranks.get(corrupt_correct_answer, 999) == 1
                            }
                    
                    swapped_corrupt_results.append({
                        'prompt_index': i,
                        'digit_logits': corrupt_digit_logits,
                        'digit_probabilities': corrupt_digit_probs,
                        'digit_ranks': corrupt_digit_ranks,
                        'digit_global_ranks': corrupt_global_ranks,
                        'candidate_performance': corrupt_candidate_performance
                    })
        
        # Calculate summary statistics (handle empty results)
        if len(swapped_clean_results) > 0:
            clean_correct_top_count = sum(1 for r in swapped_clean_results 
                                         if r['candidate_performance'].get('is_correct_top_choice', False))
            clean_avg_correct_rank = sum(r['candidate_performance'].get('correct_answer_stimulus_rank', 999) 
                                        for r in swapped_clean_results) / len(swapped_clean_results)
            clean_accuracy = clean_correct_top_count / len(swapped_clean_results)
        else:
            clean_correct_top_count = 0
            clean_avg_correct_rank = 999.0
            clean_accuracy = 0.0
            logger.warning(f"No valid clean swapped results for L{layer}H{head}")
        
        if len(swapped_corrupt_results) > 0:
            corrupt_correct_top_count = sum(1 for r in swapped_corrupt_results 
                                           if r['candidate_performance'].get('is_correct_top_choice', False))
            corrupt_avg_correct_rank = sum(r['candidate_performance'].get('correct_answer_stimulus_rank', 999) 
                                          for r in swapped_corrupt_results) / len(swapped_corrupt_results)
            corrupt_accuracy = corrupt_correct_top_count / len(swapped_corrupt_results)
        else:
            corrupt_correct_top_count = 0
            corrupt_avg_correct_rank = 999.0
            corrupt_accuracy = 0.0
            logger.warning(f"No valid corrupt swapped results for L{layer}H{head}")
        
        return {
            'clean_accuracy': clean_accuracy,
            'clean_avg_correct_rank': clean_avg_correct_rank,
            'clean_detailed_results': swapped_clean_results,
            'corrupt_accuracy': corrupt_accuracy,
            'corrupt_avg_correct_rank': corrupt_avg_correct_rank,
            'corrupt_detailed_results': swapped_corrupt_results,
            'debug_info': {
                'hook_called': hook_called,
                'clean_results_count': len(swapped_clean_results),
                'corrupt_results_count': len(swapped_corrupt_results)
            }
        }

    def multi_head_swap_ranking(self, head_list: List[Tuple[int, int]], clean_prompts: List[str], corrupt_prompts: List[str]) -> Dict[str, Any]:
        """Swap multiple heads simultaneously between clean and corrupt prompts."""
        if len(clean_prompts) != len(corrupt_prompts):
            raise ValueError("Clean and corrupt prompt lists must have the same length")
        
        logger.info(f"Running multi-head swap for {len(head_list)} heads: {head_list}")
        
        swap_activations = {}
        
        def multi_capture_and_swap_hook(act, hook, prompt_type, layer_idx):
            nonlocal swap_activations
            
            layer_key = f"layer_{layer_idx}"
            if layer_key not in swap_activations:
                swap_activations[layer_key] = {}
            
            # Find heads for this layer
            layer_heads = [head for layer, head in head_list if layer == layer_idx]
            
            for head in layer_heads:
                if head < act.shape[2]:
                    head_key = f"head_{head}"
                    
                    if prompt_type == 'clean_capture':
                        swap_activations[layer_key][f"{head_key}_clean_to_corrupt"] = act[:, :, head, :].clone()
                    elif prompt_type == 'corrupt_capture':
                        swap_activations[layer_key][f"{head_key}_corrupt_to_clean"] = act[:, :, head, :].clone()
                    elif prompt_type == 'clean_apply':
                        if f"{head_key}_corrupt_to_clean" in swap_activations[layer_key]:
                            stored_activation = swap_activations[layer_key][f"{head_key}_corrupt_to_clean"]
                            current_seq_len = act.shape[1]
                            stored_seq_len = stored_activation.shape[1]
                            
                            if current_seq_len == stored_seq_len:
                                act[:, :, head, :] = stored_activation
                            else:
                                min_len = min(current_seq_len, stored_seq_len)
                                act[:, :min_len, head, :] = stored_activation[:, :min_len, :]
                    elif prompt_type == 'corrupt_apply':
                        if f"{head_key}_clean_to_corrupt" in swap_activations[layer_key]:
                            stored_activation = swap_activations[layer_key][f"{head_key}_clean_to_corrupt"]
                            current_seq_len = act.shape[1]
                            stored_seq_len = stored_activation.shape[1]
                            
                            if current_seq_len == stored_seq_len:
                                act[:, :, head, :] = stored_activation
                            else:
                                min_len = min(current_seq_len, stored_seq_len)
                                act[:, :min_len, head, :] = stored_activation[:, :min_len, :]
            
            return act
        
        # Get unique layers
        unique_layers = list(set(layer for layer, _ in head_list))
        
        multi_swapped_clean_results = []
        multi_swapped_corrupt_results = []
        
        with torch.no_grad():
            for i, (clean_prompt, corrupt_prompt) in enumerate(zip(clean_prompts, corrupt_prompts)):
                # Reset swap storage for each pair
                swap_activations = {}
                
                # Create hook functions for each layer
                capture_hooks = []
                apply_hooks = []
                
                for layer_idx in unique_layers:
                    hook_name = f"blocks.{layer_idx}.attn.hook_result"
                    
                    def make_clean_capture_hook(layer=layer_idx):
                        return lambda act, hook: multi_capture_and_swap_hook(act, hook, 'clean_capture', layer)
                    
                    def make_corrupt_capture_hook(layer=layer_idx):
                        return lambda act, hook: multi_capture_and_swap_hook(act, hook, 'corrupt_capture', layer)
                    
                    def make_clean_apply_hook(layer=layer_idx):
                        return lambda act, hook: multi_capture_and_swap_hook(act, hook, 'clean_apply', layer)
                    
                    def make_corrupt_apply_hook(layer=layer_idx):
                        return lambda act, hook: multi_capture_and_swap_hook(act, hook, 'corrupt_apply', layer)
                    
                    capture_hooks.append((hook_name, make_clean_capture_hook()))
                    capture_hooks.append((hook_name, make_corrupt_capture_hook()))
                    apply_hooks.append((hook_name, make_clean_apply_hook()))
                    apply_hooks.append((hook_name, make_corrupt_apply_hook()))
                
                # Capture activations
                clean_tokens = self.model.to_tokens(clean_prompt, prepend_bos=True)
                corrupt_tokens = self.model.to_tokens(corrupt_prompt, prepend_bos=True)
                
                # Capture clean activations
                clean_capture_hooks = [(name, hook) for name, hook in capture_hooks if 'clean_capture' in str(hook)]
                _ = self.model.run_with_hooks(clean_tokens, fwd_hooks=clean_capture_hooks)
                
                # Capture corrupt activations  
                corrupt_capture_hooks = [(name, hook) for name, hook in capture_hooks if 'corrupt_capture' in str(hook)]
                _ = self.model.run_with_hooks(corrupt_tokens, fwd_hooks=corrupt_capture_hooks)
                
                # Apply multi-head swaps and get results
                clean_apply_hooks = [(name, hook) for name, hook in apply_hooks if 'clean_apply' in str(hook)]
                clean_swapped_logits = self.model.run_with_hooks(clean_tokens, fwd_hooks=clean_apply_hooks)
                
                corrupt_apply_hooks = [(name, hook) for name, hook in apply_hooks if 'corrupt_apply' in str(hook)]
                corrupt_swapped_logits = self.model.run_with_hooks(corrupt_tokens, fwd_hooks=corrupt_apply_hooks)
                
                # Process results (similar to single-head swap)
                clean_last_logits = clean_swapped_logits[0, -1, :]
                clean_logits_info = self.get_digit_logits_info_transformer_lens(clean_prompt, clean_last_logits)
                
                if "error" not in clean_logits_info:
                    clean_digit_logits = {k: float(v) for k, v in clean_logits_info["digit_logits"].items()}
                    clean_digit_probs = {k: float(v) for k, v in clean_logits_info["digit_softmax_probs"].items()}
                    clean_global_ranks = clean_logits_info["digit_global_ranks"]
                    
                    clean_sorted_digits = sorted(clean_digit_probs.items(), key=lambda x: x[1], reverse=True)
                    clean_digit_ranks = {digit: rank + 1 for rank, (digit, _) in enumerate(clean_sorted_digits)}
                    
                    clean_correct_answer, clean_stimulus_digits = self.extract_msit_info(clean_prompt)
                    clean_candidate_performance = {}
                    if clean_correct_answer and clean_stimulus_digits:
                        clean_stimulus_probs = {d: clean_digit_probs[d] for d in clean_stimulus_digits if d in clean_digit_probs}
                        if clean_stimulus_probs:
                            clean_sorted_stimulus = sorted(clean_stimulus_probs.items(), key=lambda x: x[1], reverse=True)
                            clean_stimulus_ranks = {digit: rank + 1 for rank, (digit, _) in enumerate(clean_sorted_stimulus)}
                            
                            clean_candidate_performance = {
                                'correct_answer': clean_correct_answer,
                                'stimulus_digits': clean_stimulus_digits,
                                'correct_answer_global_rank': clean_global_ranks.get(clean_correct_answer, 50000) + 1,
                                'correct_answer_stimulus_rank': clean_stimulus_ranks.get(clean_correct_answer, len(clean_stimulus_digits) + 1),
                                'stimulus_ranks': clean_stimulus_ranks,
                                'is_correct_top_choice': clean_stimulus_ranks.get(clean_correct_answer, 999) == 1
                            }
                    
                    multi_swapped_clean_results.append({
                        'prompt_index': i,
                        'digit_logits': clean_digit_logits,
                        'digit_probabilities': clean_digit_probs,
                        'digit_ranks': clean_digit_ranks,
                        'digit_global_ranks': clean_global_ranks,
                        'candidate_performance': clean_candidate_performance
                    })
                
                # Process corrupt results
                corrupt_last_logits = corrupt_swapped_logits[0, -1, :]
                corrupt_logits_info = self.get_digit_logits_info_transformer_lens(corrupt_prompt, corrupt_last_logits)
                
                if "error" not in corrupt_logits_info:
                    corrupt_digit_logits = {k: float(v) for k, v in corrupt_logits_info["digit_logits"].items()}
                    corrupt_digit_probs = {k: float(v) for k, v in corrupt_logits_info["digit_softmax_probs"].items()}
                    corrupt_global_ranks = corrupt_logits_info["digit_global_ranks"]
                    
                    corrupt_sorted_digits = sorted(corrupt_digit_probs.items(), key=lambda x: x[1], reverse=True)
                    corrupt_digit_ranks = {digit: rank + 1 for rank, (digit, _) in enumerate(corrupt_sorted_digits)}
                    
                    corrupt_correct_answer, corrupt_stimulus_digits = self.extract_msit_info(corrupt_prompt)
                    corrupt_candidate_performance = {}
                    if corrupt_correct_answer and corrupt_stimulus_digits:
                        corrupt_stimulus_probs = {d: corrupt_digit_probs[d] for d in corrupt_stimulus_digits if d in corrupt_digit_probs}
                        if corrupt_stimulus_probs:
                            corrupt_sorted_stimulus = sorted(corrupt_stimulus_probs.items(), key=lambda x: x[1], reverse=True)
                            corrupt_stimulus_ranks = {digit: rank + 1 for rank, (digit, _) in enumerate(corrupt_sorted_stimulus)}
                            
                            corrupt_candidate_performance = {
                                'correct_answer': corrupt_correct_answer,
                                'stimulus_digits': corrupt_stimulus_digits,
                                'correct_answer_global_rank': corrupt_global_ranks.get(corrupt_correct_answer, 50000) + 1,
                                'correct_answer_stimulus_rank': corrupt_stimulus_ranks.get(corrupt_correct_answer, len(corrupt_stimulus_digits) + 1),
                                'stimulus_ranks': corrupt_stimulus_ranks,
                                'is_correct_top_choice': corrupt_stimulus_ranks.get(corrupt_correct_answer, 999) == 1
                            }
                    
                    multi_swapped_corrupt_results.append({
                        'prompt_index': i,
                        'digit_logits': corrupt_digit_logits,
                        'digit_probabilities': corrupt_digit_probs,
                        'digit_ranks': corrupt_digit_ranks,
                        'digit_global_ranks': corrupt_global_ranks,
                        'candidate_performance': corrupt_candidate_performance
                    })
        
        # Calculate summary statistics
        if len(multi_swapped_clean_results) > 0:
            clean_correct_top_count = sum(1 for r in multi_swapped_clean_results 
                                         if r['candidate_performance'].get('is_correct_top_choice', False))
            clean_avg_correct_rank = sum(r['candidate_performance'].get('correct_answer_stimulus_rank', 999) 
                                        for r in multi_swapped_clean_results) / len(multi_swapped_clean_results)
            clean_accuracy = clean_correct_top_count / len(multi_swapped_clean_results)
        else:
            clean_correct_top_count = 0
            clean_avg_correct_rank = 999.0
            clean_accuracy = 0.0
        
        if len(multi_swapped_corrupt_results) > 0:
            corrupt_correct_top_count = sum(1 for r in multi_swapped_corrupt_results 
                                          if r['candidate_performance'].get('is_correct_top_choice', False))
            corrupt_avg_correct_rank = sum(r['candidate_performance'].get('correct_answer_stimulus_rank', 999) 
                                         for r in multi_swapped_corrupt_results) / len(multi_swapped_corrupt_results)
            corrupt_accuracy = corrupt_correct_top_count / len(multi_swapped_corrupt_results)
        else:
            corrupt_correct_top_count = 0
            corrupt_avg_correct_rank = 999.0
            corrupt_accuracy = 0.0
        
        return {
            'heads_swapped': head_list,
            'num_heads': len(head_list),
            'clean_results': multi_swapped_clean_results,
            'corrupt_results': multi_swapped_corrupt_results,
            'summary': {
                'clean_accuracy': clean_accuracy,
                'clean_avg_correct_rank': clean_avg_correct_rank,
                'clean_correct_top_count': clean_correct_top_count,
                'corrupt_accuracy': corrupt_accuracy,
                'corrupt_avg_correct_rank': corrupt_avg_correct_rank,
                'corrupt_correct_top_count': corrupt_correct_top_count,
                'clean_results_count': len(multi_swapped_clean_results),
                'corrupt_results_count': len(multi_swapped_corrupt_results)
            }
        }

    def ablate_head_ranking(self, layer: int, head: int, prompts: List[str]) -> Dict[str, Any]:
        """Ablate specific head and measure ranking performance using integrated logits function."""
        hook_called = False
        original_norm = None
        ablated_norm = None
        
        def zero_head_hook(act, hook):
            nonlocal hook_called, original_norm, ablated_norm
            hook_called = True
            
            # Store original norm for debugging
            if head < act.shape[2]:  # Check if head index is valid
                original_norm = torch.norm(act[:, :, head, :]).item()
                act[:, :, head, :] = 0.0
                ablated_norm = torch.norm(act[:, :, head, :]).item()
            
            return act
            
        hook_name = f"blocks.{layer}.attn.hook_result"
        
        # Debug: Check if hook name exists in model
        valid_hooks = [name for name, _ in self.model.named_modules()]
        if hook_name.replace('.', '_') not in [h.replace('.', '_') for h in valid_hooks]:
            logger.warning(f"Hook {hook_name} may not exist in model. Available attention hooks:")
            attn_hooks = [h for h in valid_hooks if 'attn' in h][:5]  # Show first 5
            for h in attn_hooks:
                logger.warning(f"  {h}")
        
        ablated_results = []
        
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                tokens = self.model.to_tokens(prompt, prepend_bos=True)
                
                # Run with head ablated to get modified logits
                logits = self.model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, zero_head_hook)]
                )
                
                # Focus on last token logits (where answer should be)
                last_logits = logits[0, -1, :]
                
                # Use our TransformerLens-compatible logits function with pre-computed logits
                logits_info = self.get_digit_logits_info_transformer_lens(prompt, last_logits)
                
                if "error" in logits_info:
                    logger.warning(f"Error getting logits for ablated prompt {i}: {logits_info['error']}")
                    continue
                
                # Extract data from integrated function
                digit_logits = {k: float(v) for k, v in logits_info["digit_logits"].items()}
                digit_probs = {k: float(v) for k, v in logits_info["digit_softmax_probs"].items()}
                global_ranks = logits_info["digit_global_ranks"]
                
                # Convert to format expected by rest of code
                sorted_digits = sorted(digit_probs.items(), key=lambda x: x[1], reverse=True)
                digit_ranks = {digit: rank + 1 for rank, (digit, _) in enumerate(sorted_digits)}
                
                # Extract MSIT task information
                correct_answer, stimulus_digits = self.extract_msit_info(prompt)
                
                # Calculate candidate-specific metrics
                candidate_performance = {}
                if correct_answer and stimulus_digits:
                    # Rank within stimulus digits only
                    stimulus_probs = {d: digit_probs[d] for d in stimulus_digits if d in digit_probs}
                    if stimulus_probs:
                        sorted_stimulus = sorted(stimulus_probs.items(), key=lambda x: x[1], reverse=True)
                        stimulus_ranks = {digit: rank + 1 for rank, (digit, _) in enumerate(sorted_stimulus)}
                        
                        candidate_performance = {
                            'correct_answer': correct_answer,
                            'stimulus_digits': stimulus_digits,
                            'correct_answer_global_rank': global_ranks.get(correct_answer, 50000) + 1,  # Convert to 1-indexed
                            'correct_answer_stimulus_rank': stimulus_ranks.get(correct_answer, len(stimulus_digits) + 1),
                            'stimulus_ranks': stimulus_ranks,
                            'is_correct_top_choice': stimulus_ranks.get(correct_answer, 999) == 1
                        }
                
                prompt_result = {
                    'prompt_index': i,
                    'digit_logits': digit_logits,
                    'digit_probabilities': digit_probs,
                    'digit_ranks': digit_ranks,
                    'digit_global_ranks': global_ranks,
                    'candidate_performance': candidate_performance
                }
                
                ablated_results.append(prompt_result)
        
        # Calculate summary statistics for this ablation
        correct_top_count = sum(1 for r in ablated_results 
                               if r['candidate_performance'].get('is_correct_top_choice', False))
        
        avg_correct_rank = sum(r['candidate_performance'].get('correct_answer_stimulus_rank', 999) 
                              for r in ablated_results) / len(ablated_results)
        
        # Debug logging
        if layer == 0 and head == 0:  # Only log for first head to avoid spam
            logger.info(f"DEBUG L{layer}H{head}: Hook called: {hook_called}, Original norm: {original_norm}, Ablated norm: {ablated_norm}")
        
        return {
            'accuracy': correct_top_count / len(ablated_results),
            'avg_correct_rank': avg_correct_rank,
            'detailed_results': ablated_results,
            'debug_info': {
                'hook_called': hook_called,
                'original_norm': original_norm,
                'ablated_norm': ablated_norm
            }
        }
    
    def run_sweep(self, clean_prompts: List[str], corrupt_prompts: List[str], mode: str = "ablate") -> Dict[str, Any]:
        """Run complete head ablation or swap sweep with rank-based measurement.
        
        Args:
            clean_prompts: List of clean (congruent) prompts
            corrupt_prompts: List of corrupt (incongruent) prompts  
            mode: Either "ablate" (zero out heads) or "swap" (swap head activations)
        """
        if not self.model:
            self.load_model()
            
        logger.info(f"Starting head {mode} sweep with ranking analysis...")
        
        # Baseline performance with detailed ranking
        baseline_clean = self.measure_digit_rankings(clean_prompts, "baseline_clean")
        baseline_corrupt = self.measure_digit_rankings(corrupt_prompts, "baseline_corrupt")
        
        logger.info(f"Baseline Clean - Accuracy: {baseline_clean['summary_stats']['accuracy']:.2%}, Avg Rank: {baseline_clean['summary_stats']['avg_correct_rank']:.2f}")
        logger.info(f"Baseline Corrupt - Accuracy: {baseline_corrupt['summary_stats']['accuracy']:.2%}, Avg Rank: {baseline_corrupt['summary_stats']['avg_correct_rank']:.2f}")
        
        results = {
            'baseline': {
                'clean_results': baseline_clean,
                'corrupt_results': baseline_corrupt,
                'task_discrimination': {
                    'accuracy_gap': baseline_clean['summary_stats']['accuracy'] - baseline_corrupt['summary_stats']['accuracy'],
                    'rank_gap': baseline_corrupt['summary_stats']['avg_correct_rank'] - baseline_clean['summary_stats']['avg_correct_rank']
                }
            },
            'head_effects': {},
            'important_heads': []
        }
        
        # Test each head
        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads
        
        for layer in range(n_layers):
            for head in range(n_heads):
                logger.info(f"Testing head L{layer}H{head} with {mode} mode...")
                
                if mode == "swap":
                    # Swap head activations between clean and corrupt
                    swap_results = self.swap_head_ranking(layer, head, clean_prompts, corrupt_prompts)
                    modified_clean = {
                        'accuracy': swap_results['clean_accuracy'],
                        'avg_correct_rank': swap_results['clean_avg_correct_rank'],
                        'detailed_results': swap_results['clean_detailed_results']
                    }
                    modified_corrupt = {
                        'accuracy': swap_results['corrupt_accuracy'], 
                        'avg_correct_rank': swap_results['corrupt_avg_correct_rank'],
                        'detailed_results': swap_results['corrupt_detailed_results']
                    }
                else:
                    # Ablate head and measure ranking performance
                    modified_clean = self.ablate_head_ranking(layer, head, clean_prompts)
                    modified_corrupt = self.ablate_head_ranking(layer, head, corrupt_prompts)
                
                # Calculate effect sizes based on ranking metrics
                if mode == "swap":
                    # For swap mode, measure changes from baseline
                    clean_accuracy_change = modified_clean['accuracy'] - baseline_clean['summary_stats']['accuracy']
                    corrupt_accuracy_change = modified_corrupt['accuracy'] - baseline_corrupt['summary_stats']['accuracy']
                    
                    clean_rank_change = modified_clean['avg_correct_rank'] - baseline_clean['summary_stats']['avg_correct_rank']
                    corrupt_rank_change = modified_corrupt['avg_correct_rank'] - baseline_corrupt['summary_stats']['avg_correct_rank']
                else:
                    # For ablate mode, measure drops from baseline (negative changes)
                    clean_accuracy_change = baseline_clean['summary_stats']['accuracy'] - modified_clean['accuracy']
                    corrupt_accuracy_change = baseline_corrupt['summary_stats']['accuracy'] - modified_corrupt['accuracy']
                    
                    clean_rank_change = modified_clean['avg_correct_rank'] - baseline_clean['summary_stats']['avg_correct_rank']
                    corrupt_rank_change = modified_corrupt['avg_correct_rank'] - baseline_corrupt['summary_stats']['avg_correct_rank']
                
                # Calculate logits changes for correct answers
                clean_logit_changes = []
                corrupt_logit_changes = []
                
                for i, prompt_result in enumerate(modified_clean['detailed_results']):
                    baseline_prompt = baseline_clean['prompt_results'][i]
                    correct_answer = prompt_result['candidate_performance'].get('correct_answer')
                    if correct_answer:
                        baseline_logit = baseline_prompt['digit_logits'].get(correct_answer, 0)
                        modified_logit = prompt_result['digit_logits'].get(correct_answer, 0)
                        clean_logit_changes.append(modified_logit - baseline_logit)
                
                for i, prompt_result in enumerate(modified_corrupt['detailed_results']):
                    baseline_prompt = baseline_corrupt['prompt_results'][i]
                    correct_answer = prompt_result['candidate_performance'].get('correct_answer')
                    if correct_answer:
                        baseline_logit = baseline_prompt['digit_logits'].get(correct_answer, 0)
                        modified_logit = prompt_result['digit_logits'].get(correct_answer, 0)
                        corrupt_logit_changes.append(modified_logit - baseline_logit)
                
                avg_clean_logit_change = sum(clean_logit_changes) / len(clean_logit_changes) if clean_logit_changes else 0
                avg_corrupt_logit_change = sum(corrupt_logit_changes) / len(corrupt_logit_changes) if corrupt_logit_changes else 0
                
                # Task signal changes
                if mode == "swap":
                    # For swap: measure how task discrimination changes after swapping
                    baseline_accuracy_gap = baseline_clean['summary_stats']['accuracy'] - baseline_corrupt['summary_stats']['accuracy']
                    swapped_accuracy_gap = modified_clean['accuracy'] - modified_corrupt['accuracy']
                    accuracy_signal_change = swapped_accuracy_gap - baseline_accuracy_gap
                    
                    baseline_rank_gap = baseline_corrupt['summary_stats']['avg_correct_rank'] - baseline_clean['summary_stats']['avg_correct_rank']
                    swapped_rank_gap = modified_corrupt['avg_correct_rank'] - modified_clean['avg_correct_rank']
                    rank_signal_change = swapped_rank_gap - baseline_rank_gap
                else:
                    # For ablation: measure signal drop (how much task discrimination is lost)
                    baseline_accuracy_gap = baseline_clean['summary_stats']['accuracy'] - baseline_corrupt['summary_stats']['accuracy']
                    ablated_accuracy_gap = modified_clean['accuracy'] - modified_corrupt['accuracy']
                    accuracy_signal_change = baseline_accuracy_gap - ablated_accuracy_gap
                    
                    baseline_rank_gap = baseline_corrupt['summary_stats']['avg_correct_rank'] - baseline_clean['summary_stats']['avg_correct_rank']
                    ablated_rank_gap = modified_corrupt['avg_correct_rank'] - modified_clean['avg_correct_rank']
                    rank_signal_change = baseline_rank_gap - ablated_rank_gap
                
                logit_signal_change = avg_clean_logit_change - avg_corrupt_logit_change
                
                head_result = {
                    'mode': mode,
                    'clean_accuracy_change': clean_accuracy_change,
                    'corrupt_accuracy_change': corrupt_accuracy_change,
                    'clean_rank_change': clean_rank_change,
                    'corrupt_rank_change': corrupt_rank_change,
                    'clean_logit_change': avg_clean_logit_change,
                    'corrupt_logit_change': avg_corrupt_logit_change,
                    'accuracy_signal_change': accuracy_signal_change,
                    'rank_signal_change': rank_signal_change,
                    'logit_signal_change': logit_signal_change,
                    'importance_score': abs(accuracy_signal_change) + abs(rank_signal_change) * 0.1 + abs(logit_signal_change) * 0.01,
                    'modified_clean_results': modified_clean,
                    'modified_corrupt_results': modified_corrupt,
                    # Maintain backward compatibility with old field names
                    'ablated_clean_results': modified_clean,
                    'ablated_corrupt_results': modified_corrupt
                }
                
                results['head_effects'][f"L{layer}H{head}"] = head_result
                
                # Track important heads (significant signal change)
                importance_threshold = 0.05  # Lower threshold for ranking-based measurement
                if head_result['importance_score'] > importance_threshold:
                    results['important_heads'].append({
                        'layer': layer,
                        'head': head,
                        'mode': mode,
                        'accuracy_signal_change': accuracy_signal_change,
                        'rank_signal_change': rank_signal_change,
                        'importance_score': head_result['importance_score']
                    })
                    
                logger.info(f"L{layer}H{head} ({mode}) - Acc: {accuracy_signal_change:.3f}, Rank: {rank_signal_change:.3f}, Logit: {logit_signal_change:.3f}")
        
        # Sort important heads by importance
        results['important_heads'].sort(key=lambda x: x['importance_score'], reverse=True)
        
        logger.info(f"Found {len(results['important_heads'])} important heads")
        for head_info in results['important_heads'][:5]:  # Top 5
            logger.info(f"  L{head_info['layer']}H{head_info['head']}: importance={head_info['importance_score']:.4f}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save ablation results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")

def main():
    """Run head ablation sweep on MSIT data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Head ablation sweep for MSIT")
    parser.add_argument("--model", default="gpt2-large", help="Model name (default: gpt2-large)")
    parser.add_argument("--samples", type=int, default=10, help="Number of test samples (default: 10)")
    parser.add_argument("--output", default="head_ablation_results.json", help="Output JSON filename (saved inside output_dir)")
    # Compute repo root (two levels up from this script: scripts_test -> llm_control -> repo root)
    _repo_root = Path(__file__).resolve().parents[2]
    _default_output_dir = _repo_root / "data/msit_pilot_outputs_smallnrep"
    parser.add_argument(
        "--output_dir",
        default=str(_default_output_dir),
        help=f"Directory to store JSON outputs (default: {_default_output_dir})"
    )
    parser.add_argument("--layer", type=int, help="Layer index for single-head quick verification")
    parser.add_argument("--head", type=int, help="Head index for single-head quick verification")
    parser.add_argument("--mode", choices=["ablate", "swap"], default="ablate", help="Analysis mode: 'ablate' (zero out heads) or 'swap' (swap head activations between clean/corrupt)")
    parser.add_argument("--with_examples", action="store_true", default=True, help="Include few-shot examples")
    parser.add_argument("--no_examples", action="store_true", help="Disable few-shot examples")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    # Determine whether to use examples (default True unless --no_examples)
    use_examples = args.with_examples and not args.no_examples
    
    # Run sweep with GPT2-large parameters matching msit_test_1-0_gpt2.sh
    sweep = HeadAblationSweep(args.model)
    
    # Create test data with MSIT format and examples (like the working test script)
    clean_prompts, corrupt_prompts = sweep.create_msit_test_data(args.samples, with_examples=use_examples)
    
    print(f"Using model: {args.model}")
    print(f"Analysis mode: {args.mode}")
    print(f"Test samples: {args.samples}")
    print(f"With examples: {use_examples}")
    print(f"Example clean prompt:")
    print(f"{clean_prompts[0][:200]}...")
    
    # If user specifies a single head, do a super fast verification run
    if args.layer is not None and args.head is not None:
        sweep.load_model()
        logger.info(f"Running single-head verification for L{args.layer}H{args.head} in {args.mode} mode...")

        baseline_clean = sweep.measure_digit_rankings(clean_prompts, "baseline_clean")
        baseline_corrupt = sweep.measure_digit_rankings(corrupt_prompts, "baseline_corrupt")

        if args.mode == "swap":
            # Swap head activations between clean and corrupt
            swap_results = sweep.swap_head_ranking(args.layer, args.head, clean_prompts, corrupt_prompts)
            modified_clean = {
                'accuracy': swap_results['clean_accuracy'],
                'avg_correct_rank': swap_results['clean_avg_correct_rank'],
                'detailed_results': swap_results['clean_detailed_results']
            }
            modified_corrupt = {
                'accuracy': swap_results['corrupt_accuracy'], 
                'avg_correct_rank': swap_results['corrupt_avg_correct_rank'],
                'detailed_results': swap_results['corrupt_detailed_results']
            }
        else:
            # Ablate head 
            modified_clean = sweep.ablate_head_ranking(args.layer, args.head, clean_prompts)
            modified_corrupt = sweep.ablate_head_ranking(args.layer, args.head, corrupt_prompts)

        # Calculate effect sizes based on mode
        if args.mode == "swap":
            # For swap mode, measure changes from baseline
            clean_accuracy_change = modified_clean['accuracy'] - baseline_clean['summary_stats']['accuracy']
            corrupt_accuracy_change = modified_corrupt['accuracy'] - baseline_corrupt['summary_stats']['accuracy']
            clean_rank_change = modified_clean['avg_correct_rank'] - baseline_clean['summary_stats']['avg_correct_rank']
            corrupt_rank_change = modified_corrupt['avg_correct_rank'] - baseline_corrupt['summary_stats']['avg_correct_rank']
        else:
            # For ablate mode, measure drops from baseline
            clean_accuracy_change = baseline_clean['summary_stats']['accuracy'] - modified_clean['accuracy']
            corrupt_accuracy_change = baseline_corrupt['summary_stats']['accuracy'] - modified_corrupt['accuracy']
            clean_rank_change = modified_clean['avg_correct_rank'] - baseline_clean['summary_stats']['avg_correct_rank']
            corrupt_rank_change = modified_corrupt['avg_correct_rank'] - baseline_corrupt['summary_stats']['avg_correct_rank']

        # Average logit change for correct answers
        def avg_logit_change(mod, base):
            changes = []
            for i, pr in enumerate(mod['detailed_results']):
                bpr = base['prompt_results'][i]
                ca = pr['candidate_performance'].get('correct_answer')
                if ca:
                    changes.append(pr['digit_logits'].get(ca, 0) - bpr['digit_logits'].get(ca, 0))
            return sum(changes)/len(changes) if changes else 0.0

        clean_logit_change = avg_logit_change(modified_clean, baseline_clean)
        corrupt_logit_change = avg_logit_change(modified_corrupt, baseline_corrupt)
        logit_signal_change = clean_logit_change - corrupt_logit_change

        results = {
            'mode': 'single_head_verification',
            'analysis_mode': args.mode,
            'target_head': {'layer': args.layer, 'head': args.head},
            'baseline': {
                'clean_results': baseline_clean,
                'corrupt_results': baseline_corrupt,
                'task_discrimination': {
                    'accuracy_gap': baseline_clean['summary_stats']['accuracy'] - baseline_corrupt['summary_stats']['accuracy'],
                    'rank_gap': baseline_corrupt['summary_stats']['avg_correct_rank'] - baseline_clean['summary_stats']['avg_correct_rank']
                }
            },
            'modified': {
                'clean': modified_clean,
                'corrupt': modified_corrupt
            },
            'effects': {
                'clean_accuracy_change': clean_accuracy_change,
                'corrupt_accuracy_change': corrupt_accuracy_change,
                'clean_rank_change': clean_rank_change,
                'corrupt_rank_change': corrupt_rank_change,
                'clean_logit_change': clean_logit_change,
                'corrupt_logit_change': corrupt_logit_change,
                'logit_signal_change': logit_signal_change
            }
        }
    else:
        # Run full sweep in specified mode
        results = sweep.run_sweep(clean_prompts, corrupt_prompts, mode=args.mode)
        
        # DISABLED: Auto-run multi-head swap function (causing troubles)
        # if args.mode == 'swap' and len(results.get('important_heads', [])) >= 2:
        #     print(f"\n🔄 Running automatic multi-head swap test with top {min(15, len(results['important_heads']))} heads...")
        #     print(f"📈 Using 2x sample size ({args.samples * 2}) for more robust multi-head analysis...")
        #     
        #     # Generate 2x sample size for multi-head test
        #     multi_clean_prompts, multi_corrupt_prompts = sweep.create_msit_test_data(
        #         n_samples=args.samples * 2, 
        #         with_examples=use_examples  # Fixed: was args.few_shot
        #     )
        #     
        #     # Get top 15 important heads (or all if fewer than 15)
        #     top_heads = results['important_heads'][:min(15, len(results['important_heads']))]
        #     head_list = [(h['layer'], h['head']) for h in top_heads]
        #     
        #     # Run multi-head swap
        #     multi_results = sweep.multi_head_swap_ranking(head_list, multi_clean_prompts, multi_corrupt_prompts)
        #     
        #     # Add multi-head results to main results
        #     results['multi_head_swap'] = multi_results
        #     
        #     print(f"✅ Multi-head swap completed for {len(head_list)} heads")
        
        print("ℹ️  Auto multi-head swap function is disabled")
    
    # Build timestamped run directory inside output_dir (normalize relative paths against repo root)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{ts}_{args.model}_head-{args.mode}_ssn-{args.samples}"
    base_output_dir = Path(args.output_dir)
    if not base_output_dir.is_absolute():
        base_output_dir = _repo_root / base_output_dir
    run_dir = base_output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save results JSON within the run directory
    output_path = run_dir / args.output
    sweep.save_results(results, str(output_path))
    
    # Mode-dependent output header
    mode = results.get('analysis_mode', 'ablate')
    if mode == 'swap':
        print(f"\n=== Head Swap Results ===")
    else:
        print(f"\n=== Head Ablation Results ===")
    
    print(f"Baseline task discrimination:")
    print(f"  Accuracy gap: {results['baseline']['task_discrimination']['accuracy_gap']:.2%}")
    print(f"  Rank gap: {results['baseline']['task_discrimination']['rank_gap']:.2f}")

    # Single-head printout
    if results.get('mode') == 'single_head_verification':
        th = results['target_head']
        eff = results['effects']
        analysis_mode = results.get('analysis_mode', 'ablate')
        print(f"\nSingle head verified: L{th['layer']}H{th['head']} ({analysis_mode} mode)")
        if analysis_mode == 'swap':
            print(f"  Clean acc change: {eff['clean_accuracy_change']:.3f}")
            print(f"  Corrupt acc change: {eff['corrupt_accuracy_change']:.3f}")
        else:
            print(f"  Clean acc drop: {eff['clean_accuracy_change']:.3f}")
            print(f"  Corrupt acc drop: {eff['corrupt_accuracy_change']:.3f}")
        print(f"  Clean rank change: {eff['clean_rank_change']:.3f}")
        print(f"  Corrupt rank change: {eff['corrupt_rank_change']:.3f}")
        print(f"  Logit signal change: {eff['logit_signal_change']:.3f}")
    else:
        mode_str = results.get('analysis_mode', 'ablate')
        print(f"\nImportant heads found: {len(results['important_heads'])}")
        
        # Separate heads by their primary impact on clean vs corrupt conditions
        clean_critical = []
        corrupt_critical = []
        overall_critical = []
        
        for head_info in results['important_heads']:
            clean_impact = abs(head_info.get('clean_accuracy_change', 0))
            corrupt_impact = abs(head_info.get('corrupt_accuracy_change', 0))
            
            # Classify heads by their primary impact
            if clean_impact > corrupt_impact * 1.5:
                clean_critical.append(head_info)
            elif corrupt_impact > clean_impact * 1.5:
                corrupt_critical.append(head_info)
            else:
                overall_critical.append(head_info)
        
        # Sort each category by importance score
        clean_critical.sort(key=lambda x: x.get('importance_score', 0), reverse=True)
        corrupt_critical.sort(key=lambda x: x.get('importance_score', 0), reverse=True)
        overall_critical.sort(key=lambda x: x.get('importance_score', 0), reverse=True)
        
        signal_type = 'change' if mode_str == 'swap' else 'drop'
        
        # Print clean-critical heads
        if clean_critical:
            print(f"\n🎯 Clean-Critical Heads (top {min(3, len(clean_critical))}):")
            for head_info in clean_critical[:3]:
                print(f"  L{head_info['layer']}H{head_info['head']}: clean_acc_{signal_type}={head_info.get('clean_accuracy_change', 0):.3f}, corrupt_acc_{signal_type}={head_info.get('corrupt_accuracy_change', 0):.3f}")
        
        # Print corrupt-critical heads  
        if corrupt_critical:
            print(f"\n⚡ Corrupt-Critical Heads (top {min(3, len(corrupt_critical))}):")
            for head_info in corrupt_critical[:3]:
                print(f"  L{head_info['layer']}H{head_info['head']}: clean_acc_{signal_type}={head_info.get('clean_accuracy_change', 0):.3f}, corrupt_acc_{signal_type}={head_info.get('corrupt_accuracy_change', 0):.3f}")
        
        # Print overall critical heads
        if overall_critical:
            print(f"\n🔥 Overall-Critical Heads (top {min(3, len(overall_critical))}):")
            for head_info in overall_critical[:3]:
                print(f"  L{head_info['layer']}H{head_info['head']}: clean_acc_{signal_type}={head_info.get('clean_accuracy_change', 0):.3f}, corrupt_acc_{signal_type}={head_info.get('corrupt_accuracy_change', 0):.3f}")
        
        # Multi-head swap results if available
        if 'multi_head_swap' in results:
            multi_results = results['multi_head_swap']
            head_names = [f"L{layer}H{head}" for layer, head in multi_results['heads_swapped']]
            print(f"\n🚀 Multi-Head Swap Results ({multi_results['num_heads']} heads: {', '.join(head_names[:5])}{'+...' if len(head_names) > 5 else ''}):")
            print(f"  📊 Sample size: {multi_results['summary']['clean_results_count']} (2x regular)")
            print(f"  🎯 Combined clean accuracy: {multi_results['summary']['clean_accuracy']:.3f}")
            print(f"  ⚡ Combined corrupt accuracy: {multi_results['summary']['corrupt_accuracy']:.3f}")
            print(f"  📈 Clean avg rank: {multi_results['summary']['clean_avg_correct_rank']:.3f}")
            print(f"  📉 Corrupt avg rank: {multi_results['summary']['corrupt_avg_correct_rank']:.3f}")
            
            # Compare with baseline if available
            if 'baseline' in results:
                baseline_clean_acc = results['baseline']['clean']['accuracy']
                baseline_corrupt_acc = results['baseline']['corrupt']['accuracy']
                clean_multi_change = multi_results['summary']['clean_accuracy'] - baseline_clean_acc
                corrupt_multi_change = multi_results['summary']['corrupt_accuracy'] - baseline_corrupt_acc
                print(f"  🔄 Multi-head effect on clean: {clean_multi_change:+.3f}")
                print(f"  🔄 Multi-head effect on corrupt: {corrupt_multi_change:+.3f}")
        else:
            # Multi-head swapping suggestion for future runs
            if mode_str == 'swap' and len(results['important_heads']) >= 2:
                top_heads = results['important_heads'][:3]
                head_list = [f"L{h['layer']}H{h['head']}" for h in top_heads]
                print(f"\n💡 Multi-Head Swap Suggestion:")
                print(f"   Consider swapping multiple heads simultaneously: {', '.join(head_list)}")
                print(f"   This could reveal combinatorial interference effects not visible in single-head analysis.")

if __name__ == "__main__":
    main()
