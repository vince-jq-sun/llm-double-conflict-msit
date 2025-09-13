"""
MSIT stimulus generator
-----------------------
Generates multi-source interference task (MSIT) stimuli generalized to n positions.

Conventions
- There are `ndigits` positions. Exactly one position holds the TARGET digit.
- Valid TARGET identities are integers {1, 2, ..., ndigits}.
- The neutral (non-response-mapped) digit is 0.
- "Position = identity" means: a target with identity k is placed at position index (k-1).

Conditions (booleans):
- is_simon:    whether spatial (Simon) incongruence is present (target position != identity)
- is_flanker:  whether flanker interference is present (non-target positions use valid response digits)

If both flags are False -> Congruent (baseline): flankers are 0 and target is placed at its identity-indexed position.

Design choices (mirroring canonical MSIT for n=3 and cleanly generalizing):
- Congruent:        flankers=0; target at position = identity
- Simon-only:       flankers=0; target at position != identity
- Flanker-only:     position = identity; flankers are a *single* valid digit d ≠ target
- Simon+Flanker:    position != identity; flankers are a *single* valid digit d ≠ target and d ≠ (position_index+1)

You can change FLANKER_STRATEGY to "uniform_per_slot" if you prefer each flanker slot to be sampled independently
from the allowed set (instead of the single-digit fill used in classic MSIT examples).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import random
import json
import os
from datetime import datetime

NEUTRAL_DIGIT = 0
FLANKER_STRATEGY = "single_digit_fill"  # or "uniform_per_slot"

condition_code={
    0: "Cg", ## congruent
    1: "Sm", ## Simon-only
    2: "Fk", ## Flanker-only
    3: "SmFk", ## Simon+Flanker
    4: "CgLtr", ## Congruent trials with letter as target, sth like t00, t is not a number
    5: "CgExN", ## Congruent trials with extra number as target, sth like 600, the variants are 400, 500, 600, 700, 800, 900
    6: "CgExN-R", ## Congruent trials with extra number (restricted) as target, sth like 400, the variants are only 400, 500, 600, 
    7: "SmExNFk", ## example, 115
    8: "SmFkExN", ## example, 552
    9: "SmFkIdPos", ## example, 331
    10: "CgFkExN", ## example, 525
    11: "CgAllLtr", ## Congruent trials with all letterst, sth like tuu, t and u are not numbers
}

@dataclass
class MSITTrial:
    digits: List[int]              # length ndigits (e.g., [3, 1, 3, 3, 3])
    target_identity: int           # in [1..ndigits]
    target_pos_index: int          # 0-based index where target is placed
    is_simon: bool
    is_flanker: bool
    condition: str                 # "congruent" | "simon_only" | "flanker_only" | "simon_flanker"
    flanker_value: int             # The flanker digit used (0 for neutral, or the actual flanker digit)


def _sample_target(ndigits: int, rng: random.Random) -> int:
    return rng.randint(1, ndigits)


def _sample_position(ndigits: int, identity: int, require_equal: bool, rng: random.Random) -> int:
    """Return a 0-based position index according to equality constraint with identity."""
    identity_index = identity - 1
    if require_equal:
        return identity_index
    else:
        # sample a position not equal to identity_index
        choices = [i for i in range(ndigits) if i != identity_index]
        return rng.choice(choices)


def _pick_flanker_digit(
    ndigits: int,
    exclude: List[int],
    rng: random.Random
) -> int:
    """Pick one valid response digit from 1..ndigits excluding provided identities."""
    candidates = [d for d in range(1, ndigits + 1) if d not in exclude]
    if not candidates:
        raise ValueError(
            f"No valid flanker digits remain after exclusions={exclude}; "
            f"ndigits={ndigits} too small for requested constraints."
        )
    return rng.choice(candidates)


def _fill_flankers_single_digit(
    ndigits: int,
    target_idx: int,
    flanker_digit: int,
) -> List[int]:
    digits = [flanker_digit] * ndigits
    digits[target_idx] = None  # placeholder; set by caller
    return digits


def _fill_flankers_uniform_per_slot(
    ndigits: int,
    target_idx: int,
    allowed_digits: List[int],
    rng: random.Random
) -> List[int]:
    digits = []
    for i in range(ndigits):
        if i == target_idx:
            digits.append(None)  # placeholder
        else:
            digits.append(rng.choice(allowed_digits))
    return digits


def gen_numeric_trial(
    ndigits: int,
    match_ndigits: bool = False,
    rng: Optional[random.Random] = None,
) -> MSITTrial:
    """
    Generate a single trial with numeric target from 1-9, excluding first ndigits numbers.
    For example, if ndigits=3, valid targets are 4,5,6,7,8,9.
    The target appears at a random position, other positions are 0.
    """
    if ndigits < 3:
        raise ValueError("ndigits must be >= 3.")
    
    # Check if we have enough numbers available
    # Available range: (ndigits+1) to 9
    min_target = ndigits + 1
    max_target = 9

    if match_ndigits:
        max_target = min_target+ndigits-1
        if max_target > 9:
            raise ValueError(f"Not enough numbers available for ndigits={ndigits}. "
                            f"Need ndigits <= 8 for valid targets. Consider turn off match_ndigits (bool)")
    
    if min_target > max_target:
        raise ValueError(f"Not enough numbers available for ndigits={ndigits}. "
                        f"Need ndigits <= 8 for valid targets.")
    
    rng = rng or random
    
    # Sample a target number from the available range
    target_number = rng.randint(min_target, max_target)
    
    # Randomly sample a position for the target (0-based index)
    target_pos_index = rng.randint(0, ndigits - 1)
    
    # Create digits array with neutral digits (0) and place the number at target position
    digits = [NEUTRAL_DIGIT] * ndigits
    digits[target_pos_index] = target_number
    
    return MSITTrial(
        digits=digits,
        target_identity=target_number,
        target_pos_index=target_pos_index,
        is_simon=False,
        is_flanker=False,
        condition="numeric_trial",
        flanker_value=NEUTRAL_DIGIT
    )


def gen_pure_position_trial(
    ndigits: int,
    rng: Optional[random.Random] = None
) -> MSITTrial:
    """
    Generate a single trial with pure position-based MSIT.
    The identity is an english letter, to get it,
    first, assuming that the 26 alphabet is mapped to 1..26,
    remove the first ndigits letters, remove the final 6 (u-z),
    then sample one from the remaining at random.
    """
    if ndigits < 3:
        raise ValueError("ndigits must be >= 3.")
    
    # Check if we have enough letters available
    # Remove first ndigits letters and final 6 letters (u-z)
    # Available range: (ndigits+1) to (26-6) = (ndigits+1) to 20
    min_letter_num = ndigits + 1
    max_letter_num = 20  # 26 - 6 = 20 (removing u-z)
    
    if min_letter_num > max_letter_num:
        raise ValueError(f"Not enough letters available for ndigits={ndigits}. "
                        f"Need at least {min_letter_num-ndigits} available letters.")
    
    rng = rng or random
    
    # Sample a letter number from the available range
    letter_num = rng.randint(min_letter_num, max_letter_num)
    
    # Convert to actual letter (1=a, 2=b, ..., 26=z)
    target_letter = chr(ord('a') + letter_num - 1)
    
    # Randomly sample a position for the target (0-based index)
    target_pos_index = rng.randint(0, ndigits - 1)
    
    # Create digits array with neutral digits (0) and place the letter at target position
    digits = [NEUTRAL_DIGIT] * ndigits
    # Place the actual letter character at the target position
    digits[target_pos_index] = target_letter
    
    return MSITTrial(
        digits=digits,
        target_identity=target_letter,  # Store the actual letter character
        target_pos_index=target_pos_index,
        is_simon=False,  # Pure position means position matches identity logic
        is_flanker=False,  # Flankers are neutral (0)
        condition="pure_position",
        flanker_value=NEUTRAL_DIGIT
    )
    


def gen_msit_trial(
    ndigits: int,
    is_simon: bool,
    is_flanker: bool,
    replace_identity: str = None,
    replace_flanker: str = None,
    rng: Optional[random.Random] = None
) -> MSITTrial:
    """
    Generate a single MSIT trial as a list of digits.

    Parameters
    ----------
    ndigits : int
        Number of positions (>=3). Targets are identities 1..ndigits.
    is_simon : bool
        If True, target position != target identity (Simon incongruence).
    is_flanker : bool
        If True, flankers are response-mapped digits (1..ndigits) rather than 0 (neutral).
    rng : random.Random, optional
        Provide a dedicated RNG for reproducibility; if None, uses global random.

    Returns
    -------
    MSITTrial
        dataclass with digits, target meta, and condition label.

    Raises
    ------
    ValueError if constraints are impossible for given ndigits.
    """
    if ndigits < 3:
        raise ValueError("ndigits must be >= 3.")

    rng = rng or random

    # 1) sample a target identity
    t = _sample_target(ndigits, rng)
    # 2) pick a target position: equal or not equal to identity index
    target_idx = _sample_position(ndigits, t, require_equal=not is_simon, rng=rng)

    # 3) decide flanker policy
    if not is_flanker:
        # all non-target positions are NEUTRAL_DIGIT
        digits = [NEUTRAL_DIGIT] * ndigits
        digits[target_idx] = t
        condition = "congruent" if not is_simon else "simon_only"
        flanker_value = NEUTRAL_DIGIT  # 0 for neutral flankers
        # return MSITTrial(digits, t, target_idx, is_simon, is_flanker, condition, flanker_value)

        if (replace_flanker == None) and (replace_identity == None):
            ## no need for any replacement
            return MSITTrial(digits, t, target_idx, is_simon, is_flanker, condition, flanker_value)

    # is_flanker == True
    if not is_simon:
        # Flanker-only: target position == identity-index;
        # flankers are valid digits, all equal to a digit d != target
        # (classic MSIT uses a single repeated digit for clarity).
        # Ensure position=identity holds:
        target_idx = t - 1
        exclude = [t]  # cannot equal target identity
        d = _pick_flanker_digit(ndigits, exclude, rng)

        if FLANKER_STRATEGY == "single_digit_fill":
            digits = _fill_flankers_single_digit(ndigits, target_idx, d)
        else:
            allowed = [x for x in range(1, ndigits + 1) if x not in exclude]
            digits = _fill_flankers_uniform_per_slot(ndigits, target_idx, allowed, rng)

        # set target at its position
        digits[target_idx] = t
        condition = "flanker_only"
        flanker_value = d  # Store the flanker digit used

        if (replace_flanker == None) and (replace_identity == None):
            ## no need for any replacement
            return MSITTrial(digits, t, target_idx, is_simon, is_flanker, condition, flanker_value)
        

    # Simon + Flanker: target position != identity-index;
    # flankers are valid digits, all equal to a digit d != target and d != (position_index+1)
    # This matches the provided examples like 3 1 3 (target=1 at pos 2; flankers not 2).
    exclude = [t, target_idx + 1]
    d = _pick_flanker_digit(ndigits, exclude, rng)

    if FLANKER_STRATEGY == "single_digit_fill":
        digits = _fill_flankers_single_digit(ndigits, target_idx, d)
    else:
        allowed = [x for x in range(1, ndigits + 1) if x not in exclude]
        digits = _fill_flankers_uniform_per_slot(ndigits, target_idx, allowed, rng)

    if replace_identity == None:
        pass
    elif replace_identity == "ExNum":
        ## check ndigits <5
        if ndigits > 4:
            raise ValueError("ndigits must be <5 for identity_exnum condition.")
        dict_map_now = {}
        for qq in range(1,ndigits+1):
            dict_map_now[qq] = qq+ndigits
        
        ## replace target value
        t = dict_map_now[t]


    if replace_flanker == None:
        pass

    elif replace_flanker == "ExNum":
        ## check ndigits <5
        if ndigits > 4:
            raise ValueError("ndigits must be <5 for flanker_exnum condition.")
        dict_map_now = {}
        for qq in range(1,ndigits+1):
            dict_map_now[qq] = qq+ndigits
            dict_map_now[None]=None
        
        ## replace flanker value
        d = dict_map_now[d]
        digits = [dict_map_now[dg] for dg in digits]

    elif replace_flanker == "AllExNum":
        ## find the remaining digits
        remaining_digits = [x for x in range(1,10) if x not in exclude]
        dict_map_now = {}
        for qq in range(1,ndigits+1):
            ## sample a digit from remaining_digits
            dict_map_now[qq] = rng.choice(remaining_digits)
            dict_map_now[None]=None
        
        ## replace flanker value
        d = dict_map_now[d]
        digits = [dict_map_now[dg] for dg in digits]

    elif replace_flanker == "IdPos":
        ## replace flanker value as the position of the identity position
        d = target_idx + 1
        digits = [d if dg != None else dg for dg in digits]
    else:
        raise ValueError("replace_flanker must be one of {ExNum, IdPos or None}")
        
    # place target
    digits[target_idx] = t
    condition = "simon_flanker"
    flanker_value = d  # Store the flanker digit used
    return MSITTrial(digits, t, target_idx, is_simon, is_flanker, condition, flanker_value)


def gen_msit_block(
    ndigits: int,
    n_trials: int,
    p_congruent: float = 0.25,
    p_simon_only: float = 0.25,
    p_flanker_only: float = 0.25,
    p_simon_flanker: float = 0.25,
    seed: Optional[int] = None
) -> List[MSITTrial]:
    """
    Generate a block of trials with specified proportions.

    The four probabilities must sum to 1. Trials are sampled i.i.d.
    """
    probs = [p_congruent, p_simon_only, p_flanker_only, p_simon_flanker]
    if not (abs(sum(probs) - 1.0) < 1e-9):
        raise ValueError("Condition probabilities must sum to 1.0")

    rng = random.Random(seed)
    conditions = ["congruent", "simon_only", "flanker_only", "simon_flanker"]

    out = []
    for _ in range(n_trials):
        cond = rng.choices(conditions, probs, k=1)[0]
        if cond == "congruent":
            trial = gen_msit_trial(ndigits, is_simon=False, is_flanker=False, rng=rng)
        elif cond == "simon_only":
            trial = gen_msit_trial(ndigits, is_simon=True, is_flanker=False, rng=rng)
        elif cond == "flanker_only":
            trial = gen_msit_trial(ndigits, is_simon=False, is_flanker=True, rng=rng)
        else:
            trial = gen_msit_trial(ndigits, is_simon=True, is_flanker=True, rng=rng)
        out.append(trial)
    return out


def as_prompt_line(trial: MSITTrial) -> str:
    """
    Format a single trial into a compact prompt-ready string.

    Example:
        "digits: 3 1 3 | target_identity: 1 | target_pos: 2 | condition: simon_flanker"
    """
    digits_str = " ".join(str(x) for x in trial.digits)
    return f"digits: {digits_str} | target_identity: {trial.target_identity} | target_pos: {trial.target_pos_index+1} | condition: {trial.condition}"


def demo():
    rng = random.Random(42)

    print("=== Demo: ndigits=3, one sample per condition ===")
    print(as_prompt_line(gen_msit_trial(3, False, False, rng)))
    print(as_prompt_line(gen_msit_trial(3, True,  False, rng)))
    print(as_prompt_line(gen_msit_trial(3, False, True,  rng)))
    print(as_prompt_line(gen_msit_trial(3, True,  True,  rng)))

    print("\n=== Demo: ndigits=5, 8 trials mixed ===")
    block = gen_msit_block(5, n_trials=8, seed=7)
    for t in block:
        print(as_prompt_line(t))

    print("\n=== Demo: gen_msit_mixed_sequence ===")
    
    # Example 1: Simple alternating pattern
    print("Example 1: Alternating congruent and simon [0,1,0,1]")
    stim_types = [0, 1, 0, 1]
    stim_text, answers, identities, flanker_values = gen_msit_mixed_sequence(stim_types, ndigits=3, seed=42)
    lines = stim_text.split("\\n")
    for i, line in enumerate(lines):
        condition_names = ["congruent", "simon_only", "flanker_only", "simon_flanker"]
        print(f"  Stimulus {i+1}: {line} | type: {condition_names[stim_types[i]]} | answer: {answers[i]} | identity: {identities[i]}")
    
    # Example 2: All four conditions
    print("\\nExample 2: One of each condition [0,1,2,3]")
    stim_types = [0, 1, 2, 3]
    stim_text, answers, identities, flanker_values = gen_msit_mixed_sequence(stim_types, ndigits=4, seed=123)
    lines = stim_text.split("\\n")
    for i, line in enumerate(lines):
        condition_names = ["congruent", "simon_only", "flanker_only", "simon_flanker"]
        print(f"  Stimulus {i+1}: {line} | type: {condition_names[stim_types[i]]} | answer: {answers[i]} | identity: {identities[i]}")
    
    # Example 3: Custom sequence with repetitions
    print("\\nExample 3: Custom sequence [3,3,2,1,0] (ndigits=3)")
    stim_types = [3, 3, 2, 1, 0]
    stim_text, answers, identities, flanker_values = gen_msit_mixed_sequence(stim_types, ndigits=3, seed=456)
    lines = stim_text.split("\\n")
    for i, line in enumerate(lines):
        condition_names = ["congruent", "simon_only", "flanker_only", "simon_flanker"]
        print(f"  Stimulus {i+1}: {line} | type: {condition_names[stim_types[i]]} | answer: {answers[i]} | identity: {identities[i]}")
    
    print("\\n=== Demo: gen_msit_stimuli ===")
    
    # Example 1: Basic functionality
    print("Example 1: Basic functionality - '0,1' repeated 2 times")
    s, answers, identities, types = gen_msit_stimuli(3, "0,1", 2, is_random=False, seed=42)
    print(f"  Types sequence: {types}")
    lines = s.split("\\n")
    for i, line in enumerate(lines):
        condition_names = ["congruent", "simon_only", "flanker_only", "simon_flanker"]
        print(f"  Stimulus {i+1}: {line} | type: {condition_names[types[i]]} | answer: {answers[i]} | identity: {identities[i]}")
    
    # Example 2: With randomization
    print("\\nExample 2: With randomization - '0,1,2,3' repeated 1 time, randomized")
    s, answers, identities, types = gen_msit_stimuli(4, "0,1,2,3", 1, is_random=True, seed=123)
    print(f"  Types sequence (randomized): {types}")
    lines = s.split("\\n")
    for i, line in enumerate(lines):
        condition_names = ["congruent", "simon_only", "flanker_only", "simon_flanker"]
        print(f"  Stimulus {i+1}: {line} | type: {condition_names[types[i]]} | answer: {answers[i]} | identity: {identities[i]}")
    
    # Example 3: Single type repeated
    print("\\nExample 3: Single type repeated - '3' repeated 3 times")
    s, answers, identities, types = gen_msit_stimuli(3, "3", 3, is_random=False, seed=456)
    print(f"  Types sequence: {types}")
    lines = s.split("\\n")
    for i, line in enumerate(lines):
        condition_names = ["congruent", "simon_only", "flanker_only", "simon_flanker"]
        print(f"  Stimulus {i+1}: {line} | type: {condition_names[types[i]]} | answer: {answers[i]} | identity: {identities[i]}")

    print("\\n=== Summary ===")
    print("Condition types: 0=congruent, 1=simon_only, 2=flanker_only, 3=simon_flanker")
    print("Answer = target position (1-based)")
    print("Identity = target digit value")


def _stim_type_to_flags(stim_type: int) -> Tuple[bool, bool]:
    """
    Map stim_type integer to (is_simon, is_flanker):
        0 -> congruent        -> (False, False)
        1 -> simon_only       -> (True,  False)
        2 -> flanker_only     -> (False, True)
        3 -> simon_flanker    -> (True,  True)
        4 -> pure_position    -> (None, None)  # Special case
        5 -> numeric_trial    -> (None, None)  # Special case
        6 -> numeric_trial    -> (None, None)  # Special case
    """
    mapping = {
        0: (False, False),
        1: (True,  False),
        2: (False, True),
        3: (True,  True),
        4: (None, None),  # Special case for pure_position
        5: (None, None),  # Special case for numeric_trial
        6: (None, None),  # Special case for numeric_trial
        7: (True, True),  # Special case for numeric_trial
        8: (True, True),  # Special case for numeric_trial
        9: (True, True),  # Special case for numeric_trial
        10: (False, False),  # Special case for numeric_trial
    }
    if stim_type not in mapping:
        raise ValueError("stim_type must be one of {0,1,2,3,4,5}.")
    return mapping[stim_type]


def gen_msit_sequence(
    ndigits: int,
    nstims: int,
    stim_type: int,
    separate_by: str = "\\n",
    seed: Optional[int] = None
) -> Tuple[str, List[int]]:
    """
    replaced by the function gen_msit_mixed_sequence
    使用 gen_msit_trial 生成一连串刺激，并返回：
      1) 文本形式的刺激（每个 trial 是一行，以 separate_by 连接；每行仅包含数字，以空格分隔）
      2) 对应的正确答案列表（target 的 1-based 位置）

    参数
    ----
    ndigits: >=3 的位置数量
    nstims: 生成多少个刺激
    stim_type: 0=congruent, 1=simon_only, 2=flanker_only, 3=simon_flanker
    separate_by: 连接各个刺激行的分隔符（默认换行）
    seed: 随机种子（可选）

    返回
    ----
    (stim_text, answers)
      stim_text: 字符串，如 "3 1 3\\n0 2 0\\n..."
      answers:   List[int]，每个元素是正确位置（1-based）
    """
    rng = random.Random(seed) if seed is not None else random
    
    lines = []
    answers = []
    for _ in range(nstims):
        is_simon, is_flanker = _stim_type_to_flags(stim_type)

        if stim_type == 4:  # Pure position condition
            t = gen_pure_position_trial(ndigits=ndigits, rng=rng)
        elif stim_type == 5:  # Numeric trial condition
            t = gen_numeric_trial(ndigits=ndigits, match_ndigits=False, rng=rng)
        elif stim_type == 6:  # Numeric trial condition
            t = gen_numeric_trial(ndigits=ndigits, match_ndigits=True, rng=rng)
        elif stim_type == 7: # SmExNFk:
            t = gen_msit_trial(ndigits=ndigits, is_simon=is_simon, is_flanker=is_flanker, is_identity_exnum=True, rng=rng)
        elif stim_type == 8: # SmFkExN:
            t = gen_msit_trial(ndigits=ndigits, is_simon=is_simon, is_flanker=is_flanker, is_flanker_exnum=True, rng=rng)
        elif stim_type == 9: # SmFkIdPos:
            t = gen_msit_trial(ndigits=ndigits, is_simon=is_simon, is_flanker=is_flanker, rng=rng)
        elif stim_type == 10: # CgFkExN:
            t = gen_msit_trial(ndigits=ndigits, is_simon=is_simon, is_flanker=is_flanker, is_flanker_exnum=True, rng=rng)
        else:
            t = gen_msit_trial(ndigits=ndigits, is_simon=is_simon, is_flanker=is_flanker, rng=rng)
        lines.append(" ".join(str(x) for x in t.digits))
        answers.append(t.target_pos_index + 1)

    return separate_by.join(lines), answers


def gen_msit_mixed_sequence(
    stim_types: List[int],
    ndigits: int,
    separate_by: str = "\n",
    seed: Optional[int] = None
) -> Tuple[str, List[int], List[int], List[int]]:
    """
        这个才是实际使用的函数
    Generate a mixed sequence of MSIT stimuli based on a list of stimulus types.
    
    Parameters
    ----------
    stim_types : List[int]
        List of stimulus types where each element represents the stim_type of a stimulus.
        0=congruent, 1=simon_only, 2=flanker_only, 3=simon_flanker
        Example: [0,1,0,1] generates four stimuli: congruent, simon, congruent, simon
    ndigits : int
        Number of positions (>=3). Targets are identities 1..ndigits.
    separate_by : str, optional
        Separator to join stimulus lines (default "\\n")
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Tuple[str, List[int], List[int], List[int]]
        - stim_text: String with all stimuli (digits separated by spaces, lines by separate_by)
        - answers: List of target positions (1-based) for each stimulus
        - identities: List of target identities for each stimulus
        - flanker_values: List of flanker values for each stimulus
        
    Example
    -------
    >>> stim_text, answers, identities, flanker_values = gen_msit_mixed_sequence([0,1,0,1], 3, seed=42)
    >>> # Returns 4 stimuli with types: congruent, simon_only, congruent, simon_only
    """
    if not stim_types:
        raise ValueError("stim_types list cannot be empty")
    
    for st in stim_types:
        if st not in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}:
            raise ValueError(f"Invalid stim_type {st}. Must be one of {{0,1,2,3,4,5,6,7,8,9,10}}")
    
    rng = random.Random(seed) if seed is not None else random
    nstims = len(stim_types)
    
    lines = []
    answers = []
    identities = []
    flanker_values = []
    
    for stim_type in stim_types:
        is_simon, is_flanker = _stim_type_to_flags(stim_type)
        if stim_type == 4:  # Pure position condition
            trial = gen_pure_position_trial(ndigits=ndigits, rng=rng)
        elif stim_type == 5:  # Numeric trial condition
            trial = gen_numeric_trial(ndigits=ndigits, match_ndigits=False, rng=rng)
        elif stim_type == 6:  # Numeric trial condition
            trial = gen_numeric_trial(ndigits=ndigits, match_ndigits=True, rng=rng)
        elif stim_type == 7:
            trial = gen_msit_trial(ndigits=ndigits, is_simon=is_simon, is_flanker=is_flanker, replace_identity="ExNum", rng=rng)
        elif stim_type == 8:
            trial = gen_msit_trial(ndigits=ndigits, is_simon=is_simon, is_flanker=is_flanker, replace_flanker="ExNum", rng=rng)
        elif stim_type == 9:
            trial = gen_msit_trial(ndigits=ndigits, is_simon=is_simon, is_flanker=is_flanker, replace_flanker="IdPos", rng=rng)
        elif stim_type == 10:
            trial = gen_msit_trial(ndigits=ndigits, is_simon=is_simon, is_flanker=is_flanker, replace_flanker="ExNum", rng=rng)
        else:
            trial = gen_msit_trial(ndigits=ndigits, is_simon=is_simon, is_flanker=is_flanker, rng=rng)
        
        lines.append(" ".join(str(x) for x in trial.digits))
        answers.append(trial.target_pos_index + 1)  # 1-based position
        identities.append(trial.target_identity)     # target identity value
        flanker_values.append(trial.flanker_value)   # flanker digit value
    
    return separate_by.join(lines), answers, identities, flanker_values


def gen_msit_rep_sequence(
    ndigits: int,
    stim_types_str: List[int],
    nrep: int,
    is_random: bool = False,
    separate_by: str = "\\n",
    seed: Optional[int] = None
) -> Tuple[str, List[int], List[int], List[int], List[int]]:
    """
    Generate MSIT stimuli using a string specification of stimulus types.
    
    Parameters
    ----------
    ndigits : int
        Number of positions (>=3). Targets are identities 1..ndigits.
    stim_types_str : str
        String of stimulus types separated by commas, e.g., "0,1" or "0,1,2,3"
        0=congruent, 1=simon_only, 2=flanker_only, 3=simon_flanker
    nrep : int
        Number of times to repeat the stimulus type sequence
        Example: if stim_types_str="0,1" and nrep=3, final sequence is [0,1,0,1,0,1]
    is_random : bool, optional
        If True, randomly permute the final stimulus sequence (default False)
    separate_by : str, optional
        Separator to join stimulus lines (default "\\n")
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Tuple[str, List[int], List[int], List[int], List[int]]
        - s: String with all stimuli (digits separated by spaces, lines by separate_by)
        - answers: List of target positions (1-based) for each stimulus
        - identities: List of target identities for each stimulus  
        - types: List of stimulus types used for each stimulus
        - flanker_values: List of flanker values for each stimulus
        
    Example
    -------
    >>> s, answers, identities, types, flanker_values = gen_msit_rep_sequence(3, "0,1", 2, is_random=False, seed=42)
    >>> # Generates 4 stimuli with types [0,1,0,1]: congruent, simon, congruent, simon
    """
    # Parse the string input into list of integers
    try:
        base_types = [int(x.strip()) for x in stim_types_str.split(',')]
    except ValueError:
        raise ValueError(f"Invalid stim_types_str format: '{stim_types_str}'. Expected comma-separated integers.")
    
    # Validate stimulus types
    for st in base_types:
        if st not in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}:
            raise ValueError(f"Invalid stim_type {st}. Must be one of {{0,1,2,3,4,5,6,7,8,9,10}}")
    
    if nrep <= 0:
        raise ValueError("nrep must be positive")
    
    # Create the repeated sequence
    final_types = base_types * nrep
    
    # Randomly permute if requested
    if is_random:
        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(final_types)
    
    # Generate the stimuli using gen_msit_mixed_sequence
    s, answers, identities, flanker_values = gen_msit_mixed_sequence(
        stim_types=final_types,
        ndigits=ndigits,
        separate_by=separate_by,
        seed=seed
    )
    
    return s, answers, identities, final_types, flanker_values


def generate_examples(ndigits: int, stim_types_str: str, num_examples_per_type: int = None, seed: int = 42) -> str:
    """
    Generate examples for MSIT task based on the stimulus types and ndigits.
    
    Parameters
    ----------
    ndigits : int
        Number of digit positions
    stim_types_str : str
        String of stimulus types separated by commas, e.g., "0,1,2,3"
    num_examples_per_type : int, optional
        Number of examples to generate per stimulus type. If None, generates 4 examples total
        distributed across the stimulus types (4 examples if only one type, 1 per type if multiple)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    str
        Formatted examples string with stimuli and answers
    """
    # Parse the string input into list of integers
    try:
        stim_types = [int(x.strip()) for x in stim_types_str.split(',')]
    except ValueError:
        raise ValueError(f"Invalid stim_types_str format: '{stim_types_str}'. Expected comma-separated integers.")
    
    # Determine number of examples per type
    if num_examples_per_type is None:
        if len(stim_types) == 1:
            # If only one stimulus type, generate 4 examples of that type
            num_examples_per_type = 4
        else:
            # If multiple types, generate 1 example per type
            num_examples_per_type = 1
    
    # Generate examples for each stimulus type
    examples = []
    rng = random.Random(seed)
    
    for i, stim_type in enumerate(stim_types):
        for example_num in range(num_examples_per_type):
            # Generate one stimulus of this type
            if stim_type == 4:  # Pure position condition
                trial = gen_pure_position_trial(ndigits=ndigits, rng=rng)
            elif stim_type == 5:  # Numeric trial condition
                trial = gen_numeric_trial(ndigits=ndigits, rng=rng)
            else:
                is_simon, is_flanker = _stim_type_to_flags(stim_type)
                trial = gen_msit_trial(ndigits=ndigits, is_simon=is_simon, is_flanker=is_flanker, rng=rng)
            
            # Format the stimulus and answer
            stimulus_str = " ".join(str(x) for x in trial.digits)
            answer = trial.target_pos_index + 1  # 1-based position
            
            # Emit example without numbering prefix
            examples.append(f"{stimulus_str} -> Answer: {answer}")
    
    return "\n".join(examples)


def msit_instruciton(ndigits: int, by_image: bool = False, restriction: str = "none", with_examples: bool = False, 
                     stim_types_str: str = None, examples_seed: int = 42, nstims: int | None = 1):
    
    if ndigits < 3:
        raise ValueError("ndigits must be >= 3")
    list_digits = [str(i) for i in list(range(1, ndigits + 1))]
    digits = ", ".join(list_digits)

    if nstims > 1:
        insert_y = f" in each row, "
        insert_v = "- Treat each row independently."
        insert_x = "s"
    else:
        insert_y = ", "
        insert_v = ""
        insert_x = ""

    # Header and goal
    MSIT_INSTRUCTION = f"""MSIT Task Instruction: 

    Goal:{insert_y}there's a number that only appears once; report its position in a sequence of {ndigits} numbers.
    - Counting from left to right{insert_y}the candidate positions are {digits}. 
    {insert_v}
    """
    
    # Important section depends on whether we're doing a single stimulus task
    if nstims == 1:
        MSIT_INSTRUCTION += (
            "Important:\n"
            "    - Answer with a single number only.\n"
        )
    else:
        MSIT_INSTRUCTION += (
            "Important:\n"
            "    - answer for all rows in order with only one number per row, separated by spaces. \n"
            "    - Treat each row independently.\n"
        )

    if restriction == "none":
        pass
    elif restriction == "not1by1":
        MSIT_INSTRUCTION += "- Reading out each stimulus one by one is forbidden. "
    elif restriction == "strict":
        MSIT_INSTRUCTION += "- respond with ONLY the answer{insert_x}. "
    elif restriction == "strict-rev":
        MSIT_INSTRUCTION += "Do not output anything other than the answer{insert_x}."
    else:
        raise ValueError("Invalid restriction")

    # Add examples if requested (no header line, just the examples)
    if with_examples and stim_types_str:
        examples = generate_examples(ndigits, stim_types_str, num_examples_per_type=2, seed=42)
        MSIT_INSTRUCTION += "\n\n" + "examples:\n" + examples + "\n"

    # Rename stimuli header to 'task'
    MSIT_INSTRUCTION += "\n\ntask"

    return MSIT_INSTRUCTION


def self_judge_instruction(answer: List[int]):

    instruction = f"Here are the correct answers: {answer}.\n"
    instruction += "Did you get them all correct? if not, identify the stimulus of the incorrect ones"
    
    return instruction


def save_msit_data(full_input: str, answer: List[int], self_judge: str, 
                   ndigits: int, nstims: int, by_image: bool, restriction: str = None):
    """
    Save MSIT data to JSON file in data/pilot_msit/ directory.
    
    Parameters
    ----------
    full_input : str
        The complete input prompt with instructions and stimuli
    answer : List[int]
        List of correct answers (target positions)
    self_judge : str
        Self-judgment instruction string
    ndigits : int
        Number of digit positions
    nstims : int
        Number of stimuli
    by_image : bool
        Whether the task uses images
    restriction : str, optional
        Restriction type (e.g., "not1by1", "strict")
    """
    # Create directory if it doesn't exist
    os.makedirs("data/pilot_msit", exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine modality
    modality = "image" if by_image else "text"
    
    # Generate filename
    restriction_str = restriction if restriction else "none"
    filename = f"{timestamp}_digits-{ndigits}_stims-{nstims}_{modality}_restrict-{restriction_str}.json"
    filepath = os.path.join("data/pilot_msit", filename)
    
    # Prepare data
    data = {
        "full_input": full_input,
        "answer": answer,
        "self_judge": self_judge,
        "metadata": {
            "ndigits": ndigits,
            "nstims": nstims,
            "by_image": by_image,
            "modality": modality,
            "restriction": restriction,
            "timestamp": timestamp
        }
    }
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Data saved to: {filepath}")
    return filepath


def generate_exhastive_stims_by_type(ndigits,cond=0):
    """
    Generate predefined individual stimuli by condition type.
    
    Parameters
    ----------
    ndigits : int
        Number of digit positions (3 or 4)
    cond : int
        Condition type:
        0 = congruent
        1 = simon_only  
        2 = flanker_only
        3 = simon_flanker
        8 = simon with extended flankers
        10 = congruent with extended flankers
        
    Returns
    -------
    List[MSITTrial]
        List of all possible stimuli for the given condition and ndigits
    """
    if ndigits not in [3, 4, 5]:
        raise ValueError("ndigits must be 3 or 4")
    
    if cond not in [0, 1, 2, 3, 8, 10, 11, 12, 13]:
        err_message = "cond must be "
        err_message += "\n1 (simon_only),"
        err_message += "\n2 (flanker_only),"
        err_message += "\n3 (simon_flanker),"
        err_message += "\n8 (simon with extended flankers),"
        err_message += "\n10 (congruent with extended flankers)"
        err_message += "\n11 (all letters)"
        err_message += "\n12 (congruent with # as flankers)"
        err_message += "\n13 (simon with # as flankers)"
        raise ValueError(err_message)
    
    stimuli = []
    
    if cond == 0:
        # Congruent: target at position = identity, flankers are 0
        for target_identity in range(1, ndigits + 1):
            target_pos = target_identity - 1  # 0-based index
            digits = [NEUTRAL_DIGIT] * ndigits
            digits[target_pos] = target_identity
            
            trial = MSITTrial(
                digits=digits,
                target_identity=target_identity,
                target_pos_index=target_pos,
                is_simon=False,
                is_flanker=False,
                condition="congruent",
                flanker_value=NEUTRAL_DIGIT
            )
            stimuli.append(trial)
    
    elif cond == 1:
        # Simon-only: target at position != identity, flankers are 0
        for target_identity in range(1, ndigits + 1):
            identity_pos = target_identity - 1  # 0-based index
            # Generate all possible positions except the identity position
            for target_pos in range(ndigits):
                if target_pos != identity_pos:
                    digits = [NEUTRAL_DIGIT] * ndigits
                    digits[target_pos] = target_identity
                    
                    trial = MSITTrial(
                        digits=digits,
                        target_identity=target_identity,
                        target_pos_index=target_pos,
                        is_simon=True,
                        is_flanker=False,
                        condition="simon_only",
                        flanker_value=NEUTRAL_DIGIT
                    )
                    stimuli.append(trial)
    
    elif cond == 2:
        # Flanker-only: target at position = identity, flankers are single valid digit != target
        for target_identity in range(1, ndigits + 1):
            target_pos = target_identity - 1  # 0-based index
            # Generate all possible flanker digits (valid digits except target)
            for flanker_digit in range(1, ndigits + 1):
                if flanker_digit != target_identity:
                    digits = [flanker_digit] * ndigits
                    digits[target_pos] = target_identity
                    
                    trial = MSITTrial(
                        digits=digits,
                        target_identity=target_identity,
                        target_pos_index=target_pos,
                        is_simon=False,
                        is_flanker=True,
                        condition="flanker_only",
                        flanker_value=flanker_digit
                    )
                    stimuli.append(trial)
    
    elif cond == 3:
        # Simon+Flanker: target at position != identity, flankers are single valid digit 
        # != target and != (position_index+1)
        for target_identity in range(1, ndigits + 1):
            identity_pos = target_identity - 1  # 0-based index
            # Generate all possible positions except the identity position
            for target_pos in range(ndigits):
                if target_pos != identity_pos:
                    position_identity = target_pos + 1  # 1-based position identity
                    # Generate all possible flanker digits (exclude target and position identity)
                    for flanker_digit in range(1, ndigits + 1):
                        if flanker_digit != target_identity and flanker_digit != position_identity:
                            digits = [flanker_digit] * ndigits
                            digits[target_pos] = target_identity
                            
                            trial = MSITTrial(
                                digits=digits,
                                target_identity=target_identity,
                                target_pos_index=target_pos,
                                is_simon=True,
                                is_flanker=True,
                                condition="simon_flanker",
                                flanker_value=flanker_digit
                            )
                            stimuli.append(trial)
    
    elif cond == 8:
        # Simon with extended flankers: target at position != identity, flankers range from ndigits+1 to 9
        for target_identity in range(1, ndigits + 1):
            identity_pos = target_identity - 1  # 0-based index
            # Generate all possible positions except the identity position
            for target_pos in range(ndigits):
                if target_pos != identity_pos:
                    # Generate all possible flanker digits from ndigits+1 to 9
                    for flanker_digit in range(ndigits + 1, 10):
                        digits = [flanker_digit] * ndigits
                        digits[target_pos] = target_identity
                        
                        trial = MSITTrial(
                            digits=digits,
                            target_identity=target_identity,
                            target_pos_index=target_pos,
                            is_simon=True,
                            is_flanker=True,
                            condition="simon_extended_flanker",
                            flanker_value=flanker_digit
                        )
                        stimuli.append(trial)

    elif cond == 10:
        # Congruent with extended flankers: target at position = identity, flankers range from ndigits+1 to 9
        for target_identity in range(1, ndigits + 1):
            target_pos = target_identity - 1  # 0-based index
            # Generate all possible flanker digits from ndigits+1 to 9
            for flanker_digit in range(ndigits + 1, 10):
                digits = [flanker_digit] * ndigits
                digits[target_pos] = target_identity
                
                trial = MSITTrial(
                    digits=digits,
                    target_identity=target_identity,
                    target_pos_index=target_pos,
                    is_simon=False,
                    is_flanker=True,
                    condition="congruent_extended_flanker",
                    flanker_value=flanker_digit
                )
                stimuli.append(trial)

    elif cond == 11:
        ## replace identity_value and flanker values with letters
        ## identity_value must be different from flanker value
        ## 10 letters       
        letter_pool = ["d","f","h","i","k","m","o","q","s","u"][::-1][:ndigits]
        for id_val in letter_pool:
            remaining_letters = [letter for letter in letter_pool if letter != id_val]
            for flanker in remaining_letters:
                for id_pos in range(ndigits):
                    digits = [flanker]*ndigits
                    digits[id_pos] = id_val
                    
                    trial = MSITTrial(
                        digits=digits,
                        target_identity=id_val,
                        target_pos_index=id_pos,
                        is_simon=False,
                        is_flanker=True,
                        condition="all_letters",
                        flanker_value=flanker
                    )
                    stimuli.append(trial)

    elif cond == 12:
        ## Simon condition but with the flanker value replaced with "#"
        for flanker_value in ["#","*"]:
            for target_identity in range(1, ndigits + 1):
                identity_pos = target_identity - 1  # 0-based index
                # Generate all possible positions except the identity position
                for target_pos in range(ndigits):
                    if target_pos == identity_pos:
                        digits = [flanker_value] * ndigits
                        digits[target_pos] = target_identity
                        
                        trial = MSITTrial(
                            digits=digits,
                            target_identity=target_identity,
                            target_pos_index=target_pos,
                            is_simon=True,
                            is_flanker=True,
                            condition="CgFkSymb",
                            flanker_value=flanker_value
                        )
                        stimuli.append(trial)
                

    elif cond == 13:
        ## Simon condition but with the flanker value replaced with "#"
        for flanker_value in ["#","*"]:
            for target_identity in range(1, ndigits + 1):
                identity_pos = target_identity - 1  # 0-based index
                # Generate all possible positions except the identity position
                for target_pos in range(ndigits):
                    if target_pos != identity_pos:
                        digits = [flanker_value] * ndigits
                        digits[target_pos] = target_identity
                        
                        trial = MSITTrial(
                            digits=digits,
                            target_identity=target_identity,
                            target_pos_index=target_pos,
                            is_simon=True,
                            is_flanker=True,
                            condition="SmFkSymb",
                            flanker_value=flanker_value
                        )
                        stimuli.append(trial)
                

    return stimuli


def save_msit_data(full_input: str, answer: List[int], self_judge: str, 
                   ndigits: int, nstims: int, by_image: bool, restriction: str = None):
    """
    Save MSIT data to JSON file in data/pilot_msit/ directory.
    
    Parameters
    ----------
    full_input : str
        The complete input prompt with instructions and stimuli
    answer : List[int]
        List of correct answers (target positions)
    self_judge : str
        Self-judgment instruction string
    ndigits : int
        Number of digit positions
    nstims : int
        Number of stimuli
    by_image : bool
        Whether the task uses images
    restriction : str, optional
        Restriction type (e.g., "not1by1", "strict")
    """
    # Create directory if it doesn't exist
    os.makedirs("data/pilot_msit", exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine modality
    modality = "image" if by_image else "text"
    
    # Generate filename
    restriction_str = restriction if restriction else "none"
    filename = f"{timestamp}_digits-{ndigits}_stims-{nstims}_{modality}_restrict-{restriction_str}.json"
    filepath = os.path.join("data/pilot_msit", filename)
    
    # Prepare data
    data = {
        "full_input": full_input,
        "answer": answer,
        "self_judge": self_judge,
        "metadata": {
            "ndigits": ndigits,
            "nstims": nstims,
            "by_image": by_image,
            "modality": modality,
            "restriction": restriction,
            "timestamp": timestamp
        }
    }
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Data saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    # demo()

    ### using gen_msit_sequence
    # ndigits = 4
    # nstims = 60
    # s,answer = gen_msit_sequence(
    #     ndigits=ndigits,
    #     nstims=nstims,
    #     stim_type=3,
    #     separate_by="\n",
    #     seed=None) 

    ### using gen_msit_mixed_sequence
    # stim_types = [0, 3]*3
    # ndigits = len(stim_types)
    # s,answers,identities = gen_msit_mixed_sequence(stim_types, ndigits=4, separate_by="\n", seed=None)

    ## using gen_msit_rep_sequence
    # ndigits = 4
    # stim_types = "10"
    # nrep = 1
    # s,answers,identities,types,flanker_values = gen_msit_rep_sequence(ndigits, stim_types, nrep, separate_by="\n", seed=None)

    # print(s)
    # print(answers)
    # print(identities)
    # print(types)
    # print(flanker_values)
    
    # by_image = True
    # restriction = "not1by1"
    
    # full_input = msit_instruciton(ndigits, by_image=by_image, restriction=restriction) + "\n\n" + s
    # self_judge = self_judge_instruction(answers)
    
    # Save data to JSON file
    # save_msit_data(full_input, answer, self_judge, ndigits, nstims, by_image, restriction)

    # print(full_input)
    # print(answers)
    # print(self_judge)

    a = generate_exhastive_stims_by_type(5,cond=1)
    print(a)