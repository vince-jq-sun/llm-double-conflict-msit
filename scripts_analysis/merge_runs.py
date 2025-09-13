#!/usr/bin/env python3
"""
MSIT Test Run Merger

This script merges multiple MSIT test run folders into a single combined run.
The input folders should have consistent basic parameters but can have different
numbers of sessions. The merged run maintains the same format as original runs.

Usage:
    python merge_runs.py

Configure the folders to merge in the __main__ section below.
"""

import json
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def load_run_metadata(folder_path):
    """Load run metadata from a folder."""
    metadata_path = Path(folder_path) / "run_metadata.json"
    if not metadata_path.exists():
        print(f"Warning: No run_metadata.json found in {folder_path}")
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in metadata file: {metadata_path}")
        return None


def load_session_data(session_file_path):
    """Load data from a session JSON file."""
    try:
        with open(session_file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Session file not found: {session_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {session_file_path}")
        return None


def get_session_files(folder_path):
    """Get all session JSON files from a folder, sorted by session number."""
    folder = Path(folder_path)
    session_files = list(folder.glob("session_*.json"))
    
    # Sort by session number
    def extract_session_num(filepath):
        try:
            # Extract number from filename like "session_001.json"
            filename = filepath.name
            num_str = filename.replace("session_", "").replace(".json", "")
            return int(num_str)
        except:
            return 0
    
    session_files.sort(key=extract_session_num)
    return session_files


def validate_run_compatibility(metadata_list):
    """
    Validate that runs have compatible parameters for merging.
    Returns True if compatible, False otherwise.
    """
    if not metadata_list:
        print("Error: No metadata found")
        return False
    
    # Parameters that must be identical across runs
    critical_params = ['ndigits', 'stim_types_str', 'nrep', 'restriction', 'model_name', 'max_tokens']
    
    reference_metadata = metadata_list[0]
    reference_params = reference_metadata.get('parameters', {})
    
    for i, metadata in enumerate(metadata_list[1:], 1):
        current_params = metadata.get('parameters', {})
        
        for param in critical_params:
            ref_val = reference_params.get(param)
            curr_val = current_params.get(param)
            
            if ref_val != curr_val:
                print(f"Error: Parameter '{param}' mismatch between runs:")
                print(f"  Run 0: {ref_val}")
                print(f"  Run {i}: {curr_val}")
                return False
    
    print("✓ All runs have compatible parameters")
    return True


def merge_runs(folder_paths, output_folder, merged_nick_name=None):
    """
    Merge multiple run folders into a single combined run.
    
    Args:
        folder_paths: List of paths to run folders to merge
        output_folder: Path where the merged run should be saved
        merged_nick_name: Optional nickname for the merged run
    """
    print(f"Merging {len(folder_paths)} runs...")
    
    # Load and validate metadata from all runs
    metadata_list = []
    for folder_path in folder_paths:
        metadata = load_run_metadata(folder_path)
        if metadata is None:
            print(f"Error: Could not load metadata from {folder_path}")
            return False
        metadata_list.append(metadata)
    
    # Validate compatibility
    if not validate_run_compatibility(metadata_list):
        return False
    
    # Create output directory
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all sessions from all runs
    all_sessions = []
    total_sessions = 0
    
    for i, folder_path in enumerate(folder_paths):
        print(f"Processing run {i+1}: {folder_path}")
        session_files = get_session_files(folder_path)
        
        for session_file in session_files:
            session_data = load_session_data(session_file)
            if session_data is not None:
                all_sessions.append(session_data)
        
        print(f"  Added {len(session_files)} sessions")
        total_sessions += len(session_files)
    
    print(f"Total sessions collected: {len(all_sessions)}")
    
    # Renumber sessions sequentially
    for i, session_data in enumerate(all_sessions, 1):
        session_data['session_id'] = i
    
    # Create merged metadata
    reference_metadata = metadata_list[0].copy()
    
    # Update session count and timestamp
    reference_metadata['total_sessions'] = len(all_sessions)
    reference_metadata['timestamp'] = datetime.now().isoformat()
    
    # Update nick_name if provided
    if merged_nick_name:
        reference_metadata['nick_name'] = merged_nick_name
    else:
        # Create a combined nick_name from source runs
        source_nicks = []
        for metadata in metadata_list:
            nick = metadata.get('nick_name', 'unknown')
            if nick not in source_nicks:
                source_nicks.append(nick)
        reference_metadata['nick_name'] = f"merged_{'_'.join(source_nicks)}"
    
    # Add merge information
    reference_metadata['merged_from'] = [str(Path(fp).name) for fp in folder_paths]
    reference_metadata['merge_timestamp'] = datetime.now().isoformat()
    
    # Save merged metadata
    metadata_output_path = output_path / "run_metadata.json"
    with open(metadata_output_path, 'w') as f:
        json.dump(reference_metadata, f, indent=2)
    
    print(f"✓ Saved merged metadata to {metadata_output_path}")
    
    # Save all sessions
    for session_data in all_sessions:
        session_id = session_data['session_id']
        session_filename = f"session_{session_id:03d}.json"
        session_output_path = output_path / session_filename
        
        with open(session_output_path, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    print(f"✓ Saved {len(all_sessions)} sessions to {output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("MERGE SUMMARY")
    print("="*50)
    print(f"Source runs: {len(folder_paths)}")
    for i, folder_path in enumerate(folder_paths):
        metadata = metadata_list[i]
        sessions = metadata.get('total_sessions', 'unknown')
        nick = metadata.get('nick_name', 'unknown')
        print(f"  {i+1}. {Path(folder_path).name} ({nick}) - {sessions} sessions")
    
    print(f"\nMerged run:")
    print(f"  Location: {output_path}")
    print(f"  Nick name: {reference_metadata['nick_name']}")
    print(f"  Total sessions: {len(all_sessions)}")
    model_name = reference_metadata.get('parameters', {}).get('model_name', 'unknown')
    print(f"  Model: {model_name}")
    
    return True


if __name__ == "__main__":
    # ================================================================
    # USER CONFIGURATION SECTION
    # ================================================================
    
    # List of folder paths to merge (modify these paths as needed)
    folders_to_merge = [
        # Example paths - modify these to your actual folder paths
        "/Users/vince/Documents/llm_control/data/msit_pilot_outputs/20250904_150941_model-gpt-4.1-nano_sessions-60_mixed-Rm",
        "/Users/vince/Documents/llm_control/data/msit_pilot_outputs/20250904_152123_model-gpt-4.1-nano_sessions-60_mixed-Rm",
        # Add more folder paths here as needed
    ]
    
    # Output folder for the merged run
    output_folder = "/Users/vince/Documents/llm_control/data/msit_pilot_outputs/merged_run_example"
    
    # Optional: Custom nickname for the merged run (leave None for auto-generated)
    merged_nickname = "mixed-Rm-Merged"
    
    # ================================================================
    # END USER CONFIGURATION
    # ================================================================
    
    # Validate input
    if len(folders_to_merge) < 2:
        print("Error: Need at least 2 folders to merge")
        sys.exit(1)
    
    # Check if folders exist
    for folder in folders_to_merge:
        if not Path(folder).exists():
            print(f"Error: Folder does not exist: {folder}")
            sys.exit(1)
    
    # Perform the merge
    success = merge_runs(folders_to_merge, output_folder, merged_nickname)
    
    if success:
        print(f"\n✓ Successfully merged runs into: {output_folder}")
    else:
        print("\n✗ Merge failed")
        sys.exit(1)
