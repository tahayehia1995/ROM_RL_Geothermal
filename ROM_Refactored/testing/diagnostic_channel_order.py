"""
Diagnostic script to check channel ordering in saved tensors and processed data files.
This helps identify channel mapping mismatches between training and visualization.
"""

import h5py
import numpy as np
import json
import glob
from pathlib import Path
import torch

def check_processed_data_channels(filepath):
    """Check channel order in processed data file"""
    print("=" * 80)
    print(f"📊 CHECKING PROCESSED DATA FILE: {filepath}")
    print("=" * 80)
    
    with h5py.File(filepath, 'r') as hf:
        # Check metadata
        print("\n1️⃣ METADATA:")
        if 'metadata' in hf:
            metadata = {}
            for key in hf['metadata'].attrs:
                metadata[key] = hf['metadata'].attrs[key]
                print(f"   {key}: {metadata[key]}")
        
        # Check data selections (contains training_channel_names)
        print("\n2️⃣ DATA SELECTIONS (Channel Order):")
        if 'data_selections' in hf:
            selections_group = hf['data_selections']
            if 'selections_json' in selections_group.attrs:
                selections = json.loads(selections_group.attrs['selections_json'])
                
                if 'training_channel_names' in selections:
                    channel_names = selections['training_channel_names']
                    print(f"   ✅ Found training_channel_names: {channel_names}")
                    print(f"   Channel count: {len(channel_names)}")
                    for idx, name in enumerate(channel_names):
                        print(f"      Channel {idx}: {name}")
                else:
                    print("   ⚠️ No training_channel_names found in selections")
                    if 'selected_states' in selections:
                        print(f"   Found selected_states: {selections['selected_states']}")
        else:
            print("   ⚠️ No data_selections group found")
        
        # Check STATE tensor shape
        print("\n3️⃣ STATE TENSOR SHAPE:")
        if 'train' in hf and 'STATE' in hf['train']:
            state_group = hf['train']['STATE']
            if 'data' in state_group:
                state_shape = state_group['data'].shape
                print(f"   STATE_train shape: {state_shape}")
                print(f"   Format: (n_samples, timesteps, channels, Nx, Ny, Nz)")
                print(f"   Number of channels: {state_shape[2]}")
            else:
                print("   ⚠️ No STATE data found")
        
        # Check normalization parameters
        print("\n4️⃣ NORMALIZATION PARAMETERS:")
        if 'normalization_parameters' in hf:
            norm_group = hf['normalization_parameters']
            if 'spatial_channels' in norm_group.attrs:
                spatial_channels = json.loads(norm_group.attrs['spatial_channels'])
                print(f"   Found {len(spatial_channels)} spatial channels:")
                for var_name, params in spatial_channels.items():
                    training_pos = params.get('training_position', 'N/A')
                    selected = params.get('selected_for_training', False)
                    print(f"      {var_name}: training_position={training_pos}, selected={selected}")
        else:
            print("   ⚠️ No normalization_parameters group found")
    
    print("\n" + "=" * 80)


def check_model_predictions(model_path, state_pred_path=None):
    """Check channel order in model predictions"""
    print("=" * 80)
    print(f"🤖 CHECKING MODEL PREDICTIONS")
    print("=" * 80)
    
    # Check if state_pred is saved separately
    if state_pred_path and Path(state_pred_path).exists():
        print(f"\n📁 Loading predictions from: {state_pred_path}")
        state_pred = torch.load(state_pred_path, map_location='cpu')
        print(f"   State predictions shape: {state_pred.shape}")
        print(f"   Format: (n_cases, n_timesteps, n_channels, Nx, Ny, Nz)")
        print(f"   Number of channels: {state_pred.shape[2]}")
    
    print("\n" + "=" * 80)


def check_normalization_file(json_file):
    """Check channel order in normalization parameters JSON file"""
    print("=" * 80)
    print(f"📋 CHECKING NORMALIZATION FILE: {json_file}")
    print("=" * 80)
    
    with open(json_file, 'r') as f:
        norm_config = json.load(f)
    
    # Check selection summary
    if 'selection_summary' in norm_config:
        summary = norm_config['selection_summary']
        if 'training_channels' in summary:
            channels = summary['training_channels']
            print(f"\n✅ Training channels order: {channels}")
            for idx, name in enumerate(channels):
                print(f"   Channel {idx}: {name}")
    
    # Check spatial channels with training positions
    if 'spatial_channels' in norm_config:
        print(f"\n📊 Spatial Channels with Training Positions:")
        channels_with_pos = []
        for var_name, params in norm_config['spatial_channels'].items():
            if params.get('selected_for_training', False):
                pos = params.get('training_position', None)
                channels_with_pos.append((pos, var_name))
        
        # Sort by training position
        channels_with_pos.sort(key=lambda x: x[0] if x[0] is not None else 999)
        for pos, name in channels_with_pos:
            print(f"   Position {pos}: {name}")
    
    print("\n" + "=" * 80)


def compare_channel_orders():
    """Compare channel orders across different sources"""
    print("=" * 80)
    print("🔍 COMPARING CHANNEL ORDERS")
    print("=" * 80)
    
    # Find processed data files
    processed_files = glob.glob('./processed_data/processed_data_*.h5')
    if processed_files:
        latest_file = max(processed_files, key=lambda x: Path(x).stat().st_mtime)
        print(f"\n📁 Latest processed data file: {latest_file}")
        check_processed_data_channels(latest_file)
    
    # Find normalization parameter files
    norm_files = glob.glob('./processed_data/normalization_parameters_*.json')
    if norm_files:
        latest_norm = max(norm_files, key=lambda x: Path(x).stat().st_mtime)
        print(f"\n📋 Latest normalization file: {latest_norm}")
        check_normalization_file(latest_norm)
    
    # Check grid_search_training.py channel mapping
    print("\n" + "=" * 80)
    print("📝 CHECKING grid_search_training.py CHANNEL_NAMES_MAP")
    print("=" * 80)
    try:
        import sys
        sys.path.insert(0, '.')
        from grid_search_training import CHANNEL_NAMES_MAP
        print(f"   CHANNEL_NAMES_MAP: {CHANNEL_NAMES_MAP}")
        if 4 in CHANNEL_NAMES_MAP:
            print(f"   For 4 channels: {CHANNEL_NAMES_MAP[4]}")
            print(f"   ⚠️ WARNING: This is hardcoded and may not match actual data!")
    except Exception as e:
        print(f"   ⚠️ Could not load CHANNEL_NAMES_MAP: {e}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose channel ordering issues')
    parser.add_argument('--processed-data', type=str, help='Path to processed data H5 file')
    parser.add_argument('--norm-file', type=str, help='Path to normalization parameters JSON file')
    parser.add_argument('--compare', action='store_true', help='Compare all channel orders')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_channel_orders()
    else:
        if args.processed_data:
            check_processed_data_channels(args.processed_data)
        if args.norm_file:
            check_normalization_file(args.norm_file)
        
        if not args.processed_data and not args.norm_file:
            # Default: compare all
            compare_channel_orders()

