"""
Comprehensive diagnostic script to verify channel ordering consistency.
Run this after training to ensure channel mapping is correct.
"""

import h5py
import numpy as np
import json
import glob
import torch
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import load_processed_data

def check_channel_order_in_processed_data(filepath):
    """Check channel order saved in processed data file"""
    print("=" * 80)
    print("📊 CHECKING PROCESSED DATA FILE")
    print("=" * 80)
    print(f"File: {filepath}\n")
    
    with h5py.File(filepath, 'r') as hf:
        # Check metadata
        metadata = {}
        if 'metadata' in hf:
            for key in hf['metadata'].attrs:
                metadata[key] = hf['metadata'].attrs[key]
        
        print(f"Metadata:")
        print(f"   n_channels: {metadata.get('n_channels', 'N/A')}")
        print(f"   nsteps: {metadata.get('nsteps', 'N/A')}")
        
        # Check data selections
        print(f"\nData Selections:")
        if 'data_selections' in hf:
            selections_group = hf['data_selections']
            if 'selections_json' in selections_group.attrs:
                selections = json.loads(selections_group.attrs['selections_json'])
                
                if 'training_channel_names' in selections:
                    channel_names = selections['training_channel_names']
                    print(f"   ✅ training_channel_names: {channel_names}")
                    print(f"   Channel order in saved STATE tensors:")
                    for idx, name in enumerate(channel_names):
                        print(f"      Channel {idx}: {name}")
                    return channel_names
                else:
                    print(f"   ⚠️ No training_channel_names found!")
        else:
            print(f"   ⚠️ No data_selections group found!")
    
    return None


def check_normalization_file(json_file):
    """Check channel order in normalization parameters"""
    print("\n" + "=" * 80)
    print("📋 CHECKING NORMALIZATION PARAMETERS FILE")
    print("=" * 80)
    print(f"File: {json_file}\n")
    
    with open(json_file, 'r') as f:
        norm_config = json.load(f)
    
    # Check selection summary
    if 'selection_summary' in norm_config:
        summary = norm_config['selection_summary']
        if 'training_channels' in summary:
            channels = summary['training_channels']
            print(f"✅ Training channels order: {channels}")
            return channels
    
    # Check spatial channels with training positions
    if 'spatial_channels' in norm_config:
        print(f"Spatial Channels with Training Positions:")
        channels_with_pos = []
        for var_name, params in norm_config['spatial_channels'].items():
            if params.get('selected_for_training', False):
                pos = params.get('training_position', None)
                channels_with_pos.append((pos, var_name))
        
        # Sort by training position
        channels_with_pos.sort(key=lambda x: x[0] if x[0] is not None else 999)
        channel_names = [name for _, name in channels_with_pos]
        print(f"   Channel order: {channel_names}")
        return channel_names
    
    return None


def check_model_predictions_shape(state_pred, channel_names):
    """Verify state_pred tensor shape matches channel_names"""
    print("\n" + "=" * 80)
    print("🤖 CHECKING MODEL PREDICTIONS")
    print("=" * 80)
    
    if state_pred is None:
        print("   ⚠️ state_pred is None - cannot verify")
        return
    
    print(f"   state_pred shape: {state_pred.shape}")
    print(f"   Expected format: (n_cases, n_timesteps, n_channels, Nx, Ny, Nz)")
    
    if len(state_pred.shape) == 6:
        n_cases, n_timesteps, n_channels, Nx, Ny, Nz = state_pred.shape
        print(f"   n_channels in tensor: {n_channels}")
        print(f"   n_channels from channel_names: {len(channel_names) if channel_names else 'N/A'}")
        
        if channel_names and n_channels == len(channel_names):
            print(f"\n   ✅ Channel count matches!")
            print(f"   Channel order in tensor (state_pred[:, :, channel_idx, ...]):")
            for idx, name in enumerate(channel_names):
                # Sample a few values to verify
                sample_data = state_pred[0, 0, idx, :, :, :].cpu().numpy() if torch.is_tensor(state_pred) else state_pred[0, 0, idx, :, :, :]
                data_range = f"[{np.nanmin(sample_data):.6f}, {np.nanmax(sample_data):.6f}]"
                print(f"      Channel {idx}: {name:15s} - Sample range: {data_range}")
        else:
            print(f"   ❌ MISMATCH: Channel count doesn't match!")
    else:
        print(f"   ⚠️ Unexpected tensor shape: {state_pred.shape}")


def verify_channel_mapping_consistency():
    """Verify channel mapping is consistent across all sources"""
    print("=" * 80)
    print("🔍 COMPREHENSIVE CHANNEL ORDER VERIFICATION")
    print("=" * 80)
    
    # Find processed data files
    processed_files = glob.glob('./processed_data/processed_data_*.h5')
    if not processed_files:
        print("❌ No processed data files found in ./processed_data/")
        return
    
    latest_file = max(processed_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"\n📁 Latest processed data file: {latest_file}")
    
    # Check channel order in processed data
    channel_names_from_data = check_channel_order_in_processed_data(latest_file)
    
    # Check normalization file
    norm_files = glob.glob('./processed_data/normalization_parameters_*.json')
    channel_names_from_norm = None
    if norm_files:
        latest_norm = max(norm_files, key=lambda x: Path(x).stat().st_mtime)
        channel_names_from_norm = check_normalization_file(latest_norm)
    
    # Compare
    print("\n" + "=" * 80)
    print("🔍 COMPARISON")
    print("=" * 80)
    
    if channel_names_from_data and channel_names_from_norm:
        if channel_names_from_data == channel_names_from_norm:
            print(f"✅ Channel orders MATCH!")
            print(f"   Order: {channel_names_from_data}")
        else:
            print(f"❌ MISMATCH DETECTED!")
            print(f"   From processed data: {channel_names_from_data}")
            print(f"   From norm file: {channel_names_from_norm}")
            print(f"\n   ⚠️ This mismatch will cause incorrect visualization!")
    
    # Check grid_search_training.py hardcoded mapping
    print("\n" + "=" * 80)
    print("📝 CHECKING grid_search_training.py CHANNEL_NAMES_MAP")
    print("=" * 80)
    try:
        from grid_search_training import CHANNEL_NAMES_MAP
        if channel_names_from_data:
            n_channels = len(channel_names_from_data)
            if n_channels in CHANNEL_NAMES_MAP:
                hardcoded = CHANNEL_NAMES_MAP[n_channels]
                if hardcoded == channel_names_from_data:
                    print(f"✅ Hardcoded mapping MATCHES actual data!")
                    print(f"   {hardcoded}")
                else:
                    print(f"❌ MISMATCH: Hardcoded mapping doesn't match!")
                    print(f"   Hardcoded: {hardcoded}")
                    print(f"   Actual:    {channel_names_from_data}")
                    print(f"\n   ⚠️ WARNING: grid_search_training.py CHANNEL_NAMES_MAP needs to be updated!")
            else:
                print(f"⚠️ No hardcoded mapping for {n_channels} channels")
    except Exception as e:
        print(f"⚠️ Could not check CHANNEL_NAMES_MAP: {e}")
    
    print("\n" + "=" * 80)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 80)
    print("\n💡 RECOMMENDATIONS:")
    print("   1. Ensure training_channel_names in processed data matches actual channel order")
    print("   2. Update CHANNEL_NAMES_MAP in grid_search_training.py if needed")
    print("   3. Verify channel_names are passed correctly to visualization dashboard")
    print("   4. Check that field_keys match channel_names for correct normalization")


if __name__ == "__main__":
    verify_channel_mapping_consistency()

