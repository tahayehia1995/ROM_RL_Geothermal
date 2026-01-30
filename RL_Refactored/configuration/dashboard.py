"""
RL Configuration Dashboard
Interactive dashboard for configuring RL training parameters, loading ROM models, and generating initial states.
"""
import sys
from pathlib import Path
import os
import glob
import re
import numpy as np
import torch
import h5py
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from datetime import datetime

# Add ROM_Refactored to Python path so we can import it as a package
rom_refactored_path = Path(__file__).parent.parent.parent / 'ROM_Refactored'
rom_refactored_parent = rom_refactored_path.parent

# Add parent directory to path if not already there
if str(rom_refactored_parent) not in sys.path:
    sys.path.insert(0, str(rom_refactored_parent))

# Also add ROM_Refactored itself to path so 'model' imports work
# This is needed because rom_wrapper.py uses 'from model.models.mse2c' 
# which expects 'model' to be findable as a top-level module
if str(rom_refactored_path) not in sys.path:
    sys.path.insert(0, str(rom_refactored_path))

# Import ROM_Refactored modules as packages
try:
    from ROM_Refactored.model.training.rom_wrapper import ROMWithE2C
except ImportError as e:
    print(f"Warning: Could not import ROMWithE2C: {e}")
    import traceback
    traceback.print_exc()
    ROMWithE2C = None

try:
    from ROM_Refactored.data_preprocessing import load_processed_data
except ImportError as e:
    print(f"Warning: Could not import load_processed_data: {e}")
    import traceback
    traceback.print_exc()
    load_processed_data = None

# Import Config from RL_Refactored utilities (which re-exports from ROM_Refactored)
try:
    from RL_Refactored.utilities import Config
except ImportError:
    # Fallback: try importing directly from ROM_Refactored
    try:
        from ROM_Refactored.utilities.config_loader import Config
    except ImportError:
        print("Warning: Could not import Config")
        Config = None

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    widgets = None
    display = None
    clear_output = None
    HTML = None

# =====================================
# SECTION: STATE PROCESSING & Z0 GENERATION
# =====================================

def load_state_data_from_h5(state_name, state_folder, device):
    """Load state data from H5 file"""
    state_file = os.path.join(state_folder, f'batch_spatial_properties_{state_name}.h5')
    
    if not os.path.exists(state_file):
        raise FileNotFoundError(f"State file not found: {state_file}")
    
    print(f"   Loading {state_name} from {state_file}")
    with h5py.File(state_file, 'r') as hf:
        # Load data: shape is (batch, time, Nx, Ny, Nz)
        data = np.array(hf['data'])
        print(f"   {state_name} data shape: {data.shape}")
    
    # Convert to tensor
    return torch.tensor(data, dtype=torch.float32)

def apply_dashboard_scaling(data, state_name, rl_config, device):
    """
    Apply TRAINING-ONLY normalization parameters (FIXES data leakage)
    🎯 PERFECT COMPATIBILITY: Uses corrected training-only parameters
    """
    print(f"   🔧 Applying TRAINING-ONLY normalization to {state_name}...")
    
    # 🎯 CRITICAL: Get TRAINING-ONLY normalization parameters (fixes data leakage)
    training_params = rl_config.get('training_only_normalization_params', {})
    
    # Use training-only parameters if available (prevents data leakage)
    # Otherwise fallback to preprocessing parameters, then emergency normalization
    preprocessing_params = rl_config.get('preprocessing_normalization_params', {})
    
    if training_params:
        print(f"      ✅ Using TRAINING-ONLY parameters (NO data leakage)")
        return apply_training_only_normalization(data, state_name, training_params, device)
    elif preprocessing_params:
        print(f"      ⚠️ Using preprocessing parameters (may contain data leakage)")
        return apply_preprocessing_normalization_legacy(data, state_name, preprocessing_params, device)
    else:
        print(f"      🚨 No normalization parameters found - using emergency normalization")
        return apply_emergency_fallback_normalization(data, state_name, device)

def apply_training_only_normalization(data, state_name, training_params, device):
    """
    Apply TRAINING-ONLY normalization parameters (eliminates data leakage)
    """
    if state_name not in training_params:
        print(f"      ❌ No training-only parameters for {state_name}")
        return apply_emergency_fallback_normalization(data, state_name, device)
    
    norm_params = training_params[state_name]
    param_min = float(norm_params.get('min', 0.0))
    param_max = float(norm_params.get('max', 1.0))
    norm_type = norm_params.get('type', 'minmax')
    
    print(f"      📊 Training-only {norm_type.upper()}: [{param_min:.6f}, {param_max:.6f}] → [0, 1]")
    
    # Handle inactive cells for spatial data
    if state_name in ['SW', 'SG', 'PRES', 'POROS', 'PERMI', 'PERMJ', 'PERMK', 'TEMP', 'VPOROSGEO', 'VPOROSTGEO']:
        # Identify inactive cells (same logic as preprocessing)
        if state_name in ['PRES', 'SW', 'SG', 'POROS', 'TEMP', 'VPOROSGEO', 'VPOROSTGEO']:
            active_mask = data > 0.0
        elif 'PERM' in state_name:
            active_mask = data > 0.0
        else:
            active_mask = data >= 0.0
        
        print(f"      • Active cells: {torch.sum(active_mask).item():,} / {data.numel():,}")
        
        # Start with data copy to preserve inactive cells
        scaled_data = data.clone()
        
        if param_max > param_min:
            # Apply training-only transformation
            scaled_all = (data - param_min) / (param_max - param_min)
            # Only update active cells
            scaled_data[active_mask] = scaled_all[active_mask]
            # Inactive cells remain unchanged
            
            print(f"      ✅ TRAINING-ONLY normalization applied to active cells")
        else:
            print(f"      ⚠️ Warning: param_min == param_max ({param_min:.6f})")
        
        return scaled_data.to(device)
    
    else:
        # For timeseries data (controls/observations), apply directly
        if param_max > param_min:
            scaled_data = (data - param_min) / (param_max - param_min)
            print(f"      ✅ TRAINING-ONLY normalization applied to timeseries data")
        else:
            scaled_data = data.clone()
            print(f"      ⚠️ Warning: param_min == param_max ({param_min:.6f})")
        
        return scaled_data.to(device)

def apply_preprocessing_normalization_legacy(data, state_name, preprocessing_params, device):
    """
    Apply preprocessing normalization parameters (fallback when training-only params unavailable)
    Note: May contain data leakage if preprocessing used full dataset
    """
    print(f"      ⚠️ Using preprocessing parameters (data leakage possible)")
    
    # Try to find parameters in spatial_channels (for state variables)
    if state_name in preprocessing_params.get('spatial_channels', {}):
        spatial_config = preprocessing_params['spatial_channels'][state_name]
        norm_params = spatial_config.get('parameters', {})
        norm_type = spatial_config.get('normalization_type', 'minmax')
        
        print(f"      📊 Legacy normalization type: {norm_type.upper()}")
        
        # Apply the legacy normalization
        return apply_preprocessing_normalization(data, state_name, norm_params, norm_type, device)
    
    # Try to find parameters in control_variables
    elif state_name in preprocessing_params.get('control_variables', {}):
        control_config = preprocessing_params['control_variables'][state_name]
        norm_params = control_config.get('parameters', {})
        norm_type = control_config.get('normalization_type', 'minmax')
        
        print(f"      ✅ Using legacy preprocessing parameters for control {state_name}")
        print(f"      📊 Normalization type: {norm_type.upper()}")
        
        return apply_preprocessing_normalization(data, state_name, norm_params, norm_type, device)
    
    # Try to find parameters in observation_variables  
    elif state_name in preprocessing_params.get('observation_variables', {}):
        obs_config = preprocessing_params['observation_variables'][state_name]
        norm_params = obs_config.get('parameters', {})
        norm_type = obs_config.get('normalization_type', 'minmax')
        
        print(f"      ✅ Using legacy preprocessing parameters for observation {state_name}")
        print(f"      📊 Normalization type: {norm_type.upper()}")
        
        return apply_preprocessing_normalization(data, state_name, norm_params, norm_type, device)
    
    else:
        print(f"      ❌ CRITICAL: No preprocessing parameters found for {state_name}!")
        print(f"      💡 Available parameters: {list(preprocessing_params.keys())}")
        print(f"      🔧 This indicates preprocessing dashboard hasn't been run yet")
        
        # Use emergency normalization (should not happen in normal workflow)
        print(f"      🚨 Using emergency normalization")
        return apply_emergency_fallback_normalization(data, state_name, device)

def apply_preprocessing_normalization(data, state_name, norm_params, norm_type, device):
    """
    Apply the EXACT same normalization logic as the preprocessing dashboard
    """
    if norm_type == 'none':
        print(f"      📊 No normalization applied (values preserved)")
        return data.to(device)
    
    elif norm_type == 'log':
        print(f"      📊 Applying LOG normalization (identical to preprocessing)")
        
        # Use EXACT same log normalization logic as preprocessing dashboard
        epsilon = float(norm_params.get('epsilon', 1e-8))
        log_min = float(norm_params.get('log_min', 0.0))
        log_max = float(norm_params.get('log_max', 1.0))
        min_positive = float(norm_params.get('min_positive', epsilon))
        
        # Apply identical transformation
        positive_data = data[data > 0]
        if len(positive_data) > 0:
            min_pos = min_positive
            data_shifted = torch.maximum(data, torch.tensor(min_pos, device=device))
        else:
            data_shifted = data + epsilon
        
        log_data = torch.log(data_shifted + epsilon)
        
        if log_max > log_min:
            scaled_data = (log_data - log_min) / (log_max - log_min)
        else:
            scaled_data = torch.zeros_like(log_data)
        
        print(f"      ✅ LOG normalization applied: log range [{log_min:.6f}, {log_max:.6f}] → [0, 1]")
        return scaled_data.to(device)
    
    else:  # minmax normalization (default)
        print(f"      📊 Applying MIN-MAX normalization (identical to preprocessing)")
        
        # Use EXACT same min-max parameters as preprocessing dashboard
        param_min = float(norm_params.get('min', 0.0))
        param_max = float(norm_params.get('max', 1.0))
        
        # Handle inactive cells for spatial data
        if state_name in ['SW', 'SG', 'PRES', 'POROS', 'PERMI', 'PERMJ', 'PERMK', 'TEMP', 'VPOROSGEO', 'VPOROSTGEO']:
            # Identify inactive cells (same logic as preprocessing)
            if state_name in ['PRES', 'SW', 'SG', 'POROS', 'TEMP', 'VPOROSGEO', 'VPOROSTGEO']:
                active_mask = data > 0.0
                inactive_marker = -0.145038 if state_name == 'PRES' else -1.0
            elif 'PERM' in state_name:
                active_mask = data > 0.0
                inactive_marker = -1.0
            else:
                active_mask = data >= 0.0
                inactive_marker = -1.0
            
            print(f"      • Active cells: {torch.sum(active_mask).item():,} / {data.numel():,}")
            
            # Start with data copy to preserve inactive cells
            scaled_data = data.clone()
            
            if param_max > param_min:
                # Apply same transformation as preprocessing
                scaled_all = (data - param_min) / (param_max - param_min)
                # Only update active cells
                scaled_data[active_mask] = scaled_all[active_mask]
                # Inactive cells remain unchanged
                
                print(f"      ✅ MIN-MAX applied to active cells, inactive cells preserved")
            else:
                print(f"      ⚠️ Warning: param_min == param_max ({param_min:.6f})")
            
            return scaled_data.to(device)
        
        else:
            # For timeseries data (controls/observations), apply directly
            if param_max > param_min:
                scaled_data = (data - param_min) / (param_max - param_min)
                print(f"      ✅ MIN-MAX normalization applied to timeseries data")
            else:
                scaled_data = data.clone()
                print(f"      ⚠️ Warning: param_min == param_max ({param_min:.6f})")
            
            return scaled_data.to(device)

def apply_emergency_fallback_normalization(data, state_name, device):
    """
    Emergency normalization when no parameters are available
    Should not be used in normal workflow - indicates configuration issue
    """
    print(f"      ⚠️ WARNING: Using emergency normalization for {state_name}")
    print(f"      💡 This should not happen - check normalization parameters")
    # Simple min-max normalization using data statistics
    data_min = torch.min(data)
    data_max = torch.max(data)
    if data_max > data_min:
        scaled_data = (data - data_min) / (data_max - data_min)
    else:
        scaled_data = torch.zeros_like(data)
    return scaled_data.to(device)

def calculate_training_only_normalization_params(data_dir=None, selected_states=None):
    """
    Calculate normalization parameters from training split only
    This replicates exactly what was done during training
    
    Args:
        data_dir: Directory containing H5 files (if None, uses config default)
        selected_states: List of state names to load (if None, tries to detect from available files)
    """
    # If data_dir not provided, try to get from config
    if data_dir is None:
        try:
            config_obj = Config('config.yaml')
            if hasattr(config_obj, 'paths'):
                paths = config_obj.paths
                if isinstance(paths, dict):
                    data_dir = paths.get('state_data_dir', 'sr3_batch_output')
                else:
                    data_dir = getattr(paths, 'state_data_dir', 'sr3_batch_output')
            else:
                data_dir = 'sr3_batch_output'
        except Exception:
            data_dir = 'sr3_batch_output'
    
    # Normalize path (remove double slashes and ensure proper separator)
    data_dir = os.path.normpath(data_dir)
    # Ensure trailing separator for consistency
    if not data_dir.endswith(os.sep):
        data_dir = data_dir + os.sep
    # Replace double slashes with single separator
    data_dir = data_dir.replace('//', os.sep).replace('\\\\', os.sep)
    
    print("🔍 Calculating training normalization parameters from training split only...")
    print("   🎯 This ensures no data leakage by using only training data for normalization")
    
    import h5py
    import numpy as np
    import glob
    
    # Load full dataset
    def load_raw_data(filepath, var_name):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with h5py.File(filepath, 'r') as hf:
            data = np.array(hf['data'])
        print(f"  📊 {var_name}: {data.shape}")
        return data
    
    # Determine which states to load
    if selected_states is None:
        # Try to detect available state files
        print("  🔄 Detecting available state files...")
        state_files = glob.glob(os.path.join(data_dir, 'batch_spatial_properties_*.h5'))
        selected_states = []
        for state_file in state_files:
            filename = os.path.basename(state_file)
            state_name = filename.replace('batch_spatial_properties_', '').replace('.h5', '')
            selected_states.append(state_name)
        print(f"  ✅ Detected states: {selected_states}")
    
    if not selected_states:
        print("  ⚠️ No state files found or selected!")
        return {}
    
    # Load selected state data files dynamically
    print(f"  🔄 Loading {len(selected_states)} state data files...")
    state_data = {}
    for state_name in selected_states:
        state_file = os.path.join(data_dir, f'batch_spatial_properties_{state_name}.h5')
        try:
            state_data[state_name] = load_raw_data(state_file, state_name)
        except FileNotFoundError:
            print(f"  ⚠️ {state_name} file not found - skipping")
    
    if not state_data:
        print("  ❌ No state data files could be loaded!")
        return {}
    
    # Use first state to determine sample count
    first_state = list(state_data.keys())[0]
    n_sample = state_data[first_state].shape[0]
    
    # Load timeseries data (for controls/observations normalization)
    timeseries_data = {}
    timeseries_vars = ['BHP', 'ENERGYRATE', 'WATRATRC']
    for var_name in timeseries_vars:
        timeseries_file = os.path.join(data_dir, f'batch_timeseries_data_{var_name}.h5')
        try:
            timeseries_data[var_name] = load_raw_data(timeseries_file, var_name)
        except FileNotFoundError:
            print(f"  ⚠️ {var_name} timeseries file not found - skipping")
    
    # Apply EXACT same train/test split as training
    num_train = int(0.8 * n_sample)  # Same 80/20 split as training
    print(f"  📊 Total samples: {n_sample}, Training samples: {num_train}")
    
    # Extract TRAINING data only for normalization calculation
    state_train_data = {}
    for state_name, state_raw in state_data.items():
        state_train_data[state_name] = state_raw[:num_train]
    
    timeseries_train_data = {}
    for var_name, var_raw in timeseries_data.items():
        timeseries_train_data[var_name] = var_raw[:num_train]
    
    print("  🔧 Calculating normalization parameters from TRAINING DATA ONLY...")
    
    # Calculate normalization parameters from training data only
    def calc_norm_params(data, name):
        data_min = np.min(data)
        data_max = np.max(data)
        print(f"    📏 {name}: [{data_min:.8f}, {data_max:.8f}]")
        return {'min': data_min, 'max': data_max, 'type': 'minmax'}
    
    # Calculate normalization for all loaded states
    training_norm_params = {}
    for state_name, state_train in state_train_data.items():
        training_norm_params[state_name] = calc_norm_params(state_train, state_name)
    
    # Calculate normalization for timeseries variables
    for var_name, var_train in timeseries_train_data.items():
        training_norm_params[var_name] = calc_norm_params(var_train, var_name)
    
    print("✅ Training normalization parameters calculated!")
    return training_norm_params

def save_normalization_parameters_for_rl(training_norm_params, rom_config_path=None):
    """
    Save normalization parameters in the format expected by RL training
    This ensures compatibility between E2C evaluation and RL training
    
    Args:
        training_norm_params: Dictionary of normalization parameters calculated from training data
        rom_config_path: Optional path to ROM config.yaml to get control/observation definitions
    """
    print("💾 Saving normalization parameters for RL training compatibility...")
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"normalization_parameters_{timestamp}.json"
    
    # Separate spatial channels from timeseries variables
    # Spatial channels are typically: PRES, PERMI, TEMP, VPOROSGEO, SW, SG, POROS, PERMJ, PERMK
    spatial_channel_names = []
    timeseries_vars = []
    
    # Known spatial properties (from ROM config)
    known_spatial = ['PRES', 'PERMI', 'PERMJ', 'PERMK', 'TEMP', 'VPOROSGEO', 'VPOROSTGEO', 'SW', 'SG', 'POROS']
    known_timeseries = ['BHP', 'ENERGYRATE', 'WATRATRC']
    
    for var_name in training_norm_params.keys():
        if var_name in known_spatial:
            spatial_channel_names.append(var_name)
        elif var_name in known_timeseries:
            timeseries_vars.append(var_name)
        else:
            # Default: assume spatial if not a known timeseries variable
            # This ensures we don't miss any spatial channels
            if var_name not in known_timeseries:
                spatial_channel_names.append(var_name)
    
    # Ensure all known timeseries variables are included if they exist in training_norm_params
    for var_name in known_timeseries:
        if var_name in training_norm_params and var_name not in timeseries_vars:
            timeseries_vars.append(var_name)
    
    # Load ROM config to get control/observation definitions
    control_vars = {}
    observation_vars = {}
    
    if rom_config_path and os.path.exists(rom_config_path):
        try:
            import yaml
            with open(rom_config_path, 'r') as f:
                rom_config = yaml.safe_load(f)
            
            # Get controls
            if 'data' in rom_config and 'controls' in rom_config['data']:
                controls_section = rom_config['data']['controls']
                if 'variables' in controls_section:
                    for var_name, var_def in controls_section['variables'].items():
                        if var_name in training_norm_params:
                            control_vars[var_name] = {
                                'well_names': var_def.get('well_names', []),
                                'well_type': var_def.get('well_type', 'unknown')
                            }
            
            # Get observations
            if 'data' in rom_config and 'observations' in rom_config['data']:
                obs_section = rom_config['data']['observations']
                if 'variables' in obs_section:
                    for var_name, var_def in obs_section['variables'].items():
                        if var_name in training_norm_params:
                            observation_vars[var_name] = {
                                'well_names': var_def.get('well_names', []),
                                'well_type': var_def.get('well_type', 'unknown')
                            }
        except Exception as e:
            print(f"  ⚠️ Could not load ROM config for well names: {e}")
    
    # Build spatial_channels section dynamically
    spatial_channels = {}
    for var_name in spatial_channel_names:
        if var_name in training_norm_params:
            params = training_norm_params[var_name]
            spatial_channels[var_name] = {
                "normalization_type": params.get('type', 'minmax'),
                "selected_for_training": True,
                "parameters": {
                    "type": params.get('type', 'minmax'),
                    "min": str(params['min']),
                    "max": str(params['max'])
                }
            }
    
    # Build control_variables section dynamically
    # Include ALL timeseries variables that exist in training_norm_params
    # This ensures the environment can find them even if ROM config is incomplete
    control_variables = {}
    for var_name in timeseries_vars:
        if var_name in training_norm_params:
            params = training_norm_params[var_name]
            # Get well names from ROM config if available, otherwise use empty list
            well_names = []
            if var_name in control_vars:
                well_names = control_vars[var_name].get('well_names', [])
            
            control_variables[var_name] = {
                "normalization_type": params.get('type', 'minmax'),
                "selected_wells": well_names,
                "parameters": {
                    "type": params.get('type', 'minmax'),
                    "min": str(params['min']),
                    "max": str(params['max'])
                }
            }
    
    # Build observation_variables section dynamically
    # Include ALL timeseries variables that exist in training_norm_params
    # Variables can be both controls AND observations (e.g., BHP)
    observation_variables = {}
    for var_name in timeseries_vars:
        if var_name in training_norm_params:
            params = training_norm_params[var_name]
            # Get well names from ROM config if available, otherwise use empty list
            well_names = []
            if var_name in observation_vars:
                well_names = observation_vars[var_name].get('well_names', [])
            
            # Add to observations even if also in controls (variables can be both)
            observation_variables[var_name] = {
                "normalization_type": params.get('type', 'minmax'),
                "selected_wells": well_names,
                "parameters": {
                    "type": params.get('type', 'minmax'),
                    "min": str(params['min']),
                    "max": str(params['max'])
                }
            }
    
    # Create the structure dynamically
    norm_config = {
        "spatial_channels": spatial_channels,
        "control_variables": control_variables,
        "observation_variables": observation_variables,
        "selection_summary": {
            "spatial_channels": spatial_channel_names,
            "control_variables": list(control_variables.keys()),
            "observation_variables": list(observation_variables.keys()),
            "training_channels": spatial_channel_names
        },
        "metadata": {
            "created_timestamp": timestamp,
            "source": "RL Configuration Dashboard - Dynamic Structure",
            "structure": f"Spatial: {', '.join(spatial_channel_names)}, Controls: {', '.join(control_variables.keys())}, Observations: {', '.join(observation_variables.keys())}",
            "normalization_method": "training_only_parameters",
            "data_leakage": "eliminated"
        }
    }
    
    # Save to JSON file
    try:
        with open(filename, 'w') as f:
            json.dump(norm_config, f, indent=4)
        print(f"✅ Normalization parameters saved to: {filename}")
        print(f"   📊 Available for RL training: {list(training_norm_params.keys())}")
        print(f"   🎯 Structure: Optimal configuration for ROM compatibility")
        print(f"   🔧 No data leakage: Training-only parameters")
        return filename
    except Exception as e:
        print(f"❌ Error saving normalization parameters: {e}")
        return None

def apply_dashboard_scaling(data, state_name, rl_config, device):
    """
    Apply EXACT SAME normalization as corrected_model_test.py
    NO FALLBACKS - uses training-only parameters for perfect consistency
    
    Args:
        data: input tensor to scale
        state_name: name of the state variable (e.g., 'SW', 'PRES')  
        rl_config: dashboard configuration containing training normalization params
        device: PyTorch device
        
    Returns:
        Scaled tensor using EXACT same training normalization as corrected_model_test.py
    """
    print(f"      🔧 Applying EXACT training normalization for {state_name}...")
    
    # Get TRAINING-ONLY normalization parameters (EXACT same as corrected_model_test.py)
    training_params = rl_config.get('training_only_normalization_params', {})
    
    if state_name not in training_params:
        raise ValueError(f"❌ No training normalization parameters for {state_name}! This should not happen.")
    
    norm_params = training_params[state_name]
    param_min = norm_params['min']
    param_max = norm_params['max']
    
    print(f"      ✅ Using EXACT training normalization: [{param_min:.8f}, {param_max:.8f}]")
    
    # Apply EXACT same normalization logic as corrected_model_test.py
    def apply_training_normalization(data, norm_params):
        """Apply normalization using training parameters - EXACT COPY from corrected_model_test.py"""
        data_min = norm_params['min']
        data_max = norm_params['max']
        return (data - data_min) / (data_max - data_min)
    
    # Handle inactive cells for spatial data (EXACT same logic as corrected_model_test.py)
    if state_name in ['SW', 'SG', 'PRES', 'POROS', 'PERMI', 'PERMJ', 'PERMK', 'TEMP', 'VPOROSGEO', 'VPOROSTGEO']:
        # Identify active cells (same logic as corrected_model_test.py)
        active_mask = data > 0.0
        
        print(f"      • Active cells: {torch.sum(active_mask).item():,} / {data.numel():,}")
        
        # Start with data copy to preserve inactive cells
        scaled_data = data.clone()
        
        if param_max > param_min:
            # Apply training normalization
            normalized_data = apply_training_normalization(data, norm_params)
            # Only update active cells
            scaled_data[active_mask] = normalized_data[active_mask]
            # Inactive cells remain unchanged
            
            print(f"      ✅ EXACT training normalization applied to active cells")
        else:
            print(f"      ⚠️ Warning: param_min == param_max ({param_min:.6f})")
        
        return scaled_data.to(device)
    
    else:
        # For timeseries data, apply directly (EXACT same as corrected_model_test.py)
        if param_max > param_min:
            scaled_data = apply_training_normalization(data, norm_params)
            print(f"      ✅ EXACT training normalization applied to timeseries data")
        else:
            scaled_data = data.clone()
            print(f"      ⚠️ Warning: param_min == param_max ({param_min:.6f})")
        
        return scaled_data.to(device)

def create_state_t_seq_from_dashboard(rl_config, state_folder, device, specific_case_idx=None):
    """
    Create state_t_seq tensor following user's exact pattern:
    1. Load selected states from H5 files
    2. Extract INITIAL state (first timestep) from specified case OR all cases
    3. Apply GLOBAL ROM training normalization parameters
    4. Concatenate as channels IN CANONICAL ORDER
    5. Return (num_cases OR 1, channels, Nx, Ny, Nz) tensor
    """
    selected_states = rl_config.get('selected_states', [])
    
    if not selected_states:
        raise ValueError("❌ No states selected in dashboard! Please select states first.")
    
    print(f"   Processing {len(selected_states)} selected states in canonical order: {selected_states}")
    
    # Determine if we're extracting all cases or a specific case
    if specific_case_idx is not None:
        print(f"   📊 Extracting SPECIFIC case {specific_case_idx} initial conditions")
        extract_all_cases = False
    else:
        print(f"   📊 Extracting ALL cases initial conditions for random sampling")
        extract_all_cases = True
    
    state_tensors = []
    num_cases = None
    
    for state_name in selected_states:
        # Load state data from H5 file
        state_data = load_state_data_from_h5(state_name, state_folder, device)
        
        # 🏔️ ENHANCED: Extract initial conditions from ALL cases or specific case
        # state_data shape: (batch, time, Nx, Ny, Nz)
        batch_size, time_steps = state_data.shape[0], state_data.shape[1]
        print(f"   {state_name} data shape: {state_data.shape} ({batch_size} cases, {time_steps} timesteps)")
        
        if extract_all_cases:
            # Extract ALL cases, timestep 0: [:, 0:1, ...] → (num_cases, 1, Nx, Ny, Nz)
            state_t_seq = state_data[:, 0:1, ...].to(device)
            print(f"   📊 Extracted ALL {batch_size} cases initial conditions: {state_t_seq.shape}")
            if num_cases is None:
                num_cases = batch_size
            elif num_cases != batch_size:
                raise ValueError(f"Inconsistent number of cases: {num_cases} vs {batch_size}")
        else:
            # Extract specific case, timestep 0: [case_idx:case_idx+1, 0:1, ...] → (1, 1, Nx, Ny, Nz)
            if specific_case_idx >= batch_size:
                raise ValueError(f"Case index {specific_case_idx} out of range [0, {batch_size-1}]")
            state_t_seq = state_data[specific_case_idx:specific_case_idx+1, 0:1, ...].to(device)
            print(f"   📊 Extracted case {specific_case_idx} initial conditions: {state_t_seq.shape}")
            if num_cases is None:
                num_cases = 1
        
        # 🔍 VALIDATION: Check initial conditions data (for debugging - only first case)
        sample_state = state_t_seq[0:1] if extract_all_cases else state_t_seq
        active_mask = sample_state > 0.0  # Find active cells
        if torch.sum(active_mask) > 0:
            active_data = sample_state[active_mask]
            data_std = torch.std(active_data).item()
            data_range = torch.max(active_data).item() - torch.min(active_data).item()
            
            print(f"      📈 Initial conditions statistics (sample):")
            print(f"         • Active cells: {torch.sum(active_mask).item():,}")
            print(f"         • Value range: [{torch.min(active_data):.6f}, {torch.max(active_data):.6f}]")
            print(f"         • Standard deviation: {data_std:.6f}")
            
            if data_std < 1e-8 or data_range < 1e-8:
                print(f"      ℹ️ Initial conditions appear uniform (expected for some initial states)")
            else:
                print(f"      ✅ Initial conditions have variation")
        else:
            print(f"      ⚠️ No active cells found in {state_name} sample")
        
        # 🔧 CRITICAL: Apply GLOBAL ROM training normalization parameters
        # This ensures the initial state is normalized the same way as during E2C training
        state_scaled = apply_dashboard_scaling(state_t_seq, state_name, rl_config, device)
        
        # Remove time dimension: (num_cases, 1, Nz, Nx, Ny) → (num_cases, Nz, Nx, Ny)
        state_no_time = state_scaled.squeeze(1)
        print(f"   {state_name} after removing time dim: {state_no_time.shape}")
        
        # For channel concatenation: (num_cases, Nz, Nx, Ny) → (num_cases, 1, Nz, Nx, Ny)
        state_as_channel = state_no_time.unsqueeze(1)  # Add channel dim
        print(f"   {state_name} as channel: {state_as_channel.shape}")
        
        state_tensors.append(state_as_channel)
    
    # Concatenate along channel dimension (dim=1)
    # Each tensor: (num_cases, 1, 34, 16, 25) 
    # Result: (num_cases, n_states, 34, 16, 25) where n_states = number of selected states
    state_t_seq = torch.cat(state_tensors, dim=1)
    print(f"   Final state_t_seq shape: {state_t_seq.shape}")
    
    if extract_all_cases:
        expected_shape = (num_cases, len(selected_states), 34, 16, 25)
        print(f"   Expected ROM input: (batch={num_cases}, channels={len(selected_states)}, depth=34, height=16, width=25)")
        print(f"   🎯 Ready for random sampling: {num_cases} different initial states available")
    else:
        expected_shape = (1, len(selected_states), 34, 16, 25)
    print(f"   Expected ROM input: (batch=1, channels={len(selected_states)}, depth=34, height=16, width=25)")
    
    # Validation check - states are already in canonical order from parent function
    if state_t_seq.shape == expected_shape:
        print(f"   ✅ Shape validation PASSED: {state_t_seq.shape} matches expected {expected_shape}")
        print(f"      Each state becomes one channel: {len(selected_states)} channels total (canonical order)")
    else:
        print(f"   ❌ Shape validation FAILED: {state_t_seq.shape} != expected {expected_shape}")
        print(f"      This means ROM model expects {expected_shape[1]} channels but got {state_t_seq.shape[1]}")
        raise ValueError(f"Incorrect state_t_seq shape! Got {state_t_seq.shape}, expected {expected_shape}")
    
    return state_t_seq

def generate_z0_from_dashboard(rl_config, rom_model, device):
    """
    Main function to generate realistic Z0 options from dashboard configuration.
    
    Args:
        rl_config: Dashboard configuration dictionary
        rom_model: Trained ROM model with encoder
        device: PyTorch device
        
    Returns:
        z0_options: Tensor of multiple initial latent states (num_cases, latent_dim)
        selected_states: List of states used for generation
        state_t_seq: The state tensor used for encoding (for debugging)
    """
    print("🏔️ Generating multiple realistic Z0 options from dashboard state selection...")
    
    # Get state folder from dashboard configuration
    state_folder = rl_config.get('state_folder', 'sr3_batch_output/')
    
    # Get selected states from dashboard
    selected_states = rl_config.get('selected_states', [])
    if not selected_states:
        print("❌ No states selected in dashboard!")
        print("   Please run the dashboard, select states, and apply configuration.")
        raise ValueError("State selection required for Z0 generation")

    print(f"🏔️ Selected states from dashboard: {selected_states}")
    
    # 🔧 CRITICAL FIX: Use training channel order from normalization params (no hard-coding)
    # Get training channel order from config if available
    training_channel_order = rl_config.get('training_channel_order', None)
    
    if training_channel_order:
        # Reorder selected states to match training channel order
        canonical_selected_states = []
        for training_state in training_channel_order:
            if training_state in selected_states:
                canonical_selected_states.append(training_state)
        # Add any selected states not in training order (shouldn't happen, but be safe)
        for state in selected_states:
            if state not in canonical_selected_states:
                canonical_selected_states.append(state)
    else:
        # Fallback: use selected states as-is if no training order available
        canonical_selected_states = selected_states.copy()
    
    print(f"🔧 Selected states: {selected_states}")
    print(f"✅ Canonical corrected order: {canonical_selected_states}")
    print(f"📊 Channel mapping now matches ROM training exactly!")
    
    # Update rl_config with canonical order for downstream functions
    rl_config['selected_states'] = canonical_selected_states
    
    # 🔧 SIMPLIFIED: Direct ROM compatibility (no normalization bridge needed)
    print("🌉 Using ROM compatibility normalization directly...")
    print("✅ ROM normalization already available through compatibility config")
    print("📊 Proceeding with state processing using dashboard preprocessing compatibility")

    # Create state_t_seq tensor from ALL cases (extract_all_cases=True by default)
    print("   Creating state_t_seq tensor from ALL cases for random sampling...")
    state_t_seq = create_state_t_seq_from_dashboard(rl_config, state_folder, device)

    print(f"✅ State tensor created: {state_t_seq.shape}")
    print(f"   - Batch size: {state_t_seq.shape[0]} (ALL available cases)")
    print(f"   - Channels: {state_t_seq.shape[1]} (from {len(selected_states)} selected states in canonical order)")
    print(f"   - Spatial dimensions: {state_t_seq.shape[2:5]}")

    # Generate realistic Z0 options using ROM encoder for ALL cases
    print("   Encoding ALL initial states to latent space...")
    
    # Additional validation before encoding
    print(f"   🔍 Pre-encoding validation:")
    print(f"      • Input shape: {state_t_seq.shape}")
    print(f"      • Input range: [{torch.min(state_t_seq):.6f}, {torch.max(state_t_seq):.6f}]")
    print(f"      • Contains NaN: {torch.isnan(state_t_seq).any()}")
    print(f"      • Contains Inf: {torch.isinf(state_t_seq).any()}")
    
    # Clean input if needed
    if torch.isnan(state_t_seq).any():
        print("   🚨 Cleaning NaN values from input")
        state_t_seq = torch.nan_to_num(state_t_seq, nan=0.0)
    
    if torch.isinf(state_t_seq).any():
        print("   🚨 Cleaning Inf values from input")
        state_t_seq = torch.nan_to_num(state_t_seq, posinf=1.0, neginf=0.0)
    
    # Clamp to reasonable ranges for ROM encoder
    state_t_seq = torch.clamp(state_t_seq, min=-1.0, max=10.0)  # Allow -1.0 for inactive cells
    
    with torch.no_grad():
        try:
            z0_options = rom_model.model.encoder(state_t_seq)
            
            # Validate ROM encoder output
            if torch.isnan(z0_options).any():
                print("   🚨 ROM encoder produced NaN! Using safe fallback Z0.")
                z0_options = torch.zeros_like(z0_options)
                
            if torch.isinf(z0_options).any():
                print("   🚨 ROM encoder produced Inf! Clamping to safe range.")
                z0_options = torch.nan_to_num(z0_options, posinf=1.0, neginf=-1.0)
                
            # Clamp Z0 to reasonable latent space bounds
            z0_options = torch.clamp(z0_options, min=-50.0, max=50.0)
            
        except Exception as e:
            print(f"   🚨 ROM encoder failed: {e}")
            print("   Using zero-initialized Z0 options as fallback.")
            # Create safe fallback Z0 with correct shape
            latent_dim = 64  # From config
            z0_options = torch.zeros((state_t_seq.shape[0], latent_dim), device=state_t_seq.device)

    print(f"✅ Multiple realistic Z0 options generated from ROM encoder!")
    print(f"   - Z0 options shape: {z0_options.shape} ({z0_options.shape[0]} different initial states)")
    print(f"   - Z0 device: {z0_options.device}")
    print(f"   - Z0 requires_grad: {z0_options.requires_grad}")

    # Verify Z0 options are reasonable (not all zeros or extreme values)
    z0_stats = {
        'mean': z0_options.mean().item(),
        'std': z0_options.std().item(),
        'min': z0_options.min().item(),
        'max': z0_options.max().item(),
        'per_case_means': z0_options.mean(dim=1),  # Mean for each case
        'per_case_stds': z0_options.std(dim=1)     # Std for each case
    }
    print(f"   - Z0 statistics (all cases): mean={z0_stats['mean']:.4f}, std={z0_stats['std']:.4f}")
    # Final validation
    if torch.allclose(z0_options, torch.zeros_like(z0_options), atol=1e-6):
        print("⚠️ Warning: All Z0 options are very close to zero")
    
    print("✅ Z0 options ready for RL training")
    
    return z0_options, canonical_selected_states, state_t_seq

# =====================================
# SECTION: EXISTING DASHBOARD CODE
# =====================================

# Add the auto-detection function before the RLConfigurationDashboard class

def auto_detect_action_ranges_from_h5(data_dir=None, rom_config_path=None):
    """
    Automatically detect action ranges from H5 files and synchronize with ROM config control definitions
    
    Args:
        data_dir: Directory containing H5 files (if None, uses config default)
        rom_config_path: Path to ROM config.yaml for control definitions (optional)
    
    Returns:
        dict: Action ranges detected from H5 files, synchronized with ROM config
    """
    # If data_dir not provided, try to get from config
    if data_dir is None:
        try:
            config_obj = Config('config.yaml')
            if hasattr(config_obj, 'paths'):
                paths = config_obj.paths
                if isinstance(paths, dict):
                    data_dir = paths.get('state_data_dir', 'sr3_batch_output')
                else:
                    data_dir = getattr(paths, 'state_data_dir', 'sr3_batch_output')
            else:
                data_dir = 'sr3_batch_output'
        except Exception:
            data_dir = 'sr3_batch_output'
    
    # Import required modules at the top
    import os
    from pathlib import Path
    
    # Load ROM config control definitions if available
    control_definitions = {}
    num_producers = 3  # Default
    num_injectors = 3  # Default
    well_names_map = {}  # Maps well type to well names
    control_order = []  # Initialize control_order
    
    if rom_config_path is None:
        # Try to find ROM config.yaml
        current_dir = Path(__file__).parent.parent.parent
        rom_config_path = current_dir / 'ROM_Refactored' / 'config.yaml'
    
    if os.path.exists(rom_config_path):
        try:
            import yaml
            with open(rom_config_path, 'r') as f:
                rom_config = yaml.safe_load(f)
            
            # Load control definitions
            controls_config = rom_config.get('data', {}).get('controls', {})
            if controls_config and 'variables' in controls_config:
                # IMPORTANT: Store the FULL control_vars dict (not just selected fields)
                # This preserves all fields including indices, well_names, etc.
                control_vars = controls_config['variables'].copy()  # Make a copy to preserve original
                control_order = controls_config.get('order', [])
                
                for var_name in control_order:
                    if var_name in control_vars:
                        var_config = control_vars[var_name]
                        well_type = var_config.get('well_type', '')
                        num_wells = var_config.get('num_wells', 0)
                        well_names = var_config.get('well_names', [])
                        
                        control_definitions[var_name] = {
                            'well_type': well_type,
                            'num_wells': num_wells,
                            'well_names': well_names
                        }
                        
                        if well_type == 'producers':
                            num_producers = num_wells
                            well_names_map['producers'] = well_names
                        elif well_type == 'injectors':
                            num_injectors = num_wells
                            well_names_map['injectors'] = well_names
            
            # Also load well names from well_locations if not found in controls
            well_locations = rom_config.get('data', {}).get('well_locations', {})
            if well_locations:
                # Load producer well names from well_locations
                if 'producers' in well_locations and not well_names_map.get('producers'):
                    producer_names = list(well_locations['producers'].keys())
                    well_names_map['producers'] = producer_names
                    num_producers = len(producer_names)
                
                # Load injector well names from well_locations
                if 'injectors' in well_locations and not well_names_map.get('injectors'):
                    injector_names = list(well_locations['injectors'].keys())
                    well_names_map['injectors'] = injector_names
                    num_injectors = len(injector_names)
            
            # Load observations to show what are controls vs observations
            observations_config = rom_config.get('data', {}).get('observations', {})
            obs_vars = {}
            obs_order = []
            if observations_config and 'variables' in observations_config:
                obs_vars = observations_config['variables']
                obs_order = observations_config.get('order', [])
            
            # Print summary of controls and observations
            print(f"   ✅ Loaded control definitions from ROM config:")
            print(f"      Producers: {num_producers} wells ({well_names_map.get('producers', [])})")
            print(f"      Injectors: {num_injectors} wells ({well_names_map.get('injectors', [])})")
            print(f"\n   📋 CONTROLS (from ROM config):")
            for var_name in control_order:
                if var_name in control_vars:
                    var_config = control_vars[var_name]
                    well_type = var_config.get('well_type', 'unknown')
                    well_names = var_config.get('well_names', [])
                    unit_display = var_config.get('unit_display', '')
                    display_name = var_config.get('display_name', var_name)
                    print(f"      • {var_name} ({display_name}): {well_type} - wells {well_names} ({unit_display})")
            
            print(f"\n   📊 OBSERVATIONS (from ROM config):")
            for var_name in obs_order:
                if var_name in obs_vars:
                    var_config = obs_vars[var_name]
                    well_type = var_config.get('well_type', 'unknown')
                    well_names = var_config.get('well_names', [])
                    unit_display = var_config.get('unit_display', '')
                    display_name = var_config.get('display_name', var_name)
                    indices = var_config.get('indices', [])
                    print(f"      • {var_name} ({display_name}): {well_type} - wells {well_names} ({unit_display}) - indices {indices}")
        except Exception as e:
            print(f"   ⚠️ Could not load ROM config control definitions: {e}")
            print(f"   💡 Using defaults: {num_producers} producers, {num_injectors} injectors")
    detected_ranges = {
        'water_inj_min': 0.0,    # Default fallback values for water injection
        'water_inj_max': 1000.0,
        'bhp_min': 1087.78,
        'bhp_max': 1305.34,
        'detection_successful': False,
        'detection_details': {}
    }
    
    try:
        import h5py
        import numpy as np
        import os
        
        print("🔍 AUTO-DETECTING ACTION RANGES FROM H5 FILES...")
        print("=" * 60)
        
        # Store variable ranges for all detected control variables
        variable_ranges = {}
        
        # Generic function to detect ranges for any control variable
        def detect_variable_ranges(var_name, var_config, well_type, well_names_list):
            """Detect ranges for a control variable from H5 files"""
            h5_file = os.path.join(data_dir, f'batch_timeseries_data_{var_name}.h5')
            if not os.path.exists(h5_file):
                return None
            
            try:
                with h5py.File(h5_file, 'r') as f:
                    if 'data' not in f:
                        return None
                    
                    data = np.array(f['data'])
                    unit_display = var_config.get('unit_display', '')
                    display_name = var_config.get('display_name', var_name)
                    
                    # Determine which wells to use based on well_type
                    if well_type == 'producers':
                        num_target_wells = num_producers
                        target_indices = None
                        # Try to determine producer indices
                        if data.shape[2] >= num_producers + num_injectors:
                            target_indices = list(range(num_injectors, num_injectors + num_producers))
                        elif data.shape[2] >= num_producers:
                            target_indices = list(range(num_producers))
                        else:
                            target_indices = list(range(data.shape[2]))
                    elif well_type == 'injectors':
                        num_target_wells = num_injectors
                        target_indices = None
                        # Try to determine injector indices
                        if data.shape[2] >= num_injectors:
                            target_indices = list(range(num_injectors))
                        else:
                            target_indices = list(range(data.shape[2]))
                    else:
                        # Unknown well type, use all data
                        target_indices = list(range(data.shape[2]))
                        num_target_wells = data.shape[2]
                    
                    if target_indices:
                        target_data = data[:, :, target_indices]
                        # Get non-zero values for range detection
                        non_zero_data = target_data[target_data > 0]
                        
                        if len(non_zero_data) > 0:
                            var_min = float(np.min(non_zero_data))
                            var_max = float(np.max(non_zero_data))
                        else:
                            # Fallback to all data including zeros
                            var_min = float(np.min(target_data))
                            var_max = float(np.max(target_data))
                        
                        # Store per-well ranges if possible
                        well_ranges = {}
                        if len(target_indices) == len(well_names_list):
                            for i, well_name in enumerate(well_names_list):
                                well_data = target_data[:, :, i]
                                well_non_zero = well_data[well_data > 0]
                                if len(well_non_zero) > 0:
                                    well_ranges[well_name] = {
                                        'min': float(np.min(well_non_zero)),
                                        'max': float(np.max(well_non_zero))
                                    }
                                else:
                                    well_ranges[well_name] = {
                                        'min': float(np.min(well_data)),
                                        'max': float(np.max(well_data))
                                    }
                        
                        return {
                            'var_name': var_name,
                            'display_name': display_name,
                            'well_type': well_type,
                            'well_names': well_names_list,
                            'unit_display': unit_display,
                            'min': var_min,
                            'max': var_max,
                            'well_ranges': well_ranges,
                            'shape': data.shape,
                            'target_shape': target_data.shape,
                            'target_indices': target_indices,
                            'source': f'batch_timeseries_data_{var_name}.h5'
                        }
            except Exception as e:
                print(f"   ⚠️ Error detecting ranges for {var_name}: {e}")
                return None
            
            return None
        
        # Detect ranges for all control variables from ROM config
        for var_name in control_order:
            if var_name in control_vars:
                var_config = control_vars[var_name]
                well_type = var_config.get('well_type', '')
                well_names_list = var_config.get('well_names', [])
                
                detected_range = detect_variable_ranges(var_name, var_config, well_type, well_names_list)
                if detected_range:
                    variable_ranges[var_name] = detected_range
                    print(f"✅ {var_name.upper()} RANGES DETECTED:")
                    print(f"   📊 Data shape: {detected_range['shape']}")
                    print(f"   📊 Target data shape: {detected_range['target_shape']}")
                    print(f"   📈 Range: [{detected_range['min']:.2f}, {detected_range['max']:.2f}] {detected_range['unit_display']}")
                    if detected_range['well_names']:
                        well_names_str = ', '.join(detected_range['well_names'])
                        print(f"   🎯 Wells: {well_names_str}")
        
        # Store variable ranges in detection details
        detected_ranges['detection_details']['variable_ranges'] = variable_ranges
        
        # Legacy support: Also store in old format for backward compatibility
        # Check for water rate file (WATRATRC) for injectors
        # This is the correct control for injectors according to user
        water_file = os.path.join(data_dir, 'batch_timeseries_data_WATRATRC.h5')
        if os.path.exists(water_file):
            with h5py.File(water_file, 'r') as f:
                if 'data' in f:
                    water_data = np.array(f['data'])
                    
                    # Determine injector indices from ROM config
                    injector_indices = None
                    if water_data.shape[2] >= num_injectors:
                        # Assume first N wells are injectors (where N = num_injectors from config)
                        injector_indices = list(range(num_injectors))
                    else:
                        injector_indices = list(range(min(num_injectors, water_data.shape[2])))
                    
                    if injector_indices:
                        injector_water = water_data[:, :, injector_indices]
                        # Check for active injection values
                        active_water = injector_water[injector_water > 0]  # Any positive water injection
                        
                        if len(active_water) > 0:
                            water_min = np.min(active_water)
                            water_max = np.max(active_water)
                            active_count = len(active_water)
                        else:
                            # Fallback to all injector data
                            water_min = float(np.min(injector_water))
                            water_max = float(np.max(injector_water))
                            active_count = 0
                            
                            # If all zeros, check all wells
                            all_non_zero = water_data[water_data > 0]
                            if len(all_non_zero) > 0:
                                water_min = float(np.min(all_non_zero))
                                water_max = float(np.max(all_non_zero))
                                print(f"   ⚠️ Injector wells show zero values - using global data range from all wells")
                            else:
                                # Use default range
                                water_min = 0.0
                                water_max = 1000.0
                                print(f"   ⚠️ No valid injection data found - using default range")
                    else:
                        # Fallback to all data
                        all_non_zero = water_data[water_data > 0]
                        if len(all_non_zero) > 0:
                            water_min = float(np.min(all_non_zero))
                            water_max = float(np.max(all_non_zero))
                            active_count = len(all_non_zero)
                        else:
                            water_min = float(np.min(water_data))
                            water_max = float(np.max(water_data))
                            active_count = 0
                    
                    detected_ranges['water_inj_min'] = float(water_min)
                    detected_ranges['water_inj_max'] = float(water_max)
                    detected_ranges['detection_details']['water'] = {
                        'shape': water_data.shape,
                        'injector_indices': injector_indices if injector_indices else 'all',
                        'num_injectors': num_injectors,
                        'injector_shape': injector_water.shape if 'injector_water' in locals() else water_data.shape,
                        'active_values': active_count if 'active_count' in locals() else 0,
                        'min': float(water_min),
                        'max': float(water_max),
                        'source': f'batch_timeseries_data_WATRATRC.h5 (wells {injector_indices if injector_indices else "all"})'
                    }
                    
                    print(f"✅ WATER INJECTION RANGES DETECTED:")
                    print(f"   📊 Data shape: {water_data.shape}")
                    if 'injector_water' in locals():
                        print(f"   📊 Injector data shape: {injector_water.shape}")
                        active_count_val = active_count if 'active_count' in locals() else 0
                        print(f"   💧 Active injection values (>0): {active_count_val}")
                        if active_count_val == 0:
                            non_zero_count = np.count_nonzero(injector_water)
                            if non_zero_count > 0:
                                print(f"   💡 Found {non_zero_count} non-zero values (using all injector data)")
                            else:
                                print(f"   ⚠️ All injector values are zero - may need to check data file")
                    # Get unit from WATRATRC config if available
                    water_unit = 'bbl/day'
                    if 'WATRATRC' in control_vars:
                        water_unit = control_vars['WATRATRC'].get('unit_display', 'bbl/day')
                    elif 'WATRATRC' in obs_vars:
                        water_unit = obs_vars['WATRATRC'].get('unit_display', 'bbl/day')
                    print(f"   📈 Range: [{water_min:.2f}, {water_max:.2f}] {water_unit}")
                    if injector_indices:
                        injector_names_str = ', '.join(well_names_map.get('injectors', [f'I{i+1}' for i in range(len(injector_indices))]))
                        print(f"   🎯 Will be used for injector wells: {injector_indices} ({injector_names_str})")
                    else:
                        print(f"   🎯 Will be used as Water Injection defaults")
        
        # Note: ENERGYRATE is an OBSERVATION for producers, NOT a control for injectors
        # The injector control is WATRATRC (water rate), which was detected above
                    
        # Check for BHP control ranges
        # PRIORITY 1: Use min_range/max_range from config if specified (user-defined control range)
        # PRIORITY 2: Detect from H5 file
        bhp_range_from_config = False
        if 'BHP' in control_definitions:
            bhp_config = control_definitions['BHP']
            if 'min_range' in bhp_config and 'max_range' in bhp_config:
                bhp_min = float(bhp_config['min_range'])
                bhp_max = float(bhp_config['max_range'])
                bhp_range_from_config = True
                
                detected_ranges['bhp_min'] = bhp_min
                detected_ranges['bhp_max'] = bhp_max
                detected_ranges['detection_details']['bhp'] = {
                    'min': bhp_min,
                    'max': bhp_max,
                    'source': 'ROM config.yaml (min_range/max_range)',
                    'well_type': bhp_config.get('well_type', 'producers'),
                    'well_names': bhp_config.get('well_names', [])
                }
                
                print(f"✅ PRODUCER BHP RANGES FROM CONFIG:")
                print(f"   📋 Using user-defined control range from ROM config.yaml")
                print(f"   📈 Range: [{bhp_min:.2f}, {bhp_max:.2f}] psi")
                print(f"   🎯 For {bhp_config.get('well_type', 'producers')} wells")
        
        # If not specified in config, try to detect from H5 file
        if not bhp_range_from_config:
            bhp_file = os.path.join(data_dir, 'batch_timeseries_data_BHP.h5')
            if os.path.exists(bhp_file):
                with h5py.File(bhp_file, 'r') as f:
                    if 'data' in f:
                        bhp_data = np.array(f['data'])
                        
                        # Determine producer indices from ROM config
                        producer_indices = None
                        if 'BHP' in control_definitions:
                            # BHP is a control variable - check if it's for producers or injectors
                            var_config = control_definitions['BHP']
                            if var_config['well_type'] == 'producers':
                                # Producers use BHP as control
                                # BHP is first in order, so producers are indices 0-2
                                producer_indices = list(range(num_producers))
                            elif var_config['well_type'] == 'injectors':
                                # Injectors use BHP as control - producers might be in observations
                                # For now, assume producers come after injectors
                                producer_indices = list(range(num_injectors, num_injectors + num_producers))
                        
                        # Fallback logic if config doesn't specify
                        if producer_indices is None:
                            if bhp_data.shape[2] >= num_producers + num_injectors:
                                # Assume producers come after injectors
                                producer_indices = list(range(num_injectors, num_injectors + num_producers))
                            elif bhp_data.shape[2] >= num_producers:
                                # If only producers, use first N
                                producer_indices = list(range(num_producers))
                            else:
                                # Last resort: use all available wells
                                producer_indices = list(range(bhp_data.shape[2]))
                        
                        if producer_indices:
                            producer_bhp = bhp_data[:, :, producer_indices]
                            bhp_min = np.min(producer_bhp)
                            bhp_max = np.max(producer_bhp)
                        else:
                            # Fallback to all data if shape unexpected
                            bhp_min = np.min(bhp_data)
                            bhp_max = np.max(bhp_data)
                        
                        detected_ranges['bhp_min'] = float(bhp_min)
                        detected_ranges['bhp_max'] = float(bhp_max)
                        detected_ranges['detection_details']['bhp'] = {
                            'shape': bhp_data.shape,
                            'producer_indices': producer_indices if producer_indices else 'all',
                            'producer_shape': producer_bhp.shape if 'producer_bhp' in locals() else bhp_data.shape,
                            'num_producers': num_producers,
                            'min': float(bhp_min),
                            'max': float(bhp_max),
                            'source': f'batch_timeseries_data_BHP.h5 (wells {producer_indices if producer_indices else "all"})'
                        }
                        
                        print(f"✅ PRODUCER BHP RANGES DETECTED FROM H5:")
                        print(f"   📊 Data shape: {bhp_data.shape}")
                        if 'producer_bhp' in locals():
                            print(f"   📊 Producer data shape: {producer_bhp.shape}")
                            print(f"   📊 Producer indices: {producer_indices if producer_indices else 'all'}")
                        print(f"   📈 Range: [{bhp_min:.2f}, {bhp_max:.2f}] psi")
                        if producer_indices:
                            producer_names_str = ', '.join(well_names_map.get('producers', [f'P{i+1}' for i in range(len(producer_indices))]))
                            print(f"   🎯 Will be used for producer wells: {producer_indices} ({producer_names_str})")
                        else:
                            print(f"   🎯 Will be used as Producer BHP defaults")
                        
        # Store control definitions and well counts in detection details
        detected_ranges['detection_details']['control_definitions'] = control_definitions
        detected_ranges['detection_details']['num_producers'] = num_producers
        detected_ranges['detection_details']['num_injectors'] = num_injectors
        detected_ranges['detection_details']['well_names'] = well_names_map
        
        # Store full control and observation definitions for UI display
        detected_ranges['detection_details']['control_variables'] = control_vars
        detected_ranges['detection_details']['control_order'] = control_order
        detected_ranges['detection_details']['observation_definitions'] = obs_vars
        detected_ranges['detection_details']['observation_order'] = obs_order
        
        # Check if detection was successful
        if 'gas' in detected_ranges['detection_details'] and 'bhp' in detected_ranges['detection_details']:
            detected_ranges['detection_successful'] = True
            # Determine what was successfully detected
            water_detected = 'water' in detected_ranges.get('detection_details', {})
            bhp_detected = 'bhp' in detected_ranges.get('detection_details', {})
            
            print(f"\n🎉 AUTO-DETECTION SUMMARY:")
            if water_detected and bhp_detected:
                print(f"   ✅ Both Water Injection and Producer BHP ranges detected")
            elif water_detected:
                print(f"   ✅ Water Injection ranges detected")
                print(f"   ⚠️ Producer BHP ranges NOT detected")
            elif bhp_detected:
                print(f"   ✅ Producer BHP ranges detected")
                print(f"   ⚠️ Water Injection ranges NOT detected")
            else:
                print(f"   ⚠️ No action ranges detected - using defaults")
            
            print(f"   📂 Source directory: {data_dir}")
            if control_definitions:
                print(f"   ✅ Synchronized with ROM config control definitions")
                print(f"      Producers: {num_producers} wells ({well_names_map.get('producers', [])})")
                print(f"      Injectors: {num_injectors} wells ({well_names_map.get('injectors', [])})")
            print(f"   🔄 Dashboard will use these as default action limits")
        elif 'water' in detected_ranges['detection_details'] or 'bhp' in detected_ranges['detection_details']:
            detected_ranges['detection_successful'] = True  # Partial success
            print(f"\n⚠️ PARTIAL AUTO-DETECTION:")
            if 'water' not in detected_ranges['detection_details']:
                print(f"   ❌ Water injection ranges not detected - using fallback")
            if 'bhp' not in detected_ranges['detection_details']:
                print(f"   ❌ Producer BHP ranges not detected - using fallback")
            print(f"   📂 Source directory: {data_dir}")
            if control_definitions:
                print(f"   ✅ Synchronized with ROM config control definitions")
                print(f"      Producers: {num_producers} wells ({well_names_map.get('producers', [])})")
                print(f"      Injectors: {num_injectors} wells ({well_names_map.get('injectors', [])})")
        else:
            print(f"\n❌ AUTO-DETECTION FAILED:")
            print(f"   💡 Using fallback default values")
            print(f"   📂 Check if H5 files exist in: {data_dir}")
            
    except Exception as e:
        print(f"❌ Error during auto-detection: {e}")
        print(f"💡 Using fallback default values")
        detected_ranges['detection_details']['error'] = str(e)
    
    print("=" * 60)
    return detected_ranges

class RLConfigurationDashboard:
    """
    Interactive dashboard for configuring RL training parameters
    """
    
    def __init__(self, config_path='config.yaml'):
        # Initialize dashboard components
        
        # Load config to get default paths
        try:
            if Config is not None:
                # Resolve config path relative to RL_Refactored directory
                config_file_path = Path(config_path)
                if not config_file_path.is_absolute():
                    # Make relative to RL_Refactored directory
                    config_file_path = Path(__file__).parent.parent / config_path
                
                self.config_obj = Config(str(config_file_path))
                
                # Get paths from config, defaulting to ROM_Refactored paths
                # Config uses __getattr__ to allow direct access: config.paths
                if hasattr(self.config_obj, 'paths'):
                    paths = self.config_obj.paths
                    # Handle both dict-style and attribute-style access
                    if isinstance(paths, dict):
                        self.rom_folder = os.path.normpath(paths.get('rom_models_dir', '../ROM_Refactored/saved_models/'))
                        self.state_folder = os.path.normpath(paths.get('state_data_dir', '../ROM_Refactored/sr3_batch_output/'))
                    else:
                        # Attribute-style access
                        self.rom_folder = os.path.normpath(getattr(paths, 'rom_models_dir', '../ROM_Refactored/saved_models/'))
                        self.state_folder = os.path.normpath(getattr(paths, 'state_data_dir', '../ROM_Refactored/sr3_batch_output/'))
                else:
                    # No paths section in config, use ROM_Refactored defaults
                    self.rom_folder = "../ROM_Refactored/saved_models/"
                    self.state_folder = "../ROM_Refactored/sr3_batch_output/"
            else:
                # Fallback defaults pointing to ROM_Refactored
                self.rom_folder = "../ROM_Refactored/saved_models/"
                self.state_folder = "../ROM_Refactored/sr3_batch_output/"
                self.config_obj = None
        except Exception as e:
            print(f"⚠️ Warning: Could not load config for paths: {e}")
            print("   Using default ROM_Refactored paths")
            # Fallback defaults pointing to ROM_Refactored
            self.rom_folder = "../ROM_Refactored/saved_models/"
            self.state_folder = "../ROM_Refactored/sr3_batch_output/"
            self.config_obj = None
        
        # Configuration storage
        self.config = {
            'rom_folder': self.rom_folder,
            'state_folder': self.state_folder,
            'available_states': [],
            'selected_states': {},
            'state_scaling': {},
            'action_ranges': {},
            'economic_params': {},
            'rom_models': [],
            'selected_rom': None
        }
        
        # Pre-loaded models and generated data (NEW!)
        self.loaded_rom_model = None
        self.generated_z0_options = None  # Now stores multiple Z0 options for random sampling
        self.z0_metadata = None
        self.models_ready = False
        self.device = None
        
        # Control/Observation selection storage
        self.control_selections = {}  # Dict mapping variable names to checkbox widgets
        self.observation_selections = {}  # Dict mapping variable names to checkbox widgets
        self.variable_definitions = {}  # Dict storing all variable metadata from ROM config
        self.variable_range_widgets = {}  # Dict storing range input widgets for each control variable
        
        # Available state types (MUST be defined before ROM compatibility)
        # Load training channel order and scaling from normalization parameters
        self.training_channel_order = None
        self.training_channel_scaling = {}  # Maps channel name to scaling type (log/minmax)
        
        # Try to load from normalization parameters
        preprocessing_params = self._load_preprocessing_normalization_parameters()
        if preprocessing_params:
            # Get training channel order from selection_summary
            selection_summary = preprocessing_params.get('selection_summary', {})
            training_channels = selection_summary.get('training_channels', [])
            if training_channels:
                self.training_channel_order = training_channels
                # Use training channels as the source of truth for known states
                self.known_states = training_channels.copy()
                print(f"   ✅ Loaded training channel order from normalization params: {training_channels}")
            
            # Get scaling approach from spatial_channels
            spatial_channels = preprocessing_params.get('spatial_channels', {})
            for channel_name, channel_config in spatial_channels.items():
                norm_type = channel_config.get('normalization_type', 'minmax')
                # Normalize to 'log' or 'minmax'
                if norm_type.lower() in ['log', 'logarithmic']:
                    self.training_channel_scaling[channel_name] = 'log'
                else:
                    self.training_channel_scaling[channel_name] = 'minmax'
            
            if self.training_channel_scaling:
                print(f"   ✅ Loaded scaling approach from normalization params:")
                for channel, scaling in self.training_channel_scaling.items():
                    print(f"      {channel}: {scaling}")
        
        # If no normalization params found, known_states will be populated from file scanning
        # This ensures we only work with states that actually exist
        
        # AUTO-DETECT action ranges from H5 files and sync with ROM config
        print("🔧 INITIALIZING DASHBOARD WITH AUTO-DETECTED ACTION RANGES...")
        # Try to find ROM config path
        rom_config_path = None
        try:
            import os
            from pathlib import Path
            current_dir = Path(__file__).parent.parent.parent
            rom_config_path = current_dir / 'ROM_Refactored' / 'config.yaml'
            if not os.path.exists(rom_config_path):
                rom_config_path = None
        except Exception:
            rom_config_path = None
        
        detected_ranges = auto_detect_action_ranges_from_h5(data_dir=self.state_folder, rom_config_path=rom_config_path)
        
        # Default action ranges (AUTO-DETECTED from H5 files)
        self.default_actions = {
            'bhp_min': detected_ranges.get('bhp_min', 1087.78),         # psi - Auto-detected from H5 files
            'bhp_max': detected_ranges.get('bhp_max', 1305.34),         # psi - Auto-detected from H5 files
            'water_inj_min': detected_ranges.get('water_inj_min', 0.0), # bbl/day - Auto-detected from H5 files
            'water_inj_max': detected_ranges.get('water_inj_max', 1000.0)  # bbl/day - Auto-detected from H5 files
        }
        
        # Store detection details for display
        self.detection_details = detected_ranges['detection_details']
        self.detection_successful = detected_ranges['detection_successful']
        
        if self.detection_successful:
            print(f"✅ DASHBOARD INITIALIZED WITH AUTO-DETECTED RANGES:")
            print(f"   💧 Water Injection: [{self.default_actions['water_inj_min']:.2f}, {self.default_actions['water_inj_max']:.2f}] bbl/day")
            print(f"   🔽 Producer BHP: [{self.default_actions['bhp_min']:.2f}, {self.default_actions['bhp_max']:.2f}] psi")
        else:
            print(f"⚠️ DASHBOARD INITIALIZED WITH FALLBACK RANGES:")
            print(f"   💧 Water Injection: [{self.default_actions['water_inj_min']:.2f}, {self.default_actions['water_inj_max']:.2f}] bbl/day")
            print(f"   🔽 Producer BHP: [{self.default_actions['bhp_min']:.2f}, {self.default_actions['bhp_max']:.2f}] psi")
        
        # ROM compatibility handled through training-only normalization parameters
        
        # Default economic parameters (current values from code)
        self.default_economics = {
            # Geothermal project parameters
            'energy_production_revenue': 0.0011,  # Revenue from energy production ($/kWh electrical) - POSITIVE
            'water_production_cost': 5.0,         # Cost for water production disposal ($/bbl) - NEGATIVE
            'water_injection_cost': 10.0,         # Cost for water injection ($/bbl) - NEGATIVE
            'btu_to_kwh': 0.000293071,            # BTU to kWh conversion factor
            'days_per_year': 365,                 # Days per year (each RL timestep = 1 year)
            'thermal_to_electrical_efficiency': 0.1,  # Thermal BTU to electrical BTU efficiency (~10%)
            'scale_factor': 1000000.0,            # Final scaling factor for reward normalization
            # Pre-project development parameters
            'years_before_project_start': 5,      # Years of pre-project development
            'capital_cost_per_year': 20000000.0,  # Capital cost per year during pre-project phase ($20M default)
            'fixed_capital_cost': 100000000.0     # Total capital cost ($100M = 5 years × $20M/year)
        }
        
        # Default RL model hyperparameters
        self.default_rl_hyperparams = {
            'networks': {
                'hidden_dim': 200,
                'policy_type': 'deterministic',
                'output_activation': 'sigmoid'
            },
            'sac': {
                'discount_factor': 0.986,
                'soft_update_tau': 0.005,
                'entropy_alpha': 0.0,
                'critic_lr': 0.0001,
                'policy_lr': 0.0001,
                'gradient_clipping': True,
                'max_norm': 10.0
            },
            'training': {
                'max_episodes': 100,
                'max_steps_per_episode': 29,
                'batch_size': 256,
                'replay_capacity': 100000,
                'initial_exploration': 30
            }
        }
        
        # Default action variation parameters
        self.default_action_variation = {
            'enabled': True,
            'noise_decay_rate': 0.995,
            'max_noise_std': 0.25,
            'min_noise_std': 0.01,
            'step_variation_amplitude': 0.15,
            'mode': 'adaptive',  # 'adaptive', 'exploration', 'exploitation', 'minimal'
            'well_strategies': {
                'P1': {'variation': 0.15, 'bias': 0.0, 'exploration_scale': 0.8},    # Conservative producer
                'P2': {'variation': 0.20, 'bias': 0.05, 'exploration_scale': 1.0},   # Moderate producer  
                'P3': {'variation': 0.30, 'bias': -0.05, 'exploration_scale': 1.3},  # Aggressive producer
                'I1': {'variation': 0.18, 'bias': 0.02, 'exploration_scale': 0.9},   # Conservative injector
                'I2': {'variation': 0.25, 'bias': 0.0, 'exploration_scale': 1.1},    # Moderate injector
                'I3': {'variation': 0.35, 'bias': 0.08, 'exploration_scale': 1.4}    # Aggressive injector
            },
            'enhanced_gaussian_policy': {
                'enabled': False,  # Set to True to use Gaussian instead of deterministic
                'log_std_bounds': [-1.0, 1.0],  # Wider bounds for better exploration
                'entropy_weight': 0.2  # Entropy regularization weight
            }
        }
        

        
        if not WIDGETS_AVAILABLE:
            print("❌ Interactive widgets not available. Dashboard cannot be created.")
            return
            
        self._create_widgets()
        self._setup_event_handlers()
    
    # _apply_rom_compatibility method removed - using training-only normalization approach
    
    def _load_preprocessing_normalization_parameters(self):
        """
        Load the EXACT same normalization parameters saved by data preprocessing dashboard
        🎯 PERFECT COMPATIBILITY: Reads identical JSON files
        """
        import json
        import os
        from datetime import datetime
        
        try:
            # Search for normalization parameter files in multiple locations
            search_dirs = [
                '.',  # Current directory
                '../ROM_Refactored/processed_data/',  # ROM processed data directory
                './processed_data/',  # Local processed data directory
            ]
            
            norm_files = []
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    dir_files = [os.path.join(search_dir, f) for f in os.listdir(search_dir) 
                                if f.startswith('normalization_parameters_') and f.endswith('.json')]
                    norm_files.extend(dir_files)
            
            if not norm_files:
                print(f"      ❌ No normalization parameter JSON files found")
                print(f"      💡 Expected files like: normalization_parameters_YYYYMMDD_HHMMSS.json")
                print(f"      🔍 Searched in: {', '.join(search_dirs)}")
                return None
            
            # Get the most recent file (by modification time)
            norm_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_norm_file = norm_files[0]
            print(f"      📂 Loading from: {latest_norm_file}")
            
            # Load the JSON file
            with open(latest_norm_file, 'r') as f:
                preprocessing_params = json.load(f)
            
            # Validate the structure
            expected_keys = ['spatial_channels', 'control_variables', 'observation_variables', 'selection_summary', 'metadata']
            missing_keys = [key for key in expected_keys if key not in preprocessing_params]
            
            if missing_keys:
                print(f"      ⚠️ Warning: Missing keys in normalization file: {missing_keys}")
            
            # Print detailed parameter summary
            print(f"      📊 Loaded preprocessing parameters:")
            print(f"         📅 Created: {preprocessing_params.get('metadata', {}).get('created_timestamp', 'unknown')}")
            
            # Spatial channels details
            spatial_channels = preprocessing_params.get('spatial_channels', {})
            if spatial_channels:
                print(f"         🏔️ Spatial channels ({len(spatial_channels)}):")
                for channel, config in spatial_channels.items():
                    norm_type = config.get('normalization_type', 'unknown')
                    selected = config.get('selected_for_training', False)
                    status = "✅ TRAINING" if selected else "⭕ AVAILABLE"
                    print(f"            • {channel}: {norm_type.upper()} {status}")
            
            # Control variables details
            control_vars = preprocessing_params.get('control_variables', {})
            if control_vars:
                print(f"         🎛️ Control variables ({len(control_vars)}):")
                for var, config in control_vars.items():
                    norm_type = config.get('normalization_type', 'unknown')
                    wells = config.get('selected_wells', [])
                    print(f"            • {var}: {norm_type.upper()} (wells: {wells})")
            
            # Observation variables details
            obs_vars = preprocessing_params.get('observation_variables', {})
            if obs_vars:
                print(f"         📊 Observation variables ({len(obs_vars)}):")
                for var, config in obs_vars.items():
                    norm_type = config.get('normalization_type', 'unknown')
                    wells = config.get('selected_wells', [])
                    print(f"            • {var}: {norm_type.upper()} (wells: {wells})")
            
            # Training channel verification
            selection_summary = preprocessing_params.get('selection_summary', {})
            training_channels = selection_summary.get('training_channels', [])
            if training_channels:
                print(f"         🎯 Training channels: {training_channels}")
            
            print(f"      ✅ Preprocessing parameters loaded successfully from {latest_norm_file}")
            return preprocessing_params
            
        except FileNotFoundError:
            print(f"      ❌ Normalization parameter file not found")
            return None
        except json.JSONDecodeError as e:
            print(f"      ❌ Error parsing JSON file: {e}")
            return None
        except Exception as e:
            print(f"      ❌ Error loading preprocessing parameters: {e}")
            return None
        
    def _create_widgets(self):
        """Create all dashboard widgets"""
        
        # Header
        self.header = widgets.HTML(
            value="<h1>🎮 RL Configuration Dashboard</h1>",
            layout=widgets.Layout(margin='10px 0px')
        )
        
        # === FOLDER CONFIGURATION ===
        self.folder_section = widgets.VBox([
            widgets.HTML("<h2>📁 Folder Configuration</h2>"),
            
            widgets.HBox([
                widgets.Label("ROM Models Folder:", layout=widgets.Layout(width='150px')),
                widgets.Text(
                    value=self.rom_folder,
                    placeholder="Path to ROM models folder",
                    layout=widgets.Layout(width='300px')
                )
            ]),
            
            widgets.HBox([
                widgets.Label("State Data Folder:", layout=widgets.Layout(width='150px')),
                widgets.Text(
                    value=self.state_folder,
                    placeholder="Path to state data folder",
                    layout=widgets.Layout(width='300px')
                )
            ]),
            
            widgets.Button(
                description="🔍 Scan Folders",
                button_style='primary',
                layout=widgets.Layout(width='150px')
            )
        ])
        
        # Store references to folder widgets
        self.rom_folder_input = self.folder_section.children[1].children[1]
        self.state_folder_input = self.folder_section.children[2].children[1]
        self.scan_button = self.folder_section.children[3]
        
        # Status output
        self.status_output = widgets.Output()
        
        # === TAB STRUCTURE ===
        self.tabs = widgets.Tab()
        
        # State Tab
        self.state_tab = widgets.VBox([
            widgets.HTML("<h3>🏔️ State Selection & Scaling</h3>"),
            widgets.HTML("<p><i>Select which states to use and configure their normalization</i></p>")
        ])
        
        # Action Tab
        self.action_tab = widgets.VBox([
            widgets.HTML("<h3>🎮 Action Range Configuration</h3>"),
            widgets.HTML("<p><i>Configure BHP and injection rate ranges for each well</i></p>")
        ])
        
        # Economic Tab
        self.economic_tab = widgets.VBox([
            widgets.HTML("<h3>💰 Economic Parameters</h3>"),
            widgets.HTML("<p><i>Configure NPV calculation parameters</i></p>")
        ])
        
        # RL Hyperparameters Tab
        self.rl_hyperparams_tab = widgets.VBox([
            widgets.HTML("<h3>🧠 RL Model Hyperparameters</h3>"),
            widgets.HTML("<p><i>Configure SAC algorithm and network parameters</i></p>")
        ])
        
        # Action Variation Tab
        self.action_variation_tab = widgets.VBox([
            widgets.HTML("<h3>🌊 Action Variation Enhancement</h3>"),
            widgets.HTML("<p><i>Configure advanced action variation strategies for wide exploration</i></p>")
        ])
        
        # Set up tabs
        self.tabs.children = [self.state_tab, self.action_tab, self.economic_tab, self.rl_hyperparams_tab, self.action_variation_tab]
        self.tabs.set_title(0, "🏔️ States")
        self.tabs.set_title(1, "🎮 Actions")
        self.tabs.set_title(2, "💰 Economics")
        self.tabs.set_title(3, "🧠 RL Hyperparams")
        self.tabs.set_title(4, "🌊 Variation")
        
        # === CONTROL BUTTONS ===
        self.control_buttons = widgets.HBox([
            widgets.Button(
                description="✅ Apply Configuration",
                button_style='success',
                layout=widgets.Layout(width='200px')
            ),
            widgets.Button(
                description="🔄 Reset to Defaults",
                button_style='warning',
                layout=widgets.Layout(width='150px')
            ),
            widgets.Button(
                description="💾 Save Config",
                button_style='info',
                layout=widgets.Layout(width='120px')
            )
        ])
        
        self.apply_button = self.control_buttons.children[0]
        self.reset_button = self.control_buttons.children[1]
        self.save_button = self.control_buttons.children[2]
        
        # Results output
        self.results_output = widgets.Output()
        
        # === MAIN LAYOUT ===
        self.main_widget = widgets.VBox([
            self.header,
            self.folder_section,
            self.status_output,
            self.tabs,
            self.control_buttons,
            self.results_output
        ])
        
    def _setup_event_handlers(self):
        """Setup event handlers for widgets"""
        self.scan_button.on_click(self._scan_folders)
        self.apply_button.on_click(self._apply_configuration)
        self.reset_button.on_click(self._reset_defaults)
        self.save_button.on_click(self._save_configuration)
        
    def _scan_folders(self, button):
        """Scan folders for ROM models and state data"""
        with self.status_output:
            clear_output(wait=True)
            
            self.rom_folder = self.rom_folder_input.value.strip()
            self.state_folder = self.state_folder_input.value.strip()
            
            print(f"🔍 Scanning ROM folder: {self.rom_folder}")
            print(f"🔍 Scanning state folder: {self.state_folder}")
            
            # Scan ROM models
            self._scan_rom_models()
            
            # Scan available states
            self._scan_available_states()
            
            # Re-detect action ranges with updated folder path and sync with ROM config
            print("\n🔄 RE-DETECTING ACTION RANGES WITH UPDATED FOLDER PATH...")
            # Try to find ROM config path
            rom_config_path = None
            try:
                import os
                from pathlib import Path
                current_dir = Path(__file__).parent.parent.parent
                rom_config_path = current_dir / 'ROM_Refactored' / 'config.yaml'
                if not os.path.exists(rom_config_path):
                    rom_config_path = None
            except Exception:
                rom_config_path = None
            
            detected_ranges = auto_detect_action_ranges_from_h5(data_dir=self.state_folder, rom_config_path=rom_config_path)
            
            # Update default actions
            self.default_actions = {
                'bhp_min': detected_ranges['bhp_min'],
                'bhp_max': detected_ranges['bhp_max'],
                'water_inj_min': detected_ranges.get('water_inj_min', 0.0),
                'water_inj_max': detected_ranges.get('water_inj_max', 1000.0)
            }
            
            # Update detection details
            self.detection_details = detected_ranges['detection_details']
            self.detection_successful = detected_ranges['detection_successful']
            
            # Update tabs
            self._update_state_tab()
            self._update_action_tab()
            self._update_economic_tab()
            self._update_rl_hyperparams_tab()
            self._update_action_variation_tab()
            
            print("✅ Folder scanning completed!")
    
    def _scan_rom_models(self):
        """Scan for available ROM models (searches recursively in subdirectories)"""
        rom_models = []
        
        if not os.path.exists(self.rom_folder):
            print(f"❌ ROM folder not found: {self.rom_folder}")
            return
        
        # Normalize path
        rom_folder = os.path.normpath(self.rom_folder)
        
        # Look for encoder files to identify ROM models
        # Support recursive search in subdirectories (e.g., grid_search/)
        # Support both grid search pattern (e2co_encoder_grid_*) and standard pattern (e2co_encoder_*)
        encoder_patterns = [
            os.path.join(rom_folder, "**", "e2co_encoder_grid_*.h5"),  # Grid search pattern (recursive)
            os.path.join(rom_folder, "**", "e2co_encoder_*.h5"),         # Standard pattern (recursive)
            os.path.join(rom_folder, "**", "*encoder*.h5"),              # Fallback pattern (recursive)
            os.path.join(rom_folder, "e2co_encoder_grid_*.h5"),         # Grid search pattern (root)
            os.path.join(rom_folder, "e2co_encoder_*.h5"),               # Standard pattern (root)
            os.path.join(rom_folder, "*encoder*.h5")                    # Fallback pattern (root)
        ]
        
        encoder_files = []
        for pattern in encoder_patterns:
            found_files = glob.glob(pattern, recursive=True)
            encoder_files.extend(found_files)
        
        # Remove duplicates and normalize paths
        encoder_files = list(set([os.path.normpath(f) for f in encoder_files]))
        
        if not encoder_files:
            print(f"   ⚠️ No encoder files found in {rom_folder}")
            print(f"   💡 Looking for files matching: e2co_encoder_*.h5 or *encoder*.h5")
            print(f"   🔍 Searched recursively in: {rom_folder} and subdirectories")
            # List what files are actually in the directory
            if os.path.exists(rom_folder):
                all_files = []
                for root, dirs, files in os.walk(rom_folder):
                    for file in files:
                        if file.endswith('.h5'):
                            all_files.append(os.path.join(root, file))
                if all_files:
                    print(f"   📁 Found {len(all_files)} .h5 files in directory:")
                    for f in all_files[:10]:  # Show first 10
                        print(f"      - {os.path.relpath(f, rom_folder)}")
                    if len(all_files) > 10:
                        print(f"      ... and {len(all_files) - 10} more")
            self.config['rom_models'] = []
            return
        
        print(f"   🔍 Found {len(encoder_files)} encoder files")
        
        # Group encoder/decoder/transition files by their base pattern
        model_groups = {}
        
        for encoder_file in encoder_files:
            filename = os.path.basename(encoder_file)
            dirname = os.path.dirname(encoder_file)
            
            # Extract base pattern - everything except the component name
            # For grid pattern: e2co_encoder_grid_bs32_ld32_ns2_run0001_bs32_ld32_ns2.h5
            # Base: e2co_grid_bs32_ld32_ns2_run0001_bs32_ld32_ns2.h5
            # For standard pattern: e2co_encoder_3D_native_nt800_l128_lr1e-04_ep200_steps2_channels2_wells6.h5
            # Base: e2co_3D_native_nt800_l128_lr1e-04_ep200_steps2_channels2_wells6.h5
            
            # Try to find matching decoder and transition files
            decoder_file = None
            transition_file = None
            
            # Method 1: Replace encoder with decoder/transition
            decoder_candidate1 = os.path.join(dirname, filename.replace('_encoder', '_decoder'))
            transition_candidate1 = os.path.join(dirname, filename.replace('_encoder', '_transition'))
            
            # Method 2: For grid pattern, replace encoder_grid with decoder_grid/transition_grid
            decoder_candidate2 = os.path.join(dirname, filename.replace('encoder_grid', 'decoder_grid'))
            transition_candidate2 = os.path.join(dirname, filename.replace('encoder_grid', 'transition_grid'))
            
            # Method 3: Also try searching in the same directory for any matching decoder/transition files
            # This handles cases where naming might be slightly different
            decoder_pattern = os.path.join(dirname, filename.replace('encoder', 'decoder'))
            transition_pattern = os.path.join(dirname, filename.replace('encoder', 'transition'))
            
            # Check which method works (in order of preference)
            if os.path.exists(decoder_candidate1):
                decoder_file = decoder_candidate1
            elif os.path.exists(decoder_candidate2):
                decoder_file = decoder_candidate2
            elif os.path.exists(decoder_pattern):
                decoder_file = decoder_pattern
            else:
                # Last resort: search for any decoder file with similar base name
                # Extract key identifiers from filename (run number, batch size, latent dim, etc.)
                base_parts = []
                if 'run' in filename:
                    run_match = re.search(r'run(\d+)', filename)
                    if run_match:
                        base_parts.append(f"run{run_match.group(1)}")
                if 'bs' in filename:
                    bs_match = re.search(r'bs(\d+)', filename)
                    if bs_match:
                        base_parts.append(f"bs{bs_match.group(1)}")
                if 'ld' in filename:
                    ld_match = re.search(r'ld(\d+)', filename)
                    if ld_match:
                        base_parts.append(f"ld{ld_match.group(1)}")
                
                if base_parts:
                    # Search for decoder files with matching identifiers
                    search_pattern = os.path.join(dirname, f"*decoder*{'_'.join(base_parts)}*.h5")
                    decoder_search = glob.glob(search_pattern)
                    if decoder_search:
                        decoder_file = decoder_search[0]
                    else:
                        # Even more flexible: just search for any decoder in same directory
                        all_decoders = glob.glob(os.path.join(dirname, "*decoder*.h5"))
                        if all_decoders:
                            # Try to match by run number or other identifiers
                            for dec_file in all_decoders:
                                if any(part in dec_file for part in base_parts):
                                    decoder_file = dec_file
                                    break
                            if not decoder_file and all_decoders:
                                decoder_file = all_decoders[0]  # Use first decoder as fallback
            
            if os.path.exists(transition_candidate1):
                transition_file = transition_candidate1
            elif os.path.exists(transition_candidate2):
                transition_file = transition_candidate2
            elif os.path.exists(transition_pattern):
                transition_file = transition_pattern
            else:
                # Last resort: search for any transition file with similar base name
                # Extract key identifiers from filename
                base_parts = []
                if 'run' in filename:
                    run_match = re.search(r'run(\d+)', filename)
                    if run_match:
                        base_parts.append(f"run{run_match.group(1)}")
                if 'bs' in filename:
                    bs_match = re.search(r'bs(\d+)', filename)
                    if bs_match:
                        base_parts.append(f"bs{bs_match.group(1)}")
                if 'ld' in filename:
                    ld_match = re.search(r'ld(\d+)', filename)
                    if ld_match:
                        base_parts.append(f"ld{ld_match.group(1)}")
                
                if base_parts:
                    # Search for transition files with matching identifiers
                    search_pattern = os.path.join(dirname, f"*transition*{'_'.join(base_parts)}*.h5")
                    transition_search = glob.glob(search_pattern)
                    if transition_search:
                        transition_file = transition_search[0]
                    else:
                        # Even more flexible: just search for any transition in same directory
                        all_transitions = glob.glob(os.path.join(dirname, "*transition*.h5"))
                        if all_transitions:
                            # Try to match by run number or other identifiers
                            for trans_file in all_transitions:
                                if any(part in trans_file for part in base_parts):
                                    transition_file = trans_file
                                    break
                            if not transition_file and all_transitions:
                                transition_file = all_transitions[0]  # Use first transition as fallback
            
            # If we found all three files, add to models
            if decoder_file and transition_file:
                # Extract model info from filename
                model_info = self._parse_model_filename(filename)
                
                # Create a display name
                display_name = self._create_model_display_name(filename, model_info)
                
                # Use base pattern as key to avoid duplicates
                base_pattern = filename.replace('_encoder', '').replace('encoder_grid', 'grid')
                
                if base_pattern not in model_groups:
                    model_groups[base_pattern] = {
                        'name': display_name,
                        'encoder': encoder_file,
                        'decoder': decoder_file,
                        'transition': transition_file,
                        'info': model_info,
                        'filename': filename
                    }
                    print(f"   ✅ Found complete model set: {display_name}")
                    print(f"      Encoder: {os.path.basename(encoder_file)}")
                    print(f"      Decoder: {os.path.basename(decoder_file)}")
                    print(f"      Transition: {os.path.basename(transition_file)}")
            else:
                # Warn if encoder found but decoder/transition missing
                missing = []
                if not decoder_file:
                    missing.append('decoder')
                if not transition_file:
                    missing.append('transition')
                print(f"   ⚠️ Found encoder but missing {', '.join(missing)}: {os.path.basename(encoder_file)}")
        
        # Convert to list
        rom_models = list(model_groups.values())
        
        # Sort by run number or epoch if available
        def sort_key(model):
            info = model.get('info', {})
            # Prefer run number, then epoch, then latent dim
            run_num = info.get('run', 0)
            epoch = info.get('epoch', 0)
            latent = info.get('latent', 0)
            return (run_num, epoch, latent)
        
        rom_models.sort(key=sort_key, reverse=True)
        
        self.config['rom_models'] = rom_models
        print(f"📊 Found {len(rom_models)} complete ROM model sets")
        
        for i, model in enumerate(rom_models):
            info = model['info']
            print(f"   {i+1}. {model['name']}")
            if info:
                details = []
                if 'batch_size' in info:
                    details.append(f"bs={info['batch_size']}")
                if 'latent' in info:
                    details.append(f"ld={info['latent']}")
                if 'run' in info:
                    details.append(f"run={info['run']}")
                if 'epoch' in info:
                    details.append(f"ep={info['epoch']}")
                if details:
                    print(f"      ({', '.join(details)})")
    
    def _create_model_display_name(self, filename, model_info):
        """Create a user-friendly display name for the model"""
        # For grid search models: e2co_encoder_grid_bs32_ld32_ns2_run0001_bs32_ld32_ns2.h5
        # Display: Grid Model (bs=32, ld=32, run=1)
        
        # For standard models: e2co_encoder_3D_native_nt800_l128_lr1e-04_ep200_steps2_channels2_wells6.h5
        # Display: Standard Model (ld=128, ep=200, ch=2)
        
        if 'grid' in filename.lower():
            parts = []
            if 'batch_size' in model_info:
                parts.append(f"bs={model_info['batch_size']}")
            if 'latent' in model_info:
                parts.append(f"ld={model_info['latent']}")
            if 'run' in model_info:
                parts.append(f"run={model_info['run']}")
            if parts:
                return f"Grid Model ({', '.join(parts)})"
            else:
                return "Grid Search Model"
        else:
            parts = []
            if 'latent' in model_info:
                parts.append(f"ld={model_info['latent']}")
            if 'epoch' in model_info:
                parts.append(f"ep={model_info['epoch']}")
            if 'channels' in model_info:
                parts.append(f"ch={model_info['channels']}")
            if parts:
                return f"Standard Model ({', '.join(parts)})"
            else:
                return os.path.basename(filename).replace('_encoder', '').replace('.h5', '')
    
    def _parse_model_filename(self, filename):
        """Parse model filename to extract configuration info"""
        info = {}
        
        # Extract parameters using regex
        # Support both grid search pattern and standard pattern
        patterns = {
            'channels': r'channels(\d+)',
            'epoch': r'ep(\d+)',
            'latent': r'[^a-zA-Z]l(\d+)[^a-zA-Z]',  # Match 'l' followed by digits, avoiding 'ld' confusion
            'latent_dim': r'ld(\d+)',  # Explicit latent dimension pattern
            'wells': r'wells(\d+)',
            'steps': r'steps(\d+)',
            'nsteps': r'ns(\d+)',  # Grid search uses 'ns' for nsteps
            'batch_size': r'bs(\d+)',  # Grid search uses 'bs' for batch_size
            'run': r'run(\d+)',  # Grid search run number
            'num_train': r'nt(\d+)',  # Number of training samples
            'learning_rate': r'lr([\d\.e\-]+)',  # Learning rate (may be scientific notation)
        }
        
        for param, pattern in patterns.items():
            match = re.search(pattern, filename)
            if match:
                try:
                    if param == 'learning_rate':
                        # Handle scientific notation
                        val_str = match.group(1)
                        info[param] = float(val_str)
                    else:
                        info[param] = int(match.group(1))
                except (ValueError, IndexError):
                    pass
        
        # Normalize latent dimension (prefer latent_dim over latent)
        if 'latent_dim' in info:
            info['latent'] = info['latent_dim']
        elif 'latent' not in info and 'l' in filename:
            # Try to extract from 'l' pattern if not found
            match = re.search(r'_l(\d+)_', filename)
            if match:
                info['latent'] = int(match.group(1))
        
        return info
    
    def _scan_available_states(self):
        """Scan for available state data files (searches for various naming patterns)
        Uses training channel order from normalization params if available, otherwise scans all files"""
        available_states = []
        
        if not os.path.exists(self.state_folder):
            print(f"❌ State folder not found: {self.state_folder}")
            return
        
        # Normalize path
        state_folder = os.path.normpath(self.state_folder)
        
        # Determine which states to look for:
        # Priority 1: Training channel order from normalization params (if available)
        # Priority 2: known_states (if set from normalization params)
        # Priority 3: Scan all .h5 files and extract state names dynamically
        states_to_check = []
        
        if hasattr(self, 'training_channel_order') and self.training_channel_order:
            # Use training channel order as the source of truth
            states_to_check = self.training_channel_order.copy()
            print(f"   🔍 Scanning for training channels: {states_to_check}")
        elif self.known_states:
            # Use known_states if set
            states_to_check = self.known_states.copy()
            print(f"   🔍 Scanning for known states: {states_to_check}")
        else:
            # No predefined list - scan all .h5 files and extract state names
            print(f"   🔍 No predefined state list - scanning all .h5 files to detect states")
            all_h5_files = glob.glob(os.path.join(state_folder, 'batch_spatial_properties_*.h5'))
            for h5_file in all_h5_files:
                filename = os.path.basename(h5_file)
                # Extract state name from pattern: batch_spatial_properties_{STATE_NAME}.h5
                if 'batch_spatial_properties_' in filename:
                    state_name = filename.replace('batch_spatial_properties_', '').replace('.h5', '')
                    states_to_check.append(state_name)
            
            if states_to_check:
                print(f"   ✅ Detected states from file names: {states_to_check}")
                # Update known_states with detected states
                self.known_states = states_to_check.copy()
        
        found_files = {}
        for state_name in states_to_check:
            # Try multiple patterns
            patterns = [
                os.path.join(state_folder, f'batch_spatial_properties_{state_name}.h5'),
                os.path.join(state_folder, f'{state_name}.h5'),
                os.path.join(state_folder, f'*{state_name}*.h5'),
            ]
            
            for pattern in patterns:
                matching_files = glob.glob(pattern)
                if matching_files:
                    found_files[state_name] = matching_files[0]  # Use first match
                    break
        
        # Add found states (only include states that actually have files)
        for state_name, state_file in found_files.items():
            if os.path.exists(state_file):
                available_states.append(state_name)
                print(f"   ✅ Found {state_name} data: {os.path.basename(state_file)}")
        
        # Filter available_states to match training channel order if available
        # This ensures we only show states that were actually used in training
        if hasattr(self, 'training_channel_order') and self.training_channel_order:
            # Only include states that are in both training order AND available
            available_states = [s for s in self.training_channel_order if s in available_states]
            print(f"   🎯 Filtered to training channels: {available_states}")
        
        self.config['available_states'] = available_states
        if available_states:
            print(f"📊 Found {len(available_states)} state types: {available_states}")
        else:
            print(f"⚠️ No state files found in {state_folder}")
            print(f"   💡 Looking for files matching: batch_spatial_properties_*.h5")
            # List what files are actually there
            all_h5_files = glob.glob(os.path.join(state_folder, '*.h5'))
            if all_h5_files:
                print(f"   📁 Found {len(all_h5_files)} .h5 files in directory:")
                for h5_file in all_h5_files[:10]:
                    print(f"      - {os.path.basename(h5_file)}")
                if len(all_h5_files) > 10:
                    print(f"      ... and {len(all_h5_files) - 10} more")
    
    def _get_min_positive_value(self, state_name):
        """Get minimum positive value for a state"""
        try:
            state_file = os.path.join(self.state_folder, f'batch_spatial_properties_{state_name}.h5')
            if os.path.exists(state_file):
                with h5py.File(state_file, 'r') as hf:
                    data = np.array(hf['data'])
                    positive_data = data[data > 0]
                    if len(positive_data) > 0:
                        return float(np.min(positive_data))
                    else:
                        return 0.0
            else:
                return 0.0
        except Exception as e:
            print(f"Error reading {state_name}: {e}")
            return 0.0
        
    def _update_state_tab(self):
        """Update the state selection tab"""
        if not self.config['available_states']:
            self.state_tab.children = [
                widgets.HTML("<h3>🏔️ State Selection & Scaling</h3>"),
                widgets.HTML("<p>❌ No state data found. Please check the state folder path.</p>")
            ]
            return
            
        state_widgets = [
            widgets.HTML("<h3>🏔️ State Selection & Scaling</h3>"),
            widgets.HTML("<p><i>Select states and configure normalization. Min values use minimum positive (>0) values.</i></p>")
        ]
        
        # ROM model selection
        if self.config['rom_models']:
            # Create user-friendly options with detailed info
            rom_options = []
            for i, m in enumerate(self.config['rom_models']):
                info = m.get('info', {})
                display_name = m.get('name', f"Model {i+1}")
                
                # Add additional details if available
                details = []
                if 'batch_size' in info:
                    details.append(f"bs={info['batch_size']}")
                if 'latent' in info:
                    details.append(f"ld={info['latent']}")
                if 'channels' in info:
                    details.append(f"ch={info['channels']}")
                if 'run' in info:
                    details.append(f"run={info['run']}")
                if 'epoch' in info:
                    details.append(f"ep={info['epoch']}")
                
                if details:
                    full_name = f"{display_name} ({', '.join(details)})"
                else:
                    full_name = display_name
                
                rom_options.append((full_name, i))
            
            self.rom_selector = widgets.Dropdown(
                options=rom_options,
                description='ROM Model:',
                style={'description_width': '120px'},
                layout=widgets.Layout(width='600px')
            )
            
            # Add info about selected model
            def on_rom_selection_change(change):
                if change['new'] is not None:
                    selected_idx = change['new']
                    if selected_idx < len(self.config['rom_models']):
                        selected_model = self.config['rom_models'][selected_idx]
                        encoder_file = os.path.basename(selected_model['encoder'])
                        print(f"   📌 Selected: {selected_model['name']}")
                        print(f"      File: {encoder_file}")
            
            self.rom_selector.observe(on_rom_selection_change, names='value')
            
            state_widgets.append(self.rom_selector)
            
            # Add instruction text
            state_widgets.append(widgets.HTML(
                "<p><i>💡 Select a ROM model from the dropdown above. "
                "Make sure the model's configuration matches your RL setup.</i></p>"
            ))
        
        # State selection and scaling
        state_widgets.append(widgets.HTML("<hr><h4>📊 State Selection & Scaling</h4>"))
        
        # Use training channel order if available, otherwise use available_states order
        if hasattr(self, 'training_channel_order') and self.training_channel_order:
            # Filter to only include states that are available
            display_order = [s for s in self.training_channel_order if s in self.config['available_states']]
            # Add any available states not in training order (for completeness)
            remaining_states = [s for s in self.config['available_states'] if s not in display_order]
            display_order.extend(remaining_states)
            state_widgets.append(widgets.HTML(
                f"<p><b>✅ Using training model order:</b> {self.training_channel_order}</p>"
                f"<p><i>States are displayed in the same order as the trained ROM model.</i></p>"
            ))
        else:
            display_order = self.config['available_states']
            state_widgets.append(widgets.HTML(
                "<p><i>⚠️ Training channel order not found. Using available states order.</i></p>"
            ))
        
        self.state_checkboxes = {}
        self.scaling_radios = {}
        
        for state_name in display_order:
            # State checkbox - pre-select if in training channel order
            if hasattr(self, 'training_channel_order') and self.training_channel_order:
                default_selected = state_name in self.training_channel_order
            elif hasattr(self, 'rom_normalization') and state_name in self.rom_normalization:
                default_selected = True  # ROM states are automatically selected
            else:
                default_selected = False  # Don't auto-select by default
            
            checkbox = widgets.Checkbox(
                value=default_selected,
                description=f'{state_name}',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='100px')
            )
            
            # Scaling options - use training scaling if available, otherwise fallback
            if hasattr(self, 'training_channel_scaling') and state_name in self.training_channel_scaling:
                default_scaling = self.training_channel_scaling[state_name]
            elif hasattr(self, 'rom_state_scaling') and state_name in self.rom_state_scaling:
                default_scaling = self.rom_state_scaling[state_name]
            else:
                # Fallback: use config.yaml defaults or heuristic
                default_scaling = 'log' if 'PERM' in state_name or state_name in ['PRES', 'TEMP'] else 'minmax'
            
            scaling_radio = widgets.RadioButtons(
                options=['minmax', 'log'],
                value=default_scaling,
                layout=widgets.Layout(width='150px')
            )
            
            # Min positive value info
            min_pos = self._get_min_positive_value(state_name)
            info_label = widgets.HTML(
                value=f"<small>Min+: {min_pos:.6f}</small>",
                layout=widgets.Layout(width='120px')
            )
            
            state_row = widgets.HBox([
                checkbox,
                widgets.Label('Scaling:', layout=widgets.Layout(width='60px')),
                scaling_radio,
                info_label
            ])
            
            state_widgets.append(state_row)
            
            self.state_checkboxes[state_name] = checkbox
            self.scaling_radios[state_name] = scaling_radio
        
        self.state_tab.children = state_widgets
    
    def _update_action_tab(self):
        """Update the action range configuration tab with dynamic controls/observations selection"""
        action_widgets = [
            widgets.HTML("<h3>🎮 Action Range Configuration</h3>"),
            widgets.HTML("<p><i>Select controls and observations, then configure ranges for each control variable</i></p>")
        ]
        
        # Auto-detected ranges summary
        detection_details = getattr(self, 'detection_details', {})
        if hasattr(self, 'detection_successful') and self.detection_successful:
            detection_html = "<div style='background-color: #d4edda; padding: 10px; border-left: 4px solid #28a745; margin: 10px 0;'>"
            detection_html += "<h4>✅ Auto-Detected Ranges from H5 Files</h4>"
            detection_html += "<p><b>Ranges below are automatically detected from your data files:</b></p>"
            
            # Show detected variable ranges
            variable_ranges = detection_details.get('variable_ranges', {})
            for var_name, var_range_info in variable_ranges.items():
                detection_html += f"<p>📊 <b>{var_range_info['display_name']} ({var_name}):</b> "
                detection_html += f"[{var_range_info['min']:.2f}, {var_range_info['max']:.2f}] {var_range_info['unit_display']}<br/>"
                detection_html += f"&nbsp;&nbsp;&nbsp;&nbsp;📂 Source: {var_range_info['source']} (shape: {var_range_info['shape']})<br/>"
                detection_html += f"&nbsp;&nbsp;&nbsp;&nbsp;🔧 {var_range_info['well_type'].title()}: {var_range_info['well_names']}"
                detection_html += "</p>"
            
            # Show ROM config synchronization status
            if 'well_names' in detection_details:
                well_names = detection_details['well_names']
                detection_html += "<p><b>✅ Synchronized with ROM config control definitions</b></p>"
                if well_names.get('producers'):
                    detection_html += f"<p>&nbsp;&nbsp;&nbsp;&nbsp;📊 Producers: {well_names['producers']}</p>"
                if well_names.get('injectors'):
                    detection_html += f"<p>&nbsp;&nbsp;&nbsp;&nbsp;📊 Injectors: {well_names['injectors']}</p>"
            
            detection_html += "<p><i>💡 These ranges reflect the actual data in your reservoir model and are synchronized with ROM config.</i></p>"
            detection_html += "</div>"
            
            action_widgets.append(widgets.HTML(detection_html))
        else:
            fallback_html = "<div style='background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0;'>"
            fallback_html += "<h4>⚠️ Using Fallback Default Ranges</h4>"
            fallback_html += "<p><b>Auto-detection failed. Using predefined ranges:</b></p>"
            fallback_html += f"<p>💧 Water Injection: [{self.default_actions['water_inj_min']:.2f}, {self.default_actions['water_inj_max']:.2f}] bbl/day</p>"
            fallback_html += f"<p>🔽 Producer BHP: [{self.default_actions['bhp_min']:.2f}, {self.default_actions['bhp_max']:.2f}] psi</p>"
            fallback_html += "<p><i>💡 Check that H5 files exist in sr3_batch_output/ directory.</i></p>"
            fallback_html += "</div>"
            
            action_widgets.append(widgets.HTML(fallback_html))
        
        # Add refresh button for re-detection
        refresh_button = widgets.Button(
            description="🔄 Re-detect Ranges",
            button_style='info',
            tooltip="Re-scan H5 files to detect action ranges",
            layout=widgets.Layout(width='150px', margin='10px 0px')
        )
        refresh_button.on_click(self._refresh_action_ranges)
        action_widgets.append(refresh_button)
        
        # Well configuration (synchronized with ROM config)
        num_producers = detection_details.get('num_producers', 3)
        num_injectors = detection_details.get('num_injectors', 3)
        total_wells = num_producers + num_injectors
        
        self.num_wells_input = widgets.IntSlider(
            value=total_wells,
            min=2,
            max=20,
            description='Total Wells:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='300px')
        )
        
        self.num_prod_input = widgets.IntSlider(
            value=num_producers,
            min=1,
            max=10,
            description='Producers:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='300px')
        )
        
        action_widgets.extend([
            widgets.HTML("<h4>🏭 Well Configuration</h4>"),
            self.num_wells_input,
            self.num_prod_input
        ])
        
        # ALWAYS load control and observation definitions directly from ROM config file
        # This ensures we get the latest values, not stale cached values from detection_details
        control_vars = {}
        control_order = []
        obs_vars = {}
        obs_order = []
        variable_ranges = detection_details.get('variable_ranges', {})  # Keep ranges from detection
        
        # Load directly from ROM config file (source of truth)
        try:
            import yaml
            import copy
            from pathlib import Path
            current_dir = Path(__file__).parent.parent.parent
            rom_config_path = current_dir / 'ROM_Refactored' / 'config.yaml'
            
            if rom_config_path.exists():
                with open(rom_config_path, 'r') as f:
                    rom_config = yaml.safe_load(f)
                
                # Load controls - ALWAYS from ROM config file
                controls_config = rom_config.get('data', {}).get('controls', {})
                if controls_config and 'variables' in controls_config:
                    control_vars = copy.deepcopy(controls_config['variables'])
                    control_order = controls_config.get('order', [])
                    print(f"   ✅ Loaded {len(control_vars)} control variables from ROM config file")
                    for var_name in control_order:
                        if var_name in control_vars:
                            var_cfg = control_vars[var_name]
                            print(f"      {var_name}: well_names={var_cfg.get('well_names', [])}, indices={var_cfg.get('indices', [])}, well_type={var_cfg.get('well_type', 'unknown')}")
                
                # Load observations - ALWAYS from ROM config file
                observations_config = rom_config.get('data', {}).get('observations', {})
                if observations_config and 'variables' in observations_config:
                    obs_vars = copy.deepcopy(observations_config['variables'])
                    obs_order = observations_config.get('order', [])
                    print(f"   ✅ Loaded {len(obs_vars)} observation variables from ROM config file")
            else:
                print(f"   ⚠️ ROM config file not found at {rom_config_path}")
                # Fallback to detection_details if ROM config not found
                control_vars = detection_details.get('control_variables', {})
                control_order = detection_details.get('control_order', [])
                obs_vars = detection_details.get('observation_definitions', {})
                obs_order = detection_details.get('observation_order', [])
        except Exception as e:
            print(f"⚠️ Could not load ROM config for controls/observations: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to detection_details if loading fails
            control_vars = detection_details.get('control_variables', {})
            control_order = detection_details.get('control_order', [])
            obs_vars = detection_details.get('observation_definitions', {})
            obs_order = detection_details.get('observation_order', [])
        
        
        # Combine all variables from both controls and observations for display
        # This allows user to select any variable as control or observation
        # IMPORTANT: For variables that exist in both, we need to preserve both versions
        # because controls and observations can have different indices/well_names
        all_vars = {}
        # First add observations (they have more complete info like group_name)
        all_vars.update(obs_vars)
        # Then add/update with controls (to get control-specific indices and well_names)
        # But preserve observation fields that don't exist in controls
        for var_name, var_config in control_vars.items():
            if var_name in all_vars:
                # Merge: use control config but preserve observation-specific fields
                merged_config = all_vars[var_name].copy()
                merged_config.update(var_config)
                # For controls display, prioritize control indices and well_names
                merged_config['control_indices'] = var_config.get('indices', [])
                merged_config['control_well_names'] = var_config.get('well_names', [])
                # Keep observation indices/well_names for observation display
                merged_config['observation_indices'] = all_vars[var_name].get('indices', [])
                merged_config['observation_well_names'] = all_vars[var_name].get('well_names', [])
                all_vars[var_name] = merged_config
            else:
                all_vars[var_name] = var_config.copy()
        
        # Create combined order (controls first, then observations not in controls)
        combined_order = list(control_order)
        for var_name in obs_order:
            if var_name not in combined_order:
                combined_order.append(var_name)
        
        # Initialize selections if not already done
        if not hasattr(self, 'control_selections') or not self.control_selections:
            self.control_selections = {}
            self.observation_selections = {}
            self.variable_range_widgets = {}
        
        # Controls Selection Section
        action_widgets.append(widgets.HTML("<hr style='margin: 20px 0;'>"))
        action_widgets.append(widgets.HTML("<h4>🎮 Select Control Variables</h4>"))
        action_widgets.append(widgets.HTML("<p><i>Select which variables will be used as controls (actions) for the RL agent. Variables are shown from ROM config:</i></p>"))
        
        # Show all variables (from both controls and observations) so user can select what they want
        # But default selection based on ROM config controls
        if not all_vars:
            action_widgets.append(widgets.HTML("<p style='color: #ff9800;'><i>⚠️ No variables found in ROM config. Please ensure ROM_Refactored/config.yaml has controls/observations defined.</i></p>"))
        else:
            # Show variables grouped by well type for clarity
            # First show variables for producers, then injectors
            producer_vars = []
            injector_vars = []
            other_vars = []
            
            for var_name in combined_order:
                if var_name not in all_vars:
                    continue
                var_config = all_vars[var_name]
                well_type = var_config.get('well_type', 'unknown')
                if well_type == 'producers':
                    producer_vars.append(var_name)
                elif well_type == 'injectors':
                    injector_vars.append(var_name)
                else:
                    other_vars.append(var_name)
            
            # Show producer variables first
            if producer_vars:
                action_widgets.append(widgets.HTML("<p style='font-weight: bold; margin-top: 10px; color: #1976d2;'>📊 Producer Variables:</p>"))
                for var_name in producer_vars:
                    if var_name not in all_vars:
                        continue
                    
                    var_config = all_vars[var_name]
                    display_name = var_config.get('display_name', var_name)
                    well_type = var_config.get('well_type', 'unknown')
                    
                    # For controls display, ALWAYS use control-specific well_names and indices
                    # If variable exists in controls, use control definition directly
                    if var_name in control_vars:
                        # Use control definition for controls display - this is the source of truth
                        control_var_config = control_vars[var_name]
                        # Get well_names from control config (should match ROM config exactly)
                        well_names_list = control_var_config.get('well_names', [])
                        # Get indices from control config (should match ROM config exactly)
                        indices = control_var_config.get('indices', [])
                        # Also update well_type from control config to ensure consistency
                        well_type = control_var_config.get('well_type', well_type)
                    else:
                        # Variable only in observations, use observation values
                        well_names_list = var_config.get('well_names', [])
                        indices = var_config.get('indices', [])
                    
                    unit_display = var_config.get('unit_display', '')
                    
                    # Checkbox for selecting as control
                    # Default: selected if in ROM config controls
                    default_selected = var_name in control_vars
                    if var_name not in self.control_selections:
                        checkbox = widgets.Checkbox(
                            value=default_selected,
                            description=f"{var_name}",
                            style={'description_width': '120px'},
                            layout=widgets.Layout(width='150px', margin='5px 0px')
                        )
                        self.control_selections[var_name] = checkbox
                    else:
                        checkbox = self.control_selections[var_name]
                    
                    # Get indices for controls (similar to observations)
                    indices_str = f" - indices: {indices}" if indices else ""
                    
                    # Display name and info in separate label for better layout (no truncation)
                    display_label = widgets.HTML(
                        f"<div style='margin-left: 10px; width: 600px;'>"
                        f"<b>{display_name}</b><br/>"
                        f"<span style='color: #666; font-size: 0.9em;'>{well_type.title()}: {', '.join(well_names_list)} ({unit_display}){indices_str}</span>"
                        f"</div>",
                        layout=widgets.Layout(width='650px')
                    )
                
                    action_widgets.append(widgets.HBox([checkbox, display_label], layout=widgets.Layout(width='100%')))
                    
                    # Range inputs (shown if checkbox is checked)
                    if var_name not in self.variable_range_widgets:
                        self.variable_range_widgets[var_name] = {}
                    
                    # Get detected ranges for this variable
                    var_range_info = variable_ranges.get(var_name, {})
                    default_min = var_range_info.get('min', 0.0) if var_range_info else 0.0
                    default_max = var_range_info.get('max', 1000.0) if var_range_info else 1000.0
                    
                    # Create range inputs for each well
                    range_container = widgets.VBox([])
                    range_widgets = {}
                    
                    for well_name in well_names_list:
                        # Get per-well range if available
                        well_ranges = var_range_info.get('well_ranges', {})
                        well_min = well_ranges.get(well_name, {}).get('min', default_min) if well_ranges else default_min
                        well_max = well_ranges.get(well_name, {}).get('max', default_max) if well_ranges else default_max
                        
                        min_input = widgets.FloatText(
                            value=well_min,
                            description=f'{well_name} Min:',
                            style={'description_width': '80px'},
                            layout=widgets.Layout(width='150px')
                        )
                        max_input = widgets.FloatText(
                            value=well_max,
                            description=f'Max:',
                            style={'description_width': '40px'},
                            layout=widgets.Layout(width='120px')
                        )
                        
                        range_widgets[well_name] = {'min': min_input, 'max': max_input}
                        range_container.children = list(range_container.children) + [
                            widgets.HBox([min_input, max_input], layout=widgets.Layout(margin='2px 0px 2px 30px'))
                        ]
                    
                    self.variable_range_widgets[var_name] = range_widgets
                    
                    # Show/hide range inputs based on checkbox value
                    def make_toggle_handler(var_name, range_container, widgets_list):
                        def toggle_handler(change):
                            current_children = list(self.action_tab.children)
                            if change['new']:
                                if range_container not in current_children:
                                    # Find the position after the checkbox
                                    checkbox_idx = None
                                    for i, widget in enumerate(current_children):
                                        if isinstance(widget, widgets.HBox) and len(widget.children) > 0:
                                            if hasattr(widget.children[0], 'description') and var_name in str(widget.children[0].description):
                                                checkbox_idx = i
                                                break
                                    if checkbox_idx is not None:
                                        current_children.insert(checkbox_idx + 1, range_container)
                                    else:
                                        current_children.append(range_container)
                                    self.action_tab.children = current_children
                            else:
                                if range_container in current_children:
                                    current_children.remove(range_container)
                                    self.action_tab.children = current_children
                        return toggle_handler
                    
                    checkbox.observe(make_toggle_handler(var_name, range_container, action_widgets), names='value')
                    
                    # Initially show if selected
                    if checkbox.value:
                        action_widgets.append(range_container)
            
            # Show injector variables
            if injector_vars:
                action_widgets.append(widgets.HTML("<p style='font-weight: bold; margin-top: 15px; color: #d32f2f;'>💧 Injector Variables:</p>"))
                for var_name in injector_vars:
                    if var_name not in all_vars:
                        continue
                    
                    var_config = all_vars[var_name]
                    display_name = var_config.get('display_name', var_name)
                    well_type = var_config.get('well_type', 'unknown')
                    
                    # For controls display, ALWAYS use control-specific well_names and indices
                    # If variable exists in controls, use control definition directly
                    if var_name in control_vars:
                        # Use control definition for controls display - this is the source of truth
                        control_var_config = control_vars[var_name]
                        # Get well_names from control config (should match ROM config exactly)
                        well_names_list = control_var_config.get('well_names', [])
                        # Get indices from control config (should match ROM config exactly)
                        indices = control_var_config.get('indices', [])
                        # Also update well_type from control config to ensure consistency
                        well_type = control_var_config.get('well_type', well_type)
                    else:
                        # Variable only in observations, use observation values
                        well_names_list = var_config.get('well_names', [])
                        indices = var_config.get('indices', [])
                    
                    unit_display = var_config.get('unit_display', '')
                    
                    # Checkbox for selecting as control
                    # Default: selected if in ROM config controls
                    default_selected = var_name in control_vars
                    if var_name not in self.control_selections:
                        checkbox = widgets.Checkbox(
                            value=default_selected,
                            description=f"{var_name}",
                            style={'description_width': '120px'},
                            layout=widgets.Layout(width='150px', margin='5px 0px')
                        )
                        self.control_selections[var_name] = checkbox
                    else:
                        checkbox = self.control_selections[var_name]
                    
                    # Get indices for controls (similar to observations)
                    indices_str = f" - indices: {indices}" if indices else ""
                    
                    # Display name and info in separate label for better layout (no truncation)
                    display_label = widgets.HTML(
                        f"<div style='margin-left: 10px; width: 600px;'>"
                        f"<b>{display_name}</b><br/>"
                        f"<span style='color: #666; font-size: 0.9em;'>{well_type.title()}: {', '.join(well_names_list)} ({unit_display}){indices_str}</span>"
                        f"</div>",
                        layout=widgets.Layout(width='650px')
                    )
                    
                    action_widgets.append(widgets.HBox([checkbox, display_label], layout=widgets.Layout(width='100%')))
                    
                    # Range inputs (shown if checkbox is checked)
                    if var_name not in self.variable_range_widgets:
                        self.variable_range_widgets[var_name] = {}
                    
                    # Get detected ranges for this variable
                    var_range_info = variable_ranges.get(var_name, {})
                    default_min = var_range_info.get('min', 0.0) if var_range_info else 0.0
                    default_max = var_range_info.get('max', 1000.0) if var_range_info else 1000.0
                    
                    # Create range inputs for each well
                    range_container = widgets.VBox([])
                    range_widgets = {}
                    
                    for well_name in well_names_list:
                        # Get per-well range if available
                        well_ranges = var_range_info.get('well_ranges', {})
                        well_min = well_ranges.get(well_name, {}).get('min', default_min) if well_ranges else default_min
                        well_max = well_ranges.get(well_name, {}).get('max', default_max) if well_ranges else default_max
                        
                        min_input = widgets.FloatText(
                            value=well_min,
                            description=f'{well_name} Min:',
                            style={'description_width': '80px'},
                            layout=widgets.Layout(width='150px')
                        )
                        max_input = widgets.FloatText(
                            value=well_max,
                            description=f'Max:',
                            style={'description_width': '40px'},
                            layout=widgets.Layout(width='120px')
                        )
                        
                        range_widgets[well_name] = {'min': min_input, 'max': max_input}
                        range_container.children = list(range_container.children) + [
                            widgets.HBox([min_input, max_input], layout=widgets.Layout(margin='2px 0px 2px 30px'))
                        ]
                    
                    self.variable_range_widgets[var_name] = range_widgets
                    
                    # Show/hide range inputs based on checkbox value
                    def make_toggle_handler(var_name, range_container, widgets_list):
                        def toggle_handler(change):
                            current_children = list(self.action_tab.children)
                            if change['new']:
                                if range_container not in current_children:
                                    # Find the position after the checkbox
                                    checkbox_idx = None
                                    for i, widget in enumerate(current_children):
                                        if isinstance(widget, widgets.HBox) and len(widget.children) > 0:
                                            if hasattr(widget.children[0], 'description') and var_name in str(widget.children[0].description):
                                                checkbox_idx = i
                                                break
                                    if checkbox_idx is not None:
                                        current_children.insert(checkbox_idx + 1, range_container)
                                    else:
                                        current_children.append(range_container)
                                    self.action_tab.children = current_children
                            else:
                                if range_container in current_children:
                                    current_children.remove(range_container)
                                    self.action_tab.children = current_children
                        return toggle_handler
                    
                    checkbox.observe(make_toggle_handler(var_name, range_container, action_widgets), names='value')
                    
                    # Initially show if selected
                    if checkbox.value:
                        action_widgets.append(range_container)
        
        # Observations Selection Section
        action_widgets.append(widgets.HTML("<hr style='margin: 20px 0;'>"))
        action_widgets.append(widgets.HTML("<h4>📊 Select Observation Variables</h4>"))
        action_widgets.append(widgets.HTML("<p><i>Select which variables will be used as observations (measured outputs) for the RL agent. All variables from ROM config are shown below:</i></p>"))
        
        # Show ALL variables (from both controls and observations) so user can select what they want
        # Default selection based on ROM config observations
        if not all_vars:
            action_widgets.append(widgets.HTML("<p style='color: #ff9800;'><i>⚠️ No variables found in ROM config. Please ensure ROM_Refactored/config.yaml has controls/observations defined.</i></p>"))
        else:
            # Create checkboxes for each variable
            for var_name in combined_order:
                if var_name not in all_vars:
                    continue
                
                var_config = all_vars[var_name]
                display_name = var_config.get('display_name', var_name)
                well_type = var_config.get('well_type', 'unknown')
                
                # For observations display, prioritize observation-specific well_names and indices
                # If variable exists in both controls and observations, use observation version
                if var_name in obs_vars:
                    # Use observation definition for observations display
                    obs_var_config = obs_vars[var_name]
                    well_names_list = obs_var_config.get('well_names', var_config.get('well_names', []))
                    indices = obs_var_config.get('indices', var_config.get('indices', []))
                else:
                    # Variable only in controls, use control values
                    well_names_list = var_config.get('well_names', [])
                    indices = var_config.get('indices', [])
                
                unit_display = var_config.get('unit_display', '')
                
                # Checkbox for selecting as observation
                # Default: selected if in ROM config observations
                default_selected = var_name in obs_vars
                if var_name not in self.observation_selections:
                    checkbox = widgets.Checkbox(
                        value=default_selected,
                        description=f"{var_name}",
                        style={'description_width': '120px'},
                        layout=widgets.Layout(width='150px', margin='5px 0px')
                    )
                    self.observation_selections[var_name] = checkbox
                else:
                    checkbox = self.observation_selections[var_name]
                
                # Display name and info in separate label for better layout (no truncation)
                indices_str = f" - indices: {indices}" if indices else ""
                display_label = widgets.HTML(
                    f"<div style='margin-left: 10px; width: 600px;'>"
                    f"<b>{display_name}</b><br/>"
                    f"<span style='color: #666; font-size: 0.9em;'>{well_type.title()}: {', '.join(well_names_list)} ({unit_display}){indices_str}</span>"
                    f"</div>",
                    layout=widgets.Layout(width='650px')
                )
                
                action_widgets.append(widgets.HBox([checkbox, display_label], layout=widgets.Layout(width='100%')))
        
        self.action_tab.children = action_widgets
    
    def _refresh_action_ranges(self, button):
        """Re-detect action ranges from H5 files and update the dashboard"""
        print("🔄 RE-DETECTING ACTION RANGES...")
        
        # Preserve current selections before refresh
        preserved_control_selections = {}
        preserved_observation_selections = {}
        
        if hasattr(self, 'control_selections'):
            for var_name, checkbox in self.control_selections.items():
                preserved_control_selections[var_name] = checkbox.value
        
        if hasattr(self, 'observation_selections'):
            for var_name, checkbox in self.observation_selections.items():
                preserved_observation_selections[var_name] = checkbox.value
        
        # Get current state folder
        current_state_folder = self.state_folder_input.value.strip() if hasattr(self, 'state_folder_input') else self.state_folder
        
        # Try to find ROM config path
        rom_config_path = None
        try:
            import os
            from pathlib import Path
            current_dir = Path(__file__).parent.parent.parent
            rom_config_path = current_dir / 'ROM_Refactored' / 'config.yaml'
            if not os.path.exists(rom_config_path):
                rom_config_path = None
        except Exception:
            rom_config_path = None
        
        # Re-run detection with ROM config synchronization
        detected_ranges = auto_detect_action_ranges_from_h5(data_dir=current_state_folder, rom_config_path=rom_config_path)
        
        # Update default actions (for backward compatibility)
        self.default_actions = {
            'bhp_min': detected_ranges.get('bhp_min', 1087.78),
            'bhp_max': detected_ranges.get('bhp_max', 1305.34),
            'water_inj_min': detected_ranges.get('water_inj_min', 0.0),
            'water_inj_max': detected_ranges.get('water_inj_max', 1000.0)
        }
        
        # Update detection details
        self.detection_details = detected_ranges['detection_details']
        self.detection_successful = detected_ranges['detection_successful']
        
        # Update variable range widgets with new detected values
        variable_ranges = detected_ranges['detection_details'].get('variable_ranges', {})
        if hasattr(self, 'variable_range_widgets'):
            for var_name, var_range_info in variable_ranges.items():
                if var_name in self.variable_range_widgets:
                    well_ranges = var_range_info.get('well_ranges', {})
                    for well_name, well_widgets in self.variable_range_widgets[var_name].items():
                        if well_name in well_ranges:
                            well_widgets['min'].value = well_ranges[well_name]['min']
                            well_widgets['max'].value = well_ranges[well_name]['max']
                        else:
                            # Use overall min/max if per-well ranges not available
                            well_widgets['min'].value = var_range_info.get('min', 0.0)
                            well_widgets['max'].value = var_range_info.get('max', 1000.0)
        
        # Refresh the entire action tab to show updated detection status
        self._update_action_tab()
        
        # Restore preserved selections
        if hasattr(self, 'control_selections'):
            for var_name, was_selected in preserved_control_selections.items():
                if var_name in self.control_selections:
                    self.control_selections[var_name].value = was_selected
        
        if hasattr(self, 'observation_selections'):
            for var_name, was_selected in preserved_observation_selections.items():
                if var_name in self.observation_selections:
                    self.observation_selections[var_name].value = was_selected
        
        print("✅ Action ranges refreshed successfully!")
    
    def _update_economic_tab(self):
        """Update the economic parameters tab"""
        economic_widgets = [
            widgets.HTML("<h3>💰 Economic Parameters</h3>"),
            widgets.HTML("<p><i>Configure NPV calculation parameters for reward function</i></p>")
        ]
        
        # Economic parameter inputs
        self.economic_inputs = {}
        
        # Add pre-project development parameters section
        economic_widgets.append(widgets.HTML("<hr><h4>🏗️ Pre-Project Development Phase</h4>"))
        
        # Years before project start
        self.years_before_input = widgets.IntText(
            value=self.default_economics['years_before_project_start'],
            description='Years before project start:',
            style={'description_width': '250px'},
            layout=widgets.Layout(width='450px')
        )
        economic_widgets.append(self.years_before_input)
        
        # Capital cost per year
        self.capital_per_year_input = widgets.FloatText(
            value=self.default_economics['capital_cost_per_year'],
            description='Capital cost per year ($):',
            style={'description_width': '250px'},
            layout=widgets.Layout(width='450px')
        )
        economic_widgets.append(self.capital_per_year_input)
        
        # Total capital cost (calculated automatically)
        self.total_capital_display = widgets.HTML(
            value=f"<b>Total Capital Cost: ${self.default_economics['fixed_capital_cost']:,.0f}</b>",
            layout=widgets.Layout(width='450px')
        )
        economic_widgets.append(self.total_capital_display)
        
        # Store references for calculation
        self.economic_inputs['years_before_project_start'] = self.years_before_input
        self.economic_inputs['capital_cost_per_year'] = self.capital_per_year_input
        
        # Set up automatic calculation
        def update_total_capital(*args):
            years = self.years_before_input.value
            cost_per_year = self.capital_per_year_input.value
            total_cost = years * cost_per_year
            self.total_capital_display.value = f"<b>Total Capital Cost: ${total_cost:,.0f}</b>"
        
        self.years_before_input.observe(update_total_capital, names='value')
        self.capital_per_year_input.observe(update_total_capital, names='value')
        
        # Add conversion factors section
        economic_widgets.append(widgets.HTML("<hr><h4>🔄 Conversion Factors</h4>"))
        
        # Thermal to Electrical Efficiency
        self.thermal_efficiency_input = widgets.FloatText(
            value=self.default_economics['thermal_to_electrical_efficiency'],
            description='Thermal→Electrical Efficiency:',
            style={'description_width': '250px'},
            layout=widgets.Layout(width='450px')
        )
        economic_widgets.append(self.thermal_efficiency_input)
        economic_widgets.append(widgets.HTML(
            "<p style='color: #666; margin-left: 20px; font-size: 0.9em;'>"
            "Geothermal plant efficiency: thermal BTU × efficiency = electrical BTU (default: 0.1 = 10%)</p>"
        ))
        self.economic_inputs['thermal_to_electrical_efficiency'] = self.thermal_efficiency_input
        
        # Days per year
        self.days_per_year_input = widgets.IntText(
            value=self.default_economics['days_per_year'],
            description='Days per Year:',
            style={'description_width': '250px'},
            layout=widgets.Layout(width='450px')
        )
        economic_widgets.append(self.days_per_year_input)
        economic_widgets.append(widgets.HTML(
            "<p style='color: #666; margin-left: 20px; font-size: 0.9em;'>"
            "Each RL timestep = 1 year. Daily rates × 365 = annual values</p>"
        ))
        self.economic_inputs['days_per_year'] = self.days_per_year_input
        
        # BTU to kWh conversion
        self.btu_to_kwh_input = widgets.FloatText(
            value=self.default_economics['btu_to_kwh'],
            description='BTU to kWh Factor:',
            style={'description_width': '250px'},
            layout=widgets.Layout(width='450px')
        )
        economic_widgets.append(self.btu_to_kwh_input)
        self.economic_inputs['btu_to_kwh'] = self.btu_to_kwh_input
        
        # Add operational parameters section
        economic_widgets.append(widgets.HTML("<hr><h4>💼 Operational Parameters</h4>"))
        
        # Energy production revenue
        self.energy_revenue_input = widgets.FloatText(
            value=self.default_economics['energy_production_revenue'],
            description='Energy Revenue ($/kWh):',
            style={'description_width': '250px'},
            layout=widgets.Layout(width='450px')
        )
        economic_widgets.append(self.energy_revenue_input)
        self.economic_inputs['energy_production_revenue'] = self.energy_revenue_input
        
        # Water production cost
        self.water_prod_cost_input = widgets.FloatText(
            value=self.default_economics['water_production_cost'],
            description='Water Prod. Cost ($/bbl):',
            style={'description_width': '250px'},
            layout=widgets.Layout(width='450px')
        )
        economic_widgets.append(self.water_prod_cost_input)
        self.economic_inputs['water_production_cost'] = self.water_prod_cost_input
        
        # Water injection cost
        self.water_inj_cost_input = widgets.FloatText(
            value=self.default_economics['water_injection_cost'],
            description='Water Inj. Cost ($/bbl):',
            style={'description_width': '250px'},
            layout=widgets.Layout(width='450px')
        )
        economic_widgets.append(self.water_inj_cost_input)
        self.economic_inputs['water_injection_cost'] = self.water_inj_cost_input
        
        # Scale factor
        self.scale_factor_input = widgets.FloatText(
            value=self.default_economics['scale_factor'],
            description='Scale Factor:',
            style={'description_width': '250px'},
            layout=widgets.Layout(width='450px')
        )
        economic_widgets.append(self.scale_factor_input)
        self.economic_inputs['scale_factor'] = self.scale_factor_input
        
        # Add calculated capital cost as read-only display (not editable)
        self.economic_inputs['fixed_capital_cost'] = widgets.HTML(
            value=f"${self.default_economics['fixed_capital_cost']:,.0f}",
            description='Total Capital Cost (calculated):',
            layout=widgets.Layout(width='450px')
        )
        
        # Current reward function display
        economic_widgets.extend([
            widgets.HTML("<hr><h4>📐 Geothermal Reward Function (Annual per Timestep)</h4>"),
            widgets.HTML("""
            <div style='background-color: #f0f8ff; padding: 10px; border-left: 4px solid #4CAF50;'>
            <pre>
<b>Step 1: Convert thermal energy to electrical energy</b>
  Energy_electrical_BTU = Energy_thermal_BTU × thermal_to_electrical_efficiency

<b>Step 2: Convert BTU to kWh</b>
  Energy_kWh = Energy_electrical_BTU × 0.000293071

<b>Step 3: Convert daily rates to annual (each timestep = 1 year)</b>
  Energy_kWh_year = Energy_kWh_day × 365
  Water_prod_year = Water_prod_day × 365
  Water_inj_year  = Water_inj_day × 365

<b>Step 4: Calculate annual economics</b>
  Reward = (energy_revenue × Energy_kWh_year
          - water_prod_cost × Water_prod_year
          - water_inj_cost × Water_inj_year) / scale_factor

<b>Default values:</b>
  • thermal_to_electrical_efficiency: 0.1 (10% power plant efficiency)
  • energy_revenue: $0.0011/kWh
  • water_prod_cost: $5/bbl (disposal)
  • water_inj_cost: $10/bbl (pumping)
            </pre>
            </div>
            """)
        ])
        
        self.economic_tab.children = economic_widgets
    
    def _update_rl_hyperparams_tab(self):
        """Update the RL hyperparameters configuration tab"""
        hyperparam_widgets = [
            widgets.HTML("<h3>🧠 RL Model Hyperparameters</h3>"),
            widgets.HTML("<p><i>Configure SAC algorithm and neural network parameters</i></p>")
        ]
        
        # Neural Network Configuration
        hyperparam_widgets.append(widgets.HTML("<hr><h4>🔗 Neural Network Architecture</h4>"))
        
        self.rl_hyperparams = {}
        
        # Hidden dimension
        self.rl_hyperparams['hidden_dim'] = widgets.IntSlider(
            value=self.default_rl_hyperparams['networks']['hidden_dim'],
            min=64, max=512, step=64,
            description='Hidden Dim:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        hyperparam_widgets.append(self.rl_hyperparams['hidden_dim'])
        
        # Policy type
        self.rl_hyperparams['policy_type'] = widgets.Dropdown(
            options=['deterministic', 'gaussian'],
            value=self.default_rl_hyperparams['networks']['policy_type'],
            description='Policy Type:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='300px')
        )
        hyperparam_widgets.append(self.rl_hyperparams['policy_type'])
        
        # Output activation
        self.rl_hyperparams['output_activation'] = widgets.Dropdown(
            options=['sigmoid', 'tanh'],
            value=self.default_rl_hyperparams['networks']['output_activation'],
            description='Output Activation:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='300px')
        )
        hyperparam_widgets.append(self.rl_hyperparams['output_activation'])
        
        # SAC Algorithm Parameters
        hyperparam_widgets.append(widgets.HTML("<hr><h4>🎯 SAC Algorithm Parameters</h4>"))
        
        # Discount factor
        self.rl_hyperparams['discount_factor'] = widgets.FloatSlider(
            value=self.default_rl_hyperparams['sac']['discount_factor'],
            min=0.9, max=0.999, step=0.001,
            description='Discount Factor:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        hyperparam_widgets.append(self.rl_hyperparams['discount_factor'])
        
        # Soft update tau
        self.rl_hyperparams['soft_update_tau'] = widgets.FloatLogSlider(
            value=self.default_rl_hyperparams['sac']['soft_update_tau'],
            base=10, min=-4, max=-1,
            description='Soft Update τ:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        hyperparam_widgets.append(self.rl_hyperparams['soft_update_tau'])
        
        # Entropy alpha
        self.rl_hyperparams['entropy_alpha'] = widgets.FloatSlider(
            value=self.default_rl_hyperparams['sac']['entropy_alpha'],
            min=0.0, max=1.0, step=0.01,
            description='Entropy α:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        hyperparam_widgets.append(self.rl_hyperparams['entropy_alpha'])
        
        # Learning rates
        hyperparam_widgets.append(widgets.HTML("<h5>📚 Learning Rates</h5>"))
        
        self.rl_hyperparams['critic_lr'] = widgets.FloatLogSlider(
            value=self.default_rl_hyperparams['sac']['critic_lr'],
            base=10, min=-5, max=-2,
            description='Critic LR:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        hyperparam_widgets.append(self.rl_hyperparams['critic_lr'])
        
        self.rl_hyperparams['policy_lr'] = widgets.FloatLogSlider(
            value=self.default_rl_hyperparams['sac']['policy_lr'],
            base=10, min=-5, max=-2,
            description='Policy LR:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        hyperparam_widgets.append(self.rl_hyperparams['policy_lr'])
        
        # Environment Prediction Mode Configuration
        hyperparam_widgets.append(widgets.HTML("<hr><h4>🔬 Environment Prediction Mode</h4>"))
        
        # Prediction mode selection
        self.rl_hyperparams['prediction_mode'] = widgets.RadioButtons(
            options=[
                ('State-based (Default - E2C Training Workflow)', 'state_based'),
                ('Latent-based (Faster - Pure Latent Evolution)', 'latent_based')
            ],
            value='state_based',  # Default to E2C workflow mode
            description='Prediction Mode:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='500px')
        )
        
        hyperparam_widgets.append(widgets.HTML(
            "<p><b>State-based:</b> Follows E2C training workflow (Spatial→Latent→Spatial cycle)<br/>"
            "<b>Latent-based:</b> Pure latent evolution (stays in latent space, faster)</p>"
        ))
        hyperparam_widgets.append(self.rl_hyperparams['prediction_mode'])
        
        # Training Configuration
        hyperparam_widgets.append(widgets.HTML("<hr><h4>🏋️ Training Configuration</h4>"))
        
        # Max episodes
        self.rl_hyperparams['max_episodes'] = widgets.IntSlider(
            value=self.default_rl_hyperparams['training']['max_episodes'],
            min=100, max=5000, step=100,
            description='Max Episodes:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        hyperparam_widgets.append(self.rl_hyperparams['max_episodes'])
        
        # Max steps per episode
        self.rl_hyperparams['max_steps_per_episode'] = widgets.IntSlider(
            value=self.default_rl_hyperparams['training']['max_steps_per_episode'],
            min=30, max=500, step=10,
            description='Max Steps/Episode:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        hyperparam_widgets.append(self.rl_hyperparams['max_steps_per_episode'])
        
        # Batch size
        self.rl_hyperparams['batch_size'] = widgets.Dropdown(
            options=[32, 64, 128, 256, 512],
            value=self.default_rl_hyperparams['training']['batch_size'],
            description='Batch Size:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='300px')
        )
        hyperparam_widgets.append(self.rl_hyperparams['batch_size'])
        
        # Replay memory capacity
        self.rl_hyperparams['replay_capacity'] = widgets.Dropdown(
            options=[10000, 50000, 100000, 500000, 1000000],
            value=self.default_rl_hyperparams['training']['replay_capacity'],
            description='Replay Capacity:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='300px')
        )
        hyperparam_widgets.append(self.rl_hyperparams['replay_capacity'])
        
        # Gradient clipping
        hyperparam_widgets.append(widgets.HTML("<h5>✂️ Gradient Clipping</h5>"))
        
        self.rl_hyperparams['gradient_clipping'] = widgets.Checkbox(
            value=self.default_rl_hyperparams['sac']['gradient_clipping'],
            description='Enable Gradient Clipping',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        hyperparam_widgets.append(self.rl_hyperparams['gradient_clipping'])
        
        self.rl_hyperparams['max_norm'] = widgets.FloatSlider(
            value=self.default_rl_hyperparams['sac']['max_norm'],
            min=1.0, max=50.0, step=1.0,
            description='Max Norm:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='400px')
        )
        hyperparam_widgets.append(self.rl_hyperparams['max_norm'])
        
        self.rl_hyperparams_tab.children = hyperparam_widgets
    
    def _update_action_variation_tab(self):
        """Create action variation configuration widgets"""
        
        variation_content = []
        
        # Main enable/disable toggle
        variation_content.extend([
            widgets.HTML("<h4>🌊 Enable Action Variation Enhancement</h4>"),
            widgets.Checkbox(
                value=self.default_action_variation['enabled'],
                description="Enable Wide Action Variation",
                style={'description_width': 'initial'}
            )
        ])
        
        # Variation mode selection
        variation_content.extend([
            widgets.HTML("<h4>🎯 Variation Mode</h4>"),
            widgets.Dropdown(
                options=[
                    ('Adaptive (Recommended)', 'adaptive'),
                    ('High Exploration', 'exploration'),
                    ('Balanced', 'exploitation'),
                    ('Minimal Variation', 'minimal')
                ],
                value=self.default_action_variation['mode'],
                description="Variation Strategy:",
                style={'description_width': 'initial'}
            ),
            widgets.HTML("<p><i>Adaptive: Changes variation based on training progress</i></p>")
        ])
        
        # Noise parameters
        variation_content.extend([
            widgets.HTML("<h4>🔊 Noise Parameters</h4>"),
            widgets.HBox([
                widgets.Label("Max Noise:", layout=widgets.Layout(width='100px')),
                widgets.FloatSlider(
                    value=self.default_action_variation['max_noise_std'],
                    min=0.05, max=0.5, step=0.05,
                    description="",
                    layout=widgets.Layout(width='200px')
                ),
                widgets.Label("(Strong initial exploration)", layout=widgets.Layout(width='200px'))
            ]),
            widgets.HBox([
                widgets.Label("Min Noise:", layout=widgets.Layout(width='100px')),
                widgets.FloatSlider(
                    value=self.default_action_variation['min_noise_std'],
                    min=0.001, max=0.1, step=0.005,
                    description="",
                    layout=widgets.Layout(width='200px')
                ),
                widgets.Label("(Final exploration level)", layout=widgets.Layout(width='200px'))
            ]),
            widgets.HBox([
                widgets.Label("Decay Rate:", layout=widgets.Layout(width='100px')),
                widgets.FloatSlider(
                    value=self.default_action_variation['noise_decay_rate'],
                    min=0.990, max=0.999, step=0.001,
                    description="",
                    layout=widgets.Layout(width='200px')
                ),
                widgets.Label("(How fast noise decreases)", layout=widgets.Layout(width='200px'))
            ])
        ])
        
        # Step variation
        variation_content.extend([
            widgets.HTML("<h4>📈 Step-wise Variation</h4>"),
            widgets.HBox([
                widgets.Label("Amplitude:", layout=widgets.Layout(width='100px')),
                widgets.FloatSlider(
                    value=self.default_action_variation['step_variation_amplitude'],
                    min=0.0, max=0.3, step=0.02,
                    description="",
                    layout=widgets.Layout(width='200px')
                ),
                widgets.Label("(Variation within episodes)", layout=widgets.Layout(width='200px'))
            ])
        ])
        
        # Well-specific strategies configuration
        variation_content.extend([
            widgets.HTML("<h4>🏭 Well-Specific Strategies</h4>"),
            widgets.HTML("<p><i>Different exploration strategies for each well</i></p>")
        ])
        
        # Create well strategy widgets
        well_widgets = []
        for well_name, strategy in self.default_action_variation['well_strategies'].items():
            well_type = "Producer" if well_name.startswith('P') else "Injector"
            
            well_box = widgets.VBox([
                widgets.HTML(f"<b>{well_name} ({well_type})</b>"),
                widgets.HBox([
                    widgets.Label("Variation:", layout=widgets.Layout(width='80px')),
                    widgets.FloatSlider(
                        value=strategy['variation'],
                        min=0.05, max=0.5, step=0.05,
                        description="",
                        layout=widgets.Layout(width='150px')
                    ),
                    widgets.Label("Bias:", layout=widgets.Layout(width='40px')),
                    widgets.FloatSlider(
                        value=strategy['bias'],
                        min=-0.1, max=0.1, step=0.01,
                        description="",
                        layout=widgets.Layout(width='150px')
                    )
                ])
            ])
            well_widgets.append(well_box)
        
        # Create 2x3 grid for wells
        well_grid = widgets.GridBox(
            well_widgets,
            layout=widgets.Layout(
                width='100%',
                grid_template_columns='repeat(3, 1fr)',
                grid_gap='10px'
            )
        )
        variation_content.append(well_grid)
        
        # Enhanced Gaussian Policy section
        variation_content.extend([
            widgets.HTML("<h4>🎲 Enhanced Gaussian Policy (Advanced)</h4>"),
            widgets.Checkbox(
                value=self.default_action_variation['enhanced_gaussian_policy']['enabled'],
                description="Use Gaussian Policy instead of Deterministic",
                style={'description_width': 'initial'}
            ),
            widgets.HTML("<p><i>Gaussian policy provides natural stochasticity for exploration</i></p>"),
            widgets.HBox([
                widgets.Label("Log Std Min:", layout=widgets.Layout(width='100px')),
                widgets.FloatSlider(
                    value=self.default_action_variation['enhanced_gaussian_policy']['log_std_bounds'][0],
                    min=-5.0, max=0.0, step=0.1,
                    description="",
                    layout=widgets.Layout(width='150px')
                ),
                widgets.Label("Max:", layout=widgets.Layout(width='40px')),
                widgets.FloatSlider(
                    value=self.default_action_variation['enhanced_gaussian_policy']['log_std_bounds'][1],
                    min=0.0, max=3.0, step=0.1,
                    description="",
                    layout=widgets.Layout(width='150px')
                )
            ]),
            widgets.HBox([
                widgets.Label("Entropy Weight:", layout=widgets.Layout(width='100px')),
                widgets.FloatSlider(
                    value=self.default_action_variation['enhanced_gaussian_policy']['entropy_weight'],
                    min=0.0, max=1.0, step=0.05,
                    description="",
                    layout=widgets.Layout(width='200px')
                ),
                widgets.Label("(Exploration bonus)", layout=widgets.Layout(width='150px'))
            ])
        ])
        
        # Expected results section
        variation_content.extend([
            widgets.HTML("<h4>📊 Expected Results</h4>"),
            widgets.HTML("""
            <div style='background-color: #f0f8ff; padding: 10px; border-radius: 5px;'>
                <b>Current System:</b> Action variation ~0.01 (tiny changes)<br/>
                <b>Enhanced System:</b> Action variation 0.2-0.4 (wide exploration)<br/>
                <br/>
                <b>Physical Impact:</b><br/>
                • BHP ranges: 1200-3000 psi (vs 1087-1305 psi narrow)<br/>
                • Energy injection: 50K-5M BTU/Day (vs 10-25M narrow)<br/>
                • Well differentiation: Each well explores differently<br/>
                • Temporal correlation: Actions vary smoothly within episodes
            </div>
            """)
        ])
        
        self.action_variation_tab.children = variation_content
    
    def _apply_configuration(self, button):
        """Apply configuration and pre-load ROM model + generate Z0"""
        with self.results_output:
            clear_output(wait=True)
            
            print("🔄 Applying RL configuration...")
            
            # Collect configuration
            config = self._collect_configuration()
            
            if config:
                self.config.update(config)
                print("✅ Configuration applied successfully!")
                print("Configuration Summary:")
                self._print_configuration_summary()
                
                # Store in a way that can be accessed from training script
                self._store_config_for_training()
                
                # Update ROM config file with selected controls and observations
                print("\n📝 Updating ROM config file with selected controls and observations...")
                rom_config_updated = self._update_rom_config_file()
                if rom_config_updated:
                    print("   ✅ ROM config file updated successfully!")
                else:
                    print("   ⚠️ Could not update ROM config file (non-critical)")
                
                # NEW: Load ROM model and generate Z0 immediately
                print("\n🚀 Pre-loading ROM model and generating Z0...")
                success = self._load_rom_and_generate_z0()
                
                if success:
                    print("🎉 ALL READY! Your models and Z0 are pre-loaded for RL training!")
                    print("   ✅ ROM model: Loaded and ready")
                    print("   ✅ Realistic Z0: Generated and ready")
                    print("   🚀 Main run file can now start RL training directly!")
                else:
                    print("⚠️ Configuration saved, but model loading failed.")
                    print("   Main run file will handle model loading as fallback.")
                
            else:
                print("❌ Configuration failed!")
    
    def _load_rom_and_generate_z0(self):
        """Load ROM model and generate realistic Z0 from selected states"""
        try:
            # Setup device
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() 
                                       else ("mps" if torch.backends.mps.is_available() 
                                             else "cpu"))
            print(f"   🔧 Using device: {self.device}")
            
            # Check if ROM model is selected
            if not self.config.get('selected_rom'):
                print("❌ No ROM model selected!")
                print("   💡 Please:")
                print("      1. Go to the '🏔️ States' tab")
                print("      2. Select a ROM model from the dropdown")
                print("      3. Click 'Apply Configuration' again")
                if len(self.config.get('rom_models', [])) == 0:
                    print("   📁 No ROM models found. Please:")
                    print("      1. Check the ROM Models Folder path")
                    print("      2. Click '🔍 Scan Folders' to scan for ROM models")
                return False
            
            # Check if states are selected
            if not self.config.get('selected_states'):
                print("❌ No states selected!")
                return False
            
            # 🎯 CRITICAL: Load ROM config from ROM_Refactored to ensure consistency
            # This ensures ROM model uses EXACT same config as ROM training (no parameter leakage)
            rom_config_path = Path(__file__).parent.parent.parent / 'ROM_Refactored' / 'config.yaml'
            
            if not rom_config_path.exists():
                print(f"❌ ROM config not found at {rom_config_path}")
                print("   Please ensure ROM_Refactored/config.yaml exists")
                return False
            
            print(f"   📄 Loading ROM config from: {rom_config_path}")
            rom_config = Config(str(rom_config_path))
            
            # 🎯 CRITICAL: Extract architecture parameters from selected model filename
            # and update ROM config to match the saved model architecture
            selected_rom = self.config['selected_rom']
            model_info = selected_rom.get('info', {})
            
            print("   🔧 Matching ROM config to selected model architecture...")
            config_updated = False
            
            # Update latent_dim if found in model filename
            if 'latent' in model_info or 'latent_dim' in model_info:
                latent_dim = model_info.get('latent_dim') or model_info.get('latent')
                if latent_dim and rom_config.model.get('latent_dim') != latent_dim:
                    rom_config.model['latent_dim'] = latent_dim
                    print(f"      ✅ Updated latent_dim: {rom_config.model.get('latent_dim')} → {latent_dim}")
                    config_updated = True
            
            # Update batch_size if found (for reference, though not directly used in model architecture)
            if 'batch_size' in model_info:
                batch_size = model_info['batch_size']
                print(f"      📊 Model was trained with batch_size: {batch_size}")
            
            # Update nsteps if found
            if 'nsteps' in model_info or 'steps' in model_info:
                nsteps = model_info.get('nsteps') or model_info.get('steps')
                if nsteps and rom_config.training.get('nsteps') != nsteps:
                    rom_config.training['nsteps'] = nsteps
                    print(f"      ✅ Updated nsteps: {rom_config.training.get('nsteps')} → {nsteps}")
                    config_updated = True
            
            # Update channels if found
            if 'channels' in model_info:
                channels = model_info['channels']
                if channels and rom_config.model.get('n_channels') != channels:
                    rom_config.model['n_channels'] = channels
                    print(f"      ✅ Updated n_channels: {rom_config.model.get('n_channels')} → {channels}")
                    config_updated = True
            
            # 🎯 CRITICAL: Extract transition encoder architecture from checkpoint file
            # This ensures perfect match regardless of config file state
            transition_file = selected_rom.get('transition')
            if transition_file and os.path.exists(transition_file):
                try:
                    print("   🔍 Extracting transition encoder architecture from checkpoint...")
                    checkpoint = torch.load(transition_file, map_location='cpu', weights_only=False)
                    
                    # Extract encoder hidden dimensions from checkpoint state_dict
                    encoder_hidden_dims = []
                    layer_idx = 0
                    while True:
                        weight_key = f"trans_encoder.{layer_idx}.0.weight"
                        if weight_key in checkpoint:
                            # Get output dimension from weight shape
                            weight_shape = checkpoint[weight_key].shape
                            out_dim = weight_shape[0]
                            encoder_hidden_dims.append(out_dim)
                            layer_idx += 1
                        else:
                            break
                    
                    if encoder_hidden_dims:
                        # The last layer outputs to latent_dim, so we exclude it
                        # We only want the hidden layers
                        if len(encoder_hidden_dims) > 1:
                            encoder_hidden_dims = encoder_hidden_dims[:-1]
                        
                        current_dims = rom_config.transition.get('encoder_hidden_dims', [])
                        if current_dims != encoder_hidden_dims:
                            rom_config.transition['encoder_hidden_dims'] = encoder_hidden_dims
                            print(f"      ✅ Updated encoder_hidden_dims: {current_dims} → {encoder_hidden_dims}")
                            config_updated = True
                        else:
                            print(f"      ✅ encoder_hidden_dims already match: {encoder_hidden_dims}")
                    else:
                        print(f"      ⚠️ Could not extract encoder_hidden_dims from checkpoint")
                except Exception as e:
                    print(f"      ⚠️ Could not extract transition architecture from checkpoint: {e}")
                    import traceback
                    traceback.print_exc()
            
            if config_updated:
                print("   ✅ ROM config updated to match selected model architecture")
            else:
                print("   ✅ ROM config matches selected model (no updates needed)")
            
            # Note: ROM config doesn't need RL-specific updates
            # The ROM model will be initialized with ROM config as-is
            # RL-specific parameters are handled separately in RL training
            
            # Initialize ROM model using ROM config (ensures consistency)
            print("   🧠 Initializing ROM model with matched architecture...")
            if ROMWithE2C is None:
                print("❌ ROMWithE2C not available!")
                return False
            
            self.loaded_rom_model = ROMWithE2C(rom_config).to(self.device)
            
            # Load pre-trained weights
            selected_rom = self.config['selected_rom']
            encoder_file = selected_rom['encoder']
            decoder_file = selected_rom['decoder']
            transition_file = selected_rom['transition']
            
            print("   📥 Loading pre-trained weights...")
            print(f"      • Encoder: {os.path.basename(encoder_file)}")
            print(f"      • Decoder: {os.path.basename(decoder_file)}")
            print(f"      • Transition: {os.path.basename(transition_file)}")
            
            self.loaded_rom_model.model.load_weights_from_file(encoder_file, decoder_file, transition_file)
            self.loaded_rom_model.eval()  # Set to evaluation mode
            print("   ✅ ROM model loaded successfully!")
            
            # Generate realistic Z0 options from ALL cases
            print("   🏔️ Generating multiple realistic Z0 options from selected states...")
            self.generated_z0_options, selected_states, state_t_seq = generate_z0_from_dashboard(
                self.config, self.loaded_rom_model, self.device
            )
            
            # Store metadata for multiple Z0 options
            self.z0_metadata = {
                'selected_states': selected_states,
                'z0_shape': self.generated_z0_options.shape,
                'z0_device': str(self.generated_z0_options.device),
                'num_cases': self.generated_z0_options.shape[0],
                'z0_stats': {
                    'mean': self.generated_z0_options.mean().item(),
                    'std': self.generated_z0_options.std().item(),
                    'min': self.generated_z0_options.min().item(),
                    'max': self.generated_z0_options.max().item(),
                    'per_case_means': self.generated_z0_options.mean(dim=1),
                    'per_case_stds': self.generated_z0_options.std(dim=1)
                },
                'source': f'Multiple initial states from {self.generated_z0_options.shape[0]} cases'
            }
            
            print(f"   ✅ Multiple Z0 options generated successfully!")
            print(f"      • Source states: {selected_states}")
            print(f"      • Z0 options shape: {self.generated_z0_options.shape} ({self.generated_z0_options.shape[0]} different initial states)")
            print(f"      • Z0 stats: mean={self.z0_metadata['z0_stats']['mean']:.4f}, "
                  f"std={self.z0_metadata['z0_stats']['std']:.4f}")
            print(f"      • Ready for random sampling in RL training!")
            
            # Mark as ready
            self.models_ready = True
            
            # Update global storage
            import builtins
            builtins.rl_dashboard_config = self.config
            builtins.rl_loaded_rom = self.loaded_rom_model
            builtins.rl_generated_z0_options = self.generated_z0_options  # Now stores multiple Z0 options
            builtins.rl_z0_metadata = self.z0_metadata
            builtins.rl_models_ready = True
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading ROM and generating Z0: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _collect_configuration(self):
        """Collect configuration from all tabs"""
        config = {}
        
        try:
            # State configuration
            if hasattr(self, 'state_checkboxes'):
                selected_states = []
                state_scaling = {}
                
                # Get states in training order if available
                display_order = self.config.get('available_states', [])
                if hasattr(self, 'training_channel_order') and self.training_channel_order:
                    # Filter to training order
                    display_order = [s for s in self.training_channel_order if s in display_order]
                
                for state_name in display_order:
                    if state_name in self.state_checkboxes:
                        checkbox = self.state_checkboxes[state_name]
                        if checkbox.value:
                            selected_states.append(state_name)
                            state_scaling[state_name] = self.scaling_radios[state_name].value
                
                # Store selected states in training order
                if hasattr(self, 'training_channel_order') and self.training_channel_order:
                    # Reorder to match training channel order
                    ordered_selected = [s for s in self.training_channel_order if s in selected_states]
                    # Add any selected states not in training order (shouldn't happen, but be safe)
                    for s in selected_states:
                        if s not in ordered_selected:
                            ordered_selected.append(s)
                    config['selected_states'] = ordered_selected
                else:
                    config['selected_states'] = selected_states
                
                config['state_scaling'] = state_scaling
                
                # Also store training channel order for use in generate_z0_from_dashboard
                if hasattr(self, 'training_channel_order') and self.training_channel_order:
                    config['training_channel_order'] = self.training_channel_order
            
            # ROM model selection
            if hasattr(self, 'rom_selector') and self.rom_selector is not None:
                selected_idx = self.rom_selector.value
                if selected_idx is not None and selected_idx < len(self.config.get('rom_models', [])):
                    config['selected_rom'] = self.config['rom_models'][selected_idx]
                elif len(self.config.get('rom_models', [])) == 0:
                    print("      ⚠️ Warning: No ROM models found. Please scan folders first.")
                else:
                    print("      ⚠️ Warning: No ROM model selected. Please select a ROM model from the dropdown.")
            elif len(self.config.get('rom_models', [])) == 0:
                print("      ⚠️ Warning: No ROM models available. Please scan folders first.")
            else:
                print("      ⚠️ Warning: ROM selector not available. Please scan folders first.")
            
            # Action configuration - collect selected controls and observations
            detection_details = getattr(self, 'detection_details', {})
            well_names_map = detection_details.get('well_names', {})
            producer_names = well_names_map.get('producers', [f'P{i+1}' for i in range(3)])
            injector_names = well_names_map.get('injectors', [f'I{i+1}' for i in range(3)])
            num_producers = detection_details.get('num_producers', len(producer_names))
            num_injectors = detection_details.get('num_injectors', len(injector_names))
            
            # Collect selected controls from checkboxes
            selected_controls = {}
            if hasattr(self, 'control_selections') and self.control_selections:
                for var_name, checkbox in self.control_selections.items():
                    if checkbox.value:  # If selected as control
                        # Get ranges from widgets
                        ranges = {}
                        if var_name in self.variable_range_widgets:
                            for well_name, well_widgets in self.variable_range_widgets[var_name].items():
                                ranges[well_name] = {
                                    'min': well_widgets['min'].value,
                                    'max': well_widgets['max'].value
                                }
                        selected_controls[var_name] = ranges
            
            # Collect selected observations from checkboxes
            selected_observations = []
            if hasattr(self, 'observation_selections') and self.observation_selections:
                for var_name, checkbox in self.observation_selections.items():
                    if checkbox.value:  # If selected as observation
                        selected_observations.append(var_name)
            
            # Store action configuration
            action_ranges = {
                'controls': selected_controls,
                'observations': selected_observations,
                'well_names': well_names_map,
                'num_producers': num_producers,
                'num_injectors': num_injectors,
                'num_wells': num_producers + num_injectors
            }
            
            # Legacy support: Also store in old format for backward compatibility
            # Extract BHP and water injection if they exist
            if 'BHP' in selected_controls:
                action_ranges['bhp'] = selected_controls['BHP']
            if 'WATRATRC' in selected_controls:
                action_ranges['water_injection'] = selected_controls['WATRATRC']
            # Note: ENERGYRATE is an observation, not a control in geothermal projects
            
            config['action_ranges'] = action_ranges
            
            # Economic configuration
            if hasattr(self, 'economic_inputs'):
                economic_params = {}
                for param_key, input_widget in self.economic_inputs.items():
                    if hasattr(input_widget, 'value'):
                        economic_params[param_key] = input_widget.value
                
                # Calculate total capital cost from pre-project parameters
                years_before = economic_params.get('years_before_project_start', 3)
                cost_per_year = economic_params.get('capital_cost_per_year', 6000000.0)
                economic_params['fixed_capital_cost'] = years_before * cost_per_year
                
                config['economic_params'] = economic_params
            
            # RL Hyperparameters configuration
            if hasattr(self, 'rl_hyperparams'):
                rl_hyperparams = {}
                for param_key, input_widget in self.rl_hyperparams.items():
                    rl_hyperparams[param_key] = input_widget.value
                
                config['rl_hyperparams'] = rl_hyperparams
            
            # Action Variation configuration
            if hasattr(self, 'action_variation_tab') and self.action_variation_tab.children:
                action_variation = {}
                
                # Extract values from the action variation tab widgets
                widgets_list = self.action_variation_tab.children
                
                if len(widgets_list) > 1:
                    # Enable/disable checkbox
                    action_variation['enabled'] = widgets_list[1].value if hasattr(widgets_list[1], 'value') else self.default_action_variation['enabled']
                    
                    # Variation mode dropdown  
                    if len(widgets_list) > 3:
                        action_variation['mode'] = widgets_list[3].value if hasattr(widgets_list[3], 'value') else self.default_action_variation['mode']
                    
                    # Use default values for now (could be enhanced to read from widgets)
                    action_variation.update({
                        'noise_decay_rate': self.default_action_variation['noise_decay_rate'],
                        'max_noise_std': self.default_action_variation['max_noise_std'],
                        'min_noise_std': self.default_action_variation['min_noise_std'],
                        'step_variation_amplitude': self.default_action_variation['step_variation_amplitude'],
                        'well_strategies': self.default_action_variation['well_strategies'],
                        'enhanced_gaussian_policy': self.default_action_variation['enhanced_gaussian_policy']
                    })
                
                config['action_variation'] = action_variation
            
            # ROM compatibility handled through training-only normalization parameters
            
            # 🎯 CRITICAL: Calculate TRAINING-ONLY normalization parameters (fixes data leakage)
            print("      🔄 Calculating TRAINING-ONLY normalization parameters...")
            # Get selected states from config
            selected_states = config.get('selected_states', [])
            # Normalize state folder path
            state_folder_normalized = os.path.normpath(self.state_folder)
            training_params = calculate_training_only_normalization_params(state_folder_normalized, selected_states=selected_states)
            if training_params:
                config['training_only_normalization_params'] = training_params
                print("      ✅ TRAINING-ONLY normalization parameters calculated successfully")
                print(f"         🎯 NO DATA LEAKAGE: Parameters from training split only")
                print(f"         📊 Variables: {list(training_params.keys())}")
                
                # ✨ NEW: Automatically save normalization parameters for RL training
                norm_file = save_normalization_parameters_for_rl(training_params)
                if norm_file:
                    print(f"      🔗 RL training can now use: {norm_file}")
                    config['normalization_file'] = norm_file
                
                # Compare with JSON parameters to show improvement
                preprocessing_params = self._load_preprocessing_normalization_parameters()
                if preprocessing_params:
                    config['preprocessing_normalization_params'] = preprocessing_params
                    print("      📊 Legacy preprocessing parameters also loaded for comparison")
                    
                    # Compare BHP ranges as example
                    try:
                        json_bhp_min = float(preprocessing_params['control_variables']['BHP']['parameters']['min'])
                        json_bhp_max = float(preprocessing_params['control_variables']['BHP']['parameters']['max'])
                        train_bhp_min = training_params['BHP']['min']
                        train_bhp_max = training_params['BHP']['max']
                        
                        print(f"      🔍 BHP Range Comparison:")
                        print(f"         Legacy JSON: [{json_bhp_min:.2f}, {json_bhp_max:.2f}] (data leakage)")
                        print(f"         Training-only: [{train_bhp_min:.2f}, {train_bhp_max:.2f}] (corrected)")
                        
                        if abs(json_bhp_min - train_bhp_min) > 0.01 or abs(json_bhp_max - train_bhp_max) > 0.01:
                            print(f"      ⚠️ CONFIRMED: JSON parameters include test data (DATA LEAKAGE)")
                            print(f"      ✅ Using corrected TRAINING-ONLY parameters")
                        else:
                            print(f"      ✅ JSON parameters appear to be training-only")
                    except:
                        print(f"      📊 Legacy comparison not available")
                else:
                    print("      💡 No legacy preprocessing parameters found for comparison")
            else:
                print("      ❌ Failed to calculate training-only parameters")
                # Fallback to preprocessing parameters if available
                preprocessing_params = self._load_preprocessing_normalization_parameters()
                if preprocessing_params:
                    config['preprocessing_normalization_params'] = preprocessing_params
                    print("      ⚠️ Fallback: Using legacy preprocessing parameters (may contain data leakage)")
                else:
                    print("      🚨 No normalization parameters available!")
                    print("      💡 Please ensure data files are available in state folder")
            
            # Add folder paths for state processing
            config['state_folder'] = self.state_folder
            config['rom_folder'] = self.rom_folder
            
            return config
            
        except Exception as e:
            print(f"❌ Error collecting configuration: {e}")
            return None
    
    def _store_config_for_training(self):
        """Store configuration for use in training script"""
        # Store as global variable that can be accessed
        import builtins
        builtins.rl_dashboard_config = self.config
        print("💾 Configuration stored globally as 'rl_dashboard_config'")
    
    def _print_configuration_summary(self):
        """Print a summary of the current configuration"""
        print(f"Configuration: {len(self.config.get('selected_states', []))} states selected")
        
        if 'selected_rom' in self.config and self.config['selected_rom'] is not None:
            rom = self.config['selected_rom']
            print(f"ROM: {rom.get('name', 'Unknown')}")
        else:
            print("ROM: Not selected")
        
        if 'action_ranges' in self.config and self.config['action_ranges']:
            ar = self.config['action_ranges']
            num_prod = ar.get('num_producers', 'unknown')
            num_wells = ar.get('num_wells', 6)
            num_inj = num_wells - num_prod if isinstance(num_prod, int) and isinstance(num_wells, int) else 'unknown'
            print(f"Wells: {num_prod} producers, {num_inj} injectors")
        else:
            print("Wells: Not configured")
        
        if 'rl_hyperparams' in self.config and self.config['rl_hyperparams']:
            hp = self.config['rl_hyperparams']
            print(f"\n🧠 RL Hyperparameters:")
            print(f"   • Policy Type: {hp.get('policy_type', 'unknown')}")
            print(f"   • Hidden Dim: {hp.get('hidden_dim', 'unknown')}")
            discount_factor = hp.get('discount_factor', 'unknown')
            if isinstance(discount_factor, (int, float)):
                print(f"   • Discount Factor: {discount_factor:.3f}")
            else:
                print(f"   • Discount Factor: {discount_factor}")
            critic_lr = hp.get('critic_lr', 'unknown')
            policy_lr = hp.get('policy_lr', 'unknown')
            if isinstance(critic_lr, (int, float)) and isinstance(policy_lr, (int, float)):
                print(f"   • Learning Rates: C={critic_lr:.1e}, P={policy_lr:.1e}")
            else:
                print(f"   • Learning Rates: C={critic_lr}, P={policy_lr}")
            max_episodes = hp.get('max_episodes', 'unknown')
            max_steps = hp.get('max_steps_per_episode', 'unknown')
            print(f"   • Training: {max_episodes} episodes, {max_steps} steps/episode")
            batch_size = hp.get('batch_size', 'unknown')
            replay_capacity = hp.get('replay_capacity', 'unknown')
            print(f"   • Batch Size: {batch_size}, Replay: {replay_capacity}")
        else:
            print("\n🧠 RL Hyperparameters: Not configured")
    
    def _reset_defaults(self, button):
        """Reset all settings to defaults"""
        with self.results_output:
            clear_output(wait=True)
            print("🔄 Resetting to default values...")
            
            # Reset folder paths from config (or ROM_Refactored defaults)
            if self.config_obj and hasattr(self.config_obj, 'paths'):
                paths = self.config_obj.paths
                if isinstance(paths, dict):
                    self.rom_folder_input.value = paths.get('rom_models_dir', '../ROM_Refactored/saved_models/')
                    self.state_folder_input.value = paths.get('state_data_dir', '../ROM_Refactored/sr3_batch_output/')
                else:
                    self.rom_folder_input.value = getattr(paths, 'rom_models_dir', '../ROM_Refactored/saved_models/')
                    self.state_folder_input.value = getattr(paths, 'state_data_dir', '../ROM_Refactored/sr3_batch_output/')
            else:
                # Fallback to ROM_Refactored defaults
                self.rom_folder_input.value = "../ROM_Refactored/saved_models/"
                self.state_folder_input.value = "../ROM_Refactored/sr3_batch_output/"
            
            print("✅ Reset completed! Please scan folders again.")
    
    def _save_configuration(self, button):
        """Save current configuration to file"""
        with self.results_output:
            clear_output(wait=True)
            
            config = self._collect_configuration()
            if config:
                # Save to JSON file
                config_file = "rl_config.json"
                
                # Convert numpy types to native Python types for JSON serialization
                json_config = self._convert_for_json(config)
                
                try:
                    with open(config_file, 'w') as f:
                        json.dump(json_config, f, indent=4)
                    print(f"💾 Configuration saved to {config_file}")
                    
                    # Also update ROM config file
                    print("\n📝 Updating ROM config file with selected controls and observations...")
                    rom_config_updated = self._update_rom_config_file()
                    if rom_config_updated:
                        print("   ✅ ROM config file updated successfully!")
                    else:
                        print("   ⚠️ Could not update ROM config file (non-critical)")
                except Exception as e:
                    print(f"❌ Error saving configuration: {e}")
            else:
                print("❌ No configuration to save!")
    
    def _update_rom_config_file(self):
        """
        Update ROM_Refactored/config.yaml with selected controls and observations from dashboard
        
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            import yaml
            from pathlib import Path
            
            # Get ROM config path
            current_dir = Path(__file__).parent.parent.parent
            rom_config_path = current_dir / 'ROM_Refactored' / 'config.yaml'
            
            if not rom_config_path.exists():
                print(f"   ⚠️ ROM config file not found at {rom_config_path}")
                return False
            
            # Load current ROM config
            with open(rom_config_path, 'r', encoding='utf-8') as f:
                rom_config = yaml.safe_load(f)
            
            # Get selected controls and observations from collected configuration
            action_ranges = self.config.get('action_ranges', {})
            selected_controls = action_ranges.get('controls', {})
            selected_observations = action_ranges.get('observations', [])
            
            # Debug: Print what we're updating
            print(f"   📋 Selected Controls: {list(selected_controls.keys())}")
            print(f"   📋 Selected Observations: {selected_observations}")
            
            # Get variable definitions from detection details
            detection_details = getattr(self, 'detection_details', {})
            control_vars = detection_details.get('control_variables', {})
            obs_vars = detection_details.get('observation_definitions', {})
            
            # If not in detection_details, try loading from ROM config directly
            if not control_vars or not obs_vars:
                if 'data' in rom_config:
                    controls_config = rom_config['data'].get('controls', {})
                    observations_config = rom_config['data'].get('observations', {})
                    if controls_config and 'variables' in controls_config:
                        control_vars = controls_config['variables']
                    if observations_config and 'variables' in observations_config:
                        obs_vars = observations_config['variables']
            
            # Combine all variables (controls + observations) to get full definitions
            all_vars = {}
            all_vars.update(control_vars)
            all_vars.update(obs_vars)
            
            # Ensure data section exists
            if 'data' not in rom_config:
                rom_config['data'] = {}
            
            # Update controls section
            controls_variables = {}
            controls_order = []
            num_controls = 0
            
            for var_name in selected_controls.keys():
                if var_name in all_vars:
                    var_def = all_vars[var_name].copy()
                    # Remove observation-specific fields (indices, group_name)
                    var_def.pop('indices', None)
                    var_def.pop('group_name', None)
                    # Ensure required fields are present
                    if 'name' not in var_def:
                        var_def['name'] = var_name
                    controls_variables[var_name] = var_def
                    controls_order.append(var_name)
                    # Count number of wells for this control
                    num_wells = var_def.get('num_wells', len(var_def.get('well_names', [])))
                    num_controls += num_wells
            
            rom_config['data']['controls'] = {
                'variables': controls_variables,
                'order': controls_order,
                'num_controls': num_controls
            }
            
            # Update observations section
            observations_variables = {}
            observations_order = []
            num_observations = 0
            current_index = 0
            
            for var_name in selected_observations:
                if var_name in all_vars:
                    var_def = all_vars[var_name].copy()
                    # Ensure required fields are present
                    if 'name' not in var_def:
                        var_def['name'] = var_name
                    # Calculate and set indices correctly
                    num_wells = var_def.get('num_wells', len(var_def.get('well_names', [])))
                    var_def['indices'] = list(range(current_index, current_index + num_wells))
                    current_index += num_wells
                    num_observations += num_wells
                    # Preserve group_name if it exists (for observations)
                    if 'group_name' not in var_def and var_name in obs_vars:
                        var_def['group_name'] = obs_vars[var_name].get('group_name', f'{var_def.get("display_name", var_name)} (All {var_def.get("well_type", "Wells").title()})')
                    observations_variables[var_name] = var_def
                    observations_order.append(var_name)
            
            rom_config['data']['observations'] = {
                'variables': observations_variables,
                'order': observations_order,
                'num_observations': num_observations
            }
            
            # Save updated config back to file
            with open(rom_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(rom_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
            print(f"   📝 Updated controls: {controls_order} (num_controls={num_controls})")
            print(f"   📝 Updated observations: {observations_order} (num_observations={num_observations})")
            print(f"   💾 Saved to: {rom_config_path}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error updating ROM config: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types"""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def display(self):
        """Display the dashboard"""
        if WIDGETS_AVAILABLE:
            display(self.main_widget)
        else:
            print("❌ Cannot display dashboard - ipywidgets not available")
    
    def get_configuration(self):
        """Get the current configuration"""
        return self.config.copy()


def create_rl_configuration_dashboard(config_path='config.yaml'):
    """
    Create and return an RL configuration dashboard
    
    Args:
        config_path: Path to config.yaml file (default: 'config.yaml')
    
    Returns:
        RLConfigurationDashboard: Interactive dashboard instance
    """
    if not WIDGETS_AVAILABLE:
        print("❌ Cannot create dashboard - ipywidgets not available")
        print("Please install ipywidgets: pip install ipywidgets")
        return None
    
    print("🎮 Creating RL Configuration Dashboard...")
    
    dashboard = RLConfigurationDashboard(config_path=config_path)
    
    return dashboard


def launch_rl_config_dashboard(config_path='config.yaml'):
    """
    Launch the RL configuration dashboard
    
    Args:
        config_path: Path to config.yaml file (default: 'config.yaml')
    
    Returns:
        RLConfigurationDashboard: Dashboard instance for configuration
    """
    dashboard = create_rl_configuration_dashboard(config_path=config_path)
    
    if dashboard:
        print("✅ Dashboard created successfully!")
        print("🔧 Please configure your RL parameters using the dashboard below.")
        print(f"📁 Default paths: ROM Models = {dashboard.rom_folder}, State Data = {dashboard.state_folder}")
        print("   💡 You can change these paths in the 'Folder Configuration' section")
        print("Click 'Apply Configuration' when ready to proceed.")
        dashboard.display()
        return dashboard
    else:
        print("❌ Failed to create dashboard")
        return None


# Utility functions for accessing configuration in training script
def get_rl_config():
    """Get the stored RL configuration"""
    import builtins
    if hasattr(builtins, 'rl_dashboard_config'):
        return builtins.rl_dashboard_config
    else:
        print("❌ No RL configuration found. Please run the dashboard first.")
        return None


def has_rl_config():
    """Check if RL configuration is available"""
    import builtins
    return hasattr(builtins, 'rl_dashboard_config')


def get_pre_loaded_rom():
    """Get the pre-loaded ROM model from dashboard"""
    import builtins
    if hasattr(builtins, 'rl_loaded_rom'):
        return builtins.rl_loaded_rom
    else:
        print("❌ No pre-loaded ROM model found. Please apply dashboard configuration first.")
        return None


def get_pre_generated_z0():
    """Get the pre-generated Z0 options from dashboard for random sampling"""
    import builtins
    if hasattr(builtins, 'rl_generated_z0_options'):
        return builtins.rl_generated_z0_options, builtins.rl_z0_metadata
    else:
        print("❌ No pre-generated Z0 options found. Please apply dashboard configuration first.")
        return None, None


def are_models_ready():
    """Check if ROM model and Z0 are pre-loaded and ready"""
    import builtins
    return hasattr(builtins, 'rl_models_ready') and builtins.rl_models_ready


def apply_state_scaling(data, state_name, rl_config):
    """
    Apply scaling to state data based on dashboard configuration
    
    Args:
        data: numpy array of state data
        state_name: name of the state (e.g., 'SW', 'PRES', etc.)
        rl_config: configuration from dashboard
        
    Returns:
        tuple: (scaled_data, scaling_params)
    """
    if state_name not in rl_config.get('state_scaling', {}):
        print(f"⚠️ No scaling configuration for {state_name}, using min-max")
        scaling_type = 'minmax'
    else:
        scaling_type = rl_config['state_scaling'][state_name]
    
    if scaling_type == 'log':
        # Log normalization
        epsilon = 1e-8
        positive_data = data[data > 0]
        if len(positive_data) > 0:
            min_pos = np.min(positive_data)
            data_shifted = np.maximum(data, min_pos)  # Ensure all values >= min_pos
        else:
            data_shifted = data + epsilon
        
        log_data = np.log(data_shifted + epsilon)
        log_min = np.min(log_data)
        log_max = np.max(log_data)
        
        if log_max > log_min:
            scaled_data = (log_data - log_min) / (log_max - log_min)
        else:
            scaled_data = np.zeros_like(log_data)
            
        scaling_params = {
            'type': 'log',
            'log_min': log_min,
            'log_max': log_max,
            'epsilon': epsilon,
            'min_positive': min_pos if len(positive_data) > 0 else epsilon
        }
        
    else:
        # Min-max normalization (default)
        positive_data = data[data > 0]
        if len(positive_data) > 0:
            data_min = np.min(positive_data)  # Use minimum positive value
        else:
            data_min = np.min(data)
            
        data_max = np.max(data)
        
        if data_max > data_min:
            scaled_data = (data - data_min) / (data_max - data_min)
        else:
            scaled_data = np.zeros_like(data)
            
        scaling_params = {
            'type': 'minmax',
            'min': data_min,
            'max': data_max
        }
    
    return scaled_data, scaling_params


def get_action_scaling_params(rl_config):
    """
    Get action scaling parameters from dashboard configuration
    
    Args:
        rl_config: configuration from dashboard
        
    Returns:
        dict: scaling parameters for actions
    """
    action_ranges = rl_config.get('action_ranges', {})
    
    # Get selected controls (new dynamic structure)
    selected_controls = action_ranges.get('controls', {})
    
    scaling_params = {
        'controls': {},  # Dynamic control variables
        'num_producers': action_ranges.get('num_producers', 3),
        'num_injectors': action_ranges.get('num_injectors', 3)
    }
    
    # Process each selected control variable
    for var_name, well_ranges in selected_controls.items():
        if well_ranges:
            # Aggregate min and max across all wells
            var_mins = [ranges['min'] for ranges in well_ranges.values()]
            var_maxs = [ranges['max'] for ranges in well_ranges.values()]
            
            scaling_params['controls'][var_name] = {
                'min': min(var_mins) if var_mins else 0.0,
                'max': max(var_maxs) if var_maxs else 1000.0,
                'ranges': well_ranges
            }
    
    # Legacy support: Also provide old format for backward compatibility
    if 'bhp' in action_ranges:
        bhp_ranges = action_ranges['bhp']
        if bhp_ranges:
            bhp_mins = [ranges['min'] for ranges in bhp_ranges.values()]
            bhp_maxs = [ranges['max'] for ranges in bhp_ranges.values()]
            scaling_params['bhp'] = {
                'min': min(bhp_mins) if bhp_mins else 1087.78,
                'max': max(bhp_maxs) if bhp_maxs else 1305.0,
                'ranges': bhp_ranges
            }
    
    if 'water_injection' in action_ranges:
        water_ranges = action_ranges['water_injection']
        if water_ranges:
            water_mins = [ranges['min'] for ranges in water_ranges.values()]
            water_maxs = [ranges['max'] for ranges in water_ranges.values()]
            scaling_params['water_injection'] = {
                'min': min(water_mins) if water_mins else 0.0,
                'max': max(water_maxs) if water_maxs else 1000.0,
                'ranges': water_ranges
            }
    
    if 'gas_injection' in action_ranges:
        gas_ranges = action_ranges['gas_injection']
        if gas_ranges:
            gas_mins = [ranges['min'] for ranges in gas_ranges.values()]
            gas_maxs = [ranges['max'] for ranges in gas_ranges.values()]
            scaling_params['gas_injection'] = {
                'min': min(gas_mins) if gas_mins else 10064800.2,
                'max': max(gas_maxs) if gas_maxs else 24720266.0,
                'ranges': gas_ranges
            }
    
    return scaling_params


def get_reward_function_params(rl_config):
    """
    Get reward function parameters from dashboard configuration
    
    Args:
        rl_config: configuration from dashboard
        
    Returns:
        dict: parameters for reward function
    """
    economic_params = rl_config.get('economic_params', {})
    
    # Default values for geothermal project
    defaults = {
        'energy_production_revenue': 0.0011,  # Revenue from energy production ($/kWh electrical) - POSITIVE
        'water_production_cost': 5.0,         # Cost for water production disposal ($/bbl) - NEGATIVE
        'water_injection_cost': 10.0,         # Cost for water injection ($/bbl) - NEGATIVE
        'btu_to_kwh': 0.000293071,            # BTU to kWh conversion factor
        'days_per_year': 365,                 # Days per year (each RL timestep = 1 year)
        'thermal_to_electrical_efficiency': 0.1,  # Thermal BTU to electrical BTU efficiency (~10%)
        'scale_factor': 1000000.0             # Final scaling factor for reward normalization
    }
    
    # Merge with user configuration
    reward_params = {**defaults, **economic_params}
    
    return reward_params


def update_config_with_dashboard(config, rl_config):
    """
    Update the main config object with values from dashboard configuration
    
    Args:
        config: Main Config object from config.yaml (must have rl_model section for RL config)
        rl_config: Dashboard configuration
        
    Returns:
        None: Modifies config in place
    """
    if not rl_config:
        return
    
    # Check if config has rl_model section (RL config) or not (ROM config)
    try:
        # Try to access rl_model to check if it exists
        _ = config.rl_model
        has_rl_model = True
    except (AttributeError, KeyError):
        # ROM config doesn't have rl_model section - skip RL-specific updates
        return
    
    # Update reservoir configuration (only for RL config)
    action_ranges = rl_config.get('action_ranges', {})
    if action_ranges:
        config.rl_model['reservoir']['num_producers'] = action_ranges.get('num_producers', 3)
        config.rl_model['reservoir']['num_injectors'] = action_ranges.get('num_wells', 6) - action_ranges.get('num_producers', 3)
    
    # Store action ranges in ROM training normalization section (for compatibility)
    # Handle new dynamic control structure
    selected_controls = action_ranges.get('controls', {})
    
    if 'rom_training_normalization' not in config.rl_model:
        config.rl_model['rom_training_normalization'] = {}
    
    # Store each control variable's parameters
    for var_name, well_ranges in selected_controls.items():
        if well_ranges:
            var_mins = [ranges['min'] for ranges in well_ranges.values()]
            var_maxs = [ranges['max'] for ranges in well_ranges.values()]
            config.rl_model['rom_training_normalization'][f'{var_name.lower()}_params'] = {
                'min': min(var_mins) if var_mins else 0.0,
                'max': max(var_maxs) if var_maxs else 1000.0
            }
    
    # Legacy support: Also handle old format
    if 'bhp' in action_ranges:
        bhp_ranges = action_ranges['bhp']
        if bhp_ranges:
            bhp_mins = [ranges['min'] for ranges in bhp_ranges.values()]
            bhp_maxs = [ranges['max'] for ranges in bhp_ranges.values()]
            config.rl_model['rom_training_normalization']['bhp_params'] = {
                'min': min(bhp_mins),
                'max': max(bhp_maxs)
            }
    
    if 'gas_injection' in action_ranges:
        gas_ranges = action_ranges['gas_injection']
        if gas_ranges:
            gas_mins = [ranges['min'] for ranges in gas_ranges.values()]
            gas_maxs = [ranges['max'] for ranges in gas_ranges.values()]
            # Store in ROM training section instead of action_constraints
            if 'rom_training_normalization' not in config.rl_model:
                config.rl_model['rom_training_normalization'] = {}
            config.rl_model['rom_training_normalization']['gas_injection_params'] = {
                'min': min(gas_mins),
                'max': max(gas_maxs)
            }
    
    # Update economic parameters
    economic_params = rl_config.get('economic_params', {})
    if economic_params:
        # Geothermal project parameters (primary)
        if 'energy_production_revenue' in economic_params:
            config.rl_model['economics']['prices']['energy_production_revenue'] = economic_params['energy_production_revenue']
        
        if 'water_production_cost' in economic_params:
            config.rl_model['economics']['prices']['water_production_cost'] = economic_params['water_production_cost']
        
        if 'water_injection_cost' in economic_params:
            config.rl_model['economics']['prices']['water_injection_cost'] = economic_params['water_injection_cost']
        
        # Conversion factors
        if 'thermal_to_electrical_efficiency' in economic_params:
            config.rl_model['economics']['conversion']['thermal_to_electrical_efficiency'] = economic_params['thermal_to_electrical_efficiency']
        
        if 'days_per_year' in economic_params:
            config.rl_model['economics']['conversion']['days_per_year'] = economic_params['days_per_year']
        
        if 'btu_to_kwh' in economic_params:
            config.rl_model['economics']['conversion']['btu_to_kwh'] = economic_params['btu_to_kwh']
        
        if 'scale_factor' in economic_params:
            config.rl_model['economics']['scale_factor'] = economic_params['scale_factor']
        
        # Legacy parameters (kept for backward compatibility)
        if 'gas_injection_revenue' in economic_params:
            config.rl_model['economics']['prices']['gas_injection_revenue'] = economic_params['gas_injection_revenue']
        
        if 'gas_injection_cost' in economic_params:
            config.rl_model['economics']['prices']['gas_injection_cost'] = economic_params['gas_injection_cost']
        
        if 'water_production_penalty' in economic_params:
            config.rl_model['economics']['prices']['water_production_penalty'] = economic_params['water_production_penalty']
        
        if 'gas_production_penalty' in economic_params:
            config.rl_model['economics']['prices']['gas_production_penalty'] = economic_params['gas_production_penalty']
        
        if 'lf3_to_ton_conversion' in economic_params:
            # Split conversion factor back into components
            total_conversion = economic_params['lf3_to_ton_conversion']
            config.rl_model['economics']['conversion']['lf3_to_intermediate'] = 0.1167
            config.rl_model['economics']['conversion']['intermediate_to_ton'] = total_conversion / 0.1167
    
    # Update RL hyperparameters
    rl_hyperparams = rl_config.get('rl_hyperparams', {})
    if rl_hyperparams:
        # Network parameters
        if 'hidden_dim' in rl_hyperparams:
            config.rl_model['networks']['hidden_dim'] = rl_hyperparams['hidden_dim']
        
        if 'policy_type' in rl_hyperparams:
            config.rl_model['networks']['policy']['type'] = rl_hyperparams['policy_type']
        
        if 'output_activation' in rl_hyperparams:
            config.rl_model['networks']['policy']['output_activation'] = rl_hyperparams['output_activation']
        
        # SAC parameters
        if 'discount_factor' in rl_hyperparams:
            config.rl_model['sac']['discount_factor'] = rl_hyperparams['discount_factor']
        
        if 'soft_update_tau' in rl_hyperparams:
            config.rl_model['sac']['soft_update_tau'] = rl_hyperparams['soft_update_tau']
        
        if 'entropy_alpha' in rl_hyperparams:
            config.rl_model['sac']['entropy']['alpha'] = rl_hyperparams['entropy_alpha']
        
        if 'critic_lr' in rl_hyperparams:
            config.rl_model['sac']['learning_rates']['critic'] = rl_hyperparams['critic_lr']
        
        if 'policy_lr' in rl_hyperparams:
            config.rl_model['sac']['learning_rates']['policy'] = rl_hyperparams['policy_lr']
        
        if 'gradient_clipping' in rl_hyperparams:
            config.rl_model['sac']['gradient_clipping']['enable'] = rl_hyperparams['gradient_clipping']
        
        if 'max_norm' in rl_hyperparams:
            config.rl_model['sac']['gradient_clipping']['policy_max_norm'] = rl_hyperparams['max_norm']
        
        # Training parameters
        if 'max_episodes' in rl_hyperparams:
            config.rl_model['training']['max_episodes'] = rl_hyperparams['max_episodes']
        
        if 'max_steps_per_episode' in rl_hyperparams:
            config.rl_model['training']['max_steps_per_episode'] = rl_hyperparams['max_steps_per_episode']
            config.rl_model['environment']['max_episode_steps'] = rl_hyperparams['max_steps_per_episode']
        
        # Environment prediction mode
        if 'prediction_mode' in rl_hyperparams:
            config.rl_model['environment']['prediction_mode'] = rl_hyperparams['prediction_mode']
        
        if 'batch_size' in rl_hyperparams:
            config.rl_model['replay_memory']['batch_size'] = rl_hyperparams['batch_size']
        
        if 'replay_capacity' in rl_hyperparams:
            config.rl_model['replay_memory']['capacity'] = rl_hyperparams['replay_capacity']
    
    print("✅ Config updated with dashboard values!")


def create_rl_reward_function(rl_config):
    """
    Create a reward function based on dashboard configuration
    
    Args:
        rl_config: configuration from dashboard
        
    Returns:
        function: configured reward function
    """
    reward_params = get_reward_function_params(rl_config)
    
    def reward_function(yobs, action, num_prod, num_inj):
        """
        Configured geothermal reward function based on dashboard parameters
        
        CORRECT Observation order:
        - BHP: indices [0,1,2] - INJECTOR Bottom-Hole Pressure (psi, wells 0,1,2)
        - ENERGYRATE: indices [3,4,5] - Energy Production Rate (BTU/day, producers wells 3,4,5)
        - WATRATRC: indices [6,7,8] - Water PRODUCTION Rate (bbl/day, producers wells 3,4,5)
        
        CORRECT Action/Control order:
        - BHP: indices [0,1,2] - Producer Bottom-Hole Pressure (psi, producers)
        - WATRATRC: indices [3,4,5] - Water Injection Rate (bbl/day, injectors)
        
        Args:
            yobs: observations in CORRECT order [BHP_inj(3), ENERGYRATE(3), WATRATRC_prod(3)]
            action: actions in CORRECT order [BHP(3), WATRATRC_injection(3)]  
            num_prod: number of producers
            num_inj: number of injectors
            
        Returns:
            torch.Tensor: reward value
        """
        import torch
        
        # Extract parameters
        energy_production_revenue = reward_params['energy_production_revenue']  # $/kWh
        water_production_cost = reward_params.get('water_production_cost', reward_params.get('water_production_reward', 5.0))  # $/bbl
        water_injection_cost = reward_params['water_injection_cost']  # $/bbl
        btu_to_kwh = reward_params['btu_to_kwh']  # BTU to kWh conversion
        scale = reward_params['scale_factor']
        
        # Extract observations using CORRECT order: [BHP_inj(0-2), ENERGYRATE(3-5), WATRATRC(6-8)]
        # BHP: indices 0 to num_inj (injectors)
        # ENERGYRATE: indices num_inj to num_inj+num_prod (producers)
        # WATRATRC: indices num_inj+num_prod to num_inj+num_prod*2 (producers)
        
        # Energy production (ENERGYRATE observations, indices 3-5, producers)
        energy_production_btu_day = torch.sum(yobs[:, num_inj:num_inj+num_prod], dim=1)
        energy_production_kwh_day = energy_production_btu_day * btu_to_kwh
        
        # Water PRODUCTION (WATRATRC observations, indices 6-8, producers) - already in bbl/day
        water_production_bbl_day = torch.sum(yobs[:, num_inj+num_prod:num_inj+num_prod*2], dim=1)
        
        # Extract actions using CORRECT order: [BHP(0-2), WATRATRC(3-5)]
        # Water injection (WATRATRC control, indices 3-5, injectors) - already in bbl/day
        water_injection_bbl_day = torch.sum(action[:, num_prod:num_prod+num_inj], dim=1)
        
        # Calculate geothermal reward:
        # Reward = (Energy_production_kWh * $0.11/kWh) - (Water_production_bbl * $5/bbl) - (Water_injection_bbl * $10/bbl)
        # Note: Water production is a COST (disposal/handling), not a reward
        reward = (energy_production_revenue * energy_production_kwh_day - 
                  water_production_cost * water_production_bbl_day - 
                  water_injection_cost * water_injection_bbl_day) / scale
        
        return reward
    
    return reward_function


def print_dashboard_summary():
    """Print a summary of the dashboard configuration"""
    config = get_rl_config()
    if not config:
        print("❌ No configuration found")
        return
    
    # States
    selected_states = config.get('selected_states', [])
    print(f"Configuration: {len(selected_states)} states selected")
    
    # ROM
    if 'selected_rom' in config and config['selected_rom']:
        rom = config['selected_rom']
        print(f"ROM: {rom['name']}")
    
    # Actions
    action_params = get_action_scaling_params(config)
    print(f"Wells: {action_params['num_producers']} producers, {action_params['num_injectors']} injectors")

