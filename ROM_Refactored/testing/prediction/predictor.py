"""
Test Prediction Functions
Generate predictions and prepare data for visualization
"""

import h5py
import numpy as np
import torch
import os
from testing.visualization.utils import create_visualization_dashboard
from utilities.timing import Timer, collect_prediction_metadata


def generate_test_visualization_standalone(loaded_data, my_rom, device, data_dir, num_tstep=24):
    """
    Standalone function to generate test predictions and launch visualization dashboard.
    Works independently of the dashboard object by using loaded data and selections.
    
    Args:
        loaded_data: Dictionary returned from load_processed_data() containing:
            - norm_params: Normalization parameters
            - data_selections: Data selection metadata (spatial properties, controls, observations, channel names)
            - metadata: Metadata including Nx, Ny, Nz, etc.
        my_rom: Trained ROM model
        device: PyTorch device
        data_dir: Directory containing raw data files
        num_tstep: Number of time steps for prediction
    
    Returns:
        Visualization dashboard object or None if failed
    """
    import h5py
    import os
    import numpy as np
    import torch
    
    if not isinstance(loaded_data, dict):
        raise ValueError(f"loaded_data must be a dictionary, but got {type(loaded_data)}")
    
    norm_params = loaded_data.get('norm_params')
    data_selections = loaded_data.get('data_selections')
    metadata = loaded_data.get('metadata', {})
    
    if norm_params is None:
        raise ValueError("No normalization parameters found in loaded data!")
    
    if data_selections is None:
        raise ValueError("No data selection metadata found in loaded data!")
    
    spatial_properties_to_load = data_selections.get('all_spatial_properties') or data_selections.get('selected_states', {})
    
    if not spatial_properties_to_load:
        raise ValueError("No spatial property configuration found!")
    
    # Extract selections
    selected_controls = data_selections.get('selected_controls', {})
    selected_observations = data_selections.get('selected_observations', {})
    training_channel_names = data_selections.get('training_channel_names', [])
    
    # CRITICAL: Ensure spatial_properties_to_load is ordered according to training_channel_names
    # Dictionary order may not match training_channel_names, so reorder if needed
    if training_channel_names:
        print(f"\n🔍 Verifying spatial_properties_to_load order:")
        print(f"   training_channel_names: {training_channel_names}")
        print(f"   spatial_properties_to_load keys: {list(spatial_properties_to_load.keys())}")
        
        # Check if order matches
        spatial_keys = list(spatial_properties_to_load.keys())
        if spatial_keys != training_channel_names:
            print(f"   ⚠️ WARNING: spatial_properties_to_load order doesn't match training_channel_names!")
            print(f"   🔧 Reordering spatial_properties_to_load to match training_channel_names...")
            
            # Create reordered dictionary
            reordered_spatial_properties = {}
            for var_name in training_channel_names:
                if var_name not in spatial_properties_to_load:
                    raise ValueError(f"CRITICAL ERROR: Variable '{var_name}' from training_channel_names not found in spatial_properties_to_load!")
                reordered_spatial_properties[var_name] = spatial_properties_to_load[var_name]
                print(f"      Position {len(reordered_spatial_properties)-1}: {var_name} -> {reordered_spatial_properties[var_name]}")
            
            spatial_properties_to_load = reordered_spatial_properties
            print(f"   ✅ Reordered successfully")
        else:
            print(f"   ✅ spatial_properties_to_load order matches training_channel_names")
    
    # Extract metadata
    Nx = metadata.get('Nx', 0)
    Ny = metadata.get('Ny', 0)
    Nz = metadata.get('Nz', 0)
    
    # Load ground truth data DIRECTLY from files using exact mapping saved during preprocessing
    # NO statistical matching, NO fallbacks - just direct file loading in training_channel_names order
    # Keep BOTH normalized data (for model input) and raw data (for ground truth visualization)
    test_spatial_data = {}
    raw_spatial_data = {}
    
    if not training_channel_names:
        raise ValueError("training_channel_names is required but not found in data_selections!")
    
    if not spatial_properties_to_load:
        raise ValueError("spatial_properties_to_load is required but not found in data_selections!")
    
    print(f"\n📋 Loading ground truth spatial data files DIRECTLY (in training channel order):")
    print(f"   Training channel order: {training_channel_names}")
    print(f"   Using EXACT file mapping from preprocessing (no matching, no fallbacks)")
    
    # Load each channel in training_channel_names order using direct file mapping
    for channel_idx, var_name in enumerate(training_channel_names):
        print(f"\n   Channel {channel_idx}: {var_name}")
        
        # CRITICAL: Must have file mapping for this variable
        if var_name not in spatial_properties_to_load:
            raise ValueError(f"CRITICAL ERROR: Variable '{var_name}' (channel {channel_idx}) not found in spatial_properties_to_load mapping! "
                           f"Available variables: {list(spatial_properties_to_load.keys())}")
        
        # Get filename from saved mapping
        filename = spatial_properties_to_load[var_name]
        filepath = os.path.join(data_dir, filename)
        
        # CRITICAL: File must exist
        if not os.path.exists(filepath):
            raise ValueError(f"CRITICAL ERROR: File not found for channel {channel_idx} ({var_name})! "
                           f"Expected file: {filepath}")
        
        print(f"      📁 Loading file: {filename}")
        
        # Load raw data from file
        with h5py.File(filepath, 'r') as hf:
            raw_data = np.array(hf['data'])
        
        # CRITICAL VALIDATION: Verify raw data range matches expected variable type
        # This catches cases where wrong file is loaded (e.g., TEMP file loaded for PERMI)
        sample_slice = raw_data[0, 0, :, :, :] if len(raw_data.shape) == 5 else raw_data[0, 0, 0, :, :, :]
        raw_min = float(np.nanmin(sample_slice))
        raw_max = float(np.nanmax(sample_slice))
        raw_mean = float(np.nanmean(sample_slice))
        
        # Expected ranges for each variable (based on typical geothermal reservoir values)
        # Note: Units are defined in config.yaml data_preprocessing.normalization.spatial_datasets
        expected_ranges = {
            'PRES': (1000, 5000),      # Pressure (units defined in config)
            'PERMI': (1e-15, 1e-12),   # Permeability (units defined in config)
            'TEMP': (200, 600),        # Temperature (units defined in config)
            'VPOROSGEO': (0.0, 1.0),   # Porosity (fraction)
            'VPOROSTGEO': (0.0, 1.0),  # Porosity (fraction)
        }
        
        # Check if raw data matches expected range for this variable
        if var_name.upper() in expected_ranges:
            exp_min, exp_max = expected_ranges[var_name.upper()]
            if raw_mean < exp_min or raw_mean > exp_max:
                print(f"      ⚠️ WARNING: Raw data range doesn't match expected range for {var_name}!")
                print(f"         Raw data: min={raw_min:.6e}, max={raw_max:.6e}, mean={raw_mean:.6e}")
                print(f"         Expected range: [{exp_min}, {exp_max}]")
                print(f"         File: {filename}")
                print(f"         This might indicate wrong file was loaded!")
                
                # Try to identify which variable this data actually matches
                for check_var, (check_min, check_max) in expected_ranges.items():
                    if raw_mean >= check_min and raw_mean <= check_max:
                        print(f"         ⚠️ Data appears to match {check_var} range instead!")
                        raise ValueError(f"CRITICAL ERROR: File {filename} contains {check_var} data, but was loaded for {var_name} channel!")
        
        print(f"      ✅ Raw data validation: min={raw_min:.6e}, max={raw_max:.6e}, mean={raw_mean:.6e}")
        
        # CRITICAL: Must have normalization parameters for this variable
        if var_name not in norm_params:
            raise ValueError(f"CRITICAL ERROR: No normalization parameters found for '{var_name}' (channel {channel_idx})! "
                           f"Available norm_params keys: {list(norm_params.keys())}")
        
        # Use normalize_dataset_inplace to handle both per-layer and global normalization
        from data_preprocessing.normalization import normalize_dataset_inplace
        
        norm_params_var = norm_params[var_name]
        norm_type = norm_params_var.get('type', 'minmax')
        
        # Check if per-layer normalization was used
        per_layer = norm_params_var.get('per_layer', False)
        
        # Normalize using the same function as preprocessing (handles per-layer automatically)
        normalized_data, _ = normalize_dataset_inplace(raw_data, var_name, norm_type, per_layer=per_layer)
        
        # Store raw and normalized data
        raw_spatial_data[var_name] = torch.tensor(raw_data, dtype=torch.float32)
        test_spatial_data[var_name] = torch.tensor(normalized_data, dtype=torch.float32)
        # Log raw data range for sanity checking
        sample_raw = raw_data[0, 0]
        print(f"      ✅ Loaded raw+normalized {var_name} from {filename} (shape: {raw_data.shape})")
        if per_layer:
            print(f"         Using per-layer normalization ({raw_data.shape[4]} layers)")
        print(f"         Raw sample range: min={float(np.nanmin(sample_raw)):.6e}, max={float(np.nanmax(sample_raw)):.6e}")
    
    # CRITICAL: Verify that test_spatial_data keys match training_channel_names order
    # Dictionary insertion order should match since we iterate over training_channel_names, but verify explicitly
    test_spatial_data_keys = list(test_spatial_data.keys())
    print(f"\n🔍 Verifying test_spatial_data channel order and file mappings:")
    print(f"   Expected order (training_channel_names): {training_channel_names}")
    print(f"   Actual order (test_spatial_data keys): {test_spatial_data_keys}")
    
    # CRITICAL: Verify each channel loaded the correct file
    print(f"\n   📋 File mapping verification:")
    for channel_idx, var_name in enumerate(training_channel_names):
        if var_name in test_spatial_data:
            expected_file = spatial_properties_to_load.get(var_name, 'UNKNOWN')
            print(f"      Channel {channel_idx} ({var_name}): {expected_file}")
        else:
            print(f"      ❌ Channel {channel_idx} ({var_name}): NOT LOADED!")
    
    if test_spatial_data_keys != training_channel_names:
        print(f"   ⚠️ WARNING: test_spatial_data key order doesn't match training_channel_names!")
        print(f"   🔧 Reordering test_spatial_data to match training_channel_names...")
        
        # Reorder test_spatial_data dictionary to match training_channel_names EXACTLY
        reordered_test_spatial_data = {}
        for idx, var_name in enumerate(training_channel_names):
            if var_name not in test_spatial_data:
                raise ValueError(f"CRITICAL ERROR: Channel '{var_name}' (index {idx}) from training_channel_names not found in test_spatial_data!")
            reordered_test_spatial_data[var_name] = test_spatial_data[var_name]
            print(f"      Position {idx}: {var_name} ✅")
        
        test_spatial_data = reordered_test_spatial_data
        print(f"   ✅ Reordered successfully")
        
        # Verify reordering worked
        final_keys = list(test_spatial_data.keys())
        if final_keys != training_channel_names:
            raise ValueError(f"CRITICAL ERROR: Reordering failed! Final order: {final_keys}, Expected: {training_channel_names}")
        print(f"   ✅ Verification passed: test_spatial_data now matches training_channel_names order")
    else:
        print(f"   ✅ Channel order matches correctly - no reordering needed")
    
    # Load timeseries data with SAME normalization as training
    test_timeseries_data = {}
    all_timeseries_vars = set(list(selected_controls.keys()) + list(selected_observations.keys()))
    
    for var_name in all_timeseries_vars:
        filename = f"batch_timeseries_data_{var_name}.h5"
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            continue
        
        with h5py.File(filepath, 'r') as hf:
            raw_data = np.array(hf['data'])
        
        if var_name not in norm_params:
            continue
            
        norm_params_var = norm_params[var_name]
        normalized_data = (raw_data - norm_params_var['min']) / (norm_params_var['max'] - norm_params_var['min'])
        test_timeseries_data[var_name] = torch.tensor(normalized_data, dtype=torch.float32)
    
    # Extract controls and observations
    control_components = []
    for var_name, config in selected_controls.items():
        if var_name not in test_timeseries_data:
            continue
        data = test_timeseries_data[var_name]
        selected_data = data[:, :, config['wells']]
        
        for well_idx in range(selected_data.shape[2]):
            control_components.append(selected_data[:, :, well_idx])
    
    bhp_test = torch.stack(control_components, dim=2) if control_components else torch.zeros(0)
    
    # Create observation tensor using SAME logic as training  
    obs_components = []
    for var_name, config in selected_observations.items():
        if var_name not in test_timeseries_data:
            continue
        data = test_timeseries_data[var_name]
        selected_data = data[:, :, config['wells']]
        
        for well_idx in range(selected_data.shape[2]):
            obs_components.append(selected_data[:, :, well_idx])
    
    yobs_test = torch.stack(obs_components, dim=2) if obs_components else torch.zeros(0)
    
    # Organize spatial data in training channel order
    # CRITICAL: Must use training_channel_names order explicitly to ensure consistency
    if not training_channel_names:
        raise ValueError("training_channel_names is required but not found!")
    
    # Use training_channel_names directly (not a copy) to ensure exact order match
    channel_names = list(training_channel_names)  # Convert to list to ensure order preservation
    spatial_channels = []
    
    print(f"\n📋 Building state_test tensor with channels in training order:")
    for channel_idx, var_name in enumerate(channel_names):
        if var_name not in test_spatial_data:
            raise ValueError(f"Training channel '{var_name}' (index {channel_idx}) not found in test_spatial_data!")
        spatial_channels.append(test_spatial_data[var_name])
        print(f"   Channel {channel_idx}: {var_name}")
    
    # Stack into state tensor: (n_sample, timesteps, channels, Nx, Ny, Nz)
    if not spatial_channels:
        raise ValueError("No spatial data available")
    
    state_test = torch.stack(spatial_channels, dim=2)
    n_sample, timesteps, n_channels, Nx, Ny, Nz = state_test.shape
    
    if n_channels != len(channel_names):
        raise ValueError(f"Channel count mismatch: tensor has {n_channels} channels, expected {len(channel_names)}")
    
    # Verify channel order consistency
    print(f"✅ state_test tensor created with {n_channels} channels in order: {channel_names}")
    
    available_cases = n_sample
    test_case_indices = np.arange(available_cases)
    num_case = len(test_case_indices)
    
    # Initialize prediction arrays
    state_pred = torch.zeros((num_case, num_tstep, n_channels, Nx, Ny, Nz), dtype=torch.float32).to(device)
    yobs_pred = torch.zeros((num_case, num_tstep, yobs_test.shape[2]), dtype=torch.float32).to(device)
    
    # Time step configuration
    t_steps = np.arange(0, 200, 200//num_tstep)
    dt = 10
    t_steps1 = (t_steps + dt).astype(int)
    indt_del = t_steps1 - t_steps
    indt_del = indt_del / max(indt_del)
    
    tmp = np.array(range(num_tstep)) - 1
    tmp1 = np.array(range(num_tstep))
    tmp[0] = 0
    
    # Prepare control and observation sequences
    bhp_tt1 = bhp_test[:, tmp1, :]
    bhp_t = torch.swapaxes(bhp_tt1, 1, 2).to(device)
    bhp_seq = bhp_t[test_case_indices, :, :]
    
    yobs_t_seq = torch.swapaxes(yobs_test[test_case_indices, ...], 1, 2).to(device)
    
    initial_state = state_test[test_case_indices, 0, :, :, :, :].to(device)
    state_t_seq = initial_state
    
    # Time the prediction phase
    with Timer("prediction", log_dir='./timing_logs/') as timer:
        for i_tstep in range(num_tstep):
            # Store current state prediction
            state_pred[:, i_tstep, ...] = state_t_seq
            
            # Time step for current iteration
            dt_seq = torch.tensor(np.ones((num_case, 1)) * indt_del[i_tstep], dtype=torch.float32).to(device)
            
            # Prepare inputs for model
            inputs = (state_t_seq, bhp_seq[:, :, i_tstep], yobs_t_seq[:, :, i_tstep], dt_seq)
            
            # Predict next state
            state_t1_seq, yobs_t1_seq = my_rom.predict(inputs)
            
            # Update state for next iteration
            state_t_seq = state_t1_seq
            
            yobs_pred[:, i_tstep, :] = yobs_t1_seq
        
        # Calculate average time per case (get current elapsed time)
        total_time = timer.get_elapsed()
        avg_time_per_case = total_time / num_case if num_case > 0 else 0.0
        
        # Collect metadata for timing log
        model_info = {}
        if hasattr(my_rom, 'config') and my_rom.config:
            if hasattr(my_rom.config, 'model'):
                model_info['model_method'] = my_rom.config.model.get('method', 'N/A')
                model_info['latent_dimension'] = my_rom.config.model.get('latent_dim', 'N/A')
            else:
                model_info['model_method'] = 'N/A'
                model_info['latent_dimension'] = 'N/A'
        else:
            model_info['model_method'] = 'N/A'
            model_info['latent_dimension'] = 'N/A'
        metadata = collect_prediction_metadata(num_case, num_tstep, str(device), model_info)
        metadata['average_time_per_case'] = avg_time_per_case
        timer.metadata = metadata
    
    # Construct ground truth state sequence in training channel order
    # CRITICAL: Use RAW data directly from .h5 files for ground truth visualization
    print(f"\n📋 Building state_seq_true tensor with channels in training order:")
    print(f"   Final raw_spatial_data keys order: {list(raw_spatial_data.keys())}")
    print(f"   training_channel_names order: {training_channel_names}")
    
    # CRITICAL VALIDATION: Ensure test_spatial_data keys match training_channel_names BEFORE building tensor
    final_test_keys = list(raw_spatial_data.keys())
    if final_test_keys != training_channel_names:
        raise ValueError(f"CRITICAL ERROR: raw_spatial_data keys ({final_test_keys}) != training_channel_names ({training_channel_names}) before building state_seq_true!")
    
    state_seq_true = torch.zeros((num_case, n_channels, timesteps, Nx, Ny, Nz))
    
    for channel_idx in range(len(training_channel_names)):
        var_name = training_channel_names[channel_idx]
        
        if var_name not in raw_spatial_data:
            raise ValueError(f"Channel {channel_idx} ({var_name}) missing from raw_spatial_data!")
        
        # CRITICAL: Use RAW data for ground truth (directly from .h5 files)
        channel_data = raw_spatial_data[var_name][test_case_indices, ...]
        
        # Verify data shape matches expected
        expected_shape = (len(test_case_indices), timesteps, Nx, Ny, Nz)
        if channel_data.shape != expected_shape:
            raise ValueError(f"Channel {channel_idx} ({var_name}) data shape mismatch! Expected {expected_shape}, got {channel_data.shape}")
        
        state_seq_true[:, channel_idx, :, :, :, :] = channel_data
        
        # CRITICAL VALIDATION: Print actual data range to verify correct channel assignment
        sample_channel_slice = channel_data[0, 0, :, :, :].cpu().numpy() if hasattr(channel_data, 'cpu') else channel_data[0, 0, :, :, :]
        ch_min = float(np.nanmin(sample_channel_slice))
        ch_max = float(np.nanmax(sample_channel_slice))
        ch_mean = float(np.nanmean(sample_channel_slice))
        print(f"   ✅ Channel {channel_idx}: {var_name} -> state_seq_true[:, {channel_idx}, ...]")
        print(f"      Data range in tensor: min={ch_min:.6e}, max={ch_max:.6e}, mean={ch_mean:.6e}")
        
        # Verify this matches what we loaded
        if var_name in raw_spatial_data:
            raw_sample = raw_spatial_data[var_name][0, 0, :, :, :].cpu().numpy() if hasattr(raw_spatial_data[var_name], 'cpu') else raw_spatial_data[var_name][0, 0, :, :, :]
            raw_ch_min = float(np.nanmin(raw_sample))
            raw_ch_max = float(np.nanmax(raw_sample))
            if abs(ch_min - raw_ch_min) > 1e-10 or abs(ch_max - raw_ch_max) > 1e-10:
                print(f"      ⚠️ WARNING: Tensor data doesn't match raw data!")
                print(f"         Tensor: [{ch_min:.6e}, {ch_max:.6e}]")
                print(f"         Raw: [{raw_ch_min:.6e}, {raw_ch_max:.6e}]")
    
    # Align time dimensions
    state_seq_true_aligned = state_seq_true[:, :, :num_tstep, :, :, :]
    
    # CRITICAL FINAL VALIDATION: Verify each channel contains the correct data
    print(f"\n🔍 FINAL VALIDATION: Verifying data in each channel position of state_seq_true_aligned:")
    expected_ranges = {
        'PRES': (1000, 5000),
        'PERMI': (1e-15, 1e-12),
        'TEMP': (200, 600),
        'VPOROSGEO': (0.0, 1.0),
        'VPOROSTGEO': (0.0, 1.0),
    }
    for ch_idx in range(n_channels):
        var_name = training_channel_names[ch_idx]
        ch_data = state_seq_true_aligned[0, ch_idx, 0, :, :, :].cpu().numpy()
        ch_min = float(np.nanmin(ch_data))
        ch_max = float(np.nanmax(ch_data))
        ch_mean = float(np.nanmean(ch_data))
        print(f"   Channel {ch_idx} ({var_name}): min={ch_min:.6e}, max={ch_max:.6e}, mean={ch_mean:.6e}")
        
        # Verify data matches expected range
        if var_name.upper() in expected_ranges:
            exp_min, exp_max = expected_ranges[var_name.upper()]
            if ch_mean < exp_min or ch_mean > exp_max:
                print(f"      ❌ ERROR: Channel {ch_idx} data doesn't match {var_name} range!")
                # Check which variable it actually matches
                for check_var, (check_min, check_max) in expected_ranges.items():
                    if ch_mean >= check_min and ch_mean <= check_max:
                        print(f"      ⚠️ Data appears to be {check_var} instead of {var_name}!")
                        raise ValueError(f"CRITICAL ERROR: Channel {ch_idx} labeled as '{var_name}' but contains {check_var} data! "
                                       f"Mean={ch_mean:.6e}, Expected for {var_name}: [{exp_min}, {exp_max}]")
        else:
            print(f"      ⚠️ No expected range defined for {var_name}")
    
    # Verify channel order consistency between predicted and ground truth
    print(f"\n✅ Channel order verification:")
    print(f"   training_channel_names: {training_channel_names}")
    print(f"   channel_names (for visualization): {channel_names}")
    print(f"   state_pred shape: {state_pred.shape} (channels: {state_pred.shape[2]})")
    print(f"   state_seq_true_aligned shape: {state_seq_true_aligned.shape} (channels: {state_seq_true_aligned.shape[1]})")
    
    # CRITICAL VALIDATION: Ensure channel_names matches training_channel_names exactly
    if channel_names != training_channel_names:
        raise ValueError(f"CRITICAL ERROR: channel_names ({channel_names}) != training_channel_names ({training_channel_names})!")
    
    if len(channel_names) != n_channels:
        raise ValueError(f"CRITICAL ERROR: channel_names length ({len(channel_names)}) != n_channels ({n_channels})!")
    
    train_state_pred = None
    train_state_seq_true_aligned = None
    train_yobs_pred = None
    train_yobs_seq_true = None
    train_case_indices = None
    
    STATE_train = loaded_data.get('STATE_train') if isinstance(loaded_data, dict) else None
    BHP_train = loaded_data.get('BHP_train') if isinstance(loaded_data, dict) else None
    Yobs_train = loaded_data.get('Yobs_train') if isinstance(loaded_data, dict) else None
    num_train = metadata.get('num_train', 0) if isinstance(metadata, dict) else 0
    
    if STATE_train and len(STATE_train) > 0 and num_train > 0:
        num_train_timesteps = len(STATE_train)
        train_state_data = torch.stack(STATE_train, dim=1)
        
        if BHP_train and len(BHP_train) > 0:
            train_bhp_data = torch.stack(BHP_train, dim=1)
        else:
            train_bhp_data = None
            
        if Yobs_train and len(Yobs_train) > 0:
            train_yobs_data = torch.stack(Yobs_train, dim=1)
        else:
            train_yobs_data = None
        
        train_case_indices = np.arange(num_train)
        num_train_case = len(train_case_indices)
        
        train_state_pred = torch.zeros((num_train_case, num_tstep, n_channels, Nx, Ny, Nz), dtype=torch.float32).to(device)
        if train_yobs_data is not None:
            train_yobs_pred = torch.zeros((num_train_case, num_tstep, train_yobs_data.shape[2]), dtype=torch.float32).to(device)
        else:
            train_yobs_pred = torch.zeros((num_train_case, num_tstep, yobs_test.shape[2]), dtype=torch.float32).to(device)
        
        if train_bhp_data is not None:
            train_bhp_tt1 = train_bhp_data[:, tmp1, :]
            train_bhp_t = torch.swapaxes(train_bhp_tt1, 1, 2).to(device)
            train_bhp_seq = train_bhp_t[train_case_indices, :, :]
        else:
            train_bhp_seq = torch.zeros((num_train_case, bhp_seq.shape[1], bhp_seq.shape[2]), dtype=torch.float32).to(device)
        
        if train_yobs_data is not None:
            train_yobs_t_seq = torch.swapaxes(train_yobs_data[train_case_indices, ...], 1, 2).to(device)
        else:
            train_yobs_t_seq = torch.zeros((num_train_case, yobs_t_seq.shape[1], yobs_t_seq.shape[2]), dtype=torch.float32).to(device)
        
        train_initial_state = train_state_data[train_case_indices, 0, :, :, :, :].to(device)
        train_state_t_seq = train_initial_state
        
        with Timer("training_prediction", log_dir='./timing_logs/') as train_timer:
            for i_tstep in range(num_tstep):
                train_state_pred[:, i_tstep, ...] = train_state_t_seq
                dt_seq = torch.tensor(np.ones((num_train_case, 1)) * indt_del[i_tstep], dtype=torch.float32).to(device)
                inputs = (train_state_t_seq, train_bhp_seq[:, :, i_tstep], train_yobs_t_seq[:, :, i_tstep], dt_seq)
                train_state_t1_seq, train_yobs_t1_seq = my_rom.predict(inputs)
                train_state_t_seq = train_state_t1_seq
                train_yobs_pred[:, i_tstep, :] = train_yobs_t1_seq
        
        train_state_seq_true = torch.zeros((num_train_case, n_channels, num_train_timesteps, Nx, Ny, Nz))
        for i in range(n_channels):
            train_state_seq_true[:, i, :, :, :, :] = train_state_data[train_case_indices, :, i, :, :, :]
        
        train_state_seq_true_aligned = train_state_seq_true[:, :, :num_tstep, :, :, :]
    
    # CRITICAL VALIDATION: Ensure channel_names passed to dashboard matches training_channel_names
    if channel_names != training_channel_names:
        raise ValueError(f"CRITICAL ERROR: channel_names ({channel_names}) != training_channel_names ({training_channel_names}) before passing to dashboard!")
    
    print(f"\n🎨 Passing channel_names to visualization dashboard: {channel_names}")
    print(f"   This order MUST match the channel order in state_pred and state_seq_true_aligned tensors")
    
    visualization_dashboard = create_visualization_dashboard(
        state_pred=state_pred,
        state_seq_true_aligned=state_seq_true_aligned,
        yobs_pred=yobs_pred,
        yobs_seq_true=yobs_t_seq,
        test_case_indices=test_case_indices,
        norm_params=norm_params,
        Nx=Nx, Ny=Ny, Nz=Nz,
        num_tstep=num_tstep,
        channel_names=channel_names,  # Pass the channel names for visualization (must match tensor channel order)
        my_rom=my_rom,  # Pass ROM model for comparison predictions
        test_controls=bhp_seq,  # Pass test controls for comparison predictions
        test_observations=yobs_t_seq,  # Pass test observations for comparison predictions
        device=device,  # Pass device for computation
        train_state_pred=train_state_pred,  # Pass training predictions
        train_state_seq_true_aligned=train_state_seq_true_aligned,
        train_yobs_pred=train_yobs_pred,
        train_yobs_seq_true=train_yobs_t_seq if train_state_pred is not None else None,
        train_case_indices=train_case_indices,
        loaded_data=loaded_data,
        true_data_is_raw=True
    )
    
    return visualization_dashboard


