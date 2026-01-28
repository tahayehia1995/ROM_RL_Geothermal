import numpy as np

def normalize_dataset_inplace(data, dataset_name, normalization_type='minmax', per_layer=False):
    """
    Apply normalization to a dataset and return normalization parameters
    
    Args:
        data: Input data array
        dataset_name: Name of the dataset for logging
        normalization_type: 'minmax', 'log', or 'none' normalization
        per_layer: If True and data is 5D spatial data, normalize each layer separately
    """
    if data.size == 0:
        return data, {'type': normalization_type, 'min': 0.0, 'max': 1.0}
    
    # Determine normalization strategy based on dataset structure
    shape = data.shape
    
    # Handle per-layer normalization for spatial data
    if per_layer and len(shape) == 5:  # Spatial data: (cases, timesteps, Nx, Ny, Nz)
        Nz = shape[4]
        normalized_data = np.zeros_like(data)
        layers_params = {}
        
        print(f"    📊 Normalizing {dataset_name} per-layer ({Nz} layers)...")
        
        for z_idx in range(Nz):
            layer_data = data[:, :, :, :, z_idx]
            
            if normalization_type == 'none':
                normalized_layer = layer_data.copy()
                layers_params[f'layer_{z_idx}'] = {
                    'type': 'none',
                    'min': float(np.min(layer_data)),
                    'max': float(np.max(layer_data))
                }
            elif normalization_type == 'log':
                # Log normalization for this layer
                epsilon = 1e-8
                data_shifted = layer_data + epsilon
                
                if np.any(layer_data < 0):
                    data_shifted = layer_data - np.min(layer_data) + epsilon
                    data_shift = float(np.min(layer_data))
                else:
                    data_shift = 0.0
                
                log_data = np.log(data_shifted)
                log_min = float(np.min(log_data))
                log_max = float(np.max(log_data))
                
                if log_max == log_min:
                    normalized_layer = np.zeros_like(layer_data)
                else:
                    normalized_layer = (log_data - log_min) / (log_max - log_min)
                
                layers_params[f'layer_{z_idx}'] = {
                    'type': 'log',
                    'log_min': log_min,
                    'log_max': log_max,
                    'epsilon': epsilon,
                    'data_shift': data_shift
                }
            else:  # minmax normalization
                positive_data = layer_data[layer_data > 0]
                if len(positive_data) > 0:
                    data_min = float(np.min(positive_data))
                else:
                    data_min = float(np.min(layer_data))
                
                data_max = float(np.max(layer_data))
                
                if data_max == data_min:
                    normalized_layer = np.zeros_like(layer_data)
                else:
                    normalized_layer = (layer_data - data_min) / (data_max - data_min)
                
                layers_params[f'layer_{z_idx}'] = {
                    'type': 'minmax',
                    'min': data_min,
                    'max': data_max
                }
            
            normalized_data[:, :, :, :, z_idx] = normalized_layer
        
        norm_params = {
            'type': normalization_type,
            'per_layer': True,
            'layers': layers_params
        }
        
        print(f"    ✅ Per-layer normalization complete for {dataset_name}")
        return normalized_data, norm_params
    
    # Handle 'none' normalization type (keep original values) - global
    if normalization_type == 'none':
        pass  # Keeping original values
        norm_params = {
            'type': 'none',
            'min': np.min(data),
            'max': np.max(data)
        }
        print(f"    📊 Original range: [{norm_params['min']:.6f}, {norm_params['max']:.6f}]")
        return data.copy(), norm_params
    
    if len(shape) == 5:  # Spatial data: (cases, timesteps, x, y, z)
        pass  # Spatial data structure
    elif len(shape) == 3:  # Time series data: (cases, timesteps, wells)
        pass  # Time series data structure
    else:
        pass  # Unknown structure
    
    if normalization_type == 'log':
        pass  # Log normalization
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        data_shifted = data + epsilon
        
        # Check for negative values
        if np.any(data < 0):
            print(f"    ⚠️ Negative values detected. Shifting data to positive range...")
            data_shifted = data - np.min(data) + epsilon
        
        # Apply log transformation
        log_data = np.log(data_shifted)
        
        # Then apply min-max normalization to log-transformed data
        log_min = np.min(log_data)
        log_max = np.max(log_data)
        
        if log_max == log_min:
            print(f"    ⚠️ All log values are the same ({log_min}), returning zeros")
            return np.zeros_like(data), {
                'type': 'log',
                'min': log_min,
                'max': log_max,
                'epsilon': epsilon,
                'data_shift': np.min(data) if np.any(data < 0) else 0
            }
        
        normalized = (log_data - log_min) / (log_max - log_min)
        
        norm_params = {
            'type': 'log',
            'log_min': log_min,
            'log_max': log_max,
            'epsilon': epsilon,
            'data_shift': np.min(data) if np.any(data < 0) else 0
        }
        
        pass  # Log range applied
        
    else:  # minmax normalization (default)
        pass  # Min-max normalization
        # Use minimum positive value instead of absolute minimum
        positive_data = data[data > 0]
        if len(positive_data) > 0:
            data_min = np.min(positive_data)  # Minimum positive value
        else:
            data_min = np.min(data)  # Fallback to absolute minimum if no positive values
        
        data_max = np.max(data)
        
        # Avoid division by zero
        if data_max == data_min:
            print(f"    ⚠️ All values are the same ({data_min}), returning zeros")
            return np.zeros_like(data), {'type': 'minmax', 'min': data_min, 'max': data_max}
        
        # Apply min-max normalization
        normalized = (data - data_min) / (data_max - data_min)
        
        norm_params = {
            'type': 'minmax',
            'min': data_min,
            'max': data_max
        }
        
        pass  # Min-max range applied
    
    return normalized, norm_params

def denormalize_data(normalized_data, norm_params, layer_idx=None):
    """
    Denormalize data using stored normalization parameters
    
    Args:
        normalized_data: Normalized data array
        norm_params: Normalization parameters dictionary
        layer_idx: Layer index for per-layer denormalization (required if per_layer=True)
    """
    # Check if per-layer normalization was used
    if norm_params.get('per_layer', False):
        if layer_idx is None:
            raise ValueError("layer_idx is required for per-layer denormalization")
        
        layer_key = f'layer_{layer_idx}'
        if layer_key not in norm_params['layers']:
            raise ValueError(f"Layer {layer_idx} parameters not found in norm_params")
        
        layer_params = norm_params['layers'][layer_key]
        norm_type = layer_params.get('type', 'minmax')
        
        # Handle per-layer denormalization
        if norm_type == 'none':
            return normalized_data
        elif norm_type == 'log':
            # Reverse log normalization for this layer
            log_min = layer_params['log_min']
            log_max = layer_params['log_max']
            epsilon = layer_params.get('epsilon', 1e-8)
            data_shift = layer_params.get('data_shift', 0)
            
            # CRITICAL FIX: Handle case where log_max == log_min (constant layer in training data)
            # When training data was constant, normalization sets all values to 0
            # But model predictions may have variation, which we need to preserve
            if abs(log_max - log_min) < 1e-10:  # Effectively zero range
                # Training data was constant, so original value was: exp(log_min) - epsilon + data_shift
                constant_value = np.exp(log_min) - epsilon + data_shift
                
                # Check if normalized data has variation (model predictions differ from training constant)
                data_min = np.nanmin(normalized_data)
                data_max = np.nanmax(normalized_data)
                has_variation = (data_max - data_min) > 1e-6
                
                if has_variation:
                    # Model predicted variation, but training data was constant
                    # Map normalized variation to a small percentage range around the constant value
                    # Use ±5% of constant value as the denormalization range
                    # This preserves model variation while keeping it physically reasonable
                    percent_range = 0.05  # 5% variation
                    if constant_value > 0:
                        log_range = np.log(constant_value * (1 + percent_range)) - np.log(constant_value * (1 - percent_range))
                        log_data = normalized_data * log_range + np.log(constant_value * (1 - percent_range))
                    else:
                        # Fallback: use small epsilon range if constant value is zero or negative
                        log_range_epsilon = 1e-6
                        log_data = normalized_data * log_range_epsilon + (log_min - log_range_epsilon / 2)
                else:
                    # Normalized data is also constant (matches training), return constant value
                    log_data = np.full_like(normalized_data, log_min)
            else:
                # Normal case: log_max != log_min
                log_data = normalized_data * (log_max - log_min) + log_min
            
            # Reverse log transformation
            data_shifted = np.exp(log_data)
            
            # Use expm1 for better precision if needed
            log_epsilon = np.log(epsilon)
            close_to_epsilon_mask = np.abs(data_shifted - epsilon) < epsilon * 10.0
            
            if np.any(close_to_epsilon_mask):
                log_diff = log_data - log_epsilon
                denormalized = np.where(
                    close_to_epsilon_mask,
                    np.expm1(log_diff) * epsilon + data_shift,
                    data_shifted - epsilon + data_shift
                )
            else:
                denormalized = data_shifted - epsilon + data_shift
            
            return denormalized
        else:  # minmax
            # Reverse min-max normalization for this layer
            data_min = layer_params['min']
            data_max = layer_params['max']
            denormalized = normalized_data * (data_max - data_min) + data_min
            return denormalized
    
    # Global normalization (backward compatibility)
    norm_type = norm_params.get('type', 'minmax')
    
    if norm_type == 'none':
        return normalized_data
    
    elif norm_type == 'log':
        # Reverse log normalization
        log_min = norm_params['log_min']
        log_max = norm_params['log_max']
        epsilon = norm_params.get('epsilon', 1e-8)
        data_shift = norm_params.get('data_shift', 0)
        
        # CRITICAL FIX: Handle case where log_max == log_min (constant layer in training data)
        # When training data was constant, normalization sets all values to 0
        # But model predictions may have variation, which we need to preserve
        if abs(log_max - log_min) < 1e-10:  # Effectively zero range
            # Training data was constant, so original value was: exp(log_min) - epsilon + data_shift
            constant_value = np.exp(log_min) - epsilon + data_shift
            
            # Check if normalized data has variation (model predictions differ from training constant)
            data_min = np.nanmin(normalized_data)
            data_max = np.nanmax(normalized_data)
            has_variation = (data_max - data_min) > 1e-6
            
            if has_variation:
                # Model predicted variation, but training data was constant
                # Map normalized variation to a small percentage range around the constant value
                # Use ±5% of constant value as the denormalization range
                # This preserves model variation while keeping it physically reasonable
                percent_range = 0.05  # 5% variation
                if constant_value > 0:
                    log_range = np.log(constant_value * (1 + percent_range)) - np.log(constant_value * (1 - percent_range))
                    log_data = normalized_data * log_range + np.log(constant_value * (1 - percent_range))
                else:
                    # Fallback: use small epsilon range if constant value is zero or negative
                    log_range_epsilon = 1e-6
                    log_data = normalized_data * log_range_epsilon + (log_min - log_range_epsilon / 2)
            else:
                # Normalized data is also constant (matches training), return constant value
                log_data = np.full_like(normalized_data, log_min)
        else:
            # Normal case: log_max != log_min
            log_data = normalized_data * (log_max - log_min) + log_min
        
        # Reverse log transformation
        data_shifted = np.exp(log_data)
        
        # CRITICAL FIX: For very small values, exp(log_data) can be very close to epsilon
        # When exp(log_data) ≈ epsilon, the subtraction loses precision
        # Use a more numerically stable calculation by working in log space:
        # Instead of: original_data = exp(log_data) - epsilon + data_shift
        # We use: original_data = expm1(log_data - log(epsilon)) * epsilon + data_shift
        # where expm1(x) = exp(x) - 1, which is more precise for small x
        
        # Check if values are very close to epsilon (within 3 orders of magnitude)
        # If so, use more precise calculation using expm1
        log_epsilon = np.log(epsilon)
        close_to_epsilon_mask = np.abs(data_shifted - epsilon) < epsilon * 10.0
        
        if np.any(close_to_epsilon_mask):
            # Use expm1 for better precision: expm1(x) = exp(x) - 1
            # original_data = exp(log_data) - epsilon + data_shift
            # = exp(log_data - log(epsilon) + log(epsilon)) - epsilon + data_shift
            # = exp(log_data - log_epsilon) * epsilon - epsilon + data_shift
            # = (exp(log_data - log_epsilon) - 1) * epsilon + data_shift
            # = expm1(log_data - log_epsilon) * epsilon + data_shift
            
            # For values close to epsilon, use expm1 for better precision
            log_diff = log_data - log_epsilon
            denormalized = np.where(
                close_to_epsilon_mask,
                np.expm1(log_diff) * epsilon + data_shift,
                data_shifted - epsilon + data_shift
            )
        else:
            # Standard calculation
            denormalized = data_shifted - epsilon + data_shift
        
        return denormalized
    
    else:  # minmax
        # Reverse min-max normalization
        data_min = norm_params['min']
        data_max = norm_params['max']
        
        denormalized = normalized_data * (data_max - data_min) + data_min
        
        return denormalized

