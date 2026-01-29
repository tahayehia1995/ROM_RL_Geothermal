"""
Training Orchestrator for RL Training
Manages action variation and episode tracking
"""
import math
import torch
import numpy as np
import glob
import json
import pickle
from pathlib import Path


class ActionVariationManager:
    """Comprehensive action variation management for RL training"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Variation parameters from config
        self.variation_config = config.rl_model.get('action_variation', {})
        
        # Default variation settings if not in config
        self.noise_decay_rate = self.variation_config.get('noise_decay_rate', 0.995)
        self.max_noise_std = self.variation_config.get('max_noise_std', 0.25)
        self.min_noise_std = self.variation_config.get('min_noise_std', 0.01)
        self.step_variation_amplitude = self.variation_config.get('step_variation_amplitude', 0.15)
        
        # Well-specific strategies
        self.well_strategies = {
            'P1': {'variation': 0.15, 'bias': 0.0, 'exploration_scale': 0.8},    # Conservative producer
            'P2': {'variation': 0.20, 'bias': 0.05, 'exploration_scale': 1.0},   # Moderate producer  
            'P3': {'variation': 0.30, 'bias': -0.05, 'exploration_scale': 1.3},  # Aggressive producer
            'I1': {'variation': 0.18, 'bias': 0.02, 'exploration_scale': 0.9},   # Conservative injector
            'I2': {'variation': 0.25, 'bias': 0.0, 'exploration_scale': 1.1},    # Moderate injector
            'I3': {'variation': 0.35, 'bias': 0.08, 'exploration_scale': 1.4}    # Aggressive injector
        }
        
        # Random walk state for temporal correlation
        self.random_walk_offset = None
        
        print(f"🎯 ActionVariationManager initialized:")
        print(f"   Noise decay rate: {self.noise_decay_rate}")
        print(f"   Noise range: [{self.min_noise_std:.3f}, {self.max_noise_std:.3f}]")
        print(f"   Step variation: {self.step_variation_amplitude:.3f}")
    
    def enhance_action_with_variation(self, base_action, episode, step_in_episode, max_steps, variation_mode='adaptive'):
        """
        Main function to add comprehensive action variation
        
        Args:
            base_action: Base action from policy
            episode: Current episode number
            step_in_episode: Current step within episode
            max_steps: Maximum steps per episode
            variation_mode: 'adaptive', 'exploration', 'exploitation', 'minimal'
        """
        
        if variation_mode == 'exploration':
            return self._high_variation_mode(base_action, episode, step_in_episode, max_steps)
        elif variation_mode == 'exploitation':
            return self._medium_variation_mode(base_action, episode, step_in_episode, max_steps)
        elif variation_mode == 'minimal':
            return self._low_variation_mode(base_action, episode, step_in_episode)
        else:  # adaptive
            return self._adaptive_variation_mode(base_action, episode, step_in_episode, max_steps)
    
    def _adaptive_variation_mode(self, action, episode, step, max_steps):
        """Adaptive variation based on training progress"""
        
        # Determine training phase
        max_episodes = self.config.rl_model['training']['max_episodes']
        
        if episode < max_episodes * 0.3:  # First 30% - high exploration
            return self._high_variation_mode(action, episode, step, max_steps)
        elif episode < max_episodes * 0.7:  # Middle 40% - balanced
            return self._medium_variation_mode(action, episode, step, max_steps)
        else:  # Final 30% - fine-tuning
            return self._low_variation_mode(action, episode, step)
    
    def _high_variation_mode(self, action, episode, step, max_steps):
        """High variation for exploration phase"""
        
        # 1. Episode-based noise with slower decay
        noise_std = max(self.min_noise_std, self.max_noise_std * (self.noise_decay_rate ** (episode * 0.5)))
        
        # 2. Strong step-wise variation
        step_factor = 1.0 + self.step_variation_amplitude * math.sin(step * 2 * math.pi / max_steps)
        
        # 3. Well-specific variation
        well_specific_noise = self._apply_well_specific_variation(action, episode, scale_factor=1.5)
        
        # 4. Random walk for temporal correlation
        walk_noise = self._apply_random_walk(action, step_size=0.04)
        
        # Combine all variations
        varied_action = action * step_factor + well_specific_noise + walk_noise
        
        return torch.clamp(varied_action, 0.0, 1.0)
    
    def _medium_variation_mode(self, action, episode, step, max_steps):
        """Medium variation for balanced exploration-exploitation"""
        
        # 1. Moderate noise decay
        noise_std = max(self.min_noise_std, self.max_noise_std * 0.6 * (self.noise_decay_rate ** episode))
        
        # 2. Moderate step variation
        step_factor = 1.0 + (self.step_variation_amplitude * 0.6) * math.sin(step * math.pi / max_steps)
        
        # 3. Well-specific variation with normal scaling
        well_specific_noise = self._apply_well_specific_variation(action, episode, scale_factor=1.0)
        
        # 4. Smaller random walk
        walk_noise = self._apply_random_walk(action, step_size=0.02)
        
        varied_action = action * step_factor + well_specific_noise + walk_noise
        
        return torch.clamp(varied_action, 0.0, 1.0)
    
    def _low_variation_mode(self, action, episode, step):
        """Low variation for fine-tuning phase"""
        
        # 1. Minimal noise
        noise_std = max(self.min_noise_std, self.max_noise_std * 0.2 * (0.999 ** episode))
        gaussian_noise = torch.randn_like(action) * noise_std
        
        # 2. Well-specific variation with reduced scaling
        well_specific_noise = self._apply_well_specific_variation(action, episode, scale_factor=0.5)
        
        varied_action = action + gaussian_noise + well_specific_noise
        
        return torch.clamp(varied_action, 0.0, 1.0)
    
    def _apply_well_specific_variation(self, action, episode, scale_factor=1.0):
        """Apply different variation strategies to different wells"""
        
        well_noise = torch.zeros_like(action)
        # Geothermal control order (from ROM config): [WATRATRC(I1,I2,I3), BHP(P1,P2,P3)]
        wells = ['P1', 'P2', 'P3', 'I1', 'I2', 'I3']  # Order: [BHP(3), Gas(3)]
        
        for i, well in enumerate(wells):
            if i < action.shape[-1]:  # Ensure we don't exceed action dimensions
                strategy = self.well_strategies[well]
                
                # Time-decaying bias
                bias = strategy['bias'] * (0.99 ** episode) * scale_factor
                
                # Well-specific noise
                noise_scale = strategy['variation'] * strategy['exploration_scale'] * scale_factor
                noise = torch.randn(1).item() * noise_scale
                
                well_noise[0, i] = bias + noise
        
        return well_noise
    
    def _apply_random_walk(self, action, step_size=0.03):
        """Apply random walk for temporal correlation between steps"""
        
        if self.random_walk_offset is None:
            self.random_walk_offset = torch.zeros_like(action)
        
        # Random walk step
        walk_step = torch.randn_like(action) * step_size
        self.random_walk_offset = torch.clamp(
            self.random_walk_offset + walk_step, 
            -0.2, 0.2
        )
        
        return self.random_walk_offset
    
    def reset_for_new_episode(self):
        """Reset states for new episode"""
        self.random_walk_offset = None
    
    def get_variation_statistics(self, actions_history):
        """Calculate variation statistics for monitoring"""
        if len(actions_history) < 2:
            return {'mean_variation': 0.0, 'max_variation': 0.0, 'per_well_std': [0.0] * 6}
        
        actions_tensor = torch.stack(actions_history)  # Shape: (steps, batch, actions)
        
        # Calculate statistics
        per_well_std = torch.std(actions_tensor, dim=0).squeeze().tolist()
        mean_variation = torch.std(actions_tensor).item()
        max_variation = torch.max(torch.std(actions_tensor, dim=0)).item()
        
        return {
            'mean_variation': mean_variation,
            'max_variation': max_variation,
            'per_well_std': per_well_std if isinstance(per_well_std, list) else [per_well_std]
        }


class EnhancedTrainingOrchestrator:
    """Orchestrates enhanced RL training with action variation"""
    
    def __init__(self, config, rl_config=None):
        self.config = config
        self.rl_config = rl_config  # Store dashboard configuration
        self.variation_manager = ActionVariationManager(config)
        
        # Training phase configuration
        self.training_phases = self._setup_training_phases()
        
        # Action history for analysis
        self.episode_actions = []
        self.variation_stats = []
        
        # 🔥 NEW: Enhanced episode storage for comprehensive dashboard
        self.max_stored_episodes = config.rl_model.get('dashboard', {}).get('max_stored_episodes', 50)  # Store last 50 episodes
        self.stored_episodes = {}  # Dict with episode_number as key
        self.episode_order = []    # Track episode order for efficient access
        
        # Best episode tracking for scientific visualization (kept for compatibility)
        self.best_episode_data = {
            'episode_number': -1,
            'total_reward': -np.inf,
            'actions': [],           # Physical units (psi, BTU/Day)
            'observations': [],      # Physical units (BTU/Day, bbl/day)
            'rewards': [],           # Step-wise rewards
            'states': [],            # Latent states
            'timesteps': [],         # Time information
            'economic_breakdown': [], # Detailed economic analysis per step
            'spatial_states': []     # 🔥 NEW: Spatial reservoir states for visualization
        }
        
        # Current episode tracking
        self.current_episode_data = {
            'actions': [],
            'observations': [],
            'rewards': [],
            'states': [],
            'timesteps': [],
            'economic_breakdown': [],
            'spatial_states': []  # 🔥 NEW: Spatial states during current episode
        }
        
        # Load observation and control variable definitions from ROM config for consistency
        self._load_observation_control_definitions()
        
        # 🔥 NEW: Training performance tracking for dashboard
        self.training_metrics = {
            'episode_rewards': [],
            'avg_rewards': [],
            'episodes': [],
            'policy_losses': [],  # Store policy loss per episode
            'q_losses': []       # Store Q-value loss per episode
        }
        
        # 🔥 NEW: Spatial data capture configuration
        self.capture_spatial_states = config.rl_model.get('dashboard', {}).get('capture_spatial_states', True)
        self.spatial_capture_frequency = config.rl_model.get('dashboard', {}).get('spatial_capture_frequency', 1)  # Capture every N steps
        
        # Store environment reference for spatial state access
        self.environment = None  # Will be set during training setup
        
    def set_environment(self, environment):
        """Set environment reference for spatial state capture"""
        self.environment = environment
        print(f"🔗 Environment linked to training orchestrator for spatial capture")
        
    def _load_observation_control_definitions(self):
        """Load observation and control variable definitions from ROM config.yaml for consistency"""
        self.obs_variable_map = {}  # Maps observation index to variable name
        self.obs_indices_map = {}  # Maps variable name to observation indices
        self.obs_config = {}  # Store full observation config
        self.control_variable_map = {}  # Maps control index to variable name
        self.control_indices_map = {}  # Maps variable name to control indices
        
        # Try to load from ROM config.yaml
        rom_config_path = getattr(self.config, 'rom_config_path', None)
        if rom_config_path is None:
            # Try default path
            rom_config_path = '../ROM_Refactored/config.yaml'
        
        try:
            import yaml
            import os
            
            # Resolve path relative to current file location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            rom_config_full_path = os.path.normpath(os.path.join(current_dir, '..', '..', 'ROM_Refactored', 'config.yaml'))
            
            if os.path.exists(rom_config_full_path):
                with open(rom_config_full_path, 'r') as f:
                    rom_config = yaml.safe_load(f)
                
                # Load observation definitions
                obs_config = rom_config.get('data', {}).get('observations', {})
                if obs_config and 'variables' in obs_config:
                    self.obs_config = obs_config
                    obs_vars = obs_config['variables']
                    obs_order = obs_config.get('order', ['BHP', 'ENERGYRATE', 'WATRATRC'])
                    
                    for var_name in obs_order:
                        if var_name in obs_vars:
                            var_config = obs_vars[var_name]
                            indices = var_config.get('indices', [])
                            self.obs_indices_map[var_name] = indices
                            for idx in indices:
                                self.obs_variable_map[idx] = var_name
                    
                    print(f"✅ Loaded observation definitions from ROM config: {len(obs_order)} observation types")
                
                # Load control definitions
                control_config = rom_config.get('data', {}).get('controls', {})
                if control_config and 'variables' in control_config:
                    control_vars = control_config['variables']
                    control_order = control_config.get('order', ['BHP'])
                    
                    control_idx = 0
                    for var_name in control_order:
                        if var_name in control_vars:
                            var_config = control_vars[var_name]
                            num_wells = var_config.get('num_wells', 0)
                            well_type = var_config.get('well_type', '')
                            
                            if well_type == 'injectors':
                                indices = list(range(control_idx, control_idx + num_wells))
                            elif well_type == 'producers':
                                indices = list(range(control_idx, control_idx + num_wells))
                            else:
                                indices = list(range(control_idx, control_idx + num_wells))
                            
                            self.control_indices_map[var_name] = indices
                            for idx in indices:
                                self.control_variable_map[idx] = var_name
                            
                            control_idx += num_wells
                    
                    print(f"✅ Loaded control definitions from ROM config: {len(control_order)} control types")
            else:
                print(f"⚠️ ROM config file not found at {rom_config_full_path}, using defaults")
        except Exception as e:
            print(f"⚠️ Could not load observation/control definitions from ROM config: {e}")
            print("   Using default hard-coded mappings")
    
    def _setup_training_phases(self):
        """Setup training phases based on config"""
        max_episodes = self.config.rl_model['training']['max_episodes']
        
        return {
            'exploration': (0, int(max_episodes * 0.3)),
            'balanced': (int(max_episodes * 0.3), int(max_episodes * 0.7)),
            'exploitation': (int(max_episodes * 0.7), max_episodes)
        }
    
    def get_current_training_phase(self, episode):
        """Determine current training phase"""
        for phase, (start, end) in self.training_phases.items():
            if start <= episode < end:
                return phase
        return 'exploitation'  # Default
    
    def select_enhanced_action(self, agent, state, episode, step_in_episode, exploration_steps, total_steps):
        """Enhanced action selection with variation management"""
        
        max_steps = self.config.rl_model['training']['max_steps_per_episode']
        
        # Determine action selection strategy
        if total_steps < exploration_steps:
            # Pure random exploration
            action = self._random_exploration_action()
        else:
            # Policy-based action with variation
            phase = self.get_current_training_phase(episode)
            base_action = agent.select_action(state)
            
            # Apply variation based on training phase
            action = self.variation_manager.enhance_action_with_variation(
                base_action, episode, step_in_episode, max_steps, variation_mode=phase
            )
        
        # Store action for analysis
        self.episode_actions.append(action.clone())
        
        return action
    
    def _random_exploration_action(self):
        """Generate random exploration action"""
        num_producers = self.config.rl_model['reservoir']['num_producers']
        num_injectors = self.config.rl_model['reservoir']['num_injectors']
        
        # Generate random actions using geothermal control order [WATRATRC(3), BHP(3)]
        action = torch.rand(1, num_producers + num_injectors)  # Order: [BHP(3), Gas(3)]
        return action
    
    def start_new_episode(self):
        """Start tracking a new episode"""
        self.current_episode_data = {
            'actions': [],
            'observations': [],
            'rewards': [],
            'states': [],
            'timesteps': [],
            'economic_breakdown': [],
            'spatial_states': []  # 🔥 NEW: Reset spatial states for new episode
        }
    
    def record_step_data(self, step, action, observation, reward, state, economic_breakdown=None):
        """Record data for current step including spatial states"""
        # Convert action to physical units for storage
        physical_action = self._convert_action_to_physical_units(action)
        
        # 🔧 FIX: observation is ALREADY in physical units from Environment.step()
        # No need to denormalize again - just convert to well-specific dictionary format
        physical_observation = self._convert_physical_observation_to_well_dict(observation)
        
        self.current_episode_data['actions'].append(physical_action)
        self.current_episode_data['observations'].append(physical_observation)
        self.current_episode_data['rewards'].append(reward.item() if hasattr(reward, 'item') else reward)
        self.current_episode_data['states'].append(state.detach().cpu().numpy() if hasattr(state, 'detach') else state)
        self.current_episode_data['timesteps'].append(step)
        self.current_episode_data['economic_breakdown'].append(economic_breakdown)
    
        # 🔥 NEW: Capture spatial states if enabled
        if self.capture_spatial_states and step % self.spatial_capture_frequency == 0:
            spatial_state = self._capture_current_spatial_state(step)
            if spatial_state is not None:
                self.current_episode_data['spatial_states'].append({
                    'step': step,
                    'spatial_state': spatial_state
                })
    
    def _capture_current_spatial_state(self, step):
        """Capture current spatial reservoir state from environment"""
        if self.environment is None:
            return None
            
        try:
            # Get current spatial state from environment
            if hasattr(self.environment, 'current_spatial_state') and self.environment.current_spatial_state is not None:
                # Environment maintains spatial state (state-based mode)
                spatial_state = self.environment.current_spatial_state.detach().cpu().numpy()
                return spatial_state
            elif hasattr(self.environment, 'state') and hasattr(self.environment, 'rom'):
                # Decode current latent state to spatial (latent-based mode)
                with torch.no_grad():
                    latent_state = self.environment.state
                    spatial_state = self.environment.rom.model.decoder(latent_state)
                    return spatial_state.detach().cpu().numpy()
            else:
                return None
                
        except Exception as e:
            if step == 0:  # Only warn on first step to avoid spam
                print(f"⚠️ Warning: Could not capture spatial state at step {step}: {e}")
            return None
    
    def record_training_metrics(self, episode, episode_reward, avg_reward, policy_loss=None, q_loss=None):
        """
        🔥 NEW: Record training performance metrics for dashboard visualization
        
        Args:
            episode: Episode number
            episode_reward: Episode reward
            avg_reward: Average reward over recent episodes
            policy_loss: Policy loss for this episode (optional)
            q_loss: Q-value loss for this episode (optional)
        """
        self.training_metrics['episodes'].append(episode)
        self.training_metrics['episode_rewards'].append(episode_reward)
        self.training_metrics['avg_rewards'].append(avg_reward)
        
        # Store losses if provided
        if policy_loss is not None:
            self.training_metrics['policy_losses'].append(policy_loss)
        else:
            self.training_metrics['policy_losses'].append(None)
            
        if q_loss is not None:
            self.training_metrics['q_losses'].append(q_loss)
        else:
            self.training_metrics['q_losses'].append(None)
    
    def get_training_metrics(self):
        """
        🔥 NEW: Get training performance metrics for dashboard
        
        Returns:
            dict: Training metrics for visualization
        """
        return self.training_metrics.copy()
    
    def _convert_physical_observation_to_well_dict(self, observation):
        """
        🔧 NEW: Convert already-physical observations to well dictionary format
        
        This function handles observations that are ALREADY in physical units (no denormalization needed)
        and just maps them to well-specific keys for visualization.
        
        Args:
            observation: Physical observation tensor (already denormalized)
            
        Returns:
            dict: Well-specific observations with correct physical units
        """
        try:
            obs_np = observation.detach().cpu().numpy().flatten() if hasattr(observation, 'detach') else observation.flatten()
        except Exception as e:
            print(f"⚠️ Failed to convert observation to numpy: {e}")
            return {}
        
        well_observations = {}
        
        # Use config-based observation mapping if available
        # Note: Observation structure is defined in ROM config.yaml data.observations.variables
        if hasattr(self, 'obs_indices_map') and self.obs_indices_map:
            try:
                # Map observations using config-based indices
                for var_name, indices in self.obs_indices_map.items():
                    var_config = getattr(self, 'obs_config', {}).get('variables', {}).get(var_name, {})
                    well_names = var_config.get('well_names', [])
                    well_type = var_config.get('well_type', '')
                    unit_display = var_config.get('unit_display', '')
                    
                    for idx, obs_idx in enumerate(indices):
                        if obs_idx < len(obs_np):
                            physical_value = obs_np[obs_idx]
                            
                            # Get well name from config or fallback
                            if idx < len(well_names):
                                well_name = well_names[idx]
                            else:
                                # Fallback naming
                                if well_type == 'injectors':
                                    well_name = f"I{idx+1}"
                                elif well_type == 'producers':
                                    well_name = f"P{idx+1}"
                                else:
                                    well_name = f"W{idx+1}"
                            
                            # Build observation key based on variable name and unit
                            if var_name == 'BHP':
                                well_observations[f"{well_name}_BHP_psi"] = physical_value
                            elif var_name == 'ENERGYRATE':
                                # Store as Energy_BTUday for clarity (value is in BTU/day)
                                well_observations[f"{well_name}_Energy_BTUday"] = physical_value
                                # Also keep Gas_ft3day key for backward compatibility (but it's actually BTU/day)
                                well_observations[f"{well_name}_Gas_ft3day"] = physical_value
                            elif var_name == 'WATRATRC':
                                well_observations[f"{well_name}_Water_ft3day"] = physical_value
                                # Convert to barrels for economic calculations
                                well_observations[f"{well_name}_Water_bblday"] = physical_value / 5.614583
                            else:
                                # Generic key
                                well_observations[f"{well_name}_{var_name}"] = physical_value
                
                return well_observations
            except Exception as e:
                print(f"⚠️ Error mapping observations using config: {e}, falling back to hard-coded")
        
        # Fallback to hard-coded logic if config not available
        # ROM observation order (from config.yaml): [BHP(0-2), WATRATRC(3-5), ENERGYRATE(6-8)]
        try:
            # Injector BHP (indices 0-2) - already in psi
            for i in range(min(3, len(obs_np))):
                well_name = f"I{i+1}"
                physical_value = obs_np[i]  # Already in psi
                well_observations[f"{well_name}_BHP_psi"] = physical_value
            
            # Water Production (WATRATRC, indices 3-5) - already in bbl/day
            for i in range(min(3, max(0, len(obs_np) - 3))):
                well_name = f"P{i+1}"
                if 3 + i < len(obs_np):
                    physical_value = obs_np[3 + i]  # Already in bbl/day
                    well_observations[f"{well_name}_Water_bblday"] = physical_value
            
            # Energy Production (ENERGYRATE, indices 6-8) - already in BTU/Day
            for i in range(min(3, max(0, len(obs_np) - 6))):
                well_name = f"P{i+1}"
                if 6 + i < len(obs_np):
                    physical_value = obs_np[6 + i]  # Already in BTU/Day
                    well_observations[f"{well_name}_Energy_BTUday"] = physical_value
            
            return well_observations
            
        except Exception as e:
            print(f"⚠️ Error mapping physical observations to wells: {e}")
            return {}
    
    def _convert_action_to_physical_units(self, action):
        """
        🔧 FIXED: Convert [0,1] actions to physical units using DASHBOARD ranges
        This ensures visualization shows the actual dashboard-constrained values
        """
        try:
            action_np = action.detach().cpu().numpy().flatten() if hasattr(action, 'detach') else action.flatten()
        except Exception as e:
            print(f"⚠️ Failed to convert action to numpy: {e}")
            return {}
        
        # 🎯 CRITICAL FIX: Get dashboard ranges from stored RL config
        if not self.rl_config or 'action_ranges' not in self.rl_config:
            print("⚠️ No dashboard action ranges found - using fallback conversion")
            return self._fallback_action_conversion(action_np)
        
        action_ranges = self.rl_config['action_ranges']
        action_physical = {}
        
        try:
            # Geothermal action order (from ROM config): [WATRATRC(0-2), BHP(3-5)]
            # WATRATRC: Water Injection Rate (injectors) - indices [0,1,2]
            # BHP: Bottom-Hole Pressure (producers) - indices [3,4,5]
            
            # Water Injection (WATRATRC control, indices 0-2, injectors) - USE DASHBOARD RANGES
            water_ranges = action_ranges.get('water_injection', action_ranges.get('controls', {}).get('WATRATRC', {}))
            if water_ranges:
                # Get dashboard water injection range
                if isinstance(water_ranges, dict) and all(isinstance(v, dict) and 'min' in v and 'max' in v for v in water_ranges.values()):
                    # Well-specific ranges
                    water_mins = [ranges['min'] for ranges in water_ranges.values()]
                    water_maxs = [ranges['max'] for ranges in water_ranges.values()]
                else:
                    water_mins = []
                    water_maxs = []
                
                if water_mins and water_maxs:
                    dashboard_water_min = min(water_mins)
                    dashboard_water_max = max(water_maxs)
                    
                    for i in range(min(3, len(action_np))):
                        well_name = f"I{i+1}"
                        normalized_value = action_np[i]  # [0,1] from policy
                        # Map [0,1] → dashboard range
                        physical_value = normalized_value * (dashboard_water_max - dashboard_water_min) + dashboard_water_min
                        action_physical[f"{well_name}_Water_bblday"] = physical_value
            
            # Producer BHP (indices 3-5) - USE DASHBOARD RANGES
            bhp_ranges = action_ranges.get('bhp', action_ranges.get('controls', {}).get('BHP', {}))
            if bhp_ranges:
                # Get dashboard BHP range
                if isinstance(bhp_ranges, dict) and all(isinstance(v, dict) and 'min' in v and 'max' in v for v in bhp_ranges.values()):
                    # Well-specific ranges
                    bhp_mins = [ranges['min'] for ranges in bhp_ranges.values()]
                    bhp_maxs = [ranges['max'] for ranges in bhp_ranges.values()]
                else:
                    bhp_mins = []
                    bhp_maxs = []
                
                if bhp_mins and bhp_maxs:
                    dashboard_bhp_min = min(bhp_mins)
                    dashboard_bhp_max = max(bhp_maxs)
                    
                    for i in range(min(3, max(0, len(action_np) - 3))):
                        well_name = f"P{i+1}"
                        if 3 + i < len(action_np):
                            normalized_value = action_np[3 + i]  # [0,1] from policy
                            # Map [0,1] → dashboard range
                            physical_value = normalized_value * (dashboard_bhp_max - dashboard_bhp_min) + dashboard_bhp_min
                            action_physical[f"{well_name}_BHP_psi"] = physical_value
            
            return action_physical
            
        except Exception as e:
            print(f"⚠️ Error in dashboard action conversion: {e}")
            return self._fallback_action_conversion(action_np)
    
    def _fallback_action_conversion(self, action_np):
        """
        Fallback action conversion using global ROM parameters when dashboard ranges unavailable
        """
        print("⚠️ Using fallback global ROM parameters for action conversion")
        
        # Load latest training parameters as fallback
        norm_params = self._load_latest_preprocessing_parameters()
        
        if norm_params is None:
            print("❌ No fallback parameters available")
            return {}
        
        action_physical = {}
        
        try:
            def safe_float(value):
                return float(value) if isinstance(value, str) else value
            
            # Producer BHP (indices 0-2) - using global ROM parameters
            if 'BHP' in norm_params:
                bhp_params = norm_params['BHP']
                bhp_min = safe_float(bhp_params['min'])
                bhp_max = safe_float(bhp_params['max'])
                
                for i in range(min(3, len(action_np))):
                    well_name = f"P{i+1}"
                    normalized_value = action_np[i]
                    physical_value = normalized_value * (bhp_max - bhp_min) + bhp_min
                    action_physical[f"{well_name}_BHP_psi"] = physical_value
            
            # Energy Injection (indices 3-5) - using global ROM parameters
            if 'ENERGYRATE' in norm_params:
                gas_params = norm_params['ENERGYRATE']
                gas_min = safe_float(gas_params['min'])
                gas_max = safe_float(gas_params['max'])
                
                for i in range(min(3, max(0, len(action_np) - 3))):
                    well_name = f"I{i+1}"
                    if 3 + i < len(action_np):
                        normalized_value = action_np[3 + i]
                        physical_value = normalized_value * (gas_max - gas_min) + gas_min
                        action_physical[f"{well_name}_Gas_ft3day"] = physical_value
            
            return action_physical
            
        except Exception as e:
            print(f"⚠️ Error in fallback action conversion: {e}")
            return {}
    
    def _load_latest_preprocessing_parameters(self):
        """
        Load latest preprocessing parameters using IDENTICAL method as E2C evaluation
        This ensures perfect consistency with the evaluation process
        """
        try:
            # Try to find the EXACT same normalization files that E2C evaluation uses
            json_files = glob.glob("normalization_parameters_*.json")
            pkl_files = glob.glob("normalization_parameters_*.pkl")
            
            # Sort by modification time to get the latest (same logic as E2C evaluation)
            json_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
            pkl_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
            
            # Try to load the latest JSON file first (same priority as E2C evaluation)
            if json_files:
                latest_json = json_files[0]
                with open(latest_json, 'r') as f:
                    norm_config = json.load(f)
                
                # Extract norm_params in EXACT same format as E2C evaluation
                norm_params = {}
                
                # Load spatial channel parameters (EXACT same extraction as E2C evaluation)
                for var_name, info in norm_config.get('spatial_channels', {}).items():
                    norm_params[var_name] = info['parameters']
                
                # Load control variable parameters (EXACT same extraction as E2C evaluation)
                for var_name, info in norm_config.get('control_variables', {}).items():
                    norm_params[var_name] = info['parameters']
                    
                # Load observation variable parameters (EXACT same extraction as E2C evaluation)
                for var_name, info in norm_config.get('observation_variables', {}).items():
                    norm_params[var_name] = info['parameters']
                
                return norm_params
            
            # Try pickle file as fallback (same fallback logic as E2C evaluation)
            elif pkl_files:
                latest_pkl = pkl_files[0]
                with open(latest_pkl, 'rb') as f:
                    norm_config = pickle.load(f)
                
                # Extract norm_params in EXACT same format as E2C evaluation
                norm_params = {}
                
                # Load spatial channel parameters (EXACT same extraction as E2C evaluation)
                for var_name, info in norm_config.get('spatial_channels', {}).items():
                    norm_params[var_name] = info['parameters']
                
                # Load control variable parameters (EXACT same extraction as E2C evaluation)
                for var_name, info in norm_config.get('control_variables', {}).items():
                    norm_params[var_name] = info['parameters']
                    
                # Load observation variable parameters (EXACT same extraction as E2C evaluation)
                for var_name, info in norm_config.get('observation_variables', {}).items():
                    norm_params[var_name] = info['parameters']
                
                return norm_params
            
        except Exception as e:
            print(f"⚠️ Error loading preprocessing parameters: {e}")
        
            return None
    
    def _calculate_economic_breakdown(self, physical_obs, physical_actions):
        """Calculate detailed economic breakdown for geothermal project"""
        econ_config = self.config.rl_model['economics']
        
        # Get conversion factors
        btu_to_kwh = econ_config['conversion'].get('btu_to_kwh', 0.000293071)
        
        # Get prices
        prices = econ_config['prices']
        energy_production_revenue = prices.get('energy_production_revenue', 0.11)  # $/kWh
        water_production_reward = prices.get('water_production_reward', 5.0)  # $/bbl
        water_injection_cost = prices.get('water_injection_cost', 10.0)  # $/bbl
        
        breakdown = {
            'energy_production_revenue': 0.0,  # Positive revenue from energy production
            'water_production_reward': 0.0,   # Positive reward from water production
            'water_injection_cost': 0.0,      # Negative cost for water injection
            'net_step_cashflow': 0.0,
            'operational_cashflow': 0.0
        }
        
        # Energy production revenue (BTU/day → kWh/day → $)
        # Look for ENERGYRATE observations (stored as Gas_ft3day but actually BTU/day)
        for key, value in physical_obs.items():
            if ('Gas_ft3day' in key or 'Energy' in key) and ('P' in key or 'Prod' in key):  # Producer energy
                # Value is in BTU/day, convert to kWh/day
                energy_kwh_per_day = value * btu_to_kwh
                # Convert daily rates to annual amounts (since each RL step = 1 year)
                annual_revenue = energy_kwh_per_day * energy_production_revenue * 365
                breakdown['energy_production_revenue'] += annual_revenue
        
        # Water production reward (bbl/day → $)
        for key, value in physical_obs.items():
            if 'Water_bblday' in key and ('P' in key or 'Prod' in key):  # Producer water
                # Convert daily rates to annual amounts (since each RL step = 1 year)
                annual_reward = value * water_production_reward * 365
                breakdown['water_production_reward'] += annual_reward
        
        # Water injection cost (bbl/day → $)
        # Look for WATRATRC control actions (water injection)
        for key, value in physical_actions.items():
            if 'Water' in key and ('I' in key or 'Inj' in key):  # Injector water
                # Value should be in bbl/day
                if 'bblday' in key.lower():
                    water_bbl_per_day = value
                elif 'ft3day' in key.lower():
                    # Convert ft3/day to bbl/day
                    water_bbl_per_day = value / 5.614583
                else:
                    # Assume bbl/day if unit not specified
                    water_bbl_per_day = value
                
                # Convert daily rates to annual amounts (since each RL step = 1 year)
                annual_cost = water_bbl_per_day * water_injection_cost * 365
                breakdown['water_injection_cost'] += annual_cost
        
        # Operational cashflow for this step (before any capital costs)
        # Positive revenue/reward minus costs
        breakdown['operational_cashflow'] = (breakdown['energy_production_revenue'] + 
                                            breakdown['water_production_reward'] - 
                                            breakdown['water_injection_cost'])
        
        # Net cashflow (same as operational for step-wise calculation)
        breakdown['net_step_cashflow'] = breakdown['operational_cashflow']
        
        return breakdown

    def finalize_episode(self, episode, total_reward=None):
        """
        Finalize episode and calculate statistics - now stores ALL episodes for dashboard
        
        Args:
            episode: Episode number
            total_reward: Total operational reward (capital cost applied only in post-training analysis)
            
        Returns:
            total_reward: Pure operational reward (no capital cost during training)
        """
        
        # Track statistics without verbose output
        if len(self.episode_actions) > 1:
            stats = self.variation_manager.get_variation_statistics(self.episode_actions)
            self.variation_stats.append(stats)
            
            # Calculate economic breakdown for each step
            economic_breakdowns = []
            for i, (obs, action) in enumerate(zip(self.current_episode_data['observations'], 
                                                self.current_episode_data['actions'])):
                breakdown = self._calculate_economic_breakdown(obs, action)
                economic_breakdowns.append(breakdown)
            
        # 🔥 NEW: Store complete episode data (not just the best one)
        complete_episode_data = {
                'episode_number': episode,
                'total_reward': total_reward,  # Store operational reward only
                'operational_reward': total_reward,  # Store operational reward
                'actions': self.current_episode_data['actions'].copy(),
                'observations': self.current_episode_data['observations'].copy(),
                'rewards': self.current_episode_data['rewards'].copy(),
                'states': self.current_episode_data['states'].copy(),
                'timesteps': self.current_episode_data['timesteps'].copy(),
            'economic_breakdown': economic_breakdowns,
            'spatial_states': self.current_episode_data['spatial_states'].copy()  # 🔥 NEW: Store spatial states
        }
        
        # Add to stored episodes with efficient memory management
        self._store_episode_data(episode, complete_episode_data)
        
        # Check if this is the best episode so far (use operational reward for comparison)
        if total_reward is not None and total_reward > self.best_episode_data['total_reward']:
            print(f"   🏆 NEW BEST EPISODE! Operational reward: {total_reward:.3f} (previous: {self.best_episode_data['total_reward']:.3f})")
            
            # Update best episode data
            self.best_episode_data = complete_episode_data.copy()
        
        # Reset for next episode
        self.episode_actions = []
        self.variation_manager.reset_for_new_episode()
        self.current_episode_data = {
            'actions': [],
            'observations': [],
            'rewards': [],
            'states': [],
            'timesteps': [],
            'economic_breakdown': [],
            'spatial_states': []  # 🔥 NEW: Reset spatial states
        }
        
        return total_reward
    
    def _store_episode_data(self, episode, episode_data):
        """
        Store episode data with efficient memory management
        
        Args:
            episode: Episode number
            episode_data: Complete episode data dictionary
        """
        # Add to stored episodes
        self.stored_episodes[episode] = episode_data
        self.episode_order.append(episode)
        
        # Maintain maximum storage limit
        while len(self.stored_episodes) > self.max_stored_episodes:
            oldest_episode = self.episode_order.pop(0)
            if oldest_episode in self.stored_episodes:
                del self.stored_episodes[oldest_episode]
        
        # Optional: Log storage status periodically
        if episode % 10 == 0:
            print(f"   📊 Episode storage: {len(self.stored_episodes)}/{self.max_stored_episodes} episodes stored")
    
    def get_episode_data(self, episode_number):
        """
        Get data for a specific episode
        
        Args:
            episode_number: Episode number to retrieve
            
        Returns:
            dict: Episode data or None if not found
        """
        return self.stored_episodes.get(episode_number, None)
    
    def get_available_episodes(self):
        """
        Get list of available episode numbers (sorted)
        
        Returns:
            list: Sorted list of available episode numbers
        """
        return sorted(self.stored_episodes.keys())
    
    def get_best_episode_number(self):
        """
        Get the episode number of the best episode
        
        Returns:
            int: Best episode number or -1 if no episodes stored
        """
        return self.best_episode_data['episode_number']
    
    def get_spatial_data_for_episode(self, episode_number):
        """
        Get spatial data for a specific episode formatted for visualization
        
        Args:
            episode_number: Episode number
            
        Returns:
            dict: Spatial data ready for visualization or None if not available
        """
        episode_data = self.get_episode_data(episode_number)
        if episode_data is None or 'spatial_states' not in episode_data:
            return None
        
        spatial_states = episode_data['spatial_states']
        if not spatial_states:
            return None
        
        # Format spatial data for visualization
        formatted_data = {
            'episode_number': episode_number,
            'num_timesteps': len(spatial_states),
            'spatial_states': {},  # Will be indexed by timestep
            'available_timesteps': [],
            'spatial_shape': None,  # Will be determined from first state
            'num_channels': None
        }
        
        for spatial_entry in spatial_states:
            step = spatial_entry['step']
            spatial_state = spatial_entry['spatial_state']
            
            # Store spatial state indexed by timestep
            formatted_data['spatial_states'][step] = spatial_state
            formatted_data['available_timesteps'].append(step)
        
            # Set shape information from first state
            if formatted_data['spatial_shape'] is None and spatial_state is not None:
                if len(spatial_state.shape) >= 4:  # (batch, channels, X, Y, Z)
                    formatted_data['spatial_shape'] = spatial_state.shape[2:]  # (X, Y, Z)
                    formatted_data['num_channels'] = spatial_state.shape[1]
        
        # Sort available timesteps
        formatted_data['available_timesteps'].sort()
        
        return formatted_data
    
    def get_episode_summary_stats(self):
        """
        Get summary statistics for all stored episodes
        
        Returns:
            dict: Summary statistics
        """
        if not self.stored_episodes:
            return {'num_episodes': 0}
        
        episodes = list(self.stored_episodes.values())
        rewards = [ep['total_reward'] for ep in episodes if ep['total_reward'] is not None]
        
        return {
            'num_episodes': len(episodes),
            'best_episode': self.get_best_episode_number(),
            'avg_reward': sum(rewards) / len(rewards) if rewards else 0.0,
            'max_reward': max(rewards) if rewards else 0.0,
            'min_reward': min(rewards) if rewards else 0.0,
            'episodes_with_spatial_data': sum(1 for ep in episodes if ep.get('spatial_states')),
            'available_episodes': self.get_available_episodes()
        }
    
    def get_training_summary(self):
        """Get comprehensive training summary"""
        if not self.variation_stats:
            return "No variation statistics available"
        
        all_variations = [s['mean_variation'] for s in self.variation_stats]
        
        return {
            'total_episodes': len(self.variation_stats),
            'mean_variation': sum(all_variations) / len(all_variations),
            'max_variation': max(all_variations),
            'min_variation': min(all_variations),
            'final_variation': all_variations[-1] if all_variations else 0.0
        }
    
    def get_best_episode_data(self):
        """Get the best episode data for scientific visualization (compatibility method)"""
        if self.best_episode_data['episode_number'] == -1:
            return None
        return self.best_episode_data.copy()
    
    def has_best_episode_data(self):
        """Check if best episode data is available"""
        return self.best_episode_data['episode_number'] != -1

