"""
Reward Function for RL
"""
import torch


def reward_fun(yobs, action, num_prod, num_inj, config):
    """
    Geothermal reward function using ROM config structure
    
    Observation order (from ROM config): [WATRATRC(0-2), BHP(3-5), ENERGYRATE(6-8)]
    - WATRATRC: Water Production Rate (producers) - indices [0,1,2]
    - BHP: Bottom-Hole Pressure (injectors) - indices [3,4,5]
    - ENERGYRATE: Energy Production Rate (producers) - indices [6,7,8]
    
    Action order (from ROM config): [WATRATRC(0-2), BHP(3-5)]
    - WATRATRC: Water Injection Rate (injectors) - indices [0,1,2]
    - BHP: Bottom-Hole Pressure (producers) - indices [3,4,5]
    
    Reward Formula:
    Reward = (Energy_production_kWh * $0.11/kWh) 
           + (Water_production_bbl * $5/bbl)
           - (Water_injection_bbl * $10/bbl)
    
    Args:
        yobs: Observations [WATRATRC(3), BHP(3), ENERGYRATE(3)]
        action: Actions [WATRATRC_injection(3), BHP(3)]
        num_prod: Number of production wells
        num_inj: Number of injection wells  
        config: Configuration object
        
    Returns:
        Economic reward value (scaled)
    """
    econ_config = config.rl_model['economics']
    
    # Unit conversion factors from config
    btu_to_kwh = econ_config['conversion'].get('btu_to_kwh', 0.000293071)
    
    # Economic parameters from config
    prices = econ_config['prices']
    energy_production_revenue = prices.get('energy_production_revenue', 0.11)  # $/kWh
    water_production_reward = prices.get('water_production_reward', 5.0)  # $/bbl
    water_injection_cost = prices.get('water_injection_cost', 10.0)  # $/bbl
    
    # Extract observations using ROM config order: [WATRATRC(0-2), BHP(3-5), ENERGYRATE(6-8)]
    
    # Water production (WATRATRC observations, indices 0-2, producers)
    # Already in bbl/day from ROM denormalization
    water_production_bbl_day = torch.sum(yobs[:, 0:num_prod], dim=1)
    
    # Energy production (ENERGYRATE observations, indices 6-8, producers)
    # Note: indices are num_inj+num_prod to num_inj+num_prod*2 (which is 3+3 to 3+6 = 6 to 9)
    # But ENERGYRATE is at indices 6-8, so: num_inj+num_prod = 3+3 = 6, correct!
    energy_production_btu_day = torch.sum(yobs[:, num_inj+num_prod:num_inj+num_prod*2], dim=1)
    energy_production_kwh_day = energy_production_btu_day * btu_to_kwh
    
    # Extract actions using ROM config order: [WATRATRC(0-2), BHP(3-5)]
    
    # Water injection (WATRATRC control, indices 0-2, injectors)
    # Already in bbl/day from action denormalization
    water_injection_bbl_day = torch.sum(action[:, 0:num_inj], dim=1)
    
    # Calculate geothermal reward:
    # - Energy production revenue: POSITIVE ($/kWh × kWh/day)
    # - Water production reward: POSITIVE ($/bbl × bbl/day)
    # - Water injection cost: NEGATIVE ($/bbl × bbl/day)
    reward = (energy_production_revenue * energy_production_kwh_day + 
              water_production_reward * water_production_bbl_day - 
              water_injection_cost * water_injection_bbl_day) / econ_config['scale_factor']
    
    return reward

