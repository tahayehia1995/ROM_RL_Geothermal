"""
Reward Function for RL
"""
import torch


def reward_fun(yobs, action, num_prod, num_inj, config):
    """
    Geothermal reward function using ACTUAL H5 tensor order
    
    ACTUAL Observation order (verified from denormalized values):
    - BHP: indices [0,1,2] - INJECTOR Bottom-Hole Pressure (~2000-5000 psi)
    - WATRATRC: indices [3,4,5] - Water PRODUCTION Rate (~10^4 bbl/day, producers)
    - ENERGYRATE: indices [6,7,8] - Energy Production Rate (~10^12 BTU/day, producers)
    
    ACTUAL Action/Control order:
    - BHP: indices [0,1,2] - Producer Bottom-Hole Pressure (psi, producers)
    - WATRATRC: indices [3,4,5] - Water Injection Rate (bbl/day, injectors)
    
    Reward Formula:
    Reward = (Energy_production_kWh * $0.11/kWh) 
           - (Water_production_bbl * $5/bbl)
           - (Water_injection_bbl * $10/bbl)
    
    Args:
        yobs: Observations in ACTUAL order [BHP_inj(3), WATRATRC_prod(3), ENERGYRATE(3)]
              - BHP is from INJECTORS
              - WATRATRC (water production) and ENERGYRATE are from PRODUCERS
        action: Actions in ACTUAL order [BHP(3), WATRATRC(3)]
              where BHP is for producers, WATRATRC is Water INJECTION to injectors
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
    water_production_cost = prices.get('water_production_cost', 5.0)  # $/bbl (disposal cost)
    water_injection_cost = prices.get('water_injection_cost', 10.0)  # $/bbl
    
    # Extract observations using ACTUAL order: [BHP_inj(0-2), WATRATRC(3-5), ENERGYRATE(6-8)]
    # BHP: indices 0 to num_inj (injectors) - not used in reward
    # WATRATRC: indices num_inj to num_inj+num_prod (water production from producers, ~10^4 bbl/day)
    # ENERGYRATE: indices num_inj+num_prod to num_inj+num_prod*2 (energy production from producers, ~10^12 BTU/day)
    
    # Water PRODUCTION (WATRATRC observations, indices 3-5, producers)
    water_production_bbl_day = torch.sum(yobs[:, num_inj:num_inj+num_prod], dim=1)
    
    # Energy production (ENERGYRATE observations, indices 6-8, producers)
    energy_production_btu_day = torch.sum(yobs[:, num_inj+num_prod:num_inj+num_prod*2], dim=1)
    energy_production_kwh_day = energy_production_btu_day * btu_to_kwh
    
    # Extract actions using ACTUAL order: [BHP(0-2), WATRATRC(3-5)]
    
    # Water injection (WATRATRC control, indices 3-5, injectors)
    # Already in bbl/day from action denormalization
    water_injection_bbl_day = torch.sum(action[:, num_prod:num_prod+num_inj], dim=1)
    
    # Calculate geothermal reward:
    # - Energy production revenue: POSITIVE ($/kWh × kWh/day)
    # - Water production cost: NEGATIVE ($/bbl × bbl/day) - disposal/handling cost
    # - Water injection cost: NEGATIVE ($/bbl × bbl/day) - pumping/treatment cost
    reward = (energy_production_revenue * energy_production_kwh_day - 
              water_production_cost * water_production_bbl_day -
              water_injection_cost * water_injection_bbl_day) / econ_config['scale_factor']
    
    return reward

