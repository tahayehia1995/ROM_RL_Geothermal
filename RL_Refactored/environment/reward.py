"""
Reward Function for RL
"""
import torch


def reward_fun(yobs, action, num_prod, num_inj, config):
    """
    Geothermal reward function using ACTUAL H5 tensor order
    
    IMPORTANT: Each RL timestep represents 1 YEAR of operation.
    Rates from ROM are in per-day units, so we multiply by 365 to get annual values.
    
    ACTUAL Observation order (verified from denormalized values):
    - BHP: indices [0,1,2] - INJECTOR Bottom-Hole Pressure (~2000-5000 psi)
    - WATRATRC: indices [3,4,5] - Water PRODUCTION Rate (~10^4 bbl/day, producers)
    - ENERGYRATE: indices [6,7,8] - Energy Production Rate (~10^12 BTU/day thermal, producers)
    
    ACTUAL Action/Control order:
    - BHP: indices [0,1,2] - Producer Bottom-Hole Pressure (psi, producers)
    - WATRATRC: indices [3,4,5] - Water Injection Rate (bbl/day, injectors)
    
    Reward Formula (ANNUAL values per timestep):
    1. Convert thermal BTU to electrical BTU: Energy_thermal × thermal_to_electrical_efficiency
    2. Convert BTU to kWh: Energy_electrical_BTU × 0.000293071
    3. Convert daily to annual: multiply all rates by 365
    4. Calculate economics:
       Reward = (Energy_kWh/year × $0.0011/kWh) 
              - (Water_production_bbl/year × $5/bbl)
              - (Water_injection_bbl/year × $10/bbl)
    
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
        Economic reward value (scaled) representing ANNUAL profit for this timestep
    """
    econ_config = config.rl_model['economics']
    
    # Unit conversion factors from config
    btu_to_kwh = econ_config['conversion'].get('btu_to_kwh', 0.000293071)
    days_per_year = econ_config['conversion'].get('days_per_year', 365)
    
    # Thermal to electrical efficiency (geothermal power plant efficiency ~10%)
    thermal_to_electrical_efficiency = econ_config['conversion'].get('thermal_to_electrical_efficiency', 0.1)
    
    # Economic parameters from config
    prices = econ_config['prices']
    energy_production_revenue = prices.get('energy_production_revenue', 0.0011)  # $/kWh (electrical)
    water_production_cost = prices.get('water_production_cost', 5.0)  # $/bbl (disposal cost)
    water_injection_cost = prices.get('water_injection_cost', 10.0)  # $/bbl
    
    # Extract observations using ACTUAL order: [BHP_inj(0-2), WATRATRC(3-5), ENERGYRATE(6-8)]
    # BHP: indices 0 to num_inj (injectors) - not used in reward
    # WATRATRC: indices num_inj to num_inj+num_prod (water production from producers, ~10^4 bbl/day)
    # ENERGYRATE: indices num_inj+num_prod to num_inj+num_prod*2 (energy production from producers, ~10^12 BTU/day thermal)
    
    # Water PRODUCTION (WATRATRC observations, indices 3-5, producers) - bbl/day
    water_production_bbl_day = torch.sum(yobs[:, num_inj:num_inj+num_prod], dim=1)
    # Convert to annual: bbl/day × 365 days/year = bbl/year
    water_production_bbl_year = water_production_bbl_day * days_per_year
    
    # Energy production (ENERGYRATE observations, indices 6-8, producers) - BTU/day (thermal)
    energy_production_btu_day_thermal = torch.sum(yobs[:, num_inj+num_prod:num_inj+num_prod*2], dim=1)
    # Apply thermal to electrical conversion (power plant efficiency)
    energy_production_btu_day_electrical = energy_production_btu_day_thermal * thermal_to_electrical_efficiency
    # Convert BTU to kWh
    energy_production_kwh_day = energy_production_btu_day_electrical * btu_to_kwh
    # Convert to annual: kWh/day × 365 days/year = kWh/year
    energy_production_kwh_year = energy_production_kwh_day * days_per_year
    
    # Extract actions using ACTUAL order: [BHP(0-2), WATRATRC(3-5)]
    
    # Water injection (WATRATRC control, indices 3-5, injectors) - bbl/day
    water_injection_bbl_day = torch.sum(action[:, num_prod:num_prod+num_inj], dim=1)
    # Convert to annual: bbl/day × 365 days/year = bbl/year
    water_injection_bbl_year = water_injection_bbl_day * days_per_year
    
    # Calculate geothermal reward (ANNUAL economics per timestep):
    # - Energy production revenue: POSITIVE ($/kWh × kWh/year)
    # - Water production cost: NEGATIVE ($/bbl × bbl/year) - disposal/handling cost
    # - Water injection cost: NEGATIVE ($/bbl × bbl/year) - pumping/treatment cost
    reward = (energy_production_revenue * energy_production_kwh_year - 
              water_production_cost * water_production_bbl_year -
              water_injection_cost * water_injection_bbl_year) / econ_config['scale_factor']
    
    return reward

