#!/usr/bin/env python3
"""Earth 1-D Lightning Simulation."""


from dataclasses import dataclass


PROJECT_NAME = "convective_plume_earth"


@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants used in lightning plume model calculations."""

    gravity: float = 9.81  # Gravitational acceleration [m/s2]
    universal_gas_constant: float = 8.31446  # Universal gas constant [J/mol/K]
    molar_mass_dry_air: float = 0.02896  # Molar mass of dry air [kg/mol]
    epsilon: float = 0.6222  # Ratio of molecular weights [dimensionless]
    c_p: float = 14500.0  # Specific heat capacity at constant pressure [J/kg/K]
    latent_heat_v: float = 2257000.0  # Latent heat of vaporization of water [J/kg]
    vacuum_perm: float = 8.854e-12  # Vacuum permittivity [F/m]
    e_charge: float = 1.602e-19  # Elementary charge [C]
    rho_water: float = 1000.0  # Density of liquid water [kg/m3]
    rhoro: float = 2.5  # Ratio of ice to liquid water density [dimensionless]
    drag_coef: float = 0.5  # Drag coefficient [dimensionless]
    energy_per_flash: float = 1.5e9  # Energy per lightning flash [J]
    mean_free_path_ion_coll: float = (
        4.0e-11  # Mean free path time for ion collisions [s]
    )
    temp_freeze: float = 273.15  # Freezing point of water [K]
    pa_to_bar: float = 1e-5  # Conversion factor from Pascal to bar
    surface_ten: float = 0.00072 #Surface tension between liquid condensible and air [N/m]
    rho_air: float = 1.293 #Density of liquid condensible [kg/m3]





@dataclass(frozen=True)
class SimulationParameters:
    """Configurable parameters for a single simulation run."""

    plume_base_temp: float
    base_humidity_fraction: float
    plume_base_radius: float
    temp_supercool: float
    water_collision_efficiency: float
    ice_collision_efficiency: float
    start_pressure: float = 100_000.0
    start_upward_velocity: float = 0.001
    pressure_step: float = 10.0
    growth_time_step: float = 0.01
    n_bins: int = 31
    min_radius: float = 1e-5
    max_radius: float = 0.46340950011842
    flash_rate_sampling: int = 10
    dt = 0.01