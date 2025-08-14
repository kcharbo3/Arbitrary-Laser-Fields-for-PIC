from dataclasses import dataclass, field
import numpy as np
from typing import List

# ------------ Propagation Parameters ---------------
@dataclass
class PropagationParameters:
    spatial_dimensions: int
    propagation_type: str
    monochromatic_assumption: bool
    y_input_range: float
    z_input_range: float
    N_y_input: int
    N_z_input: int
    y_vals_input: np.ndarray
    z_vals_input: np.ndarray
    y_output_range: float
    z_output_range: float
    N_y_output: int
    N_z_output: int
    y_vals_output: np.ndarray
    z_vals_output: np.ndarray
    N_t: int
    t_range: float
    times: np.ndarray
    omegas: np.ndarray
    save_data_as_files: bool
    sim_directory: str
    data_directory_path: str
    low_mem: bool

# ------------ Laser Parameters ---------------
@dataclass
class LaserParameters:
    wavelength: float
    ref_freq: float
    omega0: float
    polarization: str
    spatial_shape: str
    spatial_gaussian_order: int
    temporal_shape: str
    temporal_gaussian_order: int
    phase_offset: float
    deltax: float
    pulse_fwhm: float
    spot_size: float
    waist_in: float
    output_distance_from_focus: float
    normalize_to_a0: bool
    peak_a0: float
    total_energy: float
    l: int
    num_petals: int
    waist_in_radial: float
    waist_in_azimuthal: float

# ------------ Advanced Parameters ---------------
@dataclass
class GratingParameters:
    use_grating_eq: bool
    alpha: float
    grating_aois: List[float]
    groove_periods: List[float]
    diffraction_orders: List[int]
    grating_separations: List[float]

@dataclass
class ThickLens:
    use_thick_lens: bool
    r1_lens: float
    r2_lens: float
    lens_center_thickness: float

@dataclass
class AdvancedParameters:
    center_peak_E_at_0: bool
    grating_params: GratingParameters
    axicon_angle: float
    echelon_delay: float
    thick_lens: ThickLens

# ------------ Simulation Grid Parameters ---------------
@dataclass
class SimulationGridParameters:
    y_height: float = 0.0
    dy_sim: float = 0.0
    z_height: float = 0.0
    dz_sim: float = 0.0
    t_length: float = 0.0
    dt_sim: float = 0.0
    x_length: float = 0.0
    dx_sim: float = 0.0
    laser_start_time: float = 0.0

# ------------ Other Simulation Parameters ---------------
@dataclass
class PrePlasmaParameters:
    pre_plasma: bool = False
    char_length: float = 0.0
    cut_off_density: float = 0.0

@dataclass
class SimParameters:
    n0: np.ndarray = np.array([])
    foil_left_x: np.ndarray = np.array([])
    foil_radius: np.ndarray = np.array([])
    foil_thickness: np.ndarray = np.array([])
    centery: np.ndarray = np.array([])
    centerz: np.ndarray = np.array([])
    foil_angle: np.ndarray = np.array([])
    pre_plasma_params: PrePlasmaParameters = PrePlasmaParameters()


# ------------ Getters ---------------
def get_laser_params(wavelength, ref_freq, omega0, polarization, spatial_shape, spatial_gaussian_order, temporal_shape,
                     temporal_gaussian_order, phase_offset, deltax, pulse_fwhm, spot_size, waist_in,
                     output_distance_from_focus, normalize_to_a0, peak_a0, total_energy, l, num_petals,
                     waist_in_radial, waist_in_azimuthal):
    laser_parameters_obj = LaserParameters(
        wavelength=wavelength, ref_freq=ref_freq, omega0=omega0, polarization=polarization,
        spatial_shape=spatial_shape, spatial_gaussian_order=spatial_gaussian_order, temporal_shape=temporal_shape,
        temporal_gaussian_order=temporal_gaussian_order, phase_offset=phase_offset, deltax=deltax, pulse_fwhm=pulse_fwhm,
        spot_size=spot_size, waist_in=waist_in, output_distance_from_focus=output_distance_from_focus,
        normalize_to_a0=normalize_to_a0, peak_a0=peak_a0, total_energy=total_energy, l=l, num_petals=num_petals,
        waist_in_radial=waist_in_radial, waist_in_azimuthal=waist_in_azimuthal
    )

    return laser_parameters_obj

def get_prop_params(spatial_dimensions, propagation_type, monochromatic_assumption, y_input_range, z_input_range,
                    N_y_input, N_z_input, y_vals_input, z_vals_input, y_output_range, z_output_range, N_y_output,
                    N_z_output, y_vals_output, z_vals_output, N_t, t_range, times, omegas, save_data_as_files,
                    sim_directory, data_directory_path, low_mem):
    propagation_parameters_obj = PropagationParameters(
        spatial_dimensions=spatial_dimensions, propagation_type=propagation_type,
        monochromatic_assumption=monochromatic_assumption, y_input_range=y_input_range,
        z_input_range=z_input_range, N_y_input=N_y_input, N_z_input=N_z_input, y_vals_input=y_vals_input,
        z_vals_input=z_vals_input, y_output_range=y_output_range, z_output_range=z_output_range,
        N_y_output=N_y_output, N_z_output=N_z_output, y_vals_output=y_vals_output, z_vals_output=z_vals_output,
        N_t=N_t, t_range=t_range, times=times, omegas=omegas, save_data_as_files=save_data_as_files,
        sim_directory=sim_directory, data_directory_path=data_directory_path, low_mem=low_mem
    )
    return propagation_parameters_obj

def get_grating_params(use_grating_eq, alpha, grating_aois, groove_periods, diffraction_orders, grating_separations):
    grating_params = GratingParameters(
        use_grating_eq=use_grating_eq,
        alpha=alpha,
        grating_aois=grating_aois,
        groove_periods=groove_periods,
        diffraction_orders=diffraction_orders,
        grating_separations=grating_separations
    )

    return grating_params

def get_thick_lens_params(use_thick_lens, r1_lens, r2_lens, lens_center_thickness):
    thick_lens = ThickLens(
        use_thick_lens=use_thick_lens,
        r1_lens=r1_lens,
        r2_lens=r2_lens,
        lens_center_thickness=lens_center_thickness
    )

    return thick_lens

def get_advanced_params(center_peak_E_at_0, grating_params,
                        axicon_angle, echelon_delay, thick_lens):
    advanced_parameters_obj = AdvancedParameters(
        center_peak_E_at_0=center_peak_E_at_0,
        grating_params=grating_params,
        axicon_angle=axicon_angle,
        echelon_delay=echelon_delay,
        thick_lens=thick_lens
    )

    return advanced_parameters_obj

def get_sim_grid_params(y_height, dy_sim, z_height,
                        dz_sim, t_length, dt_sim,
                        x_length, dx_sim, laser_start_time):
    sim_grid_params = SimulationGridParameters(
        y_height=y_height, dy_sim=dy_sim, z_height=z_height, dz_sim=dz_sim,
        t_length=t_length, dt_sim=dt_sim, x_length=x_length, dx_sim=dx_sim,
        laser_start_time=laser_start_time
    )
    return sim_grid_params

def get_pre_plasma_params(pre_plasma, char_length, cut_off_density):
    pre_plasma_params = PrePlasmaParameters(
        pre_plasma=pre_plasma, char_length=char_length, cut_off_density=cut_off_density
    )

    return pre_plasma_params

def get_other_sim_params(n0, foil_left_x, foil_radius, foil_thickness, centery, centerz, foil_angle, pre_plasma_params):
    sim_params = SimParameters(
        n0=n0, foil_left_x=foil_left_x, foil_radius=foil_radius, foil_thickness=foil_thickness,
        centery=centery, centerz=centerz, foil_angle=foil_angle, pre_plasma_params=pre_plasma_params
    )

    return sim_params
