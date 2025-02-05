from fourier_prop.laser_input import (input_laser_field, laser_parameters, utils)
import numpy as np
from dataclasses import dataclass


# In microns
Y_HEIGHT = 28
DY_SIM = 1 / 16.
Z_HEIGHT = 28
DZ_SIM = 1 / 16.
T_LENGTH = 600.
DT_SIM = DY_SIM * (0.95 / np.sqrt(2.)) * (0.22/0.27)  # Extra term for CFL condition in 3D...

# Not needed for the interpolator, just for sims
X_LENGTH = 100.
DX_SIM = 1 / 16.

LASER_TIME_START = 74.93662282191235  # fs

INTERP_Y_PREFIX = 'laser_vals_y_'
INTERP_Z_PREFIX = 'laser_vals_z_'

@dataclass
class Indexes:
    lo_index_sim: int
    hi_index_sim: int
    lo_index_sim_half: int
    hi_index_sim_half: int
    lo_index_output: int
    hi_index_output: int

@dataclass
class SimGridParameters:
    num_t_vals: int
    num_y_vals: int
    num_z_vals: int
    t_indexes: Indexes
    y_indexes: Indexes
    z_indexes: Indexes
    output_times_code_units: np.ndarray
    output_y_vals_code_units: np.ndarray
    output_z_vals_code_units: np.ndarray
    sim_times_code_units: np.ndarray
    sim_y_vals_code_units: np.ndarray
    sim_y_vals_code_units_half: np.ndarray
    sim_z_vals_code_units: np.ndarray
    sim_z_vals_code_units_half: np.ndarray
    sim_times_fs: np.ndarray
    sim_y_vals_um: np.ndarray
    sim_z_vals_um: np.ndarray
    laser_time_start_code_units: float
    center_y_code_units: float
    center_z_code_units: float

def compute_sim_grid(times, y_vals_output, z_vals_output):
    ref_freq = laser_parameters.REF_FREQ
    num_wavelengths_y = utils.microns_to_norm_units(Y_HEIGHT, ref_freq) / (2*np.pi)
    num_wavelengths_z = utils.microns_to_norm_units(Z_HEIGHT, ref_freq) / (2*np.pi)
    num_periods = utils.fs_to_norm_units(T_LENGTH, ref_freq) / (2*np.pi)

    output_time_vals_code_units = utils.fs_to_norm_units(times, ref_freq)
    output_y_vals_code_units = utils.microns_to_norm_units(y_vals_output, ref_freq)
    output_z_vals_code_units = utils.microns_to_norm_units(z_vals_output, ref_freq)

    cell_height_y = (2*np.pi) * DY_SIM
    cell_height_z = (2*np.pi) * DZ_SIM
    cell_length = (2*np.pi) * DT_SIM

    y_length_code_units = num_wavelengths_y * 2*np.pi
    y_vals_sim = np.arange(-2 * cell_height_y, y_length_code_units + 3*cell_height_y, cell_height_y)
    y_vals_sim_half = y_vals_sim + cell_height_y/2.

    z_length_code_units = num_wavelengths_z * 2*np.pi
    z_vals_sim = np.arange(-2 * cell_height_z, z_length_code_units + 3*cell_height_z, cell_height_z)
    z_vals_sim_half = z_vals_sim + cell_height_z/2.

    t_length_code_units = num_periods * 2*np.pi
    t_vals_sim = np.arange(-2 * cell_length, t_length_code_units + 3*cell_length, cell_length) + cell_length/2.

    y_vals_sim_um = utils.norm_units_to_microns(y_vals_sim, ref_freq)
    y_vals_sim_um_half = utils.norm_units_to_microns(y_vals_sim_half, ref_freq)

    z_vals_sim_um = utils.norm_units_to_microns(z_vals_sim, ref_freq)
    z_vals_sim_um_half = utils.norm_units_to_microns(z_vals_sim_half, ref_freq)

    t_vals_sim_fs = utils.norm_units_to_fs(t_vals_sim, ref_freq)

    laser_time_start = utils.fs_to_norm_units(LASER_TIME_START, ref_freq)
    center_y = utils.microns_to_norm_units(Y_HEIGHT / 2.0, ref_freq)
    center_z = utils.microns_to_norm_units(Z_HEIGHT / 2.0, ref_freq)

    t_lo_index_output, t_hi_index_output = _get_output_indices_for_smaller_grid(
        laser_time_start,
        output_time_vals_code_units,
        t_vals_sim
    )
    t_lo_index_sim, t_hi_index_sim, num_t_vals = \
        _get_sim_indices_for_smaller_grid(
            t_lo_index_output, t_hi_index_output, laser_time_start, output_time_vals_code_units, t_vals_sim
        )

    t_indexes = Indexes(
        lo_index_sim=t_lo_index_sim, hi_index_sim=t_hi_index_sim,
        lo_index_sim_half=t_lo_index_sim, hi_index_sim_half=t_hi_index_sim,
        lo_index_output=t_lo_index_output, hi_index_output=t_hi_index_output
    )

    y_lo_index_output, y_hi_index_output = _get_output_indices_for_smaller_grid(
        center_y,
        output_y_vals_code_units,
        y_vals_sim
    )
    y_lo_index_sim, y_hi_index_sim, num_y_vals = \
        _get_sim_indices_for_smaller_grid(
            y_lo_index_output, y_hi_index_output, center_y, output_y_vals_code_units, y_vals_sim
        )

    y_lo_index_sim_half, y_hi_index_sim_half, num_y_vals_half = \
        _get_sim_indices_for_smaller_grid(
            y_lo_index_output, y_hi_index_output, center_y, output_y_vals_code_units, y_vals_sim_half
        )

    y_indexes = Indexes(
        lo_index_sim=y_lo_index_sim, hi_index_sim=y_hi_index_sim,
        lo_index_sim_half=y_lo_index_sim_half, hi_index_sim_half=y_hi_index_sim_half,
        lo_index_output=y_lo_index_output, hi_index_output=y_hi_index_output
    )

    z_lo_index_output, z_hi_index_output = _get_output_indices_for_smaller_grid(
        center_z,
        output_z_vals_code_units,
        z_vals_sim
    )
    z_lo_index_sim, z_hi_index_sim, num_z_vals = \
        _get_sim_indices_for_smaller_grid(
            z_lo_index_output, z_hi_index_output, center_z, output_z_vals_code_units, z_vals_sim
        )

    z_lo_index_sim_half, z_hi_index_sim_half, num_z_vals_half = \
        _get_sim_indices_for_smaller_grid(
            z_lo_index_output, z_hi_index_output, center_z, output_z_vals_code_units, z_vals_sim_half
        )

    z_indexes = Indexes(
        lo_index_sim=z_lo_index_sim, hi_index_sim=z_hi_index_sim,
        lo_index_sim_half=z_lo_index_sim_half, hi_index_sim_half=z_hi_index_sim_half,
        lo_index_output=z_lo_index_output, hi_index_output=z_hi_index_output
    )

    #print("NUM Y VALS AND NUM_Y_HALF:", num_y_vals, num_y_vals_half)

    return SimGridParameters(
        num_t_vals=num_t_vals, num_y_vals=num_y_vals, num_z_vals=num_z_vals,

        t_indexes=t_indexes, y_indexes=y_indexes, z_indexes=z_indexes,

        output_times_code_units=output_time_vals_code_units, output_y_vals_code_units=output_y_vals_code_units,
        output_z_vals_code_units=output_z_vals_code_units,

        sim_times_code_units=t_vals_sim, sim_y_vals_code_units=y_vals_sim, sim_y_vals_code_units_half=y_vals_sim_half,
        sim_z_vals_code_units=z_vals_sim, sim_z_vals_code_units_half = z_vals_sim_half,

        sim_times_fs=t_vals_sim_fs, sim_y_vals_um=y_vals_sim_um, sim_z_vals_um=z_vals_sim_um,

        laser_time_start_code_units=laser_time_start, center_y_code_units=center_y, center_z_code_units=center_z
    )


def _get_output_indices_for_smaller_grid(center_val, output_arr, sim_arr):
    lo_val = np.maximum(-center_val, output_arr.min())
    hi_val = np.minimum(sim_arr.max() - center_val, output_arr.max())
    lo_index = np.argmin(np.abs(output_arr - lo_val))
    hi_index = np.argmin(np.abs(output_arr - hi_val))

    return lo_index, hi_index


def _get_sim_indices_for_smaller_grid(lo_index_output, hi_index_output, center_val, output_arr, sim_arr):
    lo_val = output_arr[lo_index_output] + center_val
    hi_val = output_arr[hi_index_output] + center_val
    lo_index_sim = np.argmin(np.abs(sim_arr - lo_val))
    hi_index_sim = np.argmin(np.abs(sim_arr - hi_val))

    if lo_index_sim >= 4:
        lo_index_sim -= 4
    else:
        lo_index_sim = 0

    if hi_index_sim <= len(sim_arr) - 5:
        hi_index_sim += 4
    else:
        hi_index_sim = len(sim_arr) - 1
    num_vals = (hi_index_sim - lo_index_sim) + 1
    return lo_index_sim, hi_index_sim, num_vals
