import fourier_prop.laser_input.input_laser_field as laser_input
from fourier_prop.laser_input import (constants, utils)
from fourier_prop.read_laser import (read_laser, sim_grid_parameters)
import numpy as np
import matplotlib.pyplot as plt

# TODO: Clean up plotting... a lot of duplicate code here
# TODO: Option to plot intensity (Iy, Iz, or Itot)

def plot_Et_output_transverse(input_field: laser_input.InputField, time, is_Ey=True, ybound=0, zbound=0, **kwargs):
    Et_vals = np.flip(get_Et_transverse(input_field, time, is_Ey, ybound, zbound), axis=0)

    if ybound == 0:
        ybound = input_field.prop.y_vals_output.max()
    if zbound == 0:
        zbound = input_field.prop.z_vals_output.max()

    plt.imshow(Et_vals.real, aspect='auto', extent=[-zbound, zbound, -ybound, ybound], **kwargs)
    plt.ylabel("Y Axis [um]")
    plt.xlabel("Z Axis [um]")
    E_type = "Ey"
    if not is_Ey:
        E_type = "Ez"
    plt.title("Output " + E_type + " Field Cross Section at Time: " + f"{time:.2f}" + " fs")


def plot_Et_input_transverse(input_field: laser_input.InputField, time, is_Ey=True, ybound=0, zbound=0, **kwargs):
    Et_vals = np.flip(get_Et_transverse(input_field, time, is_Ey, ybound, zbound, False), axis=0)

    if ybound == 0:
        ybound = input_field.prop.y_vals_input.max()
    if zbound == 0:
        zbound = input_field.prop.z_vals_input.max()

    plt.imshow(Et_vals.real, aspect='auto', extent=[-zbound, zbound, -ybound, ybound], **kwargs)
    plt.ylabel("Y Axis [um]")
    plt.xlabel("Z Axis [um]")
    E_type = "Ey"
    if not is_Ey:
        E_type = "Ez"
    plt.title("Input " + E_type + " Field Cross Section at Time: " + f"{time:.2f}" + " fs")


def plot_Ew_output_transverse(input_field: laser_input.InputField, w, is_Ey=True, ybound=0, zbound=0, **kwargs):
    Ew_vals = np.flip(get_Ew_transverse(input_field, w, is_Ey, ybound, zbound), axis=0)

    if ybound == 0:
        ybound = input_field.prop.y_vals_output.max()
    if zbound == 0:
        zbound = input_field.prop.z_vals_output.max()

    plt.imshow(Ew_vals.real, aspect='auto', extent=[-zbound, zbound, -ybound, ybound], **kwargs)
    plt.ylabel("Y Axis [um]")
    plt.xlabel("Z Axis [um]")
    E_type = "Ey"
    if not is_Ey:
        E_type = "Ez"
    plt.title("Output " + E_type + " Field Cross Section at Omega: " + f"{w:.2f}" + " rad*PHz")


def plot_Ew_input_transverse(input_field: laser_input.InputField, w, is_Ey=True, ybound=0, zbound=0, **kwargs):
    Ew_vals = np.flip(get_Ew_transverse(input_field, w, is_Ey, ybound, zbound, False), axis=0)

    if ybound == 0:
        ybound = input_field.prop.y_vals_input.max()
    if zbound == 0:
        zbound = input_field.prop.z_vals_input.max()

    plt.imshow(Ew_vals.real, aspect='auto', extent=[-zbound, zbound, -ybound, ybound], **kwargs)
    plt.ylabel("Y Axis [um]")
    plt.xlabel("Z Axis [um]")
    E_type = "Ey"
    if not is_Ey:
        E_type = "Ez"
    plt.title("Input " + E_type + " Field Cross Section at Omega: " + f"{w:.2f}" + " rad*PHz")


def plot_Et_output_YT(input_field: laser_input.InputField, z_val, is_Ey=True, ybound=0, tbound=0, **kwargs):
    Et_vals = np.flip(get_Et_YT(input_field, z_val, is_Ey, ybound, tbound).T, axis=0)

    if ybound == 0:
        ybound = input_field.prop.y_vals_output.max()
    if tbound == 0:
        tbound = input_field.prop.times.max()

    plt.imshow(Et_vals.real, aspect='auto', extent=[tbound, -tbound, -ybound, ybound], **kwargs)
    plt.ylabel("Y Axis [um]")
    plt.xlabel("Time Axis [fs]")
    E_type = "Ey"
    if not is_Ey:
        E_type = "Ez"
    plt.title("Output " + E_type + " Field Cross Section at Z: " + f"{z_val:.2f}" + " um")

def plot_Ew_output_YW(input_field: laser_input.InputField, z_val, is_Ey=True, ybound=0, wbound=0, **kwargs):
    Ew_vals = np.flip(get_Ew_YW(input_field, z_val, is_Ey, ybound, wbound).T, axis=0)

    if ybound == 0:
        ybound = input_field.prop.y_vals_output.max()
    if wbound == 0:
        wbound = input_field.prop.omegas.max()

    plt.imshow(Ew_vals.real, aspect='auto', extent=[-wbound, wbound, -ybound, ybound], **kwargs)
    plt.ylabel("Y Axis [um]")
    plt.xlabel("Frequency Axis [rad/PHz]")
    E_type = "Ey"
    if not is_Ey:
        E_type = "Ez"
    plt.title("Output " + E_type + " Field Cross Section at Z: " + f"{z_val:.2f}" + " um")

def plot_Ew_input_YW(input_field: laser_input.InputField, z_val, is_Ey=True, ybound=0, wbound=0, **kwargs):
    Ew_vals = np.flip(get_Ew_YW(input_field, z_val, is_Ey, ybound, wbound, False).T, axis=0)

    if ybound == 0:
        ybound = input_field.prop.y_vals_input.max()
    if wbound == 0:
        wbound = input_field.prop.omegas.max()

    plt.imshow(Ew_vals.real, aspect='auto', extent=[-wbound, wbound, -ybound, ybound], **kwargs)
    plt.ylabel("Y Axis [um]")
    plt.xlabel("Frequency Axis [rad/PHz]")
    E_type = "Ey"
    if not is_Ey:
        E_type = "Ez"
    plt.title("Input " + E_type + " Field Cross Section at Z: " + f"{z_val:.2f}" + " um")

def plot_Et_output_ZT(input_field: laser_input.InputField, y_val, is_Ey=True, zbound=0, tbound=0, **kwargs):
    Et_vals = np.flip(get_Et_ZT(input_field, y_val, is_Ey, zbound, tbound).T, axis=0)

    if zbound == 0:
        zbound = input_field.prop.z_vals_output.max()
    if tbound == 0:
        tbound = input_field.prop.times.max()

    plt.imshow(Et_vals.real, aspect='auto', extent=[tbound, -tbound, -zbound, zbound], **kwargs)
    plt.ylabel("Z Axis [um]")
    plt.xlabel("Time Axis [fs]")
    E_type = "Ey"
    if not is_Ey:
        E_type = "Ez"
    plt.title("Output " + E_type + " Field Cross Section at Y: " + f"{y_val:.2f}" + " um")

def plot_Ew_output_ZW(input_field: laser_input.InputField, y_val, is_Ey=True, zbound=0, wbound=0, **kwargs):
    Ew_vals = np.flip(get_Ew_ZW(input_field, y_val, is_Ey, zbound, wbound).T, axis=0)

    if zbound == 0:
        zbound = input_field.prop.z_vals_output.max()
    if wbound == 0:
        wbound = input_field.prop.omegas.max()

    plt.imshow(Ew_vals.real, aspect='auto', extent=[-wbound, wbound, -zbound, zbound], **kwargs)
    plt.ylabel("Z Axis [um]")
    plt.xlabel("Frequency Axis [rad/PHz]")
    E_type = "Ey"
    if not is_Ey:
        E_type = "Ez"
    plt.title("Output " + E_type + " Field Cross Section at Y: " + f"{y_val:.2f}" + " um")

def plot_Ew_input_ZW(input_field: laser_input.InputField, y_val, is_Ey=True, zbound=0, wbound=0, **kwargs):
    Ew_vals = np.flip(get_Ew_ZW(input_field, y_val, is_Ey, zbound, wbound, False).T, axis=0)

    if zbound == 0:
        zbound = input_field.prop.z_vals_input.max()
    if wbound == 0:
        wbound = input_field.prop.omegas.max()

    plt.imshow(Ew_vals.real, aspect='auto', extent=[-wbound, wbound, -zbound, zbound], **kwargs)
    plt.ylabel("Z Axis [um]")
    plt.xlabel("Frequency Axis [rad/PHz]")
    E_type = "Ey"
    if not is_Ey:
        E_type = "Ez"
    plt.title("Input " + E_type + " Field Cross Section at Y: " + f"{y_val:.2f}" + " um")


def get_Et_transverse(input_field: laser_input.InputField, time, is_Ey=True, ybound=0, zbound=0, is_output=True):
    time_index = np.argmin(np.abs(input_field.prop.times - time))

    E_vals = _get_E_vals(input_field, is_output, False, is_Ey, time_index, 1)

    y_start, y_end = _get_default_bounds(ybound, is_output,
                                         input_field.prop.y_vals_input, input_field.prop.y_vals_output)

    z_start, z_end = _get_default_bounds(zbound, is_output,
                                         input_field.prop.z_vals_input, input_field.prop.z_vals_output)

    return E_vals[y_start:y_end, z_start:z_end]


def get_Ew_transverse(input_field: laser_input.InputField, w, is_Ey=True, ybound=0, zbound=0, is_output=True):
    omega_index = np.argmin(np.abs(input_field.prop.omegas - w))

    E_vals = _get_E_vals(input_field, is_output, True, is_Ey, omega_index, 0)

    y_start, y_end = _get_default_bounds(ybound, is_output,
                                         input_field.prop.y_vals_input, input_field.prop.y_vals_output)

    z_start, z_end = _get_default_bounds(zbound, is_output,
                                         input_field.prop.z_vals_input, input_field.prop.z_vals_output)

    return E_vals[y_start:y_end, z_start:z_end]


def get_Et_YT(input_field: laser_input.InputField, z_val, is_Ey=True, ybound=0, tbound=0, is_output=True):

    if is_output:
        z_index = np.argmin(np.abs(input_field.prop.z_vals_output - z_val))
    else:
        z_index = np.argmin(np.abs(input_field.prop.z_vals_input - z_val))

    E_vals = _get_E_vals(input_field, is_output, False, is_Ey, z_index, 2)

    y_start, y_end = _get_default_bounds(ybound, is_output,
                                         input_field.prop.y_vals_input, input_field.prop.y_vals_output)

    t_start, t_end = _get_default_bounds(tbound, is_output,
                                         input_field.prop.times, input_field.prop.times)

    return E_vals[y_start:y_end, t_start:t_end].T


def get_Ew_YW(input_field: laser_input.InputField, z_val, is_Ey=True, ybound=0, wbound=0, is_output=True):

    if is_output:
        z_index = np.argmin(np.abs(input_field.prop.z_vals_output - z_val))
    else:
        z_index = np.argmin(np.abs(input_field.prop.z_vals_input - z_val))

    E_vals = _get_E_vals(input_field, is_output, True, is_Ey, z_index, 2)

    y_start, y_end = _get_default_bounds(ybound, is_output,
                                         input_field.prop.y_vals_input, input_field.prop.y_vals_output)

    w_start, w_end = _get_default_bounds(wbound, is_output,
                                         input_field.prop.omegas, input_field.prop.omegas)

    return E_vals[w_start:w_end, y_start:y_end]


def get_Et_ZT(input_field: laser_input.InputField, y_val, is_Ey=True, zbound=0, tbound=0, is_output=True):
    if is_output:
        y_index = np.argmin(np.abs(input_field.prop.y_vals_output - y_val))
    else:
        y_index = np.argmin(np.abs(input_field.prop.y_vals_input - y_val))

    E_vals = _get_E_vals(input_field, is_output, False, is_Ey, y_index, 0)

    z_start, z_end = _get_default_bounds(zbound, is_output,
                                         input_field.prop.z_vals_input, input_field.prop.z_vals_output)
    t_start, t_end = _get_default_bounds(tbound, is_output,
                                         input_field.prop.times, input_field.prop.times)

    return E_vals[t_start:t_end, z_start:z_end]


def get_Ew_ZW(input_field: laser_input.InputField, y_val, is_Ey=True, zbound=0, wbound=0, is_output=True):
    if is_output:
        y_index = np.argmin(np.abs(input_field.prop.y_vals_output - y_val))
    else:
        y_index = np.argmin(np.abs(input_field.prop.y_vals_input - y_val))

    E_vals = _get_E_vals(input_field, is_output, True, is_Ey, y_index, 1)

    z_start, z_end = _get_default_bounds(zbound, is_output,
                                         input_field.prop.z_vals_input, input_field.prop.z_vals_output)

    w_start, w_end = _get_default_bounds(wbound, is_output, input_field.prop.omegas, input_field.prop.omegas)

    return E_vals[w_start:w_end, z_start:z_end]

# THIS IS NOT POLISHED AT ALL
# TODO: Make sure time axis is correct (after flippening)
def plot_Et_sim_YT(input_field: laser_input.InputField, z_val, is_Ey=True, **kwargs):
    sim_params = sim_grid_parameters.compute_sim_grid(
        input_field.prop.times, input_field.prop.y_vals_output, input_field.prop.z_vals_output
    )
    ref_freq = (2*np.pi*constants.C_SPEED) / (input_field.laser.wavelength * 1.e-6)

    z_val_code_units = utils.microns_to_norm_units(z_val, ref_freq)
    # TODO: the index needs to account for clipping
    z_vals_clipped = sim_params.sim_z_vals_code_units[
                     sim_params.z_indexes.lo_index_sim:sim_params.z_indexes.hi_index_sim + 1
                     ]
    z_index = np.argmin(np.abs(z_vals_clipped - z_val_code_units))

    if is_Ey:
        Et_file = read_laser.get_Et_sim_file_y(
            input_field, sim_params.num_t_vals,
            sim_params.num_y_vals,
            sim_params.num_z_vals
        )
    else:
        Et_file = read_laser.get_Et_sim_file_z(
            input_field, sim_params.num_t_vals,
            sim_params.num_y_vals,
            sim_params.num_z_vals
        )

    E_vals = np.flip(Et_file[:, :, z_index], axis=0)

    t_lo_val = sim_params.sim_times_fs[sim_params.t_indexes.lo_index_sim]
    t_hi_val = sim_params.sim_times_fs[sim_params.t_indexes.hi_index_sim]

    y_lo_val = sim_params.sim_y_vals_um[sim_params.y_indexes.lo_index_sim]
    y_hi_val = sim_params.sim_y_vals_um[sim_params.y_indexes.hi_index_sim]

    plt.imshow(E_vals.real, aspect='auto',
               extent=[t_lo_val, t_hi_val, y_lo_val, y_hi_val],
               interpolation="None", **kwargs)
    plt.ylabel("Y Axis [um]")
    plt.xlabel("T Axis [fs]")
    E_type = "Ey"
    if not is_Ey:
        E_type = "Ez"
    plt.title("Sim " + E_type + " Field Cross Section at Z: " + f"{z_val:.2f}" + " um")

def plot_Et_sim_ZT(input_field: laser_input.InputField, y_val, is_Ey=True, **kwargs):
    sim_params = sim_grid_parameters.compute_sim_grid(
        input_field.prop.times, input_field.prop.y_vals_output, input_field.prop.z_vals_output
    )
    ref_freq = (2*np.pi*constants.C_SPEED) / (input_field.laser.wavelength * 1.e-6)

    y_val_code_units = utils.microns_to_norm_units(y_val, ref_freq)
    y_vals_clipped = sim_params.sim_y_vals_code_units[
                     sim_params.y_indexes.lo_index_sim:sim_params.y_indexes.hi_index_sim + 1
                     ]
    y_index = np.argmin(np.abs(y_vals_clipped - y_val_code_units))

    if is_Ey:
        Et_file = read_laser.get_Et_sim_file_y(
            input_field, sim_params.num_t_vals,
            sim_params.num_y_vals,
            sim_params.num_z_vals
        )
    else:
        Et_file = read_laser.get_Et_sim_file_z(
            input_field, sim_params.num_t_vals,
            sim_params.num_y_vals,
            sim_params.num_z_vals
        )

    E_vals = np.flip(Et_file[y_index, :, :].T, axis=0)
    t_lo_val = sim_params.sim_times_fs[sim_params.t_indexes.lo_index_sim]
    t_hi_val = sim_params.sim_times_fs[sim_params.t_indexes.hi_index_sim]

    z_lo_val = sim_params.sim_z_vals_um[sim_params.z_indexes.lo_index_sim]
    z_hi_val = sim_params.sim_z_vals_um[sim_params.z_indexes.hi_index_sim]

    plt.imshow(E_vals.real, aspect='auto',
               extent=[t_lo_val, t_hi_val, z_lo_val, z_hi_val],
               interpolation="None", **kwargs)
    plt.ylabel("Z Axis [um]")
    plt.xlabel("T Axis [fs]")
    E_type = "Ey"
    if not is_Ey:
        E_type = "Ez"
    plt.title("Sim " + E_type + " Field Cross Section at Z: " + f"{y_val:.2f}" + " um")

def plot_Et_sim_YZ(input_field: laser_input.InputField, t_val, is_Ey=True, **kwargs):
    sim_params = sim_grid_parameters.compute_sim_grid(
        input_field.prop.times, input_field.prop.y_vals_output, input_field.prop.z_vals_output
    )
    ref_freq = (2*np.pi*constants.C_SPEED) / (input_field.laser.wavelength * 1.e-6)

    t_val_code_units = utils.fs_to_norm_units(t_val, ref_freq)
    t_vals_clipped = sim_params.sim_times_code_units[
                     sim_params.t_indexes.lo_index_sim:sim_params.t_indexes.hi_index_sim + 1
                     ]
    t_index = np.argmin(np.abs(t_vals_clipped - t_val_code_units))

    if is_Ey:
        Et_file = read_laser.get_Et_sim_file_y(
            input_field, sim_params.num_t_vals,
            sim_params.num_y_vals,
            sim_params.num_z_vals
        )
    else:
        Et_file = read_laser.get_Et_sim_file_z(
            input_field, sim_params.num_t_vals,
            sim_params.num_y_vals,
            sim_params.num_z_vals
        )

    E_vals = np.flip(Et_file[:, t_index, :].T, axis=0)

    y_lo_val = sim_params.sim_y_vals_um[sim_params.y_indexes.lo_index_sim]
    y_hi_val = sim_params.sim_y_vals_um[sim_params.y_indexes.hi_index_sim]

    z_lo_val = sim_params.sim_z_vals_um[sim_params.z_indexes.lo_index_sim]
    z_hi_val = sim_params.sim_z_vals_um[sim_params.z_indexes.hi_index_sim]

    # TODO: Make sure axes are correct
    plt.imshow(E_vals.real, aspect='auto',
               extent=[z_lo_val, z_hi_val, y_lo_val, y_hi_val],
               interpolation="None", **kwargs)
    plt.ylabel("Y Axis [um]")
    plt.xlabel("Z Axis [um]")
    E_type = "Ey"
    if not is_Ey:
        E_type = "Ez"
    plt.title("Sim " + E_type + " Field Cross Section at T: " + f"{t_val:.2f}" + " fs")


def _get_default_bounds(bound, is_output, arr_input, arr_output):
    if is_output:
        arr = arr_output
    else:
        arr = arr_input

    if bound == 0:
        start = 0
        end = len(arr)
    else:
        start, end = get_trim_indexes(arr, -bound, bound)
    return start, end


def _get_E_vals(input_field, is_output, is_freq_space, is_Ey, index, axis):
    if input_field.prop.spatial_dimensions == 1:
        if input_field.prop.save_data_as_files:
            if is_output:
                if is_freq_space:
                    if is_Ey:
                        E_2d = input_field.get_output_Ew_field_file_y_2d()
                    else:
                        E_2d = input_field.get_output_Ew_field_file_z_2d()
                else:
                    if is_Ey:
                        E_2d = input_field.get_output_Et_field_file_y_2d()
                    else:
                        E_2d = input_field.get_output_Et_field_file_z_2d()
            else:
                if is_freq_space:
                    if is_Ey:
                        E_2d = input_field.get_input_Ew_field_file_y_2d()
                    else:
                        E_2d = input_field.get_input_Ew_field_file_z_2d()
                else:
                    if is_Ey:
                        E_2d = input_field.get_input_Et_field_file_y_2d()
                    else:
                        E_2d = input_field.get_input_Et_field_file_z_2d()
        else:
            if is_output:
                if is_freq_space:
                    if is_Ey:
                        E_2d = input_field.output_Ew_field_y
                    else:
                        E_2d = input_field.output_Ew_field_z
                else:
                    if is_Ey:
                        E_2d = input_field.output_Et_field_y
                    else:
                        E_2d = input_field.output_Et_field_z
            else:
                if is_freq_space:
                    if is_Ey:
                        E_2d = input_field.input_Ew_field_y
                    else:
                        E_2d = input_field.input_Ew_field_z
                else:
                    if is_Ey:
                        E_2d = input_field.input_Et_field_y
                    else:
                        E_2d = input_field.input_Et_field_z
        if axis == 0:
            E_vals = E_2d[index, :]
        elif axis == 1:
            E_vals = E_2d[:, index]
        elif axis == 2:
            E_vals = E_2d[:, :]
    else:
        if input_field.prop.save_data_as_files:
            if is_output:
                if is_freq_space:
                    if is_Ey:
                        E_3d = input_field.get_output_Ew_field_file_y()
                    else:
                        E_3d = input_field.get_output_Ew_field_file_z()
                else:
                    if is_Ey:
                        E_3d = input_field.get_output_Et_field_file_y()
                    else:
                        E_3d = input_field.get_output_Et_field_file_z()
            else:
                if is_freq_space:
                    if is_Ey:
                        E_3d = input_field.get_input_Ew_field_file_y()
                    else:
                        E_3d = input_field.get_input_Ew_field_file_z()
                else:
                    if is_Ey:
                        E_3d = input_field.get_input_Et_field_file_y()
                    else:
                        E_3d = input_field.get_input_Et_field_file_z()
        else:
            if is_output:
                if is_freq_space:
                    if is_Ey:
                        E_3d = input_field.output_Ew_field_y
                    else:
                        E_3d = input_field.output_Ew_field_z
                else:
                    if is_Ey:
                        E_3d = input_field.output_Et_field_y
                    else:
                        E_3d = input_field.output_Et_field_z
            else:
                if is_freq_space:
                    if is_Ey:
                        E_3d = input_field.input_Ew_field_y
                    else:
                        E_3d = input_field.input_Ew_field_z
                else:
                    if is_Ey:
                        E_3d = input_field.input_Et_field_y
                    else:
                        E_3d = input_field.input_Et_field_z


        if axis == 0:
            E_vals = E_3d[index, :, :]
        elif axis == 1:
            E_vals = E_3d[:, index, :]
        elif axis == 2:
            E_vals = E_3d[:, :, index]
        else:
            raise Exception()

    return E_vals


def get_trim_indexes(arr, start_val, end_val):
    start_index = np.argmin(np.abs(arr - start_val))
    end_index = np.argmin(np.abs(arr - end_val))

    return start_index, end_index