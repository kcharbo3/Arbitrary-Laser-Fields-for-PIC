import fourier_prop.laser_input.input_laser_field as laser_input
import fourier_prop.laser_input.constants as constants
import fourier_prop.laser_input.utils as utils
from fourier_prop.read_laser import sim_grid_parameters as grid
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pickle


def gouy_phase(z, zr):
    return np.arctan2(z, zr)

def radius_of_curvature(z, zr):
    if z == 0:
        return 1e30
    return z + ((zr**2) / z)

def waist_func(waist0, z, zr):
    return waist0 * np.sqrt(1 + (z/zr)**2)

def gaussian_space(w, z, waist0, x, y):
    lam = (2*np.pi*constants.C_UM_FS) / w
    k = w / constants.C_UM_FS
    zr = (np.pi * waist0**2) / lam
    gouy = gouy_phase(z, zr)
    R = radius_of_curvature(z, zr)
    waist = waist_func(waist0, z, zr)
    
    first_term = np.exp(-1.0j*k*z + 1.0j*gouy) / waist
    second_term = np.exp(-(x**2 + y**2) / waist**2)
    third_term = np.exp(-1.0j * k * (x**2 + y**2)/(2 * R))
    
    return first_term*second_term*third_term

def time_envelope(fwhm, t, t_start):
    sigma = (0.5*fwhm)**2/np.log(2.0)
    return np.exp( -(t - t_start)**2 / sigma )

def siegman_pulse_transverse(t, y, z, omega, waist0, distance_from_focus, fwhm, t_start=0, y_center=0, z_center=0):
    lam = (2*np.pi*constants.C_UM_FS) / omega
    k = omega / constants.C_UM_FS
    zr = (np.pi * waist0**2) / lam
    waist = waist_func(waist0, distance_from_focus, zr)
    R = radius_of_curvature(distance_from_focus, zr)
    
    prop_term = np.exp(-1.0j * omega * t)
    gouy_term = np.exp(1.0j * np.arctan(distance_from_focus / zr))
    #gouy_term = 1
    spatial_term = np.exp(-(1 / waist**2) * ((y - y_center)**2 + (z - z_center)**2))
    radius_term = np.exp(-1.0j * k * ((y - y_center)**2 + (z - z_center)**2) * (1 / (2 * R)))
    
    return (1 / waist) * prop_term * gouy_term * spatial_term * radius_term * time_envelope(fwhm, t, t_start) * np.exp(1.j * (np.pi) * .5)



'''
Needs to:
1. Center E field array to time=0 X
2. Normalize E field X
3. Convert E field arrays to B field arrays
2. Interpolate to sim params
'''
def get_By_function(data_directory_path, grid_params: grid.SimGridParameters):



    def by_profile(y, z, t):
        # By = -Ez
        return 0.

    return by_profile

def get_Bz_function(data_directory_path, grid_params: grid.SimGridParameters):
    t_start_fs = 75. - 0.04002769142377824 - 0.02334948666387064
    y_center = 14.
    z_center = 14.
    WAVELENGTH = 1.  # um
    REF_FREQ = (2*np.pi*constants.C_SPEED) / (WAVELENGTH * 1.e-6)
    OMEGA0 = REF_FREQ * 1e-15  # rad / PHz
    SPOT_SIZE = 4.
    DISTANCE_FROM_FOCUS = -50.
    FWHM = 25. * (24.588742233055843 / 24.54557434350688) * (24.553313764862047 / 24.547112098203467) * np.sqrt(2)
    

    def bz_profile(y, z, t):
        t_fs = utils.norm_units_to_fs(t, REF_FREQ)
        y_um = utils.norm_units_to_microns(y, REF_FREQ)
        z_um = utils.norm_units_to_microns(z, REF_FREQ)
        e_val = np.exp(-2*np.pi*.1237*1j) * 100. * siegman_pulse_transverse(t_fs, y_um, z_um, OMEGA0, SPOT_SIZE, -DISTANCE_FROM_FOCUS, FWHM, t_start=t_start_fs, y_center=y_center, z_center=z_center)
        # Bz = +Ey

        #print("E:", e_val)
        return e_val

    return bz_profile

def compute_field_at_sim_grid(input_field: laser_input.InputField, comm, rank, num_processes, verbose=False):
    sim_grid_parameters = grid.compute_sim_grid(
        input_field.prop.times, input_field.prop.y_vals_output, input_field.prop.z_vals_output
    )

    if verbose and rank == 0:
        _print_parameters(sim_grid_parameters)

    comm.Barrier()
    interp_file_to_lo_val = \
        create_interpolation_functions(input_field, sim_grid_parameters, comm, rank, num_processes)

    if rank == 0:
        create_Et_sim_files(
            input_field, sim_grid_parameters.num_t_vals, sim_grid_parameters.num_y_vals,
            sim_grid_parameters.num_z_vals
        )

    comm.Barrier()

    Et_sim_y = get_Et_sim_file_y(
        input_field, sim_grid_parameters.num_t_vals, sim_grid_parameters.num_y_vals,
        sim_grid_parameters.num_z_vals
    )

    Et_sim_z = get_Et_sim_file_z(
        input_field, sim_grid_parameters.num_t_vals, sim_grid_parameters.num_y_vals,
        sim_grid_parameters.num_z_vals
    )

    t_index_lo = sim_grid_parameters.t_indexes.lo_index_sim
    t_index_hi = sim_grid_parameters.t_indexes.hi_index_sim

    y_index_lo = sim_grid_parameters.y_indexes.lo_index_sim
    y_index_hi = sim_grid_parameters.y_indexes.hi_index_sim
    y_index_lo_half = sim_grid_parameters.y_indexes.lo_index_sim_half
    y_index_hi_half = sim_grid_parameters.y_indexes.hi_index_sim_half

    z_index_lo = sim_grid_parameters.z_indexes.lo_index_sim
    z_index_hi = sim_grid_parameters.z_indexes.hi_index_sim
    z_index_lo_half = sim_grid_parameters.z_indexes.lo_index_sim_half
    z_index_hi_half = sim_grid_parameters.z_indexes.hi_index_sim_half

    if rank == 0:
        print("interp_file_to_lo_val:", interp_file_to_lo_val)
    if rank == 0:
        y_start = y_index_lo
        y_start_half = y_index_lo_half
    else:
        y_start = np.argmin(np.abs(sim_grid_parameters.sim_y_vals_code_units - interp_file_to_lo_val[rank]))
        y_start_half = np.argmin(np.abs(sim_grid_parameters.sim_y_vals_code_units_half - interp_file_to_lo_val[rank]))
    if rank == num_processes - 1:
        y_end = y_index_hi + 1
        y_end_half = y_index_hi_half + 1
    else:
        y_end = np.argmin(np.abs(sim_grid_parameters.sim_y_vals_code_units - interp_file_to_lo_val[rank + 1]))
        y_end_half = np.argmin(np.abs(sim_grid_parameters.sim_y_vals_code_units_half - interp_file_to_lo_val[rank + 1]))

    print("GETTING Ey. RANK:", rank, "Y INDICES:", y_start_half, y_end_half, "\nY_VALS:",
          sim_grid_parameters.sim_y_vals_code_units_half[y_start_half],
          sim_grid_parameters.sim_y_vals_code_units_half[y_end_half - 1], "\nZ_VALS:",
          sim_grid_parameters.sim_z_vals_code_units[z_index_lo],
          sim_grid_parameters.sim_z_vals_code_units[z_index_hi], "\nT_VALS:",
          sim_grid_parameters.sim_times_code_units[t_index_lo],
          sim_grid_parameters.sim_times_code_units[t_index_hi]
          )
    y_chunk_half, t_chunk, z_chunk = \
        np.meshgrid(
            sim_grid_parameters.sim_y_vals_code_units_half[y_start_half:y_end_half],
            sim_grid_parameters.sim_times_code_units[t_index_lo:t_index_hi + 1],
            sim_grid_parameters.sim_z_vals_code_units[z_index_lo:z_index_hi + 1],
            indexing='ij'
        )

    with open(input_field.prop.data_directory_path + grid.INTERP_Y_PREFIX + str(rank) + '.pkl', 'rb') as f:
        interp_func = pickle.load(f)

    print("Rank", rank, "filling in values")
    y_start_half_sim = y_start_half - y_index_lo_half
    y_end_half_sim = y_end_half - y_index_lo_half
    Et_sim_y[y_start_half_sim:y_end_half_sim, :, :] = \
        interp_func((y_chunk_half, t_chunk, z_chunk))
    Et_sim_y.flush()
    del Et_sim_y
    del interp_func
    del t_chunk, y_chunk_half, z_chunk

    print("GETTING Ez. RANK:", rank, "\nY_VALS:",
          sim_grid_parameters.sim_y_vals_code_units[y_start],
          sim_grid_parameters.sim_y_vals_code_units[y_end - 1], "\nZ_VALS:",
          sim_grid_parameters.sim_z_vals_code_units_half[z_index_lo_half],
          sim_grid_parameters.sim_z_vals_code_units_half[z_index_hi_half], "\nT_VALS:",
          sim_grid_parameters.sim_times_code_units[t_index_lo],
          sim_grid_parameters.sim_times_code_units[t_index_hi]
          )

    y_chunk, t_chunk, z_chunk_half = \
        np.meshgrid(
            sim_grid_parameters.sim_y_vals_code_units[y_start:y_end],
            sim_grid_parameters.sim_times_code_units[t_index_lo:t_index_hi + 1],
            sim_grid_parameters.sim_z_vals_code_units_half[z_index_lo_half:z_index_hi_half + 1],
            indexing='ij'
        )

    with open(input_field.prop.data_directory_path + grid.INTERP_Z_PREFIX + str(rank) + '.pkl', 'rb') as f:
        interp_func = pickle.load(f)
    print("Rank", rank, "Ey done, now onto Ez")
    y_start_sim = y_start - y_index_lo
    y_end_sim = y_end - y_index_lo
    Et_sim_z[y_start_sim:y_end_sim, :, :] = \
        interp_func((y_chunk, t_chunk, z_chunk_half))
    Et_sim_z.flush()
    del Et_sim_z
    del interp_func
    del t_chunk, y_chunk, z_chunk_half

    comm.Barrier()


def create_interpolation_functions(input_field: laser_input.InputField, sim_grid_parameters, comm, rank, num_processes):
    if input_field.laser.normalize_to_a0:
        normalize_to_a0(input_field, comm, rank, num_processes)
    else:
        normalize_to_energy(input_field, comm, rank, num_processes, sim_grid_parameters)
    comm.Barrier()

    chunk_size, start_index, end_index = get_chunk_info(len(input_field.prop.y_vals_output), rank, num_processes)
    Ey_file = input_field.get_output_Et_field_file_y()
    Ez_file = input_field.get_output_Et_field_file_z()
    Ey_chunk = Ey_file[start_index:end_index]
    Ez_chunk = Ez_file[start_index:end_index]
    E_mag = np.sqrt(np.abs(Ey_chunk) ** 2 + np.abs(Ez_chunk) ** 2)
    dy = sim_grid_parameters.output_y_vals_code_units[1] - sim_grid_parameters.output_y_vals_code_units[0]
    dz = sim_grid_parameters.output_z_vals_code_units[1] - sim_grid_parameters.output_z_vals_code_units[0]
    dt = sim_grid_parameters.output_times_code_units[1] - sim_grid_parameters.output_times_code_units[0]
    total_sum = _get_total_energy(E_mag, comm, rank, num_processes, dy, dz, dt)
    max_a0 = _get_max_val(E_mag, comm, rank, num_processes)
    if rank == 0:
        print("Output Total Energy:", total_sum)
        print("Output Max a0:", max_a0)
    del Ey_file
    del Ez_file
    del E_mag

    comm.Barrier()
    center_at_t0(input_field, comm, rank, num_processes)

    _waitForAllToFinish(comm, rank, num_processes)

    # Only need to interpolate up to dimensions of Simulation
    t_index_lo = sim_grid_parameters.t_indexes.lo_index_output
    t_index_hi = sim_grid_parameters.t_indexes.hi_index_output

    y_index_lo = sim_grid_parameters.y_indexes.lo_index_output
    y_index_hi = sim_grid_parameters.y_indexes.hi_index_output

    z_index_lo = sim_grid_parameters.z_indexes.lo_index_output
    z_index_hi = sim_grid_parameters.z_indexes.hi_index_output

    chunk_size, start_index, end_index = get_chunk_info(y_index_hi - y_index_lo + 1, rank, num_processes)
    start_index += y_index_lo
    end_index += y_index_lo

    Et_file_y = input_field.get_output_Et_field_file_y()
    Et_file_z = input_field.get_output_Et_field_file_z()

    # give some buffer to start and end indexes
    if start_index >= 5:
        start_index -= 5
    else:
        start_index = 0

    if end_index != len(sim_grid_parameters.output_y_vals_code_units):
        end_index += 5

    y_0_val = sim_grid_parameters.center_y_code_units
    z_0_val = sim_grid_parameters.center_z_code_units
    t_0_val = sim_grid_parameters.laser_time_start_code_units

    interp_file_to_lo_val = np.zeros((num_processes))

    for i in range(num_processes):
        _, start_index_curr, _ = get_chunk_info(y_index_hi - y_index_lo + 1, i, num_processes)
        interp_file_to_lo_val[i] = sim_grid_parameters.output_y_vals_code_units[start_index_curr + y_index_lo] + y_0_val

    print("INTERP. RANK:", rank, "\nY_VALS:",
          sim_grid_parameters.output_y_vals_code_units[start_index] + y_0_val,
          sim_grid_parameters.output_y_vals_code_units[end_index - 1] + y_0_val, "\nZ_VALS:",
          sim_grid_parameters.output_z_vals_code_units[z_index_lo] + z_0_val,
          sim_grid_parameters.output_z_vals_code_units[z_index_hi] + z_0_val, "\nT_VALS:",
          sim_grid_parameters.output_times_code_units[t_index_lo] + t_0_val,
          sim_grid_parameters.output_times_code_units[t_index_hi] + t_0_val
          )

    points = (
        sim_grid_parameters.output_y_vals_code_units[start_index:end_index] + sim_grid_parameters.center_y_code_units,
        sim_grid_parameters.output_times_code_units[t_index_lo:t_index_hi + 1]
        + sim_grid_parameters.laser_time_start_code_units,
        sim_grid_parameters.output_z_vals_code_units[z_index_lo:z_index_hi + 1] + sim_grid_parameters.center_z_code_units
    )

    Et_chunk_y = Et_file_y[start_index:end_index, t_index_lo:t_index_hi + 1, z_index_lo:z_index_hi + 1]
    interpolation_func = RegularGridInterpolator(points, Et_chunk_y.real, bounds_error=False, fill_value=0, method='cubic')
    del Et_file_y
    with open(input_field.prop.data_directory_path + grid.INTERP_Y_PREFIX + str(rank) + '.pkl', 'wb') as f:
        pickle.dump(interpolation_func, f)

    Et_chunk_z = Et_file_z[start_index:end_index, t_index_lo:t_index_hi + 1, z_index_lo:z_index_hi + 1]
    interpolation_func = RegularGridInterpolator(points, Et_chunk_z.real, bounds_error=False, fill_value=0, method='cubic')
    del Et_file_z
    with open(input_field.prop.data_directory_path + grid.INTERP_Z_PREFIX + str(rank) + '.pkl', 'wb') as f:
        pickle.dump(interpolation_func, f)

    return interp_file_to_lo_val

def create_Et_sim_files(input_field: laser_input.InputField, num_t_vals, num_y_vals, num_z_vals):
    np.memmap(
        input_field.prop.data_directory_path + constants.OUTPUT_ET_SIM_FILE_Y, dtype='complex64',
        mode='w+', shape=(num_y_vals, num_t_vals, num_z_vals)
    )

    np.memmap(
        input_field.prop.data_directory_path + constants.OUTPUT_ET_SIM_FILE_Z, dtype='complex64',
        mode='w+', shape=(num_y_vals, num_t_vals, num_z_vals)
    )

def get_Et_sim_file_y(input_field: laser_input.InputField, num_t_vals, num_y_vals, num_z_vals):
    return np.memmap(
        input_field.prop.data_directory_path + constants.OUTPUT_ET_SIM_FILE_Y, dtype='complex64',
        mode='r+', shape=(num_y_vals, num_t_vals, num_z_vals)
    )


def get_Et_sim_file_z(input_field: laser_input.InputField, num_t_vals, num_y_vals, num_z_vals):
    return np.memmap(
        input_field.prop.data_directory_path + constants.OUTPUT_ET_SIM_FILE_Z, dtype='complex64',
        mode='r+', shape=(num_y_vals, num_t_vals, num_z_vals)
    )


def center_at_t0(input_field: laser_input.InputField, comm, rank, num_processes):
    chunk_size, start_index, end_index = get_chunk_info(len(input_field.prop.y_vals_output), rank, num_processes)

    Ey_file = input_field.get_output_Et_field_file_y()
    Ez_file = input_field.get_output_Et_field_file_z()

    # Time axis needs to be flipped... not 100% sure why yet
    Ey_chunk = np.flip(Ey_file[start_index:end_index, :, :], axis=1)
    Ez_chunk = np.flip(Ez_file[start_index:end_index, :, :], axis=1)

    E_mag = np.sqrt(np.abs(Ey_chunk) ** 2 + np.abs(Ez_chunk) ** 2)
    #max_val, index = _get_max_val_with_index(E_mag, comm, rank, num_processes)

    #shift_count = (int(len(input_field.prop.times) / 2.) - index)

    #Ey_chunk = np.roll(Ey_chunk, shift_count, axis=1)
    Ey_file[start_index:end_index, :, :] = Ey_chunk
    Ey_file.flush()
    del Ey_file

    #Ez_chunk = np.roll(Ez_chunk, shift_count, axis=1)
    Ez_file[start_index:end_index, :, :] = Ez_chunk
    Ez_file.flush()
    del Ez_file


def normalize_to_a0(input_field: laser_input.InputField, comm, rank, num_processes):
    chunk_size, start_index, end_index = get_chunk_info(len(input_field.prop.y_vals_output), rank, num_processes)

    Ey_file = input_field.get_output_Et_field_file_y()
    Ez_file = input_field.get_output_Et_field_file_z()

    Ey_chunk = Ey_file[start_index:end_index]
    Ez_chunk = Ez_file[start_index:end_index]

    E_mag = np.sqrt(np.abs(Ey_chunk) ** 2 + np.abs(Ez_chunk) ** 2)

    max_val = _get_max_val(E_mag, comm, rank, num_processes)

    Ey_file[start_index:end_index] = Ey_file[start_index:end_index] * (input_field.laser.peak_a0 / max_val)
    Ey_file.flush()
    del Ey_file

    Ez_file[start_index:end_index] = Ez_file[start_index:end_index] * (input_field.laser.peak_a0 / max_val)
    Ez_file.flush()
    del Ez_file


def normalize_to_energy(input_field: laser_input.InputField, comm, rank, num_processes, sim_grid_parameters):
    chunk_size, start_index, end_index = get_chunk_info(len(input_field.prop.y_vals_output), rank, num_processes)

    Ey_file = input_field.get_output_Et_field_file_y()
    Ez_file = input_field.get_output_Et_field_file_z()

    Ey_chunk = Ey_file[start_index:end_index]
    Ez_chunk = Ez_file[start_index:end_index]

    E_mag = np.sqrt(np.abs(Ey_chunk) ** 2 + np.abs(Ez_chunk) ** 2)

    dy = sim_grid_parameters.output_y_vals_code_units[1] - sim_grid_parameters.output_y_vals_code_units[0]
    dz = sim_grid_parameters.output_z_vals_code_units[1] - sim_grid_parameters.output_z_vals_code_units[0]
    dt = sim_grid_parameters.output_times_code_units[1] - sim_grid_parameters.output_times_code_units[0]

    total_sum = _get_total_energy(E_mag, comm, rank, num_processes, dy, dz, dt)
    if rank == 0:
        print("TOTAL SUM:", total_sum, "TOTAL ENERGY:", input_field.laser.total_energy)
    Ey_file[start_index:end_index] = \
        Ey_file[start_index:end_index] * np.sqrt(input_field.laser.total_energy / total_sum)
    Ey_file.flush()
    del Ey_file

    Ez_file[start_index:end_index] = \
        Ez_file[start_index:end_index] * np.sqrt(input_field.laser.total_energy / total_sum)
    Ez_file.flush()
    del Ez_file


def _get_max_val(E_mag, comm, rank, num_processes):
    max_val = E_mag.max()

    # Rank 0 compares and finds max value
    if rank != 0:
        comm.send(max_val, dest=0)
    else:
        for i in range(num_processes - 1):
            temp_max = comm.recv(source=i+1)
            if temp_max > max_val:
                max_val = temp_max

    # Rank 0 then sends max value to rest of threads
    if rank == 0:
        for i in range(num_processes - 1):
            comm.send(max_val, dest=(i + 1))
    else:
        max_val = comm.recv(source=0)

    return max_val


def _get_max_val_with_index(E_mag, comm, rank, num_processes):
    max_val = E_mag.max()
    # Want the t index
    index = np.unravel_index(np.argmax(E_mag), E_mag.shape)[1]

    # Rank 0 compares and finds max value
    if rank != 0:
        comm.send((max_val, index), dest=0)
    else:
        for i in range(num_processes - 1):
            (temp_max_val, temp_index) = comm.recv(source=i + 1)
            if temp_max_val > max_val:
                max_val = temp_max_val
                index = temp_index

    # Rank 0 then sends max value to rest of threads
    if rank == 0:
        for i in range(num_processes - 1):
            comm.send((max_val, index), dest=(i + 1))
    else:
        (max_val, index) = comm.recv(source=0)

    return max_val, index


def _get_total_energy(E_mag, comm, rank, num_processes, dy, dz, dt):
    total_sum = (E_mag ** 2).sum() * (dy * dz * dt)

    # Rank 0 compares and finds max value
    if rank != 0:
        comm.send(total_sum, dest=0)
    else:
        for i in range(num_processes - 1):
            chunk_sum = comm.recv(source=i+1)
            total_sum += chunk_sum

    # Rank 0 then sends max value to rest of threads
    if rank == 0:
        for i in range(num_processes - 1):
            comm.send(total_sum, dest=(i + 1))
    else:
        total_sum = comm.recv(source=0)

    return total_sum

def get_chunk_start_index(total_size, rank, num_processes):
    base_chunk_size = int(total_size / num_processes)

    remainder = total_size % num_processes
    wrapped_chunk_size = base_chunk_size + 1
    total_till_no_remainder = wrapped_chunk_size * remainder

    if rank < remainder:
        start_index = wrapped_chunk_size * rank
    else:
        start_index = total_till_no_remainder + ((rank - remainder) * base_chunk_size)

    return start_index

def get_chunk_info(total_size, rank, num_processes):
    start_index = get_chunk_start_index(total_size, rank, num_processes)

    if rank == num_processes - 1:
        end_index = total_size
    else:
        end_index = get_chunk_start_index(total_size, rank + 1, num_processes)

    chunk_size = end_index - start_index
    return chunk_size, start_index, end_index

def _waitForAllToFinish(comm, rank, num_processes):
    if num_processes == 1:
        return
    if rank != 0:
        comm.send(1, dest=0)
        comm.recv(source=0)
    else:
        for i in range(1, num_processes):
            comm.recv(source=i)
        for i in range(1, num_processes):
            comm.send(1, dest=i)

def _print_parameters(sim_grid_parameters):
    print("#### Sim Resolution ####")
    print("DTime:", sim_grid_parameters.sim_times_fs[1] - sim_grid_parameters.sim_times_fs[0], "fs")
    print("DY:", sim_grid_parameters.sim_y_vals_um[1] - sim_grid_parameters.sim_y_vals_um[0], "um")
    print("DZ:", sim_grid_parameters.sim_z_vals_um[1] - sim_grid_parameters.sim_z_vals_um[0], "um")

    print("#### Sim Bounds Code Units ####")
    print("Time:", sim_grid_parameters.sim_times_code_units.max())
    print("Y:", sim_grid_parameters.sim_y_vals_code_units.max())
    print("Z:", sim_grid_parameters.sim_z_vals_code_units.max())

    print("#### Sim Field Bounds Code Units ####")
    t_indexes = sim_grid_parameters.t_indexes
    y_indexes = sim_grid_parameters.y_indexes
    z_indexes = sim_grid_parameters.z_indexes
    print("Time:", sim_grid_parameters.sim_times_code_units[t_indexes.lo_index_sim],
          sim_grid_parameters.sim_times_code_units[t_indexes.hi_index_sim])
    print("Y:", sim_grid_parameters.sim_y_vals_code_units[y_indexes.lo_index_sim],
          sim_grid_parameters.sim_y_vals_code_units[y_indexes.hi_index_sim])
    print("Z:", sim_grid_parameters.sim_z_vals_code_units[z_indexes.lo_index_sim],
          sim_grid_parameters.sim_z_vals_code_units[z_indexes.hi_index_sim])
