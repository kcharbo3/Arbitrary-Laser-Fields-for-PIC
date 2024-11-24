import fourier_prop.laser_input.input_laser_field as laser_input
import fourier_prop.laser_input.constants as constants
import numpy as np

def generate_output_Ew_field(input_field: laser_input.InputField, comm=None, rank=0, num_processes=1):
    if comm is not None:
        comm.Barrier()
    if input_field.prop.save_data_as_files:
        _generate_output_Ew_field_as_file(input_field, comm, rank, num_processes)
    else:
        _generate_output_Ew_field_in_memory(input_field, comm, rank, num_processes)


def generate_output_Et_field_from_Ew(input_field: laser_input.InputField, comm=None, rank=0, num_processes=1):
    if comm is not None:
        comm.Barrier()
    if input_field.prop.save_data_as_files:
        _generate_output_Et_field_as_file(input_field, comm, rank, num_processes)
    else:
        _generate_output_Et_field_in_memory(input_field, comm, rank, num_processes)


def generate_output_Et_field(input_field: laser_input.InputField, comm=None, rank=0, num_processes=1):
    generate_output_Ew_field(input_field, comm, rank, num_processes)
    if comm is not None:
        comm.Barrier()
    if input_field.prop.save_data_as_files:
        _generate_output_Et_field_as_file(input_field, comm, rank, num_processes)
    else:
        _generate_output_Et_field_in_memory(input_field, comm, rank, num_processes)


# Private Functions

def _generate_output_Ew_field_in_memory(input_field: laser_input.InputField, comm, rank, num_processes):
    chunk_size, start_index, end_index = laser_input.get_chunk_info(len(input_field.prop.omegas), rank, num_processes)
    omega_indexes = np.arange(start_index, end_index)

    if input_field.prop.spatial_dimensions == 2:
        input_field.output_Ew_field_y = np.zeros((
            len(input_field.prop.omegas), len(input_field.prop.y_vals_output), len(input_field.prop.z_vals_output)
        ), dtype=np.complex64)

        input_field.output_Ew_field_z = np.zeros((
            len(input_field.prop.omegas), len(input_field.prop.y_vals_output), len(input_field.prop.z_vals_output)
        ), dtype=np.complex64)
    else:
        input_field.output_Ew_field_y = np.zeros((
            len(input_field.prop.omegas), len(input_field.prop.y_vals_output)
        ), dtype=np.complex64)

        input_field.output_Ew_field_z = np.zeros((
            len(input_field.prop.omegas), len(input_field.prop.y_vals_output)
        ), dtype=np.complex64)

    if input_field.prop.low_mem:
        if input_field.prop.spatial_dimensions == 2:
            _propagate_frequencies_low_mem(input_field, omega_indexes, input_field.output_Ew_field_y, True)
            _propagate_frequencies_low_mem(input_field, omega_indexes, input_field.output_Ew_field_z, False)
        else:
            _propagate_frequencies_low_mem_2d(input_field, omega_indexes, input_field.output_Ew_field_y, True)
            _propagate_frequencies_low_mem_2d(input_field, omega_indexes, input_field.output_Ew_field_z, False)

    else:
        if input_field.prop.spatial_dimensions == 2:
            _propagate_frequencies(input_field, omega_indexes, input_field.output_Ew_field_y, True)
            _propagate_frequencies(input_field, omega_indexes, input_field.output_Ew_field_z, False)
        else:
            _propagate_frequencies_2d(input_field, omega_indexes, input_field.output_Ew_field_y, True)
            _propagate_frequencies_2d(input_field, omega_indexes, input_field.output_Ew_field_z, False)


def _generate_output_Ew_field_as_file(input_field: laser_input.InputField, comm, rank, num_processes):
    if input_field.prop.spatial_dimensions == 2:
        Ew_mem_output_y = input_field.get_output_Ew_field_file_y()
        Ew_mem_output_z = input_field.get_output_Ew_field_file_z()
    else:
        Ew_mem_output_y = input_field.get_output_Ew_field_file_y_2d()
        Ew_mem_output_z = input_field.get_output_Ew_field_file_z_2d()

    chunk_size, start_index, end_index = laser_input.get_chunk_info(len(input_field.prop.omegas), rank, num_processes)
    omega_indexes = np.arange(start_index, end_index)

    if input_field.prop.low_mem:
        if input_field.prop.spatial_dimensions == 2:
            _propagate_frequencies_low_mem(input_field, omega_indexes, Ew_mem_output_y, True)
            Ew_mem_output_y.flush()
            del Ew_mem_output_y

            _propagate_frequencies_low_mem(input_field, omega_indexes, Ew_mem_output_z, False)
            Ew_mem_output_z.flush()
            del Ew_mem_output_z
        else:
            _propagate_frequencies_low_mem_2d(input_field, omega_indexes, Ew_mem_output_y, True)
            Ew_mem_output_y.flush()
            del Ew_mem_output_y

            _propagate_frequencies_low_mem_2d(input_field, omega_indexes, Ew_mem_output_z, False)
            Ew_mem_output_z.flush()
            del Ew_mem_output_z
    else:
        if input_field.prop.spatial_dimensions == 2:
            _propagate_frequencies(input_field, omega_indexes, Ew_mem_output_y, True)
            Ew_mem_output_y.flush()
            del Ew_mem_output_y

            _propagate_frequencies(input_field, omega_indexes, Ew_mem_output_z, False)
            Ew_mem_output_z.flush()
            del Ew_mem_output_z
        else:
            _propagate_frequencies_2d(input_field, omega_indexes, Ew_mem_output_y, True)
            Ew_mem_output_y.flush()
            del Ew_mem_output_y

            _propagate_frequencies_2d(input_field, omega_indexes, Ew_mem_output_z, False)
            Ew_mem_output_z.flush()
            del Ew_mem_output_z


def _generate_output_Et_field_in_memory(input_field: laser_input.InputField, comm, rank, num_processes):
    output_Ew_field_y = np.fft.ifftshift(input_field.output_Ew_field_y, axes=0)
    output_Et_field_y = np.fft.ifft(output_Ew_field_y, axis=0)
    input_field.output_Et_field_y = np.swapaxes(np.fft.fftshift(output_Et_field_y, axes=0), 0, 1)

    output_Ew_field_z = np.fft.ifftshift(input_field.output_Ew_field_z, axes=0)
    output_Et_field_z = np.fft.ifft(output_Ew_field_z, axis=0)
    input_field.output_Et_field_z = np.swapaxes(np.fft.fftshift(output_Et_field_z, axes=0), 0, 1)


def _generate_output_Et_field_as_file(input_field: laser_input.InputField, comm, rank, num_processes):
    if input_field.prop.spatial_dimensions == 2:
        Ew_output_y = input_field.get_output_Ew_field_file_y()
        Et_output_y = input_field.get_output_Et_field_file_y()

        Ew_output_z = input_field.get_output_Ew_field_file_z()
        Et_output_z = input_field.get_output_Et_field_file_z()
    else:
        Ew_output_y = input_field.get_output_Ew_field_file_y_2d()
        Et_output_y = input_field.get_output_Et_field_file_y_2d()

        Ew_output_z = input_field.get_output_Ew_field_file_z_2d()
        Et_output_z = input_field.get_output_Et_field_file_z_2d()

    # FFT Ey first then Ez
    _convert_from_freq_to_time(input_field, Ew_output_y, Et_output_y, rank, num_processes)
    _convert_from_freq_to_time(input_field, Ew_output_z, Et_output_z, rank, num_processes)


def _convert_from_freq_to_time(input_field, Ew, Et, rank, num_processes):
    chunk_size, start_index, end_index = laser_input.get_chunk_info(len(input_field.prop.y_vals_output), rank, num_processes)

    Ew_chunk = Ew[:, start_index:end_index].copy()
    del Ew
    Ewyz_shift = np.fft.ifftshift(Ew_chunk, axes=0)
    del Ew_chunk
    Etyz = np.fft.fftshift(np.fft.ifft(Ewyz_shift, axis=0), axes=0)
    del Ewyz_shift
    Et[start_index:end_index] = np.swapaxes(Etyz, 0, 1)
    Et.flush()
    del Et, Etyz


'''
This code has been adapted and slightly modified from 
[Diffractio](https://diffractio.readthedocs.io/en/latest/readme.html)
and from the supplementary info of 
"Efficient full-path optical calculation of scalar and vector diffraction using the Bluestein method" 
by Hu et al. (https://www.nature.com/articles/s41377-020-00362-z#Sec12)
'''
def _bluestein_dft_yz(u0, f1, f2, fs, mout):
    m, n = u0.shape
    f11 = f1 + (mout * fs + f2 - f1) / (2 * mout)
    f22 = f2 + (mout * fs + f2 - f1) / (2 * mout)
    a = np.exp(1j * 2 * np.pi * f11 / fs)
    w = np.exp(-1j * 2 * np.pi * (f22 - f11) / (mout * fs))
    h = np.arange(-m + 1, max(mout, m))
    mp = m + mout - 1
    h = w ** ((h ** 2) / 2)
    ft = np.fft.fft(1 / h[0:mp + 1], 2 ** _next_pow2(mp))

    b = a ** (-(np.arange(0, m))) * h[np.arange(m - 1, 2 * m - 1)]
    tmp = np.tile(b, (n, 1)).T

    b = np.fft.fft(u0 * tmp, 2 ** _next_pow2(mp), axis=0)
    b = np.fft.ifft(b * np.tile(ft, (n, 1)).T, axis=0)

    if mout > 1:
        b = b[m-1:mp, 0:n].T * np.tile(h[m - 1:mp], (n, 1))
    else:
        b = b[0] * h[0]

    l = np.linspace(0, mout - 1, mout)
    l = l / mout * (f22 - f11) + f11

    Mshift = -m / 2
    Mshift = np.tile(np.exp(-1j * 2 * np.pi * l * (Mshift + 1 / 2) / fs), (n, 1))
    b = b * Mshift

    return b

def _bluestein_dft_y(u0, f1, f2, fs, mout):

    m = len(u0)

    f11 = f1 + (mout * fs + f2 - f1) / (2 * mout)
    f22 = f2 + (mout * fs + f2 - f1) / (2 * mout)
    a = np.exp(1j * 2 * np.pi * f11 / fs)
    w = np.exp(-1j * 2 * np.pi * (f22 - f11) / (mout * fs))
    h = np.arange(-m + 1, max(mout, m))
    mp = m + mout - 1
    h = w**((h**2) / 2)
    ft = np.fft.fft(1 / h[0:mp + 1], 2**_next_pow2(mp))

    b = a**(-(np.arange(0, m))) * h[np.arange(m - 1, 2 * m - 1)]
    tmp = b.T
    b = np.fft.fft(u0 * tmp, 2**_next_pow2(mp), axis=0)
    b = np.fft.ifft(b * ft.T, axis=0)
    b = b[m-1:mp].T * h[m - 1:mp]

    l = np.linspace(0, mout - 1, mout)
    l = l / mout * (f22 - f11) + f11

    Mshift = -m / 2
    Mshift = np.exp(-1j * 2 * np.pi * l * (Mshift + 1 / 2) / fs)
    b = b * Mshift

    return b

def _propagate_frequencies_low_mem(input_field: laser_input.InputField, omega_indexes, Ew_output, is_Ey):
    omega_vals = input_field.prop.omegas[omega_indexes]

    # TODO: Do we pull in the file at every loop instance?
    if input_field.prop.save_data_as_files:
        if is_Ey:
            eField_input_full_arr = input_field.get_input_Ew_field_file_y()
        else:
            eField_input_full_arr = input_field.get_input_Ew_field_file_z()
    else:
        if is_Ey:
            eField_input_full_arr = input_field.input_Ew_field_y
        else:
            eField_input_full_arr = input_field.input_Ew_field_z

    for base_index, omega in enumerate(omega_vals):
        curr_omega_index = base_index + omega_indexes[0]
        eField_input = eField_input_full_arr[curr_omega_index]
        wvl = (2 * np.pi * constants.C_UM_FS) / omega
        k_val = omega / constants.C_UM_FS
        if input_field.prop.monochromatic_assumption:
            wvl = (2 * np.pi * constants.C_UM_FS) / input_field.laser.omega0
            k_val = input_field.laser.omega0 / constants.C_UM_FS

        if input_field.prop.propagation_type == constants.FRESNEL:
            lens = (input_field.Y_INPUT ** 2 + input_field.Z_INPUT ** 2) / (2*input_field.focus)
        else:
            lens = np.sqrt(input_field.Y_INPUT ** 2 + input_field.Z_INPUT ** 2 + input_field.focus ** 2)
        Tx = np.array(np.exp(-1j * k_val * lens), dtype=np.complex64)
        eField_lens = np.multiply(eField_input, Tx)
        prop_distance = input_field.laser.output_distance_from_focus + input_field.focus


        if input_field.prop.propagation_type == constants.FRESNEL:
            H0 = np.exp(1j * k_val * prop_distance) * (1 / (1j * wvl * prop_distance)) \
                 * np.exp(1j * (k_val / (2 * prop_distance))
                          * (input_field.Y_OUTPUT ** 2 + input_field.Z_OUTPUT ** 2))
            H = np.exp(1j * (k_val / (2 * prop_distance))
                       * (input_field.Y_INPUT ** 2 + input_field.Z_INPUT ** 2))
        else:
            R_output = np.sqrt(input_field.Y_OUTPUT ** 2 + input_field.Z_OUTPUT ** 2 + prop_distance ** 2)
            H0 = 1 / (2 * np.pi) * np.exp(1.j * k_val * R_output) \
                 * prop_distance / R_output ** 2 * (1 / R_output - 1.j * k_val)
            R_input = np.sqrt(input_field.Y_INPUT ** 2 + input_field.Z_INPUT ** 2 + prop_distance ** 2)
            H = 1 / (2 * np.pi) * np.exp(1.j * k_val * R_input) \
                * prop_distance / R_input ** 2 * (1 / R_input - 1.j * k_val)

        u0 = eField_lens * H

        dy_input = input_field.prop.y_vals_input[1] - input_field.prop.y_vals_input[0]
        dz_input = input_field.prop.z_vals_input[1] - input_field.prop.z_vals_input[0]

        fs_y = wvl * prop_distance / dy_input  # dimension of the imaging plane
        fs_z = wvl * prop_distance / dz_input  # dimension of the imaging plane

        fy1 = input_field.prop.y_vals_output[0] + fs_y / 2
        fy2 = input_field.prop.y_vals_output[-1] + fs_y / 2

        fz1 = input_field.prop.z_vals_output[0] + fs_z / 2
        fz2 = input_field.prop.z_vals_output[-1] + fs_z / 2

        u0 = _bluestein_dft_yz(u0, fy1, fy2, fs_y, len(input_field.prop.y_vals_output)) * dy_input
        u0 = _bluestein_dft_yz(u0, fz1, fz2, fs_z, len(input_field.prop.z_vals_output)) * dz_input
        u0 = H0 * u0

        Ew_output[curr_omega_index] = u0

        if input_field.prop.propagation_type == constants.FRESNEL:
            Ew_output[curr_omega_index] = Ew_output[curr_omega_index] * np.exp(-1j * k_val * prop_distance)
        else:
            Ew_output[curr_omega_index] = Ew_output[curr_omega_index] \
                                       * np.exp(-1j * k_val * (prop_distance - input_field.focus)) \
                                       * np.exp(-1j * k_val * R_output)


def _propagate_frequencies_low_mem_2d(input_field: laser_input.InputField, omega_indexes, Ew_output, is_Ey):
    omega_vals = input_field.prop.omegas[omega_indexes]

    # TODO: Do we pull in the file at every loop instance?
    if input_field.prop.save_data_as_files:
        if is_Ey:
            eField_input_full_arr = input_field.get_input_Ew_field_file_y_2d()
        else:
            eField_input_full_arr = input_field.get_input_Ew_field_file_z_2d()
    else:
        if is_Ey:
            eField_input_full_arr = input_field.input_Ew_field_y
        else:
            eField_input_full_arr = input_field.input_Ew_field_z

    for base_index, omega in enumerate(omega_vals):

        curr_omega_index = base_index + omega_indexes[0]
        eField_input = eField_input_full_arr[curr_omega_index]
        wvl = (2 * np.pi * constants.C_UM_FS) / omega
        k_val = omega / constants.C_UM_FS
        if input_field.prop.monochromatic_assumption:
            wvl = (2 * np.pi * constants.C_UM_FS) / input_field.laser.omega0
            k_val = input_field.laser.omega0 / constants.C_UM_FS

        if input_field.prop.propagation_type == constants.FRESNEL:
            lens = (input_field.Y_INPUT ** 2) / (2*input_field.focus)
        else:
            lens = np.sqrt(input_field.Y_INPUT ** 2 + input_field.focus ** 2)
        Tx = np.array(np.exp(-1j * k_val * lens), dtype=np.complex64)
        eField_lens = np.multiply(eField_input, Tx)
        prop_distance = input_field.laser.output_distance_from_focus + input_field.focus

        if input_field.prop.propagation_type == constants.FRESNEL:
            H0 = np.exp(1j * k_val * prop_distance) * (1 / (1j * wvl * prop_distance)) \
                 * np.exp(1j * (k_val / (2 * prop_distance))
                          * (input_field.Y_OUTPUT ** 2))
            H = np.exp(1j * (k_val / (2 * prop_distance))
                       * (input_field.Y_INPUT ** 2))
        else:
            R_output = np.sqrt(input_field.Y_OUTPUT ** 2 + prop_distance ** 2)
            H0 = 1 / (2 * np.pi) * np.exp(1.j * k_val * R_output) \
                 * prop_distance / R_output ** 2 * (1 / R_output - 1.j * k_val)
            R_input = np.sqrt(input_field.Y_INPUT ** 2 + prop_distance ** 2)
            H = 1 / (2 * np.pi) * np.exp(1.j * k_val * R_input) \
                * prop_distance / R_input ** 2 * (1 / R_input - 1.j * k_val)

        u0 = eField_lens * H

        dy_input = input_field.prop.y_vals_input[1] - input_field.prop.y_vals_input[0]
        fs = wvl * prop_distance / dy_input  # dimension of the imaging plane
        fy1 = input_field.prop.y_vals_output[0] + fs / 2
        fy2 = input_field.prop.y_vals_output[-1] + fs / 2

        u0 = _bluestein_dft_y(u0, fy1, fy2, fs, len(input_field.prop.y_vals_output)) * dy_input
        u0 = H0 * u0
        Ew_output[curr_omega_index] = u0

        if input_field.prop.propagation_type == constants.FRESNEL:
            Ew_output[curr_omega_index] = Ew_output[curr_omega_index] * np.exp(-1j * k_val * prop_distance)
        else:
            Ew_output[curr_omega_index] = Ew_output[curr_omega_index] \
                                          * np.exp(-1j * k_val * (prop_distance - input_field.focus)) \
                                          * np.exp(-1j * k_val * R_output)


# TODO: add support for vectorizing omega
def _propagate_frequencies(input_field: laser_input.InputField, omega_indexes, Ew_output, is_Ey):
    omega_vals = input_field.prop.omegas[omega_indexes]
    wvls = (2 * np.pi * constants.C_UM_FS) / omega_vals  # um
    k_vals = omega_vals / constants.C_UM_FS
    if input_field.prop.monochromatic_assumption:
        wvls = (2 * np.pi * constants.C_UM_FS) / (input_field.laser.omega0 * np.ones(omega_vals.shape))
        k_vals = (input_field.laser.omega0 * np.ones(omega_vals.shape)) / constants.C_UM_FS

    WVLS, Y_OUTPUT, Z_OUTPUT = \
        np.meshgrid(wvls, input_field.prop.y_vals_output, input_field.prop.z_vals_output, indexing='ij')
    K_VALS_OUTPUT, Y_OUTPUT, Z_OUTPUT = \
        np.meshgrid(k_vals, input_field.prop.y_vals_output, input_field.prop.z_vals_output, indexing='ij')
    K_VALS_INPUT, Y_INPUT, Z_INPUT = \
        np.meshgrid(k_vals, input_field.prop.y_vals_input, input_field.prop.z_vals_input, indexing='ij')

    if input_field.prop.save_data_as_files:
        if is_Ey:
            eField_input = input_field.get_input_Ew_field_file_y()[omega_indexes]
        else:
            eField_input = input_field.get_input_Ew_field_file_z()[omega_indexes]
    else:
        if is_Ey:
            eField_input = input_field.input_Ew_field_y[omega_indexes]
        else:
            eField_input = input_field.input_Ew_field_z[omega_indexes]

    if input_field.prop.propagation_type == constants.FRESNEL:
        lens = (Y_INPUT ** 2 + Z_INPUT ** 2) / (2*input_field.focus)
    else:
        lens = np.sqrt(Y_INPUT ** 2 + Z_INPUT ** 2 + input_field.focus ** 2)
    Tx = np.array(np.exp(-1j * K_VALS_INPUT * lens), dtype=np.complex64)
    eField_lens = np.multiply(eField_input, Tx)
    prop_distance = input_field.laser.output_distance_from_focus + input_field.focus

    if input_field.prop.propagation_type == constants.FRESNEL:
        H0 = np.exp(1j * K_VALS_OUTPUT * prop_distance) * (1 / (1j * WVLS * prop_distance)) \
             * np.exp(1j * (K_VALS_OUTPUT / (2 * prop_distance)) * (Y_OUTPUT ** 2 + Z_OUTPUT ** 2))
        H = np.exp(1j * (K_VALS_INPUT / (2 * prop_distance)) * (Y_INPUT ** 2 + Z_INPUT ** 2))
    else:
        R_output = np.sqrt(Y_OUTPUT ** 2 + Z_OUTPUT ** 2 + prop_distance ** 2)
        H0 = 1 / (2 * np.pi) * np.exp(1.j * K_VALS_OUTPUT * R_output) \
             * prop_distance / R_output ** 2 * (1 / R_output - 1.j * K_VALS_OUTPUT)
        R_input = np.sqrt(Y_INPUT ** 2 + Z_INPUT ** 2 + prop_distance ** 2)
        H = 1 / (2 * np.pi) * np.exp(1.j * K_VALS_INPUT * R_input) \
            * prop_distance / R_input ** 2 * (1 / R_input - 1.j * K_VALS_INPUT)

    u0 = eField_lens * H

    dy_input = input_field.prop.y_vals_input[1] - input_field.prop.y_vals_input[0]
    dz_input = input_field.prop.z_vals_input[1] - input_field.prop.z_vals_input[0]

    fs_y = wvls * prop_distance / dy_input  # dimension of the imaging plane
    fs_z = wvls * prop_distance / dz_input  # dimension of the imaging plane

    fy1 = input_field.prop.y_vals_output[0] + fs_y / 2
    fy2 = input_field.prop.y_vals_output[-1] + fs_y / 2

    fz1 = input_field.prop.z_vals_output[0] + fs_z / 2
    fz2 = input_field.prop.z_vals_output[-1] + fs_z / 2

    # TODO: can we vectorize this better?
    for index_from_0, w_index in enumerate(omega_indexes):
        u0_output = _bluestein_dft_yz(u0[index_from_0], fy1[index_from_0], fy2[index_from_0],
                                      fs_y[index_from_0], len(input_field.prop.y_vals_output)) * dy_input

        Ew_output[w_index] = _bluestein_dft_yz(u0_output, fz1[index_from_0], fz2[index_from_0],
                                               fs_z[index_from_0], len(input_field.prop.z_vals_output)) * dz_input

    Ew_output[omega_indexes] = H0 * Ew_output[omega_indexes]
    if input_field.prop.propagation_type == constants.FRESNEL:
        Ew_output[omega_indexes] = Ew_output[omega_indexes] * np.exp(-1j * K_VALS_OUTPUT * prop_distance)
    else:
        Ew_output[omega_indexes] = Ew_output[omega_indexes] \
                                   * np.exp(-1j * K_VALS_OUTPUT * (prop_distance - input_field.focus)) \
                                   * np.exp(-1j * K_VALS_OUTPUT * R_output)

def _propagate_frequencies_2d(input_field: laser_input.InputField, omega_indexes, Ew_output, is_Ey):
    omega_vals = input_field.prop.omegas[omega_indexes]
    wvls = (2 * np.pi * constants.C_UM_FS) / omega_vals  # um
    k_vals = omega_vals / constants.C_UM_FS
    if input_field.prop.monochromatic_assumption:
        wvls = (2 * np.pi * constants.C_UM_FS) / (input_field.laser.omega0 * np.ones(omega_vals.shape))
        k_vals = (input_field.laser.omega0 * np.ones(omega_vals.shape)) / constants.C_UM_FS

    WVLS, Y_OUTPUT = np.meshgrid(wvls, input_field.prop.y_vals_output, indexing='ij')
    K_VALS_OUTPUT, Y_OUTPUT = np.meshgrid(k_vals, input_field.prop.y_vals_output, indexing='ij')
    K_VALS_INPUT, Y_INPUT = np.meshgrid(k_vals, input_field.prop.y_vals_input, indexing='ij')

    if input_field.prop.save_data_as_files:
        if is_Ey:
            eField_input = input_field.get_input_Ew_field_file_y_2d()[omega_indexes]
        else:
            eField_input = input_field.get_input_Ew_field_file_z_2d()[omega_indexes]
    else:
        if is_Ey:
            eField_input = input_field.input_Ew_field_y[omega_indexes]
        else:
            eField_input = input_field.input_Ew_field_z[omega_indexes]

    if input_field.prop.propagation_type == constants.FRESNEL:
        lens = (Y_INPUT ** 2) / (2*input_field.focus)
    else:
        lens = np.sqrt(Y_INPUT ** 2 + input_field.focus ** 2)
    Tx = np.array(np.exp(-1j * K_VALS_INPUT * lens), dtype=np.complex64)
    eField_lens = np.multiply(eField_input, Tx)
    prop_distance = input_field.laser.output_distance_from_focus + input_field.focus

    if input_field.prop.propagation_type == constants.FRESNEL:
        H0 = np.exp(1j * K_VALS_OUTPUT * prop_distance) * (1 / (1j * WVLS * prop_distance)) \
             * np.exp(1j * (K_VALS_OUTPUT / (2 * prop_distance)) * (Y_OUTPUT ** 2))
        H = np.exp(1j * (K_VALS_INPUT / (2 * prop_distance)) * (Y_INPUT ** 2))
    else:
        R_output = np.sqrt(Y_OUTPUT ** 2 + prop_distance ** 2)
        H0 = 1 / (2 * np.pi) * np.exp(1.j * K_VALS_OUTPUT * R_output) \
             * prop_distance / R_output ** 2 * (1 / R_output - 1.j * K_VALS_OUTPUT)
        R_input = np.sqrt(Y_INPUT ** 2 + prop_distance ** 2)
        H = 1 / (2 * np.pi) * np.exp(1.j * K_VALS_INPUT * R_input) \
            * prop_distance / R_input ** 2 * (1 / R_input - 1.j * K_VALS_INPUT)

    u0 = eField_lens * H

    dy_input = input_field.prop.y_vals_input[1] - input_field.prop.y_vals_input[0]
    fs_y = wvls * prop_distance / dy_input  # dimension of the imaging plane

    fy1 = input_field.prop.y_vals_output[0] + fs_y / 2
    fy2 = input_field.prop.y_vals_output[-1] + fs_y / 2

    # TODO: can we vectorize this better?
    for index_from_0, w_index in enumerate(omega_indexes):
        u0_output = _bluestein_dft_y(u0[index_from_0], fy1[index_from_0], fy2[index_from_0],
                                     fs_y[index_from_0], len(input_field.prop.y_vals_output)) * dy_input
        Ew_output[w_index] = u0_output

    Ew_output[omega_indexes] = H0 * Ew_output[omega_indexes]
    if input_field.prop.propagation_type == constants.FRESNEL:
        Ew_output[omega_indexes] = Ew_output[omega_indexes] * np.exp(-1j * K_VALS_OUTPUT * prop_distance)
    else:
        Ew_output[omega_indexes] = Ew_output[omega_indexes] \
                                   * np.exp(-1j * K_VALS_OUTPUT * (prop_distance - input_field.focus)) \
                                   * np.exp(-1j * K_VALS_OUTPUT * R_output)


def _next_pow2(x):
    y = np.ceil(np.log2(x))
    if type(x) is np.ndarray:
        y[y == -np.inf] = 0
        return y
    else:
        if y == -np.inf:
            y = 0
        return int(y)
