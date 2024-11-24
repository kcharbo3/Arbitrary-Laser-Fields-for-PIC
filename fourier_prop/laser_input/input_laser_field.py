from fourier_prop.laser_input import (
    advanced_parameters,
    propagation_parameters,
    laser_parameters,
    field_shape_functions,
    utils,
    constants
)
import numpy as np
import os


# TODO: Add support for Ez field
# TODO: Finish support for Et input field

class InputField:

    def __init__(
            self,
            propagation_params: propagation_parameters.PropagationParameters,
            laser_params: laser_parameters.LaserParameters,
            advanced_params: advanced_parameters.AdvancedParameters,
            verbose=False
    ):
        self.prop = propagation_params
        self.laser = laser_params
        self.advanced = advanced_params
        self.verbose = verbose

        # Input Fields
        self.input_Ew_field_y = None
        self.input_Et_field_y = None
        self.input_Ew_field_z = None
        self.input_Et_field_z = None

        # Output Fields
        self.output_Ew_field_y = None
        self.output_Et_field_y = None
        self.output_Ew_field_z = None
        self.output_Et_field_z = None

        if self.prop.spatial_dimensions == 2:
            Y_INPUT, Z_INPUT = np.meshgrid(self.prop.y_vals_input, self.prop.z_vals_input, indexing='ij')
            self.Y_INPUT = Y_INPUT
            self.Z_INPUT = Z_INPUT

            Y_OUTPUT, Z_OUTPUT = np.meshgrid(self.prop.y_vals_output, self.prop.z_vals_output, indexing='ij')
            self.Y_OUTPUT = Y_OUTPUT
            self.Z_OUTPUT = Z_OUTPUT
        else:
            self.Y_INPUT = self.prop.y_vals_input
            self.Z_INPUT = np.zeros((len(self.Y_INPUT)))

            self.Y_OUTPUT = self.prop.y_vals_output
            self.Z_OUTPUT = np.zeros((len(self.Y_OUTPUT)))

        self.delta_omega = utils.get_delta_omega_from_fwhm(self.laser.pulse_fwhm)  # rad / PHz

        self.focus = utils.get_focus_from_waist_in(
            self.laser.wavelength, self.laser.spot_size, self.laser.waist_in
        )

        self.shape_params = field_shape_functions.ShapeParameters(
            waist_in=self.laser.waist_in, deltax=self.laser.deltax,
            use_grating_eq=self.laser.use_grating_eq, alpha=self.laser.alpha,
            grating_separation=self.laser.grating_separation, l=self.laser.l,
            delta_omega=self.delta_omega, num_petals=self.laser.num_petals,
            spatial_gaussian_order=self.laser.spatial_gaussian_order,
            temporal_gaussian_order=self.laser.temporal_gaussian_order
        )

        beta = utils.get_beta(self.laser.alpha, self.delta_omega, self.laser.waist_in)
        self.beta_ba = utils.get_betaba(beta)

        self.angle = utils.get_angle(self.laser.alpha, self.laser.omega0,
                                     self.focus, self.laser.deltax)

        self.spatial_shape_function = field_shape_functions.SPATIAL_SHAPE_MAPPINGS[self.laser.spatial_shape]
        self.temporal_shape_function = field_shape_functions.TEMPORAL_SHAPE_MAPPINGS[self.laser.temporal_shape]

        if self.prop.save_data_as_files:
            folder_exists = os.path.exists(self.prop.data_directory_path)
            if not folder_exists:
                os.makedirs(self.prop.data_directory_path)
            if self.prop.spatial_dimensions == 2:
                self._create_input_Ew_field_file_y()
                self._create_input_Ew_field_file_z()

                self._create_output_Ew_field_file_y()
                self._create_output_Ew_field_file_z()

                self._create_output_Et_field_file_y()
                self._create_output_Et_field_file_z()
            else:
                self._create_input_Ew_field_file_y_2d()
                self._create_input_Ew_field_file_z_2d()

                self._create_output_Ew_field_file_y_2d()
                self._create_output_Ew_field_file_z_2d()

                self._create_output_Et_field_file_y_2d()
                self._create_output_Et_field_file_z_2d()

        if self.verbose:
            self._print_parameters()

    def generate_input_Ew_field(self, req_low_mem=False, comm=None, rank=0, num_processes=1):
        if self.prop.save_data_as_files:
            if self.prop.spatial_dimensions == 2:
                self._generate_input_Ew_field_as_file(req_low_mem, comm, rank, num_processes)
            else:
                self._generate_input_Ew_field_as_file_2d(req_low_mem, comm, rank, num_processes)
        else:
            if self.prop.spatial_dimensions == 2:
                self._generate_input_Ew_field_in_memory(req_low_mem, rank, num_processes)
            else:
                self._generate_input_Ew_field_in_memory_2d(req_low_mem, rank, num_processes)

    def _generate_input_Ew_field_in_memory(self, req_low_mem: bool, rank, num_processes):
        self.input_Ew_field_y = \
            np.zeros((len(self.prop.omegas), len(self.prop.y_vals_input), len(self.prop.z_vals_input)),
                     dtype=np.complex64)

        self.input_Ew_field_z = \
            np.zeros((len(self.prop.omegas), len(self.prop.y_vals_input), len(self.prop.z_vals_input)),
                     dtype=np.complex64)

        chunk_size, start_index, end_index = get_chunk_info(len(self.prop.omegas), rank, num_processes)
        if self.laser.spatial_shape == constants.PETAL_N_RADIAL:
            if req_low_mem:
                ew_field_y = self._calc_Ew_chunk_low_mem(rank, num_processes, self.spatial_shape_function[0])
                ew_field_z = self._calc_Ew_chunk_low_mem(rank, num_processes, self.spatial_shape_function[1])
            else:
                ew_field_y = self._calc_Ew_chunk(rank, num_processes, self.spatial_shape_function[0])
                ew_field_z = self._calc_Ew_chunk(rank, num_processes, self.spatial_shape_function[1])
            self.input_Ew_field_y[start_index:end_index] = ew_field_y
            self.input_Ew_field_z[start_index:end_index] = ew_field_z
        else:
            if req_low_mem:
                ew_field = self._calc_Ew_chunk_low_mem(rank, num_processes, self.spatial_shape_function)
            else:
                ew_field = self._calc_Ew_chunk(rank, num_processes, self.spatial_shape_function)

            self._handle_polarization_and_set_fields(
                self.input_Ew_field_y, self.input_Ew_field_z, ew_field, start_index, end_index
            )

    def _generate_input_Ew_field_in_memory_2d(self, req_low_mem: bool, rank, num_processes):
        self.input_Ew_field_y = np.zeros((len(self.prop.omegas), len(self.prop.y_vals_input)), dtype=np.complex64)
        self.input_Ew_field_z = np.zeros((len(self.prop.omegas), len(self.prop.y_vals_input)), dtype=np.complex64)

        chunk_size, start_index, end_index = get_chunk_info(len(self.prop.omegas), rank, num_processes)

        if req_low_mem:
            ew_field = self._calc_Ew_chunk_low_mem(rank, num_processes, self.spatial_shape_function)
        else:
            ew_field = self._calc_Ew_chunk(rank, num_processes, self.spatial_shape_function)

        self._handle_polarization_and_set_fields(
            self.input_Ew_field_y, self.input_Ew_field_z, ew_field, start_index, end_index
        )


    def _generate_input_Ew_field_as_file(self, req_low_mem: bool, comm, rank, num_processes):
        if rank == 0:
            self._create_input_Ew_field_file_y()
            self._create_input_Ew_field_file_z()
        if comm is not None:
            comm.Barrier()

        chunk_size, start_index, end_index = get_chunk_info(len(self.prop.omegas), rank, num_processes)

        Ew_mem_y = self.get_input_Ew_field_file_y()
        Ew_mem_z = self.get_input_Ew_field_file_z()

        if self.laser.spatial_shape == constants.PETAL_8_RADIAL \
                or self.laser.spatial_shape == constants.PETAL_N_RADIAL\
                or self.laser.spatial_shape == constants.PETAL_8_AZIMUTHAL:
            if req_low_mem:
                ew_field_y = self._calc_Ew_chunk_low_mem(rank, num_processes, self.spatial_shape_function[0])
                ew_field_z = self._calc_Ew_chunk_low_mem(rank, num_processes, self.spatial_shape_function[1])
            else:
                ew_field_y = self._calc_Ew_chunk(rank, num_processes, self.spatial_shape_function[0])
                ew_field_z = self._calc_Ew_chunk(rank, num_processes, self.spatial_shape_function[1])
            Ew_mem_y[start_index:end_index] = ew_field_y
            Ew_mem_z[start_index:end_index] = ew_field_z
        else:
            if req_low_mem:
                ew_field = self._calc_Ew_chunk_low_mem(rank, num_processes, self.spatial_shape_function)
            else:
                ew_field = self._calc_Ew_chunk(rank, num_processes, self.spatial_shape_function)

            self._handle_polarization_and_set_fields(Ew_mem_y, Ew_mem_z, ew_field, start_index, end_index)

        Ew_mem_y.flush()
        Ew_mem_z.flush()

        del Ew_mem_y
        del Ew_mem_z

    def _generate_input_Ew_field_as_file_2d(self, req_low_mem: bool, comm, rank, num_processes):
        if rank == 0:
            self._create_input_Ew_field_file_y_2d()
            self._create_input_Ew_field_file_z_2d()

        if comm is not None:
            comm.Barrier()

        chunk_size, start_index, end_index = get_chunk_info(len(self.prop.omegas), rank, num_processes)

        Ew_mem_y = self.get_input_Ew_field_file_y_2d()
        Ew_mem_z = self.get_input_Ew_field_file_z_2d()

        if req_low_mem:
            ew_field = self._calc_Ew_chunk_low_mem(rank, num_processes, self.spatial_shape_function)
        else:
            ew_field = self._calc_Ew_chunk(rank, num_processes, self.spatial_shape_function)

        self._handle_polarization_and_set_fields(Ew_mem_y, Ew_mem_z, ew_field, start_index, end_index)

        Ew_mem_y.flush()
        Ew_mem_z.flush()

        del Ew_mem_y
        del Ew_mem_z


    def _handle_polarization_and_set_fields(self, Ew_field_y, Ew_field_z, calculated_field, start_index, end_index):
        if self.laser.polarization == constants.LINEAR_Y:
            Ew_field_y[start_index:end_index] = calculated_field
            Ew_field_z[start_index:end_index] = 0.
        elif self.laser.polarization == constants.LINEAR_Z:
            Ew_field_y[start_index:end_index] = 0.
            Ew_field_z[start_index:end_index] = calculated_field
        elif self.laser.polarization == constants.RADIAL:
            angles = np.arctan2(self.Y_INPUT, self.Z_INPUT)
            if self.prop.spatial_dimensions == 2:
                sin_angles = (np.sin(angles))[None, :, :]
                cos_angles = (np.cos(angles))[None, :, :]
            else:
                sin_angles = (np.sin(angles))[None, :]
                cos_angles = (np.cos(angles))[None, :]

            Ew_field_y[start_index:end_index] = calculated_field * sin_angles
            Ew_field_z[start_index:end_index] = calculated_field * cos_angles
        elif self.laser.polarization == constants.AZIMUTHAL:
            angles = np.arctan2(self.Y_INPUT, self.Z_INPUT)
            if self.prop.spatial_dimensions == 2:
                sin_angles = (np.sin(angles))[None, :, :]
                cos_angles = (np.cos(angles))[None, :, :]
            else:
                sin_angles = (np.sin(angles))[None, :]
                cos_angles = (np.cos(angles))[None, :]

            Ew_field_y[start_index:end_index] = calculated_field * cos_angles
            Ew_field_z[start_index:end_index] = calculated_field * sin_angles
        elif self.laser.polarization == constants.CIRCULAR_L:
            # TODO: check circularly polarization directions....
            Ew_field_y[start_index:end_index] = calculated_field
            Ew_field_z[start_index:end_index] = -1.0j * calculated_field
        elif self.laser.polarization == constants.CIRCULAR_R:
            Ew_field_y[start_index:end_index] = calculated_field
            Ew_field_z[start_index:end_index] = 1.0j * calculated_field

    def _calc_Ew_chunk_low_mem(self, rank, num_processes, spatial_shape_function):
        chunk_size, start_index, end_index = get_chunk_info(len(self.prop.omegas), rank, num_processes)
        if self.prop.spatial_dimensions == 2:
            Ew = np.zeros((chunk_size, len(self.prop.y_vals_input), len(self.prop.z_vals_input)), dtype=np.complex64)
            for w_index, w in enumerate(self.prop.omegas[start_index:end_index]):
                Ew[w_index] = spatial_shape_function(
                    self.Y_INPUT, self.Z_INPUT, w, self.laser.omega0, self.shape_params
                ) * self.temporal_shape_function(w, self.laser.omega0, self.shape_params)
        else:
            Ew = np.zeros((chunk_size, len(self.prop.y_vals_input)), dtype=np.complex64)
            for w_index, w in enumerate(self.prop.omegas[start_index:end_index]):
                Ew[w_index] = spatial_shape_function(
                    self.Y_INPUT, w, self.laser.omega0, self.shape_params
                ) * self.temporal_shape_function(w, self.laser.omega0, self.shape_params)

        Ew = Ew * np.exp(1j * self.laser.phase_offset)

        return Ew

    def _calc_Ew_chunk(self, rank, num_processes, spatial_shape_function):
        chunk_size, start_index, end_index = get_chunk_info(len(self.prop.omegas), rank, num_processes)
        if self.prop.spatial_dimensions == 2:
            W, Y, Z = np.meshgrid(
                self.prop.omegas[start_index:end_index], self.prop.y_vals_input,
                self.prop.z_vals_input, indexing='ij'
            )

            Ew = spatial_shape_function(
                Y, Z, W, self.laser.omega0, self.shape_params
            ) * self.temporal_shape_function(W, self.laser.omega0, self.shape_params)
        else:
            W, Y = np.meshgrid(
                self.prop.omegas[start_index:end_index], self.prop.y_vals_input, indexing='ij'
            )

            Ew = spatial_shape_function(
                Y, W, self.laser.omega0, self.shape_params
            ) * self.temporal_shape_function(W, self.laser.omega0, self.shape_params)

        Ew = Ew * np.exp(1j * self.laser.phase_offset)

        return Ew

    # CREATE FILES
    def _create_input_Ew_field_file_y(self):
        np.memmap(
            self.prop.data_directory_path + constants.INPUT_EW_FILE_Y, dtype='complex64',
            mode='w+', shape=(len(self.prop.omegas), len(self.prop.y_vals_input), len(self.prop.z_vals_input))
        )

    def _create_input_Ew_field_file_y_2d(self):
        np.memmap(
            self.prop.data_directory_path + constants.INPUT_EW_FILE_Y, dtype='complex64',
            mode='w+', shape=(len(self.prop.omegas), len(self.prop.y_vals_input))
        )

    def _create_input_Ew_field_file_z(self):
        np.memmap(
            self.prop.data_directory_path + constants.INPUT_EW_FILE_Z, dtype='complex64',
            mode='w+', shape=(len(self.prop.omegas), len(self.prop.y_vals_input), len(self.prop.z_vals_input))
        )

    def _create_input_Ew_field_file_z_2d(self):
        np.memmap(
            self.prop.data_directory_path + constants.INPUT_EW_FILE_Z, dtype='complex64',
            mode='w+', shape=(len(self.prop.omegas), len(self.prop.y_vals_input))
        )

    def _create_output_Ew_field_file_y(self):
        np.memmap(
            self.prop.data_directory_path + constants.OUTPUT_EW_FILE_Y, dtype='complex64',
            mode='w+', shape=(len(self.prop.omegas), len(self.prop.y_vals_output), len(self.prop.z_vals_output))
        )

    def _create_output_Ew_field_file_y_2d(self):
        np.memmap(
            self.prop.data_directory_path + constants.OUTPUT_EW_FILE_Y, dtype='complex64',
            mode='w+', shape=(len(self.prop.omegas), len(self.prop.y_vals_output))
        )

    def _create_output_Ew_field_file_z(self):
        np.memmap(
            self.prop.data_directory_path + constants.OUTPUT_EW_FILE_Z, dtype='complex64',
            mode='w+', shape=(len(self.prop.omegas), len(self.prop.y_vals_output), len(self.prop.z_vals_output))
        )

    def _create_output_Ew_field_file_z_2d(self):
        np.memmap(
            self.prop.data_directory_path + constants.OUTPUT_EW_FILE_Z, dtype='complex64',
            mode='w+', shape=(len(self.prop.omegas), len(self.prop.y_vals_output))
        )

    def _create_output_Et_field_file_y(self):
        np.memmap(
            self.prop.data_directory_path + constants.OUTPUT_ET_FILE_Y, dtype='complex64',
            mode='w+', shape=(len(self.prop.y_vals_output), len(self.prop.omegas), len(self.prop.z_vals_output))
        )

    def _create_output_Et_field_file_y_2d(self):
        np.memmap(
            self.prop.data_directory_path + constants.OUTPUT_ET_FILE_Y, dtype='complex64',
            mode='w+', shape=(len(self.prop.y_vals_output), len(self.prop.omegas))
        )

    def _create_output_Et_field_file_z(self):
        np.memmap(
            self.prop.data_directory_path + constants.OUTPUT_ET_FILE_Z, dtype='complex64',
            mode='w+', shape=(len(self.prop.y_vals_output), len(self.prop.omegas), len(self.prop.y_vals_output))
        )

    def _create_output_Et_field_file_z_2d(self):
        np.memmap(
            self.prop.data_directory_path + constants.OUTPUT_ET_FILE_Z, dtype='complex64',
            mode='w+', shape=(len(self.prop.y_vals_output), len(self.prop.omegas))
        )

    # GETTERS
    def get_input_Ew_field_file_y(self):
        return np.memmap(
            self.prop.data_directory_path + constants.INPUT_EW_FILE_Y, dtype='complex64',
            mode='r+', shape=(len(self.prop.omegas), len(self.prop.y_vals_input), len(self.prop.z_vals_input))
        )

    def get_input_Ew_field_file_y_2d(self):
        return np.memmap(
            self.prop.data_directory_path + constants.INPUT_EW_FILE_Y, dtype='complex64',
            mode='r+', shape=(len(self.prop.omegas), len(self.prop.y_vals_input))
        )

    def get_input_Ew_field_file_z(self):
        return np.memmap(
            self.prop.data_directory_path + constants.INPUT_EW_FILE_Z, dtype='complex64',
            mode='r+', shape=(len(self.prop.omegas), len(self.prop.y_vals_input), len(self.prop.z_vals_input))
        )

    def get_input_Ew_field_file_z_2d(self):
        return np.memmap(
            self.prop.data_directory_path + constants.INPUT_EW_FILE_Z, dtype='complex64',
            mode='r+', shape=(len(self.prop.omegas), len(self.prop.y_vals_input))
        )

    def get_output_Ew_field_file_y(self):
        return np.memmap(
            self.prop.data_directory_path + constants.OUTPUT_EW_FILE_Y, dtype='complex64',
            mode='r+', shape=(len(self.prop.omegas), len(self.prop.y_vals_output), len(self.prop.z_vals_output))
        )

    def get_output_Ew_field_file_y_2d(self):
        return np.memmap(
            self.prop.data_directory_path + constants.OUTPUT_EW_FILE_Y, dtype='complex64',
            mode='r+', shape=(len(self.prop.omegas), len(self.prop.y_vals_output))
        )

    def get_output_Ew_field_file_z(self):
        return np.memmap(
            self.prop.data_directory_path + constants.OUTPUT_EW_FILE_Z, dtype='complex64',
            mode='r+', shape=(len(self.prop.omegas), len(self.prop.y_vals_output), len(self.prop.z_vals_output))
        )

    def get_output_Ew_field_file_z_2d(self):
        return np.memmap(
            self.prop.data_directory_path + constants.OUTPUT_EW_FILE_Z, dtype='complex64',
            mode='r+', shape=(len(self.prop.omegas), len(self.prop.y_vals_output))
        )

    def get_output_Et_field_file_y(self):
        return np.memmap(
            self.prop.data_directory_path + constants.OUTPUT_ET_FILE_Y, dtype='complex64',
            mode='r+', shape=(len(self.prop.y_vals_output), len(self.prop.omegas), len(self.prop.z_vals_output))
        )

    def get_output_Et_field_file_y_2d(self):
        return np.memmap(
            self.prop.data_directory_path + constants.OUTPUT_ET_FILE_Y, dtype='complex64',
            mode='r+', shape=(len(self.prop.y_vals_output), len(self.prop.omegas))
        )

    def get_output_Et_field_file_z(self):
        return np.memmap(
            self.prop.data_directory_path + constants.OUTPUT_ET_FILE_Z, dtype='complex64',
            mode='r+', shape=(len(self.prop.y_vals_output), len(self.prop.omegas), len(self.prop.z_vals_output))
        )

    def get_output_Et_field_file_z_2d(self):
        return np.memmap(
            self.prop.data_directory_path + constants.OUTPUT_ET_FILE_Z, dtype='complex64',
            mode='r+', shape=(len(self.prop.y_vals_output), len(self.prop.omegas))
        )

    def _print_parameters(self):
        print("#### Ranges ####")
        print("Spatial Dimensions:", self.prop.spatial_dimensions)
        print("Y Input Range:", self.prop.y_vals_input.min(), self.prop.y_vals_input.max())
        print("Z Input Range:", self.prop.z_vals_input.min(), self.prop.z_vals_input.max())
        print("Y Output Range:", self.prop.y_vals_output.min(), self.prop.y_vals_output.max())
        print("Z Output Range:", self.prop.z_vals_output.min(), self.prop.z_vals_output.max())
        print("Time Range:", self.prop.times.min(), self.prop.times.max())
        print("Omega Range:", self.prop.omegas.min(), self.prop.omegas.max())
        print("#### Resolutions ####")
        print("Dy Input:", self.prop.y_vals_input[1] - self.prop.y_vals_input[0])
        print("Dz Input:", self.prop.z_vals_input[1] - self.prop.z_vals_input[0])
        print("Dy Output:", self.prop.y_vals_output[1] - self.prop.y_vals_output[0])
        print("Dz Output:", self.prop.z_vals_output[1] - self.prop.z_vals_output[0])
        print("Dt:", self.prop.times[1] - self.prop.times[0])
        print("Dw:", self.prop.omegas[1] - self.prop.omegas[0])
        print("#### Other Parameters ####")
        print("Omega0:", self.laser.omega0)
        print("Delta Omega:", self.delta_omega)
        print("Entrance Waist:", self.laser.waist_in)
        print("Focus:", self.focus)
        effective_waist = self.laser.waist_in + self.laser.deltax
        print("F#:", self.focus / (2. * effective_waist))
        if self.laser.use_grating_eq:
            print("Using grating eq with separation:", self.laser.grating_separation)
        else:
            print("Using alpha:", self.laser.alpha)
            print("Angle at Focus:", self.angle)
            print("Beta BA:", self.beta_ba)

def get_input_field(verbose=False):
    return InputField(propagation_parameters.propagation_parameters_obj,
                      laser_parameters.laser_parameters_obj, advanced_parameters.advanced_parameters_obj, verbose)


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
