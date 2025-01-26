from fourier_prop.laser_input import (constants, laser_parameters)
from dataclasses import dataclass
import numpy as np

SPATIAL_DIMENSIONS = 2
PROPAGATION_TYPE = constants.RAYLEIGH_SOMMERFELD
MONOCHROMATIC_ASSUMPTION = False

# INPUT PLANE
Y_INPUT_RANGE = 10 * laser_parameters.WAIST_IN
Z_INPUT_RANGE = Y_INPUT_RANGE
N_Y_INPUT = 2 ** 9
N_Z_INPUT = 2 ** 9
Y_VALS_INPUT = np.linspace(-Y_INPUT_RANGE, Y_INPUT_RANGE, N_Y_INPUT)
Z_VALS_INPUT = np.linspace(-Z_INPUT_RANGE, Z_INPUT_RANGE, N_Z_INPUT)

# OUTPUT PLANE
Y_OUTPUT_RANGE = 30  # -50 to 50 um
Z_OUTPUT_RANGE = Y_OUTPUT_RANGE
# Recommended +1 to have center bin at 0, especially for radially polarized beams
N_Y_OUTPUT = (2 ** 8) + 1
N_Z_OUTPUT = (2 ** 8) + 1
Y_VALS_OUTPUT = np.linspace(-Y_OUTPUT_RANGE, Y_OUTPUT_RANGE, N_Y_OUTPUT)
Z_VALS_OUTPUT = np.linspace(-Z_OUTPUT_RANGE, Z_OUTPUT_RANGE, N_Z_OUTPUT)

# TIME DIMENSION
T_RANGE = 250
N_T = 2 ** 10

TIMES = np.linspace(-T_RANGE, T_RANGE, N_T)
TIMES -= 0.000001
DT = TIMES[1] - TIMES[0]
OMEGAS = np.fft.fftshift(np.fft.fftfreq(len(TIMES), DT / (2*np.pi)))
OMEGAS -= 0.0000001

# OTHER
SAVE_DATA_AS_FILES = True
DATA_DIRECTORY_PATH = "./mem_files/"
LOW_MEM = True

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
    data_directory_path: str
    low_mem: bool


propagation_parameters_obj = PropagationParameters(
    spatial_dimensions=SPATIAL_DIMENSIONS, propagation_type=PROPAGATION_TYPE,
    monochromatic_assumption=MONOCHROMATIC_ASSUMPTION, y_input_range=Y_INPUT_RANGE,
    z_input_range=Z_INPUT_RANGE, N_y_input=N_Y_INPUT, N_z_input=N_Z_INPUT, y_vals_input=Y_VALS_INPUT,
    z_vals_input=Z_VALS_INPUT, y_output_range=Y_OUTPUT_RANGE, z_output_range=Z_OUTPUT_RANGE,
    N_y_output=N_Y_OUTPUT, N_z_output=N_Z_OUTPUT, y_vals_output=Y_VALS_OUTPUT, z_vals_output=Z_VALS_OUTPUT,
    N_t=N_T, t_range=T_RANGE, times=TIMES, omegas=OMEGAS, save_data_as_files=SAVE_DATA_AS_FILES,
    data_directory_path=DATA_DIRECTORY_PATH, low_mem=LOW_MEM
)
