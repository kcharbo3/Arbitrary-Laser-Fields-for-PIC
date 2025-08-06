import numpy as np
import fourier_prop.laser_input.constants as constants

# ------------ Laser Parameters ---------------
WAVELENGTH = 1.  # um
REF_FREQ = (2*np.pi*constants.C_SPEED) / (WAVELENGTH * 1.e-6)
OMEGA0 = REF_FREQ * 1e-15  # rad / PHz

POLARIZATION = constants.LINEAR_Y
SPATIAL_SHAPE = constants.LINEAR_CHIRP_Y
SPATIAL_GAUSSIAN_ORDER = 1
TEMPORAL_SHAPE = constants.GAUSSIAN_T
TEMPORAL_GAUSSIAN_ORDER = 1
PHASE_OFFSET = 0.

WAIST_IN = 7.5e4
DELTAX = 0. * WAIST_IN
PULSE_FWHM = 25.
SPOT_SIZE = 4.
OUTPUT_DISTANCE_FROM_FOCUS = -25.

NORMALIZE_TO_A0 = False
PEAK_A0 = 21.
TOTAL_ENERGY = 981660.9897641353

# Used for Special Laser Shapes
L = 1  # LG
# Petal Beam Parameters
NUM_PETALS = 8
WAIST_IN_RADIAL = 7.5e4
WAIST_IN_AZIMUTHAL = 7.5e4

# ------------ Propagation Parameters ---------------
SPATIAL_DIMENSIONS = 2
PROPAGATION_TYPE = constants.RAYLEIGH_SOMMERFELD
MONOCHROMATIC_ASSUMPTION = False

# INPUT PLANE
Y_INPUT_RANGE = 10 * WAIST_IN
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
SIM_DIRECTORY = "./base_sim/"
DATA_DIRECTORY_PATH = SIM_DIRECTORY + "mem_files/"
LOW_MEM = True

# ------------ Advanced Parameters ---------------
# Roll the Fourier output E field so that the peak intensity is at t=0
CENTER_PEAK_EFIELD_AT_0 = True

# Use either alpha (linear) or grating separation (non-linear) for a more accurate chirp profile
USE_GRATING_EQ = True
ALPHA = 0
GRATING_SEPARATION = [30e4]
GRATING_ANGLE_OF_INCIDENCE = [np.deg2rad(52.8)]
GROOVE_PERIOD = [1 / 1480e-3]  # 1480 Grooves/mm
DIFFRACTION_ORDER = [1]

AXICON_ANGLE = 0
ECHELON_DELAY = 0

# ------------ Simulation Grid Parameters ---------------
# In microns
Y_HEIGHT = 28  # Don't forget to update CENTERY
DY_SIM = 1 / 16.
Z_HEIGHT = 28  # Don't forget to update CENTERZ
DZ_SIM = 1 / 16.
T_LENGTH = 250.
DT_SIM = DY_SIM * (0.95 / np.sqrt(3.))  # To satisfy CFL condition

# Not needed for the interpolator, just for sims
X_LENGTH = 75.
DX_SIM = 1 / 16.

LASER_TIME_START = 50.  # fs

# ------------ Other Simulation Parameters ---------------
# These are all lists to support multiple foils. They should all be of the same size.
N0 = np.array([30.])
FOIL_LEFT_X = np.array([8.5])
FOIL_RADIUS = np.array([50])
FOIL_THICKNESS = np.array([0.8])
CENTERY = np.array([Y_HEIGHT / 2.])
CENTERZ = np.array([Z_HEIGHT / 2.])

# Only used for certain foil shape functions
FOIL_ANGLE = np.array([0])

# Only used for certain foil shape functions
PRE_PLASMA = False
CHAR_LENGTH = 0.1
CUT_OFF_DENSITY = .1

