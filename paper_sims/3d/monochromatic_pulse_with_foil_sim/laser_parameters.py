from fourier_prop.laser_input import constants
from dataclasses import dataclass
import numpy as np

WAVELENGTH = 1.  # um
REF_FREQ = (2*np.pi*constants.C_SPEED) / (WAVELENGTH * 1.e-6)
OMEGA0 = REF_FREQ * 1e-15  # rad / PHz

POLARIZATION = constants.LINEAR_Y

SPATIAL_SHAPE = constants.GAUSSIAN
SPATIAL_GAUSSIAN_ORDER = 1
TEMPORAL_SHAPE = constants.GAUSSIAN_T
TEMPORAL_GAUSSIAN_ORDER = 1
PHASE_OFFSET = -2*np.pi*(0.0897)

WAIST_IN = 7.5e4
DELTAX = 0

# Use either alpha or grating separation for a more accurate chirp profile
USE_GRATING_EQ = False
ALPHA = 0  # TODO: specify alpha in terms of beta_ba?
GRATING_SEPARATION = 0e3

PULSE_FWHM = 25. * (24.588742233055843 / 24.54557434350688) * (24.553313764862047 / 24.547112098203467)  # fs
SPOT_SIZE = 4.
#FOCUS = 1.28e3 * 736.35
OUTPUT_DISTANCE_FROM_FOCUS = -8.5

NORMALIZE_TO_A0 = True
PEAK_A0 = 21.
TOTAL_ENERGY = 981660.9897641353 * (23./23.1799)  # Roughly 23J

# Used for Special Laser Shapes
L = 1  # LG
NUM_PETALS = 8  # Petal Beam

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
    use_grating_eq: bool
    alpha: float
    grating_separation: float
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


laser_parameters_obj = LaserParameters(
    wavelength=WAVELENGTH, ref_freq=REF_FREQ, omega0=OMEGA0, polarization=POLARIZATION,
    spatial_shape=SPATIAL_SHAPE, spatial_gaussian_order=SPATIAL_GAUSSIAN_ORDER, temporal_shape=TEMPORAL_SHAPE,
    temporal_gaussian_order=TEMPORAL_GAUSSIAN_ORDER, phase_offset=PHASE_OFFSET, use_grating_eq=USE_GRATING_EQ, alpha=ALPHA,
    grating_separation=GRATING_SEPARATION, deltax=DELTAX, pulse_fwhm=PULSE_FWHM, spot_size=SPOT_SIZE, waist_in=WAIST_IN,
    output_distance_from_focus=OUTPUT_DISTANCE_FROM_FOCUS, normalize_to_a0=NORMALIZE_TO_A0, peak_a0=PEAK_A0,
    total_energy=TOTAL_ENERGY, l=L, num_petals=NUM_PETALS
)

