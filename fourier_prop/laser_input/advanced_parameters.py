from dataclasses import dataclass
import numpy as np
from typing import List

# Roll the Fourier output E field so that the peak intensity is at t=0
CENTER_PEAK_EFIELD_AT_0 = True

# Use either alpha (linear) or grating separation (non-linear) for a more accurate chirp profile
USE_GRATING_EQ = True
ALPHA = 0
GRATING_SEPARATION = [30e4]
GRATING_ANGLE_OF_INCIDENCE = [np.deg2rad(52.8)]
GROOVE_PERIOD = [1 / 1480e-3]  # 1480 Grooves/mm
DIFFRACTION_ORDER = [1]

@dataclass
class GratingParameters:
    use_grating_eq: bool
    alpha: float
    grating_aois: List[float]
    groove_periods: List[float]
    diffraction_orders: List[int]
    grating_separations: List[float]

@dataclass
class AdvancedParameters:
    center_peak_E_at_0: bool
    grating_params: GratingParameters

GRATING_PARAMS = GratingParameters(
    use_grating_eq=USE_GRATING_EQ,
    alpha=ALPHA,
    grating_aois=GRATING_ANGLE_OF_INCIDENCE,
    groove_periods=GROOVE_PERIOD,
    diffraction_orders=DIFFRACTION_ORDER,
    grating_separations=GRATING_SEPARATION
)

advanced_parameters_obj = AdvancedParameters(
    center_peak_E_at_0=CENTER_PEAK_EFIELD_AT_0,
    grating_params=GRATING_PARAMS
)