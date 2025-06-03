from dataclasses import dataclass
import numpy as np

# Roll the Fourier output E field so that the peak intensity is at t=0
CENTER_PEAK_EFIELD_AT_0 = True

GRATING_ANGLE_OF_INCIDENCE = np.deg2rad(52.8)



@dataclass
class AdvancedParameters:
    center_peak_E_at_0: bool
    grating_aoi: float


advanced_parameters_obj = AdvancedParameters(
    center_peak_E_at_0=CENTER_PEAK_EFIELD_AT_0,
    grating_aoi=GRATING_ANGLE_OF_INCIDENCE
)