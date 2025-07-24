from dataclasses import dataclass

# Roll the Fourier output E field so that the peak intensity is at t=0
CENTER_PEAK_EFIELD_AT_0 = False



@dataclass
class AdvancedParameters:
    center_peak_E_at_0: bool


advanced_parameters_obj = AdvancedParameters(
    center_peak_E_at_0=CENTER_PEAK_EFIELD_AT_0,
)
