from fourier_prop.laser_input import constants
import numpy as np


# lam in m
def lam_to_ang_freq(lam):
    return (2 * np.pi * constants.C_SPEED) / lam


def get_delta_omega_from_fwhm(fwhm):
    return ((2*np.log(2) / np.pi) * 2 * np.pi) / (fwhm * np.sqrt(2 * np.log(2)))


def get_waist_in_from_waist_focus(wvl0, waist_focus, focus):
    return (wvl0 * focus) / (waist_focus * np.pi)


def get_focus_from_waist_in(wvl0, waist_focus, waist_in):
    zr = (np.pi * waist_focus**2) / wvl0
    return np.sqrt((waist_in/waist_focus)**2 - 1) * zr

# Sim Units
# l in microns
def microns_to_norm_units(l, w):
    l_meters = l * (10**-6)
    return l_meters / get_ref_length(w)

def norm_units_to_microns(l, w):
    l = l * get_ref_length(w)
    return l / (1e-6)

# t in fs
def fs_to_norm_units(t, w):
    t_s = t * (10**-15)
    return t_s / get_ref_time(w)

def norm_units_to_fs(t, w):
    t = t * get_ref_time(w)
    return t / (10**-15)

# w in rad*Hz
def get_ref_length(w):
    return constants.C_SPEED / w

# w in rad*Hz
def get_ref_time(w):
    return 1 / w

def get_angle(alpha, omega0, f, deltax):

    theta_pft = np.arctan((alpha*omega0) / f)
    theta_f = np.arctan(deltax / f)

    return np.rad2deg(theta_pft + theta_f)

def get_beta(alpha, delta_w, w_in):
    return alpha*delta_w / w_in


def get_betaba(beta):
    return np.sqrt(1 + beta**2)


def compute_thick_lens_focus(n, R1, R2, d):
    if R1 == np.inf: R1_inv = 0
    else: R1_inv = 1 / R1

    if R2 == np.inf: R2_inv = 0
    else: R2_inv = 1 / R2

    f_inv = (n - 1) * (R1_inv - R2_inv + ((n - 1) * d) / (n * R1 * R2 if R1 != np.inf and R2 != np.inf else np.inf))
    return 1 / f_inv


def thick_lens_phase(y, z, k_vals, R1, R2, n, center_thickness):
    r2 = y**2 + z**2
    sag1 = R1 - np.sqrt(np.maximum(R1**2 - r2, 0.0))
    sag2 = 0
    if R2 != np.inf:
        sag2 = -R2 + np.sqrt(np.maximum(R2**2 - r2, 0.0))
    opd = (n - 1) * (sag1 + sag2)
    material_phase = np.exp(1j * k_vals * center_thickness) * np.exp(1j * k_vals * (n - 1) * center_thickness)
    return np.exp(-1j * k_vals * opd).astype(np.complex64) * material_phase


def thick_lens_phase(y, z, R1, R2, n, center_thickness):
    r2 = y**2 + z**2
    sag1 = R1 - np.sqrt(np.maximum(R1**2 - r2, 0.0))
    sag2 = 0
    if R2 != np.inf:
        sag2 = -R2 + np.sqrt(np.maximum(R2**2 - r2, 0.0))
    opd = (n - 1) * (sag1 + sag2)
    return opd - center_thickness - ((n - 1) * center_thickness)


def n_fused_silica(wavelength_um):
    B = [0.6961663, 0.4079426, 0.8974794]
    C = [0.0684043**2, 0.1162414**2, 9.896161**2]
    lam2 = np.clip(wavelength_um**2, 0.21**2, 3.71**2)
    n2 = 1
    for Bi, Ci in zip(B, C):
        denom = lam2 - Ci
        denom = np.where(denom == 0, np.nan, denom)
        n2 += Bi * lam2 / denom

    return np.sqrt(n2)


class SingleThreadComm:
    def Barrier(self):
        return
