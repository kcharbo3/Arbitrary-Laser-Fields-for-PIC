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


class SingleThreadComm:
    def Barrier(self):
        return
