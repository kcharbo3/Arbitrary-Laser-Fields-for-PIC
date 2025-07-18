from fourier_prop.laser_input import constants, advanced_parameters
from dataclasses import dataclass
import numpy as np
import scipy


@dataclass
class ShapeParameters:
    waist_in: float
    deltax: float
    l: float
    delta_omega: float
    num_petals: int
    waist_in_radial: float
    waist_in_azimuthal: float
    spatial_gaussian_order: int
    temporal_gaussian_order: int
    polarization: str
    grating_params: advanced_parameters.GratingParameters
    axicon_angle: float
    echelon_delay: float

# BEAM SPATIAL FUNCTIONS
def lg_shape(y, z, omega, omega0, shape_params):
    R = np.sqrt(y ** 2 + z ** 2)

    return np.array(
        R ** shape_params.l
        * np.exp((-(R ** 2) / shape_params.waist_in ** 2)
                 + 1j * shape_params.l * np.arctan2(y, z)), dtype=np.complex64
    )

def lg_shape_2d(y, omega, omega0, shape_params):
    return np.array(
        np.abs(y) ** shape_params.l
        * np.exp((-(np.abs(y)) ** 2 / shape_params.waist_in ** 2)
                 + 1j * shape_params.l * np.arctan2(y, 0)), dtype=np.complex64
    )

def lg_shape_radial_chirp(y, z, omega, omega0, shape_params):
    R = np.sqrt(y ** 2 + z ** 2)
    chirp_val = get_chirp_value(omega, omega0, shape_params)

    return np.array(
        R ** shape_params.l
        * np.exp((-(R - chirp_val) ** 2 / shape_params.waist_in ** 2)
                 + 1j * shape_params.l * np.arctan2(y, z)), dtype=np.complex64
    )

def lg_shape_radial_chirp_2d(y, omega, omega0, shape_params):
    chirp_val = get_chirp_value(omega, omega0, shape_params)

    return np.array(
        np.abs(y) ** shape_params.l
        * np.exp((-(np.abs(y) - chirp_val) ** 2 / shape_params.waist_in ** 2)
                 + 1j * shape_params.l * np.arctan2(y, 0)), dtype=np.complex64
    )

def gaussian_shape(y, z, omega, omega0, shape_params):
    R = np.sqrt(y ** 2 + z ** 2)
    return np.array(np.exp(-(((R / shape_params.waist_in) ** 2)**shape_params.spatial_gaussian_order)), dtype=np.complex64)

def gaussian_shape_2d(y, omega, omega0, shape_params):
    return np.array(np.exp(-(((y / shape_params.waist_in) ** 2)**shape_params.spatial_gaussian_order)), dtype=np.complex64)

# TODO: how to handle the R term
def radial_chirp(y, z, omega, omega0, shape_params):
    R = np.sqrt(y ** 2 + z ** 2)
    chirp_val = get_chirp_value(omega, omega0, shape_params)

    return np.array(
        np.exp(-1. * ((((R - shape_params.deltax - chirp_val) / shape_params.waist_in) ** 2)**shape_params.spatial_gaussian_order))
        * R, dtype=np.complex64
    )

def radial_sinc_shape(y, z, omega, omega0, shape_params):
    R = np.sqrt(y ** 2 + z ** 2)
    chirp_val = get_chirp_value(omega, omega0, shape_params)

    k = np.pi / shape_params.waist_in
    kr = k * (R - shape_params.deltax - chirp_val)

    with np.errstate(divide='ignore', invalid='ignore'):
        profile = 2 * scipy.special.j1(kr) / kr
        profile[R == 0] = 1.0

    return np.array(profile, dtype=np.complex64)

def chevron_chirp_2d(y, omega, omega0, shape_params):
    chirp_val = get_chirp_value(omega, omega0, shape_params)

    return np.array(
        np.exp(-1*((((y-chirp_val-shape_params.deltax)/shape_params.waist_in)**2)**shape_params.spatial_gaussian_order)),
        dtype=np.complex64
    ) + np.array(
        np.exp(-1*((((y+chirp_val+shape_params.deltax)/shape_params.waist_in)**2)**shape_params.spatial_gaussian_order)),
        dtype=np.complex64
    )


def linear_chirp_y(y, z, omega, omega0, shape_params):
    chirp_val = get_chirp_value(omega, omega0, shape_params)
    return np.array(
        np.exp(-1 * ((((y + shape_params.deltax - chirp_val) / shape_params.waist_in) ** 2)**shape_params.spatial_gaussian_order
                     + (z ** 2 / shape_params.waist_in ** 2)**shape_params.spatial_gaussian_order)), dtype=np.complex64
    )

def linear_chirp_2d(y, omega, omega0, shape_params):
    chirp_val = get_chirp_value(omega, omega0, shape_params)
    return np.array(
        np.exp(-1 * ((((y + shape_params.deltax - chirp_val) / shape_params.waist_in) ** 2)**shape_params.spatial_gaussian_order)), dtype=np.complex64
    )


def linear_chirp_z(y, z, omega, omega0, shape_params):
    return linear_chirp_y(z, y, omega, omega0, shape_params)


def petal_n_Ey(y, z, omega, omega0, shape_params):
    ang = 360 / shape_params.num_petals

    u = np.zeros((len(z), len(y)), dtype=np.complex64)

    for petal in range(shape_params.num_petals):
        field = _single_petal(petal*ang, y, z, omega, omega0, shape_params, is_Ey=True, petal_num=petal)
        u += field

    return u


def petal_n_Ez(y, z, omega, omega0, shape_params):
    ang = 360 / shape_params.num_petals

    u = np.zeros((len(z), len(y)), dtype=np.complex64)

    for petal in range(shape_params.num_petals):
        field = _single_petal(petal*ang, y, z, omega, omega0, shape_params, is_Ey=False, petal_num=petal)
        u += field

    return u

def axicon_phase_shape(y, z, omega, omega0, shape_params):
    R = np.sqrt(y**2 + z**2)
    chirp_val = get_chirp_value(omega, omega0, shape_params)
    theta = shape_params.axicon_angle
    k = omega / constants.C_UM_FS

    phase = k * R * np.sin(theta)

    envelope = np.exp(-(((R - shape_params.deltax - chirp_val) / shape_params.waist_in) ** 2)**shape_params.spatial_gaussian_order)

    return np.array(envelope * np.exp(1j * phase), dtype=np.complex64)

def echelon_phase_shape(y, z, omega, omega0, shape_params):
    R = np.sqrt(y**2 + z**2)
    chirp_val = get_chirp_value(omega, omega0, shape_params)

    k = omega / constants.C_UM_FS

    tau_max = shape_params.echelon_delay
    R_max = shape_params.waist_in
    tau_D = tau_max * (R / R_max)**2
    lam0 = (constants.C_UM_FS * 2 * np.pi) / omega0
    delay_term = (constants.C_UM_FS * tau_D) / lam0
    phase_echelon = -(k / 2.) * lam0 * (np.ceil(delay_term) + np.floor(delay_term))

    envelope = np.exp(-(((R - shape_params.deltax - chirp_val) / shape_params.waist_in) ** 2)**shape_params.spatial_gaussian_order)

    return np.array(envelope * np.exp(1j * phase_echelon), dtype=np.complex64)


# BEAM TEMPORAL FUNCTIONS
def gaussian_t(omega, omega0, shape_params):
    return np.array(
        np.exp((-(((omega - omega0) ** 2) / (shape_params.delta_omega ** 2))**shape_params.temporal_gaussian_order)),
        dtype=np.complex64
    )

def _single_petal(angle, y, z, omega, omega0, shape_params, is_Ey, petal_num):
    a = np.deg2rad(angle)

    chirp_val = get_chirp_value(omega, omega0, shape_params)

    deltax = shape_params.deltax
    w_in_radial = shape_params.waist_in_radial
    w_in_azimuthal = shape_params.waist_in_azimuthal
    num_petals = shape_params.num_petals
    polarization = shape_params.polarization

    r_matrix = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    z_rotated = z*r_matrix[0, 0] + y*r_matrix[0, 1]
    y_rotated = z*r_matrix[1, 0] + y*r_matrix[1, 1]

    y_rotated_val = np.exp(-(((y_rotated - (chirp_val + deltax)) / w_in_radial)**2)**shape_params.spatial_gaussian_order)
    z_rotated_val = np.exp(-((z_rotated / w_in_azimuthal)**2)**shape_params.spatial_gaussian_order)
    base_shape = 1./np.sqrt(num_petals)/np.sqrt(2) * np.array(y_rotated_val * z_rotated_val, dtype=np.complex64)

    if polarization == constants.RADIAL:
        if is_Ey:
            u = base_shape * np.cos(a)
        else:
            u = base_shape * np.sin(a)
    elif polarization == constants.AZIMUTHAL:
        if is_Ey:
            u = base_shape * -np.sin(a)
        else:
            u = base_shape * np.cos(a)
    elif polarization == constants.CIRCULAR_L:
        if is_Ey:
            u = base_shape
        else:
            u = -1.0j * base_shape
    elif polarization == constants.CIRCULAR_R:
        if is_Ey:
            u = base_shape
        else:
            u = 1.0j * base_shape
    elif polarization == constants.CIRCULAR_OPPOSITE:
        if petal_num % 2 == 0:
            if is_Ey:
                u = -base_shape
            else:
                u = -1.0j * (-base_shape)
        else:
            if is_Ey:
                u = base_shape
            else:
                u = -1.0j * base_shape
    elif polarization == constants.CIRCULAR_RADIAL_START:
        uy = base_shape * np.cos(a) + 1.0j*base_shape*np.sin(a)
        if is_Ey:
            u = uy
        else:
            u = -1.0j * uy
    else:
        raise Exception("Unsupported polarization for PETAL_N")

    return u


def get_chirp_value(omega, omega0, shape_params):
    grating_params = shape_params.grating_params
    chirp_val = grating_params.alpha * (omega - omega0)

    if grating_params.use_grating_eq:
        num_gratings = len(grating_params.grating_aois)
        chirp_val = 0
        for i in range(num_gratings):
            chirp_val += _get_grating_chirp(
                grating_params.grating_separations[i], grating_params.groove_periods[i], omega,
                omega0, grating_params.grating_aois[i], grating_params.diffraction_orders[i]
            )
    return chirp_val


def _get_grating_chirp(separation, groove_period, omega, omega0, aoi, m):
    x_chirp = np.nan_to_num(
        separation*np.tan(np.arcsin(m*2*np.pi*constants.C_UM_FS/(omega*groove_period)-np.sin(aoi))),
        nan=1000e4
    )
    center_shift = separation * np.tan(
        np.arcsin(m*2*np.pi*constants.C_UM_FS/(omega0*groove_period)-np.sin(aoi))
    )

    return x_chirp - center_shift


SPATIAL_SHAPE_MAPPINGS = {
    constants.LG: lg_shape, constants.LG_2D: lg_shape_2d, constants.LG_RADIAL_CHIRP: lg_shape_radial_chirp,
    constants.LG_RADIAL_CHIRP_2D: lg_shape_radial_chirp_2d, constants.GAUSSIAN: gaussian_shape,
    constants.GAUSSIAN_2D: gaussian_shape_2d, constants.RADIAL_CHIRP: radial_chirp,
    constants.RADIAL_SINC: radial_sinc_shape, constants.CHEVRON_2D: chevron_chirp_2d,
    constants.LINEAR_CHIRP_Y: linear_chirp_y, constants.LINEAR_CHIRP_Z: linear_chirp_z,
    constants.LINEAR_2D: linear_chirp_2d, constants.PETAL_N: [petal_n_Ey, petal_n_Ez],
    constants.AXICON: axicon_phase_shape, constants.ECHELON: echelon_phase_shape
}

TEMPORAL_SHAPE_MAPPINGS = {constants.GAUSSIAN_T: gaussian_t}
