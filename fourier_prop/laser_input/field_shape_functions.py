from fourier_prop.laser_input import constants
from dataclasses import dataclass
import numpy as np


@dataclass
class ShapeParameters:
    waist_in: float
    deltax: float
    use_grating_eq: bool
    alpha: float
    grating_separation: float
    l: float
    delta_omega: float
    num_petals: int
    spatial_gaussian_order: int
    temporal_gaussian_order: int


# BEAM SPATIAL FUNCTIONS
def lg_shape(y, z, omega, omega0, shape_params):
    R = np.sqrt(y ** 2 + z ** 2)
    return np.array(
        R ** shape_params.l
        * np.exp((-(R + (omega - omega0)) ** 2 / shape_params.waist_in ** 2)
                 + 1j * shape_params.l * np.arctan2(y, z)), dtype=np.complex64
    )

def lg_shape_2d(y, omega, omega0, shape_params):
    return np.array(
        np.abs(y) ** shape_params.l
        * np.exp((-(np.abs(y) + (omega - omega0)) ** 2 / shape_params.waist_in ** 2)
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
        field = _single_petal(petal*ang, y, z, omega, omega0, shape_params, is_Ey=True)
        u += field

    return u


def petal_n_Ez(y, z, omega, omega0, shape_params):
    ang = 360 / shape_params.num_petals

    u = np.zeros((len(z), len(y)), dtype=np.complex64)

    for petal in range(shape_params.num_petals):
        field = _single_petal(petal*ang, y, z, omega, omega0, shape_params, is_Ey=False)
        u += field

    return u


# BEAM TEMPORAL FUNCTIONS
def gaussian_t(omega, omega0, shape_params):
    return np.array(
        np.exp((-(((omega - omega0) ** 2) / (shape_params.delta_omega ** 2))**shape_params.temporal_gaussian_order)),
        dtype=np.complex64
    )

def _single_petal(angle, y, z, omega, omega0, shape_params, is_Ey):
    a = np.deg2rad(angle)
    signz = -np.sign(np.cos(a))
    signy = -np.sign(np.sin(a))
    normz = np.abs(np.cos(a)**-1)
    normy = np.abs(np.sin(a)**-1)

    chirp_val = get_chirp_value(omega, omega0, shape_params)

    deltax = shape_params.deltax
    w_in = shape_params.waist_in
    num_petals = shape_params.num_petals
    base_shape = 1./np.sqrt(num_petals)/np.sqrt(2) \
                 * np.array(np.exp(-1*((((z+(signz*chirp_val+signz*deltax)/normz) / w_in) ** 2)**shape_params.spatial_gaussian_order
                                       + (((y+(signy*chirp_val+signy*deltax)/normy)/w_in) ** 2)**shape_params.spatial_gaussian_order)),
                            dtype=np.complex64)

    if is_Ey:
        u = -signy * base_shape / normy
    else:
        u = -signz * base_shape / normz

    return u


def get_chirp_value(omega, omega0, shape_params):
    chirp_val = shape_params.alpha * (omega - omega0)
    if shape_params.use_grating_eq:
        chirp_val = _get_grating_chirp(
            shape_params.grating_separation, constants.GROOVE_PERIOD, omega,
            omega0, constants.ANGLE_OF_INCIDENCE, constants.DIFFRACTION_ORDER
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
    constants.LG: lg_shape, constants.LG_2D: lg_shape_2d, constants.GAUSSIAN: gaussian_shape,
    constants.GAUSSIAN_2D: gaussian_shape_2d, constants.RADIAL_CHIRP: radial_chirp,
    constants.CHEVRON_2D: chevron_chirp_2d, constants.LINEAR_CHIRP_Y: linear_chirp_y,
    constants.LINEAR_CHIRP_Z: linear_chirp_z, constants.LINEAR_2D: linear_chirp_2d,
    constants.PETAL_N_RADIAL: [petal_n_Ey, petal_n_Ez],
}

TEMPORAL_SHAPE_MAPPINGS = {constants.GAUSSIAN_T: gaussian_t}
