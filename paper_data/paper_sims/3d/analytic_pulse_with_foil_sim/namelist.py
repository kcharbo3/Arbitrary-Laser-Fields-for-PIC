# ----------------------------------------------------------------------------------------
# 					SIMULATION PARAMETERS FOR THE PIC-CODE SMILEI
# ----------------------------------------------------------------------------------------
from math import pi
import numpy as np
from fourier_prop.laser_input import (laser_parameters, propagation_parameters, utils)
from fourier_prop.read_laser import sim_grid_parameters as grid
from fourier_prop.read_laser import read_laser
from fourier_prop.sim_helpers import (foil_shapes, sim_parameters)


def microns_to_norm_units(l):
    return utils.microns_to_norm_units(l, laser_parameters.REF_FREQ)


def fs_to_norm_units(t):
    return utils.fs_to_norm_units(t, laser_parameters.REF_FREQ)


# SIMULATION DIMENSIONS
l0 = 2. * pi  # laser wavelength [in code units]
t0 = l0  # optical cycle
Lsim = [
    microns_to_norm_units(grid.X_LENGTH),
    microns_to_norm_units(grid.Y_HEIGHT),
    microns_to_norm_units(grid.Z_HEIGHT)
]  # length of the simulation
Tsim = fs_to_norm_units(grid.T_LENGTH)

dt = t0 * grid.DT_SIM

sim_grid_parameters = grid.compute_sim_grid(
    propagation_parameters.TIMES,
    propagation_parameters.Y_VALS_OUTPUT,
    propagation_parameters.Z_VALS_OUTPUT
)

by_func = read_laser.get_By_function(propagation_parameters.DATA_DIRECTORY_PATH, sim_grid_parameters)
bz_func = read_laser.get_Bz_function(propagation_parameters.DATA_DIRECTORY_PATH, sim_grid_parameters)

ppc = 4

Main(
    geometry="3Dcartesian",
    solve_poisson=True,

    interpolation_order=2,

    cell_length=[l0 * grid.DX_SIM, l0 * grid.DY_SIM, l0 * grid.DZ_SIM],
    grid_length=Lsim,

    number_of_patches=[16, 16, 16],

    timestep=dt,
    simulation_time=Tsim,
    reference_angular_frequency_SI=laser_parameters.REF_FREQ,  # for ionization

    EM_boundary_conditions=[
        ['silver-muller'],
        ['silver-muller'],
        ['silver-muller']
    ],
)


def foil_shape_profile(n0):
    return foil_shapes.circular_foil(
        n0,
        microns_to_norm_units(sim_parameters.FOIL_LEFT_X),
        microns_to_norm_units(sim_parameters.FOIL_RADIUS),
        microns_to_norm_units(sim_parameters.FOIL_THICKNESS),
        microns_to_norm_units(grid.Y_HEIGHT / 2.),
        microns_to_norm_units(grid.Z_HEIGHT / 2.),
        sim_parameters.PRE_PLASMA_PARAMS["PRE_PLASMA"],
        microns_to_norm_units(sim_parameters.PRE_PLASMA_PARAMS["CHAR_LENGTH"]),
        sim_parameters.PRE_PLASMA_PARAMS["CUT_OFF_DENSITY"]
    )


n0_h = 30
frozen_time = 0
cold_or_mj = 'cold'

Species(
    name='hydrogen_ions',
    position_initialization='random',
    momentum_initialization=cold_or_mj,
    particles_per_cell=ppc,
    mass=1836. * 1,
    charge=1.,
    number_density=foil_shape_profile(
        n0_h,
    ),
    boundary_conditions=[
        ["remove", "remove"],
        ["remove", "remove"],
        ["remove", "remove"],
    ],
    time_frozen=frozen_time,
    temperature=[0.],
)

Species(
    name='hydrogen_electrons',
    position_initialization='hydrogen_ions',
    momentum_initialization=cold_or_mj,
    particles_per_cell=ppc,
    mass=1.,
    charge=-1.,
    number_density=foil_shape_profile(
        n0_h,
    ),
    boundary_conditions=[
        ["remove", "remove"],
        ["remove", "remove"],
        ["remove", "remove"],
    ],
    time_frozen=frozen_time,
    temperature=[0.],
)

Laser(
    box_side="xmin",
    space_time_profile=[by_func, bz_func]
)

##### DIAGNOSTICS #####

period_timestep = t0 / dt
data_sample_rate = 1 * period_timestep

fields = ["Ey", "Rho_hydrogen_ions"]

DiagScalar(
    every=10,
    vars=["Utot", "Ukin", "Uelm"],
    precision=10
)

DiagProbe(
    # name = "my_probe",
    every=5 * data_sample_rate,
    origin=[0., 0., Lsim[2] / 2.],
    corners=[
        [Lsim[0], 0., Lsim[2] / 2.],
        [0., Lsim[1], Lsim[2] / 2.],
    ],
    number=[1000, 1000],
    fields=fields
)

DiagProbe(
    # name = "my_probe",
    every=5 * data_sample_rate,
    origin=[0., Lsim[1] / 2., 0.],
    corners=[
        [Lsim[0], Lsim[1] / 2., 0.],
        [0., Lsim[1] / 2., Lsim[2]],
    ],
    number=[1000, 1000],
    fields=fields
)

DiagParticleBinning(
    # name = "my binning",
    deposited_quantity="weight_ekin",
    every=1 * data_sample_rate,
    time_average=1,
    species=["hydrogen_ions"],
    axes=[
        ["ekin", "auto", "auto", 400]
    ]
)
