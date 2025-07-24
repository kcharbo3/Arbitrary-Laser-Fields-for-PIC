# ----------------------------------------------------------------------------------------
# 					SIMULATION PARAMETERS FOR THE PIC-CODE SMILEI
# ----------------------------------------------------------------------------------------
from math import pi
import numpy as np
from fourier_prop.laser_input import (laser_parameters, propagation_parameters, utils)
from fourier_prop.read_laser import sim_grid_parameters_2d as grid
from fourier_prop.read_laser import read_laser_2d as read_laser
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
    microns_to_norm_units(grid.Y_HEIGHT)
]  # length of the simulation
Tsim = fs_to_norm_units(grid.T_LENGTH)

dt = t0 * grid.DT_SIM

sim_grid_parameters = grid.compute_sim_grid(
    propagation_parameters.TIMES,
    propagation_parameters.Y_VALS_OUTPUT
)

by_func = read_laser.get_By_function(propagation_parameters.DATA_DIRECTORY_PATH, sim_grid_parameters)
bz_func = read_laser.get_Bz_function(propagation_parameters.DATA_DIRECTORY_PATH, sim_grid_parameters)

ppc = 128

Main(
    geometry="2Dcartesian",
    solve_poisson=True,

    interpolation_order=2,

    cell_length=[l0 * grid.DX_SIM, l0 * grid.DY_SIM],
    grid_length=Lsim,

    number_of_patches=[16, 16],

    timestep=dt,
    simulation_time=Tsim,
    reference_angular_frequency_SI=laser_parameters.REF_FREQ,  # for ionization

    EM_boundary_conditions=[
        ['silver-muller'],
        ['silver-muller']
    ],

    random_seed=0,
)


def foil_shape_profile(n0):
    return foil_shapes.angled_flat_2d(
        n0,
        microns_to_norm_units(sim_parameters.FOIL_LEFT_X),
        microns_to_norm_units(sim_parameters.FOIL_LENGTH),
        microns_to_norm_units(sim_parameters.FOIL_THICKNESS),
        sim_parameters.FOIL_ANGLE,
        microns_to_norm_units(grid.Y_HEIGHT / 2.)
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

fields = ["Ez", "Rho_hydrogen_ions", "Rho_hydrogen_electrons"]

DiagScalar(
    every=10,
    vars=["Utot", "Ukin", "Uelm"],
    precision=10
)

DiagProbe(
    # name = "my_probe",
    every=5 * data_sample_rate,
    origin=[0., 0.],
    corners=[
        [Lsim[0], 0.],
        [0., Lsim[1]],
    ],
    number=[1000, 1000],
    fields=fields
)

# SPECTRUM
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

def weight_energy_threshold50(p):
    energy = 1836 * (np.sqrt(1 + p.px**2 + p.py**2) - 1) * 0.511

    return np.where(energy > 50, p.weight, 0)

def weight_energy_threshold100(p):
    energy = 1836 * (np.sqrt(1 + p.px**2 + p.py**2) - 1) * 0.511

    return np.where(energy > 100, p.weight, 0)

def weight_energy_threshold150(p):
    energy = 1836 * (np.sqrt(1 + p.px**2 + p.py**2) - 1) * 0.511

    return np.where(energy > 150, p.weight, 0)

def weight_energy_threshold200(p):
    energy = 1836 * (np.sqrt(1 + p.px**2 + p.py**2) - 1) * 0.511

    return np.where(energy > 200, p.weight, 0)

def weightekin_energy_threshold50(p):
    energy = 1836 * (np.sqrt(1 + p.px**2 + p.py**2) - 1) * 0.511

    return np.where(energy > 50, p.weight * (energy / 0.511), 0)

def weightekin_energy_threshold100(p):
    energy = 1836 * (np.sqrt(1 + p.px**2 + p.py**2) - 1) * 0.511

    return np.where(energy > 100, p.weight * (energy / 0.511), 0)

def weightekin_energy_threshold150(p):
    energy = 1836 * (np.sqrt(1 + p.px**2 + p.py**2) - 1) * 0.511

    return np.where(energy > 150, p.weight * (energy / 0.511), 0)

def weightekin_energy_threshold200(p):
    energy = 1836 * (np.sqrt(1 + p.px**2 + p.py**2) - 1) * 0.511

    return np.where(energy > 200, p.weight * (energy / 0.511), 0)



def weight_filtered(p):
    return np.where(p.px > 0, p.weight, 0)


def angle_y(p):
    return (180 / np.pi) * np.arctan2(p.py, p.px)

#Tester
DiagParticleBinning(
    deposited_quantity=weight_energy_threshold50,
    every=data_sample_rate,
    time_average=1,
    species=["hydrogen_ions"],
    axes=[
        [angle_y, -80, 80, 800],
        ["ekin", 0, 800, 800]
    ],
)

# Actual
DiagParticleBinning(
    deposited_quantity="weight",
    every=data_sample_rate,
    time_average=1,
    species=["hydrogen_ions"],
    axes=[
        [angle_y, -80, 80, 800],
    ],
)


DiagParticleBinning(
    deposited_quantity=weight_energy_threshold50,
    every=data_sample_rate,
    time_average=1,
    species=["hydrogen_ions"],
    axes=[
        [angle_y, -80, 80, 800],
    ],
)

DiagParticleBinning(
    deposited_quantity=weight_energy_threshold100,
    every=data_sample_rate,
    time_average=1,
    species=["hydrogen_ions"],
    axes=[
        [angle_y, -80, 80, 800],
    ],
)

DiagParticleBinning(
    deposited_quantity=weight_energy_threshold150,
    every=data_sample_rate,
    time_average=1,
    species=["hydrogen_ions"],
    axes=[
        [angle_y, -80, 80, 800],
    ],
)

DiagParticleBinning(
    deposited_quantity=weight_energy_threshold200,
    every=data_sample_rate,
    time_average=1,
    species=["hydrogen_ions"],
    axes=[
        [angle_y, -80, 80, 800],
    ],
)

# ekin divergence bins
DiagParticleBinning(
    deposited_quantity="weight_ekin",
    every=data_sample_rate,
    time_average=1,
    species=["hydrogen_ions"],
    axes=[
        [angle_y, -80, 80, 800],
    ],
)

DiagParticleBinning(
    deposited_quantity=weightekin_energy_threshold50,
    every=data_sample_rate,
    time_average=1,
    species=["hydrogen_ions"],
    axes=[
        [angle_y, -80, 80, 800],
    ],
)

DiagParticleBinning(
    deposited_quantity=weightekin_energy_threshold100,
    every=data_sample_rate,
    time_average=1,
    species=["hydrogen_ions"],
    axes=[
        [angle_y, -80, 80, 800],
    ],
)

DiagParticleBinning(
    deposited_quantity=weightekin_energy_threshold150,
    every=data_sample_rate,
    time_average=1,
    species=["hydrogen_ions"],
    axes=[
        [angle_y, -80, 80, 800],
    ],
)

DiagParticleBinning(
    deposited_quantity=weightekin_energy_threshold200,
    every=data_sample_rate,
    time_average=1,
    species=["hydrogen_ions"],
    axes=[
        [angle_y, -80, 80, 800],
    ],
)

# Beg Plots
DiagParticleBinning(
    deposited_quantity=weight_filtered,
    every=data_sample_rate,
    time_average=1,
    species=["hydrogen_ions"],
    axes=[
        [angle_y, -80, 80, 800],
        ["ekin", 0, 800, 800]
    ],
)

DiagParticleBinning(
    deposited_quantity=weight_filtered,
    every=data_sample_rate,
    time_average=1,
    species=["hydrogen_electrons"],
    axes=[
        [angle_y, -80, 80, 800],
        ["ekin", 0, 800, 800]
    ],
)

# Divergence Screens
# Number Density
DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(35), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity="weight",
    species=["hydrogen_ions"],
    axes=[[angle_y, -90, 90, 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

# Energy Density
DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(35), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity="weight_ekin",
    species=["hydrogen_ions"],
    axes=[[angle_y, -90, 90, 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

# Sum
DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(35), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity="weight",
    species=["hydrogen_ions"],
    axes=[["y", 0, Lsim[1], 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(35), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity="weight_ekin",
    species=["hydrogen_ions"],
    axes=[["y", 0, Lsim[1], 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(35), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity="weight",
    species=["hydrogen_ions"],
    axes=[["y", 0, Lsim[1], 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(35), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity="weight_ekin",
    species=["hydrogen_ions"],
    axes=[["y", 0, Lsim[1], 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(45), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity="weight",
    species=["hydrogen_ions"],
    axes=[["y", 0, Lsim[1], 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(45), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity="weight_ekin",
    species=["hydrogen_ions"],
    axes=[["y", 0, Lsim[1], 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(55), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity="weight",
    species=["hydrogen_ions"],
    axes=[["y", 0, Lsim[1], 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(55), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity="weight_ekin",
    species=["hydrogen_ions"],
    axes=[["y", 0, Lsim[1], 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

# New Screens
# Weight
DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(35), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity=weight_energy_threshold50,
    species=["hydrogen_ions"],
    axes=[[angle_y, -90, 90, 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(35), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity=weight_energy_threshold100,
    species=["hydrogen_ions"],
    axes=[[angle_y, -90, 90, 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(35), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity=weight_energy_threshold150,
    species=["hydrogen_ions"],
    axes=[[angle_y, -90, 90, 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(35), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity=weight_energy_threshold200,
    species=["hydrogen_ions"],
    axes=[[angle_y, -90, 90, 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

# ekin
DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(35), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity=weightekin_energy_threshold50,
    species=["hydrogen_ions"],
    axes=[[angle_y, -90, 90, 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(35), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity=weightekin_energy_threshold100,
    species=["hydrogen_ions"],
    axes=[[angle_y, -90, 90, 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(35), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity=weightekin_energy_threshold150,
    species=["hydrogen_ions"],
    axes=[[angle_y, -90, 90, 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(35), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity=weightekin_energy_threshold200,
    species=["hydrogen_ions"],
    axes=[[angle_y, -90, 90, 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

# Energy Density vs Position
DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(55), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity=weight_energy_threshold50,
    species=["hydrogen_ions"],
    axes=[["y", 0, Lsim[1], 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(55), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity=weight_energy_threshold100,
    species=["hydrogen_ions"],
    axes=[["y", 0, Lsim[1], 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(55), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity=weight_energy_threshold150,
    species=["hydrogen_ions"],
    axes=[["y", 0, Lsim[1], 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(55), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity=weight_energy_threshold200,
    species=["hydrogen_ions"],
    axes=[["y", 0, Lsim[1], 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

# Energy Density
DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(55), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity=weightekin_energy_threshold50,
    species=["hydrogen_ions"],
    axes=[["y", 0, Lsim[1], 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(55), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity=weightekin_energy_threshold100,
    species=["hydrogen_ions"],
    axes=[["y", 0, Lsim[1], 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(55), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity=weightekin_energy_threshold150,
    species=["hydrogen_ions"],
    axes=[["y", 0, Lsim[1], 400]],
    every=(Tsim / dt) - 1  # the entire sim
)

DiagScreen(
    # name = "my screen",
    shape="plane",
    point=[microns_to_norm_units(55), 0.],
    vector=[1., 0.],
    direction="both",
    deposited_quantity=weightekin_energy_threshold200,
    species=["hydrogen_ions"],
    axes=[["y", 0, Lsim[1], 400]],
    every=(Tsim / dt) - 1  # the entire sim
)