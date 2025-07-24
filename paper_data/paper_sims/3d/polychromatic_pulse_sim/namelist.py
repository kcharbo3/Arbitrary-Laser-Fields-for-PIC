# ----------------------------------------------------------------------------------------
# 					SIMULATION PARAMETERS FOR THE PIC-CODE SMILEI
# ----------------------------------------------------------------------------------------
from math import pi
from fourier_prop.laser_input import (laser_parameters, propagation_parameters, utils)
from fourier_prop.read_laser import sim_grid_parameters as grid
from fourier_prop.read_laser import read_laser


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

Laser(
    box_side="xmin",
    space_time_profile=[by_func, bz_func]
)

##### DIAGNOSTICS #####
period_timestep = t0 / dt
data_sample_rate = 1 * period_timestep

fields = ["Ey"]

DiagScalar(
    every=10,
    vars=["Utot", "Ukin", "Uelm"],
    precision=10
)

# YX Plane
DiagProbe(
    # name = "my_probe",
    every=5 * data_sample_rate,
    origin=[0., 0., Lsim[2] / 2.],
    corners=[
        [Lsim[0], 0., Lsim[2] / 2.],
        [0., Lsim[1], Lsim[2] / 2.],
    ],
    number=[100 / 0.0625, 28 / 0.0625],
    fields=fields
)

# ZX Plane
DiagProbe(
    # name = "my_probe",
    every=5 * data_sample_rate,
    origin=[0., Lsim[1] / 2., 0.],
    corners=[
        [Lsim[0], Lsim[1] / 2., 0.],
        [0., Lsim[1] / 2., Lsim[2]],
    ],
    number=[100 / 0.0625, 28 / 0.0625],
    fields=fields
)

# YZ Plane
DiagProbe(
    # name = "my_probe",
    every=5 * data_sample_rate,
    origin=[Lsim[0] / 2., 0., 0.],
    corners=[
        [Lsim[0] / 2., 0., Lsim[2]],
        [Lsim[0] / 2., Lsim[1], 0.],
    ],
    number=[28 / 0.0625, 28 / 0.0625],
    fields=fields
)

# 1D Probe
DiagProbe(
    # name = "my_probe",
    every=0.2 * data_sample_rate,
    origin=[0., Lsim[1] / 2., Lsim[2] / 2.],
    corners=[
        [Lsim[0], Lsim[1] / 2., Lsim[2] / 2.],
    ],
    number=[100 / 0.0625],
    fields=fields
)

