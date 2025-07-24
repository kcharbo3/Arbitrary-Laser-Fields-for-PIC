# ----------------------------------------------------------------------------------------
# 					SIMULATION PARAMETERS FOR THE PIC-CODE SMILEI
# ----------------------------------------------------------------------------------------
from math import pi
from fourier_prop.laser_input import utils
from fourier_prop.read_laser import sim_grid_parameters as grid
from fourier_prop.read_laser import read_laser
from configs import load_config
import os

config_path = os.environ.get("ALFP_CONFIG")
if not config_path:
    raise Exception("ALFP_CONFIG environment variable not set!")

alfp_config = load_config.load_config(config_path)
ref_freq = alfp_config["laser"].ref_freq
grid_params = alfp_config["sim_grid"]
prop = alfp_config["propagation"]

def microns_to_norm_units(l):
    return utils.microns_to_norm_units(l, ref_freq)

def fs_to_norm_units(t):
    return utils.fs_to_norm_units(t, ref_freq)

# SIMULATION DIMENSIONS
l0 = 2. * pi  # laser wavelength [in code units]
t0 = l0  # optical cycle
Lsim = [
    microns_to_norm_units(grid_params.x_length),
    microns_to_norm_units(grid_params.y_height),
    microns_to_norm_units(grid_params.z_height)
]  # length of the simulation
Tsim = fs_to_norm_units(grid_params.t_length)

dt = t0 * grid_params.dt_sim

full_grid_params = grid.compute_sim_grid(
    prop.times,
    prop.y_vals_output,
    prop.z_vals_output,
    grid_params,
    ref_freq
)

by_func = read_laser.get_By_function(prop.data_directory_path, full_grid_params)
bz_func = read_laser.get_Bz_function(prop.data_directory_path, full_grid_params)

Main(
    geometry="3Dcartesian",
    solve_poisson=True,

    interpolation_order=2,

    cell_length=[l0 * grid_params.dx_sim, l0 * grid_params.dy_sim, l0 * grid_params.dz_sim],
    grid_length=Lsim,

    number_of_patches=[16, 16, 16],

    timestep=dt,
    simulation_time=Tsim,
    reference_angular_frequency_SI=ref_freq,  # for ionization

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

fields = ["Ey", "Ez", "Ex"]

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
    number=[1000, 1000],
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
    number=[1000, 1000],
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
    number=[1000, 1000],
    fields=fields
)
