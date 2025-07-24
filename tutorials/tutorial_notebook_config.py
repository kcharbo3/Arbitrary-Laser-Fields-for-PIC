from inspect import getmembers
from configs import base_config
import numpy as np

# Import all constants from base_config
for k, v in getmembers(base_config):
    if k.isupper():
        globals()[k] = v

SIM_DIRECTORY = "./"
DATA_DIRECTORY_PATH = SIM_DIRECTORY + "notebook_laser_files/"

# In microns
Y_HEIGHT = 28
DY_SIM = 1 / 8.
Z_HEIGHT = 28
DZ_SIM = 1 / 8.
T_LENGTH = 250.
DT_SIM = DY_SIM * (0.95 / np.sqrt(3.))  # To satisfy CFL condition

# Not needed for the interpolator, just for sims
X_LENGTH = 50.
DX_SIM = 1 / 8.