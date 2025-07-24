from inspect import getmembers
from configs import base_config

# Import all constants from base_config
for k, v in getmembers(base_config):
    if k.isupper():
        globals()[k] = v

SIM_DIRECTORY = "./example/"
DATA_DIRECTORY_PATH = SIM_DIRECTORY + "mem_files/"
WAIST_IN = 20e4
