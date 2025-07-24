#!/bin/bash
sed -i -e 's/by_func = read_laser.get_By_function(propagation_parameters.DATA_DIRECTORY_PATH, sim_grid_parameters)/by_func=0/g' smilei.py
sed -i -e 's/bz_func = read_laser.get_Bz_function(propagation_parameters.DATA_DIRECTORY_PATH, sim_grid_parameters)/bz_func=0/g' smilei.py