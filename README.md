# About
This code enables the use of accurately focused arbitrary beam structures in particle-in-cell (PIC) simulations. The code
provides significant flexibility for simulating focused beams with complex space, time, and polarization couplings
in PIC simulations.

## Workflow
There are 5 primary stages of generating the custom laser fields in PIC simulations that this code handles:
1. Input E field in frequency space
2. Output E field in frequency space
3. Output E field in time space
4. Interpolated output E field to the PIC grid
5. PIC simulation of the field

## Using the Code
Note: All units are in microns and femtoseconds.
For stages (1-4), the notebook `/notebooks/Tutorial.ipynb` covers all of the steps to create your own laser object, propagate it,
and how to plot/visualize it.

For actual use with a PIC code, the directory `/tutorial_sim` covers using the code with the PIC software 
[Smilei](https://smileipic.github.io/Smilei/index.html) that then simulates
the user defined laser.

### Dependencies
For stages (1-3):
For non-parallel computation, only the following packages are used:
* Python3
* numpy
* scipy
* pickle

For parallel computation, the following is additionally required:
* mpi4py

### Fourier Propagation (Stages 1-3)
**Stage 1** is where the user defines their custom parameters for the laser field and for the propagation of that laser field. 
This includes setting parameters that define the input grid (Y, Z, T/W grid at the focusing optic entrance), the output grid
(Y, Z, T/W grid at the propagation distance from the focusing optic), polarization type, spot size, etc. Once all of the parameters
are set, the input grid is populated with the laser field values `E(Y, Z, W)` in frequency space. This defines the laser field
just before focusing (at the focusing optic entrance).

**Stage 2** is where the input laser field is propagated after focusing to some user-defined propagation distance. This step
will populate the output grid with the laser field values `E(Y, Z, W)` in frequency space. This defines the laser field after the
focusing optic.

**Stage 3** is where the output laser field is converted to the time space by FFTs. So the output grid will be populated with the
laser field values `E(Y, Z, T)` in time space.

### Interpolation (Stage 4)
**Stage 4** is where the output laser field `E(Y, Z, T)` is interpolated to the PIC simulation grid. Since PIC simulations often require
very high resolution grids, these values are usually output to files (set by `propagation_parameters.save_as_files`).

### PIC Simulation (Stage 5)
**Stage 5** reads the output files from stage 4 and hands the values over to the PIC simulation software as requested. The PIC simulation
will use a custom laser field block where the user can enter their own laser field values. The field values must be defined at the 
laser entrance window (`X = Position_of_laser_entrance_window`) as a function of `Y, Z, T`. As the PIC simulation runs, it reads the
files from stage 4 to get the entering laser field. Once the values are read and input into the PIC simulation, the PIC software uses
it's own E&M equations to propagate the field.

## Acknowledgements
