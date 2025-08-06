from configs import structs
import importlib.util

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("configs", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # ------------ Load Parameters ---------------
    laser_parameters_obj = structs.get_laser_params(
        module.WAVELENGTH, module.REF_FREQ, module.OMEGA0, module.POLARIZATION,
        module.SPATIAL_SHAPE, module.SPATIAL_GAUSSIAN_ORDER, module.TEMPORAL_SHAPE,
        module.TEMPORAL_GAUSSIAN_ORDER, module.PHASE_OFFSET, module.DELTAX, module.PULSE_FWHM,
        module.SPOT_SIZE, module.WAIST_IN,
        module.OUTPUT_DISTANCE_FROM_FOCUS, module.NORMALIZE_TO_A0, module.PEAK_A0,
        module.TOTAL_ENERGY, module.L, module.NUM_PETALS, module.WAIST_IN_RADIAL, module.WAIST_IN_AZIMUTHAL
    )

    propagation_parameters_obj = structs.get_prop_params(
        module.SPATIAL_DIMENSIONS, module.PROPAGATION_TYPE,
        module.MONOCHROMATIC_ASSUMPTION, module.Y_INPUT_RANGE,
        module.Z_INPUT_RANGE, module.N_Y_INPUT, module.N_Z_INPUT, module.Y_VALS_INPUT,
        module.Z_VALS_INPUT, module.Y_OUTPUT_RANGE, module.Z_OUTPUT_RANGE,
        module.N_Y_OUTPUT, module.N_Z_OUTPUT, module.Y_VALS_OUTPUT, module.Z_VALS_OUTPUT,
        module.N_T, module.T_RANGE, module.TIMES, module.OMEGAS, module.SAVE_DATA_AS_FILES,
        module.SIM_DIRECTORY, module.DATA_DIRECTORY_PATH, module.LOW_MEM
    )

    GRATING_PARAMS = structs.get_grating_params(
        module.USE_GRATING_EQ, module.ALPHA, module.GRATING_ANGLE_OF_INCIDENCE, module.GROOVE_PERIOD,
        module.DIFFRACTION_ORDER, module.GRATING_SEPARATION
    )

    advanced_parameters_obj = structs.get_advanced_params(
        module.CENTER_PEAK_EFIELD_AT_0, GRATING_PARAMS, module.AXICON_ANGLE, module.ECHELON_DELAY
    )

    sim_grid_parameters_obj = structs.get_sim_grid_params(
        module.Y_HEIGHT, module.DY_SIM, module.Z_HEIGHT, module.DZ_SIM, module.T_LENGTH, module.DT_SIM, module.X_LENGTH,
        module.DX_SIM, module.LASER_TIME_START
    )

    PRE_PLASMA_PARAMS = structs.get_pre_plasma_params(module.PRE_PLASMA, module.CHAR_LENGTH, module.CUT_OFF_DENSITY)

    other_sim_parameters_obj = structs.SimParameters(
        module.N0, module.FOIL_LEFT_X, module.FOIL_RADIUS, module.FOIL_THICKNESS, module.CENTERY, module.CENTERZ,
        module.FOIL_ANGLE, PRE_PLASMA_PARAMS
    )

    return {
        "laser": laser_parameters_obj,
        "propagation": propagation_parameters_obj,
        "advanced": advanced_parameters_obj,
        "sim_grid": sim_grid_parameters_obj,
        "other_sim_params": other_sim_parameters_obj
    }
