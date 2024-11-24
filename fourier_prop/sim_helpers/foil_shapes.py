import numpy as np


def circular_foil(n0, left_x, foil_radius, thickness, centery, centerz,
                  pre_plasma=False, pre_plasma_char_length=0, cutoff_density=0):

    if not pre_plasma:
        def circular(x, y, z):
            if x < left_x or x > left_x + thickness:
                return 0
            r = np.sqrt((y - centery)**2 + (z - centerz)**2)
            if r < foil_radius:
                return n0
            return 0
    else:
        def circular(x, y, z):
            # To the right of the foil
            if x > left_x + thickness:
                return 0

            # On foil
            r = np.sqrt((y - centery)**2 + (z - centerz)**2)
            if x >= left_x and r < foil_radius:
                return n0

            # To the left of the foil
            if x < left_x and r < foil_radius:
                distance_away = left_x - x
                density_val = n0 * np.exp(-distance_away / pre_plasma_char_length)
                if density_val < cutoff_density:
                    return 0
                else:
                    return density_val

            return 0

    return circular

def flat_foil_2d(n0, left_x, foil_height, thickness, centery,
                  pre_plasma=False, pre_plasma_char_length=0, cutoff_density=0):

    if not pre_plasma:
        def flat(x, y):
            if x < left_x or x > left_x + thickness:
                return 0
            if centery - (foil_height / 2.) < y < centery + (foil_height / 2.):
                return n0
            return 0
    else:
        def flat(x, y):
            # To the right of the foil
            if x > left_x + thickness:
                return 0

            # On foil
            if x >= left_x and centery - (foil_height / 2.) < y < centery + (foil_height / 2.):
                return n0

            # To the left of the foil
            if x < left_x and centery - (foil_height / 2.) < y < centery + (foil_height / 2.):
                distance_away = left_x - x
                density_val = n0 * np.exp(-distance_away / pre_plasma_char_length)
                if density_val < cutoff_density:
                    return 0
                else:
                    return density_val

            return 0

    return flat

def line_eq_get_y(x, m, x0, y0):
    return m*(x - x0) + y0

def line_eq_get_x(y, m, x0, y0):
    return ((1 / m) * (y - y0)) + x0

def cone3D(n0, tip_x, tip_to_end, thickness, angle, centery):
    def cone(x, y, z):
        angle_radians = angle * (np.pi / 180)
        r = np.sqrt((y - centery)**2 + (z - centery)**2)
        m = np.tan(angle_radians)
        if x < tip_x:
            return 0
        if x > tip_x + tip_to_end:
            return 0

        x_tip_inner = (thickness / np.sin(angle_radians)) + tip_x
        r_top = line_eq_get_y(x, m, tip_x, centery) - centery
        r_bottom = line_eq_get_y(x, m, x_tip_inner, centery) - centery
        if r_bottom <= r < r_top:
            return n0
        return 0

    return cone

def angled_flat_2d(n0, x_pos_center, foil_length, thickness, angle, centery):
    def flat(x, y):
        angle_radians = angle * (np.pi / 180)
        m = 1. / np.tan(angle_radians)
        width = thickness / (np.cos(angle_radians))
        x_right_center = x_pos_center + width

        length_radius = foil_length / 2.
        height_radius = length_radius * np.cos(angle_radians)
        if y < centery - height_radius or y > centery + height_radius:
            return 0

        if m == 0:
            x_left = x_pos_center
            x_right = x_pos_center + thickness
        else:
            x_left = line_eq_get_x(y, m, x_pos_center, centery)
            x_right = line_eq_get_x(y, m, x_pos_center + width, centery)

        if x_left < x < x_right:
            return n0

        return 0

    return flat
