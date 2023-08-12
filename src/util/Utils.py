import time

from level.Constants import coordinate_round


def round_cord(*args):
    if len(args) == 1:
        p = args[0]
        return [round(p[0] * coordinate_round) / coordinate_round, round(p[1] * coordinate_round) / coordinate_round]
    else:
        x = args[0]
        y = args[1]
        return [round(x * coordinate_round) / coordinate_round, round(y * coordinate_round) / coordinate_round]


def round_to_cord(value):
    return round(value * coordinate_round) / coordinate_round


def timeit(function, args):
    start_time = time.time()
    ret_data = function(**args)
    end_time = time.time()

    return ret_data, round((end_time - start_time) * 1000) / 1000

def lighten_color(color, amount=0.5):
    """
    SOURCE: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    import numpy as np
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    int_values = (np.array(colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])) * 255).astype(np.int)
    return '#%02x%02x%02x' % tuple(int_values)
