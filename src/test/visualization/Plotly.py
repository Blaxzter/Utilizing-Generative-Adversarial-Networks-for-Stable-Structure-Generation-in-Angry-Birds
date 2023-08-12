# pip install plotly
import numpy as np
import plotly.graph_objects as go

if __name__ == '__main__':
    X, Y, Z = np.mgrid[:1:20j, :1:20j, :1:20j]
    # vol = (X - 1) ** 2 + (Y - 1) ** 2 + Z ** 2

    fig = go.Figure(data = go.Volume(
        x = X.flatten(), y = Y.flatten(), z = Z.flatten(),
        value = 1,
        isomin = 0.2,
        isomax = 0.7,
        opacity = 0.2,
        surface_count = 21,
        caps = dict(x_show = False, y_show = False, z_show = False),  # no caps
    ))

    # fig.update_layout(scene_camera = dict(
    #     up = dict(x = 0, y = 0, z = 1),
    #     center = dict(x = 0, y = 0, z = 0),
    #     eye = dict(x = 0.1, y = 2.5, z = 0.1)
    # ))

    fig.show()
