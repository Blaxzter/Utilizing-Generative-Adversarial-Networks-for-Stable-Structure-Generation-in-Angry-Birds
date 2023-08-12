import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from converter.to_img_converter.LevelImgEncoder import LevelImgEncoder
from game_management.GameConnection import GameConnection
from game_management.GameManager import GameManager
from generator.baseline.Baseline import BaselineGenerator
from level.LevelReader import LevelReader
from level.LevelVisualizer import LevelVisualizer
from test.TestEnvironment import TestEnvironment
from util.Config import Config


def generate_structure():
    config = Config.get_instance()

    level_dest = config.get_data_train_path(folder = 'generated/single_structure/')
    generator = BaselineGenerator()
    generator.settings(number_levels = 3, ground_structure_range = (1, 1), air_structure_range = (0, 0))
    generator.generate_level_init(folder_path = level_dest)


def level_visualisation():
    # generate_structure()

    config = Config.get_instance()
    game_connection = GameConnection(conf = config)
    game_manager = GameManager(conf = config, game_connection = game_connection)
    game_manager.start_game(is_running = False)

    level_folder = config.get_data_train_path(folder = 'temp')
    level_path = f'{level_folder}/level-04.xml'

    # config.game_folder_path = os.path.normpath('../science_birds/{os}')
    # for level_path in sorted(Path(config.get_data_train_path(folder = 'generated/single_structure')).glob('*.xml')):
    level_reader = LevelReader()
    level_visualizer = LevelVisualizer()

    parse_level = level_reader.parse_level(str(level_path), use_blocks = True, use_pigs = True, use_platform = True)
    parse_level.filter_slingshot_platform()

    parse_level.normalize()
    parse_level.create_polygons()

    game_manager.change_level(path = str(level_path))

    fig, ax = plt.subplots(1, 3, dpi = 300, figsize = (15, 5))

    level_visualizer.visualize_screenshot(game_connection.create_level_img(structure = True), ax = ax[0])
    level_visualizer.create_img_of_level(level = parse_level, element_ids = False, use_grid = True, add_dots = False,
                                         ax = ax[1])
    level_visualizer.visualize_level_img(parse_level, dot_version = True, ax = ax[2])
    fig.suptitle(f'Level: {str(level_path)}', fontsize = 16)

    fig.tight_layout()
    plt.show()

    game_manager.stop_game()

def create_img_of_level():
    test_environment = TestEnvironment(level_folder = 'generated/single_structure')
    example_level = test_environment.get_level(0)
    test_environment.level_visualizer.create_img_of_level(example_level, add_dots = False)

def create_plotly_data(img, true_one_hot = False, seperator = True):

    if true_one_hot:
        color_range = [matplotlib.colors.to_hex(color) for color in
                       plt.get_cmap("Pastel1")(np.linspace(0., 1., img.shape[-1]))]
    else:
        color_mode = np.max(img) > 5
        if color_mode:
            color_range = [matplotlib.colors.to_hex(color) for color in plt.get_cmap("Pastel1")(np.linspace(0., 1., len(np.unique(img))))]
        else:
            if img.shape[-1] == 4:
                color_range = ['#a8d399', '#aacdf6', '#f98387', '#dbcc81']
            else:
                color_range = ['#9673A6', '#a8d399', '#aacdf6', '#f98387', '#dbcc81']

    def _create_box(x, y, layer, width = 1, height = 1, depth = 1, depth_scale = 1):
        sx = x - 0.5
        sy = y - 0.5
        return dict(
            # 8 vertices of a cube
            x = [sx, sx, sx + width, sx + width, sx, sx, sx + width, sx + width],
            y = [sy, sy + height, sy + height, sy, sy, sy + height, sy + height, sy],
            z = np.array([layer, layer, layer, layer, layer + depth, layer + depth, layer + depth, layer + depth]) * depth_scale,
            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
        )

    def _create_boxes(layer, axis = 0, color = 0):
        not_null = np.nonzero(layer)
        return dict(
            boxes = [_create_box(x, y, axis) for x, y in zip(not_null[0], not_null[1])],
            color = color_range[color]
        )

    def _combine(box_list, opacity = 1):
        boxes = box_list['boxes']
        color = box_list['color']

        return go.Mesh3d(
            # 8 vertices of a cube
            x = list(np.concatenate([np.asarray(data_dict['x']) for idx, data_dict in enumerate(boxes)])),
            y = list(np.concatenate([np.asarray(data_dict['y']) for idx, data_dict in enumerate(boxes)])),
            z = list(np.concatenate([np.asarray(data_dict['z']) for idx, data_dict in enumerate(boxes)])),

            # i, j and k give the vertices of triangles
            i = list(np.concatenate([np.asarray(data_dict['i']) + 8 * idx for idx, data_dict in enumerate(boxes)])),
            j = list(np.concatenate([np.asarray(data_dict['j']) + 8 * idx for idx, data_dict in enumerate(boxes)])),
            k = list(np.concatenate([np.asarray(data_dict['k']) + 8 * idx for idx, data_dict in enumerate(boxes)])),

            color = color,
            opacity = opacity,
            intensitymode = 'cell',
            flatshading = True,
        )

    if len(img.shape) == 2:

        boxes = []
        for color_idx, material_idx in enumerate(np.unique(img)):
            if material_idx == 0:
                continue
            current_img = img == material_idx
            boxes.append(_create_boxes(current_img, color = color_idx))

        return [_combine(box_data) for box_data in boxes if len(box_data['boxes']) != 0]

    ret_boxs = []
    iter_range = img.shape[-1]
    for last_axis in range(iter_range):
        ret_boxs.append(_create_boxes(img[:, :, last_axis], axis = last_axis, color = last_axis))

    boxes_ = [_combine(box_data) for box_data in ret_boxs if len(box_data['boxes']) != 0]
    if seperator:
        border_wall = dict(
            boxes = [_create_box(0, 0, i - 0.06, width = img.shape[0], height = img.shape[1], depth = 0.01) for i in range(iter_range)],
            color = 'grey'
        )

        boxes_ += [_combine(border_wall, opacity = 0.09)]
    return boxes_


def plotly():
    test_environment = TestEnvironment('generated/single_structure')
    level = test_environment.get_level(1)

    level_img_encoder = LevelImgEncoder()
    # level_img = level_img_encoder.create_calculated_img(level.get_used_elements())
    # multilayer_level_img = level_img_encoder.create_multi_dim_img_from_picture(level_img)

    multilayer_level_img = level_img_encoder.create_one_element_img(level.get_used_elements())

    plotly_data = create_plotly_data(multilayer_level_img)
    fig = go.Figure(data = plotly_data)
    fig.update_layout(scene = dict(aspectmode = 'data'))
    fig.show()


if __name__ == '__main__':
    # plotly()
    # level_visualisation()
    create_img_of_level()
