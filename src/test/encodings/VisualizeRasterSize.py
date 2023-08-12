import matplotlib.pyplot as plt

from level import Constants
from level.LevelVisualizer import LevelVisualizer
from test.TestEnvironment import TestEnvironment


def visualize_raster_sizes():

    test_environment = TestEnvironment('generated/single_structure')
    level = test_environment.get_level(0)

    level_visualizer = LevelVisualizer()

    level.filter_slingshot_platform()
    level.normalize()
    level.create_polygons()

    fig, ax = plt.subplots(2, 3, dpi = 100, figsize=(15, 5))

    Constants.resolution = 0.03
    level_visualizer.create_img_of_level(level, use_grid = True, add_dots = False, ax = ax[0, 0])
    level_rep1 = level_visualizer.visualize_level_img(level, dot_version = False, ax = ax[1, 0])

    Constants.resolution = 0.07
    level_visualizer.create_img_of_level(level, use_grid = True, add_dots = False, ax = ax[0, 1])
    level_rep2 = level_visualizer.visualize_level_img(level, dot_version = False, ax = ax[1, 1])

    Constants.resolution = 0.12
    level_visualizer.create_img_of_level(level, use_grid = True, add_dots = False, ax = ax[0, 2])
    level_rep3 = level_visualizer.visualize_level_img(level, dot_version = False, ax = ax[1, 2])

    fig.suptitle(f'Different level rasterisation with different grid size', fontsize = 15)
    ax[0, 0].set_title(f"Resolution 0.03 {level_rep1[0].shape}")
    ax[0, 1].set_title(f"Resolution 0.07 {level_rep2[0].shape}")
    ax[0, 2].set_title(f"Resolution 0.12 {level_rep3[0].shape}")

    fig.tight_layout()
    plt.show()




def create_level_img():

    test_environment = TestEnvironment('generated/single_structure')
    level = test_environment.get_level(0)

    level_visualizer = LevelVisualizer()

    level.filter_slingshot_platform()
    level.normalize()
    level.create_polygons()

    fig, ax = plt.subplots(1, 1, dpi = 100, figsize=(15, 5))

    level_visualizer.visualize_level_img(level, dot_version = False, ax = ax)

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_raster_sizes()
    # create_level_img()
