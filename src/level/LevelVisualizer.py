import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from level import Constants
from level.Level import Level
from level.LevelElement import LevelElement
from level.LevelUtil import calc_structure_dimensions


class LevelVisualizer:

    def __init__(self, dpi = 300, id_font_size = 5, dot_size = 1, line_size = 2):
        self.dpi = dpi
        self.id_font_size = id_font_size
        self.dot_size = dot_size
        self.line_size = line_size

    def create_img_of_structure(self, structure: [LevelElement], title = None,
                                use_grid = True, add_dots = False, element_ids = True, ax = None, scaled = False):
        show = False
        if ax is None:
            show = True
            fig, ax = plt.subplots(dpi = self.dpi)

        hsv = plt.get_cmap('brg')
        colors = hsv(np.linspace(0, 0.8, 4))

        # Add Forms to img
        for idx, element in enumerate(structure):

            # Receive the data required to draw the element shape
            bottom_left, current_color, height, width = self.get_element_data(colors, element, idx, material_color = True)

            if scaled:
                bottom_left = (bottom_left[0] / Constants.resolution, bottom_left[1] / Constants.resolution)
                height /= Constants.resolution
                width /= Constants.resolution

            # Create the matplotlib patch for each element type
            new_patch = self.create_new_patch(ax, bottom_left, current_color, element, height, width)

            # Add the patch to the Axes
            ax.add_patch(new_patch)

            # If block ids should be printed add text at the center of the element
            if element_ids:
                x_pos = element.x / (Constants.resolution if scaled else 1)
                y_pos = element.y / (Constants.resolution if scaled else 1)
                ax.text(x_pos, y_pos, str(element.id), color = "black", fontsize = self.id_font_size,
                        ha = 'center', va = 'center')

        if use_grid or add_dots:
            self.create_dots_and_grid(structure, ax, add_dots, use_grid)

        # ax.set_xticks([])
        # ax.set_yticks([])

        ax.axis('scaled')
        ax.set_title(title)

        if show:
            plt.axis('scaled')
            fig.show()
            plt.close()

    def create_img_of_level(self, level: Level, use_grid = True, add_dots = False, element_ids = True,
                            write_to_file = None, ax = None, scaled = None, material_color = False):
        # Create figure and axes
        show = False
        if ax is None:
            show = True
            fig, ax = plt.subplots(dpi = self.dpi)

        element_list: [LevelElement] = level.get_used_elements()

        if scaled is not None:
            meta_data = level.get_level_metadata()

        if material_color:
            hsv = plt.get_cmap('brg')
            colors = hsv(np.linspace(0, 0.8, 4))
        else:
            if level.structures is None:
                hsv = plt.get_cmap('Paired')
                colors = hsv(np.linspace(0, 0.8, len(element_list)))
            else:
                hsv = plt.get_cmap('brg')
                colors = hsv(np.linspace(0, 0.8, len(level.structures)))

        # Add Forms to img
        for idx, element in enumerate(element_list):

            # Receive the data required to draw the element shape
            bottom_left, current_color, height, width = self.get_element_data(colors, element, idx, level, material_color)

            if scaled is not None:
                scale_to = scaled.shape
                bottom_left[0] = bottom_left[0] * scale_to[1] / meta_data.max_x
                bottom_left[1] = bottom_left[1] * scale_to[0] / meta_data.max_y

                height *= scale_to[0] / meta_data.max_y
                width *= scale_to[1] / meta_data.max_x

            # Create the matplotlib patch for each element type
            new_patch = self.create_new_patch(ax, bottom_left, current_color, element, height, width)

            # Add the patch to the Axes
            ax.add_patch(new_patch)

            # If block ids should be printed add text at the center of the element
            if element_ids:

                x_position = element.x
                y_position = element.y

                if scaled is not None:
                    scale_to = scaled.shape
                    x_position = element.x * scale_to[1] / meta_data.max_x
                    y_position = element.y * scale_to[0] / meta_data.max_y

                ax.text(x_position, y_position, str(element.id),
                         color = "black", fontsize = self.id_font_size, ha = 'center', va = 'center')

        if use_grid or add_dots:
            self.create_dots_and_grid(element_list, ax, add_dots, use_grid)

        ax.axis('scaled')

        if show:
            # plt.title(level.path)

            if write_to_file is not None:
                imgs_folder = f'../imgs/{write_to_file + "/" if len(write_to_file) > 0 else ""}'
                if not Path(imgs_folder).is_dir():
                    os.mkdir(imgs_folder)
                plt.savefig(f'{imgs_folder}{str(level.path)[:-4]}.png')
            else:
                plt.show()

            plt.close()

    def create_dots_and_grid(self, structure, ax, use_dots, use_grid):
        min_x, min_y, max_x, max_y = calc_structure_dimensions(structure)
        if use_dots:
            x_cords = np.arange(min_x + Constants.resolution / 2, max_x + Constants.resolution, Constants.resolution)
            y_cords = np.arange(min_y + Constants.resolution / 2, max_y + Constants.resolution, Constants.resolution)
            XX, YY = np.meshgrid(x_cords, y_cords)
            ax.scatter(XX, YY, s = self.dot_size)

        if use_grid:
            ax.set_xticks(np.arange(min_x, max_x, Constants.resolution))
            ax.set_yticks(np.arange(min_y, max_y, Constants.resolution))
            ax.grid(alpha = 0.2)
            self.remove_ax_ticks(ax)

    @staticmethod
    def remove_ax_ticks(ax):
        ax.tick_params(axis = 'both', which = 'both', grid_alpha = 0, grid_color = "grey")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for tick in ax.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        for tick in ax.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)

    def get_element_data(self, colors, element: LevelElement, idx, level = None, material_color = False):
        bottom_left = element.get_bottom_left()
        width = element.size[0]
        height = element.size[1]

        # Define the color of the element, if there are strucutres then the stucture color else random
        current_color = "Black"
        if material_color:
            if element.object_type == Constants.ObjectType.Pig:
                current_color = colors[-1]
            else:
                material_idx = Constants.materials.index(element.material)
                current_color = colors[material_idx]
        else:
            if level is not None and level.structures is not None:
                for struct_idx, structure in enumerate(level.structures):
                    if element in structure:
                        current_color = colors[struct_idx]
                        break
            else:
                current_color = colors[idx]

        return bottom_left, current_color, height, width

    def create_new_patch(self, ax, bottom_left, current_color, element, height, width):
        new_patch = self.get_patch(ax, bottom_left, element, height, width)
        new_patch.set_linewidth(self.line_size)
        new_patch.set_edgecolor(current_color)
        if "Basic" not in element.type:
            new_patch.set_facecolor('none')
        else:
            new_patch.set_facecolor(current_color)
        return new_patch

    def get_patch(self, ax, bottom_left, element, height, width):
        if "Basic" in element.type or "Circle" in element.type:
            new_patch = patches.Circle(np.array(bottom_left) + width / 2, radius = width / 2)
        elif "Platform" in element.type:
            new_patch = patches.Rectangle(bottom_left, width, height)
            platform_transform = mpl.transforms.Affine2D().rotate_deg_around(element.x, element.y,
                                                                             element.rotation) + ax.transData
            new_patch.set_transform(platform_transform)
        elif "Triangle" in element.type:
            new_patch = patches.Polygon(np.asarray(
                [bottom_left, bottom_left + np.array([width, 0]), bottom_left + np.array([width, height])]),
                closed = True)
            triangle_transform = mpl.transforms.Affine2D().rotate_deg_around(element.x, element.y,
                                                                             element.rotation - 90) + ax.transData
            new_patch.set_transform(triangle_transform)
        else:
            new_patch = patches.Rectangle(bottom_left, width, height)

            # Set meta data color
        return new_patch

    def visualize_level_img(self, level: Level, per_structure = False, ax = None, dot_version = False):

        # Per structure and extern visualization not supported
        if level.structures is not None and len(level.structures) != 1 and ax is not None and per_structure:
            raise Exception("Not supported")

        level_representations = level.create_img(per_structure = per_structure, dot_version = dot_version)

        show = False
        if ax is None:
            show = True
            if len(level_representations) == 1:
                fig, ax = plt.subplots(1, 1, dpi = self.dpi)
            else:
                fig, ax = plt.subplots(
                    int(len(level_representations) / 2), int(len(level_representations) / 2), dpi = self.dpi)

        if len(level_representations) == 1:
            import sys
            np.set_printoptions(threshold = sys.maxsize, linewidth = sys.maxsize)
            # print(np.where(level_representations[0] == 0, " ", level_representations[0].astype(int))) extend = [0, level_representations[0].shape[0], 0, level_representations[0].shape[1]]
            ax.imshow(np.flip(level_representations[0], axis = 0), origin='lower')
        else:
            for level_img, cax in zip(level_representations, ax.flatten()):
                cax.imshow(np.flip(level_img, axis = 0), origin='lower')

        if show:
            if len(level_representations) == 1:
                plt.title(f"{level} method: {'dots' if dot_version else 'math'}", fontsize = 6)
            else:
                plt.suptitle(f"{level} method: {'dots' if dot_version else 'math'}", fontsize = 6)

        plt.axis('scaled')
        # ax.set_xticks([])
        # ax.set_yticks([])

        if show:
            plt.show()

        return level_representations

    def visualize_screenshot(self, img, ax = None):
        show = False
        if ax is None:
            show = True
            fig, ax = plt.subplots(1, 1, dpi = self.dpi)

        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        if show:
            plt.show()
