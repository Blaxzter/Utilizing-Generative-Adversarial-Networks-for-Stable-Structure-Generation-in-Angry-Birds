import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from shapely.geometry import Polygon

from converter import MathUtil
from converter.to_img_converter.LevelImgDecoder import LevelImgDecoder
from level import Constants
from level.Level import Level
from level.LevelVisualizer import LevelVisualizer
from test.TestEnvironment import TestEnvironment
from util.Config import Config


class LevelImgDecoderVisualization:

    def __init__(self):
        self.config = Config.get_instance()
        self.level_img_decoder = LevelImgDecoder()
        self.level_viz = LevelVisualizer()

    def create_tree_of_one_encoding(self, level_img):
        from ete3 import Tree, TreeNode, TextFace, TreeStyle
        self.t = Tree()

        ts = TreeStyle()
        ts.scale = 10
        # ts.mode = "c"
        # Disable the default tip names config
        ts.show_leaf_name = False
        ts.show_scale = False

        flipped = np.flip(level_img, axis = 0)
        level_img_8 = flipped.astype(np.uint8)
        material_id = np.max(level_img_8)

        level_img_8[level_img_8 != material_id] = 0
        contours, _ = cv2.findContours(level_img_8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        selected_contour = LevelImgDecoder.create_contour_dict(contours[0])

        rectangles, _ = MathUtil.get_rectangles(selected_contour['contour'], selected_contour['poly'])
        rect_dict = self.level_img_decoder.get_rectangle_data(rectangles)
        selected_blocks = self.visualize_select_blocks(
            rectangles = rect_dict,
            used_blocks = [],
            required_area = selected_contour['required_area'],
            poly = selected_contour['poly'],
            tree_node = self.t
        )

        ret_blocks = []
        if selected_blocks is not None:
            for selected_block in selected_blocks:
                selected_block['material'] = material_id
            ret_blocks.append(selected_blocks)

        self.t.show(tree_style = ts)

        flattend_blocks = list(itertools.chain(*ret_blocks))
        level_elements = self.level_img_decoder.create_level_elements(flattend_blocks, [])
        return Level.create_level_from_structure(level_elements)

    def visualize_contours(self, level_img):
        top_value = np.max(level_img)
        bottom_value = np.min(level_img)

        no_birds = False
        if top_value == bottom_value + 1:
            no_birds = True

        ax_plot_positions = [['original', 'original', 'original']] + \
                            [[f'thresh_{i}', f'rectangle_{i}', f'decoded_{i}']
                             for i in range(bottom_value + 1, top_value - (-1 if no_birds else + 1))]
        if not no_birds:
            ax_plot_positions += [[f'pig_thresh', f'eroded', f'positions']]

        fig, axd = plt.subplot_mosaic(
            ax_plot_positions,
            dpi = 300,
            figsize = (8, 10)
        )

        axd['original'].imshow(level_img)
        axd['original'].axis('off')

        for color_idx in range(bottom_value + 1, top_value - (-1 if no_birds else + 1)):
            axs = [axd[ax_name] for ax_name in
                   [f'thresh_{color_idx}', f'rectangle_{color_idx}', f'decoded_{color_idx}']]

            self.visualize_one_decoding(level_img, material_id = color_idx, axs = axs)
            for ax in axs:
                ax.axis('off')

        if not no_birds:
            axs = [axd[ax_name] for ax_name in [f'pig_thresh', f'eroded', f'positions']]
            self.visualize_pig_position(level_img, axs = axs)

        plt.tight_layout()
        plt.show()

    def visualize_one_decoding(self, level_img, material_id = 0, axs = None, title = None, skip_first = False):
        flipped = np.flip(level_img, axis = 0)
        level_img_8 = flipped.astype(np.uint8)
        original_img = level_img_8.copy()

        plot_image = False

        axs_idx = 0
        if axs is None:
            plot_image = True
            if not skip_first:
                skip_first = False
            fig, axs = plt.subplots(1, 4 - (1 if skip_first else 0), figsize = (10, 3), dpi = 300)
            if title is not None:
                fig.sup_title(title)

        if not skip_first:
            axs[axs_idx].imshow(level_img_8, origin = 'lower')
            self.remove_ax_ticks(axs[axs_idx])
            axs_idx += 1

        level_img_8[level_img_8 != material_id] = 0
        contours, _ = cv2.findContours(level_img_8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contour_viz = level_img_8.copy()
        cv2.drawContours(contour_viz, contours, -1, 8, 1)
        axs[axs_idx].imshow(contour_viz, origin = 'lower')
        self.remove_ax_ticks(axs[axs_idx])
        axs_idx += 1

        blocks = []

        for contour_idx, contour in enumerate(contours):
            contour_reshaped = contour.reshape((len(contour), 2))
            poly = Polygon(contour_reshaped)
            required_area = poly.area

            rectangles = [contour]
            contour_list = list(contour)
            if len(contour) > 4:
                rectangles, contour_list = MathUtil.get_rectangles(contour_reshaped, poly)

            hsv = plt.get_cmap('brg')
            dot_colors = hsv(np.linspace(0, 0.8, len(contour_list)))
            for dot_idx, (contour_point, dot_color) in enumerate(zip(contour_list, dot_colors)):
                point_flatten = contour_point.flatten()
                if len(contour_list) < 12:
                    axs[axs_idx].text(point_flatten[0], point_flatten[1], str(dot_idx), color = 'white',
                                      fontsize = 3, ha = 'center', va = 'center')
                dot = patches.Circle(point_flatten, 0.4)
                dot.set_facecolor(dot_color)
                axs[axs_idx].add_patch(dot)

            rectangle_data = []

            hsv = plt.get_cmap('brg')
            colors = hsv(np.linspace(0, 0.8, len(rectangles)))
            for rec_idx, rectangle in enumerate(rectangles):
                rect_reshaped = rectangle.reshape(4, 2)
                center = np.average(rect_reshaped, axis = 0)
                if len(contour_list) < 12:
                    axs[axs_idx].text(center[0], center[1], str(rec_idx), color = 'White', fontsize = 6,
                                      ha = 'center', va = 'center')
                new_patch = patches.Polygon(rect_reshaped, closed = True)
                new_patch.set_linewidth(0.6)
                new_patch.set_edgecolor(colors[rec_idx])
                new_patch.set_facecolor('none')
                axs[axs_idx].add_patch(new_patch)

                rectangle_data.append(self.level_img_decoder.create_rect_dict(rectangle.reshape((4, 2))))

            axs[axs_idx].imshow(original_img, origin = 'lower')
            self.remove_ax_ticks(axs[axs_idx])

            # Sort by rectangle size
            rectangles = sorted(rectangle_data, key = lambda x: x['area'], reverse = True)

            rect_dict = dict()
            for rec_idx, rec in enumerate(rectangles):
                rect_dict[rec_idx] = rec

            selected_blocks = self.visualize_select_blocks(
                rectangles = rect_dict.copy(),
                used_blocks = [],
                required_area = required_area,
                poly = poly,
                tree_node = None
            )

            if selected_blocks is not None:
                for selected_block in selected_blocks:
                    selected_block['material'] = material_id
                blocks.append(selected_blocks)

        # Maybe do a bit of block adjustment to fit better
        # Only required between selected blocks i guess :D

        flattend_blocks = list(itertools.chain(*blocks))
        axs_idx += 1
        # Create block elements out of the possible blocks and the rectangle
        level_elements = self.level_img_decoder.create_level_elements(flattend_blocks, [])
        axs[axs_idx].imshow(level_img_8, origin = 'lower')
        self.remove_ax_ticks(axs[axs_idx])
        self.level_viz.create_img_of_structure(level_elements, use_grid = False, ax = axs[axs_idx], scaled = True)

        if plot_image:
            plt.tight_layout()
            plt.show()

    def visualize_pig_position(self, level_img, axs = None):

        show = False
        if axs is None:
            show = True
            fig, axs = plt.subplots(1, 3)

        level_img_8 = level_img.astype(np.uint8)

        top_value = np.max(level_img_8)
        current_img = np.copy(level_img_8)

        current_img[current_img != top_value] = 0
        axs[0].imshow(current_img)
        axs[0].axis('off')

        kernel = MathUtil.get_circular_kernel(6)
        erosion = cv2.erode(current_img, kernel, iterations = 1)
        contours, _ = cv2.findContours(erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        axs[1].imshow(erosion)
        axs[1].axis('off')

        pig_positions = []
        for contour in contours:
            contour_reshaped = contour.reshape((len(contour), 2))
            pos = np.average(contour_reshaped, axis = 0)
            pig_positions.append(pos)

        for position in pig_positions:
            new_patch = patches.Circle((position[0], position[1]), radius = 0.5 / 2 * 1 / Constants.resolution)
            new_patch.set_facecolor('red')
            axs[2].add_patch(new_patch)

        axs[2].imshow(erosion)
        axs[2].axis('off')

        if show:
            plt.show()

        return pig_positions

    def visualize_rectangles(self, level_img, material_id = 1, axs = None):
        # Create a copy of the img to manipulate it for the contour finding
        current_img = np.ndarray.copy(level_img)
        current_img = current_img.astype(np.uint8)
        current_img[current_img != material_id] = 0

        show_img = False
        if axs is None:
            show_img = True
            fig, axs = plt.subplots(1, 3, dpi = 150, figsize = (12, 3))

        axs[0].set_title("Original Dots")
        axs[0].imshow(current_img)

        axs[1].set_title("With Added Dots")
        axs[1].imshow(current_img)

        axs[2].set_title("Found Rectangles")
        axs[2].imshow(current_img)

        # get the contours
        contours, _ = cv2.findContours(current_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour_idx, contour in enumerate(contours):
            contour_reshaped = contour.reshape((len(contour), 2))
            poly = Polygon(contour_reshaped)

            contour_list = contour_reshaped

            hsv = plt.get_cmap('brg')
            dot_colors = hsv(np.linspace(0, 0.8, len(contour_list)))
            for dot_idx, (contour_point, dot_color) in enumerate(zip(contour_list, dot_colors)):
                point_flatten = contour_point.flatten()
                if len(contour_list) < 12:
                    axs[0].text(point_flatten[0], point_flatten[1], str(dot_idx), color = 'white',
                                fontsize = 2.5, ha = 'center', va = 'center')
                dot = patches.Circle(point_flatten, 0.4)
                dot.set_facecolor(dot_color)
                axs[0].add_patch(dot)

            rectangles, contour_list = MathUtil.get_rectangles(contour_reshaped, poly)

            hsv = plt.get_cmap('brg')
            dot_colors = hsv(np.linspace(0, 0.8, len(contour_list)))
            for dot_idx, (contour_point, dot_color) in enumerate(zip(contour_list, dot_colors)):
                point_flatten = contour_point.flatten()
                axs[1].text(point_flatten[0], point_flatten[1], str(dot_idx), color = 'white',
                            fontsize = 2.5, ha = 'center', va = 'center')
                dot = patches.Circle(point_flatten, 0.4)
                dot.set_facecolor(dot_color)
                axs[1].add_patch(dot)

            hsv = plt.get_cmap('brg')
            colors = hsv(np.linspace(0, 0.8, len(rectangles)))
            for rec_idx, rectangle in enumerate(rectangles):
                new_patch = patches.Polygon(rectangle.reshape(4, 2), closed = True)
                new_patch.set_linewidth(0.6)
                new_patch.set_edgecolor(colors[rec_idx])
                new_patch.set_facecolor('none')
                axs[2].add_patch(new_patch)

                for dot, dot_color in zip(rectangle, ['red', 'green', 'blue', 'black']):
                    dot = patches.Circle(dot.flatten(), 0.4)
                    dot.set_facecolor(dot_color)
                    axs[2].add_patch(dot)

        if show_img:
            plt.tight_layout()
            plt.show()

    def visualize_rectangle(self, level_img, material_id, desired_contour_idx = None, ax = None, filtered = True):
        """
        Visualizes the rectangles of one level img
        """
        current_img = np.ndarray.copy(level_img)
        current_img = current_img.astype(np.uint8)
        current_img[current_img != material_id] = 0

        show_img = False
        if ax is None:
            show_img = True
            fig, ax = plt.subplots(1, 1, dpi = 100)

        # get the contours
        contours, _ = cv2.findContours(current_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        ax.imshow(current_img)
        for contour_idx, contour in enumerate(contours):
            if desired_contour_idx is not None and contour_idx != desired_contour_idx:
                continue

            contour_reshaped = contour.reshape((len(contour), 2))
            poly = Polygon(contour_reshaped)

            rectangles, contour_list = MathUtil.get_rectangles(contour_reshaped, poly)

            rect_dict = self.level_img_decoder.get_rectangle_data(rectangles, filter_rectangles = filtered)

            hsv = plt.get_cmap('brg')
            colors = hsv(np.linspace(0, 0.8, len(rectangles)))
            for rec_idx, rectangle in rect_dict.items():

                new_patch = patches.Polygon(rectangle['rectangle'], closed = True)

                center = np.average(rectangle['rectangle'], axis = 0)
                ax.text(center[0], center[1], str(rec_idx), color = 'Black', fontsize = 12,
                        ha = 'center', va = 'center')

                new_patch.set_linewidth(0.6)
                new_patch.set_edgecolor(colors[rec_idx])
                new_patch.set_facecolor('none')
                ax.add_patch(new_patch)

                for dot, dot_color in zip(rectangle['rectangle'], ['red', 'green', 'blue', 'black']):
                    dot = patches.Circle(dot.flatten(), 0.4)
                    dot.set_facecolor(dot_color)
                    ax.add_patch(dot)

        if show_img:
            plt.tight_layout()
            plt.show()

    def visualize_select_blocks(self, rectangles, used_blocks, required_area, poly, tree_node, occupied_area = 0):
        from ete3 import Tree, TreeNode, TextFace, TreeStyle
        # Break condition
        if occupied_area != 0 and abs(required_area / occupied_area - 1) < 0.05:

            if tree_node is not None:
                name_face = TextFace(tree_node.name, fgcolor='green', fsize=10)
                tree_node.add_face(name_face, column=0, position='branch-right')

            return used_blocks

        if occupied_area > required_area or len(rectangles) == 0:
            if tree_node is not None:
                end_child = tree_node.add_child()
                name_face = TextFace('X', fgcolor='red', fsize=10)
                end_child.add_face(name_face, column=0, position='branch-right')
            return None

        LevelImgDecoder.filter_rectangles_by_used_blocks(rectangles, used_blocks)

        if len(rectangles) == 0:
            if tree_node is not None:
                end_child = tree_node.add_child()
                name_face = TextFace('X', fgcolor='red', fsize=10)
                end_child.add_face(name_face, column=0, position='branch-right')
            return None

        # check if remaining rectangles are able to fill the shape approximetly
        combined_area = np.sum([rec['poly'].area for rec in rectangles.values()])
        if abs((combined_area + occupied_area) / required_area) < 0.8:
            if tree_node is not None:
                end_child = tree_node.add_child()
                name_face = TextFace('X', fgcolor='red', fsize=10)
                end_child.add_face(name_face, column=0, position='branch-right')
            return None

        # Go over each rectangle
        for rec_idx, rec in rectangles.items():

            if rec['height'] in self.level_img_decoder.original_possible_height and \
                    rec['width'] in self.level_img_decoder.original_possible_width:

                # Search for matching block sizes
                for block_idx, block in self.level_img_decoder.block_data.items():

                    block_width, block_height = block['dim']
                    width_diff = abs(block_width / rec['width'] - 1)
                    height_diff = abs(block_height / rec['height'] - 1)

                    if width_diff > 0.001 or height_diff > 0.001:
                        continue

                    next_rectangles = rectangles.copy()
                    del next_rectangles[rec_idx]
                    new_block = dict(
                        block_type = block,
                        rec = rec,
                        rec_idx = rec_idx
                    )

                    next_used_blocks = used_blocks.copy()

                    add_area = self.level_img_decoder.get_area_between_used_blocks(new_block, next_used_blocks)
                    if add_area == -1:
                        continue

                    next_used_blocks.append(new_block)

                    if tree_node is not None:
                        new_child = tree_node.add_child()
                        name_face = TextFace(f"{block['name']} - {rec_idx}" , fsize = 10)
                        new_child.add_face(name_face, column = 0, position = 'branch-right')

                    selected_blocks = self.visualize_select_blocks(
                        rectangles = next_rectangles,
                        used_blocks = next_used_blocks,
                        required_area = required_area,
                        occupied_area = occupied_area + rec['area'] + add_area,
                        poly = poly,
                        tree_node = new_child if tree_node is not None else None
                    )

                    if selected_blocks is not None:
                        if tree_node is not None:
                            name_face.fgcolor = 'green'
                        return selected_blocks
                    else:
                        break

        # The rectangle is bigger than any available block
        # That means it consists out of multiple smaller one
        # Divide the area into divisions of possible blocks

        for rec_idx, rec in rectangles.items():
            rx_1, rx_2, ry_1, ry_2 = (rec['min_x'], rec['max_x'], rec['min_y'], rec['max_y'])
            # Go over both orientations
            for (idx_1, primary_orientation), (idx_2, secondary_orientation) in \
                    [((1, 'height'), (0, 'width')), ((0, 'width'), (1, 'height'))]:

                # for rec_idx, rec in rectangles.items():
                # Only work with fitting blocks
                fitting_blocks = {
                    k: _block for k, _block in self.level_img_decoder.block_data.items()
                    if (_block[primary_orientation] + 2) / rec[primary_orientation] - 1 < 0.001 and
                       (_block[secondary_orientation]) / rec[secondary_orientation] - 1 < 0.001
                }

                # No combination found for this block
                if len(fitting_blocks) == 0:
                    continue

                for combination_amount in range(2, 5):
                    combinations = itertools.product(fitting_blocks.items(), repeat = combination_amount)
                    to_big_counter = 0
                    combi_counter = 0
                    for combination in combinations:
                        combi_counter += 1
                        secondary_size = combination[0][1][secondary_orientation]

                        # Only elements with the same secondary_orientation
                        secondary_direction_difference = np.sum(
                            list(map(lambda block: abs(block[1][secondary_orientation] - secondary_size), combination)))
                        if secondary_direction_difference > 0.01:
                            continue

                        # Check if the two blocks combined can fit in the space
                        combined_height = \
                            np.sum(list(map(lambda _block: _block[1][primary_orientation], combination))) \
                            + (combination_amount - 1)

                        height_difference = rec[primary_orientation] / combined_height - 1
                        if abs(height_difference) < 0.001:
                            # the two blocks fit in the space

                            next_rectangles = rectangles.copy()
                            # Check if there is remaining space in the secondary direction
                            # If so create a new rectangle there
                            # Or if the horizontal space is not filled then create a rec there
                            all_space_used = True
                            if rec[secondary_orientation] / secondary_size - 1 > 0.001:
                                rectangle = np.ndarray.copy(rec['rectangle'])
                                if primary_orientation == 'height':
                                    rectangle[0][idx_2] += secondary_size + 1
                                    rectangle[1][idx_2] += secondary_size + 1
                                else:
                                    rectangle[1][idx_2] += secondary_size + 1
                                    rectangle[2][idx_2] += secondary_size + 1

                                new_dict = self.level_img_decoder.create_rect_dict(rectangle)

                                if new_dict['width'] <= 1 or new_dict['height'] <= 1 or \
                                        new_dict['width'] not in self.level_img_decoder.possible_width or \
                                        new_dict['height'] not in self.level_img_decoder.possible_height:
                                    continue

                                next_rectangles[len(rectangles)] = new_dict
                                all_space_used = False

                            # Create the blocks of each block from bottom to top
                            next_used_blocks = used_blocks.copy()
                            start_value = ry_1 if primary_orientation == 'height' else rx_1
                            used_area = 0

                            if tree_node is not None:
                                new_child = tree_node.add_child()
                                name_faces = []

                            text_face_name = ''

                            for idx, (block_idx, block) in enumerate(combination):
                                block_rectangle = np.ndarray.copy(rec['rectangle'])
                                if primary_orientation == 'height':
                                    block_rectangle[1][idx_1] = start_value
                                    block_rectangle[2][idx_1] = start_value

                                    block_rectangle[0][idx_1] = start_value + block[primary_orientation]
                                    block_rectangle[3][idx_1] = start_value + block[primary_orientation]

                                    if not all_space_used:
                                        block_rectangle[2][idx_2] = block_rectangle[0][idx_2] + secondary_size
                                        block_rectangle[3][idx_2] = block_rectangle[0][idx_2] + secondary_size
                                else:
                                    block_rectangle[0][idx_1] = start_value
                                    block_rectangle[1][idx_1] = start_value

                                    block_rectangle[2][idx_1] = start_value + block[primary_orientation]
                                    block_rectangle[3][idx_1] = start_value + block[primary_orientation]

                                    if not all_space_used:
                                        block_rectangle[1][idx_2] = block_rectangle[0][idx_2] + secondary_size
                                        block_rectangle[2][idx_2] = block_rectangle[0][idx_2] + secondary_size

                                new_block = dict(
                                    block_type = block,
                                    rec = self.level_img_decoder.create_rect_dict(block_rectangle),
                                    rec_idx = rec_idx
                                )
                                add_area = self.level_img_decoder.get_area_between_used_blocks(new_block, next_used_blocks)
                                used_area += block['area'] + add_area - (1 if add_area > 0 else 0)
                                next_used_blocks.append(new_block)
                                start_value += block[f'{primary_orientation}'] + 1

                                if tree_node is not None:
                                    text_face_name += f"{block['name']} - {rec_idx} \n"

                            if tree_node is not None:
                                name_face = TextFace(text_face_name, fsize = 10)
                                new_child.add_face(name_face, column = 0, position = 'branch-right')
                                name_faces.append(name_face)

                            # Remove the current big rectangle
                            del next_rectangles[rec_idx]

                            selected_blocks = self.visualize_select_blocks(
                                rectangles = next_rectangles,
                                used_blocks = next_used_blocks,
                                required_area = required_area,
                                poly = poly,
                                occupied_area = occupied_area + used_area,
                                tree_node = new_child if tree_node is not None else None
                            )

                            if selected_blocks is not None:
                                if tree_node is not None:
                                    for name_face in name_faces:
                                        name_face.fgcolor = 'green'
                                return selected_blocks

                        # This means the block were to big which means doesnt fit
                        if height_difference < 0:
                            to_big_counter += 1

                    # If all blocks combined were to big, we dont need to check more block combinations
                    if to_big_counter > combi_counter:
                        break

        # We tested everything and nothing worked :(
        if tree_node is not None:
            end_child = tree_node.add_child()
            name_face = TextFace('X', fgcolor = 'red', fsize = 10)
            end_child.add_face(name_face, column = 0, position = 'branch-right')
        return None

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

if __name__ == '__main__':
    test_environment = TestEnvironment()

    level = test_environment.get_level(0)
    level_img = test_environment.level_img_encoder.create_calculated_img(level.get_used_elements())

    visualizer = LevelImgDecoderVisualization()

    # visualizer.visualize_one_decoding(level_img, material_id = 2, skip_first = True)
    # visualizer.visualize_rectangle(level_img, material_id = 2, desired_contour_idx = 1, filtered = True)
    visualizer.visualize_pig_position(level_img)
    # visualizer.visualize_rectangles(level_img, material_id = 2)
