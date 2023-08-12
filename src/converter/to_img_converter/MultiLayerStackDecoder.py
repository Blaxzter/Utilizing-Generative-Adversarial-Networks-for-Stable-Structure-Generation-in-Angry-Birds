
from typing import List, Dict

import cv2
import numpy as np
from scipy import ndimage

from converter import MathUtil
from converter.to_img_converter import DecoderUtils
from converter.to_img_converter.DecoderUtils import recalibrate_blocks
from level import Constants
from level.Level import Level
from level.LevelElement import LevelElement
from test.TestUtils import plot_matrix_complete, plot_img, plot_matrix
from test.visualization.GanDecodingVisualization import GanDecodingVisualization
from util.Config import Config


class MultiLayerStackDecoder:

    def __init__(self, level_drawer = None, add_tab = False, plot_to_file = False):
        self.config = Config.get_instance()

        self.level_drawer = level_drawer
        self.visualizer = GanDecodingVisualization(level_drawer = level_drawer, add_tab = add_tab, plot_to_file = plot_to_file)

        self.block_data = self.config.get_block_data(Constants.resolution)
        self.blocks: List[Dict] = list(self.block_data.values())

        self.max_ratio = -np.inf
        for block in self.blocks:
            ratio = block['width'] / block['height']
            self.max_ratio = np.max([self.max_ratio, ratio])

        self.delete_matrices = self.create_delete_block_matrices(self.blocks)

        # Print data
        self.display_decoding = True

        # Decoding parameter
        self.round_to_next_int = True
        self.use_negative_air_value = True
        self.negative_air_value = -2
        self.custom_kernel_scale = True
        self.minus_one_border = False
        self.cutoff_point = 0.85
        self.bird_cutoff = 0.5
        self.recalibrate_blocks = False
        self.combine_layers = True

    def decode(self, gan_output, has_air_layer = True):
        if len(gan_output.shape) == 4:
            gan_output = gan_output[0]

        level_blocks = []

        if self.combine_layers:
            flattened_idx_img = np.argmax(gan_output, axis = 2)
            flattened_idx_img[flattened_idx_img == 4] = 0

            flattened_img = np.zeros(gan_output.shape[:2])
            for i in range(1, 4):
                flattened_img[flattened_idx_img == i] = gan_output[flattened_idx_img == i, i]

            if self.display_decoding:
                self.visualizer.plot_img(flattened_img, title = 'Flattened Img', flip = True)

            level_blocks = self.decode_layer(flattened_img, -1)

            # Get block material by going over each block and check which material is the most confident at this location
            for block in level_blocks:
                start, end, bottom, top = self.get_range_of_block(block['block'], block['position'], gan_output[:, :, 1: -1])
                material = np.argmax(np.sum(gan_output[bottom:top, start:end, 1: -1], axis = (0, 1)))
                block['material'] = material
        else:
            for layer_idx in range(1 if has_air_layer else 0, gan_output.shape[-1] - 1):
                layer_blocks = self.decode_layer(gan_output[:, :, layer_idx], layer_idx)
                level_blocks += layer_blocks

        bird_positions = self.get_pig_position(gan_output[:, :, -1])
        created_level_elements = self.create_level_elements(level_blocks, bird_positions)
        if self.recalibrate_blocks:
            created_level_elements = recalibrate_blocks(created_level_elements)
        created_level = Level.create_level_from_structure(created_level_elements)

        return created_level

    def layer_to_level(self, layer):
        to_be_decoded = np.copy(layer)
        to_be_decoded[to_be_decoded == 4] = 0
        to_be_decoded[to_be_decoded > 0] = 1

        level_blocks = self.decode_layer(to_be_decoded, -1)

        multi_dim = np.zeros((layer.shape[0], layer.shape[1], 3))
        for i in range(1, 4):
            multi_dim[layer == i, i - 1] = 1

        # Get block material by going over each block and check which material is the most confident at this location
        for block in level_blocks:
            start, end, bottom, top = self.get_range_of_block(block['block'], block['position'], layer)
            material = np.argmax(np.sum(multi_dim[bottom:top, start:end], axis = (0, 1)))
            block['material'] = material

        pig_layer = np.copy(layer)
        pig_layer[pig_layer != 4] = 0
        bird_positions = self.get_pig_position(pig_layer)
        created_level_elements = self.create_level_elements(level_blocks, bird_positions)

        if self.recalibrate_blocks:
            created_level_elements = recalibrate_blocks(created_level_elements)

        created_level = Level.create_level_from_structure(created_level_elements)
        return created_level

    def decode_layer(self, layer, layer_idx):
        layer = np.flip(layer, axis = 0)
        # Normalize Layer
        layer = (layer - np.min(layer)) / (np.max(layer) - np.min(layer))

        if self.round_to_next_int:
            layer = np.rint(layer)

        # Unify the lowest values
        highest_lowest_value = self.get_cutoff_point(layer)
        layer[layer <= highest_lowest_value] = 0

        trimmed_img, trim_data = DecoderUtils.trim_img(layer, ret_trims = True)
        self.visualizer.plot_img(trimmed_img, title = 'Trimmed Img', flip = True)

        hit_probabilities, size_ranking = self.create_confidence_matrix(trimmed_img)

        if self.display_decoding:
            self.visualizer.plot_matrix_complete(hit_probabilities, blocks = self.blocks, title = 'Hit Probabilities', flipped = True)
            self.visualizer.plot_matrix_complete(size_ranking, blocks = self.blocks, title = 'Size Ranking', flipped = True)

        stop_condition = np.sum(trimmed_img[trimmed_img > 0]).item()

        # Create a ranking depending on the hit probability and the covering of the block
        current_size_ranking = hit_probabilities * size_ranking
        rounded_block_rankings = np.around(current_size_ranking, 5)  # Round to remove floating point errors

        if self.display_decoding:
            self.visualizer.plot_matrix_complete(
                rounded_block_rankings, blocks = self.blocks,
                title = 'Selection Rankings', flipped = True,
                sub_figure_names = [block['name'] + (' (Vert) ' if block['rotated'] else ' ') +
                                    f'{np.round(np.max(hit_probabilities[:, :, block_idx]).item() * 100) / 100} / ' +
                                    f'{np.round(np.max(size_ranking[:, :, block_idx]).item() * 100) / 100}'
                                    for block_idx, block in enumerate(self.blocks)]
            )

        rounded_block_rankings[hit_probabilities <= self.cutoff_point] = 0
        if self.display_decoding:
            self.visualizer.plot_matrix_complete(rounded_block_rankings, blocks = self.blocks, title = 'Hit Probabilities clipped', flipped = True)
        selected_blocks = self.select_blocks(rounded_block_rankings, [], stop_condition, _covered_area = 0)

        ret_blocks = []

        top_space, bottom_space, left_space, right_space = trim_data
        if selected_blocks is not None:
            for block in selected_blocks:
                block['position'][0] += top_space
                block['position'][1] += left_space
                block['material'] = layer_idx
                ret_blocks.append(block)

        return ret_blocks

    def create_confidence_matrix(self, layer):
        avg_results = []
        sum_results = []
        
        for idx, possible_block in enumerate(self.block_data.values()):
            block_width = possible_block['width']
            block_height = possible_block['height']
            sum_convolution_kernel = np.ones((block_height + 1, block_width + 1))

            avg_convolution_kernel = sum_convolution_kernel / np.sum(sum_convolution_kernel)

            if self.custom_kernel_scale:
                sum_convolution_kernel = sum_convolution_kernel * possible_block['scalar']

                # norm_ratio = (possible_block['width'] / possible_block['height']) / self.max_ratio
                # scalar = 1 - (norm_ratio * self.custom_kernel_scale)
                #
                # sum_convolution_kernel = sum_convolution_kernel * (1 - scalar)

            if self.minus_one_border:
                sum_convolution_kernel = np.pad(sum_convolution_kernel, 1, 'constant', constant_values = -1)

            pad_value = self.negative_air_value if self.use_negative_air_value else 0

            sum_img = np.copy(layer)
            if self.use_negative_air_value:
                sum_img = (sum_img * 2) - 1
                sum_img[sum_img == -1] = self.negative_air_value

            pad_size = np.max(sum_convolution_kernel.shape)

            sum_padded_img = np.pad(sum_img, pad_size, 'constant', constant_values = pad_value)
            sum_result = cv2.filter2D(sum_padded_img, -1, sum_convolution_kernel)[pad_size:-pad_size, pad_size:-pad_size]

            avg_padded_img = np.pad(layer, pad_size, 'constant', constant_values = 0)
            avg_result = cv2.filter2D(np.rint(avg_padded_img), -1, avg_convolution_kernel)[pad_size:-pad_size, pad_size:-pad_size]

            avg_results.append(avg_result)
            sum_results.append(sum_result)

        hit_probabilities = np.stack(avg_results, axis = -1)
        size_ranking = np.stack(sum_results, axis = -1)

        return hit_probabilities, size_ranking

    def get_pig_position(self, bird_layer):
        current_img = np.copy(bird_layer)

        current_img = np.flip(current_img, axis = 0)

        highest_lowest_value = self.get_cutoff_point(bird_layer)
        current_img[current_img <= highest_lowest_value] = 0

        kernel = MathUtil.get_circular_kernel(7)
        kernel = kernel / np.sum(kernel)
        padded_img = np.pad(current_img, 6, 'constant', constant_values = -1)
        bird_probabilities = cv2.filter2D(padded_img, -1, kernel)[6:-6, 6:-6]

        bird_probabilities[bird_probabilities < self.bird_cutoff] = 0
        trimmed_bird_img, trim_data = DecoderUtils.trim_img(bird_probabilities, ret_trims = True)

        max_height, max_width = trimmed_bird_img.shape
        top_space, bottom_space, left_space, right_space = trim_data

        bird_positions = []
        trim_counter = 0
        while not np.all(trimmed_bird_img < 0.00001):
            bird_position = np.unravel_index(np.argmax(trimmed_bird_img), trimmed_bird_img.shape)
            y, x = bird_position
            bird_positions.append([
                top_space + y, left_space + x
            ])

            # Remove the location the bird was picked
            x_cords = np.arange(x - 6, x + 6 + 1, 1)
            y_cords = np.arange(y - 6, y + 6 + 1, 1)
            x_cords[x_cords < 0] = 0
            x_cords[x_cords >= max_width] = max_width - 1
            y_cords[y_cords < 0] = 0
            y_cords[y_cords >= max_height] = max_height - 1
            x_pos, y_pos = np.meshgrid(y_cords, x_cords)

            trimmed_bird_img[x_pos, y_pos] = 0

            trim_counter += 1

        return bird_positions

    def select_blocks(self, _block_rankings, _selected_blocks: List, _stop_condition: float, _covered_area: float = 0):
        if self.display_decoding:
            print(_covered_area, _stop_condition)
        # if _stop_condition - _covered_area < 1:
        #     return _selected_blocks

        # select the most probable block that is also the biggest
        next_block = np.unravel_index(np.argmax(_block_rankings), _block_rankings.shape)

        # Extract the block
        selected_block = self.blocks[next_block[-1]]
        block_position = np.array(next_block[0:2])

        if self.display_decoding:
            description = f"Selected Block: {selected_block['name']} with {_block_rankings[next_block]} area with {len(_selected_blocks)} selected"
            print(description)

        # Remove all blocks that cant work with that blocks together
        next_block_rankings = self.delete_blocks(_block_rankings, selected_block['idx'], block_position)
        if self.display_decoding:
            self.display_block_deletion(np.around(_block_rankings, 5), block_position, selected_block)

        next_blocks = _selected_blocks.copy()
        next_blocks.append(dict(
            position = block_position,
            block = selected_block,
        ))
        next_covered_area = _covered_area + ((selected_block['width'] + 1) * (selected_block['height'] + 1))

        if np.all(next_block_rankings < 0.00001):
            return next_blocks

        return self.select_blocks(next_block_rankings, next_blocks, _stop_condition, next_covered_area)

    def display_block_deletion(self, _block_rankings, position, selected_block):
        delete_matrix_data = self.delete_matrices[selected_block['idx']]
        delete_rectangles = delete_matrix_data['delete_rectangles']
        y_pad, x_pad, _ = delete_matrix_data['matrix'].shape
        y_max, x_max, _ = _block_rankings.shape

        current_delete_rectangles = []
        for delete_rectangle in delete_rectangles:
            (del_start, del_end, del_top, del_bottom) = delete_rectangle
            start = 0 if position[1] + del_start < 0 else position[1] + del_start
            end = x_max - 1 if position[1] + del_end > x_max else position[1] + del_end
            top = 0 if position[0] + del_top < 0 else position[0] + del_top
            bottom = y_max - 1 if position[0] + del_bottom > y_max else position[0] + del_bottom
            current_delete_rectangles.append((start, end, top, bottom))

        self.visualizer.plot_matrix_complete(
            _block_rankings,
            blocks = self.blocks,
            title = f"Selected Block: {selected_block['name']}",
            selected_block = selected_block['idx'],
            position = position,
            delete_rectangles = current_delete_rectangles,
            flipped = True,
            save_name = "selected_block"
        )

    def delete_blocks(self, _block_rankings, _selected_block_idx, _block_position):
        current_delete_matrix_data = self.delete_matrices[_selected_block_idx]
        current_delete_matrix = current_delete_matrix_data['matrix']
        y_pad, x_pad, _ = current_delete_matrix.shape
        padded_block_rankings = np.pad(_block_rankings, ((y_pad, y_pad), (x_pad, x_pad), (0, 0)), 'constant', constant_values=0)

        delete_matrix_shape = current_delete_matrix_data['shape'][:2]
        top_stop, left_stop = _block_position - (delete_matrix_shape // 2) + delete_matrix_shape
        bottom_stop, right_stop = _block_position + (delete_matrix_shape // 2) + delete_matrix_shape + (delete_matrix_shape % 2)
        padded_block_rankings[top_stop: bottom_stop, left_stop: right_stop] = \
            padded_block_rankings[top_stop: bottom_stop, left_stop: right_stop] * current_delete_matrix

        return padded_block_rankings[y_pad: -y_pad, x_pad: -x_pad]

    @staticmethod
    def create_delete_block_matrices(_blocks, visualizer = None):
        """
        Function that creates a matrix that deletes the possible blocks at this location over the layer matrix
        """

        def _get_block_ranges(blocks):
            """
            Internal function that creates a list for each block for each range
            """
            ret_ranges = dict()

            for center_block_idx, center_block in enumerate(blocks):

                current_list = list()

                left_extension = (center_block['width'] + 1) // 2
                right_extension = ((center_block['width'] + 1) // 2)
                top_extension = (center_block['height'] + 1) // 2
                bottom_extension = ((center_block['height'] + 1) // 2)

                x_max = -np.inf
                y_max = -np.inf

                for outside_block_idx, outside_block in enumerate(blocks):
                    start = -(((outside_block['width'] + 1) // 2) + left_extension) + (outside_block['width'] % 2)
                    end = (((outside_block['width'] + 1) // 2) + right_extension) - (outside_block['width'] % 2)

                    top = -(((outside_block['height'] + 1) // 2) + top_extension) + (outside_block['height'] % 2)
                    bottom = (((outside_block['height'] + 1) // 2) + bottom_extension) - (outside_block['height'] % 2)

                    x_cords = np.arange(start, end + 1, 1)
                    y_cords = np.arange(top, bottom + 1, 1)

                    x_max = len(x_cords) if len(x_cords) > x_max else x_max
                    y_max = len(y_cords) if len(y_cords) > y_max else y_max

                    current_list.append((x_cords, y_cords))

                ret_ranges[center_block_idx] = dict(
                    name = center_block['name'],
                    range_list = current_list,
                    x_max = x_max,
                    y_max = y_max,
                    orig_rec = [-left_extension + (x_max // 2),
                                right_extension + (x_max // 2),
                                -top_extension + (y_max // 2),
                                bottom_extension + y_max // 2]
                )
            return ret_ranges

        block_ranges = _get_block_ranges(_blocks)
        ret_matrices = dict()

        for block_idx, block_range in block_ranges.items():
            range_list, x_max, y_max = block_range['range_list'], block_range['x_max'], block_range['y_max']
            multiply_matrix = np.ones((y_max, x_max, len(range_list)))

            center_pos_x, center_pos_y = x_max // 2, y_max // 2
            delete_rectangles = []

            for layer_idx, (x_cords, y_cords) in enumerate(range_list):
                current_x_cords = x_cords + center_pos_x
                current_y_cords = y_cords + center_pos_y
                x_pos, y_pos = np.meshgrid(current_y_cords, current_x_cords)
                multiply_matrix[x_pos, y_pos, layer_idx] = 0

                delete_rectangles.append([
                    np.min(x_cords), np.max(x_cords),
                    np.min(y_cords), np.max(y_cords)]
                )

            ret_matrices[block_idx] = dict(
                matrix = multiply_matrix,
                shape = np.array(multiply_matrix.shape),
                delete_rectangles = delete_rectangles
            )

            if visualizer is not None:
                width, height = _blocks[block_idx]['width'], _blocks[block_idx]['height']
                title = f'{block_range["name"]} {width}, {height}'
                delete_rectangles = [block_range['orig_rec'] for _ in range(len(_blocks))]
                visualizer.plot_matrix_complete(matrix = multiply_matrix,
                                                blocks = _blocks,
                                                title = title,
                                                add_max = False,
                                                delete_rectangles = delete_rectangles,
                                                position = [center_pos_y, center_pos_x])
                print(title)

        return ret_matrices

    def get_range_of_block(self, block, position, matrix):
        width, height = block['width'] + 1, block['height'] + 1
        start = position[1] - width // 2
        end = position[1] + width // 2 + width % 2
        top = position[0] - height // 2
        bottom = position[0] + height // 2 + height % 2

        end = end if end < matrix.shape[1] else matrix.shape[1] - 1
        bottom = bottom if bottom < matrix.shape[0] else matrix.shape[0] - 1

        return start, end, matrix.shape[0] - bottom, matrix.shape[0] - top

    def get_cutoff_point(self, layer):
        frequency, bins = np.histogram(layer, bins = 100)

        center = (bins[-1] - bins[0]) / 2
        highest_lowest_value = bins[0]
        for i in range(20):
            highest_count = frequency.argmax()
            highest_count_value = bins[highest_count]
            frequency[highest_count] = 0

            if highest_count_value < center:

                if highest_lowest_value < highest_count_value:
                    highest_lowest_value = highest_count_value

        return highest_lowest_value

    @staticmethod
    def create_level_elements(blocks, pig_position):
        ret_level_elements = []
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            block_attribute = dict(
                type = block['block']['name'],
                material = Constants.materials[block['material'] - 1],
                x = block['position'][1] * Constants.resolution,
                y = block['position'][0] * Constants.resolution,
                rotation = 90 if block['block']['rotated'] else 0
            )
            element = LevelElement(id = block_idx, **block_attribute)
            element.create_set_geometry()
            ret_level_elements.append(element)
        block_idx += 1

        for pig_idx, pig in enumerate(pig_position):
            pig_attribute = dict(
                type = "BasicSmall",
                material = None,
                x = pig[1] * Constants.resolution,
                y = pig[0] * Constants.resolution,
                rotation = 0
            )
            element = LevelElement(id = pig_idx + block_idx, **pig_attribute)
            element.create_set_geometry()
            ret_level_elements.append(element)

        return ret_level_elements


if __name__ == '__main__':
    decoder = MultiLayerStackDecoder()
    decoder.visualizer.plot_show_immediately = True
    delete_matrix = decoder.create_delete_block_matrices(decoder.blocks, decoder.visualizer)


