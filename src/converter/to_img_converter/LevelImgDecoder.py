import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from shapely.affinity import translate
from shapely.geometry import Polygon

from converter import MathUtil
from converter.to_img_converter import DecoderUtils
from level import Constants
from level.Level import Level
from level.LevelElement import LevelElement
from level.LevelVisualizer import LevelVisualizer
from util.Config import Config


class LevelImgDecoder:

    def __init__(self):
        self.config = Config.get_instance()
        self.block_data = self.config.get_encoding_data(f"encoding_res_{Constants.resolution}")
        if type(self.block_data) is not str:
            self.resolution = self.block_data['resolution']
            del self.block_data['resolution']

        self.original_possible_width = list(
            np.unique([width for width in map(lambda x: x['width'], list(self.block_data.values()))]))
        self.original_possible_height = list(
            np.unique([height for height in map(lambda x: x['height'], list(self.block_data.values()))]))

        self.combined_possible_width = []
        self.combined_possible_height = []
        for combination in range(2, 4):
            self.combined_possible_width = list(np.unique(
                self.combined_possible_width + list(
                    map(lambda sum_val: sum_val + (combination - 1),
                        map(np.sum,
                            map(np.array, itertools.product(self.original_possible_width, repeat = combination))))
                )))

            self.combined_possible_height = list(np.unique(
                self.combined_possible_height + list(
                    map(lambda sum_val: sum_val + (combination - 1),
                        map(np.sum,
                            map(np.array, itertools.product(self.original_possible_height, repeat = combination))))
                )
            ))

        self.possible_width = self.original_possible_width + self.combined_possible_width
        self.possible_height = self.original_possible_height + self.combined_possible_height

        self.level_viz = LevelVisualizer()
        self.iterations = 0

    def decode_level(self, level_img, recalibrate_blocks = False):
        if level_img.shape[0] == 0 or level_img.shape[1] == 0:
            return None

        flipped = np.flip(level_img, axis = 0)
        level_img_8 = flipped.astype(np.uint8)
        top_value = np.max(level_img_8)

        no_birds = top_value != 4

        ret_blocks = []
        # Go over each contour color
        for contour_color in range(1, top_value - (-1 if no_birds else + 0)):

            # Get the contour through open cv
            current_img = np.copy(level_img_8)
            current_img[current_img != contour_color] = 0
            contours, _ = cv2.findContours(current_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Create the contour data list and sort by required area
            contour_data_list = [self.create_contour_dict(contour) for contour in contours]
            contour_data_list = sorted(contour_data_list, key = lambda x: x['min_x'])

            # Go over each found contour
            for contour_idx, contour_data in enumerate(contour_data_list):

                # If more then 4 contour points then search for rectangles
                poly = contour_data['poly']
                contour = contour_data['contour']
                rectangles = [contour]
                if len(contour) > 4:
                    rectangles, _ = MathUtil.get_rectangles(contour, poly)

                rect_dict = self.get_rectangle_data(rectangles)
                self.iterations = 0
                # Select the blocks and designate the correct contour color as material
                selected_blocks = self.select_blocks(
                    rectangles = rect_dict,
                    used_blocks = [],
                    required_area = contour_data['required_area'],
                    poly = poly
                )
                print(f'Iterations {self.iterations}')

                if selected_blocks is not None:
                    for selected_block in selected_blocks:
                        selected_block['material'] = contour_color
                    ret_blocks.append(selected_blocks)

        # Maybe do a bit of block adjustment to fit better
        # Only required between selected blocks i guess :D

        flattend_blocks = list(itertools.chain(*ret_blocks))
        for block in flattend_blocks:
            print(f"Block: {block['block_type']['name']} from: {block['rec_idx']}")

        pig_positions = []
        if not no_birds:
            pig_positions = self.get_pig_position(flipped)

        # Create block elements out of the possible blocks and the rectangle
        level_elements = self.create_level_elements(flattend_blocks, pig_positions)

        if recalibrate_blocks:
            level_elements = DecoderUtils.recalibrate_blocks(level_elements)

        return Level.create_level_from_structure(level_elements)

    @staticmethod
    def create_contour_dict(contour):
        contour_reshaped = contour.reshape((len(contour), 2))
        poly = Polygon(contour_reshaped)
        required_area = poly.area
        return dict(
            contour = contour_reshaped,
            poly = poly,
            required_area = required_area,
            min_x = contour_reshaped[:, 0].min()
        )

    def get_rectangle_data(self, rectangles, filter_rectangles = True):

        def _filter_rectangles(_rect_data):
            if _rect_data['width'] <= 1 or _rect_data['height'] <= 1:
                return False

            if _rect_data['width'] not in self.possible_width:
                return False

            if _rect_data['height'] not in self.possible_height:
                return False

            return True

        # Calc data for each rectangle required in the find blocks method
        rectangles_with_data = []
        for rec_idx, rectangle in enumerate(rectangles):
            rect_data = self.create_rect_dict(rectangle.reshape((4, 2)))
            rect_data['idx'] = rec_idx
            rectangles_with_data.append(rect_data)

        possible_rects = rectangles_with_data
        if filter_rectangles:
            possible_rects = list(filter(_filter_rectangles, rectangles_with_data))

        # Sort the rectangles by area and create a dictionary out of them
        sorted_rectangles = sorted(possible_rects, key = lambda x: x['area'], reverse = True)
        rect_dict = dict()
        for rec_idx, rec in enumerate(sorted_rectangles):
            rect_dict[rec_idx] = rec

        return rect_dict

    def get_pig_position(self, level_img):

        level_img_8 = level_img.astype(np.uint8)
        top_value = np.max(level_img_8)
        current_img = np.copy(level_img_8)
        current_img[current_img != top_value] = 0

        kernel = MathUtil.get_circular_kernel(6)
        erosion = cv2.erode(current_img, kernel, iterations = 1)
        contours, _ = cv2.findContours(erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        bird_positions = []
        for contour in contours:
            contour_reshaped = contour.reshape((len(contour), 2))
            pos = np.average(contour_reshaped, axis = 0)
            bird_positions.append(pos)

        return bird_positions

    def select_blocks(self, rectangles, used_blocks, required_area, poly, occupied_area = 0):
        self.iterations += 1

        # Break condition
        if occupied_area != 0 and abs(required_area / occupied_area - 1) < 0.05:
            print(f'{required_area} / {occupied_area}')
            return used_blocks

        if occupied_area > required_area or len(rectangles) == 0:
            return None

        # Filter rectangles for existing blocks
        # check if rec overlaps a existing block significantly
        self.filter_rectangles_by_used_blocks(rectangles, used_blocks)

        if len(rectangles) == 0:
            return None

        # check if remaining rectangles are able to fill the shape approximetly
        combined_area = np.sum([rec['poly'].area for rec in rectangles.values()])
        if abs((combined_area + occupied_area) / required_area) < 0.8:
            return None

        # Go over each rectangle
        for rec_idx, rec in rectangles.items():

            if rec['height'] in self.original_possible_height and rec['width'] in self.original_possible_width:
                # Search for matching block sizes
                for block_idx, block in self.block_data.items():
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
                    add_area = self.get_area_between_used_blocks(new_block, next_used_blocks)
                    if add_area == -1:
                        continue

                    next_used_blocks.append(new_block)

                    selected_blocks = self.select_blocks(
                        rectangles = next_rectangles,
                        used_blocks = next_used_blocks,
                        required_area = required_area,
                        occupied_area = occupied_area + rec['area'] + add_area,
                        poly = poly
                    )

                    if selected_blocks is not None:
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
                    k: _block for k, _block in self.block_data.items()
                    if (_block[primary_orientation] + 2) / rec[primary_orientation] - 1 < 0.001 and \
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
                        combined_height = np.sum(list(map(lambda _block: _block[1][primary_orientation], combination))) \
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

                                new_dict = self.create_rect_dict(rectangle)

                                if new_dict['width'] <= 1 or new_dict['height'] <= 1 or \
                                        new_dict['width'] not in self.possible_width or \
                                        new_dict['height'] not in self.possible_height:
                                    continue

                                next_rectangles[len(rectangles)] = new_dict
                                all_space_used = False

                            # Create the blocks of each block from bottom to top
                            next_used_blocks = used_blocks.copy()
                            start_value = ry_1 if primary_orientation == 'height' else rx_1
                            used_area = 0
                            for block_idx, block in combination:
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
                                    rec = self.create_rect_dict(block_rectangle),
                                    rec_idx = rec_idx
                                )
                                add_area = self.get_area_between_used_blocks(new_block, next_used_blocks)
                                used_area += block['area'] + add_area - (1 if add_area > 0 else 0)
                                next_used_blocks.append(new_block)
                                start_value += block[f'{primary_orientation}'] + 1

                            # Remove the current big rectangle
                            del next_rectangles[rec_idx]

                            selected_blocks = self.select_blocks(
                                rectangles = next_rectangles,
                                used_blocks = next_used_blocks,
                                required_area = required_area,
                                poly = poly,
                                occupied_area = occupied_area + used_area
                            )

                            if selected_blocks is not None:
                                return selected_blocks

                        # This means the block were to big which means doesnt fit
                        if height_difference < 0:
                            to_big_counter += 1

                    # If all blocks combined were to big, we dont need to check more block combinations
                    if to_big_counter == combi_counter:
                        break

        # We tested everything and nothing worked :(
        return None

    @staticmethod
    def filter_rectangles_by_used_blocks(rectangles, used_blocks):
        for rec_idx, rec in rectangles.copy().items():
            for used_block in used_blocks:
                block_rec = used_block['rec']

                overlapping = rec['poly'].intersection(block_rec['poly']).area
                if overlapping > 0:
                    del rectangles[rec_idx]
                    break

                distance = rec['poly'].distance(block_rec['poly'])
                if distance == 0:
                    del rectangles[rec_idx]
                    break

    @staticmethod
    def get_area_between_used_blocks(new_block, used_blocks):

        ret_area = 0
        for selected_block in used_blocks:
            distance = new_block['rec']['poly'].distance(selected_block['rec']['poly'])
            if distance == 0:
                # That should not happen
                return -1

            if distance == 1:
                major_vec = np.rint(new_block['rec']['center_pos'] - selected_block['rec']['center_pos'])
                direct_vec = np.copy(major_vec)
                direct_vec[0] = 0 if direct_vec[0] == 0 else (-2 if direct_vec[0] > 1 else 2)
                direct_vec[1] = 0 if direct_vec[1] == 0 else (-2 if direct_vec[1] > 1 else 2)
                poly = translate(new_block['rec']['poly'], xoff = direct_vec[0], yoff = direct_vec[1])
                intersection = poly.intersection(selected_block['rec']['poly'])
                ret_area += intersection.area + 1

        return ret_area

    @staticmethod
    def create_level_elements(blocks, pig_position):
        ret_level_elements = []
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            block_attribute = dict(
                type = block['block_type']['name'],
                material = Constants.materials[block['material'] - 1],
                x = np.average(block['rec']['rectangle'][:, 0]) * Constants.resolution,
                y = np.average(block['rec']['rectangle'][:, 1]) * Constants.resolution,
                rotation = 90 if block['block_type']['rotated'] else 0
            )
            element = LevelElement(id = block_idx, **block_attribute)
            element.create_set_geometry()
            ret_level_elements.append(element)
        block_idx += 1

        for pig_idx, pig in enumerate(pig_position):
            pig_attribute = dict(
                type = "BasicSmall",
                material = None,
                x = pig[0] * Constants.resolution,
                y = pig[1] * Constants.resolution,
                rotation = 0
            )
            element = LevelElement(id = pig_idx + block_idx, **pig_attribute)
            element.create_set_geometry()
            ret_level_elements.append(element)
        return ret_level_elements

    def create_rect_dict(self, rectangle, poly = None):
        sorted_rectangle = rectangle[np.lexsort((rectangle[:, 1], rectangle[:, 0]))]
        sorted_rectangle[[2, 3]] = sorted_rectangle[[3, 2]]
        ret_dict = dict(rectangle = sorted_rectangle)
        if poly is not None:
            ret_dict['contour_area'] = poly.area

        ret_dict['poly'] = Polygon(rectangle)

        for key, value in MathUtil.get_contour_dims(rectangle).items():
            ret_dict[key] = value
        return ret_dict

    def get_rectangles(self, level_img, material_id = 1):
        # Create a copy of the img to manipulate it for the contour finding
        current_img = np.ndarray.copy(level_img)
        current_img = current_img.astype(np.uint8)
        current_img[current_img != material_id] = 0

        # get the contours
        contours, _ = cv2.findContours(current_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        ret_list = []
        for contour_idx, contour in enumerate(contours):
            contour_reshaped = contour.reshape((len(contour), 2))
            poly = Polygon(contour_reshaped)

            rectangles, contour_list = MathUtil.get_rectangles(contour_reshaped, poly)
            rect_data = list(map(lambda rectangle: self.create_rect_dict(rectangle.reshape((4, 2)), poly), rectangles))
            ret_list += rect_data

        return ret_list
