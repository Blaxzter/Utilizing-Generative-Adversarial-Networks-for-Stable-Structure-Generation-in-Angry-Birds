import multiprocessing
import time
from copy import deepcopy
from math import ceil
from pathlib import Path
from random import randint
from random import uniform

from loguru import logger
from tqdm import tqdm


# noinspection PyPep8
class BaselineGenerator:

    def __init__(self):
        # blocks number and size
        self.blocks = {'1': [0.84, 0.84], '2': [0.85, 0.43], '3': [0.43, 0.85], '4': [0.43, 0.43], '5': [0.22, 0.22],
                       '6': [0.43, 0.22], '7': [0.22, 0.43], '8': [0.85, 0.22], '9': [0.22, 0.85], '10': [1.68, 0.22],
                       '11': [0.22, 1.68], '12': [2.06, 0.22], '13': [0.22, 2.06]}

        # blocks number and name
        # (blocks 3, 7, 9, 11 and 13) are their respective block names rotated 90 derees clockwise
        self.block_names = {'1': "SquareHole", '2': "RectFat", '3': "RectFat", '4': "SquareSmall", '5': "SquareTiny",
                            '6': "RectTiny", '7': "RectTiny", '8': "RectSmall", '9': "RectSmall", '10': "RectMedium",
                            '11': "RectMedium", '12': "RectBig", '13': "RectBig"}

        # additional objects number and name
        self.additional_objects = {'1': "TriangleHole", '2': "Triangle", '3': "Circle", '4': "CircleSmall"}

        # additional objects number and size
        self.additional_object_sizes = {'1': [0.82, 0.82], '2': [0.82, 0.82], '3': [0.8, 0.8], '4': [0.45, 0.45]}

        # blocks number and probability of being selected
        self.probability_table_blocks = {'1': 0.10, '2': 0.10, '3': 0.10, '4': 0.05, '5': 0.02, '6': 0.05, '7': 0.05,
                                         '8': 0.10, '9': 0.05, '10': 0.16, '11': 0.04, '12': 0.16, '13': 0.02}

        # materials that are available
        self.materials = ["wood", "stone", "ice"]

        # bird types number and name
        self.bird_names = {'1': "BirdRed", '2': "BirdBlue", '3': "BirdYellow", '4': "BirdBlack", '5': "BirdWhite"}

        # bird types number and probability of being selected
        self.bird_probabilities = {'1': 0.35, '2': 0.2, '3': 0.2, '4': 0.15, '5': 0.1}

        self.TNT_block_probability = 0.3

        self.pig_size = [0.5, 0.5]  # size of pigs

        self.platform_size = [0.62, 0.62]  # size of platform sections

        self.edge_buffer = 0.11  # buffer uesd to push edge blocks further into the structure center (increases stability)

        self.absolute_ground = -3.5  # the position of ground within level

        self.max_peaks = 5  # maximum number of peaks a structure can have (up to 5)
        self.min_peak_split = 10  # minimum distance between two peak blocks of structure
        self.max_peak_split = 50  # maximum distance between two peak blocks of structure

        self.minimum_height_gap = 0  # y distance min between platforms
        self.platform_distance_buffer = 0.4  # x_distance min between platforms / y_distance min between platforms and ground structures

        # defines the levels area (ie. space within which structures/platforms can be placed)
        self.level_width_min = -3.0
        self.level_width_max = 3.0
        self.level_height_min = -2.0  # only used by platforms, ground structures use absolute_ground to determine their lowest point
        self.level_height_max = 4.0

        self.pig_precision = 0.01  # how precise to check for possible pig positions on ground

        self.min_ground_width = 4.5  # minimum amount of space allocated to ground structure
        # desired height limit of ground structures
        self.ground_structure_height_limit = ((self.level_height_max - self.minimum_height_gap) - self.absolute_ground) / 1.5

        self.max_attempts = 100  # number of times to attempt to place a platform before abandoning it

        # Add parameter required
        self.trihole_allowed = True
        self.tri_allowed = True
        self.cir_allowed = True
        self.cirsmall_allowed = True
        self.TNT_allowed = True

        self.ground_structure_range = (1, 1)
        self.air_structure_range = (1, 1)

        self.number_ground_structures = randint(self.ground_structure_range[0], self.ground_structure_range[1])
        self.final_pig_positions = None
        self.complete_locations = None

        self.final_TNT_positions = []
        self.final_platforms = []

        # Level Generation settings
        self.number_levels = 1
        self.pig_range = [2, 3]
        self.restricted_combination = ""
        self.use_triangles = False
        self.use_circles = False

        self.restricted_blocks = []

    def settings(self, number_levels = 1, pig_range = "1,5", use_triangles = False, use_circles = False,
                 restricted_combination = "", ground_structure_range = (1, 1), air_structure_range = (1, 1),
                 level_width_min = -3.0, level_width_max = 3.0, level_height_min = -2.0, level_height_max = 4.0,
                 materials = None, min_ground_width = 4.5):

        self.use_triangles = use_triangles
        self.use_circles = use_circles
        self.number_levels = number_levels
        self.pig_range = pig_range.split(",")
        self.restricted_combination = restricted_combination
        self.ground_structure_range = ground_structure_range
        self.air_structure_range = air_structure_range

        self.level_width_min = level_width_min
        self.level_width_max = level_width_max
        self.level_height_min = level_height_min
        self.level_height_max = level_height_max
        self.min_ground_width = min_ground_width
        self.materials = ["wood", "stone", "ice"] if materials is None else materials

    def generate_level_init(self, folder_path = "./", start_level_index = 4):
        # generate levels using input parameters

        backup_probability_table_blocks = deepcopy(self.probability_table_blocks)
        backup_materials = deepcopy(self.materials)

        restricted_combinations = self.restricted_combination.split(',')
        for i in range(len(restricted_combinations)):
            # if all materials are baned for a block type then do not use that block type
            restricted_combinations[i] = restricted_combinations[i].split()

        self.restricted_blocks = []  # block types that cannot be used with any materials
        for key, value in self.block_names.items():
            completely_restricted = True
            for material in self.materials:
                if [material, value] not in restricted_combinations:
                    completely_restricted = False
            if completely_restricted:
                self.restricted_blocks.append(value)

        self.probability_table_blocks = deepcopy(backup_probability_table_blocks)
        self.trihole_allowed = self.use_triangles
        self.tri_allowed = self.use_triangles
        self.cir_allowed = self.use_circles
        self.cirsmall_allowed = self.use_circles
        self.TNT_allowed = True

        # remove restricted block types from the structure generation process
        self.probability_table_blocks = self.remove_blocks(self.restricted_blocks)
        if "TriangleHole" in self.restricted_blocks:
            self.trihole_allowed = False
        if "Triangle" in self.restricted_blocks:
            self.tri_allowed = False
        if "Circle" in self.restricted_blocks:
            self.cir_allowed = False
        if "CircleSmall" in self.restricted_blocks:
            self.cirsmall_allowed = False

        for current_level in (pbar := tqdm(range(self.number_levels))):
            try:
                queue = multiprocessing.Queue()
                th = multiprocessing.Process(target = self.create_level, args = (
                    current_level, folder_path, restricted_combinations, start_level_index, queue))
                th.start()
                counter = 0
                if queue.empty():
                    while True:
                        if queue.empty():
                            time.sleep(0.1)
                            counter += 1
                            if counter > 100:
                                print("Kill the process")
                                th.terminate()
                                th.kill()
                                th.close()
                                break
                        elif queue.get_nowait():
                            break

                if counter < 100:
                    pbar.set_description("Processing %s" % counter)
                    th.join()

            except Exception as e:
                continue

    def create_level(self, current_level, folder_path, restricted_combinations, start_level_index, finished):
        logger.debug(f"Create Level {current_level} {finished}")
        self.number_ground_structures = randint(self.ground_structure_range[0],
                                                self.ground_structure_range[1])  # number of ground structures
        number_platforms = randint(self.air_structure_range[0], self.air_structure_range[
            1])  # number of platforms (reduced automatically if not enough space)
        # number of pigs (if set too large then can cause program to infinitely loop)
        number_pigs = randint(int(self.pig_range[0]), int(self.pig_range[1]))
        if (current_level + start_level_index) < 10:
            level_name = "0" + str(current_level + start_level_index)
        else:
            level_name = str(current_level + start_level_index)
        number_ground_structures, complete_locations, final_pig_positions = self.create_ground_structures()
        number_platforms, final_platforms, platform_centers \
            = self.create_platforms(number_platforms, complete_locations, final_pig_positions)
        complete_locations, final_pig_positions \
            = self.create_platform_structures(final_platforms, platform_centers, complete_locations,
                                              final_pig_positions)
        final_pig_positions, removed_pigs = self.remove_unnecessary_pigs(number_pigs)
        final_pig_positions = self.add_necessary_pigs(number_pigs)
        self.final_TNT_positions = self.add_TNT(removed_pigs)
        number_birds = self.choose_number_birds(final_pig_positions, number_ground_structures,
                                                number_platforms)
        possible_trihole_positions, possible_tri_positions, possible_cir_positions, possible_cirsmall_positions = self.find_additional_block_positions(
            complete_locations)
        selected_other = self.add_additional_blocks(possible_trihole_positions, possible_tri_positions,
                                                    possible_cir_positions, possible_cirsmall_positions)
        self.write_level_xml(complete_locations, selected_other, final_pig_positions, self.final_TNT_positions,
                             final_platforms, number_birds, level_name, restricted_combinations, folder_path)
        finished.put(True)
        logger.debug(f"Level Created {current_level}\n")

    def generate_subsets(self, current_tree_bottom):
        """
        Generates a list of all possible subsets for structure bottom
        :param current_tree_bottom:
        :return:
        """
        current_distances = []
        subsets = []
        current_point = 0
        while current_point < len(current_tree_bottom) - 1:
            current_distances.append(current_tree_bottom[current_point + 1][1] - current_tree_bottom[current_point][1])
            current_point = current_point + 1

        # remove similar splits causesd by floating point imprecision
        for i in range(len(current_distances)):
            current_distances[i] = round(current_distances[i], 10)

        split_points = list(set(current_distances))  # all possible x-distances between bottom blocks

        for i in split_points:  # subsets based on differences between x-distances
            current_subset = []
            start_point = 0
            end_point = 1
            for j in current_distances:
                if j >= i:
                    current_subset.append(current_tree_bottom[start_point:end_point])
                    start_point = end_point
                end_point = end_point + 1

            current_subset.append(current_tree_bottom[start_point:end_point])

            subsets.append(current_subset)

        subsets.append([current_tree_bottom])

        return subsets

    def find_subset_center(self, subset):
        """
        Finds the center positions of the given subset
        :param subset:
        :return:
        """
        if len(subset) % 2 == 1:
            return subset[(len(subset) - 1) // 2][1]
        else:
            return (subset[len(subset) // 2][1] - subset[(len(subset) // 2) - 1][1]) / 2.0 + \
                   subset[(len(subset) // 2) - 1][1]

    def find_subset_edges(self, subset):
        """
        Finds the edge positions of the given subset
        :param subset:
        :return:
        """
        edge1 = subset[0][1] - (self.blocks[str(subset[0][0])][0]) / 2.0 + self.edge_buffer
        edge2 = subset[-1][1] + (self.blocks[str(subset[-1][0])][0]) / 2.0 - self.edge_buffer
        return [edge1, edge2]

    def check_valid(self, grouping, choosen_item, current_tree_bottom, new_positions):
        """
        checks that positions for new block dont overlap and support the above blocks
        :param grouping:
        :param choosen_item:
        :param current_tree_bottom:
        :param new_positions:
        :return:
        """

        # check no overlap
        i = 0
        while i < len(new_positions) - 1:
            if (new_positions[i] + (self.blocks[str(choosen_item)][0]) / 2) > (
                    new_positions[i + 1] - (self.blocks[str(choosen_item)][0]) / 2):
                return False
            i = i + 1

        # check if each structural bottom block's edges supported by new blocks
        for item in current_tree_bottom:
            edge1 = item[1] - (self.blocks[str(item[0])][0]) / 2
            edge2 = item[1] + (self.blocks[str(item[0])][0]) / 2
            edge1_supported = False
            edge2_supported = False
            for new in new_positions:
                if ((new - (self.blocks[str(choosen_item)][0]) / 2) <= edge1 and (
                        new + (self.blocks[str(choosen_item)][0]) / 2) >= edge1):
                    edge1_supported = True
                if ((new - (self.blocks[str(choosen_item)][0]) / 2) <= edge2 and (
                        new + (self.blocks[str(choosen_item)][0]) / 2) >= edge2):
                    edge2_supported = True
            if edge1_supported == False or edge2_supported == False:
                return False
        return True

    def check_center(self, grouping, choosen_item, current_tree_bottom):
        """
        check if new block can be placed under center of bottom row blocks validly
        :param grouping:
        :param choosen_item:
        :param current_tree_bottom:
        :return:
        """

        new_positions = []
        for subset in grouping:
            new_positions.append(self.find_subset_center(subset))
        return self.check_valid(grouping, choosen_item, current_tree_bottom, new_positions)

    def check_edge(self, grouping, choosen_item, current_tree_bottom):
        """
        check if new block can be placed under edges of bottom row blocks validly
        :param grouping:
        :param choosen_item:
        :param current_tree_bottom:
        :return:
        """
        new_positions = []
        for subset in grouping:
            new_positions.append(self.find_subset_edges(subset)[0])
            new_positions.append(self.find_subset_edges(subset)[1])
        return self.check_valid(grouping, choosen_item, current_tree_bottom, new_positions)

    def check_both(self, grouping, choosen_item, current_tree_bottom):
        """
        check if new block can be placed under both center and edges of bottom row blocks validly
        :param grouping:
        :param choosen_item:
        :param current_tree_bottom:
        :return:
        """
        new_positions = []
        for subset in grouping:
            new_positions.append(self.find_subset_edges(subset)[0])
            new_positions.append(self.find_subset_center(subset))
            new_positions.append(self.find_subset_edges(subset)[1])
        return self.check_valid(grouping, choosen_item, current_tree_bottom, new_positions)

    def choose_item(self, table):
        """
        choose a random item/block from the blocks dictionary based on probability table
        :param table:
        :return:
        """
        ran_num = uniform(0.0, 1.0)
        selected_num = 0
        while ran_num > 0:
            selected_num = selected_num + 1
            ran_num = ran_num - table[str(selected_num)]
        return selected_num

    def find_structure_width(self, structure):
        """
        finds the width of the given structure
        :param structure:
        :return:
        """
        min_x = 999999.9
        max_x = -999999.9
        for block in structure:
            if round((block[1] - (self.blocks[str(block[0])][0] / 2)), 10) < min_x:
                min_x = round((block[1] - (self.blocks[str(block[0])][0] / 2)), 10)
            if round((block[1] + (self.blocks[str(block[0])][0] / 2)), 10) > max_x:
                max_x = round((block[1] + (self.blocks[str(block[0])][0] / 2)), 10)
        return (round(max_x - min_x, 10))

    def find_structure_height(self, structure):
        """
        finds the height of the given structure
        :param structure:
        :return:
        """
        min_y = 999999.9
        max_y = -999999.9
        for block in structure:
            if round((block[2] - (self.blocks[str(block[0])][1] / 2)), 10) < min_y:
                min_y = round((block[2] - (self.blocks[str(block[0])][1] / 2)), 10)
            if round((block[2] + (self.blocks[str(block[0])][1] / 2)), 10) > max_y:
                max_y = round((block[2] + (self.blocks[str(block[0])][1] / 2)), 10)
        return (round(max_y - min_y, 10))

    def add_new_row(self, current_tree_bottom, total_tree):
        """
        adds a new row of blocks to the bottom of the structure
        :param current_tree_bottom:
        :param total_tree:
        :return:
        """
        groupings = self.generate_subsets(current_tree_bottom)  # generate possible groupings of bottom row objects
        choosen_item = self.choose_item(self.probability_table_blocks)  # choosen block for new row
        center_groupings = []  # collection of viable groupings with new block at center
        edge_groupings = []  # collection of viable groupings with new block at edges
        both_groupings = []  # collection of viable groupings with new block at both center and edges

        # check if new block is viable for each grouping in each position
        for grouping in groupings:
            if self.check_center(grouping, choosen_item, current_tree_bottom):  # check if center viable
                center_groupings.append(grouping)
            if self.check_edge(grouping, choosen_item, current_tree_bottom):  # check if edges viable
                edge_groupings.append(grouping)
            if self.check_both(grouping, choosen_item, current_tree_bottom):  # check if both center and edges viable
                both_groupings.append(grouping)

        # randomly choose a configuration (grouping/placement) from the viable options
        total_options = len(center_groupings) + len(edge_groupings) + len(both_groupings)  # total number of options
        if total_options > 0:
            option = randint(1, total_options)
            if option > len(center_groupings) + len(edge_groupings):
                selected_grouping = both_groupings[option - (len(center_groupings) + len(edge_groupings) + 1)]
                placement_method = 2
            elif option > len(center_groupings):
                selected_grouping = edge_groupings[option - (len(center_groupings) + 1)]
                placement_method = 1
            else:
                selected_grouping = center_groupings[option - 1]
                placement_method = 0

            # construct the new bottom row for structure using selected block/configuration
            new_bottom = []
            for subset in selected_grouping:
                if placement_method == 0:
                    new_bottom.append([choosen_item, self.find_subset_center(subset)])
                if placement_method == 1:
                    new_bottom.append([choosen_item, self.find_subset_edges(subset)[0]])
                    new_bottom.append([choosen_item, self.find_subset_edges(subset)[1]])
                if placement_method == 2:
                    new_bottom.append([choosen_item, self.find_subset_edges(subset)[0]])
                    new_bottom.append([choosen_item, self.find_subset_center(subset)])
                    new_bottom.append([choosen_item, self.find_subset_edges(subset)[1]])

            for i in new_bottom:
                i[1] = round(i[1], 10)  # round all values to prevent floating point inaccuracy from causing errors

            current_tree_bottom = new_bottom
            total_tree.append(current_tree_bottom)  # add new bottom row to the structure
            return total_tree, current_tree_bottom  # return the new structure

        else:
            # choose a new block and try again if no options available
            return self.add_new_row(current_tree_bottom, total_tree)

    def make_peaks(self, center_point):
        """
        creates the peaks (first row) of the structurev
        """

        current_tree_bottom = []  # bottom blocks of structure
        number_peaks = randint(1, self.max_peaks)  # this is the number of peaks the structure will have
        top_item = self.choose_item(self.probability_table_blocks)  # this is the item at top of structure

        if number_peaks == 1:
            current_tree_bottom.append([top_item, center_point])

        if number_peaks == 2:
            distance_apart_extra = round(randint(self.min_peak_split, self.max_peak_split) / 100.0, 10)
            current_tree_bottom.append(
                [top_item, round(center_point - (self.blocks[str(top_item)][0] * 0.5) - distance_apart_extra, 10)])
            current_tree_bottom.append(
                [top_item, round(center_point + (self.blocks[str(top_item)][0] * 0.5) + distance_apart_extra, 10)])

        if number_peaks == 3:
            distance_apart_extra = round(randint(self.min_peak_split, self.max_peak_split) / 100.0, 10)
            current_tree_bottom.append(
                [top_item, round(center_point - (self.blocks[str(top_item)][0]) - distance_apart_extra, 10)])
            current_tree_bottom.append([top_item, round(center_point, 10)])
            current_tree_bottom.append(
                [top_item, round(center_point + (self.blocks[str(top_item)][0]) + distance_apart_extra, 10)])

        if number_peaks == 4:
            distance_apart_extra = round(randint(self.min_peak_split, self.max_peak_split) / 100.0, 10)
            current_tree_bottom.append([top_item, round(
                center_point - (self.blocks[str(top_item)][0] * 1.5) - (distance_apart_extra * 2), 10)])
            current_tree_bottom.append(
                [top_item, round(center_point - (self.blocks[str(top_item)][0] * 0.5) - distance_apart_extra, 10)])
            current_tree_bottom.append(
                [top_item, round(center_point + (self.blocks[str(top_item)][0] * 0.5) + distance_apart_extra, 10)])
            current_tree_bottom.append([top_item, round(
                center_point + (self.blocks[str(top_item)][0] * 1.5) + (distance_apart_extra * 2), 10)])

        if number_peaks == 5:
            distance_apart_extra = round(randint(self.min_peak_split, self.max_peak_split) / 100.0, 10)
            current_tree_bottom.append([top_item, round(
                center_point - (self.blocks[str(top_item)][0] * 2.0) - (distance_apart_extra * 2), 10)])
            current_tree_bottom.append(
                [top_item, round(center_point - (self.blocks[str(top_item)][0]) - distance_apart_extra, 10)])
            current_tree_bottom.append([top_item, round(center_point, 10)])
            current_tree_bottom.append(
                [top_item, round(center_point + (self.blocks[str(top_item)][0]) + distance_apart_extra, 10)])
            current_tree_bottom.append([top_item, round(
                center_point + (self.blocks[str(top_item)][0] * 2.0) + (distance_apart_extra * 2), 10)])
        return current_tree_bottom

    def make_structure(self, absolute_ground, center_point, max_width, max_height):
        """
        Recursively adds rows to base of strucutre until max_width or max_height is passed
        once this happens the last row added is removed and the structure is returned
        :param center_point:
        :param max_width:
        :param max_height:
        :return:
        """
        total_tree = []  # all self.blocks of structure (so far)

        # creates the first row (peaks) for the structure, ensuring that max_width restriction is satisfied
        current_tree_bottom = self.make_peaks(center_point)
        if max_width > 0.0:
            while self.find_structure_width(current_tree_bottom) > max_width:
                current_tree_bottom = self.make_peaks(center_point)

        total_tree.append(current_tree_bottom)

        # recursively add more rows of self.blocks to the level structure
        structure_width = self.find_structure_width(current_tree_bottom)
        structure_height = (self.blocks[str(current_tree_bottom[0][0])][1]) / 2
        if max_height > 0.0 or max_width > 0.0:
            pre_total_tree = [current_tree_bottom]
            while structure_height < max_height and structure_width < max_width:
                total_tree, current_tree_bottom = self.add_new_row(current_tree_bottom, total_tree)
                complete_locations = []
                ground = absolute_ground
                for row in reversed(total_tree):
                    for item in row:
                        complete_locations.append(
                            [item[0], item[1], round((((self.blocks[str(item[0])][1]) / 2) + ground), 10)])
                    ground = ground + (self.blocks[str(item[0])][1])
                structure_height = self.find_structure_height(complete_locations)
                structure_width = self.find_structure_width(complete_locations)
                if structure_height > max_height or structure_width > max_width:
                    total_tree = deepcopy(pre_total_tree)
                else:
                    pre_total_tree = deepcopy(total_tree)

        # make structure vertically correct (add y position to blocks)
        complete_locations = []
        ground = absolute_ground
        for row in reversed(total_tree):
            for item in row:
                complete_locations.append(
                    [item[0], item[1], round((((self.blocks[str(item[0])][1]) / 2) + ground), 10)])
            ground = ground + (self.blocks[str(item[0])][1])

        logger.debug(f"Width: {self.find_structure_width(complete_locations)}")
        logger.debug(f"Height: {self.find_structure_height(complete_locations)}")
        logger.debug(f"Block number: {len(complete_locations)}")  # number self.blocks present in the structure

        # identify all possible pig positions on top of self.blocks (maximum 2 pigs per block, checks center before sides)
        possible_pig_positions = []
        for block in complete_locations:
            block_width = round(self.blocks[str(block[0])][0], 10)
            block_height = round(self.blocks[str(block[0])][1], 10)
            pig_width = self.pig_size[0]
            pig_height = self.pig_size[1]

            if self.blocks[str(block[0])][0] < pig_width:  # dont place block on edge if block too thin
                test_positions = [[round(block[1], 10), round(block[2] + (pig_height / 2) + (block_height / 2), 10)]]
            else:
                test_positions = [[round(block[1], 10), round(block[2] + (pig_height / 2) + (block_height / 2), 10)],
                                  [round(block[1] + (block_width / 3), 10),
                                   round(block[2] + (pig_height / 2) + (block_height / 2), 10)],
                                  [round(block[1] - (block_width / 3), 10),
                                   round(block[2] + (pig_height / 2) + (block_height / 2),
                                         10)]]  # check above centre of block
            for test_position in test_positions:
                valid_pig = True
                for i in complete_locations:
                    if (round((test_position[0] - pig_width / 2), 10) < round((i[1] + (self.blocks[str(i[0])][0]) / 2),
                                                                              10) and round(
                        (test_position[0] + pig_width / 2), 10) > round((i[1] - (self.blocks[str(i[0])][0]) / 2),
                                                                        10) and round(
                        (test_position[1] + pig_height / 2), 10) > round((i[2] - (self.blocks[str(i[0])][1]) / 2),
                                                                         10) and round(
                        (test_position[1] - pig_height / 2), 10) < round((i[2] + (self.blocks[str(i[0])][1]) / 2), 10)):
                        valid_pig = False
                if valid_pig:
                    possible_pig_positions.append(test_position)

        # identify all possible pig positions on ground within structure
        left_bottom = total_tree[-1][0]
        right_bottom = total_tree[-1][-1]
        test_positions = []
        x_pos = left_bottom[1]

        while x_pos < right_bottom[1]:
            test_positions.append([round(x_pos, 10), round(absolute_ground + (pig_height / 2), 10)])
            x_pos = x_pos + self.pig_precision

        for test_position in test_positions:
            valid_pig = True
            for i in complete_locations:
                if (round((test_position[0] - pig_width / 2), 10) < round((i[1] + (self.blocks[str(i[0])][0]) / 2),
                                                                          10) and round(
                    (test_position[0] + pig_width / 2), 10) > round((i[1] - (self.blocks[str(i[0])][0]) / 2),
                                                                    10) and round((test_position[1] + pig_height / 2),
                                                                                  10) > round(
                    (i[2] - (self.blocks[str(i[0])][1]) / 2), 10) and round((test_position[1] - pig_height / 2),
                                                                            10) < round(
                    (i[2] + (self.blocks[str(i[0])][1]) / 2), 10)):
                    valid_pig = False
            if valid_pig:
                possible_pig_positions.append(test_position)

        # randomly choose a pig position and remove those that overlap it, repeat until no more valid positions
        final_pig_positions = []
        while len(possible_pig_positions) > 0:
            pig_choice = possible_pig_positions.pop(randint(1, len(possible_pig_positions)) - 1)
            final_pig_positions.append(pig_choice)
            new_pig_positions = []
            for i in possible_pig_positions:
                if (round((pig_choice[0] - pig_width / 2), 10) >= round((i[0] + pig_width / 2), 10) or round(
                        (pig_choice[0] + pig_width / 2), 10) <= round((i[0] - pig_width / 2), 10) or round(
                    (pig_choice[1] + pig_height / 2), 10) <= round((i[1] - pig_height / 2), 10) or round(
                    (pig_choice[1] - pig_height / 2), 10) >= round((i[1] + pig_height / 2), 10)):
                    new_pig_positions.append(i)
            possible_pig_positions = new_pig_positions

        logger.debug(f"Pig number:  {len(final_pig_positions)}")

        return complete_locations, final_pig_positions

    def create_ground_structures(self):
        """
        divide the available ground space between the chosen number of ground structures
        :return:
        """
        valid = False
        while not valid:
            ground_divides = []
            if self.number_ground_structures > 0:
                ground_divides = [self.level_width_min, self.level_width_max]
            for i in range(self.number_ground_structures - 1):
                ground_divides.insert(i + 1, uniform(self.level_width_min, self.level_width_max))
            valid = True
            for j in range(len(ground_divides) - 1):
                if (ground_divides[j + 1] - ground_divides[j]) < self.min_ground_width:
                    valid = False

        # determine the area available to each ground structure
        ground_positions = []
        ground_widths = []
        for j in range(len(ground_divides) - 1):
            ground_positions.append(ground_divides[j] + ((ground_divides[j + 1] - ground_divides[j]) / 2))
            ground_widths.append(ground_divides[j + 1] - ground_divides[j])

        logger.debug(f"number ground structures: {len(ground_positions)}")

        # creates a ground structure for each defined area
        self.complete_locations = []
        self.final_pig_positions = []
        for i in range(len(ground_positions)):
            max_width = ground_widths[i]
            max_height = self.ground_structure_height_limit
            center_point = ground_positions[i]
            complete_locations2, final_pig_positions2 = self.make_structure(self.absolute_ground, center_point,
                                                                            max_width, max_height)
            self.complete_locations = self.complete_locations + complete_locations2
            self.final_pig_positions = self.final_pig_positions + final_pig_positions2

        return len(ground_positions), self.complete_locations, self.final_pig_positions

    def create_platforms(self, number_platforms, complete_locations, final_pig_positions):
        """
        creates a set number of platforms within the level
        automatically reduced if space not found after set number of attempts
        """
        platform_centers = []
        attempts = 0  # number of attempts so far to find space for platform
        self.final_platforms = []
        while len(self.final_platforms) < number_platforms:
            platform_width = randint(4, 7)
            platform_position = [uniform(self.level_width_min + ((platform_width * self.platform_size[0]) / 2.0),
                                         self.level_width_max - ((platform_width * self.platform_size[0]) / 2.0)),
                                 uniform(self.level_height_min, (self.level_height_max - self.minimum_height_gap))]
            temp_platform = []

            if platform_width == 1:
                temp_platform.append(platform_position)

            if platform_width == 2:
                temp_platform.append([platform_position[0] - (self.platform_size[0] * 0.5), platform_position[1]])
                temp_platform.append([platform_position[0] + (self.platform_size[0] * 0.5), platform_position[1]])

            if platform_width == 3:
                temp_platform.append([platform_position[0] - (self.platform_size[0]), platform_position[1]])
                temp_platform.append(platform_position)
                temp_platform.append([platform_position[0] + (self.platform_size[0]), platform_position[1]])

            if platform_width == 4:
                temp_platform.append([platform_position[0] - (self.platform_size[0] * 1.5), platform_position[1]])
                temp_platform.append([platform_position[0] - (self.platform_size[0] * 0.5), platform_position[1]])
                temp_platform.append([platform_position[0] + (self.platform_size[0] * 0.5), platform_position[1]])
                temp_platform.append([platform_position[0] + (self.platform_size[0] * 1.5), platform_position[1]])

            if platform_width == 5:
                temp_platform.append([platform_position[0] - (self.platform_size[0] * 2.0), platform_position[1]])
                temp_platform.append([platform_position[0] - (self.platform_size[0]), platform_position[1]])
                temp_platform.append(platform_position)
                temp_platform.append([platform_position[0] + (self.platform_size[0]), platform_position[1]])
                temp_platform.append([platform_position[0] + (self.platform_size[0] * 2.0), platform_position[1]])

            if platform_width == 6:
                temp_platform.append([platform_position[0] - (self.platform_size[0] * 2.5), platform_position[1]])
                temp_platform.append([platform_position[0] - (self.platform_size[0] * 1.5), platform_position[1]])
                temp_platform.append([platform_position[0] - (self.platform_size[0] * 0.5), platform_position[1]])
                temp_platform.append([platform_position[0] + (self.platform_size[0] * 0.5), platform_position[1]])
                temp_platform.append([platform_position[0] + (self.platform_size[0] * 1.5), platform_position[1]])
                temp_platform.append([platform_position[0] + (self.platform_size[0] * 2.5), platform_position[1]])

            if platform_width == 7:
                temp_platform.append([platform_position[0] - (self.platform_size[0] * 3.0), platform_position[1]])
                temp_platform.append([platform_position[0] - (self.platform_size[0] * 2.0), platform_position[1]])
                temp_platform.append([platform_position[0] - (self.platform_size[0]), platform_position[1]])
                temp_platform.append(platform_position)
                temp_platform.append([platform_position[0] + (self.platform_size[0]), platform_position[1]])
                temp_platform.append([platform_position[0] + (self.platform_size[0] * 2.0), platform_position[1]])
                temp_platform.append([platform_position[0] + (self.platform_size[0] * 3.0), platform_position[1]])

            overlap = False
            for platform in temp_platform:

                if (((platform[0] - (self.platform_size[0] / 2)) < self.level_width_min) or (
                        (platform[0] + (self.platform_size[0]) / 2) > self.level_width_max)):
                    overlap = True

                for block in complete_locations:
                    if (round((platform[0] - self.platform_distance_buffer - self.platform_size[0] / 2), 10) <= round(
                            (block[1] + self.blocks[str(block[0])][0] / 2), 10) and round(
                        (platform[0] + self.platform_distance_buffer + self.platform_size[0] / 2), 10) >= round(
                        (block[1] - self.blocks[str(block[0])][0] / 2), 10) and round(
                        (platform[1] + self.platform_distance_buffer + self.platform_size[1] / 2), 10) >= round(
                        (block[2] - self.blocks[str(block[0])][1] / 2), 10) and round(
                        (platform[1] - self.platform_distance_buffer - self.platform_size[1] / 2), 10) <= round(
                        (block[2] + self.blocks[str(block[0])][1] / 2), 10)):
                        overlap = True

                for pig in final_pig_positions:
                    if (round((platform[0] - self.platform_distance_buffer - self.platform_size[0] / 2), 10) <= round(
                            (pig[0] + self.pig_size[0] / 2), 10) and round(
                        (platform[0] + self.platform_distance_buffer + self.platform_size[0] / 2), 10) >= round(
                        (pig[0] - self.pig_size[0] / 2), 10) and round(
                        (platform[1] + self.platform_distance_buffer + self.platform_size[1] / 2), 10) >= round(
                        (pig[1] - self.pig_size[1] / 2), 10) and round(
                        (platform[1] - self.platform_distance_buffer - self.platform_size[1] / 2), 10) <= round(
                        (pig[1] + self.pig_size[1] / 2), 10)):
                        overlap = True

                for platform_set in self.final_platforms:
                    for platform2 in platform_set:
                        if (round((platform[0] - self.platform_distance_buffer - self.platform_size[0] / 2),
                                  10) <= round((platform2[0] + self.platform_size[0] / 2), 10) and round(
                            (platform[0] + self.platform_distance_buffer + self.platform_size[0] / 2), 10) >= round(
                            (platform2[0] - self.platform_size[0] / 2), 10) and round(
                            (platform[1] + self.platform_distance_buffer + self.platform_size[1] / 2), 10) >= round(
                            (platform2[1] - self.platform_size[1] / 2), 10) and round(
                            (platform[1] - self.platform_distance_buffer - self.platform_size[1] / 2), 10) <= round(
                            (platform2[1] + self.platform_size[1] / 2), 10)):
                            overlap = True

                for platform_set2 in self.final_platforms:
                    for i in platform_set2:
                        if i[0] + self.platform_size[0] > platform[0] and i[0] - self.platform_size[0] < platform[0]:
                            if i[1] + self.minimum_height_gap > platform[1] and i[1] - self.minimum_height_gap < \
                                    platform[1]:
                                overlap = True

            if not overlap:
                self.final_platforms.append(temp_platform)
                platform_centers.append(platform_position)

            attempts = attempts + 1
            if attempts > self.max_attempts:
                attempts = 0
                number_platforms = number_platforms - 1

        logger.debug(f"number platforms: {number_platforms}")

        return number_platforms, self.final_platforms, platform_centers

    def create_platform_structures(self, final_platforms, platform_centers, complete_locations, final_pig_positions):
        """
        create sutiable structures for each platform
        """
        current_platform = 0
        for platform_set in final_platforms:
            platform_set_width = len(platform_set) * self.platform_size[0]

            above_blocks = []
            for platform_set2 in final_platforms:
                if platform_set2 != platform_set:
                    for i in platform_set2:
                        if i[0] + self.platform_size[0] > platform_set[0][0] and i[0] - self.platform_size[0] < \
                                platform_set[-1][0] and i[1] > platform_set[0][1]:
                            above_blocks.append(i)

            min_above = self.level_height_max
            for j in above_blocks:
                if j[1] < min_above:
                    min_above = j[1]

            center_point = platform_centers[current_platform][0]
            absolute_ground = platform_centers[current_platform][1] + (self.platform_size[1] / 2)

            max_width = platform_set_width
            max_height = (min_above - absolute_ground) - self.pig_size[1] - self.platform_size[1]

            complete_locations2, final_pig_positions2 = self.make_structure(absolute_ground, center_point, max_width,
                                                                            max_height)
            complete_locations = complete_locations + complete_locations2
            final_pig_positions = final_pig_positions + final_pig_positions2

            current_platform = current_platform + 1

        return complete_locations, final_pig_positions

    def remove_unnecessary_pigs(self, number_pigs):
        """
        remove random pigs until number equals the desired amount
        :param number_pigs:
        :return:
        """
        removed_pigs = []
        while len(self.final_pig_positions) > number_pigs:
            remove_pos = randint(0, len(self.final_pig_positions) - 1)
            removed_pigs.append(self.final_pig_positions[remove_pos])
            self.final_pig_positions.pop(remove_pos)
        return self.final_pig_positions, removed_pigs

    def add_necessary_pigs(self, number_pigs):
        """
        add pigs on the ground until number equals the desired amount
        """
        while len(self.final_pig_positions) < number_pigs:
            test_position = [uniform(self.level_width_min, self.level_width_max), self.absolute_ground]
            pig_width = self.pig_size[0]
            pig_height = self.pig_size[1]
            valid_pig = True
            for i in self.complete_locations:
                if (round((test_position[0] - pig_width / 2), 10) < round((i[1] + (self.blocks[str(i[0])][0]) / 2),
                                                                          10) and round(
                    (test_position[0] + pig_width / 2), 10) > round((i[1] - (self.blocks[str(i[0])][0]) / 2),
                                                                    10) and round((test_position[1] + pig_height / 2),
                                                                                  10) > round(
                    (i[2] - (self.blocks[str(i[0])][1]) / 2), 10) and round((test_position[1] - pig_height / 2),
                                                                            10) < round(
                    (i[2] + (self.blocks[str(i[0])][1]) / 2), 10)):
                    valid_pig = False
            for i in self.final_pig_positions:
                if (round((test_position[0] - pig_width / 2), 10) < round((i[0] + (pig_width / 2)), 10) and round(
                        (test_position[0] + pig_width / 2), 10) > round((i[0] - (pig_width / 2)), 10) and round(
                    (test_position[1] + pig_height / 2), 10) > round((i[1] - (pig_height / 2)), 10) and round(
                    (test_position[1] - pig_height / 2), 10) < round((i[1] + (pig_height / 2)), 10)):
                    valid_pig = False
            if valid_pig:
                self.final_pig_positions.append(test_position)
        return self.final_pig_positions

    def choose_number_birds(self, final_pig_positions, number_ground_structures, number_platforms):
        """
        choose the number of birds based on the number of pigs and structures present within level
        """
        number_birds = int(ceil(len(final_pig_positions) / 2))
        if (number_ground_structures + number_platforms) >= number_birds:
            number_birds = number_birds + 1
        number_birds = number_birds + 1  # adjust based on desired difficulty
        return number_birds

    def find_trihole_positions(self, complete_locations):
        """
        identify all possible triangleHole positions on top of blocks
        """
        possible_trihole_positions = []
        for block in complete_locations:
            block_width = round(self.blocks[str(block[0])][0], 10)
            block_height = round(self.blocks[str(block[0])][1], 10)
            trihole_width = self.additional_object_sizes['1'][0]
            trihole_height = self.additional_object_sizes['1'][1]

            # don't place block on edge if block too thin
            if self.blocks[str(block[0])][0] < trihole_width:
                test_positions = [
                    [round(block[1], 10), round(block[2] + (trihole_height / 2) + (block_height / 2), 10)]]
            else:
                test_positions = [
                    [round(block[1], 10), round(block[2] + (trihole_height / 2) + (block_height / 2), 10)],
                    [round(block[1] + (block_width / 3), 10),
                     round(block[2] + (trihole_height / 2) + (block_height / 2), 10)],
                    [round(block[1] - (block_width / 3), 10),
                     round(block[2] + (trihole_height / 2) + (block_height / 2), 10)]]

            for test_position in test_positions:
                valid_position = True
                for i in complete_locations:
                    if (round((test_position[0] - trihole_width / 2), 10) < round(
                            (i[1] + (self.blocks[str(i[0])][0]) / 2), 10) and round(
                        (test_position[0] + trihole_width / 2), 10) > round((i[1] - (self.blocks[str(i[0])][0]) / 2),
                                                                            10) and round(
                        (test_position[1] + trihole_height / 2), 10) > round(
                        (i[2] - (self.blocks[str(i[0])][1]) / 2), 10) and round((test_position[1] - trihole_height / 2),
                                                                                10) < round(
                        (i[2] + (self.blocks[str(i[0])][1]) / 2), 10)):
                        valid_position = False
                for j in self.final_pig_positions:
                    if (round((test_position[0] - trihole_width / 2), 10) < round((j[0] + (self.pig_size[0] / 2)),
                                                                                  10) and round(
                        (test_position[0] + trihole_width / 2), 10) > round((j[0] - (self.pig_size[0] / 2)),
                                                                            10) and round(
                        (test_position[1] + trihole_height / 2), 10) > round((j[1] - (self.pig_size[1] / 2)),
                                                                             10) and round(
                        (test_position[1] - trihole_height / 2), 10) < round((j[1] + (self.pig_size[1] / 2)), 10)):
                        valid_position = False
                for j in self.final_TNT_positions:
                    if (round((test_position[0] - trihole_width / 2), 10) < round((j[0] + (self.pig_size[0] / 2)),
                                                                                  10) and round(
                        (test_position[0] + trihole_width / 2), 10) > round((j[0] - (self.pig_size[0] / 2)),
                                                                            10) and round(
                        (test_position[1] + trihole_height / 2), 10) > round((j[1] - (self.pig_size[1] / 2)),
                                                                             10) and round(
                        (test_position[1] - trihole_height / 2), 10) < round((j[1] + (self.pig_size[1] / 2)), 10)):
                        valid_position = False
                for i in self.final_platforms:
                    for j in i:
                        if (round((test_position[0] - trihole_width / 2), 10) < round(
                                (j[0] + (self.platform_size[0] / 2)), 10) and round(
                            (test_position[0] + trihole_width / 2), 10) > round((j[0] - (self.platform_size[0] / 2)),
                                                                                10) and round(
                            (test_position[1] + self.platform_distance_buffer + trihole_height / 2),
                            10) > round((j[1] - (self.platform_size[1] / 2)), 10) and round(
                            (test_position[1] - self.platform_distance_buffer - trihole_height / 2), 10) < round(
                            (j[1] + (self.platform_size[1] / 2)), 10)):
                            valid_position = False
                if valid_position:
                    possible_trihole_positions.append(test_position)

        return possible_trihole_positions

    def find_tri_positions(self, complete_locations):
        """
        identify all possible triangle positions on top of blocks
        """
        possible_tri_positions = []
        for block in complete_locations:
            block_width = round(self.blocks[str(block[0])][0], 10)
            block_height = round(self.blocks[str(block[0])][1], 10)
            tri_width = self.additional_object_sizes['2'][0]
            tri_height = self.additional_object_sizes['2'][1]

            # don't place block on edge if block too thin
            if self.blocks[str(block[0])][0] < tri_width:
                test_positions = [[round(block[1], 10), round(block[2] + (tri_height / 2) + (block_height / 2), 10)]]
            else:
                test_positions = [[round(block[1], 10), round(block[2] + (tri_height / 2) + (block_height / 2), 10)],
                                  [round(block[1] + (block_width / 3), 10),
                                   round(block[2] + (tri_height / 2) + (block_height / 2), 10)],
                                  [round(block[1] - (block_width / 3), 10),
                                   round(block[2] + (tri_height / 2) + (block_height / 2), 10)]]

            for test_position in test_positions:
                valid_position = True
                for i in complete_locations:
                    if (round((test_position[0] - tri_width / 2), 10) < round((i[1] + (self.blocks[str(i[0])][0]) / 2),
                                                                              10) and round(
                        (test_position[0] + tri_width / 2), 10) > round((i[1] - (self.blocks[str(i[0])][0]) / 2),
                                                                        10) and round(
                        (test_position[1] + tri_height / 2), 10) > round((i[2] - (self.blocks[str(i[0])][1]) / 2),
                                                                         10) and round(
                        (test_position[1] - tri_height / 2), 10) < round((i[2] + (self.blocks[str(i[0])][1]) / 2), 10)):
                        valid_position = False
                for j in self.final_pig_positions:
                    if (round((test_position[0] - tri_width / 2), 10) < round((j[0] + (self.pig_size[0] / 2)),
                                                                              10) and round(
                        (test_position[0] + tri_width / 2), 10) > round((j[0] - (self.pig_size[0] / 2)), 10) and round(
                        (test_position[1] + tri_height / 2), 10) > round((j[1] - (self.pig_size[1] / 2)), 10) and round(
                        (test_position[1] - tri_height / 2), 10) < round((j[1] + (self.pig_size[1] / 2)), 10)):
                        valid_position = False
                for j in self.final_TNT_positions:
                    if (round((test_position[0] - tri_width / 2), 10) < round((j[0] + (self.pig_size[0] / 2)),
                                                                              10) and round(
                        (test_position[0] + tri_width / 2), 10) > round((j[0] - (self.pig_size[0] / 2)), 10) and round(
                        (test_position[1] + tri_height / 2), 10) > round((j[1] - (self.pig_size[1] / 2)), 10) and round(
                        (test_position[1] - tri_height / 2), 10) < round((j[1] + (self.pig_size[1] / 2)), 10)):
                        valid_position = False
                for i in self.final_platforms:
                    for j in i:
                        if (round((test_position[0] - tri_width / 2), 10) < round((j[0] + (self.platform_size[0] / 2)),
                                                                                  10) and round(
                            (test_position[0] + tri_width / 2), 10) > round((j[0] - (self.platform_size[0] / 2)),
                                                                            10) and round(
                            (test_position[1] + self.platform_distance_buffer + tri_height / 2), 10) > round(
                            (j[1] - (self.platform_size[1] / 2)), 10) and round(
                            (test_position[1] - self.platform_distance_buffer - tri_height / 2), 10) < round(
                            (j[1] + (self.platform_size[1] / 2)), 10)):
                            valid_position = False

                if self.blocks[str(block[0])][0] < tri_width:  # as block not symmetrical need to check for support
                    valid_position = False
                if valid_position:
                    possible_tri_positions.append(test_position)

        return possible_tri_positions

    def find_cir_positions(self, complete_locations):
        """
        identify all possible circle positions on top of blocks (can only be placed in middle of block)
        """
        possible_cir_positions = []
        for block in complete_locations:
            block_width = round(self.blocks[str(block[0])][0], 10)
            block_height = round(self.blocks[str(block[0])][1], 10)
            cir_width = self.additional_object_sizes['3'][0]
            cir_height = self.additional_object_sizes['3'][1]

            # only checks above block's center
            test_positions = [[round(block[1], 10), round(block[2] + (cir_height / 2) + (block_height / 2), 10)]]

            for test_position in test_positions:
                valid_position = True
                for i in complete_locations:
                    if (round((test_position[0] - cir_width / 2), 10) < round((i[1] + (self.blocks[str(i[0])][0]) / 2),
                                                                              10) and round(
                        (test_position[0] + cir_width / 2), 10) > round((i[1] - (self.blocks[str(i[0])][0]) / 2),
                                                                        10) and round(
                        (test_position[1] + cir_height / 2), 10) > round((i[2] - (self.blocks[str(i[0])][1]) / 2),
                                                                         10) and round(
                        (test_position[1] - cir_height / 2), 10) < round((i[2] + (self.blocks[str(i[0])][1]) / 2), 10)):
                        valid_position = False
                for j in self.final_pig_positions:
                    if (round((test_position[0] - cir_width / 2), 10) < round((j[0] + (self.pig_size[0] / 2)),
                                                                              10) and round(
                        (test_position[0] + cir_width / 2), 10) > round((j[0] - (self.pig_size[0] / 2)), 10) and round(
                        (test_position[1] + cir_height / 2), 10) > round((j[1] - (self.pig_size[1] / 2)), 10) and round(
                        (test_position[1] - cir_height / 2), 10) < round((j[1] + (self.pig_size[1] / 2)), 10)):
                        valid_position = False
                for j in self.final_TNT_positions:
                    if (round((test_position[0] - cir_width / 2), 10) < round((j[0] + (self.pig_size[0] / 2)),
                                                                              10) and round(
                        (test_position[0] + cir_width / 2), 10) > round((j[0] - (self.pig_size[0] / 2)), 10) and round(
                        (test_position[1] + cir_height / 2), 10) > round((j[1] - (self.pig_size[1] / 2)), 10) and round(
                        (test_position[1] - cir_height / 2), 10) < round((j[1] + (self.pig_size[1] / 2)), 10)):
                        valid_position = False
                for i in self.final_platforms:
                    for j in i:
                        if (round((test_position[0] - cir_width / 2), 10) < round((j[0] + (self.platform_size[0] / 2)),
                                                                                  10) and round(
                            (test_position[0] + cir_width / 2), 10) > round((j[0] - (self.platform_size[0] / 2)),
                                                                            10) and round(
                            (test_position[1] + self.platform_distance_buffer + cir_height / 2), 10) > round(
                            (j[1] - (self.platform_size[1] / 2)), 10) and round(
                            (test_position[1] - self.platform_distance_buffer - cir_height / 2), 10) < round(
                            (j[1] + (self.platform_size[1] / 2)), 10)):
                            valid_position = False
                if valid_position:
                    possible_cir_positions.append(test_position)

        return possible_cir_positions

    def find_cirsmall_positions(self, complete_locations):
        """
        identify all possible circleSmall positions on top of blocks
        """
        possible_cirsmall_positions = []
        for block in complete_locations:
            block_width = round(self.blocks[str(block[0])][0], 10)
            block_height = round(self.blocks[str(block[0])][1], 10)
            cirsmall_width = self.additional_object_sizes['4'][0]
            cirsmall_height = self.additional_object_sizes['4'][1]

            # don't place block on edge if block too thin
            if self.blocks[str(block[0])][0] < cirsmall_width:
                test_positions = [
                    [round(block[1], 10), round(block[2] + (cirsmall_height / 2) + (block_height / 2), 10)]]
            else:
                test_positions = [
                    [round(block[1], 10), round(block[2] + (cirsmall_height / 2) + (block_height / 2), 10)],
                    [round(block[1] + (block_width / 3), 10),
                     round(block[2] + (cirsmall_height / 2) + (block_height / 2), 10)],
                    [round(block[1] - (block_width / 3), 10),
                     round(block[2] + (cirsmall_height / 2) + (block_height / 2), 10)]]

            for test_position in test_positions:
                valid_position = True
                for i in complete_locations:
                    if (round((test_position[0] - cirsmall_width / 2), 10) < round(
                            (i[1] + (self.blocks[str(i[0])][0]) / 2), 10) and round(
                        (test_position[0] + cirsmall_width / 2), 10) > round((i[1] - (self.blocks[str(i[0])][0]) / 2),
                                                                             10) and round(
                        (test_position[1] + cirsmall_height / 2), 10) > round(
                        (i[2] - (self.blocks[str(i[0])][1]) / 2), 10) and round(
                        (test_position[1] - cirsmall_height / 2), 10) < round((i[2] + (self.blocks[str(i[0])][1]) / 2),
                                                                              10)):
                        valid_position = False
                for j in self.final_pig_positions:
                    if (round((test_position[0] - cirsmall_width / 2), 10) < round((j[0] + (self.pig_size[0] / 2)),
                                                                                   10) and round(
                        (test_position[0] + cirsmall_width / 2), 10) > round((j[0] - (self.pig_size[0] / 2)),
                                                                             10) and round(
                        (test_position[1] + cirsmall_height / 2), 10) > round((j[1] - (self.pig_size[1] / 2)),
                                                                              10) and round(
                        (test_position[1] - cirsmall_height / 2), 10) < round((j[1] + (self.pig_size[1] / 2)), 10)):
                        valid_position = False
                for j in self.final_TNT_positions:
                    if (round((test_position[0] - cirsmall_width / 2), 10) < round((j[0] + (self.pig_size[0] / 2)),
                                                                                   10) and round(
                        (test_position[0] + cirsmall_width / 2), 10) > round((j[0] - (self.pig_size[0] / 2)),
                                                                             10) and round(
                        (test_position[1] + cirsmall_height / 2), 10) > round((j[1] - (self.pig_size[1] / 2)),
                                                                              10) and round(
                        (test_position[1] - cirsmall_height / 2), 10) < round((j[1] + (self.pig_size[1] / 2)), 10)):
                        valid_position = False
                for i in self.final_platforms:
                    for j in i:
                        if (round((test_position[0] - cirsmall_width / 2), 10) < round(
                                (j[0] + (self.platform_size[0] / 2)), 10) and round(
                            (test_position[0] + cirsmall_width / 2), 10) > round((j[0] - (self.platform_size[0] / 2)),
                                                                                 10) and round(
                            (test_position[1] + self.platform_distance_buffer + cirsmall_height / 2),
                            10) > round((j[1] - (self.platform_size[1] / 2)), 10) and round(
                            (test_position[1] - self.platform_distance_buffer - cirsmall_height / 2), 10) < round(
                            (j[1] + (self.platform_size[1] / 2)), 10)):
                            valid_position = False
                if valid_position:
                    possible_cirsmall_positions.append(test_position)

        return possible_cirsmall_positions

    def find_additional_block_positions(self, complete_locations):
        """
        finds possible positions for valid additional block types
        """
        possible_trihole_positions = []
        possible_tri_positions = []
        possible_cir_positions = []
        possible_cirsmall_positions = []
        if self.trihole_allowed:
            possible_trihole_positions = self.find_trihole_positions(complete_locations)
        if self.tri_allowed:
            possible_tri_positions = self.find_tri_positions(complete_locations)
        if self.cir_allowed:
            possible_cir_positions = self.find_cir_positions(complete_locations)
        if self.cirsmall_allowed:
            possible_cirsmall_positions = self.find_cirsmall_positions(complete_locations)
        return possible_trihole_positions, possible_tri_positions, possible_cir_positions, possible_cirsmall_positions

    def add_additional_blocks(self, possible_trihole_positions, possible_tri_positions, possible_cir_positions,
                              possible_cirsmall_positions):
        """
        combine all possible additonal block positions into one set
        """
        all_other = []
        for i in possible_trihole_positions:
            all_other.append(['1', i[0], i[1]])
        for i in possible_tri_positions:
            all_other.append(['2', i[0], i[1]])
        for i in possible_cir_positions:
            all_other.append(['3', i[0], i[1]])
        for i in possible_cirsmall_positions:
            all_other.append(['4', i[0], i[1]])

        # randomly choose an additional block position and remove those that overlap it
        # repeat untill no more valid position

        selected_other = []
        while (len(all_other) > 0):
            chosen = all_other.pop(randint(0, len(all_other) - 1))
            selected_other.append(chosen)
            new_all_other = []
            for i in all_other:
                if (round((chosen[1] - (self.additional_object_sizes[chosen[0]][0] / 2)), 10) >= round(
                        (i[1] + (self.additional_object_sizes[i[0]][0] / 2)), 10) or round(
                    (chosen[1] + (self.additional_object_sizes[chosen[0]][0] / 2)), 10) <= round(
                    (i[1] - (self.additional_object_sizes[i[0]][0] / 2)), 10) or round(
                    (chosen[2] + (self.additional_object_sizes[chosen[0]][1] / 2)), 10) <= round(
                    (i[2] - (self.additional_object_sizes[i[0]][1] / 2)), 10) or round(
                    (chosen[2] - (self.additional_object_sizes[chosen[0]][1] / 2)), 10) >= round(
                    (i[2] + (self.additional_object_sizes[i[0]][1] / 2)), 10)):
                    new_all_other.append(i)
            all_other = new_all_other

        return selected_other

    def remove_blocks(self, restricted_blocks):
        """
        remove restricted block types from the available selection
        """
        total_prob_removed = 0.0
        new_prob_table = deepcopy(self.probability_table_blocks)
        for block_name in restricted_blocks:
            for key, value in self.block_names.items():
                if value == block_name:
                    total_prob_removed = total_prob_removed + self.probability_table_blocks[key]
                    new_prob_table[key] = 0.0
        new_total = 1.0 - total_prob_removed
        for key, value in new_prob_table.items():
            new_prob_table[key] = value / new_total
        return new_prob_table

    def add_TNT(self, potential_positions):
        """
        add TNT blocks based on removed pig positions
        """
        self.final_TNT_positions = []
        for position in potential_positions:
            if (uniform(0.0, 1.0) < self.TNT_block_probability):
                self.final_TNT_positions.append(position)
        return self.final_TNT_positions

    def write_level_xml(self, complete_locations, selected_other, final_pig_positions, final_TNT_positions,
                        final_platforms, number_birds, current_level, restricted_combinations, folder_path = "./"):
        """
        write level out in desired xml format
        """
        Path(folder_path).mkdir(parents = True, exist_ok = True)
        f = open(f"{folder_path}level-{current_level}.xml", "w")

        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<Level width ="2">\n')
        f.write('<Camera x="0" y="2" minWidth="20" maxWidth="30"/>\n')
        f.write('<Birds>\n')
        for i in range(number_birds):  # bird type is chosen using probability table
            f.write('<Bird type="%s"/>\n' % self.bird_names[str(self.choose_item(self.bird_probabilities))])
        f.write('</Birds>\n')
        f.write('<Slingshot x="-8" y="-2.5"/>\n')
        f.write('<GameObjects>\n')

        for i in complete_locations:
            material = self.materials[randint(0, len(self.materials) - 1)]  # material is chosen randomly
            while [material, self.block_names[
                str(i[0])]] in restricted_combinations:  # if material if not allowed for block type then pick again
                material = self.materials[randint(0, len(self.materials) - 1)]
            rotation = 0
            if (i[0] in (3, 7, 9, 11, 13)):
                rotation = 90
            f.write('<Block type="%s" material="%s" x="%s" y="%s" rotation="%s" />\n' % (
                self.block_names[str(i[0])], material, str(i[1]), str(i[2]), str(rotation)))

        for i in selected_other:
            material = self.materials[randint(0, len(self.materials) - 1)]  # material is chosen randomly
            while [material, self.additional_objects[
                str(i[0])]] in restricted_combinations:  # if material if not allowed for block type then pick again
                material = self.materials[randint(0, len(self.materials) - 1)]
            if i[0] == '2':
                facing = randint(0, 1)
                f.write('<Block type="%s" material="%s" x="%s" y="%s" rotation="%s" />\n' % (
                    self.additional_objects[i[0]], material, str(i[1]), str(i[2]), str(facing * 90.0)))
            else:
                f.write('<Block type="%s" material="%s" x="%s" y="%s" rotation="0" />\n' % (
                    self.additional_objects[i[0]], material, str(i[1]), str(i[2])))

        for i in final_pig_positions:
            f.write('<Pig type="BasicSmall" material="" x="%s" y="%s" rotation="0" />\n' % (str(i[0]), str(i[1])))

        for i in final_platforms:
            for j in i:
                f.write('<Platform type="Platform" material="" x="%s" y="%s" />\n' % (str(j[0]), str(j[1])))

        for i in final_TNT_positions:
            f.write('<TNT type="" material="" x="%s" y="%s" rotation="0" />\n' % (str(i[0]), str(i[1])))

        f.write('</GameObjects>\n')
        f.write('</Level>\n')

        f.close()


if __name__ == '__main__':
    baseline_generator = BaselineGenerator()
    baseline_generator.generate_level_init(folder_path = "../generated_level/BaseLine/")
