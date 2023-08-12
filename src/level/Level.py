import numpy as np
from loguru import logger

from converter.to_img_converter.LevelImgEncoder import LevelImgEncoder
from level import Constants
from level.LevelElement import LevelElement
from level.LevelUtil import calc_structure_meta_data, calc_structure_dimensions
from util import RunConfig


class Level:

    def __init__(self, path: str = None, original_doc = None, blocks = True, pigs = True, platform = False):

        self.path = path
        self.original_doc = original_doc

        self.level_encoder = LevelImgEncoder()
        self.slingshot = LevelElement(id = -1, type = "Slingshot", material = None, x = 0, y = 1)

        self.blocks: [LevelElement] = []
        self.pigs: [LevelElement] = []
        self.platform: [LevelElement] = []
        self.birds: [LevelElement] = []

        self.use_blocks = blocks
        self.use_pigs = pigs
        self.use_platform = platform

        self.is_normalized = False

        self.structures = None

    def __getitem__(self, item):
        if item == "Block":
            return self.blocks
        elif item == "Pig":
            return self.pigs
        elif item == "Platform":
            return self.platform
        elif item == "Bird":
            return self.birds
        elif item == "Slingshot":
            return self.slingshot

    def __str__(self):
        return f"Path: {self.path}, Blocks: {len(self.blocks)} Pigs: {len(self.pigs)} Platform: {len(self.platform)} Bird: {len(self.birds)}"

    def separate_structures(self):
        test_list = self.create_element_list(self.use_blocks, self.use_pigs, self.use_platform, sort_list = True)

        # check if the first element has ploygons
        if test_list[0].shape_polygon is None:
            for element in test_list:
                element.create_set_geometry()

        # A structure is a list of
        self.structures = []
        for element in test_list:

            current_element_id = element.id

            if len(self.structures) == 0:
                self.structures.append([element])
                continue

            closest_structures = []

            # Calculate distance between groups
            for structure in self.structures:

                for struct_element in structure:
                    dist_to_element = element.distance(struct_element)
                    if RunConfig.verbose:
                        logger.debug(f"Block {current_element_id} -> {struct_element.id}: {float(dist_to_element)}")
                    if dist_to_element < 0.2:
                        closest_structures.append(structure)
                        break

            # Go over the structures closest to the element
            if len(closest_structures) == 0:
                # If there is no structure close means that it could be a new structure
                if RunConfig.verbose:
                    logger.debug("Create new Structure")
                self.structures.append([element])
            elif len(closest_structures) == 1:
                # Just one structure means it belongs to it
                if RunConfig.verbose:
                    logger.debug("Add to closest structure")
                closest_structures[0].append(element)
            else:
                # More than one structure means it adds all structures together
                if RunConfig.verbose:
                    logger.debug("Merge all closest structures")
                merge_into = closest_structures[0]
                for closest_structure in closest_structures[1:]:
                    for merge_element in closest_structure:
                        merge_into.append(merge_element)
                    self.structures.remove(closest_structure)
                merge_into.append(element)

        if RunConfig.verbose:
            for structure in self.structures:
                logger.debug(f"Structure amount: {len(structure)}")

        return self.structures

    def create_img(self, per_structure = True, dot_version = False):
        logger.debug("Create level img")
        if not per_structure:
            element_lists: [[LevelElement]] = [
                self.create_element_list(self.use_blocks, self.use_pigs, self.use_platform)]
        else:
            # Check if the level has been structurised
            if self.structures is None:
                self.separate_structures()

            element_lists: [[LevelElement]] = self.structures

        return self.level_encoder.create_img_of_structures(element_lists, dot_version)

    def normalize(self):

        test_list: [LevelElement] = self.create_element_list(self.use_blocks, self.use_pigs, self.use_platform)

        min_x, min_y, max_x, max_y = calc_structure_dimensions(test_list)

        for element in test_list:
            element.x -= min_x
            element.y -= min_y
            element.coordinates[0] -= min_x
            element.coordinates[1] -= min_y

        self.is_normalized = True

    def print_elements(self, as_table = False, group_by_material = True):
        logger.debug(f"Print level: {self.path}")

        if not as_table:
            for element in self.blocks + self.pigs + self.platform:
                logger.debug(element)
        else:
            from tabulate import tabulate

            level_data = [[element.type, element.material, element.x, element.y, element.rotation, element.size] for
                         element in self.blocks + self.pigs + self.platform]
            data = sorted(
                level_data,
                key = lambda entry: (Constants.materials.index(entry[1])
                if group_by_material and entry[1] != '' else 4, entry[2])
            )

            print(tabulate(data, headers = ["type", "material", "x", "y", "rotation", "sizes"]))

    def contains_od_rotation(self):

        test_list = self.create_element_list(self.use_blocks, self.use_pigs, self.use_platform)

        for element in test_list:
            orientation = element.rotation / 90
            next_integer = round(orientation)
            dist_to_next_int = abs(next_integer - orientation)
            if dist_to_next_int > 0.1:
                logger.debug(str(element))
                return True

        return False

    def create_polygons(self):
        test_list = self.create_element_list(self.use_blocks, self.use_pigs, self.use_platform)
        for element in test_list:
            element.shape_polygon = element.create_geometry()

    def get_used_elements(self):
        return self.create_element_list(self.use_blocks, self.use_pigs, self.use_platform)

    def create_element_list(self, blocks, pigs, platform, sort_list = False):
        test_list = []
        if blocks:   test_list += self.blocks
        if pigs:     test_list += self.pigs
        if platform: test_list += self.platform

        if sort_list:
            norm_list = np.linalg.norm(list(map(lambda x: x.coordinates, test_list)), axis = 1)
            index_list = np.argsort(norm_list)

            return list(np.array(test_list)[index_list])

        return test_list

    def filter_slingshot_platform(self):
        if len(self.platform) == 0:
            # logger.debug("No Platform")
            return

        platform_coords = np.asarray(list(map(lambda p: p.coordinates, self.platform)))
        dist_to_slingshot = platform_coords - self.slingshot.coordinates
        norms = np.linalg.norm(dist_to_slingshot, axis = 1)

        remove_platforms = []

        for idx, norm in enumerate(norms):
            if norm < 4:
                remove_platforms.append(self.platform[idx])

        for remove_platform in remove_platforms:
            self.platform.remove(remove_platform)

    def get_level_metadata(self):
        current_elements = self.get_used_elements()
        return calc_structure_meta_data(current_elements)

    @staticmethod
    def create_level_from_structure(level_elements: [LevelElement]):
        ret_level = Level(path = 'created')
        for level_element in level_elements:
            ret_level[level_element.object_type.name].append(
                level_element
            )

        return ret_level
