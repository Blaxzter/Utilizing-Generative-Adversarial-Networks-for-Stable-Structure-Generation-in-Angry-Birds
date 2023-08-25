import numpy as np
from shapely.geometry import Point

from level import Constants
from level.Constants import ObjectType
from level.LevelUtil import calc_structure_dimensions
from util.Config import Config


class LevelImgEncoder:

    def __init__(self, config = None):
        self.config = config

        if self.config is None:
            self.config = Config.get_instance()

    def create_img_of_structures(self, element_lists, dot_version):
        ret_images = []
        if not dot_version:
            for element_list in element_lists:
                ret_images.append(
                    self.create_calculated_img(element_list)
                )

        else:
            for element_list in element_lists:
                ret_images.append(
                    self.create_dot_img(element_list)
                )
        return ret_images

    def create_dot_img(self, element_list):
        working_list = element_list.copy()

        min_x, min_y, max_x, max_y = calc_structure_dimensions(working_list)
        resolution = Constants.resolution

        x_cords = np.arange(min_x + resolution / 2, max_x - resolution / 2, resolution)
        y_cords = np.arange(min_y + resolution / 2, max_y - resolution / 2, resolution)

        picture = np.zeros((len(y_cords), len(x_cords)))

        coordinate_lists = np.array([[element.x, element.y] for element in working_list])

        for i, y_cord in enumerate(y_cords):
            for j, x_cord in enumerate(x_cords):
                in_location = []

                norm = np.linalg.norm(coordinate_lists - np.array([x_cord, y_cord]), axis = 1)
                sorted = np.argsort(norm)

                for element_idx in sorted:
                    element = working_list[element_idx.item()]
                    if element.shape_polygon.intersects(Point(x_cord, y_cord)):
                        in_location.append(element)
                        break

                if len(in_location) == 0:
                    continue

                elif len(in_location) >= 1:
                    picture[len(y_cords) - i - 1, j] = in_location[0].get_identifier()

        return picture

    def create_calculated_img(self, element_list):
        min_x, min_y, max_x, max_y = calc_structure_dimensions(element_list)

        resolution = Constants.resolution

        cord_list = []
        # logger.debug(f"New Structure {(round((max_x - min_x) / resolution), round((max_y - min_y) / resolution))}")
        for element in element_list:

            left_block_pos = element.x - element.width / 2 - min_x
            right_block_pos = element.x + element.width / 2 - min_x
            bottom_block_pos = element.y - element.height / 2 - min_y
            top_block_pos = element.y + element.height / 2 - min_y

            x_cord_range = np.linspace(left_block_pos + resolution / 2, right_block_pos - resolution / 2, num=100) + 0.00001
            y_cord_range = np.linspace(bottom_block_pos + resolution / 2, top_block_pos - resolution / 2, num=100) + 0.00001

            x_cords = np.unique(np.round(x_cord_range / resolution)).astype(np.int)
            y_cords = np.unique(np.round(y_cord_range / resolution)).astype(np.int)

            if len(x_cords) != element.int_width:
                right_stop = right_block_pos - resolution

                if len(x_cords) < element.int_width:
                    right_stop = right_block_pos + resolution / 2
                    pass

                x_cord_range = np.linspace(left_block_pos + resolution / 2, right_stop, num=100)
                x_cords = np.unique(np.round(x_cord_range / resolution)).astype(np.int)

            if len(y_cords) != element.int_height:
                top_stop = top_block_pos - resolution

                if len(x_cords) < element.int_width:
                    top_stop = top_block_pos + resolution / 2
                    pass

                y_cord_range = np.linspace(bottom_block_pos + resolution / 2, top_stop, num=100)
                y_cords = np.unique(np.round(y_cord_range / resolution)).astype(np.int)

            cord_list.append(self.extract_element_data(element, x_cords, y_cords))

        picture = self.convert_into_img(cord_list)

        return self.remove_empty_line(picture)

    def create_one_element_img(self, element_list, air_layer = False, multilayer = False, true_one_hot = False):
        min_x, min_y, max_x, max_y = calc_structure_dimensions(element_list)
        resolution = Constants.resolution

        pig_index = 40

        img_width = round(max_y / resolution)
        img_height = round(max_x / resolution)
        if multilayer:
            # last_layer = 14 if true_one_hot else 4 # Only wood run
            last_layer = 40 if true_one_hot else 4
            if air_layer:
                last_layer += 1
            picture = np.zeros((img_width, img_height, last_layer))
        else:
            picture = np.zeros((img_width, img_height))

        # Set the first layer is zero
        if air_layer and multilayer:
            picture[:, :, 0] = 1

        # logger.debug(f"New Structure {(round((max_x - min_x) / resolution), round((max_y - min_y) / resolution))}")
        for element in element_list:

            element_idx = 14

            if element.object_type == ObjectType.Block or element.object_type == ObjectType.SpecialBlock:
                material_idx = Constants.materials.index(element.material)
                type_idx = list(Constants.block_names.values()).index(element.type) + 1
                if element.is_vertical:
                    type_idx += 1
            else:
                material_idx = 3
                type_idx = element_idx

            x_pos = round(element.x / resolution)
            y_pos = round(element.y / resolution)

            if multilayer:
                if air_layer:
                    picture[y_pos, x_pos, 0] = 0

                if true_one_hot:
                    store_idx = type_idx + 13 * material_idx if element.object_type == ObjectType.Block else pig_index
                    picture[y_pos, x_pos, store_idx - (0 if air_layer else 1)] = 1
                else:
                    picture[y_pos, x_pos, material_idx + (1 if air_layer else 0)] = type_idx
            else:
                store_idx = type_idx + 13 * material_idx if element.object_type == ObjectType.Block else pig_index

                picture[y_pos, x_pos] = store_idx

        return np.flip(picture, axis = 0).astype(np.uint8)

    def remove_empty_line(self, picture):
        ret_img = picture[0, :]
        for y_value in range(1, picture.shape[0]):
            if np.max(picture[y_value, :]) != 0:
                ret_img = np.row_stack([ret_img, picture[y_value, :]])

        return ret_img

    def extract_element_data(self, element, x_cords, y_cords):
        return dict(
            x_cords = x_cords,
            y_cords = y_cords,
            is_pig = element.object_type == ObjectType.Pig,
            min_x = np.min(x_cords),
            min_y = np.min(y_cords),
            max_x = np.max(x_cords),
            max_y = np.max(y_cords),
            width = np.max(x_cords) - np.min(x_cords),
            height = np.max(y_cords) - np.min(y_cords),
            material = element.get_identifier()
        )

    def convert_into_img(self, cord_list):
        min_x = np.min(list(map(lambda x: x['min_x'], cord_list)))
        min_y = np.min(list(map(lambda x: x['min_y'], cord_list)))

        img_width = np.max(list(map(lambda x: x['max_x'] - min_x, cord_list)))
        img_height = np.max(list(map(lambda x: x['max_y'] - min_y, cord_list)))
        picture = np.zeros((img_height + 1, img_width + 1))

        for cords in cord_list:
            cords['y_cords'] -= min_y
            cords['x_cords'] -= min_x
            x_pos, y_pos = np.meshgrid(img_height - cords['y_cords'], cords['x_cords'])
            if cords['is_pig']:
                x_center = np.average(x_pos)
                y_center = np.average(y_pos)
                r = np.sqrt((x_pos - x_center) ** 2 + (y_pos - y_center) ** 2)
                inside = r < 3.17
                picture[x_pos[inside], y_pos[inside]] = cords['material']
            else:
                picture[x_pos, y_pos] = cords['material']

        return picture

    @staticmethod
    def create_multi_dim_img_from_picture(level_img, with_air_layer = False):
        level_img_shape = level_img.shape

        temp_img = np.copy(level_img)
        if len(level_img_shape) == 3:
            if level_img_shape[-1] != 1:
                raise Exception("Only one dimension Images Pls")
            else:
                temp_img = temp_img.reshape(level_img_shape[:-1])

        ret_img = np.zeros((level_img_shape[0], level_img_shape[1], 4 + (1 if with_air_layer else 0)))

        for layer_idx, material_id in enumerate(range((0 if with_air_layer else 1), 5)):
            ret_img[:, :, layer_idx][temp_img == material_id] = 1

        return ret_img.astype(dtype = np.int16)

    def create_calculated_img_no_size_check(self, element_list):
        min_x, min_y, max_x, max_y = calc_structure_dimensions(element_list)
        resolution = Constants.resolution

        cord_list = []

        print(f'\n')

        # logger.debug(f"New Structure {(round((max_x - min_x) / resolution), round((max_y - min_y) / resolution))}")
        for element in element_list:
            left_block_pos = element.x - element.width / 2 - min_x
            right_block_pos = element.x + element.width / 2 - min_x
            bottom_block_pos = element.y - element.height / 2 - min_y
            top_block_pos = element.y + element.height / 2 - min_y

            x_cord_range = np.linspace(left_block_pos + resolution / 2, right_block_pos - resolution / 2) + 0.001
            y_cord_range = np.linspace(bottom_block_pos + resolution / 2, top_block_pos - resolution / 2) + 0.001

            x_cords = np.unique(np.round(x_cord_range / resolution)).astype(np.int)
            y_cords = np.unique(np.round(y_cord_range / resolution)).astype(np.int)

            # print(f'ID: {element.id} -> ({len(x_cords)}, {len(y_cords)})')
            cord_list.append(self.extract_element_data(element, x_cords, y_cords))

        picture = self.convert_into_img(cord_list)

        return self.remove_empty_line(picture)

    # Wrapper function

    def create_one_element_img_multilayer(self, element_list):
        return self.create_one_element_img(element_list, multilayer = True)

    def create_one_element_true_one_hot(self, element_list):
        return self.create_one_element_img(element_list, multilayer = True, true_one_hot = True)

    def create_one_element_true_one_hot_with_air(self, element_list):
        return self.create_one_element_img(element_list, multilayer = True, true_one_hot = True, air_layer = True)

    def create_multilayer_with_air(self, element_list):
        level_img = self.create_calculated_img(element_list)
        return self.create_multi_dim_img_from_picture(level_img, with_air_layer = True)

    def create_multilayer_without_air(self, element_list):
        level_img = self.create_calculated_img(element_list)
        return self.create_multi_dim_img_from_picture(level_img, with_air_layer = False)

    def create_one_layer_img(self, element_list):
        return self.create_calculated_img(element_list)
