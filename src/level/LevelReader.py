import os
from xml.dom.minidom import parse, Document, Element

from level import Constants
from level.Constants import ObjectType, min_distance_to_slingshot
from level.Level import Level
from level.LevelCreator import create_basis_level_node
from level.LevelElement import LevelElement
from level.LevelUtil import calc_structure_dimensions


class LevelReader:
    def __init__(self):
        pass

    def parse_level(self, path, use_blocks = True, use_pigs = True, use_platform = False) -> Level:
        level_doc: Document = parse(path)
        ret_level = Level(path, level_doc, use_blocks, use_pigs, use_platform)
        counter = 0
        for level_part in ["Block", "Pig", "Platform", "Bird"]:
            elements: [Element] = level_doc.getElementsByTagName(level_part)
            for element in elements:
                element_attributes = dict()
                for attribute in Constants.attributes:
                    if attribute in element.attributes:
                        element_attributes[attribute] = element.attributes[attribute].value

                ret_level[level_part].append(
                    LevelElement(id = counter, **element_attributes)
                )
                counter += 1
        slingshot = level_doc.getElementsByTagName("Slingshot")[0]
        ret_level.slingshot = LevelElement(
            id = counter, type = "Slingshot", material = None,
            x = slingshot.attributes['x'].value, y = slingshot.attributes['y'].value
        )

        return ret_level

    def write_level_to_file(self, ret_level: Level, new_level_name = None):
        level_name = ret_level.path if new_level_name is None else new_level_name
        self.write_xml_file(ret_level.original_doc, level_name)

    def write_xml_file(self, xml_file, name):

        # check if folder of file exists
        if not os.path.exists(os.path.dirname(name)):
            # recursively create folder
            os.makedirs(os.path.dirname(name), exist_ok = True, mode = 0o777)

        writer = open(name, 'w')
        xml_file.writexml(writer, indent = " ", addindent = " ", newl = '\n')

    def write_xml_file_from_string(self, xml_string, name):
        with open(name, 'w') as f:
            f.write(xml_string)

    def create_level_from_structure(self, structure: [LevelElement], level: Level = None, move_to_ground: bool = True, move_closer: bool = True, red_birds = True):
        doc, level_node = create_basis_level_node(level, red_birds = red_birds)

        if level is None:
            level = Level()
            level.blocks = structure

        data = None
        if move_to_ground:
            data = calc_structure_dimensions(structure, use_original = True)

        level.separate_structures()

        def _sort_elements(_element):
            if _element.object_type == ObjectType.Platform:
                return 14
            elif _element.object_type == ObjectType.Pig:
                return 13
            return list(Constants.block_names.values()).index(str(_element.type))

        game_objects = doc.createElement('GameObjects')
        for structure in level.structures:
            struct_data = calc_structure_dimensions(structure, use_original = True)
            for level_element in sorted(structure, key = _sort_elements):
                block_name = 'Block'
                if level_element.object_type == ObjectType.Platform:
                    block_name = 'Platform'
                elif level_element.object_type == ObjectType.Pig:
                    block_name = 'Pig'

                current_element_doc = doc.createElement(block_name)
                current_element_doc.setAttribute("type", str(level_element.type))
                current_element_doc.setAttribute("material", str(level_element.material))

                if move_closer and level is not None:
                    current_element_doc.setAttribute(
                        "x",
                        str(level_element.x - abs(level.slingshot.x + min_distance_to_slingshot - struct_data[0]))
                    )
                else:
                    current_element_doc.setAttribute("x", str(level_element.x))

                if move_to_ground:
                    current_element_doc.setAttribute("y", str(level_element.y + Constants.absolute_ground - struct_data[1]))
                else:
                    current_element_doc.setAttribute("y", str(level_element.y))

                current_element_doc.setAttribute("rotation", str(level_element.rotation))

                if level_element.object_type == ObjectType.Platform:
                    if level_element.size[0] != 0.62:
                        current_element_doc.setAttribute("scaleX", str(level_element.size[0] * (1 / 0.62)))
                    if level_element.size[1] != 0.62:
                        current_element_doc.setAttribute("scaleY", str(level_element.size[1] * (1 / 0.62)))

                game_objects.appendChild(current_element_doc)

        level_node.appendChild(game_objects)
        doc.appendChild(level_node)
        return doc
