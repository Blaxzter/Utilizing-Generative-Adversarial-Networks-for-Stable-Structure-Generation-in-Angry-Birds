import glob
import os
from pathlib import Path

from loguru import logger

from data_scripts.CreateDataScript import config
from level.LevelElement import LevelElement
from level.LevelReader import LevelReader


def read_all_files(path = './converted_levels/**/*.xml'):
    files = glob.glob(path, recursive = True)

    return sorted(files)


def utf16_to_utf8():
    files = read_all_files()

    for file_name in files:
        with open(file_name, encoding = 'utf-8', mode = "r+") as file:
            content = file.read()
            rep_content = content.replace("utf-16", "utf-8")
            file.seek(0)
            file.write(rep_content)
            file.truncate()


def fix_camera():
    level_path = config.get_instance().get_data_train_path('generated/single_structure')
    level_names = list(Path(level_path).glob("*"))

    for file_name in level_names:
        with open(file_name, encoding = 'utf-8', mode = "r+") as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                if '<Camera x="0" y="2" minWidth="20" maxWidth="30"/>' in line:
                    lines[idx] = '<Camera x="0" y="0" minWidth="20" maxWidth="30"/>'
            file.seek(0)
            file.writelines(lines)

def fix_faulty_xml():
    files = read_all_files()

    for file_name in files:
        with open(file_name, encoding = 'utf-8', mode = "r+") as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                if 'Camera' in line or 'Slingshot' in line:
                    lines[idx] = line.replace('">', '"/>')
            file.seek(0)
            file.writelines(lines)

def filter_levels():
    for level in read_all_files("./converted_levels/NoRotation/*.xml"):
        os.remove(level)

    files = read_all_files()

    level_counter = 0

    for idx, file_name in enumerate(files):
        level_reader = LevelReader()
        level = level_reader.parse_level(file_name)
        if not level.contains_od_rotation() and len(level.blocks) != 0:
            node = level.original_doc.getElementsByTagName("Level")
            name_element = level.original_doc.createElement("PrevLevelName")
            name_element.setAttribute("name", file_name)
            node[0].appendChild(name_element)

            new_level_idx = '0' + str(level_counter + 5) if len(str(level_counter + 5)) == 1 else level_counter + 5
            new_level_name = f"./converted_levels/NoRotation/level-{new_level_idx}.xml"
            level_reader.write_level_to_file(level, new_level_name = new_level_name)
            level_counter += 1

        logger.debug(f"Level contains od rotation: {file_name} \n")


def filter_level(elements, parameter, level):
    for element in elements:
        for level_filter in parameter['filter']:
            if 'type' in level_filter and level_filter['type'] in element.type:
                logger.debug(f'Found due to type {level}')
                return


def filter_level_for(*args):
    parameter = args[0]
    files = read_all_files(parameter['path'])

    for idx, file_name in enumerate(files):
        level_reader = LevelReader()
        level = level_reader.parse_level(file_name)

        elements: [LevelElement] = level.create_element_list(blocks = True, pigs = True, platform = True)
        filter_level(elements, parameter, level)


def remove_pngs():
    for level in read_all_files("./converted_levels/NoRotation/*.png"):
        os.remove(level)


if __name__ == '__main__':
    # filter_levels()
    fix_camera()
    # filter_level_for(
    #    dict(
    #        path = './converted_levels/NoRotation/*.xml',
    #        filter = [
    #             dict(
    #                 type = "Triangle"
    #             )
    #         ]
    #     )
    # )
