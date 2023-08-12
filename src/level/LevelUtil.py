from dataclasses import dataclass
from typing import Optional

from level.Constants import ObjectType
from level.LevelElement import LevelElement
from util.Utils import round_to_cord


@dataclass
class StructureMetaData:
    """ Class that represents a levels / structures meta data """
    min_x: float = None
    max_x: float = None
    min_y: float = None
    max_y: float = None
    height: float = None
    width: float = None
    block_amount: int = None
    platform_amount: int = None
    pig_amount: int = None
    special_block_amount: int = None
    total: int = None
    ice_blocks: int = None
    stone_blocks: int = None
    wood_blocks: int = None
    stable: Optional[bool] = None

    def __eq__(self, other):
        if not isinstance(other, StructureMetaData):
            return False

        if self.block_amount != other.block_amount:
            return False
        if self.pig_amount != other.pig_amount:
            return False
        if self.platform_amount != other.platform_amount:
            return False
        if self.special_block_amount != other.special_block_amount:
            return False
        if self.ice_blocks != other.ice_blocks:
            return False
        if self.stone_blocks != other.stone_blocks:
            return False
        if self.wood_blocks != other.wood_blocks:
            return False
        if self.stable != other.stable:
            return False

        # Check for width through delta
        if abs(self.width - other.width) > 0.1:
            return False
        if abs(self.height - other.height) > 0.1:
            return False

        return True

    def __getitem__(self, item):
        return getattr(self, item)


def normalize_structure(structure, offset = False):

    min_x, min_y, max_x, max_y = calc_structure_dimensions(structure)

    for element in structure:
        element.x -= min_x
        element.y -= min_y
        element.coordinates[0] -= min_x
        element.coordinates[1] -= min_y

    if offset:
        for element in structure:
            element.x += abs(min_x)
            element.y += abs(min_y)
            element.coordinates[0] += abs(min_x)
            element.coordinates[1] += abs(min_y)

def calc_structure_meta_data(element_list: [LevelElement]) -> StructureMetaData:
    min_x, min_y, max_x, max_y = calc_structure_dimensions(element_list)
    block_amount = len([x for x in element_list if x.object_type == ObjectType.Block])
    pig_amount = len([x for x in element_list if x.object_type == ObjectType.Pig])
    platform_amount = len([x for x in element_list if x.object_type == ObjectType.Platform])
    special_block_amount = len([x for x in element_list if x.object_type == ObjectType.SpecialBlock])
    stone_blocks = len([x for x in element_list if x.object_type == ObjectType.Block and x.material == 'stone'])
    ice_blocks = len([x for x in element_list if x.object_type == ObjectType.Block and x.material == 'ice'])
    wood_blocks = len([x for x in element_list if x.object_type == ObjectType.Block and x.material == 'wood'])
    return StructureMetaData(
        min_x = min_x,
        max_x = max_x,
        min_y = min_y,
        max_y = max_y,
        height = max_y - min_y,
        width = max_x - min_x,
        block_amount = block_amount,
        pig_amount = pig_amount,
        platform_amount = platform_amount,
        special_block_amount = special_block_amount,
        total = block_amount + pig_amount + platform_amount + special_block_amount,
        stone_blocks = stone_blocks,
        wood_blocks = wood_blocks,
        ice_blocks = ice_blocks,
        stable = None,
    )


def calc_structure_dimensions(element_list: [LevelElement], use_original = False, round = False):
    """
    Calculates with the given elements the metadata wanted
    :param element_list: The list of level elements which are included in the calculation
    """
    min_x, min_y, max_x, max_y = 10000, 10000, -10000, -10000
    for element in element_list:
        min_x = min(min_x, (element.x if not use_original else element.original_x) - element.width / 2)
        min_y = min(min_y, (element.y if not use_original else element.original_y) - element.height / 2)
        max_x = max(max_x, (element.x if not use_original else element.original_x) + element.width / 2)
        max_y = max(max_y, (element.y if not use_original else element.original_y) + element.height / 2)

    if round:
        return round_to_cord(min_x), round_to_cord(min_y), round_to_cord(max_x), round_to_cord(max_y)
    else:
        return min_x, min_y, max_x, max_y
