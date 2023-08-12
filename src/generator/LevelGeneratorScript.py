from generator.baseline.Baseline import BaselineGenerator

from loguru import logger

from util.Config import Config

logger.disable('generator.baseline.Baseline')

if __name__ == '__main__':
    config = Config.get_instance()

    level_dest = config.get_data_train_path(folder = 'generated/single_structure_small/')
    generator = BaselineGenerator()
    generator.settings(
        number_levels = 5000,
        ground_structure_range = (1, 1),
        air_structure_range=(0, 0),
        level_width_min = -2.0,
        level_width_max = 2.0,
        level_height_max = 2.0,
        min_ground_width = 2,
        materials = ["wood", "wood", "wood"]
    )
    generator.generate_level_init(folder_path = level_dest)
