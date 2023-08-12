import pickle
from multiprocessing import Manager, Pool
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from converter.to_img_converter.LevelImgEncoder import LevelImgEncoder
from game_management.GameConnection import GameConnection
from game_management.GameManager import GameManager
from level.LevelReader import LevelReader
from level.LevelUtil import calc_structure_meta_data
from util.Config import Config

logger.disable('level.Level')

continue_at_level = 0

config = Config.get_instance()
game_connection = GameConnection(conf = config)
level_reader = LevelReader()

orig_level_folder = config.get_data_train_path(folder = 'generated/single_structure')

use_screen_shot = False
use_ai = False
test_on_live_game = False

if test_on_live_game:
    game_manager = GameManager(conf = config, game_connection = game_connection)
    game_manager.start_game(is_running = False)

data_file = config.get_data_set('multilayer_with_air', 'original.pickle')

level_encoder = LevelImgEncoder()

encoding_algorithm = level_encoder.create_multilayer_with_air


def create_level_data_multi_structure(original_data_level, p_dict, lock):

    parsed_level = level_reader.parse_level(
        str(original_data_level), use_blocks = True, use_pigs = True, use_platform = True)
    parsed_level.filter_slingshot_platform()
    parsed_level.normalize()
    parsed_level.create_polygons()
    level_structures = parsed_level.separate_structures()

    level_counter = 0
    for idx, structure in enumerate(level_structures):
        meta_data = calc_structure_meta_data(structure)

        if meta_data.pig_amount == 0:
            continue

        struct_doc = level_reader.create_level_from_structure(
            structure = structure,
            level = parsed_level,
            move_to_ground = True
        )

        new_level_path = config.get_data_train_path("structures") + "level-04.xml"
        level_reader.write_xml_file(struct_doc, new_level_path)

        if test_on_live_game:
            game_manager.change_level(path = new_level_path, delete_level = True)

        new_level = level_reader.parse_level(new_level_path, use_platform = True)
        new_level.normalize()
        new_level.create_polygons()
        new_level.filter_slingshot_platform()
        new_level.separate_structures()

        if use_screen_shot:
            level_screenshot = game_connection.create_level_img(structure = True)

        if use_ai:
            game_connection.startAi(start_level = 4, end_level = 4, print_ai_log = True)

        ret_pictures = new_level.create_img(per_structure = True, dot_version = True)
        if use_ai:
            all_levels_played = game_connection.wait_till_all_level_played()
            logger.debug(f'All levels Played: {all_levels_played}')
            game_connection.stopAI()
            if not all_levels_played:
                continue

        if test_on_live_game:
            new_level_data = game_connection.get_data()
            logger.debug(new_level_data)

        dict_idx = f'{str(original_data_level).strip(".xml")}_{level_counter}'
        p_dict[dict_idx] = dict(
            meta_data = meta_data,
            img_data = ret_pictures
        )

        if test_on_live_game:
            p_dict[dict_idx]['game_data'] = new_level_data

        if use_screen_shot:
            p_dict[dict_idx]['level_screenshot'] = level_screenshot,

        level_counter += 1

    lock.acquire()
    with open(data_file, 'wb') as handle:
        pickle.dump(dict(p_dict), handle, protocol = pickle.HIGHEST_PROTOCOL)
    lock.release()


def create_level_data_single_structure(original_data_level, p_dict, lock, store_immediately = False):

    parsed_level = level_reader.parse_level(
        str(original_data_level), use_blocks = True, use_pigs = True, use_platform = True)

    # Preprocess the level
    parsed_level.normalize()
    parsed_level.create_polygons()
    parsed_level.filter_slingshot_platform()

    meta_data = calc_structure_meta_data(parsed_level.get_used_elements())
    if meta_data.pig_amount == 0:
        return

    if test_on_live_game:
        game_manager.change_level(path = str(original_data_level), delete_level = True)

    if use_screen_shot:
        level_screenshot = game_connection.create_level_img(structure = True)

    if use_ai:
        game_connection.startAi(start_level = 4, end_level = 4, print_ai_log = True)

    ret_pictures = encoding_algorithm(parsed_level.get_used_elements())

    if use_ai:
        all_levels_played = game_connection.wait_till_all_level_played()
        logger.debug(f'All levels Played: {all_levels_played}')
        game_connection.stopAI()
        if not all_levels_played:
            return

    if test_on_live_game:
        new_level_data = game_connection.get_data()
        logger.debug(new_level_data)

    dict_idx = f'{str(original_data_level).strip(".xml")}_{0}'
    p_dict[dict_idx] = dict(
        meta_data = meta_data,
        img_data = ret_pictures
    )

    if test_on_live_game:
        p_dict[dict_idx]['game_data'] = new_level_data,

    if use_screen_shot:
        p_dict[dict_idx]['level_screenshot'] = level_screenshot,

    if store_immediately and lock is not None:
        lock.acquire()
        with open(data_file, 'wb') as handle:
            pickle.dump(dict(p_dict), handle, protocol = pickle.HIGHEST_PROTOCOL)
        lock.release()


def create_data_simple():
    data_dict = dict()
    load_data_dict(data_dict)

    levels = sorted(Path(orig_level_folder).glob('*.xml'))

    for level_idx, original_data_level in tqdm(enumerate(levels), total = len(levels)):
        if level_idx < continue_at_level:
            continue

        create_level_data_single_structure(original_data_level, data_dict, None)

    with open(data_file, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)


def create_data_multiprocess():
    process_manager = Manager()
    lock = process_manager.Lock()
    p_dict = process_manager.dict()

    pool = Pool(None)
    load_data_dict(p_dict)
    levels = sorted(Path(orig_level_folder).glob('*.xml'))
    results = []
    p_bar = tqdm(enumerate(levels), total = len(levels))

    def update(*a):
        p_bar.update()

    for level_idx, original_data_level in enumerate(levels):
        if level_idx < continue_at_level:
            continue

        # create_level_data_single_structure(original_data_level, p_dict, lock)

        res = pool.apply_async(func = create_level_data_single_structure, args = (original_data_level, p_dict, lock),
                               callback = update)
        results.append(res)
        # create_level_data(original_data_level, p_dict, lock)
    [result.wait() for result in results]
    # game_manager.stop_game()


def load_data_dict(p_dict):
    if continue_at_level > 0:
        with open(data_file, 'rb') as f:
            data_dict = pickle.load(f)
            for key, value in data_dict.items():
                p_dict[key] = value


if __name__ == '__main__':
    create_data_simple()
