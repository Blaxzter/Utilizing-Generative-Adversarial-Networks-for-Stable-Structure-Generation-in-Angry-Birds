import json
import os
import pickle
import platform
import sys
import time
from enum import Enum
from pathlib import Path

from exceptions.Exceptions import ParameterException
from generator.baseline.Baseline import BaselineGenerator
from util.ProgramArguments import get_program_arguments

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

class GeneratorOptions(Enum):
    baseline = 1
    gan = 2


class Config:
    instance = None

    def __init__(self, args):

        self.strftime = time.strftime("%Y%m%d-%H%M%S")

        # check if args has generator is valid
        if 'generator' in args and args.generator is not None:
            self.generator = args.generator
            if self.generator not in GeneratorOptions:
                raise ParameterException(f"The selected generator is not an option: {GeneratorOptions}")
        else:
            self.generator = GeneratorOptions.baseline

        self.plotting_enabled = True
        self.plot_to_file = False
        self.plt_img_counter = 0

        # Tensorflow data creation setting
        self.max_height = 128
        self.max_width = 128

        self.convert_to_multi_dim = False

        self.current_path = Path(".")
        found_src = False
        while not found_src:
            for file in self.current_path.glob("*"):
                if file.name == "src":
                    found_src = True
                    break
            if not found_src:
                self.current_path = Path(f'{self.current_path}/..')
        self.current_path = str(self.current_path)[:-2]

        self.level_amount: int = args.level_amount if 'level_amount' in args and args.level_amount else 1
        self.data_train_path = os.path.normpath(os.path.join(self.current_path, 'resources/data/source_files/'))
        self.generated_level_path: str = args.generated_level_path + os.sep if 'generated_level_path' in args and args.generated_level_path else \
            os.path.normpath(os.path.join(self.current_path, 'resources/data/generated_level/'))
        self.game_folder_path: str = args.game_folder_path if 'game_folder_path' in args and args.game_folder_path else \
            os.path.normpath(os.path.join(self.current_path, 'resources/science_birds/{os}'))
        if '{os}' in self.game_folder_path:
            os_name = platform.system()
            if os_name == 'Windows':
                self.game_folder_path = self.game_folder_path.replace('{os}', 'win-new')
                self.game_path = os.path.join(self.game_folder_path, "ScienceBirds.exe")
                self.copy_dest = os.path.normpath('ScienceBirds_Data/StreamingAssets/Levels/')
            elif os_name == 'Darwin':
                self.game_folder_path = self.game_folder_path.replace('{os}', 'osx-new')
                self.game_path = os.path.join(self.game_folder_path, "ScienceBirds.app")
                self.copy_dest = os.path.normpath('Sciencebirds.app/Contents/Resources/Data/StreamingAssets/Levels')

        self.is_meta_data_comp = True

        self.ai_path = args.ai_path if 'ai_path' in args else os.path.normpath(
            os.path.join(self.current_path, 'ai/Naive-Agent-standalone-Streamlined.jar')
        )
        self.rescue_level_path = os.path.normpath(
            os.path.join(self.current_path, 'resources/data/source_files/level_archive/{timestamp}/')
        )

        self.evaluate = args.evaluate if 'evaluate' in args else False
        self.rescue_level = args.rescue_level if 'rescue_level' in args else True

        # Ml stuff
        self.create_tensorflow_writer = True
        self.one_encoding = False

        self.inner_tqdm = False
        self.outer_tqdm = True

        self.source_file_root = os.path.normpath(
            os.path.join(self.current_path, 'resources/data/source_files/')
        )

        self.data_root = os.path.normpath(
            os.path.join(self.current_path, 'resources/data/')
        )
        self.run_data_root = os.path.normpath(
            os.path.join(self.current_path, 'resources/run_data/')
        )
        self.pickle_folder = os.path.normpath(
            os.path.join(self.current_path, 'resources/data/pickles')
        )
        self.dataset_folder = os.path.normpath(
            os.path.join(self.current_path, 'resources/data/pickles/dataset/')
        )
        self.epoch_run_data = os.path.normpath(
            os.path.join(self.current_path, 'resources/data/pickles/run_data')
        )
        self.eval_root = os.path.normpath(
            os.path.join(self.current_path, 'resources/data/eval/')
        )
        self.metrics_root = os.path.normpath(
            os.path.join(self.current_path, 'resources/data/eval/fids/')
        )
        self.grid_search_root = os.path.normpath(
            os.path.join(self.current_path, 'resources/data/eval/grid_search/')
        )
        self.good_generated_levels = os.path.normpath(
            os.path.join(self.current_path, 'resources/data/eval/good_generated_levels/')
        )
        self.quality_root = os.path.normpath(
            os.path.join(self.current_path, 'resources/data/eval/quality_pictures/')
        )
        self.encoding_folder = os.path.normpath(
            os.path.join(self.current_path, 'resources/data/encoding_data')
        )
        self.tf_records_name = os.path.normpath(
            os.path.join(self.current_path, 'resources/data/tfrecords/{dataset_name}.tfrecords')
        )
        self.log_file_root = os.path.normpath(
            os.path.join(self.current_path, 'resources/logs/')
        )
        self.train_log_dir = os.path.normpath(
            os.path.join(self.current_path, 'resources/logs/{current_run}/{timestamp}/train')
        )
        self.image_root = os.path.normpath(
            os.path.join(self.current_path, 'resources/imgs/')
        )
        self.conv_debug = os.path.normpath(
            os.path.join(self.current_path, 'resources/imgs/conv_debug/')
        )
        self.image_store = os.path.normpath(
            os.path.join(self.current_path, 'resources/imgs/generated/{timestamp}/')
        )
        self.checkpoint_dir = os.path.normpath(
            os.path.join(self.current_path, 'resources/models/{current_run}/{timestamp}/')
        )
        self.new_model_path = os.path.normpath(
            os.path.join(self.current_path, '../models/{current_run}')
        )
        self.gan_img_store_dir = os.path.normpath(
            os.path.join(self.current_path, 'resources/data/gan_images/')
        )

        self.save_checkpoint_every = 15
        self.keep_checkpoints = 2

        self.tag = None

    def __str__(self):
        return f'Config:' \
               f'\tstrftime = {self.strftime} \n' \
               f'\tgenerator = {self.generator} \n' \
               f'\tcurrent_path = {self.current_path} \n' \
               f'\tlevel_amount = {self.level_amount} \n' \
               f'\tlevel_path = {self.generated_level_path} \n' \
               f'\tgame_folder_path = {self.game_folder_path} \n' \
               f'\tai_path = {self.ai_path} \n' \
               f'\trescue_level_path = {self.rescue_level_path} \n' \
               f'\tevaluate = {self.evaluate} \n' \
               f'\trescue_level = {self.rescue_level} \n' \
               f'\tcreate_tensorflow_writer = {self.create_tensorflow_writer} \n' \
               f'\ttf_records_name = {self.tf_records_name} \n' \
               f'\ttrain_log_dir = {self.train_log_dir} \n' \
               f'\timage_store = {self.image_store} \n' \
               f'\tsave_checkpoint_every = {self.save_checkpoint_every} \n' \
               f'\tcheckpoint_dir = {self.checkpoint_dir} \n'

    @staticmethod
    def get_instance(args = None):
        if Config.instance is None:
            parser = get_program_arguments()
            parsed_args = parser.parse_args(args = args)
            Config.instance = Config(parsed_args)

        return Config.instance

    @staticmethod
    def get_instance_noargs(args):
        if Config.instance is None:
            Config.instance = Config(args)

        return Config.instance

    def get_generated_image_store(self):
        return self.image_store.replace("{timestamp}", self.strftime)

    def get_log_dir(self, run_name, strftime):
        return self.train_log_dir.replace("{timestamp}", strftime).replace("{current_run}", run_name)

    def get_current_log_dir(self, run_name):
        return self.train_log_dir.replace("{timestamp}", self.strftime).replace("{current_run}", run_name)

    def get_current_checkpoint_dir(self, run_name):
        return self.checkpoint_dir.replace("{timestamp}", self.strftime).replace("{current_run}", run_name)

    def get_checkpoint_dir(self, run_name, strftime):
        return self.checkpoint_dir.replace("{timestamp}", strftime).replace("{current_run}", run_name)

    def get_new_model_path(self, run_name):
        return self.new_model_path.replace("{current_run}", run_name)

    def get_tf_records(self, dataset_name: str):
        return self.tf_records_name.replace("{dataset_name}", dataset_name)

    def get_leve_path(self):
        return self.generated_level_path

    def get_game_path(self):
        return self.game_path

    def get_game_folder_path(self):
        return self.game_folder_path

    def get_game_level_path(self):
        return os.path.join(self.game_folder_path, self.copy_dest)

    def get_ai_path(self):
        return self.ai_path

    def get_data_train_path(self, folder = None):

        if folder is None:
            return self.data_train_path
        else:
            if folder[-1] != os.sep:
                folder += os.sep

            return os.path.join(os.path.normpath(self.data_train_path), os.path.normpath(folder))

    def get_pickle_folder(self):
        return self.pickle_folder

    def get_data_set(self, folder_name, file_name):
        if '.pickle' not in file_name:
            file_name += '.pickle'
        folder = os.path.join(self.dataset_folder, folder_name)
        Path(folder).mkdir(parents = True, exist_ok = True)
        file = os.path.normpath(os.path.join(folder, file_name))
        return file

    def get_pickle_file(self, file_name):
        if '.pickle' not in file_name:
            file_name += '.pickle'

        for path in Path(self.pickle_folder).rglob(file_name):
            return str(path)
        return None

    def get_epoch_run_data(self, run_name, epoch):
        folder_name = os.path.join(self.epoch_run_data, run_name.replace('.pickle', ''))
        Path(folder_name).mkdir(parents = True, exist_ok = True)

        file_name = f'epoch_{epoch}_{self.strftime}.pickle'

        return os.path.join(folder_name, file_name)

    def get_epoch_run_data_files(self, run_name):
        folder_name = os.path.join(self.epoch_run_data, run_name.replace('.pickle', ''))
        return Path(folder_name).glob('*.pickle')

    def get_data_root(self):
        return self.data_root

    def get_log_file(self, log_name):
        for path in Path(self.log_file_root).rglob(log_name):
            split = str(path).split(os.sep)
            return str(path), split[split.index('logs') + 1]
        else:
            return None

    def get_img_path(self, img_folder = None):
        if img_folder is not None:
            return os.path.join(self.image_root, img_folder)
        else:
            return self.image_root

    def get_data_tag(self):
        if self.tag is None:
            raise Exception("Pls make a meaningful tag du hupen")

        return self.tag

    def get_run_data(self, folder):
        if folder is not None:
            return os.path.join(self.run_data_root, folder) + ".pickle"
        else:
            return self.run_data_root

    def get_text_data(self, file):
        text_folder = os.path.join(self.data_root, "text")
        return os.path.join(text_folder, file) + ".txt"

    def get_eval_file(self, file_name):
        text_folder = os.path.join(self.eval_root, file_name)
        return text_folder + ".json"

    def get_fids_file(self, file_name):
        text_folder = os.path.join(self.metrics_root, file_name)
        return text_folder + ".json"

    def get_grid_search_file(self, file_name):
        text_folder = os.path.join(self.grid_search_root, file_name)
        return text_folder + ".pickle"

    def get_quality_search_folder(self, folder_name):
        text_folder = os.path.join(self.quality_root, folder_name)
        return text_folder + ".pdf"

    def get_conv_debug_img_file(self, file_name):
        img_file = os.path.join(self.conv_debug, file_name)
        return img_file + ".svg"

    def good_generated_level(self, file_name):
        img_file = os.path.join(self.good_generated_levels, file_name)
        return img_file + ".xml"

    def get_event_file(self, log_dir):
        for path in Path(log_dir).rglob('events.out.tfevents.*'):
            return str(path)
        return None

    def get_encoding_data(self, file_name):
        if file_name is not None:
            path = os.path.join(self.encoding_folder, file_name) + ".json"
            if os.path.isfile(path):
                with open(path, 'r') as f:
                    return json.loads(''.join(f.readlines()))
            else:
                return path
        else:
            return self.run_data_root

    def get_deconverted_file(self):
        next_free_level = 4
        for path in Path(self.log_file_root).glob('*.xml'):
            next_free_level = min(int(path.name[5:7]), next_free_level)
        return os.path.join(self.source_file_root, 'deconverted_levels') + os.sep + "level_" + (str(next_free_level) if len(str(next_free_level)) > 1 else '0' + str(next_free_level)) + ".xml"

    def get_gan_img_store(self, loaded_model):
        path = os.path.join(self.gan_img_store_dir, loaded_model) + (".pickle" if ".pickle" not in loaded_model else '')
        if not Path(path).exists():
            with open(path, 'wb') as handle:
                pickle.dump(dict(), handle, protocol = pickle.HIGHEST_PROTOCOL)
        return path

    def get_block_data(self, resoultion):
        block_data = self.get_encoding_data(f"encoding_res_{resoultion}")
        if type(block_data) is not str:
            resolution = block_data['resolution']
            del block_data['resolution']

        for block_idx, (key, value) in enumerate(block_data.items()):
            block_data[key]['idx'] = block_idx

        return block_data

    def set_game_folder_props(self, folder):
        os_name = platform.system()
        if os_name == 'Windows':
            self.game_folder_path = folder
            self.game_path = os.path.join(self.game_folder_path, "ScienceBirds.exe")
            self.copy_dest = os.path.normpath('ScienceBirds_Data/StreamingAssets/Levels/')
        elif os_name == 'Darwin':
            self.game_folder_path = folder
            self.game_path = os.path.join(self.game_folder_path, "ScienceBirds.app")
            self.copy_dest = os.path.normpath('Sciencebirds.app/Contents/Resources/Data/StreamingAssets/Levels')


if __name__ == '__main__':
    config = Config.get_instance()
    print(config)
