import pickle
from pathlib import Path

from icecream import ic

from util.Config import Config

config = Config.get_instance()


def load_test_images_of_model(model_name):
    loaded_model = model_name.replace(' ', '_').lower()
    store_imgs_pickle_file = config.get_gan_img_store(loaded_model)

    with open(store_imgs_pickle_file, 'rb') as f:
        loaded_outputs = pickle.load(f)

    return loaded_outputs, store_imgs_pickle_file


if __name__ == '__main__':
    for file in Path(config.gan_img_store_dir).glob('*'):
        file_name = str(file.name)
        ic(file_name)
        outputs, file_path = load_test_images_of_model(file_name)

        out_dict = dict()

        for key, output in outputs.items():
            out_dict[key.strip()] = output

            # outputs[key]['output'] = outputs[key]['output'].numpy()
            # outputs[key]['seed'] = outputs[key]['seed'].numpy()
            # outputs[key]['comment'] = outputs[key]['comment'].strip()

        with open(file_path, 'wb') as handle:
            pickle.dump(out_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)
