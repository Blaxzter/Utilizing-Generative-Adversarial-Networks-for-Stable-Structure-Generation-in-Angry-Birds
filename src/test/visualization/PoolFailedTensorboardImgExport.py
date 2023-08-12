import os
from multiprocessing import Manager, Pool

import imageio
from tqdm.auto import tqdm

from util.Config import Config


def extract_data(element, p_dict, decode_func, decode_img, tftype):
    for value in element.summary.value:
        if value.metadata.plugin_data.plugin_name == 'scalars':
            if value.tag not in p_dict.keys():
                p_dict[value.tag] = dict()

            p_dict[value.tag][element.step] = \
                decode_func(value.tensor.tensor_content, tftype).numpy().item()

        elif value.metadata.plugin_data.plugin_name == 'images':
            if 'image' not in p_dict.keys():
                p_dict['image'] = dict()

            s = value.tensor.string_val[2]  # first elements are W and H
            tf_img = decode_img(s)  # [H, W, C]
            np_img = tf_img.numpy()
            p_dict['image'][element.step] = np_img[96:337, 607:1186]

def save_images_from_event(event_filename, tag, output_dir = './'):
    import tensorflow as tf
    from tensorflow.core.util import event_pb2
    assert (os.path.isdir(output_dir))

    data_set = tf.data.TFRecordDataset(event_filename)
    serialized = list(map(event_pb2.Event.FromString, map(lambda x: x.numpy(), data_set)))

    process_pool = Pool(None)
    managed_dict = Manager().dict()
    progress_bar = tqdm(total = len(serialized))

    results = []

    def update(*a):
        progress_bar.update()

    for data_element in serialized:
        # extract_data(data_element, managed_dict)
        res = process_pool.apply_async(
            func = extract_data,
            args = (data_element, managed_dict, tf.io.decode_raw, tf.image.decode_image, tf.float32),
            callback = update
        )
        results.append(res)

    [result.wait() for result in results]

    print(managed_dict)

    imageio.mimsave('movie.mp4', managed_dict['images'].values())


if __name__ == '__main__':
    config: Config = Config.get_instance()
    img_folder = config.get_img_path("generated")
    log_file = config.get_log_file("events.out.tfevents.1654852719.ubuntu.25142.0.v2")

    save_images_from_event(log_file, 'Training data', img_folder)
