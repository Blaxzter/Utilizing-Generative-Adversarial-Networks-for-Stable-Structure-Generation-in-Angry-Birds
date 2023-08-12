
from tqdm import tqdm
import os
from glob import glob
import shutil

from converter.to_text_converter.xml2text import xml2txt
from util.Config import Config


def main():
    config = Config.get_instance()
    orig_level_folder = config.get_data_train_path(folder = 'generated/single_structure')
    train_out_file = config.get_text_data('train')

    converter = xml2txt(orig_level_folder)
    train_data, remove_file_list = converter.xml2vector(True)
    for i, data in enumerate(train_data):
        data_ = []
        for d in data:
            if sum(d) == 0:
                break
            data_.append(d)
        train_data[i] = data_

    with open(train_out_file, "w") as f:
        for train_d in tqdm(train_data):
            train_d_ = ""
            for d in train_d:
                d = list(map(str, d))
                c = " ".join(d)
                train_d_ += c + "  "
            f.write(str(train_d_)+"\n")


if __name__ == "__main__":
    main()
