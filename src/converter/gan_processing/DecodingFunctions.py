import numpy as np


class DecodingFunctions:

    def __init__(self, shift_value = None, max_value = None, rescaling = None, threshold_callback = lambda: 0.5):

        self.rescaling = rescaling
        self.max_value = max_value
        self.shift_value = shift_value
        self.threshold_callback = threshold_callback

    def set_rescaling(self, rescaling):
        self.rescaling = rescaling

    def update_rescale_values(self, max_value, shift_value):
        self.max_value = max_value
        self.shift_value = shift_value

    def rescale(self, orig_img):
        return self.rescaling(self.max_value / 2)(orig_img + self.shift_value).numpy()

    def default_rint_rescaling(self, orig_img):
        norm_img = self.rescaling(self.max_value / 2)(orig_img + self.shift_value).numpy()
        return np.rint(norm_img), norm_img

    def threshold_rint_rescaling(self, orig_img):
        threshold = float(self.threshold_callback())

        norm_img = self.rescaling(self.max_value / 2)(orig_img + self.shift_value).numpy()
        norm_img[norm_img < threshold] = 0

        return np.rint(norm_img), norm_img

    def default_rint_rescaling(self, orig_img):
        norm_img = self.rescaling(self.max_value / 2)(orig_img + self.shift_value).numpy()
        return np.rint(norm_img), norm_img

    def argmax_multilayer_decoding(self, orig_img):
        threshold = float(self.threshold_callback())

        norm_img = self.rescaling(self.max_value / 2)(orig_img + self.shift_value).numpy()
        stacked_img = np.dstack((np.zeros((128, 128)) + threshold, norm_img))
        return np.argmax(stacked_img, axis = 2), norm_img

    def orig_multilayer_decoding(self, orig_img):
        threshold = float(self.threshold_callback())

        norm_img = self.rescaling(self.max_value / 2)(orig_img + self.shift_value).numpy()
        stacked_img = np.dstack((np.zeros((128, 128)) + threshold, norm_img))
        arg_max = np.argmax(stacked_img, axis = 2)
        ret_img = np.zeros(stacked_img.shape[:2])

        for dim in range(1, stacked_img.shape[-1]):
            ret_img[arg_max == dim] = stacked_img[arg_max == dim, dim]

        return ret_img

    def argmax_multilayer_decoding_with_air(self, orig_img, rescale = True):
        if rescale:
            norm_img = self.rescaling(self.max_value / 2)(orig_img + self.shift_value).numpy()
        else:
            norm_img = orig_img

        threshold = float(self.threshold_callback())
        norm_img[norm_img[:, :, 0] < threshold, 0] = 0

        return np.argmax(norm_img, axis = 2), norm_img

    def one_element_multilayer(self, orig_img):
        threshold = float(self.threshold_callback())

        norm_img = self.rescaling(self.max_value / 2)(orig_img + self.shift_value).numpy()
        return self.create_single_layer_img(multilayer_img = norm_img, air_threshold = threshold), norm_img

    @staticmethod
    def create_single_layer_img(multilayer_img, air_threshold = 0.5):
        stacked_img = np.dstack((np.zeros(multilayer_img.shape[:2]) + air_threshold, multilayer_img))

        arg_max = np.argmax(stacked_img, axis = 2)
        ret_img = np.zeros(multilayer_img.shape[:2])

        for dim in range(1, multilayer_img.shape[-1]):
            ret_img[arg_max == dim] = np.rint(stacked_img[arg_max == dim, dim]) + (dim - 1) * 13

        ret_img[arg_max == 4] = 40

        return ret_img