import os
import pickle
from tkinter import ttk

import matplotlib
import numpy as np
from loguru import logger
from mpl_toolkits.axes_grid1 import make_axes_locatable

import test.encodings.ConvolutionTest
from converter.gan_processing.DecodingFunctions import DecodingFunctions
from converter.to_img_converter import DecoderUtils
from converter.to_img_converter.MultiLayerStackDecoder import MultiLayerStackDecoder
from generator.gan.SimpleGans import SimpleGAN88212

matplotlib.use("TkAgg")

from tkinter import *
from matplotlib import pyplot as plt

from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from util.Config import Config


class GeneratorApplication:

    def __init__(self, frame = None, level_drawer = None):
        self.config: Config = Config.get_instance()
        self.canvas = None

        self.level_drawer = level_drawer

        self.toolbar = None
        self.gan = None

        self.top_frame = frame

        self.single_element = False
        self.rescaling = None
        self.seed = None
        self.small_version = False

        self.model_loads = {
            'Standard GAN 1': self.load_model_0,
            'Standard GAN 2': self.load_model_1,
            'W-GAN SGD': self.load_model_2,
            'W-GAN ADAM': self.load_model_3,
            'Big Gan Multilayer': self.load_multilayer_encoding,
            'Multilayer With Air (AIIDE)': self.multilayer_with_air,
            'Multilayer With Air - RELU': self.multilayer_with_air_new,
            'One Element Encoding': self.load_one_element_encoding,
            'One Element Multilayer': self.load_one_element_multilayer,
            'True One Hot': self.load_true_one_hot,
            'Small True One Hot With Air': self.small_true_one_hot_with_air
        }

        if frame is None:
            self.create_window()

        self.selected_model = StringVar()
        self.load_stored_img = StringVar()
        self.create_buttons()

        if self.gan is not None:
            self.seed = self.gan.create_random_vector()

        self.decoding_functions = DecodingFunctions(threshold_callback = lambda: self.threshold_text.get('0.0', 'end'))
        self.img_decoding = self.decoding_functions.default_rint_rescaling

        self.multilayer_stack_decoder = MultiLayerStackDecoder(level_drawer = level_drawer, add_tab = True)

        if frame is None:
            # run the gui
            self.window.mainloop()

    def generate_img(self, created_img = None):
        import tensorflow as tf
        with tf.device('/CPU:0'):
            if created_img is None:
                orig_img, pred = self.gan.create_img(seed = self.seed)
            else:
                orig_img, pred = created_img

            if self.level_drawer is None:
                if self.canvas:
                    self.canvas.get_tk_widget().destroy()

                if self.toolbar:
                    self.toolbar.destroy()

                fig, ax = plt.subplots(1, 1, dpi = 100)
            else:
                self.level_drawer.clear_figure_canvas()

            # Use the defined decoding algorithm
            img, norm_img = self.img_decoding(orig_img[0])

            if len(orig_img[0].shape) == 3 and not self.single_element:
                if str(type(orig_img)) != '<class \'numpy.ndarray\'>':
                    orig_img = orig_img.numpy()

                layer_amount = orig_img[0].shape[-1]
                if layer_amount > 1:
                    for layer in range(1 if self.uses_air_layer else 0, layer_amount):
                        orig_img[0, orig_img[0, :, :, layer] > 0, layer] += layer + self.decoding_functions.max_value

                viz_img = np.max(orig_img[0], axis = 2)
            else:
                viz_img = img

            trimmed_img, trim_data = DecoderUtils.trim_img(img.reshape(img.shape[0:2]), ret_trims = True)
            self.main_visualization = trimmed_img

            depth = orig_img.shape[-1]

            if depth == 1:
                rounded_image = norm_img
            else:
                rounded_image = viz_img

            top_space, bottom_space, left_space, right_space = trim_data
            bottom_trim = rounded_image.shape[0] - bottom_space
            right_trim = rounded_image.shape[1] - right_space

            fig, ax = self.level_drawer.new_fig()
            self.create_plt_img(ax, fig, f'Probability {pred.item()}', viz_img)
            self.level_drawer.add_tab_to_fig_canvas(fig, ax, name = f'Full Img')

            # fig, ax = self.level_drawer.new_fig()
            # self.create_plt_img(ax, fig, f'Probability {pred.item()}', self.decoding_functions.orig_multilayer_decoding(orig_img[0]))
            # self.level_drawer.add_tab_to_fig_canvas(fig, ax, name = f'Original Value space')

            fig, ax = self.level_drawer.new_fig()
            self.create_plt_img(ax, fig, f'Probability {pred.item()}', np.rint(rounded_image[top_space: bottom_trim, left_space: right_trim]))
            self.level_drawer.add_tab_to_fig_canvas(fig, ax, name = f'Rounded to next Integer')

            if self.level_drawer is None:
                self.canvas = FigureCanvasTkAgg(fig, master = self.window)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack()

                # creating the Matplotlib toolbar
                self.toolbar = NavigationToolbar2Tk(self.canvas, self.window)
                self.toolbar.update()
            else:
                # Draw combined img to level

                self.level_drawer.grid_drawer.create_tab_panes(1 if depth == 1 or depth > 10 else depth + 1)
                self.level_drawer.draw_level(trimmed_img, tab = 0)
                if depth != 1:

                    # Dont visualize every layer if the amount of layers is to high
                    if depth > 10:
                        return

                    for i in range(1, depth + 1):
                        fig, ax = self.level_drawer.new_fig()
                        self.create_plt_img(ax, fig, f'Layer {i}', orig_img[0, :, :, i - 1])
                        self.level_drawer.add_tab_to_fig_canvas(fig, ax, name = f'Layer {i}')

                        top_space, bottom_space, left_space, right_space = trim_data
                        bottom_trim = rounded_image.shape[0] - bottom_space
                        right_trim = rounded_image.shape[1] - right_space
                        fig, ax = self.level_drawer.new_fig()
                        self.create_plt_img(ax, fig, f'Probability {pred.item()}',
                                            np.rint(norm_img[top_space: bottom_trim, left_space: right_trim, i - 1]))
                        self.level_drawer.add_tab_to_fig_canvas(fig, ax, name = f'Layer {i} rint')

                        self.level_drawer.draw_level(
                            np.rint(norm_img[top_space: bottom_trim, left_space: right_trim, i - 1]), tab = i)

    def create_plt_img(self, ax, fig, plt_title, viz_img):
        im = ax.imshow(viz_img)
        # ax.set_title(plt_title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size = '5%', pad = 0.05)
        fig.colorbar(im, cax = cax, orientation = 'vertical')

    def load_gan(self):
        import tensorflow as tf

        with tf.device('/CPU:0'):

            # Load the model
            self.model_loads[self.selected_model.get()]()
            self.seed = self.gan.create_random_vector()
            self.level_drawer.draw_mode.set('OneElement' if self.single_element else 'LevelImg')
            self.level_drawer.combobox.set(self.level_drawer.draw_mode.get())

            self.load_stored_imgs()
            self.decoding_functions.set_rescaling(rescaling = tf.keras.layers.Rescaling)

            checkpoint = tf.train.Checkpoint(
                generator_optimizer = tf.keras.optimizers.Adam(1e-4),
                discriminator_optimizer = tf.keras.optimizers.Adam(1e-4),
                generator = self.gan.generator,
                discriminator = self.gan.discriminator
            )
            manager = tf.train.CheckpointManager(
                checkpoint, self.checkpoint_dir, max_to_keep = 2
            )
            checkpoint.restore(manager.latest_checkpoint)
            if manager.latest_checkpoint:
                logger.debug("Restored from {}".format(manager.latest_checkpoint))
            else:
                logger.debug("Initializing from scratch.")

            orig_img, pred = self.gan.create_img(self.seed)
            self.generate_img(created_img = (orig_img, pred))

    def new_seed(self):
        import tensorflow as tf
        with tf.device('/CPU:0'):
            self.seed = self.gan.create_random_vector()

    def create_window(self):
        self.window = Tk()
        self.window.title('Gan Level Generator')
        self.window.geometry("1200x800")

        self.top_frame = Frame(self.window, width = 1200, height = 50, pady = 3)
        self.top_frame.pack()

    def create_buttons(self):
        wrapper = Canvas(self.top_frame)
        wrapper.pack(side = LEFT, padx = (10, 10), pady = (20, 10))
        label = Label(wrapper, text = "Loaded Model:")
        label.pack(side = TOP)

        self.combobox = ttk.Combobox(wrapper, textvariable = self.selected_model, width = 30, state = "readonly")
        self.combobox['values'] = list(self.model_loads.keys())
        self.combobox['state'] = 'readonly'
        self.combobox.bind('<<ComboboxSelected>>', lambda event: self.load_gan())
        self.combobox.pack(side = TOP)

        plot_button = Button(
            master = self.top_frame,
            command = lambda: self.generate_img(),
            height = 2,
            width = 10,
            text = "Generate Img")

        plot_button.pack(side = LEFT, pady = (20, 10), padx = (10, 10))

        plot_button = Button(
            master = self.top_frame,
            command = lambda: self.new_seed(),
            height = 2,
            width = 10,
            text = "New Seed")

        plot_button.pack(side = LEFT, pady = (20, 10), padx = (10, 10))

        wrapper = Canvas(self.top_frame)
        wrapper.pack(side = LEFT, padx = (10, 10), pady = (20, 10))
        label = Label(wrapper, text = "Multilayer Threshold")
        label.pack(side = LEFT)

        self.threshold_text = Text(wrapper, height = 1, width = 10)
        self.threshold_text.insert('0.0', '0.5')
        self.threshold_text.pack(side = LEFT, padx = (10, 10))

        wrapper = Canvas(self.top_frame)
        wrapper.pack(side = LEFT, padx = (20, 10), pady = (20, 10))
        label = Label(wrapper, text = "Store Comment:")
        label.pack(side = LEFT)

        self.img_store_comment = Text(wrapper, height = 1, width = 30)
        self.img_store_comment.insert('0.0', ' ')
        self.img_store_comment.pack(side = LEFT, padx = (10, 10))

        store_img = Button(
            master = self.top_frame,
            command = lambda: self.store_gan_img(),
            height = 2,
            width = 15,
            text = "Store GAN Output")
        store_img.pack(side = LEFT, pady = (20, 10), padx = (10, 10))

        # Display loaded images
        wrapper = Canvas(self.top_frame)
        wrapper.pack(side = LEFT, padx = (10, 10), pady = (20, 10))
        label = Label(wrapper, text = "Loaded output:")
        label.pack(side = TOP)

        self.stored_images = ttk.Combobox(wrapper, textvariable = self.load_stored_img, width = 30, state = "readonly")
        self.stored_images['state'] = 'readonly'
        self.stored_images.bind('<<ComboboxSelected>>', lambda event: self.display_loaded_img())
        self.stored_images.pack(side = TOP)

        store_img = Button(
            master = self.top_frame,
            command = lambda: self.delete_selected_img(),
            height = 2,
            width = 2,
            text = "X")
        store_img.pack(side = LEFT, pady = (20, 10), padx = (10, 10))


        store_img = Button(
            master = self.top_frame,
            command = lambda: self.get_decode_parameter(callback = self.decode_gan_img),
            height = 2,
            width = 15,
            text = "Decode Gan Img")
        store_img.pack(side = LEFT, pady = (20, 10), padx = (10, 10))

    def load_stored_imgs(self):
        loaded_model = self.selected_model.get().replace(' ', '_').lower()
        self.store_imgs_pickle_file = self.config.get_gan_img_store(loaded_model)
        # check if store_imgs_pickle_file exists and create it if not
        if not os.path.exists(self.store_imgs_pickle_file):
            self.loaded_outputs = dict()
            with open(self.store_imgs_pickle_file, 'wb') as handle:
                pickle.dump(self.loaded_outputs, handle, protocol = pickle.HIGHEST_PROTOCOL)

        with open(self.store_imgs_pickle_file, 'rb') as f:
            self.loaded_outputs = pickle.load(f)

        self.stored_images['values'] = list(self.loaded_outputs.keys())
        self.stored_images.set('')
        self.img_store_comment.delete('0.0', END)

    def store_gan_img(self):
        orig_img, prediction = self.gan.create_img(self.seed)
        comment = self.img_store_comment.get('0.0', 'end')
        comment = comment.strip()
        self.loaded_outputs[comment] = dict(
            output = orig_img.numpy(),
            prediction = prediction,
            seed = self.seed,
            comment = comment
        )

        with open(self.store_imgs_pickle_file, 'wb') as handle:
            pickle.dump(self.loaded_outputs, handle, protocol = pickle.HIGHEST_PROTOCOL)

        self.stored_images['values'] = list(self.loaded_outputs.keys())
        self.img_store_comment.delete('0.0', END)

    def display_loaded_img(self):
        loaded_data = self.loaded_outputs[self.stored_images.get()]
        orig_img = loaded_data['output']
        prediction = loaded_data['prediction']
        self.seed = loaded_data['seed']
        depth = orig_img.shape[-1]

        self.level_drawer.grid_drawer.create_tab_panes(1 if depth == 1 else depth + 1)
        self.level_drawer.create_img_tab_panes(1 if depth == 1 else depth + 1)
        self.generate_img(created_img = (orig_img, prediction))

    def delete_selected_img(self):
        del self.loaded_outputs[self.stored_images.get()]

        with open(self.store_imgs_pickle_file, 'wb') as handle:
            pickle.dump(self.loaded_outputs, handle, protocol = pickle.HIGHEST_PROTOCOL)

        self.stored_images['values'] = list(self.loaded_outputs.keys())

    def get_decode_parameter(self, callback):
        self.parameter_dict = dict(
            use_drawn_level = dict(type = 'bool', default = True),
            round_to_next_int = dict(type = 'bool', default = False),
            use_negative_air_value = dict(type = 'bool', default = True),
            negative_air_value = dict(type = 'number', default = -2),
            custom_kernel_scale = dict(type = 'bool', default = True),
            minus_one_border = dict(type = 'bool', default = False),
            cutoff_point = dict(type = 'number', default = 0.85),
            bird_cutoff = dict(type = 'number', default = 0.5),
            recalibrate_blocks = dict(type = 'bool', default = False),
            combine_layers = dict(type = 'bool', default = False),
            disable_plotting = dict(type = 'bool', default = True)
        )
        self.level_drawer.create_parameter_popup(self.parameter_dict, ok_button_text = 'Decode Img', callback = callback)

    def decode_gan_img(self):

        self.level_drawer.clear_figure_canvas()

        for key, value in self.parameter_dict.items():
            if hasattr(self.multilayer_stack_decoder, key):
                setattr(self.multilayer_stack_decoder, key, value['data'])

        if self.parameter_dict['disable_plotting']['data'] == 1:
            self.multilayer_stack_decoder.visualizer.disable = True
        else:
            self.multilayer_stack_decoder.visualizer.disable = False

        if self.gan is not None and self.parameter_dict['use_drawn_level']['data'] == 0:
            orig_img, prediction = self.gan.create_img(self.seed)
            orig_img = orig_img.numpy()

            if not self.single_element:
                self.level_drawer.level = self.multilayer_stack_decoder.decode(orig_img, self.uses_air_layer)
            else:
                self.level_drawer.level = self.level_drawer.level_id_img_decoder.decode_one_hot_encoding(orig_img)
        else:
            self.level_drawer.level = self.multilayer_stack_decoder.layer_to_level(self.level_drawer.grid_drawer.current_elements())

        fig, ax = self.level_drawer.new_fig()
        self.level_drawer.level_visualizer.create_img_of_structure(
            self.level_drawer.level.get_used_elements(), use_grid = False, ax = ax, scaled = True
        )
        self.level_drawer.add_tab_to_fig_canvas(fig, ax, name = 'Decoded Level')

    def load_model_0_0(self):
        from generator.gan.SimpleGans import SimpleGAN88212
        self.checkpoint_dir = self.config.get_new_model_path('simple_gan')
        self.gan = SimpleGAN88212()
        self.decoding_functions.update_rescale_values(max_value = 4, shift_value = 0)
        self.img_decoding = self.decoding_functions.default_rint_rescaling
        self.single_element = False
        self.small_version = False
        self.uses_air_layer = False

    def load_model_0(self):
        from generator.gan.SimpleGans import SimpleGAN100112
        self.checkpoint_dir = self.config.get_new_model_path('Standard GAN 1')
        self.gan = SimpleGAN100112()
        self.decoding_functions.update_rescale_values(max_value = 4, shift_value = 0)
        self.img_decoding = self.decoding_functions.default_rint_rescaling
        self.single_element = False
        self.small_version = False
        self.uses_air_layer = False

    def load_model_1(self):
        from generator.gan.SimpleGans import SimpleGAN100116
        self.checkpoint_dir = self.config.get_new_model_path('Standard GAN 2')
        self.gan = SimpleGAN100116()
        self.decoding_functions.update_rescale_values(max_value = 4, shift_value = 0)
        self.img_decoding = self.decoding_functions.default_rint_rescaling
        self.single_element = False
        self.small_version = False
        self.uses_air_layer = False

    def load_model_2(self):
        from generator.gan.SimpleGans import SimpleGAN100116
        self.checkpoint_dir = self.config.get_new_model_path('W-GAN SGD')
        self.gan = SimpleGAN100116()
        self.decoding_functions.update_rescale_values(max_value = 4, shift_value = 0)
        self.img_decoding = self.decoding_functions.default_rint_rescaling
        self.single_element = False
        self.small_version = False
        self.uses_air_layer = False

    def load_model_3(self):
        from generator.gan.SimpleGans import SimpleGAN100116
        self.checkpoint_dir = self.config.get_new_model_path('W-GAN ADAM')
        self.gan = SimpleGAN100116()
        self.decoding_functions.update_rescale_values(max_value = 4, shift_value = 0)
        self.img_decoding = self.decoding_functions.default_rint_rescaling
        self.single_element = False
        self.small_version = False
        self.uses_air_layer = False

    def load_multilayer_encoding(self):
        from generator.gan.BigGans import WGANGP128128_Multilayer
        self.checkpoint_dir = self.config.get_new_model_path('Big Gan Multilayer')
        self.gan = WGANGP128128_Multilayer()
        self.decoding_functions.update_rescale_values(max_value = 1, shift_value = 1)
        self.img_decoding = self.decoding_functions.argmax_multilayer_decoding
        self.single_element = False
        self.small_version = False
        self.uses_air_layer = False

    def load_one_element_encoding(self):
        from generator.gan.BigGans import WGANGP128128
        self.checkpoint_dir = self.config.get_new_model_path('One Element Encoding')
        self.decoding_functions.update_rescale_values(max_value = 40, shift_value = 1)
        self.img_decoding = self.decoding_functions.threshold_rint_rescaling
        self.gan = WGANGP128128()
        self.single_element = True
        self.small_version = False
        self.uses_air_layer = False

    def load_one_element_multilayer(self):
        from generator.gan.BigGans import WGANGP128128_Multilayer
        self.checkpoint_dir = self.config.get_new_model_path('One Element Multilayer')
        self.decoding_functions.update_rescale_values(max_value = 14, shift_value = 1)
        self.img_decoding = self.decoding_functions.one_element_multilayer
        self.gan = WGANGP128128_Multilayer()
        self.single_element = True
        self.small_version = False
        self.uses_air_layer = False

    def load_true_one_hot(self):
        from generator.gan.BigGans import WGANGP128128_Multilayer
        self.checkpoint_dir = self.config.get_new_model_path('True One Hot')
        self.decoding_functions.update_rescale_values(max_value = 1, shift_value = 1)
        self.img_decoding = self.decoding_functions.argmax_multilayer_decoding
        self.gan = WGANGP128128_Multilayer(last_dim = 40)
        self.single_element = True
        self.small_version = False
        self.uses_air_layer = False

    def small_true_one_hot_with_air(self):
        from generator.gan.BigGans import WGANGP128128_Multilayer
        self.checkpoint_dir = self.config.get_new_model_path('Small True One Hot With Air')
        self.decoding_functions.update_rescale_values(max_value = 1, shift_value = 1)
        self.img_decoding = self.decoding_functions.argmax_multilayer_decoding_with_air
        self.gan = WGANGP128128_Multilayer(last_dim = 15)
        self.single_element = True
        self.small_version = True
        self.uses_air_layer = False

    def multilayer_with_air(self):
        from generator.gan.BigGans import WGANGP128128_Multilayer
        self.checkpoint_dir = self.config.get_new_model_path('Multilayer With Air (AIIDE)')
        self.decoding_functions.update_rescale_values(max_value = 1, shift_value = 1)
        self.img_decoding = self.decoding_functions.argmax_multilayer_decoding_with_air
        self.gan = WGANGP128128_Multilayer(last_dim = 5)
        self.single_element = False
        self.small_version = False
        self.uses_air_layer = True

    def multilayer_with_air_new(self):
        from generator.gan.BigGans import WGANGP128128_Multilayer
        self.checkpoint_dir = self.config.get_new_model_path('Multilayer With Air - RELU')
        self.decoding_functions.update_rescale_values(max_value = 1, shift_value = 1)
        self.img_decoding = self.decoding_functions.argmax_multilayer_decoding_with_air
        self.gan = WGANGP128128_Multilayer(last_dim = 5, last_layer = 'ReLU')
        self.single_element = False
        self.small_version = False
        self.uses_air_layer = True
