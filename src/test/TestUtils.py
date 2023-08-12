import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from util.Config import Config

config = Config.get_instance()


def plot_img(img, title = None, ax = None, flip = False, plot_always = False):
    if not plot_always and not config.plotting_enabled:
        return
    if ax is None:
        fig, ax = plt.subplots()
        im = ax.imshow(img)
        if title is not None:
            ax.set_title(title)
        plt.colorbar(im)

        if config.plot_to_file:
            fig_name = config.get_conv_debug_img_file(
                f'{config.plt_img_counter}_{title.replace(" ", "_").lower() if title is not None else "Img"}')
            fig.savefig(fig_name)
            config.plt_img_counter += 1
            plt.close(fig)
        else:
            plt.show()
    else:
        if flip:
            ax.imshow(np.flip(img, axis = 0))
        else:
            ax.imshow(img)
        ax.set_title(title)

def create_plt_array(dpi = 100, plot_always = False):
    if not plot_always and not config.plotting_enabled:
        return None, None

    fig, axs = plt.subplots(5, 3, figsize = (15, 10), dpi = dpi)
    axs = axs.flatten()
    fig.delaxes(axs[-1])
    fig.delaxes(axs[-2])
    return fig, axs


def plot_matrix(matrix, blocks, axs):
    if not config.plotting_enabled:
        return
    for layer in range(matrix.shape[-1]):
        _plot_img = matrix[:, :, layer]
        plot_img(_plot_img, blocks[layer]['name'], ax = axs[layer])
    plt.tight_layout()

    if config.plot_to_file:
        plt.savefig(config.get_conv_debug_img_file(f'{config.plt_img_counter}_matrix'))
        config.plt_img_counter += 1
        plt.close()
    else:
        plt.show(block = False)


def plot_matrix_complete(matrix, blocks = None, title = None, add_max = True, block = False, position = None,
                         delete_rectangles = None, flipped = False, selected_block = None, plot_always = False,
                         save_name = None):

    if not plot_always and not config.plotting_enabled:
        return

    fig, axs = create_plt_array(dpi = 100, plot_always = False)

    for layer_idx in range(matrix.shape[-1]):
        _plot_img = matrix[:, :, layer_idx]

        ax_title = ''
        if blocks is not None:
            ax_title = blocks[layer_idx]['name'] + (f' {np.max(_plot_img).item()}' if add_max else '')

        plot_img(_plot_img, ax_title, ax = axs[layer_idx], flip = flipped, plot_always = plot_always)
        color = 'blue' if (selected_block is not None and selected_block == layer_idx) else 'red'

        if position is not None:
            height = _plot_img.shape[0]
            if flipped:
                axs[layer_idx].scatter([position[1]], [height - position[0] - 1], color = 'red', s = 1)
            else:
                axs[layer_idx].scatter([position[1]], [position[0]], color = 'red', s = 1)

        if delete_rectangles is not None:
            height = _plot_img.shape[0]
            (start, end, top, bottom) = delete_rectangles[layer_idx]
            if flipped:
                axs[layer_idx].add_patch(
                    Rectangle((start, height - top - 1), end - start, top - bottom,
                              fill = False, color = color, linewidth = 1))
            else:
                axs[layer_idx].add_patch(
                    Rectangle((start, top), end - start, bottom - top,
                              fill = False, color = color, linewidth = 1))

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    if config.plot_to_file:
        file_name = f'{config.plt_img_counter}_{title.replace(" ", "_").lower() if title is not None else "matrix"}'
        if save_name is not None:
            file_name = f'{config.plt_img_counter}_{save_name}'
        fig.savefig(config.get_conv_debug_img_file(file_name))
        config.plt_img_counter += 1
        plt.close(fig)
    else:
        plt.show(block = block)
