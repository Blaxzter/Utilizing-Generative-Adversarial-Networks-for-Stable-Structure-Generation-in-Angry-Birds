import base64
import io

import matplotlib.image as mpimg
from loguru import logger
from matplotlib import pyplot as plt

from game_management.GameConnection import GameConnection


def run_ai_on_level():
    game_connection = GameConnection(None)
    try:
        game_connection.start()

        logger.debug("Start game")
        game_connection.start_game()
        game_connection.wait_for_game_window()
        game_connection.load_level_menu()

        logger.debug("\nChange Level")
        game_connection.change_level(index = 3)

        logger.debug("\nGet Data")
        game_connection.get_data()

        logger.debug("Start AI")
        game_connection.startAi(start_level = 3, end_level = 4)
        game_connection.wait_till_all_level_played()

        logger.debug("\nGet Data Again")
        game_connection.get_data()

        game_connection.stop_components()
    except (KeyboardInterrupt, OSError) as e:
        logger.debug(e)
        game_connection.stop_components()


def picture():
    game_connection = GameConnection(None)
    try:
        game_connection.start()

        logger.debug("Start game")
        game_connection.start_game()
        game_connection.wait_for_game_window()
        game_connection.load_level_menu()

        logger.debug("Change Level")
        game_connection.change_level(index = 3)

        img_str = game_connection.get_img_data()

        i = base64.b64decode(img_str.strip('data:image/png;base64'))
        i = io.BytesIO(i)
        i = mpimg.imread(i, format = 'png')

        plt.imshow(i, interpolation = 'nearest')
        plt.show(dpi = 300)

        game_connection.stop_components()
    except (KeyboardInterrupt, OSError) as e:
        logger.debug(e)
        game_connection.stop_components()
        exit(1)


if __name__ == '__main__':
    picture()
