from loguru import logger
from game_management.GameConnection import GameConnection
from util.Config import Config


class Evaluator:

    def __init__(self, conf: Config, game_connection: GameConnection):
        self.start_level = 4
        self.end_level = 4

        self.data = {}

        self.conf: Config = conf
        self.game_connection: GameConnection = game_connection

    def evaluate_levels(self, start_level = 4, end_level = -1):
        logger.debug("\nChange Level")
        self.game_connection.change_level(index = start_level)

        logger.debug("Start AI")
        self.game_connection.startAi(start_level = start_level, end_level = end_level)
        self.game_connection.wait_till_all_level_played()

        logger.debug("\nGet Data")
        data = self.game_connection.get_data()
        logger.debug(data)
