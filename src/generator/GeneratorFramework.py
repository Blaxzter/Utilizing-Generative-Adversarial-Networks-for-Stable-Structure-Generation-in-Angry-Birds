from game_management.GameConnection import GameConnection
from game_management.GameManager import GameManager
from util.Config import Config
from util.Evaluator import Evaluator


class GeneratorFramework:
    def __init__(self, generator, conf: Config):
        # Initialize the generated level path from the configuration
        self.generated_level_path = conf.generated_level_path

        # Initialize the level generator using the configuration
        self.generator = generator

        # Initialize the game connection using the configuration
        self.game_connection = GameConnection(conf = conf)

        # Initialize the game manager using the configuration and game connection
        self.game_manager = GameManager(conf = conf, game_connection = self.game_connection)

        # Initialize the evaluator using the configuration and game connection
        self.evaluator = Evaluator(conf = conf, game_connection = self.game_connection)

    def run(self):
        # Run the generation and evaluation processes
        self.generate()
        self.evaluate()

    def stop(self):
        # Stop the game using the game manager
        self.game_manager.stop_game()

    def generate(self):
        # Generate the initial level using the generator and the folder path
        self.generator.generate_level_init(
            folder_path = self.generated_level_path,
        )

    def evaluate(self):
        # Start the game using the game manager
        self.game_manager.start_game()

        # Copy the game levels and iterate through them
        copied_levels = self.game_manager.copy_game_levels()
        for copied_level in copied_levels:
            # Extract the level index from the copied level name
            level_index = int(copied_level.name[6:8])

            # Evaluate the levels starting from the extracted index
            self.evaluator.evaluate_levels(start_level = level_index)
