import base64
import io
import json
import os
import platform
import subprocess
import threading
import time
from subprocess import Popen as new
from typing import Optional
from wsgiref.simple_server import make_server

import matplotlib.image as mpimg
from loguru import logger
from wsocket import WSocketApp, logger, FixedHandler, ThreadingWSGIServer

from util.Config import Config


class GameConnection(threading.Thread):
    def __init__(self, conf: Config = None, host = 'localhost', port = 9001, stop_if_game_windows_close = True):
        super().__init__()

        self.conf = conf
        if conf is None:
            self.conf = Config.get_instance()

        self.ai_path = self.conf.get_ai_path()
        self.condition_object = threading.Condition()

        self.host = host
        self.port = port

        self.wsocket_app = WSocketApp()
        self.wsocket_app.onconnect += self.new_client
        self.wsocket_app.onmessage += self.game_response
        self.wsocket_app.onclose += self.game_window_closed
        self.server = None

        self._stop_event = threading.Event()
        self.stop_if_game_windows_close = stop_if_game_windows_close

        self.response = None

        self.client = None
        self.game_process: Optional[subprocess.Popen[str]] = None
        self.ai_process: Optional[subprocess.Popen[str]] = None
        self.counter = 0

        self.wait_for_all_levels_played_tries = 10

    def run(self):
        self.server = make_server(self.host, self.port, self.wsocket_app, ThreadingWSGIServer, FixedHandler)
        try:
            self.server.serve_forever()
        except Exception:
            logger.debug("\nServer stopped.")
            self.server.server_close()

    def wait_for_response(self):
        # Wait for game response
        if self.response is None:
            logger.debug("Wait for message")
            self.condition_object.acquire()
            self.condition_object.wait()
            self.condition_object.release()
            msg = self.response
            self.response = None
        else:
            msg = self.response
            self.response = None

        # print(f'Received({self.port}): {msg}')
        return msg

    def game_response(self, message: str, client):
        if len(message) > 200:
            logger.debug(f"Game Response: {message[:200]}")
        else:
            logger.debug(f"Game Response: {message}")
        self.response = json.loads(message)
        self.condition_object.acquire()
        self.condition_object.notify()
        self.condition_object.release()

    def game_window_closed(self, message, client):
        logger.debug(f"Game Window Closed")
        self.stop_game()

        if self.ai_process:
            self.ai_process.terminate()
            self.ai_process.wait()

        self.client = None

        if self.stop_if_game_windows_close:
            logger.debug("Stop server")
            self.server.shutdown()

    def start_game(self, game_path = '../science_birds/win-new/ScienceBirds.exe'):
        os_name = platform.system()
        if os_name == 'Windows':
            print(f"Start game: {game_path}")
            self.game_process = new([game_path, f'generatorPort', str(self.port)], shell = False, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT)
        elif os_name == 'Darwin':
            os.system(f"open {game_path}")

    def startAi(self, start_level = -1, end_level = -1, print_ai_log = True, ai_path = None):

        if self.ai_process:
            self.ai_process.terminate()
            self.ai_process.wait()
            self.ai_process = None

        # Set the game into ai modus and define which level to be played
        message = [0, 'aimodus', f'{{"mode": true, "startLevel" : {start_level}, "endLevel": {end_level}}}']
        logger.debug(message)
        self.send(message)
        self.wait_for_response()

        if ai_path is None:
            ai_path = self.ai_path

        if print_ai_log:
            self.ai_process = new(f"java -jar {ai_path}", shell = False)
        else:
            self.ai_process = new(f"java -jar {ai_path}", shell = False, stdout = subprocess.DEVNULL,
                                  stderr = subprocess.STDOUT)

    def stopAI(self):

        message = [0, 'aimodus', f'{{"mode": false }}']
        self.send(message)
        self.wait_for_response()

        if self.ai_process:
            self.ai_process.terminate()
            self.ai_process.wait()

    def stop_components(self):
        self.stop_game()
        if self.ai_process:
            self.ai_process.terminate()
        if self.server:
            self.server.server_close()
        self._stop_event.set()

    def new_client(self, client):
        logger.debug(f"New game window connected {client}")
        self.client = client
        self.condition_object.acquire()
        self.condition_object.notify()
        self.condition_object.release()

    def send(self, msg):
        print(f"Send({self.port}): {msg}")
        self.response = None
        self.client.send(json.dumps(msg))

    def wait_for_game_window(self):
        counter = 0
        self.condition_object.acquire()
        while self.client is None:
            logger.debug(f"Waiting for client window: {counter}")
            self.condition_object.wait()

    def load_level_menu(self):
        logger.debug("Load level scene")
        message = [0, 'loadscene', {'scene': 'LevelSelectMenu'}]
        self.send(message)
        self.wait_for_response()
        # response = None
        # while True:
        #     if response is not None and 'data' in response[1] and response[1]['data'] == "True":
        #         break
        #
        #     logger.debug(f"In loop: {response}")
        #     logger.debug("Level Menu is loaded")
        #     message = [0, 'levelsloaded']
        #     self.send(message)
        #     response = self.wait_for_response()
        #     time.sleep(0.2)

    def wait_till_all_level_played(self):
        logger.debug("Wait for level played")
        request_try = 0
        while True:
            logger.debug("Wait for level played")
            message = [0, 'alllevelsplayed']
            self.send(message)
            response = self.wait_for_response()

            if response is not None and 'data' in response[1] and response[1]['data'] == "True":
                logger.debug("All level Played")
                break
            else:
                if request_try >= self.wait_for_all_levels_played_tries:
                    break
                time.sleep(2)
                request_try += 1
        return response[1]['data']

    def change_level(self, index = 0, wait_for_stable = True, stopTime = False):
        if stopTime and wait_for_stable:
            raise Exception("Dont stop time and wait for stable at the same time")
        message = [0, 'selectlevel', {'levelIndex': index, 'waitForStable': wait_for_stable, 'stopTime': stopTime}]
        self.send(message)
        return self.wait_for_response()

    def simulate_all_levels(self, start_idx = 0, end_idx = False, wait_for_stable = False, wait_for_response = True):
        message = [0, 'simulatealllevels', {'startIndex': start_idx, 'waitForStable': wait_for_stable, 'endIndex': end_idx}]
        self.send(message)
        if wait_for_response:
            return self.wait_for_response()
        else:
            return None

    def go_to_menu(self):
        message = [0, 'loadscene', {'scene': 'MainMenu'}]
        self.send(message)
        return self.wait_for_response()

    def stop_game(self):
        os_name = platform.system()
        if os_name == 'Windows':
            if self.game_process:
                self.game_process.terminate()
        elif os_name == 'Darwin':
            os.system("pkill ScienceBirds")

    def get_data(self):
        message = [0, 'getdata']
        self.send(message)
        received_data = self.wait_for_response()
        parsed = json.loads(received_data[1]['data'])
        logger.debug(json.dumps(parsed, indent = 4, sort_keys = True))

        return received_data

    def get_img_data(self, structure = True):
        message = [0, 'screenshotstructure' if structure else 'screenshot']
        self.send(message)
        received_data = self.wait_for_response()
        return received_data[1]['data']

    def create_level_img(self, structure = True):
        img_data = self.get_img_data(structure = structure)
        i = base64.b64decode(img_data.strip('data:image/png;base64'))
        i = io.BytesIO(i)
        return mpimg.imread(i, format = 'png')
