import logging
from datetime import datetime


class LogUtils:
    def __init__(self, verbose: bool = False):
        self.formatted_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        log_formatter = logging.Formatter(
            '%(asctime)s %(levelname)-8s %(name)-10s - [%(filename)s:%(lineno)d]: %(message)s')

        file_handler = logging.FileHandler(f"logs/{self.formatted_start_time}_game_logs.log")
        file_handler.setFormatter(log_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)



