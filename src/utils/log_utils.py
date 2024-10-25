import logging
from datetime import datetime


class LogUtils:
    def __init__(self, log_level: str):
        self.formatted_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        logging.basicConfig(level=log_level)

        log_formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)-10s - [%(filename)s:%(lineno)d]: %(message)s"
        )

        file_handler = logging.FileHandler(
            f"logs/{self.formatted_start_time}_game_logs.log"
        )
        file_handler.setFormatter(log_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
