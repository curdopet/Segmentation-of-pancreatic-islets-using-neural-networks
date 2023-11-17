from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum


class LogColors:
    """Colors: https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797"""
    HEADER = '\033[95m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ERROR = '\033[38;5;196m'
    WARNING = '\033[38;5;214m'
    INFO = '\033[38;5;253m'
    SUCCESS = '\033[38;5;40m'
    DEBUG = '\033[38;5;45m'


class Severity(IntEnum):
    DEBUG = 1
    SUCCESS = 2
    INFO = 3
    WARNING = 4
    ERROR = 5


@dataclass
class PerImageLog:
    current_image_num: int = 0
    total_images_cnt: int = 0
    image_name: str = ""


class Log:
    def __init__(self, total_images_cnt: int = 0):
        self.per_image_log_info = PerImageLog(
            total_images_cnt=total_images_cnt,
        )

    @staticmethod
    def log(severity: Severity, message: str):
        extended_mesage = "{} | {} \t| {}".format(datetime.now(), severity.name, message)

        if severity == Severity.ERROR:
            print(LogColors.ERROR + extended_mesage + LogColors.ENDC)
        elif severity == Severity.WARNING:
            print(LogColors.WARNING + extended_mesage + LogColors.ENDC)
        elif severity == Severity.INFO:
            print(LogColors.INFO + extended_mesage + LogColors.ENDC)
        elif severity == Severity.SUCCESS:
            print(LogColors.SUCCESS + extended_mesage + LogColors.ENDC)
        elif severity == Severity.DEBUG:
            print(LogColors.DEBUG + extended_mesage + LogColors.ENDC)

    def per_image_log(self, severity: Severity, message: str):
        extended_message = ""

        if self.per_image_log_info.total_images_cnt != 0:
            extended_message += "Image: [{}/{}]  {}\t | ".format(
                self.per_image_log_info.current_image_num,
                self.per_image_log_info.total_images_cnt,
                self.per_image_log_info.image_name,
            )
        extended_message += message

        self.log(severity, extended_message)

    def update_per_image_log(self, image_name: str):
        self.per_image_log_info.current_image_num += 1
        self.per_image_log_info.image_name = image_name

    def reset_image_log(self):
        self.per_image_log_info.current_image_num = 0
        self.per_image_log_info.image_name = ""
