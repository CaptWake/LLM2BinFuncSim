from __future__ import annotations

from enum import Enum
import logging
import sys
from typing import Optional, TextIO

_logger: Optional[SimpleLogger] = None


class colors(Enum):
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"

    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    LIGHT_YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    WHITE = "\033[1;37m"
    RESET = "\033[0;39m"


class SimpleLogger:
    """
    We implement a simple logger to not rely on additional python packages,
    e.g., rich. This is written in way that, by default, log messages (e.g.,
    debug/info/...) are sent to stderr.
    """

    def __init__(self, use_colors: bool = False):
        self.level = logging.WARNING
        self.use_colors: bool = use_colors

    def setLevel(self, level: int) -> None:
        self.level: int = level

    def raw_print_to_stdout(self, msg: str) -> None:
        self.raw_print(msg, file=sys.stdout)

    def raw_print(self, msg: str, file: TextIO = sys.stdout) -> None:
        print(msg, file=file)

    def debug(self, msg: str) -> None:
        if logging.DEBUG >= self.level:
            if self.use_colors:
                self.raw_print(f"{colors.GREEN}DEBUG: {msg}{colors.RESET}")
            else:
                self.raw_print(f"DEBUG: {msg}")

    def info(self, msg: str) -> None:
        if logging.INFO >= self.level:
            self.raw_print(f"INFO: {msg}")

    def warning(self, msg: str) -> None:
        if logging.WARNING >= self.level:
            if self.use_colors:
                self.raw_print(f"{colors.YELLOW}WARNING: {msg}{colors.RESET}")
            else:
                self.raw_print(f"WARNING: {msg}")

    def error(self, msg: str) -> None:
        if logging.ERROR >= self.level:
            if self.use_colors:
                self.raw_print(f"{colors.RED}ERROR: {msg}{colors.RESET}")
            else:
                self.raw_print(f"ERROR: {msg}")


def get_logger(use_colors: bool = False) -> SimpleLogger:
    global _logger

    if _logger is None:
        _logger = SimpleLogger(use_colors=use_colors)

    return _logger
