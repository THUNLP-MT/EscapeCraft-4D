import logging

GLOBAL_LOG_LEVEL = logging.DEBUG

# This log level is for item log
ITEM_LOG_LEVEL_NAME = "ITEM"
ITEM_LOG_LEVEL = 25

logging.addLevelName(ITEM_LOG_LEVEL, ITEM_LOG_LEVEL_NAME)


class ColoredFormatter(logging.Formatter):
    COLOR_CODES = {
        "DEBUG": "\033[94m",        # blue
        "INFO": "\033[92m",         # green
        "WARNING": "\033[93m",      # yellow
        "ERROR": "\033[91m",        # red
        "CRITICAL": "\033[91m",     # red
        "ITEM": "\033[95m",         # purple
        "RESET": "\033[0m",         # reset color
    }

    def format(self, record):
        msg = super().format(record)
        color = self.COLOR_CODES.get(record.levelname, self.COLOR_CODES["RESET"])
        return f'{color}{msg}{self.COLOR_CODES["RESET"]}'


def configure_logger(name):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(GLOBAL_LOG_LEVEL)

        ch = logging.StreamHandler()
        ch.setLevel(GLOBAL_LOG_LEVEL)

        formatter = ColoredFormatter(
            "[%(asctime)s] - file: %(name)s - level: [%(levelname)s] : %(message)s"
        )
        ch.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(ch)

    return logger


def item_log(self, message, *args, **kws):
    if self.isEnabledFor(ITEM_LOG_LEVEL):
        self._log(ITEM_LOG_LEVEL, message, args, **kws)


logging.Logger.item = item_log


def set_log_level(level):
    assert level in ["info", "error", "debug"]
    _level_dict = {
        "info": logging.INFO,
        "error": logging.ERROR,
        "debug": logging.DEBUG,
        "item": ITEM_LOG_LEVEL,
    }
    global GLOBAL_LOG_LEVEL
    GLOBAL_LOG_LEVEL = _level_dict[level]
    logging.getLogger().setLevel(GLOBAL_LOG_LEVEL)

