import logging
import sys


FORMAT = "%(levelname)s: %(asctime)s|[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"


class Logger:
    @staticmethod
    def getLogger(name, logging_level=logging.DEBUG):
        return Logger._setup_logger(name, logginglevel=logging_level)

    @staticmethod
    def _setup_logger(filename, logginglevel=logging.INFO):
        logging.basicConfig(
            format=FORMAT, datefmt="%Y-%m-%d %H:%M:%S %Z", stream=sys.stdout
        )

        logger = logging.getLogger(filename)
        logger.setLevel(logginglevel)

        return logger
