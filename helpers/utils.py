from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
from helpers.config import Config


# Torch convention
def get_score(skfunction, y_pred, y_true):
    score = skfunction(y_true, y_pred)
    return score


# LOGGER = init_logger()
def init_logger(log_file=Config.OUTPUT_DIR+'train.log'):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger
