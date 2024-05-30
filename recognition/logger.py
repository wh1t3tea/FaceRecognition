import logging



def setup_train_logging(log_file):
    """
    Set up logging for training process.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Logger object for training.

    """
    logger = logging.getLogger("Trainer:")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_file, mode='a')
    formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def setup_callbacks_logging(log_file):
    """
    Set up logging for callbacks.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Logger object for callbacks.

    """
    logger = logging.getLogger("Callback:")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_file, mode="a")
    formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logging
