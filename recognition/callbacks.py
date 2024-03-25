import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler(f"train_logs.log", mode='w')
formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

handler.setFormatter(formatter)
logger.addHandler(handler)


class EarlyStop:
    def __init__(self, path, patience=7):

        self.patience = patience
        self.path = path
        self.best_score = 0
        self.counter = 0
        self.early_stop = False

    def __call__(self, accuracy_score, epoch):
        if accuracy_score > self.best_score:
            logger.info(f"EarlyStop: lfw-pairs Accuracy increased {self.best_score} -> {accuracy_score}")
            self.best_score = accuracy_score
            self.counter = 0
        else:
            if self.counter > self.patience:
                self.early_stop = True
                logger.info(
                    f"EarlyStop: counter out of patience: {self.counter}/{self.patience}. Early stop training on epoch: {epoch}")
            self.counter += 1
