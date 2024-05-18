import os
import os.path as osp
import torch
from logger import setup_callbacks_logging


class EarlyStop:
    def __init__(self,
                 model,
                 arcface,
                 optim,
                 path,
                 log_path,
                 patience=7):

        self.use_params = {"model.pth": model,
                           "arcface.pth": arcface,
                           "optimizer.pth": optim}
        self.patience = patience
        self.path = path
        self.best_score = 0
        self.counter = 0
        self.early_stopping = False
        self.logger = setup_callbacks_logging(log_path)

    def __call__(self, accuracy_score, epoch):
        if accuracy_score > self.best_score:
            self.logger.info(f"EarlyStop: lfw-pairs Accuracy increased {self.best_score} -> {accuracy_score}")
            self.best_score = accuracy_score
            self.counter = 0
            self.save_checkpoint(epoch)
        else:
            if self.counter > self.patience:
                self.early_stopping = True
                self.logger.info(
                    f"EarlyStop: counter out of patience: {self.counter}/{self.patience}. Early stop training on epoch: {epoch}")
            self.counter += 1

    def current_params(self):
        params = [(key, value.state_dict()) for key, value in self.use_params.items()]
        return params

    def save_checkpoint(self, epoch):
        checkpoint_name = f"checkpoint_epch{epoch}"
        checkpoint_path = osp.join(self.path, checkpoint_name)
        print(checkpoint_path)
        os.makedirs(checkpoint_path, exist_ok=True)
        for key, value in self.current_params():
            param_path = osp.join(checkpoint_path, key)
            torch.save(value, param_path)
            self.logger.info(f"EarlyStop: checkpoint saved - {checkpoint_name}")

