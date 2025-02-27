import os
import os.path as osp
import torch
from logger import setup_callbacks_logging


class EarlyStop:
    """
    Implements early stopping during training based on validation accuracy.

    Args:
        model (torch.nn.Module): The model being trained.
        arcface (torch.nn.Module): ArcFace module.
        optim (torch.optim.Optimizer): The optimizer used for training.
        path (str): Directory where checkpoints will be saved.
        log_path (str): Path to the log file.
        patience (int, optional): Number of epochs to wait for improvement before stopping. Defaults to 7.
    """
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
        """
        Check whether to perform early stopping based on the current accuracy score.

        Args:
            accuracy_score (float): Current accuracy score.
            epoch (int): Current epoch number.
        """
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
        """
        Get the current parameters of the model, arcface, and optimizer.

        Returns:
            list: List of tuples containing the parameter names and their corresponding state dictionaries.
        """
        params = [(key, value.state_dict()) for key, value in self.use_params.items()]
        return params

    def save_checkpoint(self, epoch):
        """
        Save a checkpoint of the model, arcface, and optimizer parameters.

        Args:
            epoch (int): Current epoch number.
        """
        checkpoint_name = f"checkpoint_epch{epoch}"
        checkpoint_path = osp.join(self.path, checkpoint_name)
        print(checkpoint_path)
        os.makedirs(checkpoint_path, exist_ok=True)
        for key, value in self.current_params():
            param_path = osp.join(checkpoint_path, key)
            torch.save(value, param_path)
            self.logger.info(f"EarlyStop: checkpoint saved - {checkpoint_name}")

