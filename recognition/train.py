import torch
import logger
import argparse
from dataset import get_dataloader, get_mx_dataloader
from eval.lfw_benchmark import Evaluate
from pytorch_metric_learning.losses import ArcFaceLoss
from backbones.iresnet import get_iresnet
from torch import optim
from torch.optim import lr_scheduler
from ellzaf_ml.models import GhostFaceNetsV2
from cfg.cfg import load_cfg
from callbacks import EarlyStop
import numpy as np
from logger import setup_train_logging
from optimizer import get_optimizer
from loss.arcface import ArcMarginProduct
from torch import nn


def train(arg):
    # Load configuration file
    cfg = load_cfg(arg)

    # Setup logging
    log_path = cfg["logging"]["path"]
    logger = setup_train_logging(log_path)

    # Device configuration (CPU/GPU)
    device = cfg["device"]

    # Load test metrics configuration
    test_metrics = cfg['metric']["lfw-pair accuracy"]

    # Number of epochs to train
    epochs = cfg["epochs"]

    # Prepare dataloader for training data
    train_loader = get_dataloader(root_dir=cfg["data_path"], batch_size=cfg["batch_size"])

    # Initialize evaluation metric for LFW benchmark
    metric = Evaluate(test_metrics["pairs_dir"],
                      test_metrics["tar@far_data_dir"],
                      test_metrics["fpr"])

    # Loss function configuration
    if cfg["loss"].get("arcface", False):
        loss_cfg = cfg["loss"]["arcface"]
        loss_name = "arcface"
        arc_margin = ArcMarginProduct(in_features=cfg["embedding_size"],
                                      out_features=loss_cfg["num_classes"],
                                      s=loss_cfg["scale"], m=0.5).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception(f"Unexpected loss: {cfg['loss']}")

    # Backbone model configuration
    if cfg["backbone"][:7] == "iresnet":
        backbone = get_iresnet(cfg["backbone"]).to(device)
    elif cfg["backbone"] == "ghostfacenetv2":
        backbone = GhostFaceNetsV2(image_size=112,
                                   num_classes=None, dropout=0.).to(device)
    else:
        raise Exception(f"Unknown backbone: {cfg['backbone']}")

    # List of trainable entities
    trainable_entities = [backbone, arc_margin]

    # Initialize optimizer and scheduler
    optimizer, scheduler = get_optimizer(trainable_entities, cfg["optimizer"])

    logger.info("Start training with current configuration")

    iter_counter = 0
    show_lr = cfg["logging"].get("show_lr", False)

    # Early stopping configuration
    if cfg["callbacks"].get("earlystop", False):
        early_stop_cfg = cfg["callbacks"]["earlystop"]
        early_stop = EarlyStop(path=early_stop_cfg["chkpt_path"],
                               patience=early_stop_cfg["patience"],
                               model=backbone,
                               arcface=arc_margin,
                               optim=optimizer,
                               log_path=log_path)
    else:
        early_stop = False

    accuracy = None
    end_training = False
    iters_to_log = cfg["log_iters"]

    # Gradient accumulation configuration
    if cfg["gradient_accumulation_steps"] > 1:
        grad_accumulation = True
        loss_accumulation = 0
        grad_accum_iters = cfg["gradient_accumulation_steps"]
    else:
        grad_accumulation = False
        grad_accum_iters = 1

    trainable_lst = [cfg["backbone"], "arcface"]

    for epoch in range(epochs):
        if end_training:
            break

        backbone.train()
        epoch_train_loss = 0

        current_lr = scheduler.get_last_lr()

        for images, labels in train_loader:
            iter_counter += 1
            images, labels = images.to(device), labels.to(device)
            embeddings = backbone(images)
            logits = arc_margin(embeddings, labels)
            if grad_accumulation:
                train_loss = criterion(logits, labels)
                train_loss.backward()

                epoch_train_loss += train_loss.detach()
                loss_accumulation += train_loss.detach()

                if iter_counter % grad_accum_iters == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                train_loss = criterion(logits, labels)
                train_loss.backward()
                epoch_train_loss += train_loss.detach()
                optimizer.step()

            if iter_counter % iters_to_log == 0:
                logger.info(f"Iter: {iter_counter}/{epochs * len(train_loader)} | train_loss: {train_loss}" +
                            f" | current LR: {current_lr if show_lr else 0} | last_accuracy: {accuracy}")

            elif grad_accumulation and iter_counter % grad_accum_iters == 0:
                logger.info(
                    f"Iter: {iter_counter}/{epochs * len(train_loader)} | train_loss: {loss_accumulation.mean()}" +
                    f" | current LR: {current_lr if show_lr else 0} | last_accuracy: {accuracy}")
                loss_accumulation = 0

        scheduler.step()

        with torch.inference_mode():
            backbone.eval()
            epoch_train_loss = np.round(epoch_train_loss.cpu() / len(train_loader), decimals=5)

            accuracy, thresh = metric.accuracy(model=backbone,
                                               metrics=["accuracy"],
                                               fpr=test_metrics["fpr"])
            accuracy = np.round(accuracy["accuracy"], 4)
            logger.info(f"Epoch {epoch} / {epochs} results: train_loss: {np.round(epoch_train_loss, 8)}" +
                        f" | current LR: {current_lr if show_lr else 0} | lfw-accuracy: {accuracy} | threshold: {np.round(thresh, 5)}")

            if early_stop:
                early_stop(accuracy, epoch)
                end_training = early_stop.early_stopping

    logger.info(f"Training ends on Epoch: {epoch}/{epochs}.")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Arcface Training")
    parser.add_argument("config", type=str, help="path to your .json cfg")
    train(parser.parse_args())
