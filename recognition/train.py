import torch
import logger
import argparse
from dataset import get_dataloader
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


def train(arg):
    cfg = load_cfg(arg)

    log_path = cfg["logging"]["path"]
    logger = setup_train_logging(log_path)

    device = cfg["device"]

    test_metrics = cfg['metric']["lfw-pair accuracy"]

    epochs = cfg["epochs"]

    train_loader = get_dataloader(root_dir=cfg["data_path"],
                                  batch_size=cfg["batch_size"],
                                  )
    metric = Evaluate(test_metrics["pairs_dir"],
                      test_metrics["tar@far_data_dir"],
                      test_metrics["fpr"])

    if cfg["loss"].get("arcface", False):
        loss = ArcFaceLoss(cfg["loss"]["arcface"]["num_classes"],
                           cfg["embedding_size"],
                           scale=cfg["loss"]["arcface"]["scale"]).to(device)
    else:
        raise

    if cfg["backbone"][:7] == "iresnet":
        backbone = get_iresnet(cfg["backbone"]).to(device)
    elif cfg["backbone"] == "ghostfacenetv2":
        backbone = GhostFaceNetsV2(image_size=112,
                                   num_classes=None, dropout=0.).to(device)
    else:
        raise Exception(f"Unknown backbone: {cfg['backbone']}")

    if cfg["optimizer"].get("sgd", False):
        optim_cfg = cfg["optimizer"]["sgd"]
        optimizer = optim.SGD([
            {'params': backbone.parameters()},
            {'params': loss.parameters(), "weight_deacy": 0.}],
            lr=optim_cfg["lr"],
            momentum=optim_cfg["momentum"],
            weight_decay=optim_cfg["weight_decay"])

        scheduler_cfg = cfg["optimizer"]["sgd"]["scheduler"]
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             scheduler_cfg["reduce_epochs"],
                                             scheduler_cfg["gamma"])
    else:
        raise Exception("Train with other optimizers not implemented yeat")

    logger.info(f"Start training with current configuration")

    iter_counter = 0

    show_lr = False
    if cfg["logging"]["show_lr"]:
        show_lr = True

    if cfg["callbacks"].get("earlystop", False):
        early_stop_cfg = cfg["callbacks"]["earlystop"]
        early_stop = EarlyStop(path=early_stop_cfg["chkpt_path"],
                               patience=early_stop_cfg["patience"],
                               model=backbone,
                               arcface=loss,
                               optim=optimizer,
                               log_path=log_path)
    else:
        early_stop = False

    accuracy = None
    end_training = False

    for epoch in range(epochs):

        if end_training:
            break

        backbone.train()
        epoch_train_loss = 0

        for images, labels in train_loader:
            iter_counter += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            embeddings = backbone(images)

            train_loss = loss(embeddings, labels)
            epoch_train_loss += train_loss.detach()
            train_loss.backward()

            optimizer.step()
            logger.info(f"Iter: {iter_counter}/{epochs * len(train_loader)} | train_loss: {train_loss} \
             {f'| current LR: {scheduler.get_last_lr()}' if show_lr else 0} | last_accuracy: {accuracy}")

        scheduler.step()

        with torch.inference_mode():
            backbone.eval()
            epoch_train_loss = np.round(epoch_train_loss.cpu() / len(train_loader), decimals=5)

            accuracy, thresh = metric.accuracy(model=backbone,
                                               metrics=["accuracy"],
                                               fpr=test_metrics["fpr"])
            accuracy = np.round(accuracy["accuracy"], 4)

            logger.info(f"Epoch {epoch} / {epochs} results: train_loss: {epoch_train_loss} \
            | {f'| current LR: {scheduler.get_last_lr()}' if show_lr else 0} | lfw-accuracy: {accuracy} | threshold: {np.round(thresh, 5)}")

            if early_stop:
                early_stop(accuracy, epoch)
                end_training = early_stop.early_stopping

    logger.info(f"Training ends on Epoch: {epoch}/{epochs}.")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Arcface Training")
    parser.add_argument("config",
                        type=str,
                        help="path to your .json cfg")
    train(parser.parse_args())
