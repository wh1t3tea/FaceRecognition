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
    cfg = load_cfg(arg)

    log_path = cfg["logging"]["path"]
    logger = setup_train_logging(log_path)

    device = cfg["device"]

    test_metrics = cfg['metric']["lfw-pair accuracy"]

    epochs = cfg["epochs"]

    train_loader = get_mx_dataloader(root_dir=cfg["data_path"],
                                  batch_size=cfg["batch_size"])
    metric = Evaluate(test_metrics["pairs_dir"],
                      test_metrics["tar@far_data_dir"],
                      test_metrics["fpr"])

    if cfg["loss"].get("arcface", False):
        loss_cfg = cfg["loss"]["arcface"]
        loss_name = "arcface"
        #loss = ArcFaceLoss(cfg["loss"]["arcface"]["num_classes"],
        #                  cfg["embedding_size"],
        #                   scale=cfg["loss"]["arcface"]["scale"]).to(device)
        arc_margin = ArcMarginProduct(in_features=cfg["embedding_size"],
                                      out_features=loss_cfg["num_classes"],
                                      s=loss_cfg["scale"], m=0.5).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception(f"Unexpected loss: {cfg['loss']}")

    if cfg["backbone"][:7] == "iresnet":
        backbone = get_iresnet(cfg["backbone"]).to(device)
    elif cfg["backbone"] == "ghostfacenetv2":
        backbone = GhostFaceNetsV2(image_size=112,
                                   num_classes=None, dropout=0.).to(device)
    else:
        raise Exception(f"Unknown backbone: {cfg['backbone']}")

    trainable_entities = [backbone, arc_margin]

    optimizer, scheduler = get_optimizer(trainable_entities, cfg["optimizer"])

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
                               arcface=arc_margin,
                               optim=optimizer,
                               log_path=log_path)
    else:
        early_stop = False

    accuracy = None
    end_training = False
    last_lr = 0.1
    for epoch in range(epochs):
        if epoch == 1:
            arc_margin.s = 64.0

        if end_training:
            break

        epoch_train_loss = 0

        loss_accumulation = 0
        for images, labels in train_loader:
            iter_counter += 1
            backbone.train()
            if iter_counter % 40 == 0:
                for param in optimizer.param_groups:
                    param["lr"] = param["lr"] - 0.00005
                    last_lr = param["lr"]

            images, labels = images.to(device), labels.to(device)
            embeddings = backbone(images)
            logits = arc_margin(embeddings, labels)
            train_loss = criterion(logits, labels)
            train_loss = train_loss / 4
            loss_accumulation += train_loss
            train_loss.backward()
            if iter_counter % 4 == 0:
                if epoch_train_loss != 0:
                    epoch_train_loss = epoch_train_loss.detach().cpu()
                epoch_train_loss += loss_accumulation.detach().cpu()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                optimizer.step()
                loss_accumulation = 0
                optimizer.zero_grad()
            if iter_counter % 40 == 0:
                logger.info(
                    f"Iter: {iter_counter // 4}/{epochs * len(train_loader) // 4} | train_loss: {'{:.8f}'.format(epoch_train_loss / iter_counter * 4)} {f'| current LR: {last_lr}' if show_lr else 0} | last_accuracy: {accuracy}")

            if iter_counter % 2000 == 0:
                with torch.inference_mode():
                    backbone.eval()
                    epoch_train_loss = np.round(epoch_train_loss.detach().cpu(), decimals=5)

                    accuracy, thresh = metric.accuracy(model=backbone,
                                                       metrics=["accuracy"],
                                                       fpr=test_metrics["fpr"])
                    accuracy = np.round(accuracy["accuracy"], 4)

                    logger.info(
                        f"Iter {iter_counter // 4} / {epochs * len(train_loader) // 4} results: train_loss: {'{:.8f}'.format(epoch_train_loss / iter_counter * 4)} | {f'| current LR: {last_lr}' if show_lr else 0} | lfw-accuracy: {accuracy} | threshold: {np.round(thresh, 5)}")

        scheduler.step()

        with torch.inference_mode():
            backbone.eval()
            epoch_train_loss = np.round(epoch_train_loss.detach().cpu() / iter_counter, decimals=5)

            accuracy, thresh = metric.accuracy(model=backbone,
                                               metrics=["accuracy"],
                                               fpr=test_metrics["fpr"])
            accuracy = np.round(accuracy["accuracy"], 4)

            logger.info(f"Epoch {epoch} / {epochs} results: train_loss: {'{:.8f}'.format(epoch_train_loss * 4)} | {f'| current LR: {current_lr}' if show_lr else 0} | lfw-accuracy: {accuracy} | threshold: {np.round(thresh, 5)}")

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
