import torch
import logging
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


def train(arg):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(f"train_logs.log", mode='w')
    formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    cfg = load_cfg(arg)

    device = cfg["device"]
    test_metrics = cfg['metric']["lfw-pair accuracy"]

    epochs = cfg["epochs"]

    train_loader = get_dataloader(root_dir=cfg["data_path"],
                                  batch_size=cfg["batch_size"],
                                  )
    metric = Evaluate(test_metrics["pair_dir"],
                      test_metrics["tar@far_data_dir"],
                      test_metrics["fpr"])
    if cfg["loss"] == "arcface":
        loss = ArcFaceLoss(cfg["num_classes"],
                           cfg["embedding_size"],
                           scale=cfg["loss"]["arcface"]["scale"]).to(device)
    else:
        raise

    if cfg["backbone"][:7] == "iresnet":
        backbone = get_iresnet(cfg["backbone"]).to(device)
    elif cfg["backbone"] == "ghostfacenetv2":
        backbone = GhostFaceNetsV2(image_size=112,
                                   num_classes=None).to(device)
    else:
        raise Exception(f"Unknown backbone: {cfg['backbone']}")

    if cfg["optimizer"] == "sgd":
        optim_cfg = cfg["optimizer"]["sgd"]
        optimizer = optim.SGD(params=[{"params": backbone.parameters()},
                                      {"params": loss.parameters()}],
                              lr=optim_cfg["lr"],
                              momentum=optim_cfg["momentum"],
                              weight_decay=optim_cfg["weight_decay"])
    else:
        raise Exception("Train with other optimizers not implemented yeat")

    scheduler_cfg = cfg["optimizer"]["sgd_scheduler"]
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         scheduler_cfg["reduce_epochs"],
                                         scheduler_cfg["gamma"])
    logger.info(f"Start training with current configuration: {cfg}")

    iter_counter = 0

    show_lr = False
    if cfg["logging"]["lr"]:
        show_lr = True

    if cfg["callbacks"].get("earlystop", False) == "earlystop":
        earlystop_cfg = cfg["callbacks"]["earlystop"]
        earlystop = EarlyStop(path=earlystop_cfg["chkpt_path"], patience=earlystop_cfg["patience"])

    accuracy = None

    for epoch in range(epochs):

        model.train()
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
            model.eval()
            epoch_train_loss = torch.round(epoch_train_loss / len(train_loader), decimals=5)

            accuracy = metric.accuracy(model=backbone,
                                       metrics=["accuracy"])["accuracy"]
            accuracy = torch.round(accuracy, 4)

            earlystop(accuracy)

            logger.info(f"Epoch {epoch} / {epochs} results: train_loss: {epoch_train_loss} \
            | {f'| current LR: {scheduler.get_last_lr()}' if show_lr else 0} | lfw-accuracy: {accuracy}")

            if earlystop.early_stop:
                break

    logger.info(f"Training ends on Epoch: {epoch}/{epochs}.")



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Arcface Training")
    parser.add_argument("config",
                        type=str,
                        help="path to your .json cfg file")
    train(parser.parse_args())
