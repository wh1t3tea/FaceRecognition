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


def train(arg):
    cfg = arg.config
    device = cfg["device"]
    test_metrics = cfg['metric']["lfw-pair accuracy"]

    train_loader = get_dataloader(root_dir=cfg["data_path"],
                                  batch_size=cfg["batch_size"],
                                  )
    metric = Evaluate(test_metrics["pair_dir"],
                      test_metrcis["tar@far_data_dir"],
                      test_metrics["fpr"])
    if cfg["loss"] == "arcface":
        loss = ArcFaceLoss(cfg["num_classes"],
                           cfg["embedding_size"],
                           scale=cfg["loss"]["arcface"]["scale"]).to(device)
    else:
        raise

    if cfg["backbone"][:7] == "iresnet":
        backbone = get_iresnet(cfg["backbone"]).to(device)
    elif backbone == "ghostfacenetv2":
        backbone = GhostFaceNetsV2(image_size=112,
                                   num_classes=None).to(device)

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
    pass


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
