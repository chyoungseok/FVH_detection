import argparse
import os
import yaml

import torch
from torch.utils.data import DataLoader

from datasets.flair2d_binary import Flair2DBinaryDataset
from models.plainnet import PlainNet
from engine.train_loop import evaluate
from utils.common import load_checkpoint


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dcfg = cfg["data"]
    split = dcfg.get("test_split", None) if dcfg.get("split_col", None) else None

    ds = Flair2DBinaryDataset(
        csv_path=dcfg["csv_path"],
        root_dir=dcfg.get("root_dir", ""),
        split=split,
        image_col=dcfg.get("image_col", "file_path"),
        label_col=dcfg.get("label_col", "label"),
        split_col=dcfg.get("split_col", "split"),
        image_size=tuple(dcfg.get("image_size", [224, 224])),
        is_train=False,
    )
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=dcfg.get("num_workers", 4), pin_memory=True)

    mcfg = cfg["model"]
    model = PlainNet(
        backbone=mcfg.get("backbone", "resnet18"),
        in_channels=mcfg.get("in_channels", 1),
        pretrained=False,
    ).to(device)

    ckpt = load_checkpoint(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    metrics = evaluate(model, loader, device)
    print("[EVAL]", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    main(args)
