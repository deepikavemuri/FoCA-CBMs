import argparse
import torch
import os
import pandas as pd
from datasets import get_datamodule_fromconfig
from models.conceptBank import get_concept_bank_fromconfig
from utils.train_helper import TrainHelper as _TrainHelper
from mmengine.config import DictAction
from utils.config import Config
from models.cbms import CaptionCBM


class TrainHelper(_TrainHelper):
    def __init__(self, config, Model):
        super(TrainHelper, self).__init__(config, Model)
        self.captions = self.generate_captions()

    def generate_captions(self):
        caption = os.path.join(self.config.exp_root, "captions.csv")
        if not os.path.exists(caption):
            concept_bank = get_concept_bank_fromconfig(self.config)
            concept_bank.to(torch.device(self.config.device))
            datamodule = get_datamodule_fromconfig(
                self.config, clip_encoder=concept_bank.clip_encoder
            )
            image_features, labels = datamodule.get_train_data()
            captions = {
                "caption": concept_bank.translate(
                    features=image_features, batch_size=128
                ),
                "label": labels.tolist(),
            }
            captions = pd.DataFrame(captions)
            captions.to_csv(caption, index=False)
        else:
            captions = pd.read_csv(caption)

        select_cap = os.path.join(self.config.exp_root, "selected_caption.csv")
        if not os.path.exists(select_cap):
            per_class_num_features = (
                self.config.num_dynamic_concept // self.config.num_class
            )
            select_caps = captions.groupby("label").apply(
                lambda x: x.sample(n=per_class_num_features)
            )
            # using image caption to initialize dynamic concept bank
            select_caps.to_csv(select_cap, index=False)
        else:
            select_caps = pd.read_csv(select_cap)
        return select_caps["caption"].tolist()


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/HybridCBM/CUB/CUB_allshot.py",
        help="path to config file",
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="whether to enable test mode"
    )
    parser.add_argument("--device", type=int, default=7, help="which gpu to use")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="overwrite parameters in cfg from commandline",
    )
    parser.add_argument(
        "--exp_root",
        type=str,
        default="exp/HybridCBM/CUB/CUB_allshot",
        help="experiment root",
    )
    args = parser.parse_args()
    config = Config.from_args(args)
    config.use_normalize = True
    config.exp_root = config.exp_root.replace("exp/HybridCBM", "exp/CaptionCBM")
    config.lambda_discri = 0
    config.lambda_ort = 0
    config.lambda_align = 0
    return config


if __name__ == "__main__":
    config = get_config()
    trainner = TrainHelper(config, CaptionCBM)
    if not config.test:
        Config.save_config(config)
        trainner.train()
    trainner.test()
