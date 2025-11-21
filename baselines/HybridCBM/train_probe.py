import argparse

import torch.cuda

from datasets import get_datamodule_fromconfig
from models.linearProbe import LogisticRegressionSearch
from utils.utils import config_logging


def train():
    from mmengine.config import DictAction
    from utils.config import Config

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/LProbe/CUB.py', help='path to config file')
    parser.add_argument('--test',
                        action='store_true',
                        default=False,
                        help='whether to enable test mode')
    parser.add_argument('--device',
                        type=lambda x: int(x) if x.isdigit() else x,
                        default='cpu',
                        help='which gpu to use'
                        )
    parser.add_argument('--cfg-options',
                        nargs='+',
                        action=DictAction,
                        help='overwrite parameters in cfg from commandline')
    args = parser.parse_args()
    config = Config.from_args(args, generate_config=False)
    datamodule = get_datamodule_fromconfig(config)
    datamodule.setup(None)
    train_data = datamodule.get_train_data(numpy=True)
    val_data = datamodule.get_val_data(numpy=True)
    test_data = datamodule.get_test_data(numpy=True)
    logger = config_logging(log_file=f'{config.exp_root}/train_{config.n_shots}shot.log')
    logger.info('-' * 50)
    logger.info(f"{config.dataset} {config.n_shots}shot")
    torch.cuda.empty_cache()
    model = LogisticRegressionSearch(n_runs=config.n_runs, steps=config.steps,
                                     device='gpu' if config.dataset == 'ImageNet' and config.n_shots == 'all' else 'cpu')
    model.search(train_data, val_data, test_data)


if __name__ == "__main__":
    train()
