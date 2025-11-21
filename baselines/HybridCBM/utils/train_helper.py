import glob
import os
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from datasets import get_datamodule_fromconfig
from models.conceptBank import get_concept_bank_fromconfig
from .utils import config_logging
from .config import Config


def init_data_bank(config, captions):
    concept_bank = get_concept_bank_fromconfig(config)
    concept_bank.to(torch.device(config.device))
    datamodule = get_datamodule_fromconfig(config, clip_encoder=concept_bank.clip_encoder)
    concept_bank.initialize(img_features=datamodule.img_features['train'],
                            num_images_per_class=datamodule.num_images_per_class,
                            captions=captions)
    concept_bank.to(torch.device(config.device))
    return concept_bank, datamodule


class TrainHelper:
    def __init__(self, config=None, Model=None, runner=None, seed=42):
        self.config = config if config is not None else Config.config()
        self.Model = Model
        self.seed = seed
        self.runner = runner
        self.captions = None

    def init_data_bank(self):
        concept_bank = get_concept_bank_fromconfig(self.config)
        try:
            concept_bank.to(torch.device(self.config.device))
        except RuntimeError:
            print(f"CUDA {self.config.device} ERROR")
        datamodule = get_datamodule_fromconfig(self.config, clip_encoder=concept_bank.clip_encoder)
        concept_bank.initialize(img_features=datamodule.img_features['train'],
                                num_images_per_class=datamodule.num_images_per_class,
                                captions=self.captions)
        concept_bank.to(torch.device(self.config.device))
        return concept_bank, datamodule

    def load_checkpoint(self, use_last_ckpt=False):
        checkpoint_dir = os.path.join(self.config.exp_root, 'checkpoints')
        if ('use_last_ckpt' in self.config and self.config['use_last_ckpt'] is not None) or use_last_ckpt:
            checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
            if not checkpoints:
                print(f"No checkpoints found in {checkpoint_dir}")
                return None
            checkpoint_path = max(checkpoints, key=os.path.getctime)
        else:
            checkpoints = glob.glob(os.path.join(checkpoint_dir, '*val_acc*.ckpt'))
            if not checkpoints:
                print(f"No checkpoints found in {checkpoint_dir}")
                return None

            def get_val_acc(ckpt):
                filename = os.path.basename(ckpt)
                val_acc = filename.split('val_acc=')[-1].split('.ckpt')[0].split('-')[0]
                return float(val_acc)

            checkpoint_path = max(checkpoints, key=get_val_acc)
        return checkpoint_path

    def test(self, log_file='test.log'):
        seed_everything(self.seed)  # seed matches first run of linear probe
        # 初始化数据模块和概念库
        concept_bank, datamodule = self.init_data_bank()

        logger = config_logging(log_file=f'{self.config.exp_root}/{log_file}')
        checkpoint_path = self.load_checkpoint()

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        # 加载模型
        model = self.Model.load_from_checkpoint(checkpoint_path,
                                                map_location=torch.device(self.config.device),
                                                config=self.config,
                                                conceptbank=concept_bank,
                                                strict=False)
        model.to(self.config.device)
        model.eval()
        logger.info(f'conceptBank requires_grad: {model.conceptbank.dynamic_bank.concept_features.requires_grad}')
        # 创建Trainer
        trainer = Trainer(devices=[self.config.device],
                          callbacks=[RichProgressBar()],
                          logger=TensorBoardLogger(
                              name=f'{self.config.dataset}_{self.config.n_shots}shot_{self.config.num_dynamic_concept}_test',
                              save_dir=self.config.exp_root),
                          default_root_dir=self.config.exp_root)
        trainer.test(model, datamodule=datamodule)
        return

    def train(self):
        proj_name = self.config.dataset
        # concept seletion method
        seed_everything(self.seed)  # seed matches first run of linear probe
        concept_bank, datamodule = self.init_data_bank()
        logger = config_logging(log_file=f'{self.config.exp_root}/train.log')
        logger.info("use asso concept with dot product loader, faster")
        model = self.Model(self.config, conceptbank=concept_bank)
        if self.config.dataset == "ImageNet" and self.config.n_shots == "all":
            check_interval = 5
        else:
            check_interval = 10

        logger.info(f"check interval = {check_interval}")
        checkpoint_dir = os.path.join(self.config.exp_root, 'checkpoints')
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{epoch}-{step}-{val_acc:.4f}',
            monitor='val_acc',
            mode='max',
            save_top_k=3,
            every_n_epochs=check_interval)

        trainer = Trainer(devices=[self.config.device],
                          callbacks=[checkpoint_callback, RichProgressBar(),
                                     LearningRateMonitor(logging_interval='epoch')],
                          logger=TensorBoardLogger(
                              name=f'{proj_name}_{self.config.n_shots}shot_{self.config.num_dynamic_concept}_train',
                              save_dir=self.config.exp_root),
                          check_val_every_n_epoch=check_interval,
                          default_root_dir=self.config.exp_root,
                          max_epochs=self.config.max_epochs,
                          )

        trainer.fit(model, datamodule=datamodule, ckpt_path=self.load_checkpoint(use_last_ckpt=True))

    def run(self):
        if self.config.multi and self.runner is not None:
            self.runner.search_parameters()
            self.runner.run()
            exit(0)
        else:
            if not self.config.test and not os.path.exists(os.path.join(self.config.exp_root, 'test.log')):
                Config.save_config(self.config, force=True)
                self.train()
                self.test()
            elif os.path.exists(self.config.exp_root) and self.config.test:
                try:
                    self.config = Config.load_config(self.config.exp_root)
                except FileNotFoundError:
                    print(f"config file not found in {self.config.exp_root}, Generate a new one")
                    Config.save_config(self.config, force=False)
                self.test()
            elif not os.path.exists(os.path.join(self.config.exp_root, 'clip_metrics.csv')):
                self.test()
