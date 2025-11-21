import os
from utils.config import Config
from utils.train_helper import TrainHelper
from models.cbms import LinearCBM

if __name__ == "__main__":
    config = Config.config()
    config.exp_root = config.exp_root.replace("HybridCBM", f"HybridCBM_{config.seed}")
    trainner = TrainHelper(config=config, Model=LinearCBM, seed=config.seed)
    trainner.run()
