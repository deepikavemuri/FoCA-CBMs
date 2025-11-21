import argparse
import os

from mmengine import DictAction
from mmengine.config import Config as BaseConfig


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == "":
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


class Config(BaseConfig):
    @staticmethod
    def config():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config",
            default="config/HybridCBM/CUB/CUB_allshot.py",
            help="path to config file",
        )
        parser.add_argument(
            "--test",
            action="store_true",
            default=False,
            help="whether to enable test mode",
        )
        parser.add_argument(
            "--multi",
            action="store_true",
            default=False,
            help="whether to enable test mode",
        )
        parser.add_argument("--device", type=int, default=0, help="which gpu to use")
        parser.add_argument(
            "--cfg-options",
            nargs="+",
            action=DictAction,
            help="overwrite parameters in cfg from commandline",
        )
        parser.add_argument(
            "--exp_root",
            type=str,
            default="exp/HybridCBM/",
            help="experiment root",
        )
        args = parser.parse_args()
        config = Config.from_args(args)
        return config

    @staticmethod
    def from_args(args, generate_config=True, **kwargs):
        config = Config.fromfile(args.config)
        config.merge_from_dict(vars(args))
        for k, v in kwargs.items():
            setattr(config, k, v)
        if args.cfg_options is not None:
            print(args.cfg_options)
            for item in args.cfg_options:
                setattr(config, item, args.cfg_options[item])
        if generate_config:
            config = Config.generate_config(config)
        return config

    @staticmethod
    def load_config(exp_root):
        files = [name for name in os.listdir(exp_root) if ".py" in name]
        if len(files) == 0:
            raise FileNotFoundError(f"No config file found in {exp_root}")
        files = files[0]
        config = Config.fromfile(os.path.join(exp_root, files))
        config.exp_root = exp_root
        return config

    @staticmethod
    def save_config(config, force=False):
        path = os.path.join(config.exp_root, os.path.basename(config.config))
        if not os.path.exists(path) or force:
            mkdir_or_exist(config.exp_root)
            config.dump(file=path)

    @staticmethod
    def generate_config(config):
        translator_path = f"/baselines/HybridCBM/saved_models/translator.pt"
        data_root = f"datasets/{config.dataset}"
        if config.dataset == "awa2":
            data_root = "/DATA/Animals_with_Attributes2"
        elif config.dataset == "cifar100":
            data_root = "/DATA/cifar100_images"
        elif config.dataset == "inet100":
            data_root = "/DATA/inet100"
        num_dynamic_concept = round(
            config.num_class
            * config.num_concept_per_class
            * config.dynamic_concept_ratio
        )
        num_static_concept = (
            config.num_class * config.num_concept_per_class - num_dynamic_concept
        )
        if num_dynamic_concept > 0:
            assert (
                num_dynamic_concept >= config.num_class
            ), "Number of dynamic concepts should be larger than number of classes"
        setattr(config, "translator_path", translator_path)
        setattr(config, "data_root", data_root)
        setattr(config, "num_static_concept", num_static_concept)
        setattr(config, "num_dynamic_concept", num_dynamic_concept)

        if len(config.exp_root.split("/")) <= 3:
            exp_root = Config.config_to_root(config)
            print(f"generated exp_root: {exp_root}")
            setattr(config, "exp_root", exp_root)
        return config

    @staticmethod
    def config_to_root(config):
        exp_root = config.exp_root
        clip_model = config.clip_model.replace("/", "-")
        dirname = "/".join(exp_root.split("/")[:2])
        dirname = os.path.join(
            dirname, f"{config.dataset}_{config.weight_init_method.capitalize()}"
        )
        if config.lambda_l1 > 0:
            dirname += f"_L1"
        if not config.use_normalize:
            dirname += "_NoNorm"
        if config.scale != 1:
            dirname += f"_Scale={config.scale}"
        exp_root = os.path.join(
            dirname,
            f"BottleNeck{config.num_static_concept + config.num_dynamic_concept}="
            f"S{config.num_static_concept}+D{config.num_dynamic_concept}-LR{config.lr}",
            f"{config.n_shots}shot_{clip_model}",
        )
        if config.concept_select_fn == "random":
            exp_root += "_RandSelect"
        if config.num_dynamic_concept == 0 and (
            config.lambda_discri > 0 or config.lambda_ort > 0 or config.lambda_align > 0
        ):
            raise ValueError(
                "Dynamic concepts are not used, but discriminative, orthogonal, or alignment loss is used"
            )
        if config.lambda_discri > 0:
            exp_root += (
                f"_Discri={config.lambda_discri_alpha}_{config.lambda_discri_beta}"
            )
        if config.lambda_ort > 0:
            exp_root += f"_Ort={config.lambda_ort}"
        if config.lambda_align > 0:
            exp_root += f"_Align={config.lambda_align}"
        return exp_root

    @staticmethod
    def root_to_config(exp_root):
        config = Config(dict(exp_root=exp_root))

        # Parse the dataset and weight_init_method
        parts = exp_root.split("/")
        dataset_weight = parts[2].split("_")
        config.dataset = dataset_weight[0]
        config.weight_init_method = dataset_weight[1].lower()
        if "L1" in parts[2]:
            config.lambda_l1 = 0.001
        if "NoNorm" in parts[2]:
            config.use_normalize = False
        # Extract static and dynamic concepts from the bottleneck part
        bottleneck_info = parts[3].split("=")[1].split("-")[0]
        lr = parts[3].split(bottleneck_info + "-")[-1]
        config.lr = float(lr[2:])
        static_dynamic = bottleneck_info.split("+")
        config.num_static_concept = int(static_dynamic[0][1:])
        config.num_dynamic_concept = int(static_dynamic[1][1:])

        # Parse n_shots and clip_model
        shot_info, clip_model = parts[4].split("_")[:2]
        config.n_shots = int(shot_info[:-4]) if "all" not in shot_info else "all"
        config.clip_model = "ViT-" + "/".join(clip_model.split("-")[1:])

        # Parse optional L1, Random Select, Static, and Dynamic flags
        tail = "_".join(parts[4].split("_")[2:])

        config.concept_select_fn = "random" if "_RandSelect" in tail else "submodular"

        if "_Static" in tail:
            config.num_dynamic_concept = 0
        elif "_Dynamic" in tail:
            config.num_static_concept = 0

        # Parse discriminative, orthogonal, and alignment lambdas
        if "Discri=" in tail:
            discri_info = tail.split("Discri=")[1].split("_")[:2]
            alpha, beta = map(
                lambda x: int(x) if x.isdigit() else float(x), discri_info
            )
            config.lambda_discri = 1
            config.lambda_discri_alpha = alpha
            config.lambda_discri_beta = beta
        else:
            config.lambda_discri = 0

        if "_Ort=" in tail:
            config.lambda_ort = map(
                lambda x: int(x) if x.isdigit() else float(x),
                tail.split("_Ort=")[1].split("_")[0],
            ).__next__()
        else:
            config.lambda_ort = 0

        if "_Align=" in tail:
            config.lambda_align = map(
                lambda x: int(x) if x.isdigit() else float(x),
                tail.split("_Align=")[1].split("_")[0],
            ).__next__()
        else:
            config.lambda_align = 0

        return config
