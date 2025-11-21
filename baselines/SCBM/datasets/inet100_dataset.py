import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, RandomResizedCrop, CenterCrop, Normalize, ToTensor, RandomHorizontalFlip, ColorJitter
import json
from collections import OrderedDict
from PIL import Image
from glob import glob
import random
import numpy as np

IMAGENET100_CLASSES = OrderedDict(
    {
        "n01968897": "chambered nautilus, pearly nautilus, nautilus",
        "n01770081": "harvestman, daddy longlegs, Phalangium opilio",
        "n01818515": "macaw",
        "n02011460": "bittern",
        "n01496331": "electric ray, crampfish, numbfish, torpedo",
        "n04347754": "submarine, pigboat, sub, U-boat'",
        "n01687978": "agama",
        "n01740131": "night snake, Hypsiglena torquata",
        "n01537544": "indigo bunting, indigo finch, indigo bird, Passerina cyanea",
        "n01491361": "tiger shark, Galeocerdo cuvieri",
        "n02007558": "flamingo",
        "n01735189": "garter snake, grass snake",
        "n01630670": "common newt, Triturus vulgaris",
        "n01440764": "tench, Tinca tinca",
        "n01819313": "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita",
        "n02002556": "white stork, Ciconia ciconia",
        "n01667778": "terrapin",
        "n01755581": "diamondback, diamondback rattlesnake, Crotalus adamanteus",
        "n01924916": "flatworm, platyhelminth",
        "n01751748": "sea snake",
        "n01984695": "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish",
        "n01729977": "green snake, grass snake",
        "n01614925": "bald eagle, American eagle, Haliaeetus leucocephalus",
        "n01608432": "kite",
        "n01443537": "goldfish, Carassius auratus",
        "n01770393": "scorpion",
        "n01855672": "goose",
        "n01560419": "bulbul",
        "n01592084": "chickadee",
        "n01914609": "sea anemone, anemone",
        "n01582220": "magpie",
        "n01667114": "mud turtle",
        "n01784675": "centipede",
        "n01820546": "lorikeet",
        "n01773797": "garden spider, Aranea diademata",
        "n02006656": "spoonbill",
        "n01986214": "hermit crab",
        "n01484850": "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
        "n01749939": "green mamba",
        "n01828970": "bee eater",
        "n02018795": "bustard",
        "n01695060": "Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis",
        "n01729322": "hognose snake, puff adder, sand viper",
        "n01677366": "common iguana, iguana, Iguana iguana",
        "n01734418": "king snake, kingsnake",
        "n01843383": "toucan",
        "n01806143": "peacock",
        "n01773549": "barn spider, Araneus cavaticus",
        "n01775062": "wolf spider, hunting spider",
        "n01728572": "thunder snake, worm snake, Carphophis amoenus",
        "n01601694": "water ouzel, dipper",
        "n01978287": "Dungeness crab, Cancer magister",
        "n01930112": "nematode, nematode worm, roundworm",
        "n01739381": "vine snake",
        "n01883070": "wombat",
        "n01774384": "black widow, Latrodectus mactans",
        "n02037110": "oystercatcher, oyster catcher",
        "n01795545": "black grouse",
        "n02027492": "red-backed sandpiper, dunlin, Erolia alpina",
        "n01531178": "goldfinch, Carduelis carduelis",
        "n01944390": "snail",
        "n01494475": "hammerhead, hammerhead shark",
        "n01632458": "spotted salamander, Ambystoma maculatum",
        "n01698640": "American alligator, Alligator mississipiensis",
        "n01675722": "banded gecko",
        "n01877812": "wallaby, brush kangaroo",
        "n01622779": "great grey owl, great gray owl, Strix nebulosa",
        "n01910747": "jellyfish",
        "n01860187": "black swan, Cygnus atratus",
        "n01796340": "ptarmigan",
        "n01833805": "hummingbird",
        "n01685808": "whiptail, whiptail lizard",
        "n01756291": "sidewinder, horned rattlesnake, Crotalus cerastes",
        "n01514859": "hen",
        "n01753488": "horned viper, cerastes, sand viper, horned asp, Cerastes cornutus",
        "n02058221": "albatross, mollymawk",
        "n01632777": "axolotl, mud puppy, Ambystoma mexicanum",
        "n01644900": "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui",
        "n02018207": "American coot, marsh hen, mud hen, water hen, Fulica americana",
        "n01664065": "loggerhead, loggerhead turtle, Caretta caretta",
        "n02028035": "redshank, Tringa totanus",
        "n02012849": "crane",
        "n01776313": "tick",
        "n02077923": "sea lion",
        "n01774750": "tarantula",
        "n01742172": "boa constrictor, Constrictor constrictor",
        "n01943899": "conch",
        "n01798484": "prairie chicken, prairie grouse, prairie fowl",
        "n02051845": "pelican",
        "n01824575": "coucal",
        "n02013706": "limpkin, Aramus pictus",
        "n01955084": "chiton, coat-of-mail shell, sea cradle, polyplacophore",
        "n01773157": "black and gold garden spider, Argiope aurantia",
        "n01665541": "leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea",
        "n01498041": "stingray",
        "n01978455": "rock crab, Cancer irroratus",
        "n01693334": "green lizard, Lacerta viridis",
        "n01950731": "sea slug, nudibranch",
        "n01829413": "hornbill",
        "n02093859": "Kerry blue terrier",
    }
)


def get_ImageNet100_CBM_dataloader(datapath):
    data_dir = os.path.join(datapath, "inet100")
    json_file = os.path.join(datapath, "concepts", "inet100_concepts.json")
    image_datasets = {
        "train": Imagenet100ConceptDataset(
            data_root=data_dir,
            json_file=json_file,
            split='train',
        ),
        "val": Imagenet100ConceptDataset(
            data_root=data_dir,
            json_file=json_file,
            split='val',
        ),
        "test": Imagenet100ConceptDataset(
            data_root=data_dir,
            json_file=json_file,
            split='test',
        ),
    }

    return image_datasets["train"], image_datasets["val"], image_datasets["test"]


class Imagenet100ConceptDataset(Dataset):
    def __init__(
        self,
        data_root,
        json_file,
        split="train",
        transforms=None,
        fraction="full",
    ):
        self.num_classes = len(IMAGENET100_CLASSES.keys())
        self.class_concept_dict = dict(
            json.load(open(json_file, "r"), object_pairs_hook=OrderedDict)
        )
        self.split = split
        self.class_list = self.class_concept_dict.keys()
        self.concept_list = []
        for v in self.class_concept_dict.values():
            self.concept_list += v
        self.concept_list = list(set(self.concept_list))
        self.num_attrs = len(self.concept_list)
        self.class_label_map = {
            i: k for i, k in zip(np.arange(0, self.num_classes), self.class_list)
        }
        self.label_class_map = {
            k: i for i, k in zip(np.arange(0, self.num_classes), self.class_list)
        }
        self.concept_label_dict = dict()

        self.dir_idx = {k: v for v, k in enumerate(IMAGENET100_CLASSES.keys())}
        class_dirs = IMAGENET100_CLASSES.keys()

        if transforms is None:
            if split == "train":
                self.transforms = Compose(
                    [
                        RandomResizedCrop(224, interpolation=Image.BILINEAR),
                        RandomHorizontalFlip(),
                        ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                        ),
                        ToTensor(),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            else:
                self.transforms = Compose(
                    [
                        Resize(size=256, interpolation=Image.BILINEAR),
                        CenterCrop(224),
                        ToTensor(),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
        else:
            self.transforms = transforms

        self.data = []
        for d in class_dirs:
            label = self.dir_idx[d]
            # label = d
            if split == "train":
                images = glob(os.path.join(data_root, "train", d, "*.JPEG"))
                images.sort()
                if fraction == "half":
                    images = images[: len(images) // 2]
                elif fraction == "quarter":
                    images = images[: len(images) // 4]
                random.shuffle(images)
            elif split == "val":
                images = glob(os.path.join(data_root, "val", d, "*.JPEG"))
            elif split == "test":
                images = glob(os.path.join(data_root, "test_set", d, "*.JPEG"))
            else:
                raise ValueError("Invalid split: {}".format(split))
            self.data.extend(list(zip(images, [label] * len(images))))

        np.random.shuffle(self.data)
        class_list = list(range(len(self.class_concept_dict.keys())))

        for cls in class_list:
            attrs = self.class_concept_dict[self.class_label_map[cls]]
            for attr in attrs:
                if attr not in self.concept_label_dict.keys():
                    self.concept_label_dict[attr] = len(self.concept_label_dict.keys())

    def __len__(self):
        return len(self.data)

    def get_concept_count(self):
        return len(self.concept_label_dict)

    def __getitem__(self, index):
        image, label = self.data[index]
        if type(image) == str:
            image = Image.open(image).convert("RGB")
            image = self.transforms(image)

        attrs = self.class_concept_dict[self.class_label_map[label]]
        attr_values = []

        for attr in attrs:
            attr_values.append(self.concept_label_dict[attr])

        concept_vector = np.zeros(self.get_concept_count())
        concept_vector[attr_values] = 1

        return {
                    "img_code": image,
                    "labels": label,
                    "features": image,
                    "concepts": concept_vector.astype(np.float32),
                }