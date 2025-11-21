import os

# CUB Constants
# CUB data is downloaded from the CBM release.
# Dataset: https://worksheets.codalab.org/rest/bundles/0xd013a7ba2e88481bbc07e787f73109f5/
# Metadata and splits: https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683
CUB_DATA_DIR = "/raid/ai24mtech12011/projects/temp/fca4nn/DATA/CUB_200_2011"
CUB_PROCESSED_DIR = "/raid/ai24mtech12011/projects/temp/fca4nn/DATA/CUB_200_2011/CUB112"


# Derm data constants
# Derm7pt is obtained from : https://derm.cs.sfu.ca/Welcome.html
DERM7_FOLDER = "/path/to/derm7pt/"
DERM7_META = os.path.join(DERM7_FOLDER, "meta", "meta.csv")
DERM7_TRAIN_IDX = os.path.join(DERM7_FOLDER, "meta", "train_indexes.csv")
DERM7_VAL_IDX = os.path.join(DERM7_FOLDER, "meta", "valid_indexes.csv")

# Ham10000 can be obtained from : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
HAM10K_DATA_DIR = "/path/to/broden/"


# BRODEN concept bank
BRODEN_CONCEPTS = "/path/to/broden/"

# ImageNet100 concept bank and dataset
IMAGENET100_CONCEPTS = "/path/to/imagenet100/"
IMAGENET100_DATA_DIR = "/path/to/imagenet100/"


# AwA2 Dataset
AWA2_TRAIN = (
    "/raid/ai24mtech12011/projects/temp/fca4nn/DATA/Animals_with_Attributes2/train.csv"
)
AWA2_VAl = (
    "/raid/ai24mtech12011/projects/temp/fca4nn/DATA/Animals_with_Attributes2/val.csv"
)
AWA2_TEST = (
    "/raid/ai24mtech12011/projects/temp/fca4nn/DATA/Animals_with_Attributes2/test.csv"
)
AWA2_DATA = (
    "/raid/ai24mtech12011/projects/temp/fca4nn/DATA/Animals_with_Attributes2/JPEGImages"
)
AWA2_CONCEPTS = (
    "/raid/ai24mtech12011/projects/temp/fca4nn/DATA/concepts/awa2_concepts.txt"
)
AWA2_CLASSES = "/raid/ai24mtech12011/projects/temp/fca4nn/DATA/classes/awa2_classes.txt"
