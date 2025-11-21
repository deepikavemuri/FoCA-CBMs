_base_ = "../base.py"

# dataset
dataset = "cifar100"
data_root = f"./../DATA/cifar100_images"

num_class = 100

concept_select_fn = "submodular"  # 'submodular' or 'random'
submodular_weights = [1e7, 0.1]
num_concept_per_class = 10
dynamic_concept_ratio = 0.5

use_normalize = True
scale = 1
lr = 5e-5
max_epochs = 5000

# weight matrix
weight_init_method = "zero"  # 'rand' or 'zero' or 'cosine' or 'topk' or 'label'

# train mode
train_mode = "joint"  # 'joint' or 'concept_{epochs}' or 'concept_stop_{epochs}'

# loss function
lambda_discri_alpha = 2
lambda_discri_beta = 0.1
lambda_ort = 0.1
lambda_align = 0.01
lambda_l1 = 0.001
