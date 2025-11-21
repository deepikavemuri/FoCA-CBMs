exp_root = f"exp/HybridCBM"
# Backbone
clip_model = "ViT-L/14"
translator_path = f"./saved_models/translator.pt"

# dataset
use_img_features = True
pin_memory = False
num_workers = 0
force_compute = False

# train
scale = 0.1
concept_lr = 1e-3
max_epochs = 10000

# loss function
lambda_discri = 1
lambda_discri_alpha = 0.5
lambda_discri_beta = 0.5
lambda_ort = 0.1
lambda_align = 0.01
lambda_l1 = 0.001

# weight matrix
init_val = 1.0

use_normalize = False

cls_name_init = None
cls_sim_prior = None
