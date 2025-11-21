gpu=$2
dataset=$1

CUDA_VISIBLE_DEVICES=$gpu python trainLinear.py --config config/HybridCBM/${dataset}/${dataset}_allshot.py \
--cfg-options clip_model=RN50 \
--cfg-options concept_select_fn=submodular \
--cfg-options num_concept_per_class=10 \
--cfg-options dynamic_concept_ratio=0.5 \
--cfg-options lambda_discri_alpha=2 \
--cfg-options lambda_discri_beta=0.1 \
--cfg-options lambda_ort=0.1 \
--cfg-options lambda_align=0.01 \
--cfg-options seed=12345