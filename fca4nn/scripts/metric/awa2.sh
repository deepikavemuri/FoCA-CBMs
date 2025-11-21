gpu_id=0

CUDA_VISIBLE_DEVICES=$gpu_id python metric_calculator.py \
            --seed 42 \
            --dataset awa2 \
            --model_name VIT::deit_base_patch16_224 \
            --model_weights < model_weights_paths > \
            --data_path ././DATA/Animals_with_Attributes2/ \
            --lattice_path ././DATA/lattices/awa2_context.pkl \
            --lattice_levels 1 3 \
            --backbone_layer_ids 3 4 \
            --metadata_path ./saved_models/metric_metadata/ \
            --separation_score davies_bouldin \
            --clustering_method kmeans


### Different Model Names (--model_name): 

## Baselines -> only change --model_weights along with --model_name
# - CEM::resnet18
# - SCBM::resnet18
# - PCBM::resnet18
# - CBM::resnet18
# - PYTORCH::resnet18
# - CLIP::RN50
# - VIT::deit_base_patch16_224
# - MLPCBM::resnet18

## Ours -> change --model_weights, --lattice_levels, --backbone_layer_ids along with --model_name according to the ones used during training
# - MCLCBM::resnet18
# - OURS-1FCA::resnet18
# - OURS-2FCA::resnet18
# - OURS-1FCA::vit_base_patch16_224
# - OURS-2FCA::vit_base_patch16_224