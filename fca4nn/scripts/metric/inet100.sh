gpu_id=0

CUDA_VISIBLE_DEVICES=$gpu_id python metric_calculator.py \
            --seed 42 \
            --dataset inet100 \
            --model_name SCBM::resnet50 \
            --model_weights < model_weights_paths > \
            --data_path ././DATA/inet100/ \
            --lattice_path ././DATA/lattices/inet100_context.pkl \
            --lattice_levels 1 3\
            --backbone_layer_ids 3 4 \
            --metadata_path ./saved_models/metric_metadata/ \
            --separation_score davies_bouldin \
            --clustering_method kmeans


### Different Model Names (--model_name): 

## Baselines -> only change --model_weights along with --model_name
# - CEM::resnet50
# - SCBM::resnet50
# - PCBM::resnet50
# - CBM::resnet50
# - PYTORCH::resnet50
# - CLIP::RN50
# - VIT::deit_base_patch16_224
# - MLPCBM::resnet50

## Ours -> change --model_weights, --lattice_levels, --backbone_layer_ids along with --model_name according to the ones used during training
# - MCLCBM::resnet50
# - OURS-1FCA::resnet50
# - OURS-2FCA::resnet50
# - OURS-1FCA::vit_base_patch16_224
# - OURS-2FCA::vit_base_patch16_224