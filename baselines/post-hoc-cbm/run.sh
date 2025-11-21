OUTPUT_DIR=./outputs
gpu_id=15
dataset="awa2" # inet100
data_path="/raid/DATASETS/${dataset}/"

mkdir -p $OUTPUT_DIR

for seed in 0 42 12345
do
        
        CUDA_VISIBLE_DEVICES=$gpu_id python learn_concepts_multimodal.py \
                        --classes=$dataset \
                        --backbone-name="clip:RN50" \
                        --concept_list="/DATA/concepts/${dataset}_concepts.txt" \
                        --out-dir=$OUTPUT_DIR \
                        > "${OUTPUT_DIR}/learn_concepts_multimodal_rn50_${dataset}_seed_${seed}.log"

        sleep 10

        CUDA_VISIBLE_DEVICES=$gpu_id python train_pcbm.py \
                        --concept-bank="${OUTPUT_DIR}/multimodal_concept_clip:RN50_${dataset}_recurse:1.pkl" \
                        --dataset=$dataset \
                        --backbone-name="clip:RN50" \
                        --out-dir=$OUTPUT_DIR \
                        --data_path=$data_path \
                        --lam=2e-4 \
                        --seed $seed \
                        > "${OUTPUT_DIR}/train_pcbm_multimodal_${dataset}_seed_${seed}.log"

        sleep 10

        CUDA_VISIBLE_DEVICES=$gpu_id python train_pcbm_h.py \
                --concept-bank="${OUTPUT_DIR}/multimodal_concept_clip:RN50_${dataset}_recurse:1.pkl" \
                --pcbm-path="${OUTPUT_DIR}/pcbm_${dataset}__clip:RN50__multimodal_concept_clip:RN50_${dataset}_recurse:1__lam:0.0002__alpha:0.99__seed:${seed}.ckpt" \
                --out-dir=$OUTPUT_DIR \
                --dataset=$dataset \
                --data_path=$data_path \
                > "${OUTPUT_DIR}/train_pcbm_multimodal_${dataset}_hybrid_seed_${seed}.log"
        
        sleep 10

done