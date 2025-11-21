gpu_id=0

for seed in 42 0 12345
do
    echo "Running Imagenet100 with seed $seed on GPU $gpu_id"

        CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
                    --do_train_full \
                    --do_test \
                    --seed $seed \
                    --model resnet50 \
                    --concept_wts 0.01 \
                    --cls_wts 0.01 \
                    --dataset imagenet100 \
                    --data_root ./../DATA/inet100 \
                    --concept_file ./../DATA/concepts/inet100_concepts.json \
                    --lattice_path ./../DATA/lattices/inet100_context.pkl \
                    --num_clfs 2 \
                    --lattice_levels 1 3 \
                    --backbone_layer_ids 2 3 \
                    --lr 1e-4 \
                    --epochs 30 \
                    --batch_size 256 \
                    --verbose 100 \
                    --clf_special_init \
                    --save_model_dir ./saved_models/

    sleep 30
done