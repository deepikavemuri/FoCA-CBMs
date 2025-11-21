gpu_id=0

for seed in 42 0 12345
do
    echo "Running AWA2 with seed $seed on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
                --do_train_full \
                --do_test \
                --seed $seed \
                --dataset awa2 \
                --model_type vit \
                --model deit_base_patch16_224 \
                --concept_wts 0.01 \
                --cls_wts 0.01 \
                --data_root ././DATA/Animals_with_Attributes2/ \
                --concept_file ././DATA/concepts/awa2_concepts.json \
                --lattice_path ././DATA/lattices/awa2_context.pkl \
                --num_clfs 2 \
                --lattice_levels 1 3 \
                --backbone_layer_ids 3 4 \
                --lr 3e-4 \
                --epochs 1 \
                --batch_size 256 \
                --verbose 10 \
                --keep_top_k 2 \
                --clf_special_init \
                --save_model_dir ./saved_models/

    sleep 30
done