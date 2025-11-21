gpu_id=14

for seed in 42 0 12345
do
    echo "Running AWA2 with seed $seed on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id python train_cbm.py \
                    --do_train \
                    --do_test \
                    --seed $seed \
                    --dataset cifar100 \
                    --model resnet50 \
                    --concept_wts 0.1 \
                    --data_dir ./../DATA/cifar100/ \
                    --concept_file ./../DATA/concepts/cifar100_concepts.json \
                    --lr 3e-4 \
                    --epochs 75 \
                    --batch_size 256 \
                    --verbose 10 \
                    --keep_top_k 2 \
                    --save_model_dir ./saved_models/

    sleep 30
done