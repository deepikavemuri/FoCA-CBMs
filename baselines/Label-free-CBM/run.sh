gpu_id=15
for seed in 42 0 12345 
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train_cbm.py \
        --seed $seed \
        --dataset imagenet100 \
        --backbone resnet50 \
        --concept_set ./DATA/concepts/inet100_concepts.txt \
        --clip_cutoff 0.28 \
        --n_iters 1000 \
        --lam 0.0001

    CUDA_VISIBLE_DEVICES=$gpu_id python train_cbm.py \
        --seed $seed \
        --dataset awa2 \
        --data_root ./DATA/Animals_with_Attributes2 \
        --backbone resnet18 \
        --concept_set ./DATA/concepts/awa2_concepts.txt \
        --clip_cutoff 0.26 \
        --n_iters 1000 \
        --lam 0.0001

    CUDA_VISIBLE_DEVICES=$gpu_id python train_cbm.py \
        --seed $seed \
        --dataset cifar100 \
        --data_root ./DATA/cifar100 \
        --backbone resnet50 \
        --concept_set ./DATA/concepts/cifar100_concepts.txt \
        --clip_cutoff 0.26 \
        --n_iters 1000 \
        --lam 0.0001

    sleep 10
done