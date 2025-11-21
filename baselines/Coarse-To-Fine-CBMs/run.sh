gpu=$1

CUDA_VISIBLE_DEVICES=$gpu python main.py --dataset cifar100 --compute_similarities --batch_size 256 --epochs 30

sleep 30

CUDA_VISIBLE_DEVICES=$gpu python main.py --dataset awa2 --compute_similarities --batch_size 256 --epochs 30

sleep 30

CUDA_VISIBLE_DEVICES=$gpu python main.py --dataset inet100 --compute_similarities --batch_size 256 --epochs 30
