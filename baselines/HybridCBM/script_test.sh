gpu=$1
path=$2

CUDA_VISIBLE_DEVICES=$gpu python trainLinear.py --exp_root  ${path} --test