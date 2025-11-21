# $1: cfg path
# $2: ckpt path
CUDA_VISIBLE_DEVICES=4 python main.py --cfg $1 --func asso_opt_main --test --cfg-options bs=512 ckpt_path=$2 ${@:3}