# $1: number of shots -> always all
# $2: dataset (awa2/CIFAR100/inet100)

CUDA_VISIBLE_DEVICES=1 python main.py --cfg cfg/asso_opt/$2/$2_$1shot_fac.py --work-dir exp/asso_opt/$2/$2_$1shot_fac --func asso_opt_main ${@:3}