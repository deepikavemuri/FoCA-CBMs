gpu=$1


# Train 
CUDA_VISIBLE_DEVICES=$gpu python main.py --cfg cfg/asso_opt/CIFAR100/CIFAR100_allshot_fac.py --work-dir exp/asso_opt/CIFAR100/CIFAR100_allshot_fac --func asso_opt_main ${@:2}

sleep 30

CUDA_VISIBLE_DEVICES=$gpu python main.py --cfg cfg/asso_opt/awa2/awa2_allshot_fac.py --work-dir exp/asso_opt/awa2/awa2_allshot_fac --func asso_opt_main ${@:2}

sleep 30

CUDA_VISIBLE_DEVICES=$gpu python main.py --cfg cfg/asso_opt/inet100/inet100_allshot_fac.py --work-dir exp/asso_opt/inet100/inet100_allshot_fac --func asso_opt_main ${@:2}

sleep 60
# Test
# run labo_test.sh with proper args