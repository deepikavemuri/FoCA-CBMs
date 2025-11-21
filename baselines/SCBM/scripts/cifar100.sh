data='cifar100'
tag='SCBM_experiments'
encoder_arch='resnet50'
gpu=0

for i in 42 0 12345
do
    python -u train.py +model=SCBM +data=$data model.cov_type='amortized' model.reg_precision='l1' model.reg_weight=1 experiment_name="${data}_SCBM_amortized_${i}" seed=$i logging.project=SCBM logging.mode=offline model.tag=$tag model.encoder_arch=$encoder_arch model.j_epochs=70 model.c_epochs=70 model.t_epochs=70 model.train_batch_size=32 model.val_batch_size=32 gpu=$gpu model.learning_rate=0.00003 > /home/ec2-user/sayantaa/fca4nn/baselines/SCBM/main_logs/${data}_SCBM_amortized_${i}.log

    sleep 30
done