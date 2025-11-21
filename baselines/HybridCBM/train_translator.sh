export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12355

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --rdzv_id=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_translator.py \
    --out_dir ./weights/translator \
    --epochs 1000 \
    --save_every 20 \
    --batch_size 256 \
    --clip_model RN50 \
    --augment