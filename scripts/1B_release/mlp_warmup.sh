#!/bin/bash
# How to run: `bash scripts/1B_release/mlp_warmup.sh  <number-of-total-nodes> <head-node-name | OPTIONAL> <rank-of-current-node | OPTIONAL>`
# For single-node:
# E.g., `bash scripts/1B_release/mlp_warmup.sh 1`
# For multinode:
# E.g., `bash scripts/1B_release/mlp_warmup.sh 2 0 node1`
# E.g., `bash scripts/1B_release/mlp_warmup.sh 2 1 node2`

# Load the environment flags
set -a
source .env
set +a

NNODES=$1
if [ -z "$NNODES" ]; then
    echo "Error: Number of nodes (NNODES) is not specified."
    exit 1
fi
# Specify the CLI args (sys.argv)
if [ "$NNODES" -eq 1 ]; then
    head_node=$(hostname)
    RANK=0
else
    head_node=$2
    RANK=$3
    set -a
    source .mnenv
    set +a
fi



head_node_ip=`getent hosts $head_node | awk '{ print $1 }'`
echo "Head Node: ${head_node}"
echo "Head Node IP: ${head_node_ip}"
echo "master_port: ${master_port}"
echo "NNODES: ${NNODES}"
echo "RDZV_ID: ${RDZV_ID}"
# Get node rank based on custom ordering
ACCUM_STEPS=$((32/32/${NNODES}))

############### Alignment ################

BASE_RUN_NAME="instellavl-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-warmup"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
echo -e "\n\n"

# ACCELERATE_CPU_AFFINITY=1 
NUM_GPUS=$((8*NNODES))

torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}"  \
    --rdzv_id=${RDZV_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node:$master_port \
    instellavl/train/train.py \
    --deepspeed configs/zero_configs/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path configs/data_configs/online_warmup.yaml \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $CKPT_DIR/projectors/${BASE_RUN_NAME} \
    --resume_from_checkpoint False \
    --num_gpus $NUM_GPUS \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $ACCUM_STEPS \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_pin_memory False \
    --dispatch_batches False \
    --online_training True \
    --run_name $BASE_RUN_NAME 
