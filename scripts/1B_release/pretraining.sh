#!/bin/bash
# How to run: `bash scripts/1B_release/pretraining.sh  <number-of-total-nodes> <head-node-name | OPTIONAL> <rank-of-current-node | OPTIONAL>`
# For single-node:
# E.g., `bash scripts/1B_release/pretraining.sh 1`
# For multinode:
# E.g., `bash scripts/1B_release/pretraining.sh 2 0 node1`
# E.g., `bash scripts/1B_release/pretraining.sh 2 1 node2`

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
ACCUM_STEPS=$((32/16/${NNODES}))

mkdir -p ${MIOPEN_USER_DB_PATH}
touch ${MIOPEN_USER_DB_PATH}/gfx942130.HIP.3_2_0_36bb7fd4a-dirty.ufdb.txt
touch ${MIOPEN_USER_DB_PATH}/gfx942130.HIP.3_2_0_36bb7fd4a-dirty.udb.txt
touch ${MIOPEN_USER_DB_PATH}/gfx942130.ukdb

############### Pretrain ################
BASE_RUN_NAME="instellavl-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-warmup"
ADAPTER_PATH="${CKPT_DIR}/projectors/${BASE_RUN_NAME}"

echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
echo -e "\n\n"

MID_RUN_NAME="instellavl-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-pretrain"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"
echo -e "\n\n"

# this could also be the previous stage checkpoint
CKPT_PATH=$LLM_VERSION


# ACCELERATE_CPU_AFFINITY=1 
NUM_GPUS=$((8*NNODES))
torchrun --nproc_per_node=8 --nnodes=${NNODES} --node_rank=$RANK \
    --rdzv_id=${RDZV_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node:$master_port \
    instellavl/train/train.py \
    --deepspeed configs/zero_configs/zero3_offload.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path configs/data_configs/online_pretraining.yaml \
    --pretrain_mm_mlp_adapter="${ADAPTER_PATH}/mm_projector.bin" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 336), (336, 672), (336, 1008), (336, 1344), (336, 1680), (672, 336), (672, 672), (1008, 336), (1344, 336), (1680, 336)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "${CKPT_DIR}/${MID_RUN_NAME}" \
    --resume_from_checkpoint False \
    --num_gpus $NUM_GPUS \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${ACCUM_STEPS} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --online_training True \
    --dispatch_batches False
