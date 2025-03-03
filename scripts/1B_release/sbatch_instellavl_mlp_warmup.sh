#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=144:00:00
#SBATCH --job-name=docker_job
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --gres=gpu:8 
#SBATCH --mem=0
#SBATCH --partition=                             #specify your partition (e.g.: AITeam, iHUB)
#SBATCH --account=                               #specify your relevant group/account
#SBATCH --output=/home/aditysin/PROJECTS/TEST/logs/job_outputW.txt
#SBATCH --nodelist=node1,node2


## running style: sbatch /home/user/InstellaVL/scripts/1B_release/sbatch_instellavl_warmup.sh /home/user/InstellaVL

export WORKDIR=$1
if [ -z "$WORKDIR" ]; then
    echo "Error: WORKDIR is not specified."
    exit 1
fi

# Load environment variables
set -a
source $WORKDIR/.env
set +a

# Determine the head node and node list
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
ALL_NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))

if [ ${#ALL_NODES[@]} -gt 1 ]; then
    set -a
    source $WORKDIR/.mnenv
    set +a
fi

# Verify that head_node is among the allocated nodes
if [[ ! " ${ALL_NODES[@]} " =~ " $head_node " ]]; then
    echo "Error: head_node $head_node is not allocated to this job."
    exit 1
fi

# Build NODELIST with head_node first
NODELIST=("$head_node")
for node in "${ALL_NODES[@]}"; do
    if [ "$node" != "$head_node" ]; then
        NODELIST+=("$node")
    fi
done

# Create a colon-separated list of nodes
NODELIST_CSV=$(IFS=: ; echo "${NODELIST[*]}")
export NODELIST_CSV
export NODELIST
export head_node

# Set master port and other variables
# master_port=$((20000 + $RANDOM % 40000))
echo $SLURM_JOB_NODELIST
head_node_ip=`getent hosts $head_node | awk '{ print $1 }'`
NNODES=$SLURM_NNODES
echo "Head Node: ${head_node}"
echo "Head Node IP: ${head_node_ip}"
echo "master_port: ${master_port}"
echo "NNODES: ${NNODES}"
echo "RDZV_ID: ${RDZV_ID}"
ACCUM_STEPS=$((32/32/${NNODES}))
export ACCUM_STEPS

# Create necessary directories
mkdir -p ${DATA_CACHE}
mkdir -p ${WORKDIR}/logs
mkdir -p ${MIOPEN_USER_DB_PATH}
touch ${MIOPEN_USER_DB_PATH}/gfx942130.HIP.3_2_0_36bb7fd4a-dirty.ufdb.txt
touch ${MIOPEN_USER_DB_PATH}/gfx942130.HIP.3_2_0_36bb7fd4a-dirty.udb.txt
touch ${MIOPEN_USER_DB_PATH}/gfx942130.ukdb

# Set up model specifics
BASE_RUN_NAME="instellavl-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-warmup"
export BASE_RUN_NAME
echo -e "\n\n"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
echo -e "\n\n"

NUM_GPUS=$((8*NNODES))
export NUM_GPUS

################## SRUN ####################
# Remove any existing container named instellavl_container
srun -l bash -c "$DOCKER_COMMAND rm -f $CONTAINER_NAME 2>/dev/null || true  "

export IMAGE_NAME="${IMAGE}:${TAG}"

srun -l bash -c "
    echo 'Checking if Docker image is already loaded on node.'
    if $DOCKER_COMMAND images --format '{{.Repository}}:{{.Tag}}' | grep -q '^$IMAGE_NAME$'; then
        echo 'Docker image is already loaded on node.'
    else
        # Ensure your image is at the common source.        
        echo 'Loading Docker image on node.'
        $DOCKER_COMMAND load -i $DOCKER_SOURCE/instellavl.rar
        echo 'Docker image loaded on node.'
    fi
"

# Construct the command to run
FULL_CMD="
    NODE_HOSTNAME=\$(hostname)
    echo -e \"\n\n\"
    echo \"We're in \$NODE_HOSTNAME\"

    IFS=: read -ra NODELIST <<< \"\$NODELIST_CSV\"
    NODE_RANK=-1

    for i in \"\${!NODELIST[@]}\"; do
        if [ \"\${NODELIST[\$i]}\" == \"\$NODE_HOSTNAME\" ]; then
            NODE_RANK=\$i
            break
        fi
    done

    if [ \"\$NODE_RANK\" == \"-1\" ]; then
        echo \"Error: NODE_HOSTNAME \$NODE_HOSTNAME not found in NODELIST\"
        exit 1
    fi

    NNODES=\${#NODELIST[@]}
    echo \"Running on node \$NODE_HOSTNAME with node_rank \$NODE_RANK and nnodes \$NNODES\"

    DOCKER_RUN_COMMAND=(docker run --rm --network host
    --device=/dev/dri
    --device=/dev/kfd
    --device=/dev/infiniband
    --ipc=host
    --group-add=video
    --cap-add=SYS_PTRACE
    --shm-size 8G
    --security-opt seccomp=unconfined
    --privileged
    -v /cephfs:/cephfs
    -v $WORKDIR:$WORKDIR
    -v $CACHE_DIR:$CACHE_DIR
    --name $CONTAINER_NAME
    -w ${WORKDIR}
    -e WANDB_API_KEY=$WANDB_API_KEY
    -e OMP_NUM_THREADS=200
    -e HSA_FORCE_FINE_GRAIN_PCIE=1
    -e NCCL_DEBUG=INFO
    -e NCCL_ENABLE_DMABUF_SUPPORT=1
    -e NCCL_IB_GID_INDEX=3
    -e NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9
    -e NCCL_IB_DISABLE=0
    -e NCCL_NSOCKS_PERTHREAD=12
    -e NCCL_SOCKET_IFNAME=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
    -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    -e RCCL_MSCCL_ENABLE=0
    -e RCCL_MSCCLPP_ENABLE=0
    -e HF_TOKEN=$HF_TOKEN
    -e HF_HOME=$CACHE_DIR
    -e S3_ENDPOINT_URL=$S3_ENDPOINT_URL
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
    -e TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM
    -e CACHE_DIR=$CACHE_DIR
    -e DATA_CACHE=$DATA_CACHE
    -e MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH
    -e MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
    -e MIOPEN_DEBUG_DISABLE_FIND_DB=1
    -e MIOPEN_DEBUG_DISABLE_SQL_WAL=1
    -e MIOPEN_DISABLE_CACHE=true
    )
    cmd=(
        \"\${DOCKER_RUN_COMMAND[@]}\"  
        \"\$IMAGE_NAME\"  
        bash -c \"pip install -e . --no-deps && torchrun --nproc_per_node=8 --nnodes=\$NNODES --node_rank=\$NODE_RANK --rdzv_id=\$RDZV_ID --rdzv_backend=c10d --rdzv_endpoint=\${head_node}:\${master_port} instellavl/train/train.py \\  
        --deepspeed configs/zero_configs/zero3.json \\  
        --model_name_or_path \${LLM_VERSION} \\  
        --version \${PROMPT_VERSION} \\  
        --data_path configs/data_configs/online_warmup.yaml \\  
        --vision_tower \${VISION_MODEL_VERSION} \\  
        --mm_tunable_parts=\"mm_mlp_adapter\" \\  
        --mm_vision_select_layer -2 \\  
        --mm_projector_type mlp2x_gelu \\  
        --mm_use_im_start_end False \\  
        --mm_use_im_patch_token False \\  
        --bf16 True \\  
        --output_dir \$CKPT_DIR/projectors/\${BASE_RUN_NAME} \\  
        --resume_from_checkpoint False \\  
        --num_gpus \$NUM_GPUS \\  
        --num_train_epochs 1 \\  
        --per_device_train_batch_size 32 \\  
        --per_device_eval_batch_size 4 \\  
        --gradient_accumulation_steps \$ACCUM_STEPS \\  
        --eval_strategy \"no\" \\  
        --save_strategy \"steps\" \\  
        --save_steps 50000 \\  
        --learning_rate 1e-3 \\  
        --weight_decay 0. \\  
        --warmup_ratio 0.03 \\  
        --lr_scheduler_type \"cosine\" \\  
        --logging_steps 1 \\  
        --tf32 False \\  
        --model_max_length 8192 \\  
        --gradient_checkpointing True \\  
        --dataloader_num_workers 16 \\  
        --lazy_preprocess True \\  
        --report_to wandb \\  
        --dataloader_pin_memory False \\  
        --dispatch_batches False \\  
        --run_name \$BASE_RUN_NAME\"  
    )
    echo \"Command to run: \${cmd[@]}\" 

    # Execute the command  
    \"\${cmd[@]}\" > \"\${WORKDIR}\"/logs/job_outputW_\"\$attempt\"_\"\$NODE_RANK\".txt 2>&1

"

# Run the command with retry logic
max_attempts=10
attempt=1
export attempt

until srun --export=ALL -l bash -c "$FULL_CMD"; do
    echo "Attempt $attempt of $max_attempts failed with exit code $?. Retrying in 5 seconds..."
    echo -e "\n\n"
    if [ $attempt -ge $max_attempts ]; then
        echo "Maximum attempts reached. Giving up."
        exit 1
    fi
    attempt=$((attempt+1))
    sleep 5
done  
