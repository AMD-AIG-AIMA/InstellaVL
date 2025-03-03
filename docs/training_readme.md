Here we will elucidate on different training style one can adopt based on the settings that favours them most. 

> [!NOTE]
> Since we have two formats of training: a) **Online** and b) **Offline** (decided by `--online_training` flag in each of the training script), we need to have:
> 1. **For Online mode**: Everything (*data* as well as *prior model checkpoints*, if any-- `adapter` in case of Pre-training and `pretrained checkpoint` in case of InstrcutionTuning, should be in S3 bucket.)
>    - <ins>Warmup</ins>: `CKPT_DIR` in `.env` should point to an online location (e.g., if checkpoints folder in S3 bucket looks something like `s3://object_name/path/to/checkpoints/`, then `CKPT_DIR` = `path/to/checkpoints`.)
>    - <ins>Pre-training</ins>: Now the projector is at online location - (e.g., `s3://object_name/path/to/checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin`). Set `--pretrain_mm_mlp_adapter=$CKPT_DIR/projectors/${BASE_RUN_NAME}/mm_projector.bin` in the pre-training bash script. While ensure the `--output_dir="${CKPT_DIR}/${MID_RUN_NAME}"`.
>     - <ins>Instruction-tuning</ins>: Now the pretrained model is at `"${CKPT_DIR}/${MID_RUN_NAME}"` of S3 bucket. And ensure `--output_dir="${CKPT_DIR}/${SI_RUN_NAME}"`.
> 2. **For Offline mode**: Again we have two modes of data loading (as detailed :point_right: [here](../README.md#hammer_and_wrench-how-to-curate-the-above-datasets-both-for-offline-and-online-setting)). Whichever way suits better, mention it at `--data_path` flag in training-script. It would be either `offline_<train-step>.yaml` or `offline_s3_<train-step>.yaml`, where `<train-step>` = *mlp_warmup* / *pretraining* / *instruction_tuning*. And the model checkpoints should be inside `InstellaVL/checkpoints` folder.
>    - Now the only change is `CKPT_DIR` in `.env`. Change it to an offline location (e.g., `checkpoints` folder of `InstellaVL`.)

1. If you have `slurm` in your cluster:
    1. **Multi Node**: For multi node setting do the following:
        ```bash
        # Note: Here <train-step> = mlp_warmup, pretraining, instruction_tuning
        user@landing-node:~$ sbatch scripts/1B_release/sbatch_instellavl_<train-step>.sh <absolute-path-to-InstellaVL-repo>
        ```
    2. **Single Node**: For single node just change the `SBATCH` flags, namely `--nodes=1`, `--ntasks=1`, and `--nodelist=<name-of-node>` if any, in the above script.

2. ELse,
    1. **Single Node**: For single node purpose, put the following script in a bash (`.sh`) file (say `docker_run_instellavl.sh`) and execute it using `bash <path/to/docker_run_instellavl.sh>`.
        ```bash
        # Note: Here <train-step> = mlp_warmup, pretraining, instruction_tuning
        docker run --rm --network=host \
            --device=/dev/kfd \
            --device=/dev/dri \
            --device=/dev/infiniband \
            --group-add=video \
            --ipc=host \
            --cap-add=SYS_PTRACE \
            --security-opt seccomp=unconfined \
            --privileged \
            -v /cephfs:/cephfs \
            --shm-size 8G \
            -v "<absolute-path/to/parent-directory-of-InstellaVL-repo>":"<absolute-path/to/parent-directory-of-InstellaVL-repo>" \
            -w "<path/to/InstellaVL-repo>" \
            --name="instellavl_container" \
            rocm/instellavl:latest \
            bash -c "bash scripts/1B_release/<train-step>.sh 1"
        # Here 1 at the end denotes the number of nodes.
        ```
    2. **Multi Node**,
        1. We need to have interactive `tmux`-like session in the respective nodes we desire to use for multinode training.
        2. Initialize docker interactive container inside each node.
            ```bash
            # Since workdir=InstellaVL repo, after container activation we will automatically land in Instella-VL repo
            user@node:~$ docker run -it --network=host --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --privileged -v /cephfs:/cephfs --shm-size 8G -v "<path/to/parent-directory-of-InstellaVL-repo>":"<path/to/parent-directory-of-InstellaVL-repo>" -w "<path/to/InstellaVL-repo>" --name="instellavl_container" rocm/instellavl:latest /bin/bash
            ``` 
        3. Now run the training code using `bash` command inside docker container. But before that decide one node to be master-node and assign `rank=0` by specifying `0` as sys argv at the end of bash execution (shown below) while other nodes can take `rank != 0`:
            ```bash
            # Note: Inside docker container user@node changes to root@node. And suppose that node1 is master node. And we have 4 nodes for training.
            # Note: Here <train-step> = mlp_warmup, pretraining, instruction_tuning
            root@node1:~$ bash scripts/1B_release/<train-step>.sh 4 node1 0
            root@node2:~$ bash scripts/1B_release/<train-step>.sh 4 node1 1
            root@node3:~$ bash scripts/1B_release/<train-step>.sh 4 node1 2
            root@node4:~$ bash scripts/1B_release/<train-step>.sh 4 node1 3
            ```


> [!CAUTION]
> Since we have 3 ways of loading datasets into model (described in main [README.md](../README.md#hammer_and_wrench-how-to-curate-the-above-datasets-both-for-offline-and-online-setting)), we must be cautious on how to load them by correctly mentioning the corresponding `.yaml` files at [`data_configs`](./configs/data_configs/). *Also don't forget to modify the absolute path to each dataset mentioned therein.*
