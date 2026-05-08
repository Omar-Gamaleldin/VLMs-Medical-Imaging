#! /usr/bin/env bash
# Here you can add bash commands:
#SBATCH --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,/usr/lib64/slurm:/usr/lib64/slurm,/usr/lib64/libhwloc.so:/usr/lib64/libhwloc.so,/usr/lib64/libhwloc.so.15:/usr/lib64/libhwloc.so.15,/pfs/work9/workspace/scratch/ul_ekd37-gepa-optimization/:/pfs/work9/workspace/scratch/ul_ekd37-gepa-optimization
#SBATCH --container-mount-home
#SBATCH --output=slurm_log/gepa/%x_%j.out  # Output file for job logs %x for job-name and %j for job-id
#SBATCH --container-name=vllm_qwen3.5
#SBATCH --job-name=Qwen3.5_9B_Server
#SBATCH --gres=gpu:1

cd $HOME/VLMs-Medical-Imaging

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export XDG_CACHE_HOME=/tmp/cache

vllm serve models/Qwen3.5-9B \
	--port 8001 \
	--mm-encoder-tp-mode data \
	--mm-processor-cache-type shm \
	--reasoning-parser qwen3 \
	--enable-prefix-caching \
	--served-model-name qwen3.5-9b\

