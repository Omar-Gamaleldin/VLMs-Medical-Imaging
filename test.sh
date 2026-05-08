#! /usr/bin/env bash
# Here you can add bash commands:
#SBATCH --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,/usr/lib64/slurm:/usr/lib64/slurm,/usr/lib64/libhwloc.so:/usr/lib64/libhwloc.so,/usr/lib64/libhwloc.so.15:/usr/lib64/libhwloc.so.15,/pfs/work9/workspace/scratch/ul_ekd37-gepa-optimization/:/pfs/work9/workspace/scratch/ul_ekd37-gepa-optimization
#SBATCH --container-mount-home
#SBATCH --output=slurm_log/gepa/%x_%j.out  # Output file for job logs %x for job-name and %j for job-id
#SBATCH --container-name=vllm_qwen3.5
#SBATCH --job-name=Qwen3.5_9B_Server
#SBATCH --gres=gpu:1

cd $HOME/VLMs-Medical-Imaging

# Get Data form Workspace to GPU Node -------------------------------
export WORKSPACE_PATH=/pfs/work9/workspace/scratch/ul_ekd37-gepa-optimization

# Extract compressed input dataset on local SSD
echo "Extracting dataset..."
tar -C $TMPDIR/ -xzf $WORKSPACE_PATH/dataset.tgz

export DATAPATH=$TMPDIR/dataset

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export XDG_CACHE_HOME=/tmp/cache

wait_for_server() {
  local url=$1
  local max_attempts=${2:-30}
  local interval=${3:-10}

  echo "Waiting for server at $url..."
  for ((i=1; i<=max_attempts; i++)); do
    if curl -sf "$url" > /dev/null 2>&1; then
      echo "Server is ready!"
      return 0
    fi
    echo "Attempt $i/$max_attempts — retrying in ${interval}s..."
    sleep "$interval"
  done

  echo "Server did not become ready in time."
  return 1
}

(vllm serve models/Qwen3.5-9B \
	--port 8001 \
	--mm-encoder-tp-mode data \
	--mm-processor-cache-type shm \
	--reasoning-parser qwen3 \
	--enable-prefix-caching \
	--served-model-name qwen3.5-9b\ ) &

wait_for_server "http://localhost:8001/health"

python3 -m pip install -q pillow dspy transformers==5.5.0

python3 gepa/qwen3.5-gepa.py --data_dir=$DATAPATH
