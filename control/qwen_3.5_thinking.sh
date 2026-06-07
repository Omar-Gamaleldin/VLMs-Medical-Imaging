#! /usr/bin/env bash 

# Here you can add bash commands:
#SBATCH --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,/usr/lib64/slurm:/usr/lib64/slurm,/usr/lib64/libhwloc.so:/usr/lib64/libhwloc.so,/usr/lib64/libhwloc.so.15:/usr/lib64/libhwloc.so.15,/pfs/work9/workspace/scratch/ul_ekd37-gepa-optimization:/pfs/work9/workspace/scratch/ul_ekd37-gepa-optimization
#SBATCH --container-mount-home
#SBATCH --output=slurm_log/%x_%j.out  # Output file for job logs %x for job-name and %j for job-id
#SBATCH --container-name=vllm_qwen3.5
#SBATCH --gres=gpu:1
#SBATCH --job-name=Qwen3.5_Thinking_Control

cd $HOME/VLMs-Medical-Imaging

# Get Data form Workspace to GPU Node -------------------------------
export WORKSPACE_PATH=/pfs/work9/workspace/scratch/ul_ekd37-gepa-optimization

# Extract compressed input dataset on local SSD (removed -v for speed)
echo "Extracting dataset..."
tar -C $TMPDIR/ -xzf $WORKSPACE_PATH/dataset.tgz

export DATAPATH=$TMPDIR/dataset
#-------

# Environment Setup
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export XDG_CACHE_HOME=/tmp/cache

echo "================================================================================"
echo "Running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "================================================================================"
nvidia-smi

vllm serve models/Qwen3.5-9B \
    --served-model-name "Qwen3.5-9B" \
    --host 0.0.0.0 \
    --port 8001 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --reasoning-parser qwen3 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32768 \
    --trust-remote-code \
    --dtype bfloat16 &

VLLM_PID=$!

# ── Wait until the server is ready ───────────────────────────────────────────
echo "Waiting for server to be ready..."
MAX_WAIT=300   # seconds before giving up (model load can be slow)
ELAPSED=0
 
until curl -sf "http://localhost:8001/health" > /dev/null 2>&1; do
    # Check the server process is still alive
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died. Check logs above."
        exit 1
    fi
 
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: Server did not become ready within ${MAX_WAIT}s."
        kill $VLLM_PID
        exit 1
    fi
 
    echo "  still loading... (${ELAPSED}s elapsed)"
    sleep 10
    ELAPSED=$((ELAPSED + 10))
done
 
echo "Server is ready after ${ELAPSED}s!"
 
# ── Run the benchmark ─────────────────────────────────────────────────────────
 
# python3 -m venv ~/venvs/qwen3.5_control
# source ~/venvs/qwen3.5_control/bin/activate
python3 -m pip install pillow transformers==5.5.0

# Start inference
echo "Starting Qwen 3.5 Inference..."
python3 -u control/qwen3.5_thinking.py --data_dir=$DATAPATH
	
