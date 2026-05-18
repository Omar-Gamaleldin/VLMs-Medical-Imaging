#! /usr/bin/env bash 

# Here you can add bash commands:
#SBATCH --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,/usr/lib64/slurm:/usr/lib64/slurm,/usr/lib64/libhwloc.so:/usr/lib64/libhwloc.so,/usr/lib64/libhwloc.so.15:/usr/lib64/libhwloc.so.15,/pfs/work9/workspace/scratch/ul_ekd37-gepa-optimization:/pfs/work9/workspace/scratch/ul_ekd37-gepa-optimization
#SBATCH --container-mount-home
#SBATCH --output=slurm_log/%x_%j.out  # Output file for job logs %x for job-name and %j for job-id
#SBATCH --container-name=vllm_qwen3.5

# Get Data form Workspace to GPU Node -------------------------------
export WORKSPACE_PATH=/pfs/work9/workspace/scratch/ul_ekd37-test-run

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
echo "The job time limit is: $(scontrol show job $SLURM_JOB_ID | grep -oP 'TimeLimit=\K\S+')"
echo "================================================================================"
nvidia-smi

# Fix dependency conflicts and install required packages
echo "Cleaning and installing dependencies..."
python3 -m pip install -q pillow transformers==5.5.0

# Start inference
echo "Starting Qwen 3.5 Inference..."
python3 2_inference_code/qwen.py --data_dir=$DATAPATH
	
