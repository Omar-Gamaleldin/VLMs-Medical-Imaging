#! /usr/bin/env bash 

# Here you can add bash commands:
#SBATCH --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,/usr/lib64/slurm:/usr/lib64/slurm,/usr/lib64/libhwloc.so:/usr/lib64/libhwloc.so,/usr/lib64/libhwloc.so.15:/usr/lib64/libhwloc.so.15,/pfs/work9/workspace/scratch/ul_ekd37-gepa-optimization:/pfs/work9/workspace/scratch/ul_ekd37-gepa-optimization
#SBATCH --container-mount-home
#SBATCH --output=slurm_log/%x_%j.out  # Output file for job logs %x for job-name and %j for job-id
#SBATCH --container-name=vllm_qwen3.5
#SBATCH --gres=gpu:1
#SBATCH --job-name=Qwen3.5_Control

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

# python3 -m venv ~/venvs/qwen3.5_control
# source ~/venvs/qwen3.5_control/bin/activate

pip3 install pillow transformers==5.5.0

# Start inference
echo "Starting Qwen 3.5 Inference..."
python3 -u control/qwen3.5_instruct.py --data_dir=$DATAPATH
	
