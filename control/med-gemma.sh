#! /usr/bin/env bash 

# Here you can add bash commands:
#SBATCH --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,/usr/lib64/slurm:/usr/lib64/slurm,/usr/lib64/libhwloc.so:/usr/lib64/libhwloc.so,/usr/lib64/libhwloc.so.15:/usr/lib64/libhwloc.so.15,/pfs/work9/workspace/scratch/ul_ekd37-test-run:/pfs/work9/workspace/scratch/ul_ekd37-test-run
#SBATCH --container-mount-home
#SBATCH --output=slurm_log/%x_%j.out  # Output file for job logs %x for job-name and %j for job-id

# Get Data form Workspace to GPU Node -------------------------------
# set the workspace path. To figure out this path, you can run $(ws_find <workspace_name>) in the terminal (i.e. $(ws_find synthetic_data)) and copy the path
export WORKSPACE_PATH=/pfs/work9/workspace/scratch/ul_ekd37-test-run

# Extract compressed input dataset on local SSD
tar -C $TMPDIR/ -xvzf $WORKSPACE_PATH/dataset.tgz

export DATAPATH=$TMPDIR/dataset
#-------

# Other Stuff: 
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export DOME_CPUSET=$(scontrol show job $SLURM_JOB_ID --details | grep 'CPU_IDs' | awk '{print $2}' | awk -F= '{print $2}')
export DOME_MEMSET=$(scontrol show job $SLURM_JOB_ID --details | grep 'MemPerTres' | awk -F= '{print $2}' | awk -F: '{print $2}')
export XDG_CACHE_HOME=/tmp/cache
JOB_TIME_LIMIT=$(scontrol show job $SLURM_JOB_ID | grep -oP 'TimeLimit=\K\S+')
echo "================================================================================"
echo "Running on $(hostname). This worker has IP $(ip a | grep 134.60.70 | awk '{print $2}')"
echo "Running as $USER, with groups: $(groups)"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "The job time limit is: $JOB_TIME_LIMIT"
echo "Using CPU Cores $DOME_CPUSET"
echo "Using $DOME_MEMSET MiB RAM"
echo "================================================================================"
nvidia-smi


# Here you can add some pip installs
pip install torch pillow transformers accelerate

# Example to start my code: (Change that!)
python 2_inference_code/medgemma.py \
	--data_dir=$DATAPATH
