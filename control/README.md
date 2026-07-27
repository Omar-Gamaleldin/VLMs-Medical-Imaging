# Control

This folder contains the main inference scripts used to run the medical imaging benchmark with different model and prompting setups.

## Contents

- `gemma4_thinking.py` / `gemma_4_thinking.sh` - Gemma 4 thinking-mode inference.
- `gemma4_intruct.py` / `gemma4_instruct.sh` - Gemma 4 instruction-style inference.
- `medgemma.py` / `medgemma.sh` - MedGemma inference entry points.
- `qwen3.5_thinking.py` / `qwen_3.5_thinking.sh` - Qwen 3.5 thinking-mode inference.
- `qwen3.5_instruct.py` / `qwen_3.5.sh` - Qwen 3.5 instruction-style inference.
- `results/` - evaluation scripts and stored outputs.

The scripts expect the benchmark dataset layout used by the project and write run outputs into the corresponding results directories Each script contains all required sbatch parameters besides time and gpu partition which will be decided by user at when using sbatch. 

For all the bash scripts you must first change the workspace directory to your own and specify where to find the dataseet using the`--data-dir` argument for the python script