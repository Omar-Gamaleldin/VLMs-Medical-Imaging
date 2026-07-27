# GEPA

This folder contains the GEPA optimization workflow used to improve model behavior on the benchmark.

## Contents

- `gepa_optimize.py` - runs the optimization pipeline and coordinates scoring, reflection, and data loading.
- `gepa.sh` - shell entry point for launching the optimization run.

The code is set up to work with local model servers and the benchmark data stored under `control/`.