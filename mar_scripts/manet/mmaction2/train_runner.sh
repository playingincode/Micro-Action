#!/bin/bash
#OAR -p gpu='YES' and host='nefgpu46.inria.fr'
#OAR -l /nodes=1/gpunum=1,walltime=72:00:00
#OAR --name manet_videomaev2_sgp
#OAR --stdout nef_logs/%jobname%.%jobid%.out
#OAR --stderr nef_logs/%jobname%.%jobid%.err





# Activate conda environment

source activate openmlab_1|| { echo "Conda environment not found"; exit 1; }

# Display python version and path
export CUBLAS_WORKSPACE_CONFIG=:4096:8
which python
export PYTHONPATH=$(pwd):$PYTHONPATH

# Add to PATH



python -u tools/train.py configs/recognition/manet/manet.py --seed=0 --deterministic