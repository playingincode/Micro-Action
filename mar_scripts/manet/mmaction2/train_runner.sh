#!/bin/bash
#OAR -p gpu='YES' and host='nefgpu56.inria.fr'
#OAR -l /nodes=1/gpunum=1,walltime=72:00:00
#OAR --name second_expert_running_for_mpii_again
#OAR --stdout nef_logs/%jobname%.%jobid%.out
#OAR --stderr nef_logs/%jobname%.%jobid%.err



module load conda/2020.48-python3.8 cuda/12.2 gcc/9.2.0

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
source activate openmmlab|| { echo "Conda environment not found"; exit 1; }

# Display python version and path
python --version
which python

# Add to PATH
export PATH="/home/npoddar/:$PATH"

# Run ffmpeg and nvidia-smi to check availability
ffmpeg
nvidia-smi || { echo "NVIDIA driver issue"; exit 1; }

export PATH=/pytorch_env/bin:$PATH

export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u tools/train.py configs/recognition/manet/manet.py --seed=0 --deterministic --resume-from /data/stars/user/npoddar/MANET_original_six_classes/Micro-Action/mar_scripts/manet/mmaction2/work_dirs/all_experts_without_cross_attention_with_trainable_weighter/epoch_20.pth