#!/bin/bash
#SBATCH --job-name=roberta_hyperparameter_search
#SBATCH --open-mode=append
#SBATCH --output=/scratch/rh3015/MLLU/results/baseline_cola_res.txt
#SBATCH --error=/scratch/rh3015/MLLU/results/baseline_cola.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --mail-user=rh3015@nyu.edu
#SBATCH --mail-type=END  
#SBATCH --mail-type=BEGIN

singularity exec --nv --overlay $SCRATCH/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python finetune_cola.py
"
