#!/bin/bash
#SBATCH --job-name=eval_one_from_min_pair
#SBATCH --open-mode=append
#SBATCH --output=/scratch/rh3015/MLLU/results/cola_eval.txt
#SBATCH --error=/home/rh3015/%j_%x.err
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
python eval_lang.py English ../data_preprocessing/clean_eng_evalset/test_sets ../models/finetuned_xlmr_cola
python eval_lang.py French ../CLAMS/French ../models/finetuned_xlmr_cola
python eval_lang.py Hebrew ../CLAMS/Hebrew ../models/finetuned_xlmr_cola
python eval_lang.py German ../CLAMS/German ../models/finetuned_xlmr_cola
python eval_lang.py Russian ../CLAMS/Russian ../models/finetuned_xlmr_cola
"
