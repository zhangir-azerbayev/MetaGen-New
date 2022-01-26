#!/bin/bash 
#SBATCH --mem=8g
#SBATCH --output=results/gtv_retinanet/%a.out
#SBATCH --error=results/gtv_retinanet/error%a.error
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH -t 1:30:00

module load miniconda
conda activate mg 

python3 scene_baseline.py 
