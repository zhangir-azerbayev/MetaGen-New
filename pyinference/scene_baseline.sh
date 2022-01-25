#!/bin/bash 
#SBATCH --mem=4g
#SBATCH --output=results/baseline_retinanet/%a.out
#SBATCH --output=results/baseline_retinanet/error%a.error

module load miniconda
conda activate mg 

python3 scene_baseline.py 
