#!/bin/bash
#SBATCH --job-name=decoding
#SBATCH --output=/scratch/users/bensonb/international-brain-lab/paper-brain-wide-map/brainwidemap/logs/slurm/dw_bwmjan_18_3_.%a.out
#SBATCH --error=/scratch/users/bensonb/international-brain-lab/paper-brain-wide-map/brainwidemap/logs/slurm/dw_bwmjan_18_3_.%a.err
#SBATCH --partition=normal
#SBATCH --array=4801-5400
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bensonb@stanford.edu
#SBATCH --time=24:00:00

#extracting settings from $SLURM_ARRAY_TASK_ID
echo slurm_task $SLURM_ARRAY_TASK_ID

export PYTHONPATH="$PWD":$PYTHONPATH
# calling script

echo
# change to conda  => which python
python 03_decode_single_session.py $SLURM_ARRAY_TASK_ID

# example of running a subset of the BWM dataset with only 3 subjects, must uncomment lines in 03_... script
# python 03_decode_single_session.py $SLURM_ARRAY_TASK_ID DY_011 CSHL059 SWC_054
