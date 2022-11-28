#!/bin/bash
#SBATCH --job-name=decoding
#SBATCH --output=/scratch/users/bensonb/international-brain-lab/paper-brain-wide-map/brainwidemap/logs/slurm/dcmetaanalysis_1_.%a.out
#SBATCH --error=/scratch/users/bensonb/international-brain-lab/paper-brain-wide-map/brainwidemap/logs/slurm/dcmetaanalysis_1_.%a.err
#SBATCH --partition=normal
#SBATCH --array=1-10
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
