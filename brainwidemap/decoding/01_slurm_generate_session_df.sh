#!/bin/bash
#SBATCH --job-name=decsummary
#SBATCH --output=/scratch/users/bensonb/international-brain-lab/paper-brain-wide-map/brainwidemap/logs/slurm/decodinggendf.%A.out
#SBATCH --error=/scratch/users/bensonb/international-brain-lab/paper-brain-wide-map/brainwidemap/logs/slurm/decodinggendf.%A.err
#SBATCH --partition=normal
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bensonb@stanford.edu
#SBATCH --time=4:00:00

export PYTHONPATH="$PWD":$PYTHONPATH
# calling script

echo
# change to conda  => which python
python 01_generate_session_df.py
