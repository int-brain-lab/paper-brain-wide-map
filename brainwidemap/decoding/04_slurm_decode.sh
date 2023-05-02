#!/bin/bash
#SBATCH --job-name=decoding
#SBATCH --output=/scratch/users/bensonb/international-brain-lab/paper-brain-wide-map/brainwidemap/logs/slurm/dw_bwmapr_01_1_.%a.out
#SBATCH --error=/scratch/users/bensonb/international-brain-lab/paper-brain-wide-map/brainwidemap/logs/slurm/dw_bwmapr_01_1_.%a.err
#SBATCH --partition=normal
#SBATCH --array=3301-3521
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bensonb@stanford.edu
#SBATCH --time=48:00:00

#extracting settings from $SLURM_ARRAY_TASK_ID
echo slurm_task $SLURM_ARRAY_TASK_ID

export PYTHONPATH="$PWD":$PYTHONPATH
# calling script

echo
# change to conda  => which python
python 04_decode_single_session.py $SLURM_ARRAY_TASK_ID

# example of running a subset of the BWM dataset with only the given subjects 
# You can add any number of subjects to filter the BWM dataset: 10 are shown
#python 04_decode_single_session.py $SLURM_ARRAY_TASK_ID DY_011 CSHL059 SWC_054 KS042 ZFM-02373 ibl_witten_26 CSH_ZAD_029 UCLA037 SWC_038 NYU-39
#python 04_decode_single_session.py $SLURM_ARRAY_TASK_ID DY_011 CSHL059 SWC_054
#python 04_decode_single_session.py $SLURM_ARRAY_TASK_ID 5bcafa14-71cb-42fa-8265-ce5cda1b89e0

