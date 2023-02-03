#!/bin/bash
#SBATCH --job-name=decoding
#SBATCH --output=/scratch/users/bensonb/international-brain-lab/paper-brain-wide-map/brainwidemap/logs/slurm/dw_bwmchecknulljan_20_1_.%a.out
#SBATCH --error=/scratch/users/bensonb/international-brain-lab/paper-brain-wide-map/brainwidemap/logs/slurm/dw_bwmchecknulljan_20_1_.%a.err
#SBATCH --partition=normal
#SBATCH --array=1-999
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bensonb@stanford.edu
#SBATCH --time=24:00:00

#extracting settings from $SLURM_ARRAY_TASK_ID
echo slurm_task $SLURM_ARRAY_TASK_ID

export PYTHONPATH="$PWD":$PYTHONPATH
# calling script

echo
# change to conda  => which python
#python 03_decode_single_session.py $SLURM_ARRAY_TASK_ID

# example of running a subset of the BWM dataset with only the given subjects 
# You can add any number of subjects to filter the BWM dataset: 10 are shown
python 03_decode_single_session.py $SLURM_ARRAY_TASK_ID DY_011 CSHL059 SWC_054 KS042 ZFM-02373 ibl_witten_26 CSH_ZAD_029 UCLA037 SWC_038 NYU-39

