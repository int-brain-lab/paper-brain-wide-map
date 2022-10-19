# Standard library
import os
import pickle
from datetime import date
from pathlib import Path

# Third party libraries
import numpy as np
import sklearn.linear_model as skl
from sklearn.model_selection import GridSearchCV

# IBL libraries
import neurencoding.linear as lm
import neurencoding.utils as mut

# Brainwide repo imports
from brainwidemap.encoding.params import GLM_CACHE, GLM_FIT_PATH
from brainwidemap.encoding.utils import make_batch_slurm

# SLURM params
BATCHFILE = '/home/gercek/bwm_impostor_glm.sh'
JOBNAME = 'bwm_GLMs_impostor'
PARTITION = 'shared-cpu'
TIME = '06:30:00'
CONDAPATH = Path('/home/gercek/mambaforge/')
ENVNAME = 'iblenv'
LOGPATH = Path('/home/gercek/worker-logs/')
JOB_CORES = 4
MEM = '12GB'
SUBMIT_BATCH = False


# Model parameters
def tmp_binf(t):
    return np.ceil(t / params['binwidth']).astype(int)


# Define fitting parameters for workers
params = {
    'binwidth': 0.02,
    'iti_prior': [-0.4, -0.1],
    'fmove_offset': -0.2,
    'wheel_offset': -0.3,
    'contnorm': 5.,
    'reduce_wheel_dim': True,
    'dataset_fn': '2022-10-17_dataset_metadata.pkl',
    'impostor_fn': '2022-01-19_impostor_df.pkl',  # TODO: Regen impostor sessions
    'model': lm.LinearGLM,
    'alpha_grid': {
        'alpha': np.logspace(-2, 1.5, 25)
    },
    'contiguous': False,
    'impostor': True,
    'prior_estimate': False,
    'n_impostors': 500,
}

# Estimator relies on alpha grid in case of GridSearchCV, needs to be defined after main params
# Likewise for bases functions which rely on binwidth.
params['bases'] = {
    'stim': mut.raised_cosine(0.4, 5, tmp_binf),
    'feedback': mut.raised_cosine(0.4, 5, tmp_binf),
    'wheel': mut.raised_cosine(0.3, 3, tmp_binf),
    'fmove': mut.raised_cosine(0.2, 3, tmp_binf),
}

params['estimator'] = GridSearchCV(skl.Ridge(), params['alpha_grid'])
params['verif_binwidth'] = params['binwidth']

# Output parameters file for workers
currdate = str(date.today())
parpath = Path(GLM_FIT_PATH).joinpath(f'{currdate}_glm_fit_pars.pkl')
datapath = Path(GLM_CACHE).joinpath(params['dataset_fn'])
impostorpath = Path(GLM_CACHE).joinpath(params['impostor_fn'])
with open(parpath, 'wb') as fw:
    pickle.dump(params, fw)
with open(datapath, 'rb') as fo:
    njobs = pickle.load(fo)['dataset_filenames'].index.max()
print("Parameters file located at:", parpath)
print("Dataset file used:", datapath)

# Generate batch script
make_batch_slurm(BATCHFILE,
                 Path(__file__).parents[1].joinpath('cluster_worker.py'),
                 job_name=JOBNAME,
                 partition=PARTITION,
                 time=TIME,
                 condapath=CONDAPATH,
                 envname=ENVNAME,
                 logpath=LOGPATH,
                 cores_per_job=JOB_CORES,
                 memory=MEM,
                 array_size=f'1-{njobs}',
                 f_args=[
                     str(datapath), str(parpath),
                     r'${SLURM_ARRAY_TASK_ID}', currdate,
                     '--impostor', str(impostorpath)
                 ])

# If SUBMIT_BATCH, then actually execute the batch job
if SUBMIT_BATCH:
    os.system(f'sbatch {BATCHFILE}')
else:
    print(f'Batch file generated at {BATCHFILE};'
          ' user must submit it themselves. Good luck!')
