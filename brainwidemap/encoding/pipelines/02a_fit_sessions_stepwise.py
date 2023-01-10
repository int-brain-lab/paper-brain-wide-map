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
from brainwidemap.encoding.utils import make_batch_slurm, make_batch_slurm_singularity

# SLURM params
BATCHFILE = "/home/gercek/bwm_stepwise_glm_leaveoneout.sh"
JOBNAME = "bwm_GLMs_LOO"
PARTITION = "shared-cpu"
TIME = "12:00:00"
SINGULARITY = True
if SINGULARITY:
    parfile = "paper-brain-wide-map/brainwidemap/encoding/params.py"
    SINGULARITY_MODULES = ["GCC/9.3.0", "Singularity/3.7.3-Go-1.14"]
    SINGULARITY_IMAGE = "~/iblcore.sif"
    SINGULARITY_BIND = {
        f"/home/gercek/Projects/{parfile}": f"/data/{parfile}",
    }
    SINGULARITY_CONDA = "/opt/conda"
    SINGULARITY_ENV = "iblenv"

CONDAPATH = Path("/home/gercek/mambaforge/")
ENVNAME = "iblenv"
LOGPATH = Path("/home/gercek/worker-logs/")
JOB_CORES = 32
MEM = "12GB"
SUBMIT_BATCH = False


# Model parameters
def tmp_binf(t):
    return np.ceil(t / params["binwidth"]).astype(int)


# Define fitting parameters for workers
params = {
    "binwidth": 0.02,
    "iti_prior": [-0.4, -0.1],
    "fmove_offset": -0.2,
    "wheel_offset": -0.3,
    "contnorm": 5.0,
    "reduce_wheel_dim": False,
    "dataset_fn": "2022-10-24_dataset_metadata.pkl",
    "model": lm.LinearGLM,
    "alpha_grid": {"alpha": np.logspace(-2, 1.5, 50)},
    "contiguous": False,
    "prior_estimate": False,
    "null": None,
    "n_impostors": 100,
    "seqsel_kwargs": {"direction": "backward", "n_features_to_select": 8},
    "seqselfit_kwargs": {"full_scores": True},
}

params["bases"] = {
    "stim": mut.nonlinear_rcos(0.4, 5, 0.1, tmp_binf),
    "feedback": mut.nonlinear_rcos(0.4, 5, 0.1, tmp_binf),
    "wheel": mut.nonlinear_rcos(0.3, 3, 0.05, tmp_binf)[::-1],
    "fmove": mut.nonlinear_rcos(0.2, 3, 0.05, tmp_binf)[::-1],
}
# Estimator relies on alpha grid in case of GridSearchCV, needs to be defined after main params
params["estimator"] = GridSearchCV(skl.Ridge(), params["alpha_grid"])

# Output parameters file for workers
currdate = str(date.today())
parpath = Path(GLM_FIT_PATH).joinpath(f"{currdate}_glm_fit_pars.pkl")
datapath = Path(GLM_CACHE).joinpath(params["dataset_fn"])
with open(parpath, "wb") as fw:
    pickle.dump(params, fw)
with open(datapath, "rb") as fo:
    njobs = pickle.load(fo)["dataset_filenames"].index.max()
print("Parameters file located at:", parpath)
print("Dataset file used:", datapath)

# Generate batch script
if SINGULARITY:
    make_batch_slurm_singularity(
        BATCHFILE,
        Path(__file__).parents[1].joinpath("cluster_worker.py"),
        job_name=JOBNAME,
        partition=PARTITION,
        time=TIME,
        singularity_modules=SINGULARITY_MODULES,
        container_image=SINGULARITY_IMAGE,
        img_condapath=SINGULARITY_CONDA,
        img_envname=SINGULARITY_ENV,
        local_pathadd=Path(__file__).parents[3],
        logpath=LOGPATH,
        cores_per_job=JOB_CORES,
        memory=MEM,
        array_size=f"1-{njobs}",
        f_args=[str(datapath), str(parpath), r"${SLURM_ARRAY_TASK_ID}", currdate],
    )
else:
    make_batch_slurm(
        BATCHFILE,
        Path(__file__).parents[1].joinpath("cluster_worker.py"),
        job_name=JOBNAME,
        partition=PARTITION,
        time=TIME,
        condapath=CONDAPATH,
        envname=ENVNAME,
        logpath=LOGPATH,
        cores_per_job=JOB_CORES,
        memory=MEM,
        array_size=f"1-{njobs}",
        f_args=[str(datapath), str(parpath), r"${SLURM_ARRAY_TASK_ID}", currdate],
    )

# If SUBMIT_BATCH, then actually execute the batch job
if SUBMIT_BATCH:
    os.system(f"sbatch {BATCHFILE}")
else:
    print(f"Batch file generated at {BATCHFILE}; user must submit it themselves. Good luck!")
