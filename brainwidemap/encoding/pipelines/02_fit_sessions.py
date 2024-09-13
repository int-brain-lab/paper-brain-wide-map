# Standard library
import argparse
import pickle
from datetime import date
from pathlib import Path

# IBL libraries
import neurencoding.linear as lm
import neurencoding.utils as mut

# Third party libraries
import numpy as np
import sklearn.linear_model as skl
from sklearn.model_selection import GridSearchCV

# Brainwide repo imports
from brainwidemap.encoding.params import GLM_CACHE, GLM_FIT_PATH
from brainwidemap.encoding.utils import make_batch_slurm_singularity

parser = argparse.ArgumentParser(
    description="Produce batch scripts for fitting GLMs"
    " on a SLURM cluster. Requires a compiled ibl singularity image."
    " This image can be generated using the image uploaded at"
    " docker://bdgercek/iblcore:latest. See your cluster admin"
    " for more info about using singularity images. Additionally,"
    " parameters for the actual GLM fitting are defined within the script itself."
    " The arguments passed to the script via this parser are only for cluster control."
    " If you would like to change parameters of the actual fit please adjust the contents"
    ' of the "parameters" section in the file.'
)
parser.add_argument(
    "--basefilepath",
    type=Path,
    default=Path("~/").expanduser().joinpath("jobscripts/bwm_stepwise_glm_leaveoneout"),
    help="Base filename for batch scripts",
)
parser.add_argument(
    "--jobname",
    type=str,
    default="bwm_GLMs_LOO",
)
parser.add_argument(
    "--partition",
    type=str,
    default="shared-cpu",
)
parser.add_argument(
    "--timelimit",
    type=str,
    default="12:00:00",
)
parser.add_argument(
    "--singularity_modules",
    type=str,
    nargs="+",
    default=[],
    help="Modules to load when using singularity containers.",
)
parser.add_argument(
    "--singularity_image",
    type=Path,
    default=Path("~/").expanduser().joinpath("iblcore.sif"),
    help="Path to singularity image with iblenv installed.",
)
parser.add_argument(
    "--singularity_conda",
    type=str,
    default="/opt/conda",
    help="Path to conda installation within singularity image.",
)
parser.add_argument(
    "--singularity_env",
    type=str,
    default="iblenv",
    help="Name of conda environment within singularity image.",
)
parser.add_argument(
    "--logpath",
    type=str,
    default=Path("~/").expanduser().joinpath("worker-logs/"),
    help="Path to store log files from workers.",
)
parser.add_argument(
    "--job_cores", type=int, default=32, help="Number of cores to request per job."
)
parser.add_argument("--mem", type=str, default="12GB", help="Memory to request per job.")

args = parser.parse_args()


# Model parameters
# The GLM constructor class requires a function that converts time to bin index, here we define it
# using the binwidth parameter created shortly.
def tmp_binf(t):
    return np.ceil(t / params["binwidth"]).astype(int)


######### PARAMETERS #########
params = {
    "binwidth": 0.02,
    "iti_prior": [-0.4, -0.1],
    "fmove_offset": -0.2,
    "wheel_offset": -0.3,
    "contnorm": 5.0,
    "reduce_wheel_dim": False,
    "dataset_fn": "2024-08-12_dataset_metadata.pkl",
    "model": lm.LinearGLM,
    "alpha_grid": {"alpha": np.logspace(-3, 2, 50)},
    "contiguous": False,
    "prior_estimate": False,
    "null": None,
    "n_impostors": 100,
    "seqsel_kwargs": {"direction": "backward", "n_features_to_select": 8},
    "seqselfit_kwargs": {"full_scores": True},
    "seed": 0,
    "rt_thresh": "session_median",
}

params["bases"] = {
    "stim": mut.nonlinear_rcos(0.4, 5, 0.1, tmp_binf),
    "feedback": mut.nonlinear_rcos(0.4, 5, 0.1, tmp_binf),
    "wheel": mut.nonlinear_rcos(0.3, 3, 0.05, tmp_binf)[::-1],
    "fmove": mut.nonlinear_rcos(0.2, 3, 0.05, tmp_binf)[::-1],
}
# Estimator relies on alpha grid in case of GridSearchCV, needs to be defined after main params
params["estimator"] = GridSearchCV(skl.Ridge(), params["alpha_grid"])
if "rt_thresh" in params:
    earlyrt_flag = "--earlyrt"
    latert_flag = "--latert"
    earlyrt_fn = "_early_rt"
else:
    earlyrt_flag = ""
    latert_flag = ""
    earlyrt_fn = ""

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
make_batch_slurm_singularity(
    str(args.basefilepath) + earlyrt_fn,
    str(Path(__file__).parents[1].joinpath("cluster_worker.py")),
    job_name=args.jobname,
    partition=args.partition,
    time=args.timelimit,
    singularity_modules=args.singularity_modules,
    container_image=args.singularity_image,
    img_condapath=args.singularity_conda,
    img_envname=args.singularity_env,
    local_pathadd=Path(__file__).parents[3],
    logpath=args.logpath,
    cores_per_job=args.job_cores,
    memory=args.mem,
    array_size=f"1-{njobs}",
    f_args=[earlyrt_flag, str(datapath), str(parpath), r"${SLURM_ARRAY_TASK_ID}", currdate],
)
if len(earlyrt_fn) > 0:
    make_batch_slurm_singularity(
        str(args.basefilepath) + "_late_rt",
        str(Path(__file__).parents[1].joinpath("cluster_worker.py")),
        job_name=args.jobname,
        partition=args.partition,
        time=args.timelimit,
        singularity_modules=args.singularity_modules,
        container_image=args.singularity_image,
        img_condapath=args.singularity_conda,
        img_envname=args.singularity_env,
        local_pathadd=Path(__file__).parents[3],
        logpath=args.logpath,
        cores_per_job=args.job_cores,
        memory=args.mem,
        array_size=f"1-{njobs}",
        f_args=[latert_flag, str(datapath), str(parpath), r"${SLURM_ARRAY_TASK_ID}", currdate],
    )

# If SUBMIT_BATCH, then actually execute the batch jo
print(
    f"Batch file generated at {str(args.basefilepath) + '_batch.sh'};"
    " user must submit it themselves. Good luck!"
)
