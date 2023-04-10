# Standard library
import logging
import os
import pickle
from datetime import datetime as dt
from pathlib import Path

# Third party libraries
import dask
import numpy as np
import pandas as pd
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

# IBL libraries
import brainbox.io.one as bbone
from iblutil.util import Bunch
from iblutil.numerical import ismember
from one.api import ONE
from brainwidemap.encoding.params import GLM_CACHE
from brainwidemap.encoding.utils import load_trials_df
from brainwidemap.bwm_loading import load_trials_and_mask, bwm_query, bwm_units

_logger = logging.getLogger("brainwide")


def load_regressors(
    session_id,
    pid,
    t_before=0.0,
    t_after=0.0,
    binwidth=0.02,
    abswheel=False,
    clu_criteria="bwm",
    one=None,
):
    """
    Load in regressors for given session and probe. Returns a dictionary with the following keys:

    Parameters
    ----------
    session_id : str
        EID of the session to load
    pid : str
        Probe ID to load associated with the session
    t_before : float, optional
        Time before stimulus onset to include in output trial_start column of df, by default 0.
    t_after : float, optional
        Time after feedback to include in output trial_end column of df, by default 0.
    binwidth : float, optional
        Binwidth for wheel signal. Needs to match that of GLM, by default 0.02
    abswheel : bool, optional
        Load in wheel speed instead of velocity, by default False
    ret_qc : bool, optional
        Whether to recompute cluster metrics and return a full dataframe of the result,
        by default False
    clu_criteria : str, optional
        Criteria for saving clusters, 'all' for all units, 'bwm' for criteria matching that of
        brain-wide map (all metrics passing). No others supported for now., by default 'bwm'
    one : ONE, optional
        Instance of ONE, by default None

    Returns
    -------
    trialsdf, spk_times, spk_clu, clu_regions, clu_qc, clu_df, clu_qc (optional)
        Output regressors for GLM
    """
    one = ONE() if one is None else one

    _, mask = load_trials_and_mask(one=one, eid=session_id)
    mask = mask.index[np.nonzero(mask.values)]
    trialsdf = load_trials_df(
        session_id,
        t_before=t_before,
        t_after=t_after,
        wheel_binsize=binwidth,
        ret_abswheel=abswheel,
        ret_wheel=not abswheel,
        addtl_types=["firstMovement_times"],
        one=one,
        trials_mask=mask,
    )

    clusters = {}
    ssl = bbone.SpikeSortingLoader(one=one, pid=pid)
    origspikes, tmpclu, channels = ssl.load_spike_sorting()
    if "metrics" not in tmpclu:
        tmpclu["metrics"] = np.ones(tmpclu["channels"].size)
    clusters[pid] = ssl.merge_clusters(origspikes, tmpclu, channels)
    clu_df = pd.DataFrame(clusters[pid]).set_index(["cluster_id"])
    clu_df["pid"] = pid

    if clu_criteria == "bwm":
        allunits = (
            bwm_units(one=one)
            .rename(columns={"cluster_id": "clu_id"})
            .set_index(["eid", "pid", "clu_id"])
        )
        keepclu = clu_df.index.intersection(allunits.loc[session_id, pid, :].index)
    elif clu_criteria == "all":
        keepclu = clu_df.index
    else:
        raise ValueError("clu_criteria must be 'bwm' or 'all'")

    clu_df = clu_df.loc[keepclu]
    keepmask = np.isin(origspikes.clusters, keepclu)
    spikes = Bunch({k: v[keepmask] for k, v in origspikes.items()})
    sortinds = np.argsort(spikes.times)
    spk_times = spikes.times[sortinds]
    spk_clu = spikes.clusters[sortinds]
    clu_regions = clusters[pid].acronym
    return trialsdf, spk_times, spk_clu, clu_regions, clu_df


def cache_regressors(
    subject, session_id, pid, regressor_params, trialsdf, spk_times, spk_clu, clu_regions, clu_df
):
    """
    Take outputs of load_regressors() and cache them to disk in the folder defined in the params.py
    file in this repository, using a nested subject -> session folder structure.

    If an existing file in the directory already contains identical data, will not write a new file
    and instead return the existing filenames.

    Returns the metadata filename and regressors filename.
    """
    subpath = Path(GLM_CACHE).joinpath(subject)
    if not subpath.exists():
        os.mkdir(subpath)
    sesspath = subpath.joinpath(session_id)
    if not sesspath.exists():
        os.mkdir(sesspath)
    fnbase = DATE
    metadata_fn = sesspath.joinpath(fnbase + f"_{pid}_metadata.pkl")
    data_fn = sesspath.joinpath(fnbase + f"_{pid}_regressors.pkl")
    regressors = {
        "trialsdf": trialsdf,
        "spk_times": spk_times,
        "spk_clu": spk_clu,
        "clu_regions": clu_regions,
        "clu_df": clu_df,
    }
    metadata = {"subject": subject, "session_id": session_id, "pid": pid, **regressor_params}

    with open(metadata_fn, "wb") as fw:
        pickle.dump(metadata, fw)
    with open(data_fn, "wb") as fw:
        pickle.dump(regressors, fw)
    return metadata_fn, data_fn


@dask.delayed
def delayed_loadsave(subject, session_id, pid, params):
    outputs = load_regressors(session_id, pid, **params)
    return cache_regressors(subject, session_id, pid, params, *outputs)


# Parameters
SESS_CRITERION = "resolved-behavior"
DATE = str(dt.now().date())
T_BEF = 0.6
T_AFT = 0.6
BINWIDTH = 0.02
ABSWHEEL = False
QC = True
EPHYS_IMPOSTOR = False
CLU_CRITERIA = "bwm"
# End parameters

# Construct params dict from above
params = {
    "t_before": T_BEF,
    "t_after": T_AFT,
    "binwidth": BINWIDTH,
    "abswheel": ABSWHEEL,
    "clu_criteria": CLU_CRITERIA,
}

one = ONE()
dataset_futures = []

sessdf = bwm_query().set_index("pid")

for pid, rec in sessdf.iterrows():
    subject = rec.subject
    eid = rec.eid
    save_future = delayed_loadsave(subject, eid, pid, params)
    dataset_futures.append([subject, eid, pid, save_future])

N_CORES = 2
cluster = SLURMCluster(
    cores=N_CORES,
    memory="32GB",
    processes=1,
    queue="shared-cpu",
    walltime="01:15:00",
    log_directory="/home/gercek/dask-worker-logs",
    interface="ib0",
    worker_extra_args=["--lifetime", "60m", "--lifetime-stagger", "10m"],
    job_cpu=N_CORES,
    job_script_prologue=[
        f"export OMP_NUM_THREADS={N_CORES}",
        f"export MKL_NUM_THREADS={N_CORES}",
        f"export OPENBLAS_NUM_THREADS={N_CORES}",
    ],
)
cluster.scale(40)
client = Client(cluster)

tmp_futures = [client.compute(future[3]) for future in dataset_futures]
# impostor_df = get_impostor_df(
#     '',
#     one,
#     ephys=EPHYS_IMPOSTOR,
#     tdf_kwargs={k: v for k, v in params.items() if k not in ['resolved_alignment', 'ret_qc']},
#     ret_template=True)

# Run below code AFTER futures have finished!
dataset = [
    {
        "subject": x[0],
        "eid": x[1],
        "probes": x[2],
        "meta_file": tmp_futures[i].result()[0],
        "reg_file": tmp_futures[i].result()[1],
    }
    for i, x in enumerate(dataset_futures)
    if tmp_futures[i].status == "finished"
]

# Run below if run finished but you lost the python session
regfns = list(Path(GLM_CACHE).rglob(f"*{DATE}*_regressors.pkl"))

dataset = [
    {
        "subject": path.parts[-3],
        "eid": path.parts[-2],
        "probes": path.parts[-1].split("_")[1],
        "meta_file": str(path).split("_regressors.pkl")[0] + "_metadata.pkl",
        "reg_file": path,
    }
    for path in regfns
]
dataset = pd.DataFrame(dataset)

outdict = {"params": params, "dataset_filenames": dataset}
with open(Path(GLM_CACHE).joinpath(DATE + "_dataset_metadata.pkl"), "wb") as fw:
    pickle.dump(outdict, fw)
