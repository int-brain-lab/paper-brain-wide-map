"""
Utility functions for the paper-brain-wide-map repository
"""
# Standard library
import logging
from pathlib import Path

# Third party libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

# IBL libraries
from brainbox.io.one import SessionLoader
from ibllib.atlas import BrainRegions
from iblutil.numerical import ismember
from one.api import ONE
from brainwidemap.encoding.timeseries import TimeSeries, sync

_logger = logging.getLogger("brainwide")


def get_impostor_df(
    subject, one, ephys=False, tdf_kwargs={}, progress=False, ret_template=False, max_sess=10000
):
    """
    Produce an impostor DF for a given subject, i.e. a dataframe which joins all trials from
    ephys sessions for that mouse. Will have an additional column listing the source EID of each
    trial.

    Parameters
    ----------
    subject : str
        Subject nickname
    one : oneibl.one.ONE instance
        ONE instance to use for data loading
    ephys : bool
        Whether to use ephys sessions (True) or behavior sessions (False)
    tdf_kwargs : dict
        Dictionary of keyword arguments for brainbox.io.one.load_trials_df
    progress : bool
        Whether or not to display a progress bar of sessions processed
    ret_template : bool
        Whether to return the template session identity (for ephys sessions)
    """
    if ephys:
        sessions = one.alyx.rest(
            "insertions",
            "list",
            django="session__project__name__icontains,"
            "ibl_neuropixel_brainwide_01,"
            "session__subject__nickname__icontains,"
            f"{subject},"
            "session__task_protocol__icontains,"
            "_iblrig_tasks_ephysChoiceWorld,"
            "session__qc__lt,50,"
            "session__extended_qc__behavior,1",
        )
        eids = [item["session_info"]["id"] for item in sessions]
    else:
        eids = one.search(
            project="ibl_neuropixel_brainwide_01",
            task_protocol="biasedChoiceWorld",
            subject=[subject] if len(subject) > 0 else [],
        )
    if len(eids) > max_sess:
        rng = np.random.default_rng()
        eids = rng.choice(eids, size=max_sess, replace=False)

    dfs = []
    timing_vars = [
        "feedback_times",
        "goCue_times",
        "stimOn_times",
        "trial_start",
        "trial_end",
        "firstMovement_times",
    ]
    for eid in tqdm(eids, desc="Eid :", leave=False, disable=not progress):
        try:
            tmpdf = load_trials_df(eid, one=one, **tdf_kwargs)
        except Exception as e:
            _logger.warning(f"eid {eid} df load failed with exception {e}.")
            continue
        if ret_template and ephys:
            det = one.get_details(eid, full=True)
            tmpdf["template"] = det["json"]["SESSION_ORDER"][det["json"]["SESSION_IDX"]]
        tmpdf[timing_vars] = tmpdf[timing_vars].subtract(tmpdf["trial_start"], axis=0)
        tmpdf["orig_eid"] = eid
        tmpdf["duration"] = tmpdf["trial_end"] - tmpdf["trial_start"]
        dfs.append(tmpdf)
    return pd.concat(dfs).reset_index()


def remap(ids, source="Allen", dest="Beryl", output="acronym", br=BrainRegions()):
    _, inds = ismember(ids, br.id[br.mappings[source]])
    ids = br.id[br.mappings[dest][inds]]
    if output == "id":
        return br.id[br.mappings[dest][inds]]
    elif output == "acronym":
        return br.get(br.id[br.mappings[dest][inds]])["acronym"]


def get_id(acronym, brainregions=BrainRegions()):
    return brainregions.id[np.argwhere(brainregions.acronym == acronym)[0, 0]]


def sessions_with_region(acronym, one=None):
    if one is None:
        one = ONE()
    query_str = (
        f"channels__brain_region__acronym__icontains,{acronym},"
        "probe_insertion__session__project__name__icontains,ibl_neuropixel_brainwide_01,"
        "probe_insertion__session__qc__lt,50,"
        "~probe_insertion__json__qc,CRITICAL"
    )
    traj = one.alyx.rest(
        "trajectories", "list", provenance="Ephys aligned histology track", django=query_str
    )
    eids = np.array([i["session"]["id"] for i in traj])
    sessinfo = [i["session"] for i in traj]
    probes = np.array([i["probe_name"] for i in traj])
    return eids, sessinfo, probes


def melt_masterscores_stepwise(masterscores, n_cov, exclude_cols=[]):
    tmp_master = masterscores.reset_index()
    rawpoints = tmp_master.melt(
        value_vars=(vv := [f"{i}cov_diff" for i in range(1, n_cov + 1)]),
        id_vars=tmp_master.columns.difference([*vv, *exclude_cols]),
    )
    rawpoints["covname"] = rawpoints.apply(
        lambda x: x[str(int(x.variable[0])) + "cov_name"], axis=1
    )
    rawpoints["position"] = rawpoints.apply(lambda x: int(x.variable[0]), axis=1)
    rawpoints = rawpoints.drop(columns=[f"{i}cov_name" for i in range(1, n_cov + 1)])
    rawpoints["percentile"] = rawpoints.groupby(["covname", "region"]).value.rank(pct=True)
    return rawpoints


def make_batch_slurm(
    filename,
    scriptpath,
    job_name="brainwide",
    partition="shared-cpu",
    time="01:00:00",
    condapath=Path("~/mambaforge/"),
    envname="iblenv",
    logpath=Path("~/worker-logs/"),
    cores_per_job=4,
    memory="16GB",
    array_size="1-100",
    f_args=[],
):
    fw = open(filename, "wt")
    fw.write("#!/bin/sh\n")
    fw.write(f"#SBATCH --job-name={job_name}\n")
    fw.write(f"#SBATCH --time={time}\n")
    fw.write(f"#SBATCH --partition={partition}\n")
    fw.write(f"#SBATCH --array={array_size}\n")
    fw.write(f"#SBATCH --output={logpath.joinpath(job_name)}.%a.out\n")
    fw.write("#SBATCH --ntasks=1\n")
    fw.write(f"#SBATCH --cpus-per-task={cores_per_job}\n")
    fw.write(f"#SBATCH --mem={memory}\n")
    fw.write("\n")
    fw.write(f"source {condapath}/bin/activate\n")
    fw.write(f"conda activate {envname}\n")
    fw.write(f'python {scriptpath} {" ".join(f_args)}\n')
    fw.close()
    return


def make_batch_slurm_singularity(
    filebase,
    scriptpath,
    job_name="brainwide",
    partition="shared-cpu",
    time="01:00:00",
    singularity_modules=["GCC/9.3.0", "Singularity/3.7.3-Go-1.14"],
    container_image="~/iblcore.sif",
    mount_paths={},
    img_condapath="/opt/conda/",
    img_envname="iblenv",
    local_pathadd="~/Projects/paper-brain-wide-map/",
    logpath=Path("~/worker-logs/"),
    cores_per_job=4,
    memory="16GB",
    array_size="1-100",
    f_args=[],
):
    """
    Make a batch file for slurm to run a singularity container.

    Filebase is used to generate a _batch.sh and _workerscript.sh file, the former of which
    is submitted via sbatch, and the latter of which is run by the worker nodes.

    Parameters
    ----------
    filebase : str
        The base name of the batch file to be generated.
    scriptpath : str
        The path to the python script to be run by the worker nodes.
    job_name : str, optional
        The name of the job, by default "brainwide"
    partition : str, optional
        The partition to run on, by default "shared-cpu"
    time : str, optional
        The time limit for the job, by default "01:00:00"
    singularity_modules : list, optional
        The modules to load for singularity, by default ["GCC/9.3.0", "Singularity/3.7.3-Go-1.14"]
    container_image : str, optional
        The path to the singularity container image, by default "~/iblcore.sif"
    mount_paths : dict, optional
        A dictionary of path pairs to mount in the container. Usually singularity will by
        default mount the home directory of the user, but if additional directories are needed
        specify them here. The keys are the paths on the host machine, and the values are the
        paths in the container. By default {}.
    img_condapath : str, optional
        The path to conda in the container, by default "/opt/conda/"
    img_envname : str, optional
        The name of the conda environment to activate in the container, by default "iblenv"
    local_pathadd : str, optional
        The path to add to the PYTHONPATH in the container, which will be seen by the python
        executable as being part of pythons normal path. Similar to using conda develop.
        by default "~/Projects/paper-brain-wide-map/"
    logpath : Path, optional
        The path to write the log files to, by default Path("~/worker-logs/")
    cores_per_job : int, optional
        The number of cores to request per job, by default 4
    memory : str, optional
        The amount of memory to request per job, by default "16GB"
    array_size : str, optional
        The size of the slurm job array to run. These are the jobs that will be run in parallel
        and the index of the job for a current worker node is available as the environment
        variable $SLURM_ARRAY_TASK_ID, which can be used in f_args. By default "1-100"
    f_args : list, optional
        A list of string arguments to pass to the python script. These will be passed concatenated
        by spaces. By default []
    """
    batchfile = filebase + "_batch.sh"
    workerscript = filebase + "_workerscript.sh"
    fw = open(batchfile, "wt")
    fw.write("#!/bin/sh\n")
    fw.write(f"#SBATCH --job-name={job_name}\n")
    fw.write(f"#SBATCH --time={time}\n")
    fw.write(f"#SBATCH --partition={partition}\n")
    fw.write(f"#SBATCH --array={array_size}\n")
    fw.write(f"#SBATCH --output={logpath.joinpath(job_name)}.%a.out\n")
    fw.write("#SBATCH --ntasks=1\n")
    fw.write(f"#SBATCH --cpus-per-task={cores_per_job}\n")
    fw.write(f"#SBATCH --mem={memory}\n")
    fw.write("\n")
    fw.write(f"module load {' '.join(singularity_modules)}\n")
    bindstr = "" if len(mount_paths) == 0 else "-B "
    mountpairs = ",".join([f"{k}:{v}" for k, v in mount_paths.items()])
    fw.write(
        f"singularity run {bindstr} {mountpairs} {container_image} /bin/bash {workerscript}"
    )
    fw.close()

    fw = open(workerscript, "wt")
    fw.write("#!/bin/bash\n")
    fw.write(f"source {img_condapath}/bin/activate\n")
    fw.write(f"conda activate {img_envname}\n")
    fw.write(f"export PYTHONPATH={local_pathadd}\n")
    fw.write("python " + scriptpath + " " + " ".join(f_args) + "\n")
    return


def load_trials_df(
    eid,
    one=None,
    t_before=0.0,
    t_after=0.2,
    ret_wheel=False,
    ret_abswheel=False,
    wheel_binsize=0.02,
    addtl_types=[],
    trials_mask=None,
):
    """
    Generate a pandas dataframe of per-trial timing information about a given session.
    Each row in the frame will correspond to a single trial, with timing values indicating timing
    session-wide (i.e. time in seconds since session start). Can optionally return a resampled
    wheel velocity trace of either the signed or absolute wheel velocity.

    The resulting dataframe will have a new set of columns, trial_start and trial_end, which define
    via t_before and t_after the span of time assigned to a given trial.
    (useful for bb.modeling.glm)

    Parameters
    ----------
    eid : [str, UUID, Path, dict]
        Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
    one : one.api.OneAlyx, optional
        one object to use for loading. Will generate internal one if not used, by default None
    t_before : float, optional
        Time before stimulus onset to include for a given trial, as defined by the trial_start
        column of the dataframe. If zero, trial_start will be identical to stimOn, by default 0.
    t_after : float, optional
        Time after feedback to include in the trail, as defined by the trial_end
        column of the dataframe. If zero, trial_end will be identical to feedback, by default 0.
    ret_wheel : bool, optional
        Whether to return the time-resampled wheel velocity trace, by default False
    ret_abswheel : bool, optional
        Whether to return the time-resampled absolute wheel velocity trace, by default False
    wheel_binsize : float, optional
        Time bins to resample wheel velocity to, by default 0.02
    addtl_types : list, optional
        List of additional types from an ONE trials object to include in the dataframe. Must be
        valid keys to the dict produced by one.load_object(eid, 'trials'), by default empty.
    trials_mask : list, optional
        List of trial indices to include in the dataframe. If None, all trials will be included.

    Returns
    -------
    pandas.DataFrame
        Dataframe with trial-wise information. Indices are the actual trial order in the original
        data, preserved even if some trials do not meet the maxlen criterion. As a result will not
        have a monotonic index. Has special columns trial_start and trial_end which define start
        and end times via t_before and t_after
    """
    if not one:
        one = ONE()

    if ret_wheel and ret_abswheel:
        raise ValueError("ret_wheel and ret_abswheel cannot both be true.")

    # Define which datatypes we want to pull out
    trialstypes = [
        "choice",
        "probabilityLeft",
        "feedbackType",
        "feedback_times",
        "contrastLeft",
        "contrastRight",
        "goCue_times",
        "stimOn_times",
    ]
    trialstypes.extend(addtl_types)

    loader = SessionLoader(one=one, eid=eid)
    loader.load_session_data(pose=False, motion_energy=False, pupil=False, wheel=True)
    loader.load_wheel(fs=1000, corner_frequency=70, order=8)

    trials = loader.trials
    starttimes = trials.stimOn_times
    endtimes = trials.feedback_times
    trialsdf = trials[trialstypes]
    if trials_mask is not None:
        trialsdf = trialsdf.loc[trials_mask]
    trialsdf["trial_start"] = trialsdf["stimOn_times"] - t_before
    trialsdf["trial_end"] = trialsdf["feedback_times"] + t_after
    tdiffs = trialsdf["trial_end"] - np.roll(trialsdf["trial_start"], -1)
    if np.any(tdiffs[:-1] > 0):
        logging.warning(
            f"{sum(tdiffs[:-1] > 0)} trials overlapping due to t_before and t_after "
            "values. Try reducing one or both!"
        )
    if not ret_wheel and not ret_abswheel:
        return trialsdf

    wheel = one.load_object(eid, "wheel", collection="alf")
    whlpos, whlt = wheel.position, wheel.timestamps
    starttimes = trialsdf["trial_start"]
    endtimes = trialsdf["trial_end"]
    wh_endlast = 0
    trials = []
    for (start, end) in np.vstack((starttimes, endtimes)).T:
        wh_startind = np.searchsorted(whlt[wh_endlast:], start) + wh_endlast
        wh_endind = np.searchsorted(whlt[wh_endlast:], end, side="right") + wh_endlast + 4
        wh_endlast = wh_endind
        tr_whlpos = whlpos[wh_startind - 1 : wh_endind + 1]
        tr_whlt = whlt[wh_startind - 1 : wh_endind + 1] - start
        tr_whlt[0] = 0.0  # Manual previous-value interpolation
        whlseries = TimeSeries(tr_whlt, tr_whlpos, columns=["whlpos"])
        whlsync = sync(wheel_binsize, timeseries=whlseries, interp="previous")
        trialstartind = np.searchsorted(whlsync.times, 0)
        trialendind = np.ceil((end - start) / wheel_binsize).astype(int)
        trpos = whlsync.values[trialstartind : trialendind + trialstartind]
        whlvel = trpos[1:] - trpos[:-1]
        whlvel = np.insert(whlvel, 0, 0)
        if np.abs((trialendind - len(whlvel))) > 0:
            raise IndexError("Mismatch between expected length of wheel data and actual.")
        if ret_wheel:
            trials.append(whlvel)
        elif ret_abswheel:
            trials.append(np.abs(whlvel))
    trialsdf["wheel_velocity"] = trials
    return trialsdf
