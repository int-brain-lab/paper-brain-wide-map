"""
Utility functions for the paper-brain-wide-map repository
"""
# Standard library
import logging
from pathlib import Path

# Third party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors_
# IBL libraries
from one.api import ONE
from iblutil.util import Bunch
import brainbox.io.one as bbone
from brainbox.io.one import SessionLoader

# Brainwidemap repo imports
from brainwidemap.bwm_loading import load_good_units, load_trials_and_mask, bwm_units
from brainwidemap.encoding.timeseries import TimeSeries, sync


def load_regressors(
    session_id,
    pid,
    one=None,
    t_before=0.0,
    t_after=0.2,
    binwidth=0.02,
    abswheel=False,
    clu_criteria="bwm",
    one_url="https://openalyx.internationalbrainlab.org",
    one_pw="international"
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
    if one is None:
        one = ONE(base_url=one_url, password=one_pw, silent=True)

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

    if clu_criteria == "bwm":
        spikes, clu_df = load_good_units(one=one, pid=pid)
    spk_times = spikes["times"]
    spk_clu = spikes["clusters"]
    clu_df["pid"] = pid
    clu_regions = clu_df.acronym
    return trialsdf, spk_times, spk_clu, clu_regions, clu_df


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
    is submitted via sbatch, and the latter of which is run by the worker nodes. Important to note
    is that the _workerscript file should absolutely not be modified during a run, as it would
    result in different workers running different code.

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
    if not len(singularity_modules) == 0:
        fw.write(f"module load {' '.join(singularity_modules)}\n")
    bindstr = "" if len(mount_paths) == 0 else "-B "
    mountpairs = ",".join([f"{k}:{v}" for k, v in mount_paths.items()])
    fw.write(f"singularity run {bindstr} {mountpairs} {container_image} /bin/bash {workerscript}")
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
    t_before=0.0,
    t_after=0.2,
    ret_wheel=False,
    ret_abswheel=False,
    wheel_binsize=0.02,
    addtl_types=[],
    trials_mask=None,
    one=None,
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
    if one is None:
        raise ValueError("one must be defined.")
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
    loader.load_session_data(pose=False, motion_energy=False, pupil=False, wheel=False)

    trialsdf = loader.trials[trialstypes]
    if trials_mask is not None:
        trialsdf = trialsdf.loc[trials_mask]
    trialsdf["trial_start"] = trialsdf["stimOn_times"] - t_before
    trialsdf["trial_end"] = trialsdf["feedback_times"] + t_after
    tdiffs = trialsdf["trial_end"] - np.roll(trialsdf["trial_start"], -1)
    if np.any(tdiffs.iloc[:-1] > 0):
        logging.warning(
            f"{sum(tdiffs[:-1] > 0)} trials overlapping due to t_before and t_after "
            "values. Try reducing one or both!"
        )
    if not ret_wheel and not ret_abswheel:
        return trialsdf
    # Load in the wheel data manually so we can do the resampling on our own terms rather than
    # through the SessionLoader
    wheel = one.load_object(eid, "wheel", collection="alf")
    whlpos, whlt = wheel.position, wheel.timestamps
    starttimes = trialsdf["trial_start"]
    endtimes = trialsdf["trial_end"]
    wh_endlast = 0  # Last index of the previously processed trial wheel idx
    trials = []  # List of trial wheel traces
    for (start, end) in np.vstack((starttimes, endtimes)).T:
        # Do a sorted search on the wheel timestamps to find the start and end indices
        wh_startind = np.searchsorted(whlt[wh_endlast:], start) + wh_endlast
        # Add a few bins past the canonical end index so when we resample to fixed t intervals
        # We know we will cover all of the trial time. 4 bins is a bit arbitrary but seems to work
        wh_endind = np.searchsorted(whlt[wh_endlast:], end, side="right") + wh_endlast + 4
        wh_endlast = wh_endind  # Even though we're 4 bins past the end, usually the ITI is far
                                # longer than 4 bins so we can just use that value still
        # Isolate trial-related positions and timestamps
        tr_whlpos = whlpos[wh_startind - 1 : wh_endind + 1]
        tr_whlt = whlt[wh_startind - 1 : wh_endind + 1] - start
        tr_whlt[0] = 0.0  # Manual previous-value interpolation
        # Resample to fixed time bins
        whlseries = TimeSeries(tr_whlt, tr_whlpos, columns=["whlpos"])
        whlsync = sync(wheel_binsize, timeseries=whlseries, interp="previous")
        # Extract the trial wheel trace and add to the list
        trialstartind = np.searchsorted(whlsync.times, 0)
        trialendind = np.ceil((end - start) / wheel_binsize).astype(int)
        trpos = whlsync.values[trialstartind : trialendind + trialstartind]
        # Raw wheel velocity with no smoothing
        whlvel = trpos[1:] - trpos[:-1]
        # Insert a zero at the beginning to make the length match the trial. Wheel speed needs
        # to be zero at this point anyway because of the enforced quiescence period.
        whlvel = np.insert(whlvel, 0, 0)
        if np.abs((trialendind - len(whlvel))) > 0:
            raise IndexError("Mismatch between expected length of wheel data and actual.")
        if ret_wheel:
            trials.append(whlvel)
        elif ret_abswheel:
            trials.append(np.abs(whlvel))
    trialsdf["wheel_velocity"] = trials
    return trialsdf


def single_cluster_raster(
    spike_times,
    events,
    trial_idx,
    dividers,
    colors,
    labels,
    pre_time=0.4,
    post_time=1.0,
    raster_bin=0.01,
    raster_cbar=False,
    raster_interp="none",
    show_psth=False,
    psth_bin=0.05,
    weights=None,
    fr=True,
    frac_tr=1,
    norm=False,
    axs=None,
):
    """
    Generate rasterplot for a single cluster, optionall with PSTH above.

    Parameters
    ----------
    spike_times : array-like
        Array of spike times for a single cluster
    events : array-like
        Time to which to align each trial
    trial_idx : array-like
        Indices of each trial
    dividers : array-like
        Times at which to put a divider marker on the raster?
    colors : array-like
        Colors for the dividers
    labels : Labels for parts of the raster
        replace with above
    pre_time : float, optional
        Time before event to plot, by default 0.4
    post_time : float, optional
        Time after event to plot, by default 1.
    raster_bin : float, optional
        Time bin size for the raster, by default 0.01
    raster_cbar : bool, optional
        Whether to include a color bar for the raster, which uses binned spike counts.
    raster_interp : str, optional
        Passed to matplotlib.pyplot.imshow, by default "none"
    psth : bool, optional
        Whether to plot the PSTH, by default False
    psth_bin : float, optional
        Time bin size for the PSTH, by default 0.05
    weights : array-like?, optional
        No clue, by default None
    fr : bool, optional
        Whether to plot as a firing rate or spike count, by default True
    norm : bool, optional
        Whether to normalize PSTH, by default False
    frac_tr: int
        WWhich fraction of trials to show    
    axs : matplotlib.pyplot.axes, optional
        Either a single axis if no PSTH or an array of 2 axes, by default None

    Returns
    -------
    fig, axs
        Figure and Axes objects

    """
    raster, t_raster = bin_spikes(
        spike_times,
        events.values,
        pre_time=pre_time,
        post_time=post_time,
        bin_size=raster_bin,
        weights=weights,
    )
    psth, t_psth = bin_spikes(
        spike_times,
        events.values,
        pre_time=pre_time,
        post_time=post_time,
        bin_size=psth_bin,
        weights=weights,
    )

    if fr:
        psth = psth / psth_bin

    if norm:
        psth = psth - np.repeat(psth[:, 0][:, np.newaxis], psth.shape[1], axis=1)
        raster = raster - np.repeat(raster[:, 0][:, np.newaxis], raster.shape[1], axis=1)

    dividers = [0] + dividers + [len(trial_idx)]
    if axs is None and show_psth:
        fig, axs = plt.subplots(
            2, 1, figsize=(4, 6), gridspec_kw={"height_ratios": [1, 3], "hspace": 0}, sharex=True
        )
        psth_ax = axs[0]
        raster_ax = axs[1]
    elif axs is None and not show_psth:
        fig, axs = plt.subplots(1, 1, figsize=(4, 3))
        raster_ax = axs
    else:
        if show_psth:
            if not hasattr(axs, "shape"):
                raise ValueError("axs must be a list of axes if psth is True")
            psth_ax = axs[0]
            raster_ax = axs[1]
            fig = axs[0].get_figure()
        else:
            if hasattr(axs, "shape"):
                raise ValueError("axs must be a single axis if psth is False")
            raster_ax = axs
            fig = axs.get_figure()

    label, lidx = np.unique(labels, return_index=True)
    label_pos = []
    for lab, lid in zip(label, lidx):
        idx = np.where(np.array(labels) == lab)[0]
        for iD in range(len(idx)):
            if iD == 0:
                t_ids = trial_idx[dividers[idx[iD]] + 1 : dividers[idx[iD] + 1] + 1]
                t_ints = dividers[idx[iD] + 1] - dividers[idx[iD]]
            else:
                t_ids = np.r_[t_ids, trial_idx[dividers[idx[iD]] + 1 : dividers[idx[iD] + 1] + 1]]
                t_ints = np.r_[t_ints, dividers[idx[iD] + 1] - dividers[idx[iD]]]

        psth_div = np.nanmean(psth[t_ids], axis=0)
        std_div = np.nanstd(psth[t_ids], axis=0) / np.sqrt(len(t_ids))

        if show_psth:
            psth_ax.fill_between(
                t_psth, psth_div - std_div, psth_div + std_div, alpha=0.4, color=colors[lid]
            )
            psth_ax.plot(t_psth, psth_div, alpha=1, color=colors[lid])

        lab_max = idx[np.argmax(t_ints)]
        label_pos.append((dividers[lab_max + 1] - dividers[lab_max]) / 2 + dividers[lab_max])

    cmap = colors_.LinearSegmentedColormap.from_list(
        "custom_binary", ["white", "black"])
    norm_ = colors_.Normalize(vmin=0, vmax=0.5)

    raster_ax.imshow(
        raster[trial_idx][::frac_tr],
        cmap=cmap,
        norm=norm_,
        origin="lower",
        extent=[np.min(t_raster), np.max(t_raster), 0, len(raster)],
        interpolation=raster_interp,
        aspect="auto",
    )


    width = raster_bin * 4
    for iD in range(len(dividers) - 1):
        raster_ax.fill_between(
            [post_time + raster_bin / 2, post_time + raster_bin / 2 + width],
            [dividers[iD + 1], dividers[iD + 1]],
            [dividers[iD], dividers[iD]],
            color=colors[iD],
        )

    raster_ax.set_xlim([-1 * pre_time, post_time + raster_bin / 2 + width])
    
    raster_ax.set_yticks(dividers)
    raster_ax.set_yticklabels([0,'',len(trial_idx[::frac_tr])])
    
    if raster_cbar:
        plt.colorbar(raster_ax.get_images()[0], ax=raster_ax, label="Spike count")
    secax = raster_ax.secondary_yaxis("right")

    secax.set_yticks(label_pos)
    secax.set_yticklabels(label, rotation=90, rotation_mode="anchor", ha="center")
    for ic, c in enumerate(np.array(colors)[lidx]):
        secax.get_yticklabels()[ic].set_color(c)
    if show_psth:
        axs[0].axvline(
            0, *axs[0].get_ylim(), c="k", ls="--", zorder=10
        )  # TODO this doesn't always work
    raster_ax.axvline(0, *raster_ax.get_ylim(), c="k", ls="--", zorder=10)

    return fig, axs


def bin_spikes(spike_times, align_times, pre_time, post_time, bin_size, weights=None):

    n_bins_pre = int(np.ceil(pre_time / bin_size))
    n_bins_post = int(np.ceil(post_time / bin_size))
    n_bins = n_bins_pre + n_bins_post
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    ts = np.repeat(align_times[:, np.newaxis], tscale.size, axis=1) + tscale
    epoch_idxs = np.searchsorted(spike_times, np.c_[ts[:, 0], ts[:, -1]])
    bins = np.zeros(shape=(align_times.shape[0], n_bins))

    for i, (ep, t) in enumerate(zip(epoch_idxs, ts)):
        xind = (np.floor((spike_times[ep[0] : ep[1]] - t[0]) / bin_size)).astype(np.int64)
        w = weights[ep[0] : ep[1]] if weights is not None else None
        r = np.bincount(xind, minlength=tscale.shape[0], weights=w)
        if w is not None:
            r_norm = np.bincount(xind, minlength=tscale.shape[0])
            r_norm[r_norm == 0] = 1
            r = r / r_norm
        bins[i, :] = r[:-1]

    tscale = (tscale[:-1] + tscale[1:]) / 2

    return bins, tscale


def find_trial_ids(trials, sort="idx"):
    """
    Sort trials in a df based on various parameters

    Parameters
    ----------
    trials : pandas.DataFrame
        Dataframe of trials, at a minimum containing the "contrastRight", "contrastLeft", and "feedbackType" columns
    sort : str, optional
        Sorting method, which can be:
            - "idx": trial index
            - "fdbk": feedback type, ordered by correct then incorrect
            - "side": side of stimulus, ordered by left and then right
            - "fdbk and side": stimulus side and feedback type, ordered by correct left,
                correct right, incorrect left, incorrect right
            - "movement": movement direction, ordered by left and then right
            - "block": Order of blocks, ordered by block identity (without 0.5 block) L then R
        by default "idx"

    Returns
    -------
    numpy.ndarray, list
        Array of trial indices and list of the indices in that array
            which separate based on condition.
    """
    # Find different permutations of trials
    # correct right
    cor_r = np.where(
        np.bitwise_and(trials["feedbackType"] == 1, np.isfinite(trials["contrastRight"]))
    )[0]
    # correct left
    cor_l = np.where(
        np.bitwise_and(trials["feedbackType"] == 1, np.isfinite(trials["contrastLeft"]))
    )[0]
    # incorrect right
    incor_r = np.where(
        np.bitwise_and(
            trials["feedbackType"] == -1, np.isfinite(trials["contrastRight"])
        )
    )[0]
    # incorrect left
    incor_l = np.where(
        np.bitwise_and(trials["feedbackType"] == -1, np.isfinite(trials["contrastLeft"]))
    )[0]

    # left and right blocks
    block_l = np.where(trials["probabilityLeft"] == 0.8)[0]
    block_r = np.where(trials["probabilityLeft"] == 0.2)[0]

    dividers = []

    # Find the trial id for all possible combinations
    if sort == "idx":
        trial_id = np.sort(np.r_[cor_r, cor_l, incor_r, incor_l])
    elif sort == "fdbk":
        trial_id = np.r_[np.sort(np.r_[cor_l, cor_r]), np.sort(np.r_[incor_l, incor_r])]
        dividers.append(np.r_[cor_l, cor_r].shape[0])
    elif sort == "side":
        trial_id = np.r_[np.sort(np.r_[cor_l, incor_l]), np.sort(np.r_[cor_r, incor_r])]
        dividers.append(np.r_[cor_l, incor_l].shape[0])
    elif sort == "fdbk and side":
        trial_id = np.r_[np.sort(cor_l), np.sort(incor_l), np.sort(cor_r), np.sort(incor_r)]
        dividers.append(cor_l.shape[0])
        dividers.append(np.r_[cor_l, incor_l].shape[0])
        dividers.append(np.r_[cor_l, incor_l, cor_r].shape[0])
    elif sort == "movement":
        trial_id = np.r_[np.sort(np.r_[cor_l, incor_r]), np.sort(np.r_[incor_l, cor_r])]
        dividers.append(np.r_[cor_l, incor_r].shape[0])
    elif sort == "block":
        trial_id = np.r_[block_l, block_r]
        dividers.append(block_l.shape[0])

    return trial_id, dividers
