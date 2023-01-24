# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import correlate, correlation_lags
from sklearn.preprocessing import normalize

# IBL libraries
import brainbox.io.one as bbone
from brainbox.metrics.single_units import quick_unit_metrics
from brainbox.processing import bincount2D
from iblutil.util import Bunch
from one.api import ONE

# which sesssion and probe to look at, bin size
BINWIDTH = 0.02
CORRWIND = (-0.8, 0.02)  # seconds
MIN_RATE = 1.0  # Minimum rate, in hertz, for a neuron to be included in xcorr analysis
PASSING_FRAC = 2 / 3

# Do some data loading
one = ONE()


# Build a basic vector to work with and also bin spikes
def binf(t):
    return np.ceil(t / BINWIDTH).astype(int)


def load_sess(eid, probe):
    """
    Fetch spike info and
    """
    one = ONE()
    spikes, _ = bbone.load_spike_sorting(
        eid,
        probe=probe,
        one=one,
        dataset_types=["spikes.times", "spikes.clusters", "spikes.amps", "spikes.depths"],
    )
    trialsdf = bbone.load_trials_df(eid, one=one, addtl_types=["firstMovement_times"])
    return spikes[probe], trialsdf


def load_sess_region(eid, probe, acronym="VISp"):
    """
    Fetch spike info and
    """
    one = ONE()
    spikes, clusters, _ = bbone.load_spike_sorting_with_channel(
        eid,
        probe=probe,
        one=one,
        dataset_types=["spikes.times", "spikes.clusters", "spikes.amps", "spikes.depths"],
    )
    trialsdf = bbone.load_trials_df(eid, one=one, addtl_types=["firstMovement_times"])
    return spikes[probe], trialsdf, clusters[probe].acronym


def get_spikes_events(spikes, trialsdf, passing_fraction=0.0):
    if passing_fraction > 0:
        metrics = quick_unit_metrics(spikes.clusters, spikes.times, spikes.amps, spikes.depths)
        pass_units = np.argwhere(metrics.label >= passing_fraction).squeeze()
        passmask = np.isin(spikes.clusters, pass_units)
        spikes = Bunch({k: v[passmask] for k, v in spikes.items()})

    # Get information about the details of our session such as start time etc
    t_start = 0
    t_end = trialsdf["trial_end"].max()

    events = {
        "leftstim": trialsdf[trialsdf.contrastLeft.notna()].stimOn_times,
        "rightstim": trialsdf[trialsdf.contrastRight.notna()].stimOn_times,
        "gocue": trialsdf.goCue_times,
        "movement": trialsdf.firstMovement_times,
        "correct": trialsdf[trialsdf.feedbackType == 1].feedback_times,
        "incorrect": trialsdf[trialsdf.feedbackType == -1].feedback_times,
    }
    return spikes, t_start, t_end, events


def get_binned(spikes, t_start, t_end):
    tmask = spikes.times < t_end  # Only get spikes in interval
    binned = bincount2D(
        spikes.times[tmask], spikes.clusters[tmask], xlim=[t_start, t_end], xbin=BINWIDTH
    )[0]
    ratemask = np.argwhere(np.mean(binned, axis=1) >= (BINWIDTH * MIN_RATE)).squeeze()
    binned = binned[ratemask]
    return binned


def get_event_vec(t_start, t_end, event_times, event_name):
    vecshape = binf(t_end + BINWIDTH) - binf(t_start)
    evec = np.zeros(vecshape)
    evinds = event_times[event_name].dropna().apply(binf)
    evec[evinds] = 1
    return evec


def xcorr_window(binned, evec):
    lags = correlation_lags(binned.shape[1], evec.shape[0]) * BINWIDTH  # Value of correlation lags
    start, end = np.searchsorted(lags, CORRWIND[0]), np.searchsorted(lags, CORRWIND[1]) + 1
    lagvals = lags[start:end]  # Per-step values of the lag
    corrarr = np.zeros((binned.shape[0], end - start))
    for i in range(binned.shape[0]):
        corrarr[i] = correlate(binned[i], evec)[start:end]
    return corrarr, lagvals


def heatmap_xcorr(corrarr, lagvals, ax=None, norm=True):
    ax = ax if ax is not None else plt.subplots(1, 1)[1]
    normarr = normalize(corrarr) if norm else corrarr
    sortinds = np.argsort(normarr.argmax(axis=1))
    sns.heatmap(pd.DataFrame(normarr[sortinds], columns=np.round(lagvals, 3)), ax=ax)
    ax.vlines(np.searchsorted(lagvals, 0) + 0.5, ax.get_ylim()[0], ax.get_ylim()[1], color="white")
    return ax


if __name__ == "__main__":
    # Third party libraries
    import dask
    from dask.distributed import Client
    from dask_jobqueue import SLURMCluster

    # Brainwidemap repo imports
    from brainwidemap.bwm_loading import bwm_query

    evkeys = [
        "leftstim",
        "rightstim",
        "gocue",
        "movement",
        "correct",
        "incorrect",
    ]

    sessions = bwm_query()
    N_CORES = 1
    cluster = SLURMCluster(
        cores=N_CORES,
        memory="12GB",
        processes=1,
        queue="shared-cpu",
        walltime="01:15:00",
        interface="ib0",
        extra=["--lifetime", "70m", "--lifetime-stagger", "4m"],
        job_cpu=N_CORES,
        env_extra=[
            f"export OMP_NUM_THREADS={N_CORES}",
            f"export MKL_NUM_THREADS={N_CORES}",
            f"export OPENBLAS_NUM_THREADS={N_CORES}",
        ],
    )
    cluster.adapt(minimum_jobs=0, maximum_jobs=400)
    client = Client(cluster)

    corrarrs = {k: [] for k in evkeys}
    for i, (eid, probe) in sessions[["eid", "probe"]].iterrows():
        spikes, trialsdf = dask.delayed(load_sess, nout=2)(eid, probe)
        spikes, t_start, t_end, events = dask.delayed(get_spikes_events, nout=4)(
            spikes, trialsdf, PASSING_FRAC
        )
        binned = dask.delayed(get_binned)(spikes, t_start, t_end)
        for event in evkeys:
            evec = dask.delayed(get_event_vec)(t_start, t_end, events, event)
            corrarr, lagvals = dask.delayed(xcorr_window, nout=2)(binned, evec)
            corrarrs[event].append(corrarr)

    corrcomp = {k: client.compute(v) for k, v in corrarrs.items()}
    stacked_corrs = {k: dask.delayed(np.vstack)(corrarrs[k]) for k in evkeys}
