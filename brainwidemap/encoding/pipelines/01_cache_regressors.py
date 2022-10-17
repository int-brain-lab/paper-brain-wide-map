# Standard library
import hashlib
import logging
import os
import pickle
import re
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
from one.api import ONE
from brainwidemap.encoding.params import GLM_CACHE
from brainwidemap.encoding.utils import load_trials_df
from brainwidemap.bwm_loading import load_trials_and_mask, bwm_query
from brainwidemap.encoding.utils import get_impostor_df

_logger = logging.getLogger('brainwide')


def load_regressors(session_id,
                    pid,
                    t_before=0.,
                    t_after=0.,
                    binwidth=0.02,
                    abswheel=False,
                    clu_criteria='bwm',
                    one=None):
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

    _, mask = load_trials_and_mask(one=one, eid=eid)
    trialsdf = load_trials_df(session_id,
                              t_before=t_before,
                              t_after=t_after,
                              wheel_binsize=binwidth,
                              ret_abswheel=abswheel,
                              ret_wheel=not abswheel,
                              addtl_types=['firstMovement_times'],
                              one=one,
                              trials_mask=mask,
                              )

    clusters = {}
    ssl = bbone.SpikeSortingLoader(one=one, pid=pid)
    origspikes, tmpclu, channels = ssl.load_spike_sorting()
    if 'metrics' not in tmpclu:
        tmpclu['metrics'] = np.ones(tmpclu['channels'].size)
    clusters[pid] = ssl.merge_clusters(origspikes, tmpclu, channels)
    clu_df = pd.DataFrame(clusters[pid]).set_index(['cluster_id'])
    clu_df['pid'] = pid

    if clu_criteria == 'bwm':
        keepclu = clu_df.index[clu_df.label == 1]
    elif clu_criteria == 'all':
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


def cache_regressors(subject, session_id, pid, regressor_params, trialsdf, spk_times, spk_clu,
                     clu_regions, clu_qc, clu_df):
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
    curr_t = dt.now()
    fnbase = str(curr_t.date())
    metadata_fn = sesspath.joinpath(fnbase + '_metadata.pkl')
    data_fn = sesspath.joinpath(fnbase + '_regressors.pkl')
    regressors = {
        'trialsdf': trialsdf,
        'spk_times': spk_times,
        'spk_clu': spk_clu,
        'clu_regions': clu_regions,
        'clu_qc': clu_qc,
        'clu_df': clu_df,
    }
    reghash = _hash_dict(regressors)
    metadata = {
        'subject': subject,
        'session_id': session_id,
        'pid': pid,
        'regressor_hash': reghash,
        **regressor_params
    }
    prevdata = [
        sesspath.joinpath(f) for f in os.listdir(sesspath) if re.match(r'.*_metadata\.pkl', f)
    ]
    matchfile = False
    for f in prevdata:
        with open(f, 'rb') as fr:
            frdata = pickle.load(fr)
            if metadata == frdata:
                matchfile = True
        if matchfile:
            _logger.info(f'Existing cache file found for {subject}: {session_id}, '
                         'not writing data.')
            old_data_fn = sesspath.joinpath(f.name.split('_')[0] + '_regressors.pkl')
            return f, old_data_fn
    # If you've reached here, there's no matching file
    with open(metadata_fn, 'wb') as fw:
        pickle.dump(metadata, fw)
    with open(data_fn, 'wb') as fw:
        pickle.dump(regressors, fw)
    return metadata_fn, data_fn


def _hash_dict(d):
    hasher = hashlib.md5()
    sortkeys = sorted(d.keys())
    for k in sortkeys:
        v = d[k]
        if type(v) == np.ndarray:
            hasher.update(v)
        elif isinstance(v, (pd.DataFrame, pd.Series)):
            hasher.update(v.to_string().encode())
        else:
            try:
                hasher.update(v)
            except Exception:
                _logger.warning(f'Key {k} was not able to be hashed. May lead to failure to update'
                                ' in cached files if something was changed.')
    return hasher.hexdigest()


@dask.delayed
def delayed_load(session_id, pid, params, force_load=False):
    try:
        return load_regressors(session_id, pid, **params)
    except KeyError as e:
        if force_load:
            params['resolved_alignment'] = False
            return load_regressors(session_id, pid, **params)
        else:
            raise e


@dask.delayed(pure=False, traverse=False)
def delayed_save(subject, session_id, pid, params, outputs):
    return cache_regressors(subject, session_id, pid, params, *outputs)


# Parameters
SESS_CRITERION = 'resolved-behavior'
DATE = str(dt.today())
T_BEF = 0.6
T_AFT = 0.6
BINWIDTH = 0.02
ABSWHEEL = True
QC = True
EPHYS_IMPOSTOR = False
FORCE = True  # If load_spike_sorting_fast doesn't return _channels, use _channels function
# End parameters

# Construct params dict from above
params = {
    't_before': T_BEF,
    't_after': T_AFT,
    'binwidth': BINWIDTH,
    'abswheel': ABSWHEEL,
}

one = ONE()
dataset_futures = []

sessdf = bwm_query(freeze="2022_10_initial").set_index("pid")

for pid, rec in sessdf.iterrows():
    subject = rec.subject
    eid = rec.eid
    load_outputs = delayed_load(eid, pid, params)
    save_future = delayed_save(subject, eid, pid, params, load_outputs)
    dataset_futures.append([subject, eid, pid, save_future])

N_CORES = 4
cluster = SLURMCluster(cores=N_CORES,
                       memory='32GB',
                       processes=1,
                       queue="shared-cpu",
                       walltime="01:15:00",
                       log_directory='/home/gercek/dask-worker-logs',
                       interface='ib0',
                       extra=["--lifetime", "60m", "--lifetime-stagger", "10m"],
                       job_cpu=N_CORES,
                       env_extra=[
                           f'export OMP_NUM_THREADS={N_CORES}',
                           f'export MKL_NUM_THREADS={N_CORES}',
                           f'export OPENBLAS_NUM_THREADS={N_CORES}'
                       ])
cluster.scale(20)
client = Client(cluster)

tmp_futures = [client.compute(future[3]) for future in dataset_futures]
params['maxlen'] = params['max_len']
params.pop('max_len')
# impostor_df = get_impostor_df(
#     '',
#     one,
#     ephys=EPHYS_IMPOSTOR,
#     tdf_kwargs={k: v for k, v in params.items() if k not in ['resolved_alignment', 'ret_qc']},
#     ret_template=True)

# Run below code AFTER futures have finished!
dataset = [{
    'subject': x[0],
    'eid': x[1],
    'probes': x[2],
    'meta_file': tmp_futures[i].result()[0],
    'reg_file': tmp_futures[i].result()[1]
} for i, x in enumerate(dataset_futures) if tmp_futures[i].status == 'finished']
dataset = pd.DataFrame(dataset)

outdict = {'params': params, 'dataset_filenames': dataset}
with open(Path(GLM_CACHE).joinpath(DATE + '_dataset_metadata.pkl'), 'wb') as fw:
    pickle.dump(outdict, fw)
