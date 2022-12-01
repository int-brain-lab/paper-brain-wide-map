# Standard library
import os
import subprocess
import sys
from copy import deepcopy

# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold

# IBL libraries
import neurencoding.utils as mut
from brainbox.plot import peri_event_time_histogram
from neurencoding.linear import LinearGLM

# Brainwidemap repo imports
from brainwidemap.encoding.cluster_worker import get_cached_regressors
from brainwidemap.encoding.design import generate_design
from brainwidemap.encoding.params import GLM_CACHE, GLM_FIT_PATH

# Get fit parameters from the most recent fit
CACHEDATE = "2022-10-24"
params = pd.read_pickle(GLM_FIT_PATH + "2022-10-24_glm_fit_pars.pkl")
fastbases_pars = deepcopy(params)


def binf(t):
    return np.ceil(t / 0.02).astype(int)


fastbases_pars["bases"]["stim"] = mut.full_rcos(0.4, 10, binf)
yanliang_units = pd.read_csv("/home/berk/Downloads/BWM_single_cell_VISp_sig.csv")
mwU_pvals = yanliang_units.rename(columns={"cluster_id": "clu_id"}).set_index(
    ["eid", "pid", "clu_id"]
)
mwU_pvals = mwU_pvals.assign(cachefn=None)
totcells = len(mwU_pvals) - 8

fig, ax = plt.subplots(sqrt := np.ceil(np.sqrt(totcells)).astype(int), sqrt, figsize=(11, 11))
ax = ax.flatten()

cellcounter = 0
comparisonmetrics = []
for pid in mwU_pvals.index.get_level_values("pid").unique():
    fns = (
        subprocess.run(
            ["find", GLM_CACHE, "-name", f"*{CACHEDATE}_{pid}_regressors.pkl"], capture_output=True
        )
        .stdout.decode("utf-8")
        .split("\n")
    )
    fn = fns[0]
    mwU_pvals.loc[pd.IndexSlice[:, pid, :], "cachefn"] = fn
    stdf, sspkt, sspkclu, sclureg, scluqc = get_cached_regressors(fn)
    targetclu = mwU_pvals.loc[:, pid, :].index.get_level_values("clu_id")
    mask = np.isin(sspkclu, targetclu)
    sspkt = sspkt[mask]
    sspkclu = sspkclu[mask]
    design = generate_design(stdf, stdf["probabilityLeft"], 0.6, **params)
    fastdesign = generate_design(stdf, stdf["probabilityLeft"], 0.6, **fastbases_pars)
    try:
        nglm = LinearGLM(design, sspkt, sspkclu, params["binwidth"], estimator=params["estimator"])
        nglm_fast = LinearGLM(
            fastdesign, sspkt, sspkclu, params["binwidth"], estimator=params["estimator"]
        )
    except UserWarning:
        continue
    sfs = mut.SequentialSelector(nglm, len(design.covar.keys()) - 1, "backward")
    sfs_fast = mut.SequentialSelector(nglm_fast, len(design.covar.keys()) - 1, "backward")
    splitter = KFold(n_splits=5, shuffle=True, random_state=0)
    alldiffs = []
    allfastdiffs = []
    for i, (train, test) in enumerate(splitter.split(design.trialsdf.index)):
        trainidx = design.trialsdf.index[train]
        sfs.fit(train_idx=trainidx, full_scores=True)
        sfs_fast.fit(train_idx=trainidx, full_scores=True)
        diffs = pd.DataFrame(
            {
                name: sfs.basescores_test_ - sfs.full_scores_test_[k].loc[:, 0]
                for name, k in zip(design.covar.keys(), sfs.full_scores_test_.columns)
            }
        )
        fastdiffs = pd.DataFrame(
            {
                name: sfs_fast.basescores_test_ - sfs_fast.full_scores_test_[k].loc[:, 0]
                for name, k in zip(design.covar.keys(), sfs_fast.full_scores_test_.columns)
            }
        )
        diffs.index.name = "clu_id"
        fastdiffs.index.name = "clu_id"
        diffs = diffs.assign(fold=i)
        fastdiffs = fastdiffs.assign(fold=i)
        alldiffs.append(diffs)
        allfastdiffs.append(fastdiffs)
    alldiffs = pd.concat(alldiffs).set_index("fold", append=True)
    allfastdiffs = pd.concat(allfastdiffs).set_index("fold", append=True)
    meandiffs = alldiffs.groupby("clu_id").agg("mean")
    meanfastdiffs = allfastdiffs.groupby("clu_id").agg("mean")
    comparisonmetrics.append({"pid": pid, "diffs": meandiffs, "fastdiffs": meanfastdiffs})
    for cell, data in meandiffs.iterrows():
        peri_event_time_histogram(
            sspkt,
            sspkclu,
            stdf.stimOn_times[np.isfinite(stdf.contrastRight) & (stdf.contrastRight > 0)],
            cell,
            t_before=0.1,
            t_after=0.4,
            bin_size=0.02,
            smoothing=0.001,
            error_bars="sem",
            pethline_kwargs={"color": "darkorange", "lw": 2, "label": "Right stim"},
            errbar_kwargs={"color": "darkorange", "alpha": 0.2},
            ax=ax[cellcounter],
        )
        oldlim = ax[cellcounter].get_ylim()
        oldticks = ax[cellcounter].get_yticks()
        peri_event_time_histogram(
            sspkt,
            sspkclu,
            stdf.stimOn_times[np.isfinite(stdf.contrastLeft) & (stdf.contrastLeft > 0)],
            cell,
            t_before=0.1,
            t_after=0.4,
            bin_size=0.02,
            smoothing=0.001,
            error_bars="sem",
            pethline_kwargs={"color": "navy", "lw": 2, "label": "Left stim"},
            errbar_kwargs={"color": "navy", "alpha": 0.2},
            ax=ax[cellcounter],
        )
        newlim = ax[cellcounter].get_ylim()
        newticks = ax[cellcounter].get_yticks()
        if oldlim[1] > newlim[1]:
            ax[cellcounter].set_ylim([0, oldlim[1]])
            ax[cellcounter].set_yticks(oldticks)
        ax[cellcounter].legend()
        ax[cellcounter].set_title(
            f"{pid}\ndiff={data.stimonR}\n"
            f"fastdiff={meanfastdiffs.loc[cell, 'stimonR']}\n"
            f"MWU pval={mwU_pvals.loc[pd.IndexSlice[:, pid, cell], 'p_value_stim'].values}",
            fontsize=6,
        )
        cellcounter += 1
