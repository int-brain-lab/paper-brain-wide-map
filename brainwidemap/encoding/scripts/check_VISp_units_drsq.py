import sys
import os
import subprocess

# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold

# IBL libraries
from neurencoding.linear import LinearGLM
import neurencoding.utils as mut
from brainbox.plot import peri_event_time_histogram

# Brainwidemap repo imports
from brainwidemap.encoding.cluster_worker import get_cached_regressors
from brainwidemap.encoding.design import generate_design
from brainwidemap.encoding.params import GLM_CACHE, GLM_FIT_PATH


# Get fit parameters from the most recent fit
CACHEDATE = "2022-10-24"
params = pd.read_pickle(GLM_FIT_PATH + "2022-10-24_glm_fit_pars.pkl")
yanliang_units = pd.read_csv("/home/berk/Downloads/BWM_single_cell_VISp_sig.csv")
mwU_pvals = yanliang_units.rename(columns={"cluster_id": "clu_id"}).set_index(
    ["eid", "pid", "clu_id"]
)
mwU_pvals = mwU_pvals.assign(cachefn=None)
totcells = len(mwU_pvals) - 8

fig, ax = plt.subplots(sqrt := np.ceil(np.sqrt(totcells)).astype(int), sqrt, figsize=(11, 11))
ax = ax.flatten()

cellcounter = 0
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
    try:
        nglm = LinearGLM(design, sspkt, sspkclu, params["binwidth"], estimator=params["estimator"])
    except UserWarning:
        continue
    sfs = mut.SequentialSelector(nglm, len(design.covar.keys()) - 1, "backward")
    splitter = KFold(n_splits=5, shuffle=True, random_state=0)
    alldiffs = []
    for i, (train, test) in enumerate(splitter.split(design.trialsdf.index)):
        trainidx = design.trialsdf.index[train]
        sfs.fit(train_idx=trainidx, full_scores=True)
        diffs = pd.DataFrame(
            {
                name: sfs.basescores - sfs.full_scores_test_[k].loc[:, 0]
                for name, k in zip(design.covar.keys(), sfs.full_scores_test_.columns)
            }
        )
        diffs.index.name = "clu_id"
        diffs = diffs.assign(fold=i)
        alldiffs.append(diffs)
    alldiffs = pd.concat(alldiffs).set_index("fold", append=True)
    meandiffs = alldiffs.groupby("clu_id").agg("mean")
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
        if oldlim[1] > newlim[1]:
            ax[cellcounter].set_ylim([0, oldlim[1]])
        ax[cellcounter].legend()
        ax[cellcounter].set_title(
            f"{pid}\ndiff={data.stimonR}\n"
            f"MWU pval={mwU_pvals.loc[pd.IndexSlice[:, pid, cell], 'p_value_stim'].values}",
            fontsize=6,
        )
        cellcounter += 1
