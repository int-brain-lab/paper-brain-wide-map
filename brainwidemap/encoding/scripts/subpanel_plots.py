from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from one.api import ONE
from brainwidemap.bwm_loading import bwm_query
from brainwidemap.encoding.design import generate_design
from brainwidemap.encoding.glm_predict import GLMPredictor, predict
from brainwidemap.encoding.params import GLM_CACHE
from brainwidemap.encoding.utils import load_regressors, single_cluster_raster, find_trial_ids

import neurencoding.linear as lm
from neurencoding.utils import remove_regressors


def plot_twocond(
    eid,
    pid,
    clu_id,
    align_time,
    aligncol,
    aligncond1,
    aligncond2,
    t_before,
    t_after,
    regressors,
):
    stdf, sspkt, sspkclu, design, spkmask, nglm = load_unit_fit_model(eid, pid, clu_id)
    pred = GLMPredictor(stdf, nglm, sspkt, sspkclu)
    fig, ax = plt.subplots(3, 4, figsize=(12, 12), sharey="row")
    oldticks = []
    pred.psth_summary(
        align_time,
        clu_id,
        t_before,
        t_after,
        trials=stdf[aligncond1(stdf[aligncol])].index,
        ax=ax[:, 0],
    )
    oldticks.extend(ax[0, 0].get_yticks())
    pred.psth_summary(
        align_time,
        clu_id,
        t_before,
        t_after,
        trials=stdf[aligncond2(stdf[aligncol])].index,
        ax=ax[:, 1],
    )
    oldticks.extend(ax[0, 1].get_yticks())
    noreg_dm = remove_regressors(design, regressors)
    nrnglm = lm.LinearGLM(
        noreg_dm, sspkt[spkmask], sspkclu[spkmask], estimator=glm_params["estimator"], mintrials=0
    )
    nrnglm.fit()
    nrpred = GLMPredictor(stdf, nrnglm, sspkt, sspkclu)
    nrpred.psth_summary(
        align_time,
        clu_id,
        t_before,
        t_after,
        trials=stdf[aligncond1(stdf[aligncol])].index,
        ax=ax[:, 2],
    )
    oldticks.extend(ax[0, 2].get_yticks())
    nrpred.psth_summary(
        align_time,
        clu_id,
        t_before,
        t_after,
        trials=stdf[aligncond2(stdf[aligncol])].index,
        ax=ax[:, 3],
    )
    oldticks.extend(ax[0, 3].get_yticks())
    ax[0, 0].set_ylim([0, np.max(oldticks) * 1.1])
    ax[0, 0].set_yticks(oldticks)
    return fig, ax, sspkt, sspkclu, stdf


def load_unit_fit_model(eid, pid, clu_id):
    stdf, sspkt, sspkclu, _, __ = load_regressors(
        eid,
        pid,
        one,
        t_before=0.6,
        t_after=0.6,
        binwidth=glm_params["binwidth"],
        abswheel=True,
    )
    design = generate_design(stdf, stdf["probabilityLeft"], t_before=0.6, **glm_params)
    spkmask = sspkclu == clu_id
    nglm = lm.LinearGLM(
        design, sspkt[spkmask], sspkclu[spkmask], estimator=glm_params["estimator"], mintrials=0
    )
    nglm.fit()
    return stdf, sspkt, sspkclu, design, spkmask, nglm


# Please use the saved parameters dict from 02_fit_sessions.py as params
PLOTPATH = Path("/home/berk/Documents/Projects/results/plots/prediction_summaries")
N_TOP_UNITS = 20
RAST_BINSIZE = 0.002
one = ONE()
plt.rcParams["svg.fonttype"] = "none"
# Sets of align_time as key with aligncol, aligncond1/2 functions, t_before/t_after,
# and the name of the associated model regressors as values
alignsets = {
    "stimOn_times": (
        "contrastRight",  # Column name in df to use for filtering
        lambda c: np.isnan(c),  # Condition 1 function (left stim)
        lambda c: np.isfinite(c),  # Condition 2 function (right stim)
        0.1,  # Time before align_time to include in trial psth/raster
        0.4,  # Time after align_time to include in trial psth/raster
        "stimonL",  # Condition 1 label within the GLM design matrix
        "stimonR",  # Condition 2 label within the GLM design matrix
    ),
    "firstMovement_times": (
        "choice",
        lambda c: c == 1,
        lambda c: c == -1,
        0.2,
        0.05,
        "fmoveL",
        "fmoveR",
    ),
    "feedback_times": (
        "feedbackType",
        lambda f: f == 1,
        lambda f: f == -1,
        0.1,
        0.4,
        "correct",
        "incorrect",
    ),
}

glm_params = pd.read_pickle(Path(__file__).parent.joinpath("glm_params.pkl"))

# Which units we're going to use for plotting
targetunits = {  # eid, pid, clu_id, region, drsq, alignset key
    "stim": (
        "e0928e11-2b86-4387-a203-80c77fab5d52",  # EID
        "799d899d-c398-4e81-abaf-1ef4b02d5475",  # PID
        209,  # clu_id
        "VISp",  # region
        0.04540706,  # drsq (from 02_fit_sessions.py)
        "stimOn_times",  # Alignset key
    ),
    "choice": (
        "671c7ea7-6726-4fbe-adeb-f89c2c8e489b",
        "04c9890f-2276-4c20-854f-305ff5c9b6cf",
        123,
        "GRN",
        0.000992895,  # drsq
        "firstMovement_times",
    ),
    "feedback": (
        "a7763417-e0d6-4f2a-aa55-e382fd9b5fb8",
        "57c5856a-c7bd-4d0f-87c6-37005b1484aa",
        98,
        "IRN",
        0.3077195113,  # drsq
        "feedback_times",
    ),
    "block": (
        "7bee9f09-a238-42cf-b499-f51f765c6ded",
        "26118c10-35dd-4ab1-9f0f-b9a89a1da070",
        207,
        "MOp",
        0.0043285,  # drsq
        "stimOn_times",
    ),
}


# This is a hack to use the "find trial IDs" function provided by Mayo to get the trial IDs
# corresponding to each condition for the different variables
sortlookup = {"stim": "side", "choice": "movement", "feedback": "fdbk", "wheel": "movement"}

for variable, (eid, pid, clu_id, region, drsq, aligntime) in targetunits.items():
    # Skip block (separate logic) and make folder structure before continuing if folders don't exist
    if variable == "block":
        continue
    varfolder = Path(PLOTPATH).joinpath(variable)  # Variable gets its own folder
    rasterfolder = varfolder.joinpath("rasters")  # Rasters in a separate subfolder for variable
    if not varfolder.exists():
        varfolder.mkdir()
    if not rasterfolder.exists():
        rasterfolder.mkdir()
    if not varfolder.joinpath("png").exists():  # PNGs separated too
        varfolder.joinpath("png").mkdir()
    if not rasterfolder.joinpath("png").exists():  # PNGs separated too
        rasterfolder.joinpath("png").mkdir()
    aligncol, aligncond1, aligncond2, t_before, t_after, reg1, reg2 = alignsets[aligntime]
    fig, ax, sspkt, sspkclu, stdf = plot_twocond(
        eid,
        pid,
        clu_id,
        aligntime,
        aligncol,
        aligncond1,
        aligncond2,
        t_before,
        t_after,
        [reg1, reg2] if variable != "wheel" else ["wheel"],
    )
    # Wheel only has one regressor, unlike all the others. Handle this properly
    if variable != "wheel":
        remstr = f"\n[{reg1}, {reg2}] regressors rem."
    else:
        remstr = "\nwheel regressor rem."
    names = [reg1, reg2, reg1 + remstr, reg2 + remstr]
    for subax, title in zip(ax[0, :], names):
        subax.set_title(title)
    plt.savefig(varfolder.joinpath(f"{eid}_{pid}_clu{clu_id}_{region}_{variable}_predsummary.svg"))
    plt.savefig(
        varfolder.joinpath(f"png/{eid}_{pid}_clu{clu_id}_{region}_{variable}_predsummary.png")
    )
    plt.close()

    stdf["response_times"] = stdf["stimOn_times"]
    trial_idx, dividers = find_trial_ids(stdf, sort=sortlookup[variable])
    fig, ax = single_cluster_raster(
        sspkt[sspkclu == clu_id],
        stdf[aligntime],
        trial_idx,
        dividers,
        ["b", "r"],
        [reg1, reg2],
        pre_time=t_before,
        post_time=t_after,
        raster_cbar=True,
        raster_bin=RAST_BINSIZE,
    )
    ax.set_title("{} unit {} : $\log \Delta R^2$ = {:.2f}".format(region, clu_id, np.log(drsq)))
    plt.savefig(rasterfolder.joinpath(f"{eid}_{pid}_clu{clu_id}_{region}_{variable}_raster.svg"))
    plt.savefig(
        rasterfolder.joinpath(f"png/{eid}_{pid}_clu{clu_id}_{region}_{variable}_raster.png")
    )
    plt.close()

## Treat block separately since it's a different type of plot
variable = "block"
varfolder = Path(PLOTPATH).joinpath(variable)
if not varfolder.exists():
    varfolder.mkdir()

eid, pid, clu_id, region, drsq, aligntime = targetunits[variable]
stdf, sspkt, sspkclu, design, spkmask, nglm = load_unit_fit_model(eid, pid, clu_id)
pred, trlabels = predict(nglm, glm_type="linear", retlab=True)
mask = design.dm[:, design.covar["pLeft"]["dmcol_idx"]] != 0
itipred = pred[clu_id][mask]
iticounts = nglm.binnedspikes[mask, :]
labels = trlabels[mask]
rates = pd.DataFrame(
    index=stdf.index[stdf.probabilityLeft != 0.5],
    columns=["firing_rate", "pred_rate", "pLeft"],
    dtype=float,
)
for p_val in [0.2, 0.8]:
    trials = stdf.index[stdf.probabilityLeft == p_val]
    for trial in trials:
        trialmask = labels == trial
        rates.loc[trial, "firing_rate"] = np.mean(iticounts[trialmask]) / design.binwidth
        rates.loc[trial, "pred_rate"] = np.mean(itipred[trialmask]) / design.binwidth
        rates.loc[trial, "pLeft"] = p_val
fig, ax = plt.subplots(1, 2, figsize=(6, 6), sharey=True)
sns.boxplot(rates, x="pLeft", y="firing_rate", ax=ax[0])
sns.boxplot(rates, x="pLeft", y="pred_rate", ax=ax[1])
ax[0].set_title(f"{region} {clu_id} firing rate by block")
ax[1].set_title(f"{region} {clu_id} predicted rate by block")
ax[0].set_ylabel("Firing rate (spikes/s)")
plt.savefig(varfolder.joinpath(f"{eid}_{pid}_clu{clu_id}_{region}_{variable}_predsummary.svg"))
plt.savefig(varfolder.joinpath(f"{eid}_{pid}_clu{clu_id}_{region}_{variable}_predsummary.png"))
plt.close()
