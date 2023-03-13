from brainwidemap.bwm_loading import bwm_query
from pathlib import Path
from brainwidemap.encoding.glm_predict import GLMPredictor
import neurencoding.linear as lm
from neurencoding.utils import remove_regressors
import matplotlib.pyplot as plt
import brainwidemap.encoding.cluster_worker as cw
import pandas as pd
import numpy as np
from brainwidemap.encoding.params import GLM_FIT_PATH, GLM_CACHE

# Please use the saved parameters dict form 02_fit_sessions.py as params
PLOTPATH = Path("/home/berk/Documents/Projects/results/plots/prediction_summaries")
N_TOP_UNITS = 5
IMGFMT = "png"
alignsets = {  # Sets of align_time as key with aligncol, aligncond1/2 functions, and t_before/t_after as the values
    "stimOn_times": (
        "contrastRight",
        lambda c: np.isnan(c),
        lambda c: np.isfinite(c),
        0.1,
        0.4,
        "stimonL",
        "stimonR",
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

targetreg = {  # Function to produce the target metric, the target regions, and alignset key for each plottable
    "stim": (lambda df: df["stimonR"] - df["stimonL"], ["VISp"], "stimOn_times"),
    "choice": (lambda df: df["fmoveR"] - df["fmoveL"], ["GRN"], "firstMovement_times"),
    "feedback": (lambda df: df["correct"] - df["incorrect"], ["IRN"], "feedback_times"),
}

params = pd.read_pickle(GLM_FIT_PATH + "/2023-03-07_glm_fit_pars.pkl")
meanscores = pd.read_pickle(GLM_FIT_PATH + "/2023-03-02_glm_fit.pkl")[
    "mean_fit_results"
].set_index("region", append=True)


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
    sessdf = bwm_query()
    subject = sessdf[sessdf["eid"] == eid]["subject"].iloc[0]
    eidfn = Path(GLM_CACHE).joinpath(Path(f"{subject}/{eid}/2022-12-22_{pid}_regressors.pkl"))
    stdf, sspkt, sspkclu, sclureg, clu_df = cw.get_cached_regressors(eidfn)
    design = cw.generate_design(stdf, stdf["probabilityLeft"], t_before=0.6, **params)
    spkmask = sspkclu == clu_id
    nglm = lm.LinearGLM(
        design, sspkt[spkmask], sspkclu[spkmask], estimator=params["estimator"], mintrials=0
    )
    nglm.fit()
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
        noreg_dm, sspkt[spkmask], sspkclu[spkmask], estimator=params["estimator"], mintrials=0
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
    return fig, ax


for variable, (targetmetricfun, regions, aligntime) in targetreg.items():
    varfolder = Path(PLOTPATH).joinpath(variable)
    if not varfolder.exists():
        varfolder.mkdir()
    targetmetric = targetmetricfun(meanscores)
    aligncol, aligncond1, aligncond2, t_before, t_after, reg1, reg2 = alignsets[aligntime]
    for region in regions:
        topunits = (
            targetmetric.loc[:, :, :, region].sort_values(ascending=False).iloc[:N_TOP_UNITS]
        )
        for (eid, pid, clu_id), drsq in topunits.items():
            fig, ax = plot_twocond(
                eid,
                pid,
                clu_id,
                aligntime,
                aligncol,
                aligncond1,
                aligncond2,
                t_before,
                t_after,
                [reg1, reg2],
            )
            remstr = f"\n[{reg1}, {reg2}] regressors rem."
            names = [reg1, reg2, reg1 + remstr, reg2 + remstr]
            for subax, title in zip(ax[0, :], names):
                subax.set_title(title)
            plt.savefig(
                varfolder.joinpath(
                    f"{eid}_{pid}_clu{clu_id}_{region}_{variable}_predsummary.{IMGFMT}"
                )
            )
            plt.close()
