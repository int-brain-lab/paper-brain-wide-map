from brainwidemap.bwm_loading import bwm_query
from pathlib import Path
from brainwidemap.encoding.glm_predict import GLMPredictor
import neurencoding.linear as lm
import matplotlib.pyplot as plt
import brainwidemap.encoding.cluster_worker as cw
import pandas as pd
from brainwidemap.encoding.params import GLM_FIT_PATH

# Please use the saved parameters dict form 02_fit_sessions.py as params
PLOTPATH = Path("/home/berk/Documents/Projects/results/plots/prediction_summaries/")
params = pd.read_pickle(GLM_FIT_PATH + "/2023-01-16_glm_fit")
meanscores = pd.read_pickle(GLM_FIT_PATH + "/2023-01-16_glm_fit.pkl")["mean_fit_results"]


def plot_twocond(
    eid, pid, clu_id, align_time, aligncol, aligncond1, aligncond2, t_before, t_after
):
    sessdf = bwm_query()
    subject = sessdf[sessdf["eid"] == eid]["subject"].iloc[0]
    eidfn = Path(f"./{subject}/{eid}/2022-12-22_{pid}_regressors.pkl")
    stdf, sspkt, sspkclu, sclureg, clu_df = cw.get_cached_regressors(eidfn)
    design = cw.generate_design(stdf, stdf["probabilityLeft"], t_before=0.6, **params)
    nglm = lm.LinearGLM(design, sspkt, sspkclu, estimator=params["estimator"])
    nglm.fit()
    pred = GLMPredictor(stdf, nglm, sspkt, sspkclu)
    fig, ax = plt.subplots(3, 2, figsize=(12, 9), sharey="row")
    pred.psth_summary(
        align_time,
        clu_id,
        t_before,
        t_after,
        trials=stdf[stdf[aligncol] == aligncond1].index,
        ax=ax[:, 0],
    )
    pred.psth_summary(
        align_time,
        clu_id,
        t_before,
        t_after,
        trials=stdf[stdf[aligncol] == aligncond2].index,
        ax=ax[:, 1],
    )
    return fig, ax
