from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brainwidemap import bwm_query

BASEPATH = Path("/home/berk/Documents/Projects/results/glms/merged_results")
OLDDATE = "2022-03-22"
NEWDATE = "2022-10-24"
oldfit = BASEPATH.joinpath(f"{OLDDATE}_impostor_run")
newfit = BASEPATH.joinpath("{NEWDATE}_impostor_run")

firstprobes = (
    bwm_query(freeze="2022_10_update").query("probe_name == 'probe00'").pid.unique()
)

plotshapes = {5: (2, 3), 3: (1, 3), 1: (1, 1)}
plotsizes = {5: (8, 8), 3: (7, 4), 1: (5, 5)}
for kern in [
    "stimonR",
    "stimonL",
    "correct",
    "incorrect",
    "fmoveR",
    "fmoveL",
    "wheel",
    "pLeft",
    "pLeft_tr",
]:
    old_var = pd.read_parquet(oldfit.joinpath(f"{kern}_percentiles.parquet"))
    new_var = pd.read_parquet(newfit.joinpath(f"{kern}_percentiles.parquet"))
    try:
        old_var = old_var[new_var.columns]
    except KeyError:
        print(f"Kernel {kern} did not have matching columns")
        continue
    new_var = new_var[new_var.index.isin(firstprobes, level="probe")]
    old_grp = old_var.groupby(["eid", "clu_id"]).agg("mean")
    new_grp = new_var.groupby(["eid", "clu_id"]).agg("mean")
    ncols = len(new_var.columns)
    old_grp.columns = range(1, ncols + 1)
    new_grp.columns = range(1, ncols + 1)
    joindf = old_grp.join(new_grp, lsuffix="_old", rsuffix="_new", how="inner")
    fig, ax = plt.subplots(
        *plotshapes[ncols],
        sharex=True,
        sharey=True,
        figsize=plotsizes[ncols],
    )
    if not ncols == 1:
        ax = ax.flatten()
        for i in range(ncols, plotshapes[ncols][0] * plotshapes[ncols][1]):
            ax[i].remove()
    else:
        ax = np.array([ax])
    for i in range(1, ncols + 1):
        xcol = f"{i}_old"
        ycol = f"{i}_new"
        sns.kdeplot(
            data=joindf,
            x=xcol,
            y=ycol,
            ax=ax[i - 1],
            bw_adjust=0.7 if kern.find("pLeft") == -1 else 0.4,
            thresh=0.05,
        )
        sns.scatterplot(data=joindf, x=xcol, y=ycol, alpha=0.1, ax=ax[i - 1])
        ax[i - 1].set_title(
            f"Basis function {i}: $R^2 = {np.corrcoef(joindf[xcol], joindf[ycol])[0, 1]:0.3f}$"
        )
    plt.suptitle(f"{OLDDATE} vs {NEWDATE}:\n{kern} p-values per-neuron compared.")
    plt.tight_layout()
