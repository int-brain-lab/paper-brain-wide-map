# Standard library
from pathlib import Path

# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# IBL libraries
from ibllib.atlas import BrainRegions
from ibllib.atlas.flatmaps import plot_swanson

# Brainwidemap repo imports
from brainwidemap.encoding.params import GLM_FIT_PATH

FITDATE = "2023-01-16"
VARIABLES = [
    "stimonR",
    "stimonL",
    "correct",
    "incorrect",
    "fmoveR",
    "fmoveL",
    "pLeft",
    "pLeft_tr",
    "wheel",
]
ALPHA = 0.02
MIN_UNITS = 20
COLOR_MAPS = {
    "stimonR": "Greens",
    "stimonL": "Greens",
    "correct": "Reds",
    "incorrect": "Reds",
    "fmoveR": "Oranges",
    "fmoveL": "Oranges",
    "pLeft": "Blues",
    "pLeft_tr": "Blues",
    "wheel": "Purples",
}
COLOR_RANGE = [0.0, 0.5]
ANNOTATE = False
IMGFMT = "pdf"

fitfile = Path(GLM_FIT_PATH).joinpath(FITDATE + "_glm_fit.pkl")

if not fitfile.exists():
    raise FileNotFoundError(f"Fit folder {fitfile} does not exist")

br = BrainRegions()


def flatmap_variable(df, cmap):
    fig = plt.figure(figsize=(8, 4) if not ANNOTATE else (16, 8))
    ax = fig.add_subplot(111)
    ax = plot_swanson(
        df.index,
        df.values,
        hemisphere="left",
        cmap=cmap,
        br=br,
        ax=ax,
        annotate=ANNOTATE,
        vmin=COLOR_RANGE[0],
        vmax=COLOR_RANGE[1],
    )
    fig.subplots_adjust(right=0.85)
    cb_ax = fig.add_axes([0.88, 0.25, 0.02, 0.5])
    plt.colorbar(mappable=ax.images[0], cax=cb_ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    return fig, ax, cb_ax


fitdata = pd.read_pickle(fitfile)
# Distribution of full model R^2 values, and std. dev between folds
meanscores = fitdata["mean_fit_results"]
full_model = meanscores["full_model"].copy()
full_model_std = (
    fitdata["fit_results"].groupby(["eid", "pid", "clu_id"]).agg({"full_model": "std"})
)
joindf = full_model_std.join(full_model, how="inner", lsuffix="_std")
sns.histplot(
    data=joindf,
    x="full_model",
    y="full_model_std",
    bins=(np.arange(-0.05, 0.25, 0.002), np.arange(0, 0.05, 3e-4)),
    cbar=True,
)

# plot relative distributions of each regressor's drsq
fig = plt.fig()
sns.kdeplot(meanscores[VARIABLES], clip=[-0.015, 0.04])
ax = plt.gca()
ax.set_xlim(-0.015, 0.03)
plt.tight_layout()



for variable in VARIABLES:
    var_percentiles = pd.read_parquet(fitfile.joinpath(f"{variable}_percentiles.parquet"))
    nfolds = var_percentiles.groupby("fold").ngroups
    unitcounts = (var_percentiles.groupby("region").size() / nfolds).astype(int)
    keepreg = unitcounts[unitcounts >= MIN_UNITS].index
    mask = var_percentiles.index.isin(keepreg, level="region")
    var_percentiles = var_percentiles.loc[:, :, :, mask, :]
    twotail = np.abs(var_percentiles - 0.5) * 2
    meanp = twotail.groupby(["eid", "probe", "clu_id", "region"]).mean()
    corr_alpha = ALPHA / len(var_percentiles.columns)  # Bonferroni correction
    significance = np.any(meanp >= (1 - corr_alpha), axis=1)
    propsig = significance.groupby("region").mean()
    fig, ax, cb_ax = flatmap_variable(propsig, COLOR_MAPS[variable])
    fig.suptitle(f"{variable} significance levels")
    fig.savefig(
        fitfolder.joinpath(f"{variable}_significance{'_annotated' * ANNOTATE}.{IMGFMT}"),
        format=IMGFMT,
        dpi=600,
    )
