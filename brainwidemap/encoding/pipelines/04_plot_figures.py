# Standard library
from pathlib import Path

# Third party libraries
import matplotlib.pyplot as plt
from matplotlib import colors
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
COLOR_RANGE = [5, 95]
GLOBAL_CMAP = True
ANNOTATE = False
IMGFMT = "pdf"
SAVEPATH = "/home/berk/Documents/Projects/results/plots/"

fitfile = Path(GLM_FIT_PATH).joinpath(FITDATE + "_glm_fit.pkl")

if not fitfile.exists():
    raise FileNotFoundError(f"Fit folder {fitfile} does not exist")

br = BrainRegions()


def flatmap_variable(df, cmap, cmin=COLOR_RANGE[0], cmax=COLOR_RANGE[1]):
    fig = plt.figure(figsize=(8, 4) if not ANNOTATE else (16, 8))
    ax = fig.add_subplot(111)
    cmap_kwargs = {
        "vmin": cmin,
        "vmax": cmax
    } if not GLOBAL_CMAP else {
        "norm": colors.LogNorm(vmin=cmin, vmax=cmax, clip=True)
    }
    ax = plot_swanson(df.index,
                      df.values,
                      hemisphere="left",
                      cmap=cmap,
                      br=br,
                      ax=ax,
                      annotate=ANNOTATE,
                      **cmap_kwargs)
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
full_model_std = (fitdata["fit_results"].groupby(["eid", "pid", "clu_id"
                                                  ]).agg({"full_model":
                                                          "std"}))
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

unitcounts = meanscores.groupby("region").size().astype(int)
keepreg = unitcounts[unitcounts >= MIN_UNITS].index
if GLOBAL_CMAP:
    allmeans = meanscores.set_index(
        "region", append=True)[VARIABLES].groupby("region").mean()
    cmin = np.percentile(allmeans.values.flatten(), COLOR_RANGE[0])
    if cmin < 0:
        cmin = 1e-5
    cmax = np.percentile(allmeans.values.flatten(), COLOR_RANGE[1])
for variable in VARIABLES:
    varscores = meanscores.set_index("region", append=True)[variable]
    mask = varscores.index.isin(keepreg, level="region")
    varscores = varscores.loc[:, :, :, mask, :]
    regmeans = varscores.groupby("region").mean()
    if not GLOBAL_CMAP:
        cmin = np.percentile(regmeans, 0)
        cmax = np.percentile(regmeans, 100)
    fig, ax, cb_ax = flatmap_variable(regmeans, COLOR_MAPS[variable], cmin,
                                      cmax)
    fig.suptitle(f"{variable} significance levels")
    fig.savefig(
        SAVEPATH.joinpath(
            f"{variable}_significance{'_annotated' * ANNOTATE}{'_global_cmap' * GLOBAL_CMAP}.{IMGFMT}"
        ),
        format=IMGFMT,
        dpi=450,
    )
    plt.close()
