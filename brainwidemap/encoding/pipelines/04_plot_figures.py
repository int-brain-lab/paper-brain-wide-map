# Standard library
from pathlib import Path

# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# IBL libraries
from iblatlas.atlas import BrainRegions
from iblatlas.plots import plot_swanson
from matplotlib import colors

# Brainwidemap repo imports
from brainwidemap.encoding.params import GLM_FIT_PATH

FITDATE = "2024-07-16"
VARIABLES = [
    "stimonR",
    "stimonL",
    "correct",
    "incorrect",
    "fmoveR",
    "fmoveL",
    # "fmoveR_early",  # Comment/uncomment if early RT split is used.
    # "fmoveL_early",
    "pLeft",
    "pLeft_tr",
    "wheel",
]
DIFFPAIRS = {
    "stim": ["stimonR", "stimonL"],
    "fback": ["correct", "incorrect"],
    "choice": ["fmoveR", "fmoveL"],
}
MIN_UNITS = 20  # Minimum num units in a region to be plotted on swanson
COLOR_RANGE = [5, 95]  # Percentiles of observations to use for capping color map ranges
GLOBAL_CMAP = False  # Whether to use a single, log scale global cmap for all variables
DISTPLOTS = False  # Whether to plot distributions of variables
DIFFPLOTS = True  # Whether to plot differences in drsq for paired variables
ABSDIFF = True  # Whether to plot absolute value of difference or signed difference
ANNOTATE = False  # Whether to annotate brain regions
IMGFMT = "png"  # Format of output image
SAVEPATH = Path("/home/gercek/Projects/results/plots/swanson_maps/")  # Path to save plots

if not SAVEPATH.exists():
    SAVEPATH.mkdir()

if GLOBAL_CMAP:
    if not SAVEPATH.joinpath("global_cmap").exists():
        SAVEPATH.joinpath("global_cmap").mkdir()
    SAVEPATH = SAVEPATH.joinpath("global_cmap")
else:
    if not SAVEPATH.joinpath("local_cmap").exists():
        SAVEPATH.joinpath("local_cmap").mkdir()
    SAVEPATH = SAVEPATH.joinpath("local_cmap")

if DIFFPAIRS:
    DIFFPATH = SAVEPATH.joinpath("diffplots")
    if not DIFFPATH.exists():
        DIFFPATH.mkdir()

if ANNOTATE:
    SAVEPATH = SAVEPATH.joinpath("annotated")
    if not SAVEPATH.exists():
        SAVEPATH.mkdir()
    DIFFPATH = DIFFPATH.joinpath("annotated")
    if not DIFFPATH.exists():
        DIFFPATH.mkdir()

if IMGFMT != "svg":
    SAVEPATH = SAVEPATH.joinpath(IMGFMT + "_plots")
    DIFFPATH = DIFFPATH.joinpath(IMGFMT + "_plots")
    if not SAVEPATH.exists():
        SAVEPATH.mkdir()
    if not DIFFPATH.exists():
        DIFFPATH.mkdir()

fitfile = Path(GLM_FIT_PATH).joinpath(FITDATE + "_glm_fit.pkl")

if not fitfile.exists():
    raise FileNotFoundError(f"Fit folder {fitfile} does not exist")

br = BrainRegions()


def flatmap_variable(
    df, cmap, cmin=COLOR_RANGE[0], cmax=COLOR_RANGE[1], norm=None, plswan_kwargs={}
):
    fig = plt.figure(figsize=(8, 4) if not ANNOTATE else (16, 8))
    ax = fig.add_subplot(111)
    if norm is not None:
        cmap_kwargs = {"norm": norm, **plswan_kwargs}
    elif GLOBAL_CMAP:
        cmap_kwargs = {"norm": colors.LogNorm(vmin=cmin, vmax=cmax, clip=True), **plswan_kwargs}
    else:
        cmap_kwargs = {"vmin": cmin, "vmax": cmax, **plswan_kwargs}
    ax = plot_swanson(
        df.index,
        df.values,
        hemisphere="left",
        cmap=cmap,
        br=br,
        ax=ax,
        annotate=ANNOTATE,
        **cmap_kwargs,
    )
    plt.colorbar(mappable=ax.images[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    return fig, ax


def get_cmap(split):
    """
    for each split, get a colormap defined by Yanliang
    """
    varmaps = {
        "stimonR": "stim",
        "stimonL": "stim",
        "fmoveR": "choice",
        "fmoveL": "choice",
        "correct": "fback",
        "incorrect": "fback",
        "pLeft": "block",
        "pLeft_tr": "block",
        "wheel": "wheel",
    }
    dc = {
        "stim": ["#ffffff", "#D5E1A0", "#A3C968", "#86AF40", "#517146"],
        "choice": ["#ffffff", "#F8E4AA", "#F9D766", "#E8AC22", "#DA4727"],
        "fback": ["#ffffff", "#F1D3D0", "#F5968A", "#E34335", "#A23535"],
        "block": ["#ffffff", "#D0CDE4", "#998DC3", "#6159A6", "#42328E"],
        "wheel": ["#ffffff", "#C2E1EA", "#95CBEE", "#5373B8", "#324BA0"],
    }
    return colors.LinearSegmentedColormap.from_list("mycmap", dc[varmaps[split]])


fitdata = pd.read_pickle(fitfile)
# Distribution of full model R^2 values, and std. dev between folds
meanscores = fitdata["mean_fit_results"]
full_model = meanscores["full_model"].copy()
full_model_std = (
    fitdata["fit_results"].groupby(["eid", "pid", "clu_id"]).agg({"full_model": "std"})
)
joindf = full_model_std.join(full_model, how="inner", lsuffix="_std")

if DISTPLOTS:
    sns.histplot(
        data=joindf,
        x="full_model",
        y="full_model_std",
        bins=(np.arange(-0.05, 0.25, 0.002), np.arange(0, 0.05, 3e-4)),
        cbar=True,
    )

    # plot relative distributions of each regressor's drsq
    fig = plt.Figure()
    sns.kdeplot(meanscores[VARIABLES], clip=[-0.015, 0.04])
    ax = plt.gca()
    ax.set_xlim(-0.015, 0.03)
    plt.tight_layout()

unitcounts = meanscores.groupby("region").size().astype(int)
keepreg = unitcounts[unitcounts >= MIN_UNITS].index
if GLOBAL_CMAP:
    allmeans = meanscores.set_index("region", append=True)[VARIABLES].groupby("region").mean()
    cmin = np.percentile(allmeans.values.flatten(), COLOR_RANGE[0])
    if cmin < 0:
        cmin = 1e-5
    cmax = np.percentile(allmeans.values.flatten(), COLOR_RANGE[1])
    if cmax < 1e-2:
        cmax = 1e-2  # HACK to make sure the 10^-2 tick gets drawn
for variable in VARIABLES:
    varscores = meanscores.set_index("region", append=True)[variable]
    mask = varscores.index.isin(keepreg, level="region")
    varscores = varscores.loc[:, :, :, mask, :]
    regmeans = varscores.groupby("region").mean()
    if not GLOBAL_CMAP:
        cmin = np.percentile(regmeans, COLOR_RANGE[0])
        cmax = np.percentile(regmeans, COLOR_RANGE[1])
    fig, ax = flatmap_variable(regmeans, get_cmap(variable), cmin, cmax)
    fig.suptitle(f"{variable} $\Delta R^2$")
    fig.savefig(
        SAVEPATH.joinpath(
            f"{variable}{'_annotated' * ANNOTATE}{'_global_cmap' * GLOBAL_CMAP}.{IMGFMT}"
        ),
        format=IMGFMT,
        dpi=450,
    )
    plt.close()

if DIFFPLOTS:
    for key, (var1, var2) in DIFFPAIRS.items():
        cmap = "coolwarm" if not ABSDIFF else get_cmap(var1)
        swan_kwargs = {"empty_color": "silver" if cmap != "coolwarm" else "white"}
        varscores1 = meanscores.set_index("region", append=True)[var1]
        varscores2 = meanscores.set_index("region", append=True)[var2]
        mask = varscores1.index.isin(keepreg, level="region")
        varscores1 = varscores1.loc[:, :, :, mask, :]
        varscores2 = varscores2.loc[:, :, :, mask, :]
        indivdiff = varscores1 - varscores2
        if ABSDIFF:
            indivdiff = indivdiff.abs()
        diff = indivdiff.groupby("region").mean()
        if not GLOBAL_CMAP:
            cmin = np.percentile(diff, COLOR_RANGE[0])
            cmax = np.percentile(diff, COLOR_RANGE[1])
        if cmin < 0:
            norm = colors.TwoSlopeNorm(vcenter=0, vmin=cmin, vmax=cmax)
        else:
            if ABSDIFF:
                norm = colors.Normalize(vmin=0, vmax=cmax)
            else:
                norm = colors.TwoSlopeNorm(vcenter=0, vmin=-cmax, vmax=cmax)
        fig, ax = flatmap_variable(diff, cmap, norm=norm)
        fig.suptitle(f"{var1} $\Delta R^2$ - {var2} $\Delta R^2$")
        fig.savefig(
            DIFFPATH.joinpath(
                f"{var1}_{var2}_{'abs' * ABSDIFF}diff{'_annotated' * ANNOTATE}.{IMGFMT}"
            ),
            format=IMGFMT,
            dpi=450,
        )
        plt.close()
