from ibllib.atlas import BrainRegions
import numpy as np
import pandas as pd
from ibllib.atlas.flatmaps import plot_swanson
from pathlib import Path
from brainwidemap.encoding.params import GLM_FIT_PATH
import matplotlib.pyplot as plt

FITDATE = "2022-10-24"
VARIABLES = [
    "stimonR", "stimonL", "correct", "incorrect", "fmoveR", "fmoveL", "pLeft", "pLeft_tr", "wheel"
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
IMGFMT = 'pdf'

fitfolder = Path(GLM_FIT_PATH).joinpath("merged_results").joinpath(FITDATE + "_impostor_run")

if not fitfolder.exists():
    raise FileNotFoundError(f"Fit folder {fitfolder} does not exist")

br = BrainRegions()


def flatmap_variable(df, cmap):
    fig = plt.figure(figsize=(8, 4) if not ANNOTATE else (16, 8))
    ax = fig.add_subplot(111)
    ax = plot_swanson(df.index,
                      df.values,
                      hemisphere="left",
                      cmap=cmap,
                      br=br,
                      ax=ax,
                      annotate=ANNOTATE,
                      vmin=COLOR_RANGE[0],
                      vmax=COLOR_RANGE[1])
    fig.subplots_adjust(right=0.85)
    cb_ax = fig.add_axes([0.88, 0.25, 0.02, 0.5])
    plt.colorbar(mappable=ax.images[0], cax=cb_ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    return fig, ax, cb_ax


for variable in VARIABLES:
    var_percentiles = pd.read_parquet(fitfolder.joinpath(f"{variable}_percentiles.parquet"))
    nfolds = var_percentiles.groupby("fold").ngroups
    unitcounts = (var_percentiles.groupby('region').size() / nfolds).astype(int)
    keepreg = unitcounts[unitcounts >= MIN_UNITS].index
    mask = var_percentiles.index.isin(keepreg, level='region')
    var_percentiles = var_percentiles.loc[:, :, :, mask, :]
    twotail = np.abs(var_percentiles - 0.5) * 2
    meanp = twotail.groupby(['eid', 'probe', 'clu_id', 'region']).mean()
    corr_alpha = ALPHA / len(var_percentiles.columns)  # Bonferroni correction
    significance = np.any(meanp >= (1 - corr_alpha), axis=1)
    propsig = significance.groupby('region').mean()
    fig, ax, cb_ax = flatmap_variable(propsig, COLOR_MAPS[variable])
    fig.suptitle(f"{variable} significance levels")
    fig.savefig(fitfolder.joinpath(f"{variable}_significance{'_annotated' * ANNOTATE}.{IMGFMT}"),
                format=IMGFMT,
                dpi=600)
