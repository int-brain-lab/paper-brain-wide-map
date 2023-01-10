# Standard library
import os
from pathlib import Path

# Third party libraries
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.transforms import Affine2D


# IBL libraries
from ibllib.atlas import AllenAtlas, BrainRegions
from ibllib.atlas.plots import plot_scalar_on_slice
from ibllib.atlas.flatmaps import plot_swanson


def atlas_variable(df, cmap, vmin=0, vmax=1, axes=None, fig=None, cbar=False, atlas=None):
    if axes is not None and fig is None:
        raise ValueError("If axes is not None, fig must be provided")
    if axes is not None and len(axes) != 3:
        raise ValueError("Axes must be a list of 3 axes for 3 slices")
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    if atlas is None:
        atlas = AllenAtlas()
    oldshape = axes.shape
    axes = axes.flatten()
    # Plot top view
    plot_scalar_on_slice(
        df.index,
        df.values.reshape(-1),
        slice="top",
        hemisphere="left",
        cmap=cmap,
        mapping="Beryl",
        background="boundary",
        clevels=[vmin, vmax],
        brain_atlas=atlas,
        ax=axes[0],
    )
    limsx = axes[0].get_xlim()
    limsy = axes[0].get_ylim()
    axes[0].set_ylim(limsx)
    axes[0].set_xlim(limsy)
    r = Affine2D().rotate_deg(-90) + axes[0].transData
    for ob in axes[0].images + axes[0].lines + axes[0].collections:
        ob.set_transform(r)
    axes[0].set_ylim([0, 5300])

    plot_scalar_on_slice(
        df.index,
        df.values.reshape(-1),
        coord=-1500,
        slice="sagittal",
        cmap=cmap,
        mapping="Beryl",
        background="boundary",
        clevels=[vmin, vmax],
        brain_atlas=atlas,
        ax=axes[1],
    )
    plot_swanson(
        df.index,
        df.values.reshape(-1),
        cmap=cmap,
        hemisphere="left",
        br=atlas.regions,
        vmin=vmin,
        vmax=vmax,
        ax=axes[2],
    )
    cb_ax = None
    if cbar:
        fig.subplots_adjust(right=0.90)
        cb_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
        plt.colorbar(mappable=axes[2].images[0], cax=cb_ax)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
    axes = axes.reshape(oldshape)
    return fig, axes, cb_ax


if __name__ == "__main__":
    import itertools as it
    BASEPATH = Path("/home/berk/Documents/Projects/results/glms/merged_results")
    FITDATE = "2022-10-24"
    ALPHA = 0.01
    TWOTAIL = True
    KERNCOLORS = False
    NOTIMECOURSE = True
    ALLFOLDS_SIG = False
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
    notkerns = [
        "stimdiffs",
    ]
    if not KERNCOLORS:
        COLOR_MAPS = {k: "viridis" for k in COLOR_MAPS.keys()}
    COLOR_RANGE = [0.0, 0.5] if not ALLFOLDS_SIG else [0.0, 0.3]
    MINUNITS = 20
    atlas = AllenAtlas()
    br = BrainRegions()
    fitpath = BASEPATH.joinpath(f"{FITDATE}_impostor_run")
    fitfiles = [fn for fn in os.listdir(fitpath) if fn.find("percentiles.parquet") > 0]
    kerns = [fn[: fn.find("_percentiles.parquet")] for fn in fitfiles]

    argtuples = it.product([True, False], [True, False], [True, False])
    for TWOTAIL, NOTIMECOURSE, ALLFOLDS_SIG in argtuples:
        COLOR_RANGE = [0.0, 0.5] if not ALLFOLDS_SIG else [0.0, 0.25]
        print(f"TWOTAIL: {TWOTAIL}, NOTIMECOURSE: {NOTIMECOURSE}, ALLFOLDS_SIG: {ALLFOLDS_SIG}")
        for file, kern in zip(fitfiles, kerns):
            if kern in notkerns:
                continue
            fitdata = pd.read_parquet(fitpath.joinpath(file))
            if FITDATE == "2022-03-22" and len(fitdata.columns) > 1:
                fitdata = fitdata.loc[:, fitdata.columns[4::5]]
            ncols = len(fitdata.columns)
            corr_alpha = ALPHA / ncols
            regcounts = fitdata.index.get_level_values("region").value_counts()
            nfolds = fitdata.index.get_level_values("fold").max() + 1
            keepreg = regcounts[regcounts > (MINUNITS * nfolds)].index
            fitdata = fitdata.loc[fitdata.index.get_level_values("region").isin(keepreg)]
            if TWOTAIL:
                if NOTIMECOURSE:
                    sig = (2 * (fitdata - 0.5).abs() > (1 - corr_alpha)).any(axis=1)
                else:
                    sig = 2 * (fitdata - 0.5).abs() > (1 - ALPHA)
                if ALLFOLDS_SIG:
                    sig = sig.groupby(list(filter(lambda n: n != "fold", sig.index.names))).all()
                propsig = sig.groupby("region").agg("mean")
            else:
                if NOTIMECOURSE:
                    uppersig = (fitdata > (1 - corr_alpha)).any(axis=1).to_frame(name="significance")
                    lowersig = (fitdata < corr_alpha).any(axis=1).to_frame(name="significance")
                else:
                    uppersig = fitdata > (1 - ALPHA)
                    lowersig = fitdata < ALPHA
                if ALLFOLDS_SIG:
                    uppersig = uppersig.groupby(
                        list(filter(lambda n: n != "fold", uppersig.index.names))
                    ).all()
                    lowersig = lowersig.groupby(
                        list(filter(lambda n: n != "fold", lowersig.index.names))
                    ).all()
                lowerprop = lowersig.groupby("region").agg("mean")
                upperprop = uppersig.groupby("region").agg("mean")
                propsig = lowerprop.join(upperprop, lsuffix="_lower", rsuffix="_upper")

            if NOTIMECOURSE:
                ncols = 1
            fig, ax = plt.subplots(
                3 * ((not TWOTAIL) + 1),
                ncols,
                figsize=(ncols * 5, 8 * ((not TWOTAIL) + 1)),
            )
            plt.suptitle(
                f"{kern} Proportion neurons significant per basis\n"
                + (not TWOTAIL) * "(sep tail tested)"
                + f" ($\\alpha = {ALPHA}$)"
            )
            plt.tight_layout()
            if ncols == 1:
                ax = ax.reshape(-1, 1)
            plotcols = fitdata.columns if not NOTIMECOURSE else ["significance"]
            for i, col in enumerate(plotcols):
                cbar = i == ncols - 1
                colname = "" if ncols == 1 else f"{float(col):0.3f}s "
                if TWOTAIL:
                    ax[0, i].set_title(colname)
                    atlas_variable(
                        propsig[col] if not NOTIMECOURSE else propsig,
                        cmap=COLOR_MAPS[kern],
                        vmin=COLOR_RANGE[0],
                        vmax=COLOR_RANGE[1],
                        axes=ax[:, i],
                        fig=fig,
                        cbar=cbar,
                    )
                else:
                    atlas_variable(
                        propsig[col + "_lower"],
                        cmap=COLOR_MAPS[kern],
                        vmin=COLOR_RANGE[0],
                        vmax=COLOR_RANGE[1],
                        axes=ax[::2, i],
                        fig=fig,
                    )
                    atlas_variable(
                        propsig[col + "_upper"],
                        cmap=COLOR_MAPS[kern],
                        vmin=COLOR_RANGE[0],
                        vmax=COLOR_RANGE[1],
                        axes=ax[1::2, i],
                        fig=fig,
                        cbar=cbar,
                    )
                    for iax in ax[::2, i]:
                        iax.set_title(colname + "lower sig")
                    for iax in ax[1::2, i]:
                        iax.set_title(colname + "upper sig")
            plt.savefig(
                fitpath.joinpath(
                    f"{kern}_proportion_neurons_sig"
                    + (not TWOTAIL) * "_septails"
                    + (NOTIMECOURSE) * "_allbases"
                    + (ALLFOLDS_SIG) * "_allfoldssig"
                    + ".svg"
                ),
                dpi=300,
            )
            plt.close()
