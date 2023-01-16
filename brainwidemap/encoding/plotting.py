# Third party libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from sklearn.decomposition import PCA

# IBL libraries
import brainbox.io.one as bbone
from brainbox.plot import peri_event_time_histogram
from brainbox.singlecell import calculate_peths
from ibllib.atlas import AllenAtlas
from one.api import ONE


def peth_from_eid_blocks(eid, probe_idx, unit, one=None):
    if not one:
        one = bbone.ONE()
    trialsdf = bbone.load_trials_df(eid, one=one, t_before=0.6, t_after=0.6)
    trialsdf = trialsdf[np.isfinite(trialsdf.stimOn_times)]
    probestr = 'probe0' + str(probe_idx)
    spikes, clusters = bbone.load_spike_sorting(eid, one=one, probe=probestr)
    spkt, spk_clu = spikes[probestr].times, spikes[probestr].clusters
    fig, ax = plt.subplots(2, 1, figsize=(4, 12), gridspec_kw={'height_ratios': [1, 2]})
    highblock_t = trialsdf[trialsdf.probabilityLeft == 0.8].stimOn_times
    lowblock_t = trialsdf[trialsdf.probabilityLeft == 0.2].stimOn_times
    peri_event_time_histogram(spkt,
                              spk_clu,
                              highblock_t,
                              unit,
                              t_before=0.6,
                              t_after=0.6,
                              error_bars='sem',
                              ax=ax[0],
                              pethline_kwargs={
                                  'lw': 2,
                                  'color': 'orange',
                                  'label': 'High probability L'
                              },
                              errbar_kwargs={
                                  'color': 'orange',
                                  'alpha': 0.5
                              })
    yscale_orig = ax[0].get_ylim()
    yticks_orig = ax[0].get_yticks()[1:]
    peri_event_time_histogram(spkt,
                              spk_clu,
                              lowblock_t,
                              unit,
                              t_before=0.6,
                              t_after=0.6,
                              error_bars='sem',
                              ax=ax[0],
                              pethline_kwargs={
                                  'lw': 2,
                                  'color': 'blue',
                                  'label': 'Low probability L'
                              },
                              errbar_kwargs={
                                  'color': 'blue',
                                  'alpha': 0.5
                              })
    yscale_new = ax[0].get_ylim()
    ax[0].set_ylim([min(yscale_orig[0], yscale_new[0]), max(yscale_orig[1], yscale_new[1])])
    ax[0].set_yticks(np.append(ax[0].get_yticks(), yticks_orig))
    ax[0].legend()
    _, binned = calculate_peths(spkt,
                                spk_clu, [unit],
                                trialsdf.stimOn_times,
                                pre_time=0.6,
                                post_time=0.6,
                                bin_size=0.02)
    binned = np.squeeze(binned)
    ax[1].imshow(binned, aspect='auto', cmap='gray_r')
    ax[1].fill_betweenx(range(binned.shape[0]),
                        0,
                        binned.shape[1], (trialsdf.probabilityLeft == 0.8).values,
                        label='P(Left) = 0.8',
                        color='orange',
                        alpha=0.05)
    ax[1].fill_betweenx(range(binned.shape[0]),
                        0,
                        binned.shape[1], (trialsdf.probabilityLeft == 0.2).values,
                        label='P(Left) = 0.2',
                        color='blue',
                        alpha=0.05)
    ticks = [0, 30, 60]
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels([-0.6, 0, 0.6])
    ax[1].set_xlim([0, 60])
    return fig, ax


def ridge_plot(df, xcol, ycol, palette=sns.cubehelix_palette(10, rot=-.25, light=.7)):
    g = sns.FacetGrid(df, row=ycol, hue=ycol, aspect=15., height=.5, palette=palette)
    g.map(sns.kdeplot, xcol, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0,
                .2,
                label,
                fontweight="bold",
                color=color,
                ha="left",
                va="center",
                transform=ax.transAxes)

    g.map(label, xcol)
    g.figure.subplots_adjust(hspace=-.25)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    return g


def plot_scalar_on_slice(regions,
                         values,
                         coord=-1000,
                         slice='coronal',
                         mapping='Allen',
                         hemisphere='left',
                         cmap='viridis',
                         background='image',
                         clevels=None,
                         brain_atlas=None,
                         ax=None):
    """
    Function to plot scalar value per allen region on histology slice
    :param regions: array of acronyms of Allen regions
    :param values: array of scalar value per acronym. If hemisphere is 'both' and different values
        want to be shown on each
    hemispheres, values should contain 2 columns, 1st column for LH values, 2nd column for RH
        values
    :param coord: coordinate of slice in um (not needed when slice='top')
    :param slice: orientation of slice, options are 'coronal', 'sagittal', 'horizontal', 'top'
        (top view of brain)
    :param mapping: atlas mapping to use, options are 'Allen', 'Beryl' or 'Cosmos'
    :param hemisphere: hemisphere to display, options are 'left', 'right', 'both'
    :param background: background slice to overlay onto, options are 'image' or 'boundary'
    :param cmap: colormap to use
    :param clevels: min max color levels [cim, cmax]
    :param brain_atlas: AllenAtlas object
    :param ax: optional axis object to plot on
    :return:
    """

    if clevels is None:
        clevels = (np.min(values), np.max(values))

    ba = brain_atlas or AllenAtlas()
    br = ba.regions

    # Find the mapping to use
    map_ext = '-lr'
    map = mapping + map_ext

    region_values = np.zeros_like(br.id) * np.nan

    if len(values.shape) == 2:
        for r, vL, vR in zip(regions, values[:, 0], values[:, 1]):
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0][0]] = vR
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0][1]] = vL
    else:
        for r, v in zip(regions, values):
            region_values[np.where(br.acronym[br.mappings[map]] == r)[0]] = v

        lr_divide = int((br.id.shape[0] - 1) / 2)
        if hemisphere == 'left':
            region_values[0:lr_divide] = np.nan
        elif hemisphere == 'right':
            region_values[lr_divide:] = np.nan
            region_values[0] = np.nan

    if ax:
        fig = ax.figure
    else:
        fig, ax = plt.subplots()

    if background == 'boundary':
        cmap_bound = matplotlib.cm.get_cmap("bone_r").copy()
        cmap_bound.set_under([1, 1, 1], 0)

    if slice == 'coronal':

        if background == 'image':
            ba.plot_cslice(coord / 1e6, volume='image', mapping=map, ax=ax)
            ba.plot_cslice(coord / 1e6,
                           volume='value',
                           region_values=region_values,
                           mapping=map,
                           cmap=cmap,
                           vmin=clevels[0],
                           vmax=clevels[1],
                           ax=ax)
        else:
            ba.plot_cslice(coord / 1e6,
                           volume='value',
                           region_values=region_values,
                           mapping=map,
                           cmap=cmap,
                           vmin=clevels[0],
                           vmax=clevels[1],
                           ax=ax)
            ba.plot_cslice(coord / 1e6,
                           volume='boundary',
                           mapping=map,
                           ax=ax,
                           cmap=cmap_bound,
                           vmin=0.01,
                           vmax=0.8)

    elif slice == 'sagittal':
        if background == 'image':
            ba.plot_sslice(coord / 1e6, volume='image', mapping=map, ax=ax)
            ba.plot_sslice(coord / 1e6,
                           volume='value',
                           region_values=region_values,
                           mapping=map,
                           cmap=cmap,
                           vmin=clevels[0],
                           vmax=clevels[1],
                           ax=ax)
        else:
            ba.plot_sslice(coord / 1e6,
                           volume='value',
                           region_values=region_values,
                           mapping=map,
                           cmap=cmap,
                           vmin=clevels[0],
                           vmax=clevels[1],
                           ax=ax)
            ba.plot_sslice(coord / 1e6,
                           volume='boundary',
                           mapping=map,
                           ax=ax,
                           cmap=cmap_bound,
                           vmin=0.01,
                           vmax=0.8)

    elif slice == 'horizontal':
        if background == 'image':
            ba.plot_hslice(coord / 1e6, volume='image', mapping=map, ax=ax)
            ba.plot_hslice(coord / 1e6,
                           volume='value',
                           region_values=region_values,
                           mapping=map,
                           cmap=cmap,
                           vmin=clevels[0],
                           vmax=clevels[1],
                           ax=ax)
        else:
            ba.plot_hslice(coord / 1e6,
                           volume='value',
                           region_values=region_values,
                           mapping=map,
                           cmap=cmap,
                           vmin=clevels[0],
                           vmax=clevels[1],
                           ax=ax)
            ba.plot_hslice(coord / 1e6,
                           volume='boundary',
                           mapping=map,
                           ax=ax,
                           cmap=cmap_bound,
                           vmin=0.01,
                           vmax=0.8)

    elif slice == 'top':
        if background == 'image':
            ba.plot_top(volume='image', mapping=map, ax=ax)
            ba.plot_top(volume='value',
                        region_values=region_values,
                        mapping=map,
                        cmap=cmap,
                        vmin=clevels[0],
                        vmax=clevels[1],
                        ax=ax)
        else:
            ba.plot_top(volume='value',
                        region_values=region_values,
                        mapping=map,
                        cmap=cmap,
                        vmin=clevels[0],
                        vmax=clevels[1],
                        ax=ax)
            ba.plot_top(volume='boundary',
                        mapping=map,
                        ax=ax,
                        cmap=cmap_bound,
                        vmin=0.01,
                        vmax=0.8)

    return fig, ax
