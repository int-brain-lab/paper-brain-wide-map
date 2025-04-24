import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.use("Qt5Agg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from pathlib import Path

from collections import Counter, OrderedDict
from functools import reduce
import os, sys
import itertools
from scipy import stats
from statsmodels.stats.multitest import multipletests
import subprocess


from one.api import ONE
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
from iblatlas.plots import plot_swanson_vector, plot_scalar_on_slice
import iblatlas


from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text
import matplotlib.collections
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from PIL import Image
import dataframe_image as dfi
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.ticker as tck
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from brainwidemap import download_aggregate_tables, bwm_units, bwm_query
from brainwidemap.encoding.design import generate_design
from brainwidemap.encoding.glm_predict import GLMPredictor, predict
from brainwidemap.encoding.utils import (load_regressors, single_cluster_raster, find_trial_ids)

from brainwidemap.manifold.state_space_bwm import plot_strip_sampling
from brainwidemap.meta.oscillations import single_cell_psd, T_BIN

from brainbox.plot import peri_event_time_histogram
from reproducible_ephys_functions import LAB_MAP as labs

import neurencoding.linear as lm
from neurencoding.utils import remove_regressors

import warnings
#warnings.filterwarnings("ignore")

from ibl_style.style import figure_style
from ibl_style.utils import get_coords, add_label, MM_TO_INCH
import figrid as fg
import matplotlib.ticker as ticker

'''
This script is used to plot the main result figures of the BWM paper.
The raw results from each analysis can be found in bwm_figs_res.
There are 4 analyses: population trajectory, decoding, glm, single-cell
See first function in code block of this script for each analysis type
for data format conversion.
'''


ba = AllenAtlas()
br = BrainRegions()
one = ONE()
# one = ONE(base_url='https://openalyx.internationalbrainlab.org',
#          password='international', silent=True)

# -------------------------------------------------------------------------------------------------
# Constants and definitions
# -------------------------------------------------------------------------------------------------

VARIABLES = ['stim', 'choice', 'fback']
variverb = dict(zip(VARIABLES, ['Stimulus', 'Choice', 'Feedback']))
variverb.update({'speed': 'Wheel-speed', 'velocity': 'Wheel-velocity'})


ANALYSIS_TYPES = {
    'decoding_effect': {
        'label': 'Decoding $R^2$ over null',
        'analysis': 'Decoding',
        'info': 'Regularized logistic regression'
    },
    'mannwhitney_effect': {
        'label': 'Fraction of \n significant cells',
        'analysis': 'Single-cell statistics',
        'info': 'C.C Mann-Whitney test'
    },
    'euclidean_effect': {
        'label': 'Normalized euclidean \n distance (Hz)',
        'analysis': 'Population trajectory',
        'info': 'Distance between trajectories'
    },
    'euclidean_latency': {
        'label': 'Latency of distance (s)',
        'analysis': 'Population trajectory',
        'info': 'Time near peak'
    },
    'glm_effect': {
        'label': 'Absolute difference $\\Delta R^2$ (log)',
        'analysis': 'Encoding',
        'info': 'General linear model'}
}

EVENT_LABELS = {
    'stim': 'Time from stim (s)',
    'choice': 'Time from move (s)',
    'fback': 'Time from feedback (s)',
    'speed': 'Time from move (s)',
    'velocity': 'Time from move (s)',
}

EVENT_LINES = {
    'stim': 'Stimulus onset',
    'choice': 'Movement onset',
    'fback': 'Feedback onset',
    'speed': 'Movement onset',
    'velocity': 'Movement onset',
}

EVENT_LINE_LOC = {
    'stim': 1.2,
    'choice': 2,
    'fback': 3.9
}

# Canonical colors for different trial types
BLUE = [0.13850039, 0.41331206, 0.74052025]
RED = [0.66080672, 0.21526712, 0.23069468]

# -------------------------------------------------------------------------------------------------
# Paths to results
# -------------------------------------------------------------------------------------------------
 
# pooled results
meta_pth = Path(one.cache_dir, 'bwm_res', 'bwm_figs_data', 'meta')
meta_pth.mkdir(parents=True, exist_ok=True)

# main fig panels
imgs_pth = Path(one.cache_dir, 'bwm_res', 'bwm_figs_imgs')
for variable in VARIABLES:
    Path(imgs_pth, variable).mkdir(parents=True, exist_ok=True)         

def imgs_variable_path(variable):
    return imgs_pth.joinpath(variable)

# decoding results
dec_pth = Path(one.cache_dir, 'bwm_res', 'bwm_figs_data','decoding')
dec_pth.mkdir(parents=True, exist_ok=True)  

# population trajectory results
man_pth = Path(one.cache_dir, 'bwm_res', 'bwm_figs_data','trajectory')
man_pth.mkdir(parents=True, exist_ok=True)

# encoding results
enc_pth = Path(one.cache_dir, 'bwm_res', 'bwm_figs_data', 'encoding')
enc_pth.mkdir(parents=True, exist_ok=True)


# single_cell (MannWhitney) results
sc_pth = Path(one.cache_dir, 'bwm_res', 'bwm_figs_data', 'single_cell')
sc_pth.mkdir(parents=True, exist_ok=True)

pth_res = Path(one.cache_dir,'bwm_res', 'manifold', 'res')
pth_res.mkdir(parents=True, exist_ok=True)

sigl = 0.05  # significance level (for stacking, plotting, fdr)

plt.ion()  # interactive plotting on

# -------------------------------------------------------------------------------------------------
# Figure styling
# -------------------------------------------------------------------------------------------------

f_size = 8  # font size large
f_size_s = 0.7 * f_size # font size small

# mpl.rcParams['figure.autolayout']  = True
# mpl.rcParams.update({'font.size': f_size})


figure_style()
f_size = mpl.rcParams['font.size']
f_size_s = mpl.rcParams['xtick.labelsize']

title_size = 7
label_size = 7
text_size = 6
f_size_l = title_size
f_size = label_size
f_size_s = text_size
f_size_xs = 5

mpl.rcParams['xtick.minor.visible'] = False
mpl.rcParams['ytick.minor.visible'] = False

mpl.rcParams['pdf.fonttype']=42
# mpl.rcParams['xtick.major.size'] = 4
# mpl.rcParams['ytick.major.size'] = 4

handle_length = 1
handle_pad = 0.5
#
# # Set the default MaxNLocator for ticks
# mpl.rcParams['axes.xmargin'] = 0  # Prevent extra margins
# mpl.rcParams['axes.ymargin'] = 0
# mpl.rcParams['xtick.major.pad'] = 2
# mpl.rcParams['ytick.major.pad'] = 2

def set_max_ticks(ax, num_ticks=4):
    x_ticks = len(ax.get_xticks())
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=np.min([x_ticks, num_ticks])))
    y_ticks = len(ax.get_yticks())
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=np.min([y_ticks, num_ticks])))

# -------------------------------------------------------------------------------------------------
# Plotting utils
# -------------------------------------------------------------------------------------------------

def adjust_subplots(fig, adjust=5, extra=2):
    width, height = fig.get_size_inches() / MM_TO_INCH
    if not isinstance(adjust, int):
        assert len(adjust) == 4
    else:
        adjust = [adjust] *  4
    fig.subplots_adjust(top=1 - adjust[0] / height, bottom=(adjust[1] + extra) / height,
                        left=adjust[2] / width, right=1 - adjust[3] / width)


def plot_vertical_swanson(acronyms, scores, mask=None, ax=None, cmap='viridis', vmin=None, vmax=None, fontsize=7,
                          annotate_kwargs=dict(), cbar=True, cbar_label=None, cbar_shrink=0.8, legend=True,
                          mask_label=None):
    """
    Plot a vertical swanson figure with optional colorbar and legend
    :return:
    """

    ax = plot_swanson_vector(acronyms, scores, hemisphere=None, orientation='portrait', cmap=cmap, br=br, ax=ax,
                             empty_color='white', linewidth=0.1, mask=mask, mask_color='silver', vmin=vmin, vmax=vmax,
                             fontsize=fontsize, **annotate_kwargs)

    ax.set_axis_off()
    ax.axes.invert_xaxis()

    if cbar:
        cbar_kwargs = {'shrink': cbar_shrink, 'aspect': 12, 'pad': 0.025}
        cax = add_cbar(cmap, vmin, vmax, ax, cbar_label, cbar_kwargs=cbar_kwargs)

    if legend:
        leg = add_sig_legends(ax, mask_label)
        leg.set_bbox_to_anchor((0.65, 0.11))

    return ax, cax if cbar else None


def plot_horizontal_swanson(acronyms, scores, mask=None, ax=None, cmap='viridis', vmin=None, vmax=None, fontsize=7,
                            annotate_kwargs=dict(), cbar=True, cbar_label=None, cbar_shrink=0.45, legend=True,
                            mask_label=None):

    """
    Plot a horzontal swanson figure with optional colorbar and legend
    :return:
    """

    ax = plot_swanson_vector(acronyms, scores, hemisphere=None, orientation='landscape', cmap=cmap, br=br, ax=ax,
                             empty_color='white', linewidth=0.1, mask=mask, mask_color='silver', vmin=vmin, vmax=vmax,
                             fontsize=fontsize, **annotate_kwargs)

    ax.set_axis_off()

    if cbar:
        cbar_kwargs = {'shrink': cbar_shrink, 'aspect': 12, 'pad': 0.025}
        cax = add_cbar(cmap, vmin, vmax, ax, cbar_label, cbar_kwargs=cbar_kwargs)

    if legend:
        if cbar:
            bbox = cax.ax.get_position()
            cax.ax.set_position([bbox.x0 - 0.15, bbox.y0, bbox.width, bbox.height])
        leg = add_sig_legends(ax, mask_label)
        leg.set_bbox_to_anchor((1, -0.05))

    return ax, cax if cbar else None


def add_sig_legends(ax, mask_label='Not significant'):
    """
    Add legend elements to an axis to indicate regions in mask (silver) and regions not analyzed (white)
    :return:
    """

    legend_elements = [
        Rectangle((0, 0), 5, 5, facecolor='silver', edgecolor='black', linewidth=0.5,
                  label=mask_label),
        Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', linewidth=0.5,
                  label='Not analyzed')
    ]

    legend = ax.legend(handles=legend_elements, frameon=False,
                       fontsize=f_size_s, handlelength=0.7, handletextpad=0.5)
    return legend


def add_cbar(cmap, vmin, vmax, ax, label, cbar_kwargs=dict(), associated=True):
    """
    Add a colorbar to an axis
    :return:
    """
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    if associated: # whether the axis passed in is stand alone or associated to another plot
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                            orientation="horizontal", **cbar_kwargs)
    else:
        cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')

    ticks = np.round(np.linspace(vmin, vmax, num=3), 2)
    cbar.set_ticks(ticks)
    cbar.ax.xaxis.set_tick_params(pad=5, labelsize=f_size_s)
    cbar.outline.set_visible(False)
    cbar.set_label(label, fontsize=f_size, ha='center', va='top')

    return cbar


def get_cmap_for_variable(variable):
    """
    Get colormap for each variable
    Defined by Yanliang (updated by Chris)
    """

    cmap_lookup = {
        'stim': ["#EAF4B3","#D5E1A0", "#A3C968",
                    "#86AF40", "#517146","#33492E"],
        'choice': ["#F8E4AA","#F9D766","#E8AC22",
                   "#DA4727","#96371D"],
        'fback': ["#F1D3D0","#F5968A","#E34335",
                  "#A23535","#842A2A"],
        'block': ["#D0CDE4","#998DC3","#6159A6",
                  "#42328E", "#262054"],
        'speed': ["#D0CDE4","#998DC3","#6159A6",
                  "#42328E", "#262054"],
        'velocity': ["#D0CDE4","#998DC3","#6159A6",
                  "#42328E", "#262054"],
        'move': ["#c1e1ea", "#91d0fa", "#5da5f1",
                 "#2a71d7", "#0045b1"]}

    if '_' in variable:
        variable = variable.split('_')[0]

    return LinearSegmentedColormap.from_list("mycmap", cmap_lookup[variable])


def adjust_vertical_swansons(positions, ax):

    # x values are absolute (3300 is the left side, -300 the right side),
    # y values are relative to the original position (-ve shifts up, +ve shifts down)

    xlim = ax.get_xlim()
    x_thres = xlim[0] / 2
    regions = list(positions.keys())
    text_objects = [obj for obj in ax.get_children() if isinstance(obj, Text)]

    for object in text_objects:
        text = object.get_text()
        if text in br.acronym:
            if text in regions:
                x0, y0 = object.get_position()
                x = positions[text]['x']
                y = y0 + positions[text]['y']
                object.set_position((x, y))
                line = Line2D((x, x0), (y, y0), color='k', lw=0.7)
                ax.add_line(line)
                # ax.annotate('', xy=(x0, y0), xytext=(x, y),
                #             arrowprops=dict(arrowstyle='-', color='k', lw=0.7), annotation_clip=False)
                object.set_horizontalalignment('left') if x < x_thres else object.set_horizontalalignment('right')
            else:
                object.remove()

def adjust_horizontal_swansons(positions, ax):

    # y values are absolute (3300 is the bottom, 0 is top),
    # x values are relative to the original position (-ve shifts left, +ve shifts right)

    xlim = ax.get_xlim()
    x_thres = xlim[1] / 2
    regions = list(positions.keys())
    text_objects = [obj for obj in ax.get_children() if isinstance(obj, Text)]

    for object in text_objects:
        text = object.get_text()
        if text in br.acronym:
            if text in regions:
                x0, y0 = object.get_position()
                x = x0 + positions[text]['x']
                y = positions[text]['y']
                object.set_position((x, y))
                line = Line2D((x, x0), (y, y0), color='k', lw=0.7)
                ax.add_line(line)
                # ax.annotate('', xy=(x0, y0), xytext=(x, y),
                #             arrowprops=dict(arrowstyle='-', color='k', lw=0.7), annotation_clip=False)
                object.set_horizontalalignment('left') if x > x_thres else object.set_horizontalalignment('right')
                object.set_verticalalignment('top') if y > 3000 else object.set_verticalalignment('bottom')
            else:
                object.remove()


# -------------------------------------------------------------------------------------------------
# Atlas utils
# -------------------------------------------------------------------------------------------------

def beryl_to_cosmos(beryl_acronym, br):
    beryl_id = br.id[br.acronym==beryl_acronym]
    return br.get(ids=br.remap(beryl_id, source_map='Beryl', target_map='Cosmos'))['acronym'][0]


def swanson_to_beryl_hex(beryl_acronym, br):
    beryl_id = br.id[br.acronym==beryl_acronym]
    rgb = br.get(ids=beryl_id)['rgb'][0].astype(int)
    return '#' + rgb_to_hex((rgb[0],rgb[1],rgb[2]))


def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb


def get_allen_info():
    data = np.load(Path(one.cache_dir, 'dmn', 'alleninfo.npy'), allow_pickle=True).flat[0]

    # cosmos_indices = np.unique(br.mappings['Cosmos'])
    # acronyms = br.acronym[cosmos_indices]
    # colors = br.rgb[cosmos_indices]

    return data['dfa'], data['palette']

_, REGION_COLS = get_allen_info()

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# META (combined results)
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

ADJUST_SWANSONS = {
    'stim': {
        'decoding_effect': {
            'CL': {'x': -50, 'y': -500}, 'NOT': {'x': -200, 'y': 300}, 'TRN': {'x': 3300, 'y': 500},
            'PRNc': {'x': 3300, 'y': 700}
        },
        'mannwhitney_effect': {
            'VISl': {'x': -50, 'y': -500}, 'NOT': {'x': -200, 'y': 300}, 'PF': {'x': -200, 'y': -100}
        },
        'euclidean_effect': {
            'MOp': {'x': 400, 'y': 300}, 'MOs': {'x': 600, 'y': 300}, 'VISal': {'x': -50, 'y': 0},
            'VISp': {'x': -75, 'y': 50}, 'VISam': {'x': 50, 'y': -300}, 'VISa': {'x': -100, 'y': -600},
            'CL': {'x': -200, 'y':0}, 'PF': {'x': -200, 'y': 100}, 'SNr': {'x': 3300, 'y': 1300},
            'PRNc': {'x': 3300, 'y': 400}, 'GRN': {'x': 3300, 'y': 200}
        },
        'euclidean_latency': {
            'LSv': {'x': 3300, 'y': 0}, 'LGd': {'x': 3300, 'y': 2000}, 'NOT': {'x': -200, 'y': 300},
            'RSPv': {'x': 0, 'y': -100}
        },
        'glm_effect': {
            'SNr': {'x': 3300, 'y': 600}, 'V': {'x': 3300, 'y': 0}, 'PRNc': {'x': 3300, 'y': 400},
            'PF': {'x': -200, 'y': -100}, 'UVU': {'x': 500, 'y': 500}
        },
    },
    'fback': {
        'decoding_effect': {
            'AUDp': {'x': 100, 'y': -400}, 'MARN': {'x': 3300, 'y': 0}, 'GRN': {'x': 3300, 'y': 200},
            'VCO': {'x': 150, 'y': -1000}
        },
        'mannwhitney_effect': {
            'AUDd': {'x': 100, 'y': -400}, 'GPi': {'x': 3300, 'y': -1200}, 'CL': {'x': -200, 'y': -100},
            'NPC': {'x': -200, 'y': 100}, 'GRN': {'x': 3300, 'y': -500}
        },
        'euclidean_effect': {
            'AUDp': {'x': 100, 'y': -400}, 'GPi': {'x': 3300, 'y': -1400}, 'SNc': {'x': 3300, 'y': 1700},
            'NB': {'x': -200, 'y': -100}, 'DN': {'x': 200, 'y': -200}
        },
        'euclidean_latency': {
            'AUDv': {'x': 100, 'y': -400}, 'PRNr': {'x': 3300, 'y': 500}, 'NLL': {'x': 200, 'y': -100},
            'MOB': {'x': 3300, 'y': -100}, 'CUN': {'x': -200, 'y': -1000}
        },
        'glm_effect': {
            'SUT': {'x': 3300, 'y': 0}, 'V': {'x': 3300, 'y': 0}, 'VII': {'x': 3300, 'y': 0},
            'POST': {'x': 50, 'y': -1500}, 'UVU': {'x': 700, 'y': 500}
        },
    },
    'choice': {
        'decoding_effect': {
            'CL': {'x': -200, 'y': -300}, 'SNr': {'x': -200, 'y': 100}, 'GRN': {'x': 3300, 'y': 400},
            'VII': {'x': 3300, 'y': 200}, 'PRNr': {'x': 3300, 'y': 600}
        },
        'mannwhitney_effect': {
            'CL': {'x': -200, 'y': -300}, 'NPC': {'x': -200, 'y': 400}, 'GRN': {'x': 3300, 'y': -400},
            'VII': {'x': 3300, 'y': -500},
        },
        'euclidean_effect': {
            'CL': {'x': -200, 'y': -300}, 'NPC': {'x': -200, 'y': 400}, 'GRN': {'x': 3300, 'y': 200},
            'SNr': {'x': 3300, 'y': 1400},
        },
        'euclidean_latency': {
            'AUDd': {'x': 100, 'y': -400}, 'VISam': {'x': 0, 'y': -100}, 'VISl': {'x': -50, 'y': -200},
            'ECT': {'x': -100, 'y': -200}, 'LD': {'x': 3300, 'y': -1500}
        },
        'glm_effect': {
            'ANcr1': {'x': 0, 'y': 300}, 'V': {'x': 3300, 'y': 0}, 'GRN': {'x': 3300, 'y': 0},
            'UVU': {'x': 700, 'y': 500}
        },
    },
    'speed': {
        'decoding_effect': {
            'LPO': {'x': 3300, 'y': -1400}, 'CLI': {'x': -100, 'y': 200}, 'MARN': {'x': 3300, 'y': 100},
            'GRN': {'x': 3300, 'y': 200}
        },
        'glm_effect': {
            'CLI': {'x': -100, 'y': 100}, 'PRNr': {'x': 3300, 'y': 50}, 'PRNc': {'x': 3300, 'y': 0},
            'VII': {'x': 3300, 'y': 0}, 'MARN': {'x': 3300, 'y': 100}, 'GRN': {'x': 3300, 'y': 200}
        },
    },
    'velocity': {
        'decoding_effect': {
            'LPO': {'x': 3300, 'y': -1400}, 'FOTU': {'x': 200, 'y': 300}, 'MARN': {'x': 3300, 'y': 100},
            'GRN': {'x': 3300, 'y': 200}
        },
        'glm_effect': {
            'SPVO': {'x': 200, 'y': 300}, 'VII': {'x': 3300, 'y': 0}, 'GRN': {'x': 3300, 'y': 100},
        },
    }
}


# -------------------------------------------------------------------------------------------------
# Data pooling
# -------------------------------------------------------------------------------------------------

def pool_results_across_analyses(return_raw=False):

    '''
    input are various csv files from 
    4 different analysis types ['glm','euc', 'mw', 'dec']
    VARIABLES ['stim', ' choice', 'fback']

    some files need conversion to csv (trajectory, glm); 
    see first functions for in the subsequent sections

    '''

    D = {}
    '''
    # encoding (glm)   
    '''

    t = pd.read_pickle(
            Path(enc_pth,'2024-08-12_GLM_WheelSpeed_fit.pkl'))['mean_fit_results']

    # for GLM data restriction
    units_df = bwm_units(one)
    valid_pairs = set(zip(units_df['pid'], units_df['cluster_id']))
    
    t = t.loc[[index for index in t.index if
                     (index[1], index[2]) in valid_pairs]]
                        
    res = [abs(t['stimonL'] - t['stimonR']),
           abs(t['fmoveR']  - t['fmoveL']),
           abs(t['correct']  - t['incorrect'])]

    d0 = dict(zip(VARIABLES,res))
    d = {i: d0[i].to_frame().reset_index() for i in d0}
    
    rr = t['region'].reset_index()
    acs = rr['region'].values

    dd = {}
    for variable in d:
        df = pd.DataFrame(d[variable])
        
        df['region'] = acs
        
        # drop void and root
        df = df[~df['region'].isin(['root','void'])]
        df.drop(['eid', 'pid', 'clu_id'], axis=1, inplace=True)
        df = df.groupby(['region']).mean()
        df.rename(columns = {0: 'glm_effect'}, inplace=True)
        dd[variable] = df               

    D['glm'] = dd
    print('intergated glm results')  
    
    ''' 
    # euclidean (euc) 
    '''
    
    d = {}
    

    for vari in VARIABLES:
        r = []
        variable = vari + '_restr'
        columns = ['region','nclus', 
                'p_euc_can', 'amp_euc_can',
                'lat_euc_can']

        dd = np.load(Path(man_pth, f'{variable}.npy'),
                    allow_pickle=True).flat[0] 
        
        for reg in dd:

            r.append([reg, dd[reg]['nclus'],
                      dd[reg]['p_euc_can'],
                      dd[reg]['amp_euc_can'],
                      dd[reg]['lat_euc_can']])

        d[vari]  = pd.DataFrame(data=r,columns=columns)
    
        d[vari]['euclidean_significant'] = d[vari].p_euc_can.apply(
                                               lambda x: x<sigl)
                                               

        d[vari].rename(columns = {'amp_euc_can': 'euclidean_effect',
                                  'lat_euc_can': 'euclidean_latency'},
                                  inplace=True)
                                  
        d[vari] = d[vari][['region',
                           'euclidean_effect',
                           'euclidean_latency',
                           'euclidean_significant']]

    D['euclidean'] = d
    print('intergated trajectory results')    
    
    '''
    # Mann Whitney (single_cell)
    '''
    
    d = {}   
    mw = pd.read_csv(Path(sc_pth,  
            'Updated_Single_cell_analysis_July_10_2024 - Sheet1.csv'))

    for vari in VARIABLES:
        # that fixes typos
        d[vari] = mw[['Acronym',
                    f'[{vari}] fraction of significance',
                    f'[{vari}] significance']].rename(
                    columns = {'Acronym': 'region',
                    f'[{vari}] fraction of significance':  
                    'mannwhitney_effect',
                    f'[{vari}] significance': 
                    'mannwhitney_significant'})
        d[vari].mannwhitney_significant.replace(
                                np.nan, False,inplace=True)
        d[vari].mannwhitney_significant.replace(
                                1, True,inplace=True)


    
    D['mannwhitney'] = d
    print('intergated MannWhitney results')
    
    '''   
    # decoding (dec)
    '''
   
    d = {}     
              
    for vari in VARIABLES:
    
        d[vari] = pd.read_parquet(Path(dec_pth,
                    f'{vari}_stage3.pqt'))[[
                    'region','valuesminusnull_median',
                    'sig_combined_corrected']].rename(columns = {
                    'valuesminusnull_median': 'decoding_effect',
                    'sig_combined_corrected': 'decoding_significant'})
                
        d[vari].dropna(axis=0,how='any',
            subset=['decoding_effect'], inplace=True)

        
    D['decoding'] = d   
    print('intergated decoding results')
    if return_raw:
        return D
       
    # merge frames across analyses    
    for vari in VARIABLES:
    
        # merge such that only regions are kept for which all 
        # analyses have a score (how='inner')
        
        df_ = reduce(lambda left,right: 
                     pd.merge(left,right,on='region', how='inner'), 
                     [D[ana][vari] for ana in D])
                     
        df_.replace(to_replace=[True, False], 
                      value=[1, 0], inplace=True)
                      
        # ## Apply logarithm to GLM results             
        df_.glm_effect = np.log10(
                df_.glm_effect.clip(lower=1e-5))
                
        # ## Apply logarithm to trajectory results              
        df_.euclidean_effect = np.log10(df_.euclidean_effect)                
                
        
        # Reorder columns to match ordering in Figure       
        df_ = df_[['region',
                   'euclidean_latency',
                   'euclidean_effect',
                   'glm_effect',
                   'mannwhitney_effect',
                   'decoding_effect',
                   'decoding_significant',
                   'mannwhitney_significant',
                   'euclidean_significant']]                        
                                
        df_.to_pickle(meta_pth / f"{vari}.pkl")

        
    print('pooled and saved results at')
    print(meta_pth)   


def pool_wheel_res():

    '''
    For encoding and decoding, 
    pool wheel velocity and speed results
    
    variable in ['speed', 'velocity']
    '''
    D = {}
 
    '''
    encoding (glm) 
    '''

    d = {}
    fs = {'speed': '2024-08-12_GLM_WheelSpeed_fit.pkl',
          'velocity': '2024-09-09_GLM_WheelVel_fit.pkl'} 

    # for GLM data restriction
    units_df = bwm_units(one)
    valid_pairs = set(zip(units_df['pid'], units_df['cluster_id']))
    
    # get regions that have after pooling at least 20 neurons
    rr = units_df['Beryl'].value_counts() >= 20
    valid_regs = rr[rr == True].index.tolist()
    
    
    for v in fs:
    
        # filter GLM results bu units_df 
        data = pd.read_pickle(Path(enc_pth, fs[v]))['mean_fit_results']
        
        data = data.loc[[index for index in data.index if
                         (index[1], index[2]) in valid_pairs]]

        d[v] = data.groupby(
            "region").agg({"wheel":"mean"})
        d[v].reset_index(inplace=True)
        d[v].columns = ['region', 'glm_effect']
        
    D['glm'] = d           

    '''
    decoding
    '''
   
    d = {}    
    fs = {'speed': 'wheel-speed_stage3',
          'velocity': 'wheel-velocity_stage3'}
          
    for vari in fs:
        d[vari] = pd.read_parquet(Path(dec_pth,
                    f'{fs[vari]}.pqt'))[[
                    'region','valuesminusnull_median',
                    'sig_combined_corrected']].rename(columns = {
                    'valuesminusnull_median': 'decoding_effect',
                    'sig_combined_corrected': 'decoding_significant'})
                
        d[vari].dropna(axis=0,how='any',subset=['decoding_effect'],
                       inplace=True)    
    
    D['dec'] = d


    
    # merge frames across analyses    
    for vari in fs:
        df_ = reduce(lambda left,right: 
                     pd.merge(left,right,on='region', how='inner'), 
                     [D[ana][vari] for ana in D])
                     
        df_.replace(to_replace=[True, False], 
                      value=[1, 0], inplace=True)
                      
        # ## Apply logarithm to GLM results             
        df_.glm_effect = np.log10(
                df_.glm_effect.clip(lower=1e-5))
               
        
        # Reorder columns to match ordering in Figure       
        df_ = df_[['region',
                   'glm_effect',
                   'decoding_effect',
                   'decoding_significant']]
                                           
        df_ = df_[df_['region'].isin(valid_regs)]
                                
        df_.to_pickle(meta_pth / f"{vari}.pkl")

        
    print('pooled and saved results at')
    print(meta_pth)      
    

# -------------------------------------------------------------------------------------------------
# Data loading utils
# -------------------------------------------------------------------------------------------------
def load_data_for_variable(variable):

    data = pd.read_pickle(meta_pth.joinpath(f"{variable}.pkl"))

    return data

def get_data_for_analysis(analysis, data):

    assert analysis in ANALYSIS_TYPES.keys(), (f'{analysis} not recognised, '
                                               f'please choose from {ANALYSIS_TYPES.keys()}')

    analysis_sig = '_'.join([analysis.split('_')[0], 'significant'])
    latency = True if 'latency' in analysis else False

    if 'glm' not in analysis:
        acronyms = data[data[f'{analysis_sig}'] == True]['region'].values
        scores = data[data[f'{analysis_sig}'] == True][f'{analysis}'].values

        if latency:
            mask = data[np.bitwise_or(data[f'{analysis_sig}'] == False,
                                      np.isnan(data[f'{analysis}']))]['region'].values
        else:
            # Remove regions from mask with nan amps (not analyzed)
            mask = data[np.bitwise_and(data[f'{analysis_sig}'] == False,
                                       ~np.isnan(data[f'{analysis}']))]['region'].values

    else:
        acronyms = data['region'].values
        scores = data[f'{analysis}'].values
        mask = []

    vmax = np.max(scores)
    vmin = np.min(scores)

    return scores, acronyms, mask, vmin, vmax, latency


# -------------------------------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------------------------------


def plot_swansons_for_variable(variable, axs=None, annotate_list=None, adjust=False, save=True, save_path=None):

    """
    For a single variable, plot 5 results on swansons,
    4 effects for the 4 analyses and latencies for trajectory
    (Previously called plot_swansons)
    :param variable:
    :param fig:
    :param axs:
    :return:
    """

    data = load_data_for_variable(variable)
    analysis = ['decoding_effect', 'mannwhitney_effect', 'euclidean_effect',
                'euclidean_latency', 'glm_effect']

    cmap = get_cmap_for_variable(variable)

    if axs is None:
        fig, axs = plt.subplots(1, len(analysis), figsize=(180 * MM_TO_INCH, 80 * MM_TO_INCH),
                              gridspec_kw={'wspace': 0.5})
        tight = True
    else:
        assert len(axs) == len(analysis)
        fig = axs[0].get_figure()
        tight = False

    for col, ana in enumerate(analysis):

        ax = axs[col]

        scores, acronyms, mask, vmin, vmax, is_lat = get_data_for_analysis(ana, data)

        if 'glm' in ana and variable == 'stim':
            vmax = np.percentile(scores, 95)
            vmin = np.percentile(scores, 5)

        if adjust:
            annotate_list = list(ADJUST_SWANSONS[variable][ana].keys())

        annotate_kwargs = {
            'annotate': True,
            'annotate_n': 5,
            'annotate_order': 'bottom' if is_lat else 'top',
            'annotate_list': annotate_list, # If annotate list is given this overides the arguments above
        }

        ana_label = ANALYSIS_TYPES[ana]['label']
        ana_type = ANALYSIS_TYPES[ana]['analysis']
        ana_info = ANALYSIS_TYPES[ana]['info']

        ax, cax = plot_vertical_swanson(acronyms, scores, mask, ax=ax, cmap=cmap.reversed() if is_lat else cmap,
                                        vmin=vmin, vmax=vmax, fontsize=f_size_xs, cbar_shrink=0.4,
                                        annotate_kwargs=annotate_kwargs, cbar=True, cbar_label=ana_label,
                                        legend=True if col == 0 else False, mask_label='Not significant')

        ax.text(-0.25, 0.5, ana_type, fontsize=f_size_l, ha='center',va = 'center',
                rotation='vertical', transform=ax.transAxes)
        ax.text(-0.1, .5, ana_info, fontsize=f_size_s, ha='center',va = 'center',
                rotation='vertical', transform=ax.transAxes)
        ax.text(0.9, 0.95, f' {len(scores)}/{len(scores) + len(mask)}',
                fontsize=f_size_s, ha='center', transform=ax.transAxes)

        ax.set_xlim(-100, 3250)
        ax.invert_xaxis()

        if '\n' in ana_label:
            ana_label = ' '.join(ana_label.split(' \n '))
        if ana in ['decoding_effect', 'glm_effect']:
            cax.set_label(ana_label, fontsize=f_size_s, labelpad=0)
        else:
             cax.set_label(ana_label, fontsize=f_size_s, labelpad=1)
        # Print regions with largest (smallest) amplitude (latency) scores
        # print(f'{ana}:')
        # print(acronyms[np.argsort(scores)][:7] if is_lat else acronyms[np.argsort(scores)][-7:])

        if adjust:
            adjust_vertical_swansons(ADJUST_SWANSONS[variable][ana], ax)

    if tight:
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=1)

    if save:
        save_path = save_path or imgs_variable_path(variable)
        fig.savefig(save_path.joinpath(f'{variable}_swansons.svg'))
        fig.savefig(save_path.joinpath(f'{variable}_swansons.pdf'))


def plot_slices_for_variable(variable, axs=None, save=True):

    """
    For a single variable, plot effects for the 4 analyses and
    latencies of population trajectories onto brain slices
    (Previously called plot_slices)
    :param variable:
    :return:
    """

    data = load_data_for_variable(variable)
    analysis = ['decoding_effect', 'mannwhitney_effect', 'euclidean_effect',
                'euclidean_latency', 'glm_effect']

    cmap = get_cmap_for_variable(variable)

    nrows = 4  # Number of rows for plots

    if axs is None:
        fig, axs = plt.subplots(nrows, len(analysis), figsize=(180 * MM_TO_INCH, 130 * MM_TO_INCH),
                                gridspec_kw={'height_ratios': [1] * (nrows -1) + [1.2], 'hspace': 0})
        tight = True
    else:
        fig = axs[0][0].get_figure()
        tight = False

    for col, ana in enumerate(analysis):

        scores, acronyms, mask, vmin, vmax, is_lat = get_data_for_analysis(ana, data)

        row = 0  #row index
      
        for st in ['sagittal', 'top']:
            if st == 'sagittal':
                coords = [-1800, -800, -200]  # coordinate in [um]
            else:
                coords = [-1800]  # ignored for top view

        
            for coord in coords:

                ana_label = ANALYSIS_TYPES[ana]['label']
                ana_type = ANALYSIS_TYPES[ana]['analysis']
                ana_info = ANALYSIS_TYPES[ana]['info']

                ax = axs[row][col]

                plot_scalar_on_slice(acronyms, scores, coord=coord, 
                                     slice=st, mapping='Beryl', 
                                     hemisphere='left', background='boundary', 
                                     cmap=cmap.reversed() if is_lat else cmap,
                                     brain_atlas=ba, ax=ax,
                                     empty_color='white', mask=mask, mask_color='silver',
                                     vector=True)
                ax.axis("off")
                ax.set_aspect('equal')
                if row == 0:
                    ax.text(0.5, 1.3, ana_type, fontsize=f_size_l, ha='center', transform=ax.transAxes)
                    ax.text(0.5, 1.1, ana_info, fontsize=f_size_s, ha='center', transform=ax.transAxes)

                if row == nrows - 1:
                    cbar_kwargs = {'shrink': 0.6, 'aspect': 12, 'pad': 0.1}
                    add_cbar(cmap, vmin, vmax, ax, ana_label, cbar_kwargs=cbar_kwargs)

                row += 1

    if tight:
        fig.subplots_adjust(top=0.96, bottom=0.08, left=0.02, right=0.98)

    if save:
        fig.savefig(Path(imgs_pth, 'si', f'n6_supp_figure_{variverb[variable]}_raw.eps'), dpi=150)
        fig.savefig(Path(imgs_pth, 'si', f'n6_supp_figure_{variverb[variable]}_raw.pdf'))
        fig.savefig(Path(imgs_pth, 'si', f'n6_supp_figure_{variverb[variable]}_raw.svg'), dpi=200)
          

def plot_swansons_across_analysis(axs=None, save=True):
    """
    SI figure swansons for all three main variables and analyses
    scores are normalized across variables per analysis
    (Previously called plot_all_swansons)
    :param axs:
    :param save:
    :return:
    """

    analysis = ['decoding_effect', 'mannwhitney_effect', 'euclidean_effect', 'glm_effect']
    cmap = 'viridis_r'

    if axs is None:
        fig, axs = plt.subplots(len(VARIABLES), len(analysis),
                                figsize=([180 * MM_TO_INCH, 200 * MM_TO_INCH]),
                                gridspec_kw={'hspace': 0, 'wspace': 0})
        tight = True
    else:
        fig = axs.ravel()[0].get_figure()
        tight = False

    for col, ana in enumerate(analysis):
        ana_data = {'vmin': [], 'vmax': []}

        # For each analysis load in the data for each variable
        for variable in VARIABLES:
            var_data = dict()
            data = load_data_for_variable(variable)
            var_data['scores'], var_data['acronyms'], var_data['mask'], *_ = (
                get_data_for_analysis(ana, data))
            ana_data[ana] = var_data
            ana_data['vmin'].append(np.min(var_data['scores']))
            ana_data['vmax'].append(np.max(var_data['scores']))


        # Find the vmin and vmax across all variables
        vmin = np.min(ana_data['vmin'])
        vmax = np.max(ana_data['vmax'])

        for row, variable in enumerate(VARIABLES):

            ax = axs[row][col]
            data = load_data_for_variable(variable)
            scores, acronyms, mask, *_ = get_data_for_analysis(ana, data)
            ana_label = ANALYSIS_TYPES[ana]['label']
            ana_type = ANALYSIS_TYPES[ana]['analysis']
            ana_info = ANALYSIS_TYPES[ana]['info']

            legend = True if row == len(VARIABLES) - 1 and col == 0 else False

            _, cax = plot_vertical_swanson(
                acronyms, scores, mask=mask, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar=True,
                cbar_label=ana_label, cbar_shrink=0.5, legend=legend , mask_label='Not significant')

            if row != len(VARIABLES) - 1:
                cax.ax.set_visible(False)

            ax.text(1.25, 0.9, f'{len(scores)}/{len(scores) + len(mask)}',
                    fontsize=f_size_s, ha='right', transform=ax.transAxes)

            if col == 0:
                ax.text(-0.15, 0.5, variverb[variable], fontsize=f_size_l, rotation='vertical',
                        va='center', ha='right', transform=ax.transAxes)

            if row == 0:
                ax.text(0.5, 1.1, ana_type,fontsize=f_size_l, ha='center', transform=ax.transAxes)
                ax.text(0.5, 1.05, ana_info, fontsize=f_size_s, ha='center', transform=ax.transAxes)

            if legend:
                leg = ax.legend_
                leg.set_bbox_to_anchor((0.7, 0.14))

    if tight:
        fig.subplots_adjust(top=0.95, bottom=0.025, left=0, right=1)

    if save:
        fig.savefig(Path(imgs_pth, 'si', 'n6_ed_fig2_all_variables.pdf'))
        fig.savefig(Path(imgs_pth, 'si', 'n6_ed_fig2_all_variables.eps'), dpi=150)


def plot_glm_swansons(axs=None, annotate_list=None, save=True):

    for i, ax in enumerate(axs):
        if i < 3:
            variable = 'stim'
            label = '$\\Delta R^2$ right stimulus onset'
            cticks = [10, 80]
            cticklabels = [0, 0.005]
        else:
            variable = 'choice'
            label = '$\\Delta R^2$ right movement onset'
            cticks = [20, 90]
            cticklabels = [0, 0.002]

        variable = 'stim' if i < 3 else 'choice'

        cmap = get_cmap_for_variable(variable)

        data = load_data_for_variable(variable)
        scores, acronyms, mask, vmin, vmax, _ = get_data_for_analysis('glm_effect', data)

        _, cax = plot_vertical_swanson(
            acronyms, scores, mask=mask, ax=ax, cmap=cmap, vmin=0, vmax=100, cbar=True,
            cbar_label=label, cbar_shrink=1, legend=False, mask_label='Not significant')

        if i not in [1, 4]:
            cax.ax.set_visible(False)
        else:
            cax.set_ticks(cticks)
            cax.set_ticklabels(cticklabels)



def plot_swansons_for_wheel(axs=None, annotate_list=None, adjust=False, save=True):

    """
    For decoding and encoding, plot Swansons for speed and velocity
    Previously called plot_wheel_swansons
    :param axs:
    :param save:
    :return:
    """

    analysis = ['decoding_effect', 'glm_effect']
    variables = ['speed', 'velocity']
     
    cmap = get_cmap_for_variable('speed')

    if axs is None:
        fig, axs = plt.subplots(1, len(analysis) * len(variables),
                              figsize=(180 * MM_TO_INCH, 80 * MM_TO_INCH),
                              gridspec_kw={'wspace': 0.5})
        tight = True
    else:
        fig = axs[0].get_figure()
        tight = False

    col = 0
    for variable in variables:
        data = load_data_for_variable(variable)
        for ana in analysis:

            ax = axs[col]
            scores, acronyms, mask, vmin, vmax, is_lat = get_data_for_analysis(ana, data)

            if adjust:
                annotate_list = list(ADJUST_SWANSONS[variable][ana].keys())

            annotate_kwargs = {
                'annotate': True,
                'annotate_n': 5,
                'annotate_order': 'bottom' if is_lat else 'top',
                'annotate_list': annotate_list,  # If annotate list is given this overides the arguments above
            }

            ana_label = ANALYSIS_TYPES[ana]['label']
            ana_type = ANALYSIS_TYPES[ana]['analysis']
            ana_info = ANALYSIS_TYPES[ana]['info']

            ax, cax = plot_vertical_swanson(acronyms, scores, mask, ax=ax, cmap=cmap.reversed() if is_lat else cmap,
                                  vmin=vmin, vmax=vmax, fontsize=f_size_xs, cbar_shrink=0.5,
                                  annotate_kwargs=annotate_kwargs, cbar=True, cbar_label=ana_label,
                                  legend=True if col == 0 else False, mask_label='Not significant')

            ax.text(-0.25, 0.5, ana_type,fontsize=f_size, ha='center',va = 'center', rotation='vertical',
                    transform=ax.transAxes)
            ax.text(-0.1, .5, ana_info, fontsize=f_size, ha='center', va='center', rotation='vertical',
                    transform=ax.transAxes)

            ax.text(0.85, 0.95, f' {len(scores)}/{len(scores) + len(mask)}', fontsize=f_size_s,
                    ha='center', transform=ax.transAxes)

            ax.set_xlim(-100, 3250)
            ax.invert_xaxis()

            if '\n' in ana_label:
                ana_label = ' '.join(ana_label.split(' \n '))
            if ana in ['decoding_effect', 'glm_effect']:
                cax.set_label(ana_label, fontsize=f_size_s, labelpad=0)
            else:
                cax.set_label(ana_label, fontsize=f_size_s, labelpad=1)

            if col == 0:
                leg = ax.legend_
                leg.set_bbox_to_anchor((0.75, 0.11))

            col += 1

            if adjust:
                adjust_vertical_swansons(ADJUST_SWANSONS[variable][ana], ax)

    if tight:
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    if save:
        fig.savefig(Path(imgs_pth, 'wheel_swansons.svg'))


def plot_table(variable, axs=None, save=True, export=False, save_path=None):
    """
     Plot comparison table
    :param variable:
    :return:
    """

    data = load_data_for_variable(variable)
    cmap = get_cmap_for_variable(variable)

    # Normalize values in each amplitude column to interval [0,1]
    # assuming columns
    if variable in ['speed', 'velocity']:
        analysis = ['decoding', 'glm']
        si, se = 1, 3
        column_labels = {
            'region': 'region',
            'glm_effect': 'Encoding',
            'decoding_effect': 'Decoding'
        }
    else:
        analysis = ['decoding', 'mannwhitney', 'glm', 'euclidean']
        si, se = 2, 6
        column_labels = {
            'region': 'region',
            'glm_effect': 'Encoding',
            'euclidean_effect': 'Trajectory',
            'mannwhitney_effect': 'Single-cell',
            'decoding_effect': 'Decoding'
        }

    ana_effects = ['_'.join([ana, 'effect']) for ana in analysis]

    assert (set(list(data.iloc[:, si:se].keys())) == set(ana_effects))

    data.iloc[:, si:se] = ((data.iloc[:, si:se] - data.iloc[:, si:se].min()) /
                           (data.iloc[:, si:se].max() - data.iloc[:, si:se].min()) + 1e-4)

    # order rows (regions) canonically, omitting those without data
    # p = (Path(iblatlas.__file__).parent / 'beryl.npy')
    # TODO fix me
    p = Path('/Users/admin/int-brain-lab/iblatlas/iblatlas/beryl.npy')
    acronyms = list(data['region'])
    acronyms_beryl = br.id2acronym(np.load(p), mapping='Beryl')
    acronyms_order = []
    for acr in acronyms_beryl:
        if acr in acronyms:
            acronyms_order.append(acr)

    data['region'] = pd.Categorical(data['region'], categories=acronyms_order, ordered=True)
    data = data.sort_values('region')
    data= data.reset_index(drop=True)

    for ana in analysis:
        if ana == 'glm':
            continue
        data[f'{ana}_effect'] = data[f'{ana}_effect'
                               ] * data[f'{ana}_significant']

    data = data[['region'] + ana_effects]

    data = data.rename(columns=column_labels)
    effs = list(column_labels.values())[1:]
    columns_order = ['region'] + list(effs[::-1])
    data = data[columns_order]


    midpoint = len(data) // 2
    df1 = data.iloc[:midpoint].reset_index(drop=True)
    df2 = data.iloc[midpoint:].reset_index(drop=True)

    if export:
        export_to_table(cmap, effs, [df1, df2], variable)

    else:
        df1 = df1.set_index('region')
        df2 = df2.set_index('region')

        if axs is None:
            fig, axs = plt.subplots(2, 1, figsize=(200 * MM_TO_INCH, 60 * MM_TO_INCH),
                                    gridspec_kw={'hspace': 0.5})
            tight = True
        else:
            fig = axs[0].get_figure()
            tight = False

        cmap.set_bad('silver')

        mask1 = np.where(df1 == 0, True, False)
        sns.heatmap(df1.T, mask=mask1.T, linewidths=0.7, linecolor='w', cmap=cmap, cbar=False, ax=axs[0],
                    xticklabels=df1.index.values, yticklabels=df1.columns.values)

        mask2 = np.where(df2 == 0, True, False)
        sns.heatmap(df2.T, mask=mask2.T, linewidths=0.7, linecolor='w', cmap=cmap, cbar=False, ax=axs[1],
                    xticklabels=df2.index.values, yticklabels=df2.columns.values)

        for ax in axs:
            ax.tick_params(left=False, bottom=False)
            ax.tick_params(axis='x', pad=-2)
            ax.set_xlabel('')
            for i, tick_label in enumerate(ax.get_xticklabels()):
                col = REGION_COLS[tick_label.get_text()]
                tick_label.set_color(col)
                tick_label.set_fontsize(f_size_xs)

        if tight:
            fig.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.9)

        if save:
            save_path = save_path or imgs_variable_path(variable)
            fig.savefig(save_path.joinpath(f'{variable}_table.pdf'))
            fig.savefig(save_path.joinpath(f'{variable}_table.pdf'))


def export_to_table(cmap, analysis, dfs, variable):

    cmap = cmap
    analysis = analysis

    def region_formatting(x):
        """
        Formatting for acronym strings
        """
        color = mpl.colors.to_hex(REGION_COLS[x])
        
        return 'background-color: w; color: ' + color


    def effect_formatting(x):
        """
        Formatting for effect columns
        """

        if x==0:  # not significant (values were set to zero)
            color = 'silver'
        elif pd.isna(x):  # not analysed
            color = 'w'    
        else:
            rgb = cmap(x)
            color =  ('#' + rgb_to_hex((int(255*rgb[0]), int(255*rgb[1]), int(255*rgb[2]))))
                         
        return 'background-color: ' + color

    # Format table  
    def make_pretty(styler):
        
        styler.applymap(effect_formatting, subset=analysis)
        styler.applymap(region_formatting, subset=['region'])
        styler.set_properties(subset=analysis, **{'width': '8px'})
        styler.set_properties(subset=analysis, **{'font-size': '0pt'})
        styler.set_properties(subset=['region'], **{'width': 'max-content'})
        styler.set_properties(subset=['region'],**{'font-size': '5pt'})
        styler.hide(axis="index")

        #styler.hide(axis="columns")  # Hide column headers
       
        styler.set_table_styles([         
            {"selector": "tr", "props": "line-height: 5px"}, #was 9
            {"selector": "td, th", 
                "props": "line-height: inherit; padding: 0 "},               
            {"selector": "tbody td", 
                "props": [("border", "1px solid white")]},
            {'selector': 'th.col_heading', 
                        'props': [('writing-mode', 'vertical-rl'),
                              ('text-align', 'center'),
                              ('font-size', '3pt'),
                              ('padding', '1px 0'),
                              ('white-space', 'nowrap')]}])
     
        return styler

    for i, df in enumerate(dfs):

        table = df.style.pipe(make_pretty)
        save_path = Path(imgs_pth, variable)
        save_path.mkdir(parents=True, exist_ok=True)
        dfi.export(table, Path(save_path, f'{variable}_table_{i}.png'), max_rows=-1, dpi = 200)



def scatter_for_analysis_pair(variable, analysis_pair, sig_only=False, ax=None, save=False):
    """
    Scatter plot: comparison of two analysis amplitudes for a given variable
    Previously called scatter_for_analysis_effects

    Parameters:
        variable (str): One of 'stim', 'choice', 'fback'
        analysis_pair (tuple): Pair of analyses, e.g., 
        ('glm_effect', 'decoding_effect')
        meta_pth (Path): Path to the directory containing data files
    
    """

    # Load the data for the specified variable
    data = load_data_for_variable(variable)
    data['glm_significant'] = 1

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
        tight = True
        ss = 20
    else:
        fig = ax.get_figure()
        tight = False
        ss = 5

    if sig_only:
        # Fetch data for both analyses, sig regions only
        data = data.query(f'{analysis_pair[0].split("_")[0]}_significant == 1 &'
                             f'{analysis_pair[1].split("_")[0]}_significant == 1')

    scores1 = data[analysis_pair[0]].values
    scores2 = data[analysis_pair[1]].values
    acronyms = data['region'].values
    cols = np.array([REGION_COLS[acr] for acr in acronyms])
    
    # Remove NaN values
    valid_indices = ~np.isnan(scores1) & ~np.isnan(scores2)
    scores1 = scores1[valid_indices]
    scores2 = scores2[valid_indices]
    acronyms = acronyms[valid_indices]
    cols = cols[valid_indices]
    
    # Scatter plot
    ax.scatter(scores1, scores2, color=cols, s=ss)
    # Set labels
    ax.set_xlabel(ANALYSIS_TYPES[analysis_pair[0]]['analysis'], fontsize=f_size)
    ax.set_ylabel(ANALYSIS_TYPES[analysis_pair[1]]['analysis'], fontsize=f_size)
    
    if tight:
        # Annotating each point with the region code
        for i, reg in enumerate(acronyms):
            ax.annotate(' ' + reg, (scores1[i], scores2[i]), fontsize=f_size_s, color=cols[i])
        # Set tight layout
        fig.tight_layout()

   # Calculate and display correlation coefficients
   # cors, ps = spearmanr(scores1, scores2)
   # corp, pp = pearsonr(scores1, scores2)

    if save:
        fig.savefig(Path(imgs_pth, f'{variable}_{analysis_pair[0]}_{analysis_pair[1]}.png'))


def plot_scatter_analysis_pairs(sig_only=True, axs=None, save=True):

    """
    For the three main variables and 4 analyses,
    plot grid of scatters of amplitudes
    Previously called scatter_analysis_effects_grid
    :param sig_only:
    :param axs:
    :param save:
    :return:
    """

    analysis = ['euclidean_effect', 'glm_effect', 'mannwhitney_effect', 'decoding_effect']
    analysis_combinations = list(itertools.combinations(analysis, 2))

    if axs is None:
        fig, axs = plt.subplots(nrows =len(analysis_combinations),ncols=len(VARIABLES),
                                figsize=(180 * MM_TO_INCH, 200 * MM_TO_INCH))
        tight = True
    else:
        fig = axs.ravel()[0].get_figure()
        tight = False

    for col, variable in enumerate(VARIABLES):
        for ic, combination in enumerate(analysis_combinations):
            row = np.mod(ic, len(analysis_combinations))
            ax = axs[row, col]
            scatter_for_analysis_pair(variable, combination, sig_only=sig_only, ax=ax)

            if row == 0:
                ax.set_title(f'{variverb[variable]}', fontsize=f_size_l, va='bottom')
            else:
                ax.set_title(None)

            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    if tight:
        fig.tight_layout()

    if save:
        fig.savefig(Path(imgs_pth, 'si', f'n6_ed_fig3_analyses_amp_pairs_grid.eps'), dpi=150)
        fig.savefig(Path(imgs_pth, 'si', f'n6_ed_fig3_analyses_amp_pairs_grid.pdf'))
        

def plot_bar_neuron_count(table_only=False, ssvers='_rerun'):

    '''
    bar plot for neuron count per region 
    second bar for recording count per region
    
    for BWM intro figure;
    
    Adding additional info in the table, including effect sizes
    
    ssvers: spike sorting version in ['_rerun', '']
    '''

    file_ = download_aggregate_tables(one)
    df = pd.read_parquet(file_)
    dfa, palette = get_allen_info()
    df['Beryl'] = br.id2acronym(df['atlas_id'], mapping='Beryl')
    df['Cosmos'] = br.id2acronym(df['atlas_id'], mapping='Cosmos')    
    cosregs = dict(list(Counter(zip(df['Beryl'],df['Cosmos']))))
    
    # number of clusters per region, c0
    c0 = dict(Counter(df['Beryl']))
    del c0['root']
    del c0['void']   
    
    # good neurons per region
    c_good = dict(Counter(df['Beryl'][df['label'] == 1]))
    
    for reg in c0:
        if reg not in c_good:
            c_good[reg] = 0
    
    # number of recordings per region
    ps = list(set(['_'.join(x) for x in zip(df['Beryl'],df['pid'])]))    
    n_ins = Counter([x.split('_')[0] for x in ps])

    # order by beryl
    regs0 = list(c0.keys()) 
    p = (Path(iblatlas.__file__).parent / 'beryl.npy')
    regs = br.id2acronym(np.load(p), mapping='Beryl')
    regs1 = []
    for reg in regs:
        if reg in regs0:
            regs1.append(reg)

    nclus_nins = [[c0[reg], n_ins[reg],c_good[reg]] for reg in regs1]

 
    cols = [palette[reg]  
            if cosregs[reg] not in ['CBX', 'CBN']
            else mpl.colors.to_rgba('#757a3d') for reg in regs1]

    # reverse numbers
    cols.reverse()
    nclus_nins.reverse()
    regs1.reverse()

    nclus_nins = np.array(nclus_nins)
    print('raw bwm dataset')
    print(f'{len(regs1)} regions')
    print(len(df['Beryl']), 'neurons')
    print(len(df['Beryl'][df['label'] == 1]),' good neurons')
    #print(len(np.unique(df['pid'])), 'recordings'

    if table_only:
        vs0 = ['stim', 'choice', 'fback']
        ans0 = ['dec', 'man', 'euc', 'glm', ]
        effects = [('_').join([x,y]) for x in vs0 for y in ans0]
        
        vs1 = ['speed', 'velocity']
        ans1 = ['dec', 'glm']
        effects += [('_').join([x,y]) for x in vs1 for y in ans1]        

        vs = vs0 + vs1
        ans = ans0 + ans1
        
        res = {}
        
        for v in vs:
            res[v] = pd.read_pickle(meta_pth / f"{v}.pkl")
   
        res2 = {}
        for reg in regs1:
            res3 = {}
            for v in vs:
                for a in ans:
                    if (v in vs1) and (a not in ans1):
                        continue 
                
                    # Find the effect key and its corresponding significance key
                    effect_key = next((x for x in res[v].keys() if (a in x and 'effect' in x)), None)
                    sig_key = next((x for x in res[v].keys() if (a in x and 'significant' in x)), None)

                    # Default values if keys are missing
                    effect_score = None
                    sig_score = None

                    if effect_key and reg in res[v]['region'].values:
                        effect_score = np.round(res[v][res[v]['region'] == reg][effect_key].item(), 2)

                    if sig_key and reg in res[v]['region'].values:
                        sig_score = res[v][res[v]['region'] == reg][sig_key].item()

                    # Store effect score
                    res3[('_').join([v, a])] = effect_score

                    # Store significance score if available
                    if sig_key:
                        res3[('_').join([v, a, 'sig'])] = sig_score
                            
                    res2[reg] = res3   
    
                    
        # get all regions in the canonical set
        units_df = bwm_units(one)
        gregs = Counter(units_df['Beryl'])
        
        print('canonical bwm dataset')
        print(f"{len(np.unique(units_df['Beryl']))} regions")
        print(len(units_df['Beryl']), 'good neurons')
    
        sig_effects = [e + '_sig' for e in effects if any(e + '_sig' in res2[reg] for reg in regs1)]

        # Full column list
        columns = (['Beryl', 'Beryl', 'Cosmos', 'Cosmos', 
                '# recordings', '# neurons', 
                '# good neurons', 'canonical'] + effects + sig_effects)

        r = []                       
        for k in range(len(regs1)):
            cano = True if regs1[k] in gregs else False

            # Extract only existing values in res2[regs1[k]]
            row_values = [regs1[k], get_name(regs1[k]),
                        cosregs[regs1[k]],
                        get_name(cosregs[regs1[k]]), nclus_nins[k,1],
                        nclus_nins[k,0], nclus_nins[k,2], cano] 

            # Ensure row length matches column length
            for col in effects + sig_effects:
                row_values.append(res2[regs1[k]].get(col, None))  

            r.append(row_values)            

        df  = pd.DataFrame(data=r, columns=columns)
        df.dropna(axis=1, how='all', inplace=True)
        df  = df.reindex(index=df.index[::-1])
        df.reset_index(drop=True, inplace=True)
        df.to_csv(meta_pth / 'region_info.csv', index=False)
                 
        print('saving table only')
        return 


    # plotting; figure options
    
    ncols = 3 
    fig, ax = plt.subplots(ncols=ncols, figsize=(ncols*4/3,ncols* 7/3))
    
    fs = 5
    barwidth = 0.6
    
    cols_ = cols
    nclus_nins_ = nclus_nins
    regs1_ = regs1
    L = len(regs1)
     
    for k, k0 in zip(range(3), reversed(range(3))):

        cols = cols_[k0 * L//ncols : (k0+1) * L//ncols]
        nclus_nins = nclus_nins_[k0 * L//ncols : (k0+1) * L//ncols]
        regs1 = regs1_[k0 * L//ncols : (k0+1) * L//ncols]


        Bars = ax[k].barh(range(len(regs1)), nclus_nins[:,0],
                   fill=False, edgecolor = cols, height=barwidth)
        ax[k].barh(range(len(regs1)), nclus_nins[:,2],color=cols, 
                   height=barwidth)
        ax[k].set_xscale("log")
        ax[k].tick_params(axis='y', pad=19,left = False) 
        ax[k].set_yticks(range(len(regs1)))
        ax[k].set_yticklabels([reg for reg in regs1],
                              fontsize=fs, ha = 'left')
                              
        # indicate 10% with black line                         
        y_start = np.array([plt.getp(item, 'y') for item in Bars])
        y_end   = y_start+[plt.getp(item, 'height') for item in Bars]

        ax[k].vlines(0.1 * nclus_nins[:,0], y_start, y_end, 
                  color='k', linewidth =1)                     
                              
                              
        for ytick, color in zip(ax[k].get_yticklabels(), cols):
            ytick.set_color(color)     
            
       
        # #recs
        ax2 = ax[k].secondary_yaxis("right")
        ax2.tick_params(right = False)
        ax2.set_yticks(range(len(regs1)))
        ax2.set_yticklabels(nclus_nins[:,1],fontsize=fs)
        for ytick, color in zip(ax2.get_yticklabels(), cols):
            ytick.set_color(color)
        ax2.spines['right'].set_visible(False)    

        ax[k].spines['bottom'].set_visible(False)
        ax[k].spines['right'].set_visible(False)
        ax[k].spines['left'].set_visible(False)
        ax[k].xaxis.set_ticks_position('top')
        ax[k].xaxis.set_label_position('top')
        ax[k].set_xlabel('Neurons')
        ax[k].xaxis.set_minor_locator(MultipleLocator(450))
        plt.setp(ax[k].get_xminorticklabels(), visible=False)
        #ax[k].xaxis.set_major_locator(plt.MaxNLocator(10))

        
    fig.tight_layout()



# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# SINGLE CELL ANALYSIS
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# Loading utils
# -------------------------------------------------------------------------------------------------
MOVE_VARIABLES = ['overall', 'nose', 'pupil', 'paw', 'tongue']
move2panels = dict(zip(MOVE_VARIABLES, ['b', 'c', 'c', 'c', 'c']))
TASK_VARIABLES = ['correct_vs_baseline', 'incorrect_vs_baseline', 'task']


def load_single_cell_data_for_variable(variable):

    if variable in MOVE_VARIABLES:
        file = 'Updated_Single_cell_analysis_July_10_2024 - Movement(DLC) variables.csv'
        strmatch = f'[panel {move2panels[variable]}] ({variable})'
    else:
        file = 'Updated_Single_cell_analysis_July_10_2024 - Sheet1.csv'
        strmatch = f'[{variable}]'

    data = pd.read_csv(Path(sc_pth, file))
    data = (data[['Acronym',f'{strmatch} fraction of significance', f'{strmatch} significance']]
            .rename(columns = {'Acronym': 'region',
                               f'{strmatch} fraction of significance': 'mannwhitney_effect',
                               f'{strmatch} significance': 'mannwhitney_significant'}))

    return data

# -------------------------------------------------------------------------------------------------
# Plotting utils
# -------------------------------------------------------------------------------------------------

def plot_swanson_for_single_cell_variable(variable, ax=None, save=True, cbar=True, legend=True):
    """
    SI figure swanson for single-cell results
    vari in ['correct_vs_baseline', 'incorrect_vs_baseline', 'task']
    Previously called swansons_SI
    :param variable:
    :param ax:
    :return:
    """

    cmap = get_cmap_for_variable('fback') if variable in TASK_VARIABLES else get_cmap_for_variable('move')

    if ax is None:
        fig, ax = plt.subplots(figsize=(120 * MM_TO_INCH, 60 * MM_TO_INCH))
    else:
        fig = ax.get_figure()

    data = load_single_cell_data_for_variable(variable)
    ana = 'mannwhitney'

    acronyms = data[data[f'{ana}_significant'] == True]['region'].values
    scores = data[data[f'{ana}_significant'] == True][f'{ana}_effect'].values
    # Turn fraction into percentage
    scores = scores * 100

    # Mask regions are those that have values (not nan) and are not significant (0)
    mask = data[np.bitwise_and(data[f'{ana}_significant'] == 0, ~np.isnan(data[f'{ana}_effect']))]['region'].values

    vmin, vmax = (np.min(scores), np.max(scores)) if variable == 'task' else (0, 100)

    annotate_kwargs = {'annotate': False}
    _, cax = plot_horizontal_swanson(acronyms, scores, mask=mask, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                                     annotate_kwargs=annotate_kwargs, cbar=cbar, legend=legend,
                                     cbar_label='Fraction of significant cells (%)',
                                     mask_label='Zero significant cells (%)')

    ax.set_title(variable.capitalize(), fontsize=f_size_l)
    ax.title.set_position((0.1, 0.8))

    if save:
        fig.savefig(Path(imgs_pth, 'si', f'mannwhitney_SI_{variable}.svg'))

    return ax, cax


# -------------------------------------------------------------------------------------------------
# BEHAVIOR ANALYSIS
# -------------------------------------------------------------------------------------------------
'''
#####
Behavioral SI figure
#####
'''

def perf_scatter(rerun=False, axs=None, save=True):

    """
    Two scatter plots, a point a mouse from the BWM set,
    left panel (x, y):
    (# bias training sessions, # pre-bias training sessions
    right panel (x, y):
    (# bias training sessions, % trials correct during ephys)
    :param rerun:
    :return:
    """

    # Define path to the parquet file
    data_path = Path(one.cache_dir, 'bwm_res', 'bwm_figs_data', 'training_perf.pqt')

    # Only reprocess data if file does not exist or rerun is set to True
    if not data_path.exists() or rerun:

        # Retrieve unique subject IDs
        eids = bwm_units(one)['eid'].unique()
        pths = one.eid2path(eids)
        subs = np.unique([str(x).split('/')[-3] for x in pths])

        # Get lab metadata
        rr = labs()
        sub_labs = {str(x).split('/')[-3]: rr[1][str(x).split('/')[-5]] for x in pths}
        sub_cols = {sub: rr[-1][lab] for sub, lab in sub_labs.items()}

        # Initialize results list and column names
        r = []
        columns = ['subj', '#sess biasedChoiceWorld', '#sess trainingChoiceWorld', 
                   'perf ephysChoiceWorld', 'lab', 'lab_color']

        # Process each subject's data
        for sub in subs:
            # Load trials and training data
            trials = one.load_aggregate('subjects', sub, '_ibl_subjectTrials.table')
            training = one.load_aggregate('subjects', sub, '_ibl_subjectTraining.table')

            # Join and sort by session start time
            trials = (trials.set_index('session').join(training.set_index('session')).
                      sort_values('session_start_time', kind='stable'))

            # Separate behavior sessions based on task protocol
            session_types = {
                'training': ['_iblrig_tasks_trainingChoiceWorld', 'trainingChoiceWorld'],
                'biased': ['_iblrig_tasks_biasedChoiceWorld', 'biasedChoiceWorld'],
                'ephys': ['_iblrig_tasks_ephys', 'ephysChoiceWorld']
            }

            # Create filtered sessions based on task_protocol
            filtered_sessions = {k: trials[trials['task_protocol'].str.startswith(tuple(v))]
                                 for k, v in session_types.items()}

            # Calculate metrics
            nbiased = filtered_sessions['biased'].index.nunique()
            nunbiased = filtered_sessions['training'].index.nunique()
            perf_ephys = np.nanmean((filtered_sessions['ephys'].feedbackType + 1) / 2)

            # Append results for the current subject
            r.append([sub, nbiased, nunbiased, perf_ephys, sub_labs[sub], sub_cols[sub]])

        # Create DataFrame and save to parquet
        df = pd.DataFrame(r, columns=columns)
        df.to_parquet(data_path)

    # Read the parquet file into DataFrame
    df = pd.read_parquet(data_path)

    # convert to hex for seaborn
    df['lab_color'] = df['lab_color'].apply(lambda x: mcolors.to_hex(x))

    # Create a dictionary to map lab to its color
    lab_palette = dict(zip(df['lab'], df['lab_color']))

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=([7.99, 3.19]))
        tight = True
    else:
        fig = axs[0].get_figure()
        tight = False

    # Left scatter plot: (#sess biasedChoiceWorld, #sess trainingChoiceWorld)
    sns.scatterplot(ax=axs[0], data=df, x='#sess biasedChoiceWorld', y='#sess trainingChoiceWorld', hue='lab',
                    palette=lab_palette, legend=False, s=20)

    # Drop NaNs for Pearson correlation calculation
    valid_left = df[['#sess biasedChoiceWorld', '#sess trainingChoiceWorld']].dropna()
    r_left, p_left = stats.pearsonr(valid_left['#sess biasedChoiceWorld'], valid_left['#sess trainingChoiceWorld'])

    # Annotate plot with r and p-value
    axs[0].text(0.8, 1, f'r = {r_left:.3f}\np = {p_left:.3f}', transform=axs[0].transAxes, verticalalignment='top',
                fontsize=f_size_s)

    axs[0].set_xlabel('Number of biased training sessions', fontsize=f_size)
    axs[0].set_ylabel('Number of non-biased training sessions', fontsize=f_size)

    # Right scatter plot: (#sess biasedChoiceWorld, perf ephysChoiceWorld)
    sns.scatterplot(ax=axs[1], data=df, x='#sess biasedChoiceWorld', y='perf ephysChoiceWorld', hue='lab',
                    palette=lab_palette, legend=True, s=20)

    valid_right = df[['#sess biasedChoiceWorld', 'perf ephysChoiceWorld']].dropna()
    r_right, p_right = stats.pearsonr(valid_right['#sess biasedChoiceWorld'], valid_right['perf ephysChoiceWorld'])

    # Annotate plot with r and p-value
    axs[1].text(0.8, 1, f'r = {r_right:.3f}\np = {p_right:.3f}', transform=axs[1].transAxes, verticalalignment='top',
                fontsize=f_size_s)

    axs[1].set_xlabel('Number of biased training sessions', fontsize=f_size)
    axs[1].set_ylabel('Percentage of trials correct \n (recording sessions only)', fontsize=f_size)

    axs[1].legend(loc='upper left', bbox_to_anchor=(1.05, 0.75), frameon=False, fontsize=f_size_s)

    # Adjust layout and display plot
    if tight:
        fig.tight_layout()

    if save:
        fig.savefig(Path(imgs_pth, 'si', f'n6_supp_figure_learning_stats.pdf'), dpi=150)

    return fig

# -------------------------------------------------------------------------------------------------
# DECODING ANALYSIS
# -------------------------------------------------------------------------------------------------

dec_d = {'stim': 'stimside', 'choice': 'choice',
            'fback': 'feedback'}

ADJUST_DECODING_SWANSONS = {
    'stim': {
        'NB': {'y': 1500, 'x': 100}, 'TRN': {'y': 1800, 'x': 1300}, 'PRNc': {'y': 3300, 'x': 200},
    },
    'fback': {
        'BST': {'y': 3300, 'x': -100}, 'PVH': {'y': 3300, 'x': -100}, 'CS': {'y': 3300, 'x': 100},
        'V': {'y': 3300, 'x': 100}, 'LDT': {'y': 1350, 'x': -300}, 'NI': {'y': 1500, 'x': -500}
    },
    'choice': {
        'VISal': {'y': 200, 'x': -200}, 'SAG': {'y': 1450, 'x': 100}, 'LDT': {'y': 3300, 'x': 300},
        'IF': {'y': 3300, 'x': 100}, 'RL': {'y': 3300, 'x': 400}, 'GR': {'y': 1750, 'x': 200},
        'LIN': {'y': 3300, 'x': 50}, 'ISN': {'y': 3300, 'x': 400}
    },
}


# -------------------------------------------------------------------------------------------------
# Loading utils
# -------------------------------------------------------------------------------------------------

def group_into_regions():

    ''' 
    grouping per-session decoding results into 
    per-region, also doing FDR correction
    '''    

    MIN_UNITS = 5
    MIN_TRIALS = 0
    MIN_SESSIONS_PER_REGION = 2
    ALPHA_LEVEL = 0.05
    Q_LEVEL = 0.01

    def significance_by_region(group):
        result = pd.Series()
        # only get p-values for sessions with min number of trials
        if 'n_trials' in group:
            trials_mask = group['n_trials'] >= MIN_TRIALS
        else:
            trials_mask = np.ones(group.shape[0]).astype('bool')
        pvals = group.loc[trials_mask, 'p-value'].values
        pvals = np.array([p if p > 0 else 1.0 / (N_PSEUDO + 1) for p in pvals])
        # count number of good sessions
        n_sessions = len(pvals)
        result['n_sessions'] = n_sessions
        # only compute combined p-value if there are enough sessions
        if n_sessions < MIN_SESSIONS_PER_REGION:
            result['pval_combined'] = np.nan
            result['n_units_mean'] = np.nan
            result['values_std'] = np.nan
            result['values_median'] = np.nan
            result['null_median_of_medians'] = np.nan
            result['valuesminusnull_median'] = np.nan
            result['frac_sig'] = np.nan
            result['values_median_sig'] = np.nan
            result['sig_combined'] = np.nan
        else:
            scores = group.loc[trials_mask, 'score'].values
            result['pval_combined'] = stats.combine_pvalues(pvals, 
                method='fisher')[1]
            result['n_units_mean'] = group.loc[trials_mask, 'n_units'].mean()
            result['values_std'] = np.std(scores)
            result['values_median'] = np.median(scores)
            result['null_median_of_medians'] = group.loc[trials_mask, 
                'median-null'].median()
            result['valuesminusnull_median'] = result[
                'values_median'] - result['null_median_of_medians']
            result['frac_sig'] = np.mean(pvals < ALPHA_LEVEL)
            result['values_median_sig'] = np.median(scores[pvals < ALPHA_LEVEL])
            result['sig_combined'] = result['pval_combined'] < ALPHA_LEVEL
        return result

    # indicate in file name constraint
    exx = '' if MIN_TRIALS == 0 else ('_' + str(MIN_TRIALS))
    
    for vari in VARIABLES:
        pqt_file = os.path.join(dec_pth,f"{dec_d[vari]}_stage2.pqt")
        df1 = pd.read_parquet(pqt_file)
        # compute combined p-values for each region
        df2 = df1[df1.n_units >= MIN_UNITS].groupby([
            'region']).apply(
            lambda x: significance_by_region(x)).reset_index()
        # run FDR correction on p-values
        mask = ~df2['pval_combined'].isna()
        _, pvals_combined_corrected, _, _ = multipletests(
            pvals=df2.loc[mask, 'pval_combined'],
            alpha=Q_LEVEL,
            method='fdr_bh',
        )
        df2.loc[mask, 'pval_combined_corrected'] = pvals_combined_corrected
        df2.loc[:, 'sig_combined_corrected'] = df2.pval_combined_corrected < Q_LEVEL
        # save out
        
        filename = os.path.join(dec_pth, f"{vari}_stage3{exx}.pqt")
        print(f"saving parquet {vari}")
        df2.to_parquet(filename)
        print("parquet saved")


def acronym2name(acronym):
    return br.name[np.argwhere(br.acronym==acronym)[0]][0]


def get_xy_vals(xy_table, eid, region):
    xy_vals = xy_table.loc[xy_table['eid_region']==f'{eid}_{region}']
    assert xy_vals.shape[0] == 1
    return xy_vals.iloc[0]


def get_res_vals(res_table, eid, region):
    er_vals = res_table[(res_table['eid']==eid) & (res_table['region']==region)]
    assert len(er_vals)==1
    return er_vals.iloc[0]


def load_decoding_for_variable(variable):

    variable2file = {
        'stim': 'stimside_e0928e11-2b86-4387-a203-80c77fab5d52_VISp_merged_probes_pseudo_ids_-1_100.pkl',
        'choice': 'choice_671c7ea7-6726-4fbe-adeb-f89c2c8e489b_GRN_merged_probes_pseudo_ids_-1_100.pkl',
        'fback': 'feedback_e012d3e3-fdbc-4661-9ffa-5fa284e4e706_IRN_merged_probes_pseudo_ids_-1_100.pkl',
    }

    data = pd.read_pickle(open(Path(dec_pth, variable2file[variable]), 'rb'))

    fit = data['fit'][0]

    # Get average predictions from all runs
    run_idxs = np.unique([i for i, fit in enumerate(data['fit']) if fit['pseudo_id'] == -1])
    preds = np.concatenate([np.squeeze(data['fit'][i]['predictions_test'])[:, None] for i in run_idxs], axis=1)
    preds = np.mean(preds, axis=1)

    # Get decoding target (binary)
    targets = np.squeeze(fit['target'])

    # Get mask
    mask = fit['mask'][0]

    # Get signed_contrasts
    contrasts = (np.nan_to_num(fit['trials_df'].contrastLeft) -np.nan_to_num(fit['trials_df'].contrastRight))

    # Get the first region
    reg = data['region'][0]

    return preds, targets, mask, contrasts, reg

# -------------------------------------------------------------------------------------------------
# Plotting utils
# -------------------------------------------------------------------------------------------------

def plot_decoding_line(ax=None, save=True, save_path=None):
    """
    Plot decoding extra panels for bwm main figure
    Previously called stim_dec_line
    :param fig:
    :param ax:
    :return:
    """

    if ax is None:
        tight = True
        fig, ax = plt.subplots(figsize=(60*MM_TO_INCH, 38*MM_TO_INCH))
    else:
        tight = False
        fig = ax.get_figure()

    preds, targets, mask, contrasts, region = load_decoding_for_variable('stim')

    # Compute neurometric curve
    contrasts = contrasts[mask]
    unique_contrasts = np.unique(contrasts)
    neurometric_curve = 1 - np.array([np.mean(preds[contrasts==c]) for c in unique_contrasts])
    neurometric_curve_err = np.array([2 * np.std(preds[contrasts==c]) /
                                      np.sqrt(np.sum(contrasts==c)) for c in unique_contrasts])

    ax.set_title(f"{region}, single session", fontsize=f_size_l)
    ax.plot(-unique_contrasts, neurometric_curve, c='k')
    ax.plot(-unique_contrasts, neurometric_curve, 'ko', ms=3)
    ax.errorbar(-unique_contrasts, neurometric_curve, neurometric_curve_err, color='k')
    ax.set_ylim(0,1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim(-1.03,1.03)
    ax.set_xticks([-1.0, -0.25, -0.125, -0.0625,  0, 0.0625, 0.125, 0.25, 1.0])

    ax.set_xticklabels([-1] + ['']*7 + [1])             
    ax.set_xlabel('Stimulus contrast', fontsize=f_size)
    ax.set_ylabel('Predicted \n P(stim = right)', fontsize=f_size)
    ax.set_title(f"{region}, single session", fontsize=f_size_l)

    if tight:
        fig.tight_layout()

    if save:
        save_path = save_path or imgs_variable_path('stim')
        fig.savefig(save_path.joinpath('stim_dec_line.svg'))
        fig.savefig(save_path.joinpath('stim_dec_line.pdf'))



def plot_decoding_scatter(variable, ax=None, save=True, save_path=None):

    """
    Plot decoding scatter for
    variable in [choice, fback]
    Previously called dec_scatter
    :param variable:
    :param fig:
    :param ax:
    :return:
    """

    if ax is None:
        tight = True
        fig, ax = plt.subplots(figsize=(60*MM_TO_INCH, 38*MM_TO_INCH))
    else:
        tight = False
        fig = ax.get_figure()

    assert variable in ['choice', 'fback'], 'variable must be in [choice, fback]'

    preds, targets, mask, contrasts, region = load_decoding_for_variable(variable)

    trials = np.arange(len(mask))[[m==1 for m in mask]]

    ax.plot(trials[targets == -1], preds[targets == -1] if variable == 'fback' else 1 - preds[targets == -1],
            'o', c=RED, ms=2)
    ax.plot(trials[targets == 1], preds[targets == 1] if variable == 'fback' else 1 - preds[targets == 1],
            'o', c=BLUE, ms=2)

    if variable == 'fback':
        leg = ['Incorrect', 'Correct']
        ylabel = f'Predicted\n {leg[1].lower()}'
    elif variable == 'choice':
        leg = ['Right choice', 'Left choice']
        ylabel = f'Predicted\n {leg[0].lower()}'

    loc = (1.4, 0.8) if variable == 'fback' else (1.5, 1.1)
    ax.legend(leg, frameon=False, fontsize=f_size_s, bbox_to_anchor=loc, loc='upper right',
              handletextpad=0.1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(100, 400)
    ax.set_xlabel('Trials', fontsize=f_size)
    ax.set_ylabel(ylabel, fontsize=f_size)
    ax.set_title(f"{region}, single session", fontsize=f_size_l)

    if tight:
        # TODO FIX me figure this out
        fig.tight_layout()
    if save:
        save_path = save_path or imgs_variable_path(variable)
        fig.savefig(save_path.joinpath(f'{variable}_dec_scatter.svg'))
        fig.savefig(save_path.joinpath(f'{variable}_dec_scatter.pdf'))


def wheel_decoding_ex(variable, axs=None, save=True):

    """"
    For vari in speed, velocity
    show example trials for decoding
    """


    n_pseudo = 2 if variable == 'velocity' else 4
    session_file = (f'wheel-{variable}_671c7ea7-6726-4fbe-adeb-f89c2c8e489b_GRN_merged_probes_pseudo_ids_-1_{n_pseudo}.pkl')
    data = pd.read_pickle(open(Path(dec_pth, session_file), 'rb'))

    trials = [113, 216]

    if variable == 'speed':
        ymin, ymax = -0.9, 8.75
    else:
        ymin, ymax = -8.5, 8.5    
        
    # base fit
    fit = data['fit'][0]

    # get average predictions from all runs
    run_idxs = np.unique([i for i, fit in enumerate(data['fit']) if fit['pseudo_id'] == -1])
    preds = []
    n_preds = len(fit['predictions_test'])
    for n in range(n_preds):
        preds_tmp = np.concatenate([np.squeeze(data['fit'][i]['predictions_test'][n])[:, None] for i in run_idxs], axis=1,)
        preds.append(np.mean(preds_tmp, axis=1))
    # get decoding target
    targs = fit['target']
    # get good trials
    mask = fit['mask'][0]
    trial_idxs = np.where(mask)[0]
    # get some other metadata
    eid = data['eid']
    region = data['region']
    r2 = np.mean([data['fit'][i]['scores_test_full'] for i in range(2)])

    # build plot
    if axs is None:
        tight = True
        fig, axs = plt.subplots(1, len(trials), figsize=(4.32, 1.63))
        print(f"session: {eid} \n region: {region}")
        print(f" \n $R^2$ = {r2:.3f} (average across 2 models)")
    else:
        tight = False
        fig = axs[0].get_figure()

    movetime = 0.2

    for i, (t, ax) in enumerate(zip(trials, axs)):
        targs_curr, preds_curr, trial_curr = targs[t], preds[t], trial_idxs[t]
        ax.plot((np.arange(len(targs_curr))-10) * 0.02 + movetime, targs_curr, 'k')
        ax.plot((np.arange(len(targs_curr))-10) * 0.02 + movetime, preds_curr, 'r')
        ax.plot(np.zeros(50) + movetime, np.linspace(ymin, ymax), 'k--',)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(f'{variverb[variable]} (rad./s)', fontsize=f_size)
        ax.set_xticks([0, .20, 0.5, 1.0], ['-0.2', '0.0', '0.5', '1.0'])
        ax.set_xlabel(EVENT_LABELS[variable], fontsize=f_size)
        ax.set_title(f'Trial {trial_curr} - {region[0]}', fontsize=f_size_l)
        ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))

        if i == 1:
            ax.text(0.22, 0.8, EVENT_LINES[variable], transform=ax.transAxes, fontsize=f_size_s)
            ax.set_ylabel('')
        else:
            ax.legend(['Actual', 'Predicted'], frameon=False, fontsize=f_size_s, bbox_to_anchor=(0.5, 1))

    if tight:
        fig.tight_layout()

    if save:
        fig.savefig(Path(imgs_pth, variable, 'wheel_decoding_ex.svg'),bbox_inches='tight')
                     

def plot_decoding_speed_velocity(axs=None, save=True):

    """
    SI figure comparing wheel velocity/speed decoding
    Previously called plot_SI_speed_velocity
    :return:
    """

    MIN_UNITS = 5
    MIN_TRIALS = 250
    MIN_SESSIONS_PER_REGION = 2    
    
    targets = ['wheel-velocity', 'wheel-speed']
    results = {target: {} for target in targets}
    for target in targets: #, 'wheel-speed']:
        
        # Load results combined across sessions for each region
        file_stage3_results = os.path.join(dec_pth, f'{target}_stage3.pqt')
        res_table_final = pd.read_parquet(file_stage3_results)

        # 0nly get regions from final results table
        regions = res_table_final[res_table_final.n_sessions >= MIN_SESSIONS_PER_REGION]['region'].values
        
        results[target]['vals'] = res_table_final
        results[target]['regions'] = regions
        
    # load raw targets
    dfs = {}
    for target in targets:
        imposter_file = os.path.join(dec_pth, f'imposterSessions_{target}.pqt')
        df_tmp = pd.read_parquet(imposter_file)
        vals = df_tmp.loc[:, target].to_numpy()
        dfs[target] = np.concatenate([v[None, :] for v in vals], axis=0)    

    metrics_to_plot = {
        'valuesminusnull_median': '$R^2$, null corrected',
        'values_median': '$R^2$',
        'null_median_of_medians': '$R^2$, median null',
    }
    n_metrics = len(metrics_to_plot.keys())
    n_panels = n_metrics + 1

    if axs is None:
        fig, axs = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3))
        tight = True
    else:
        fig = axs[0].get_figure()
        tight = False

    _, pal = get_allen_info()

    # -----------------------
    # metrics
    # -----------------------
    for ax, metric in zip(axs[:len(metrics_to_plot)], metrics_to_plot.keys()):

        # Plot values
        cols = [pal[reg] for reg in results['wheel-velocity']['vals'].region]
        ax.scatter(results['wheel-velocity']['vals'][metric], results['wheel-speed']['vals'][metric], s=1, color=cols)

        # Plot diagonal line
        xs = np.linspace(0, 0.55)
        if metric != 'null_median_of_medians':
            ax.plot(xs, xs, 'k--')
        # Plot text
        for x, y, s in zip(
            results['wheel-velocity']['vals'][metric],
            results['wheel-speed']['vals'][metric],
            results['wheel-velocity']['vals'].region,
        ):
            if np.isnan(x) or np.isnan(y):
                continue
            ax.text(x, y, s, fontsize=f_size_xs, color=pal[s])

        ax.set_xlabel(f'Wheel-velocity ({metrics_to_plot[metric]})', fontsize=f_size)
        ax.set_ylabel(f'Wheel-speed ({metrics_to_plot[metric]})', fontsize=f_size)

    # -----------------------
    # target shapes
    # -----------------------
    ts = np.arange(60) * 0.020 - 0.2
    axs[-1].plot(ts, np.median(dfs['wheel-speed'], axis=0))
    axs[-1].plot(ts, np.median(dfs['wheel-velocity'], axis=0))
    axs[-1].fill_between(ts, np.percentile(dfs['wheel-speed'], 5, axis=0),
                         np.percentile(dfs['wheel-speed'], 95, axis=0), alpha=0.2, color='C0')
    axs[-1].fill_between(ts, np.percentile(dfs['wheel-velocity'], 5, axis=0),
                         np.percentile(dfs['wheel-velocity'], 95, axis=0), alpha=0.2, color='C1')
    axs[-1].legend(['Wheel-speed', 'Wheel-velocity'], fontsize=f_size_s, frameon=False, bbox_to_anchor=(1, 0.2),
                   handlelength=handle_length, handletextpad=handle_pad)
    axs[-1].axvline(x=0, linestyle='--', c='k')
    axs[-1].text(0, 4.9, "  Movement onset", color='k', fontsize=f_size_s, horizontalalignment='left')
    axs[-1].set_xlabel('Time from move (s)', fontsize=f_size)
    axs[-1].set_ylabel('Within trial speed/ velocity', fontsize=f_size)

    if tight:
        fig.tight_layout()

    if save:
        fig.savefig(Path(imgs_pth, 'si', f'n6_supp_figure_decoding_wheelspeedvsvel.pdf'), dpi=150)

    return fig


def plot_swansons_decoding(variable, ax=None, adjust=False, save=True, annotate_list=None):

    """
    SI figure swanson for decoding results above slices
    (needs manual assembly, putting swansons on top of slices)
    showing the fraction of significant sessions per region

    Previously called swansons_SI_dec

    :param variable:
    :return:
    """

    cmap = get_cmap_for_variable(variable)

    if ax is None:
        fig, ax = plt.subplots(figsize=(120 * MM_TO_INCH, 60 * MM_TO_INCH))
    else:
        fig = ax.get_figure()

    # Load data
    df = pd.read_parquet(dec_pth.joinpath(f"{dec_d[variable]}_stage2.pqt"))

    # Fraction of significant sessions per region
    regs = np.unique(df['region'])
    res = {reg: len(np.unique(df[(df['region'] == reg) & (df['p-value'] < sigl)]['eid'])) /
                len(np.unique(df[df['region'] == reg]['eid'])) for reg in regs}

    acronyms = np.array(list(res.keys()))
    scores = np.array(list(res.values()))

    # Turn regions with zero sig sessions to grey
    mask = acronyms[scores == 0]

    acronyms = acronyms[scores != 0]
    scores = scores[scores != 0]

    # Turn fraction into percentage
    scores = scores * 100

    vmin, vmax = (0, 100)

    annotate_kwargs = {'annotate_list': annotate_list, 'annotate': True}
    _, cax = plot_horizontal_swanson(acronyms, scores, mask=mask, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                                     annotate_kwargs=annotate_kwargs, cbar=True, legend=True, fontsize=f_size_xs,
                                     cbar_label='Fraction of significant sessions (%)', cbar_shrink=0.3,
                                     mask_label='Zero significant sessions (%)')

    # Adjust the swansons so that we can annotate with regions
    ax.set_ylim(0, 3250)
    ax.invert_yaxis()

    # Add title
    ax.set_title('Decoding',fontsize=f_size_l)
    ax.title.set_position((0.1, 0.8))

    if adjust:
        adjust_horizontal_swansons(ADJUST_DECODING_SWANSONS[variable], ax)

    if save:
        fig.savefig(Path(imgs_pth, 'si', f'decSwanson_SI_{variable}.svg'))


    return ax, cax


def plot_female_male_repro(axs=None, save=True):

    """"
    For the 5 repro regions, and all variables (stimulus, choice, feedback),
    plot results split by female and male mice with two strips of dots 
    (blue for male, red for female) for each region.
    """

    repro_regs = ["VISa/am", "CA1", "DG", "LP", "PO"]

    # subject = one.get_details(eid)['subject']
    # info = one.alyx.rest('subjects', 'read', id=subject)
    # sex = info['sex']  # 'M', 'F', or 'U'

    # Load the male/female subject dictionary
    data = np.load(os.path.join(dec_pth, 'male_female.npy'), allow_pickle=True).flat[0]

    # Set up the plot structure (3 rows for each variable, 1 column)
    if axs is None:
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8.26, 4.37), sharey=True, sharex=True)
        tight = True
    else:
        fig = axs[0].get_figure()
        tight = False

    for row_idx, variable in enumerate(VARIABLES):

        pqt_file = dec_pth.joinpath(f"{dec_d[variable]}_stage2.pqt")
        df = pd.read_parquet(pqt_file)

        # Combine 'VISa' and 'VISam' into 'VISa/am'
        df['region'] = df['region'].replace({'VISa': 'VISa/am', 'VISam': 'VISa/am'})

        # Restrict the DataFrame to the 5 repro regions
        df = df[df['region'].isin(repro_regs)]

        # Map the sex information to the DataFrame
        df['sex'] = df['subject'].map(data)

        # Separate by sex
        female_df = df[df['sex'] == 'F']
        male_df = df[df['sex'] == 'M']

        # Combine male and female data into a new DataFrame for better comparison in the plot
        df_combined = pd.concat([female_df.assign(gender='Female'), male_df.assign(gender='Male')])

        # Plot in the corresponding row
        ax = axs[row_idx]
        
        # Strip plot for both male (blue) and female (red) side by side for each region
        sns.stripplot(data=df_combined, x='region', y='score', hue='gender', palette={'Male': 'blue', 'Female': 'red'},
                      dodge=True, jitter=True, size=3, ax=ax)

        # Add mean null score (white dot) for each gender and region
        # Ensure the x-alignment is consistent (no dodge for null scores)
        for region in repro_regs:
            # Get the x position for this region (for both male and female)
            x_female = repro_regs.index(region) - 0.2  # Slight left shift for female (consistent with dodge)
            x_sale = repro_regs.index(region) + 0.2    # Slight right shift for male (consistent with dodge)

            # Plot null score for females (red)
            female_null = female_df[female_df['region'] == region]['median-null'].mean()
            ax.scatter(x_female, female_null, color='white', edgecolor='red', zorder=3, s=30)

            # Plot null score for males (blue)
            male_null = male_df[male_df['region'] == region]['median-null'].mean()
            ax.scatter(x_sale, male_null, color='white', edgecolor='blue', zorder=3, s=30)

        # Set x labels and titles only for bottom row
        if row_idx == 2:
            ax.set_xlabel("Brain Regions")
        else:
            ax.set_xticklabels([])
            ax.set_xlabel('')

        ax.set_ylabel(f"{variverb[variable].capitalize()} \n \n Decoding accuracy", fontsize=f_size)
        
        # Remove legend from individual plots to keep the layout clean
        ax.get_legend().remove()
        for i in range(1, len(repro_regs)):
            ax.axvline(x=i-0.5, color='grey', linestyle='--', linewidth=1)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='white', label='Male',
                   markerfacecolor='blue', markeredgecolor='blue', markersize=2.5),
        plt.Line2D([0], [0], marker='o', color='white', label='Female',
                   markerfacecolor='red', markeredgecolor='red', markersize=2.5),
        plt.Line2D([0], [0], marker='o', color='white', label='Null (Male)',
                   markerfacecolor='white', markeredgecolor='blue', markersize=5),
        plt.Line2D([0], [0], marker='o', color='white', label='Null (Female)',
                   markerfacecolor='white', markeredgecolor='red', markersize=5)
    ]

    axs[0].legend(handles=legend_handles, loc='upper center', frameon=False, ncols=2, fontsize=f_size_s,
                  bbox_to_anchor=(0.92, 0.99), columnspacing=0.1, handletextpad=0.2).set_draggable(True)

    # Adjust layout
    if tight:
        fig.tight_layout()
    if save:
        fig.savefig(Path(imgs_pth, 'si', f'n6_supp_dec_repro_male_female.pdf'), dpi=150)

    return fig


def print_age_weight():

    '''
    for all BWM mice print age and weight and means
    '''
    subjects = bwm_query(one)['subject'].unique()
    d = {}
    for sub in subjects:
        print(sub)
        info = one.alyx.rest('subjects', 'read', id=sub)
        d[sub] = [info['age_weeks'], info['reference_weight']]

    print('mean age [weeks]',  np.round(np.mean([d[s][0] for s in d]),2))
    print('median age [weeks]',  np.round(np.median([d[s][0] for s in d]),2))
    print('max, min [weeks]', np.max([d[s][0] for s in d]), 
                              np.min([d[s][0] for s in d]))

    print('mean weight [gr]',  np.round(np.median([d[s][1] for s in d]),2))
    print('median weight [gr]',  np.round(np.mean([d[s][1] for s in d]),2))                                     
    print('max, min [gr]', np.max([d[s][1] for s in d]), 
                              np.min([d[s][1] for s in d]))

# -------------------------------------------------------------------------------------------------
# ENCODING ANALYSIS
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# Loading utils
# -------------------------------------------------------------------------------------------------
regressor2full = {
    'stimonL': 'Left stim',
    'stimonR': 'Right stim',
    'incorrect': 'Incorrect',
    'correct': 'Correct',
    'fmoveL': 'Left\nchoice',
    'fmoveR': 'Right\nchoice',
}

def load_glm_params():
    # Please use the saved parameters dict from 02_fit_sessions.py as params
    return pd.read_pickle(enc_pth /"glm_params.pkl")

def load_unit_fit_model(eid, pid, clu_id):
    glm_params = load_glm_params()

    stdf, sspkt, sspkclu, reg, clu_df = load_regressors(eid, pid, one, t_before=0.6, t_after=0.6,
                                                        binwidth=glm_params["binwidth"], abswheel=True)

    design = generate_design(stdf, stdf["probabilityLeft"], t_before=0.6, **glm_params)
    clu_idx = np.where(clu_df['cluster_id'] == clu_id)[0][0]
    spkmask = sspkclu == clu_idx
    nglm = lm.LinearGLM(design, sspkt[spkmask], sspkclu[spkmask], estimator=glm_params["estimator"], mintrials=0)
    nglm.fit()

    return stdf, sspkt, sspkclu, design, spkmask, nglm, clu_idx

# -------------------------------------------------------------------------------------------------
# Loading utils
# -------------------------------------------------------------------------------------------------

def plot_twocond(eid, pid, clu_id, align_time, aligncol, aligncond1, aligncond2, t_before, t_after,
                 regressors, ax=None):

    glm_params = load_glm_params()
    # Load in data and fit model for a particular cluster
    (stdf, sspkt, sspkclu, design, spkmask, nglm, clu_idx) = load_unit_fit_model(eid, pid, clu_id)

    # Construct GLM prediction object that does our model predictions
    pred = GLMPredictor(stdf, nglm, sspkt, sspkclu)
  
    # Construct design matrix without regressors of interest
    noreg_dm = remove_regressors(design, regressors)

    # Fit model without regressors of interest
    nrnglm = lm.LinearGLM(noreg_dm, sspkt[spkmask], sspkclu[spkmask], estimator=glm_params["estimator"], mintrials=0)
    nrnglm.fit()

    # Construct GLM prediction object that does model predictions without regressors of interest
    nrpred = GLMPredictor(stdf, nrnglm, sspkt, sspkclu)

    # Compute model predictions for each condition
    keyset1 = pred.compute_model_psth(align_time, t_before, t_after,trials=stdf[aligncond1(stdf[aligncol])].index)
    cond1pred = pred.full_psths[keyset1][clu_idx][0]

    keyset2 = pred.compute_model_psth(align_time, t_before, t_after,trials=stdf[aligncond2(stdf[aligncol])].index)
    cond2pred = pred.full_psths[keyset2][clu_idx][0]

    nrkeyset1 = nrpred.compute_model_psth(align_time, t_before, t_after,trials=stdf[aligncond1(stdf[aligncol])].index)
    nrcond1pred = nrpred.full_psths[nrkeyset1][clu_idx][0]

    nrkeyset2 = nrpred.compute_model_psth(align_time, t_before, t_after, trials=stdf[aligncond2(stdf[aligncol])].index)
    nrcond2pred = nrpred.full_psths[nrkeyset2][clu_idx][0]

    # Plot PSTH of original units and model predictions in both cases
    if not ax:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey="row")
    else:
        fig = ax[0].get_figure()
        
    lw = mpl.rcParams['lines.linewidth']
    x = np.arange(-t_before, t_after, nglm.binwidth)
    for rem_regressor in [False, True]:
        i = int(rem_regressor)
        oldticks = []
        peri_event_time_histogram(sspkt, sspkclu, stdf[aligncond1(stdf[aligncol])][align_time], clu_idx,
                                  t_before, t_after, bin_size=nglm.binwidth, error_bars="sem", ax=ax[i],
                                  smoothing=0.01, pethline_kwargs={"color": BLUE, "linewidth": lw},
                                  errbar_kwargs={"color": BLUE, "alpha": 0.5})
        oldticks.extend(ax[i].get_yticks())

        peri_event_time_histogram(sspkt, sspkclu, stdf[aligncond2(stdf[aligncol])][align_time], clu_idx,
                                  t_before, t_after, bin_size=nglm.binwidth, error_bars="sem", ax=ax[i],
                                  smoothing=0.01, pethline_kwargs={"color": RED, "linewidth": lw},
                                  errbar_kwargs={"color": RED, "alpha": 0.5})

        oldticks.extend(ax[i].get_yticks())
        pred1 = cond1pred if not rem_regressor else nrcond1pred
        pred2 = cond2pred if not rem_regressor else nrcond2pred
        ax[i].step(x, pred1, color="skyblue", linewidth=lw)
        oldticks.extend(ax[i].get_yticks())
        ax[i].step(x, pred2, color="#F08080", linewidth=lw)
        oldticks.extend(ax[i].get_yticks())
        ax[i].set_ylim([0, np.max(oldticks) * 1.1])

    return fig, ax, sspkt, sspkclu, stdf, clu_idx


def get_example_results():
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
            "fmoveL", #choice left
            "fmoveR", #choice right
        ),
        "feedback_times": (
            "feedbackType",
            lambda f: f == 1,
            lambda f: f == -1,
            0.1,
            0.4,
            "correct", #correct
            "incorrect", #incorrect
        ),
    }

    # Which units we're going to use for plotting
    targetunits = {  # eid, pid, clu_id, region, drsq, alignset key
        "stim": (
            'e0928e11-2b86-4387-a203-80c77fab5d52',  # EID 
            '799d899d-c398-4e81-abaf-1ef4b02d5475',  # PID 
             598, #235,  # clu_id, was 235 -- online 218 looks good
            "VISp",  # region
            0.04540706,  # drsq (from 02_fit_sessions.py)
            "stimOn_times",  # Alignset key
        ),
        "choice": (
            "a7763417-e0d6-4f2a-aa55-e382fd9b5fb8",#"671c7ea7-6726-4fbe-adeb-f89c2c8e489b"
            "57c5856a-c7bd-4d0f-87c6-37005b1484aa",#"04c9890f-2276-4c20-854f-305ff5c9b6cf"
            80, # 74 was 143
            "GRN",
            0.000992895,  # drsq
            "firstMovement_times",
        ),
        "fback": (
            "a7763417-e0d6-4f2a-aa55-e382fd9b5fb8",
            "57c5856a-c7bd-4d0f-87c6-37005b1484aa",
            359, #83 clu-id was 83 before
            "IRN",
            0.3077195113,  # drsq
            "feedback_times",
        ),
    }

    sortlookup = {"stim": "side", 
                  "choice": "movement", 
                  "fback": "fdbk", 
                  "wheel": "movement"}
    
    return targetunits, alignsets, sortlookup
    
    
def plot_encoding_for_variable(variable, cluster_id=None, axs=None, save=True, save_path=None):

    """
    Plot raster and two line plots
    ax = [ax_raster, ax_line0, ax_line1]
    Previously called encoding raster lines
    :param variable:
    :param clu_id0:
    :param axs:
    :param frac_tr:
    :return:
    """

    frac_tr = 10 if variable == 'fback' else 3

    if axs is None:
        tight = True
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(80 * MM_TO_INCH, 140 * MM_TO_INCH),
                                sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    else:
        tight = False
        fig = axs[0].get_figure()

    targetunits, alignsets, sortlookup = get_example_results()
    eid, pid, clu_id, region, drsq, aligntime = targetunits[variable]

    if cluster_id:
        clu_id = cluster_id

    (aligncol, aligncond1, aligncond2, t_before, t_after, reg1, reg2) = alignsets[aligntime]
        
    _, _, sspkt, sspkclu, stdf, clu_idx = (
        plot_twocond(eid, pid, clu_id, aligntime, aligncol, aligncond1, aligncond2, t_before, t_after,
                     [reg1, reg2] if variable != "wheel" else ["wheel"], ax = [axs[1],axs[2]]))

    for ax in [axs[1], axs[2]]:
        for coll in ax.collections:
            if isinstance(coll, matplotlib.collections.LineCollection):
                coll.remove()
        ax.axvline(x=0, linestyle='--', c='k')
    
    axs[1].set_ylabel('Firing rate (Hz)', fontsize=f_size)
    axs[2].set_ylabel('Firing rate (Hz)', fontsize=f_size)

    axs[1].yaxis.set_major_locator(plt.MaxNLocator(4))
    axs[2].yaxis.set_major_locator(plt.MaxNLocator(4))

    names = ['Full model', 'Model without selected regressors']
    for ax, title in zip([axs[1],axs[2]], names):
        ax.set_title(title, fontsize=f_size) #, va='top')

    # custom legend
    all_lines = axs[2].get_lines()
    legend_labels = [regressor2full[reg1], regressor2full[reg2], 'Model', 'Model']
    axs[2].legend(all_lines, legend_labels, loc='upper right',  bbox_to_anchor=(1.45, 0.85), fontsize=f_size_s,
                  frameon=False, handlelength=handle_length, handletextpad=handle_pad)

    stdf["response_times"] = stdf["stimOn_times"]
    trial_idx, dividers = find_trial_ids(stdf, sort=sortlookup[variable])

    _, _ = single_cluster_raster(sspkt[sspkclu == clu_idx], stdf[aligntime], trial_idx, dividers,
                                 [BLUE, RED], [regressor2full[reg1], regressor2full[reg2]], pre_time=t_before,
                                 post_time=t_after, raster_cbar=False, raster_bin=0.002, frac_tr = frac_tr,
                                 axs=axs[0])

    axs[0].axhline(y=dividers[0], c='k', linewidth=0.5)
    axs[0].set_ylabel('Trials', fontsize=f_size)

    axs[0].set_title("{} unit {} $\log \Delta R^2$ = {:.2f}".format(region, clu_id, np.log10(drsq)),
                     fontsize=f_size_l)
    axs[0].sharex(axs[2])
    axs[1].set_xlabel(None)
    axs[2].set_xlabel(EVENT_LABELS[variable], fontsize=f_size)
                  
    if tight:
        fig.tight_layout()
    if save:
        save_path = save_path or imgs_variable_path(variable)
        fig.savefig(save_path.joinpath(f'{variable}_encoding_raster_lines.svg'))
        fig.savefig(save_path.joinpath(f'{variable}_encoding_raster_lines.pdf'))


def encoding_wheel_boxen(ax=None, save=True):

    d = {}
    fs = {'speed': 'GLMs_wheel_speed.pkl',
          'velocity': 'GLMs_wheel_velocity.pkl'} 

    for v in fs:
        d[v] = pd.read_pickle(Path(enc_pth, fs[v]))['mean_fit_results']["wheel"].to_frame()

    joinwheel = d['speed'].join(d['velocity'], how="inner", rsuffix="_velocity", lsuffix="_speed")
    meltwheel = joinwheel.melt()

    if ax is None:
        tight = True
        fig, ax = plt.subplots(figsize=[85 * MM_TO_INCH, 70 * MM_TO_INCH ])
    else:
        tight = False
        fig = ax.get_figure()
    
    ax = sns.boxenplot(data=meltwheel, y="value", x="variable", hue="variable", dodge=False, ax=ax, legend=False,
                       palette={'wheel_speed': sns.color_palette()[0], 'wheel_velocity': sns.color_palette()[1]})
                                 
    ax.set_ylim([-0.015, 0.04])
    ax.set_xticklabels(['Wheel-speed', 'Wheel-velocity'], fontsize=f_size)
    ax.set_xlabel('')
    ax.set_ylabel(r'Distribution of population $\Delta R^2$')

    if tight:
        fig.tight_layout()
    if save:
        fig.savefig(Path(imgs_pth, 'speed', 'glm_boxen.svg'))


def encoding_wheel_2d_density(ax=None, fig=None):
    # Load data and configurations
    d = {}
    fs = {'speed': 'GLMs_wheel_speed.pkl',
          'velocity': 'GLMs_wheel_velocity.pkl'} 

    for v in fs:
        d[v] = pd.read_pickle(
            Path(enc_pth, fs[v]))['mean_fit_results']["wheel"].to_frame()

    # Join data
    joinwheel = d['speed'].join(d['velocity'], how="inner", 
                                rsuffix="_velocity", lsuffix="_speed")
    
    # Extract the columns to plot
    x = joinwheel.iloc[:, 0]  # wheel_speed values
    y = joinwheel.iloc[:, 1]  # wheel_velocity values
    
    # Check if we are plotting alone or on provided axis
    alone = False
    if not ax:
        alone = True
        fig, ax = plt.subplots(constrained_layout=True, figsize=[5, 5])  
    
    # Plot 2D density using seaborn's kdeplot
    sns.kdeplot(x=x, y=y, ax=ax, cmap="Blues", fill=True)

    # Label axes
    ax.set_xlabel('Wheel-speed')
    ax.set_ylabel('Wheel-velocity')

    ax.set_aspect('equal')
    # Save plot if alone
    if alone:
        fig.tight_layout()
        fig.savefig(Path(imgs_pth, 'speed', 'glm_2d_density.svg'))



# -------------------------------------------------------------------------------------------------
# POPULATION TRAJECTORY
# -------------------------------------------------------------------------------------------------


TRAJECTORY_EXAMPLE_REGIONS = {
    'stim': ['VISp', 'LGd', 'PRNc', 'VISam', 'IRN', 'VISl', 'VISpm', 'VM', 'MS', 'VISli'],
    'choice': ['GRN', 'PRNc', 'VISal', 'PRNr', 'LSr', 'SIM', 'APN', 'MRN', 'RT', 'LGd', 'MV', 'ORBm'],
    'fback': ['IRN', 'SSp-n', 'PRNr', 'IC', 'MV', 'AUDp', 'CENT3', 'SSp-ul', 'GPe']
}

REGION_COLS['all'] = (0.32156863, 0.74901961, 0.01568627, 1)

# canonical colors for left and right trial types
blue_left = BLUE
red_right = RED
ntravis = 30  # #trajectories for vis, first 2 real, rest pseudo

ADJUST_TRAJECTORIES = {
    'stim': {'LGd': {'x_l': 0, 'y_l': -0.4, 'x_s': -0.5, 'y_s': 1.5},
             'VISp': {'x_l': -10, 'y_l': 2.2, 'x_s': -0.5, 'y_s': 3},
             'PRNc': {'x_l': 0, 'y_l': 0, 'x_s': -1, 'y_s': 2},
             'VISam': {'x_l': 0, 'y_l': 0.8, 'x_s': 0.8, 'y_s': 0.3},
             'IRN': {'x_l': 0, 'y_l': 0, 'x_s': 2, 'y_s': -1},
             'VISl': {'x_l': 0, 'y_l': -1.5, 'x_s': -0.75, 'y_s': -3.5},
             'VISpm': {'x_l': 0, 'y_l': -1.2, 'x_s': 1, 'y_s': 4.5},
             'VM': {'x_l': -3.5, 'y_l': 1, 'x_s': 1, 'y_s': 6, 'ha_s': 'center'},
             'MS': {'x_l': -9, 'y_l': -2.5, 'x_s': -0.5, 'y_s': 0},
             'VISli': {'x_l': -3.8, 'y_l': -2.1, 'x_s': 0.5, 'y_s': -2.5}

             },

    'choice': {'PRNc': {'x_l': 0, 'y_l': 0, 'x_s': 2.7, 'y_s': 1.5},
               'VISal': {'x_l': -4.2, 'y_l': 3, 'x_s': 0, 'y_s': 3},
               'PRNr': {'x_l': 0, 'y_l': -2, 'x_s': -5, 'y_s': 1},
               'LSr': {'x_l': 0, 'y_l': 1.8, 'x_s': -1.5, 'y_s': -1.5},
               'SIM': {'x_l': 0, 'y_l': -0.5, 'x_s': 3.2, 'y_s': -2},
               'APN': {'x_l': 0, 'y_l': -1, 'x_s': -3, 'y_s': 3},
               'MRN': {'x_l': 0, 'y_l': -0.5, 'x_s': -3, 'y_s': 3},
               'RT': {'x_l': 0, 'y_l': 0.3, 'x_s': 3, 'y_s': 0},
               'LGd': {'x_l': 0, 'y_l': -2, 'x_s': 1, 'y_s': -2},
               'GRN': {'x_l': 0, 'y_l': 0, 'x_s': -2, 'y_s': 0},
               'MV': {'x_l': 0, 'y_l': -0.8, 'x_s': -2, 'y_s': 2.5},
               'ORBm': {'x_l': 0, 'y_l': -0.5, 'x_s': -2.5, 'y_s': 3.5, 'ha_s': 'center'},
               },

    'fback': {'IRN': {'x_l': 0, 'y_l': 0, 'x_s': 1, 'y_s': -0.5},
              'SSp-n': {'x_l': 0, 'y_l': 0, 'x_s': 1.5, 'y_s': 3.5},
              'PRNr': {'x_l': 0, 'y_l': -2.5, 'x_s': 1, 'y_s': 4},
              'IC': {'x_l': -2.5, 'y_l': 21, 'x_s': 1, 'y_s': 1},
              'MV': {'x_l': 0, 'y_l': 0.5, 'x_s': 0.4, 'y_s': -2},
              'AUDp': {'x_l': -6.8, 'y_l': 30, 'x_s': 0.25, 'y_s': 0},
              'CENT3': {'x_l': 0, 'y_l': -1.3, 'x_s': 0.9, 'y_s': 2},
              'SSp-ul': {'x_l': 0, 'y_l': -1.8, 'x_s': 0.9, 'y_s': -0.2},
              'GPe': {'x_l': 0, 'y_l': 1, 'x_s': 0.5, 'y_s': -11},
              }
}


def get_name(brainregion):
    regid = br.id[np.argwhere(br.acronym == brainregion)][0, 0]
    return br.name[np.argwhere(br.id == regid)[0, 0]]
    
         
def pre_post(variable, can=False):
    """
    [pre_time, post_time] relative to alignment event
    variable could be contr or restr variant, then
    use base window
    
    ca: If true, use canonical time windows
    """

    pre_post0 = {'stim': [0, 0.15],
                 'choice': [0.15, 0],
                 'fback': [0, 0.7]}

    # canonical windows
    pre_post_can =  {'stim': [0, 0.1],
                     'choice': [0.1, 0],
                     'fback': [0, 0.2]}

    pp = pre_post_can if can else pre_post0

    if '_' in variable:
        return pp[variable.split('_')[0]]
    else:
        return pp[variable]


def grad(c, nobs, fr=1):
    """
    color gradient for plotting trajectories
    c: color map type
    nobs: number of observations
    """

    cmap = mpl.cm.get_cmap(c)

    return [cmap(fr * (nobs - p) / nobs) for p in range(nobs)]

# -------------------------------------------------------------------------------------------------
# Loading utils
# -------------------------------------------------------------------------------------------------
def get_trajectory_variable(variable):
    if '_' in variable:
        variable = variable.split('_')[0]
    return variable

def load_trajectory_for_variable(variable, averages=False):

    if averages:
        data = np.load(Path(man_pth, f'{variable}_grand_averages.npy'), allow_pickle=True).flat[0]
    else:
        data = np.load(Path(man_pth, f'{variable}.npy'), allow_pickle=True).flat[0]

    return get_trajectory_variable(variable), data

# -------------------------------------------------------------------------------------------------
# Plotting utils
# -------------------------------------------------------------------------------------------------

ADJUST_3D_TRAJECTORIES = {
    'stim': {
        'view': {'elev': 30, 'azim': -153, 'roll': 1},
        'zoom': 0.6, #0.65,
        'trans': {'shift_x': -2, 'shift_y': 0, 'shift_z': 0},
        'origin': {'x':0.3, 'y': 0.1, 'l': 0.1},
        'annotations': [
            {'text': 't = 0', 'x': 0.9, 'y': 1, 'c': 'k', 'r': 0},
            {'text': 't = 50 ms', 'x': 0.8, 'y': -0.1, 'c': 'k', 'r': 0},
            {'text': 'Left stimulus', 'x': 0.45, 'y': 0.85, 'c': BLUE, 'r': 0},
            {'text': 'Right stimulus', 'x': 0.35, 'y': 0.55, 'c': RED, 'r': 0},
            {'text': 'Control', 'x': 0.8, 'y': 0.2, 'c': 'grey', 'r': 0}
        ]
    },
    'fback': {
        'view': {'elev': 10, 'azim': -96, 'roll': 8},
        #'view': {'elev': 26, 'azim': -84, 'roll': 4},
        'zoom': 0.7, #0.8,
        'trans': {'shift_x': -1.5, 'shift_y': 0, 'shift_z': 0},
        'origin': {'x':0.175, 'y': 0.1, 'l': 0.1},
        'annotations': [
            {'text': 't = 0', 'x': 0.6, 'y': 1, 'c': 'k', 'r': 0},
            {'text': 't = 700 ms', 'x': 0.35, 'y': 0.4, 'c': 'k', 'r': 0},
            {'text': 'Incorrect', 'x': 0.4, 'y': 0.9, 'c': RED, 'r': 0},
            {'text': 'Correct', 'x': 0.85, 'y': -0.1, 'c': BLUE, 'r': 0},
            {'text': 'Control', 'x': 0.5, 'y': 0.5, 'c': 'grey', 'r': 0}]
    },
    'choice': {
        'view': {'elev': 13, 'azim': -98, 'roll': -5},
        'zoom': 0.65, #0.8
        'trans': {'shift_x': -2, 'shift_y': -1, 'shift_z': 0},
        'origin': {'x': 0.25, 'y': 0.15, 'l': 0.1},
        'annotations': [
            {'text': 't = 0', 'x': 1, 'y': 0.6, 'c': 'k', 'r': 0},
            {'text': 't = -150 ms', 'x': 0.23, 'y': 0.6, 'c': 'k', 'r': 0},
            {'text': 'Right choice', 'x': 0.5, 'y': 0.8, 'c': RED, 'r': 10},
            {'text': 'Left choice', 'x': 0.45, 'y': 0.5, 'c': BLUE, 'r': -40},
            {'text': 'Control', 'x': 0.7, 'y': 0.5, 'c': 'grey', 'r': 0}]
    },
}

ADJUST_AVERAGE_3D_TRAJECTORIES = {
    'stim': {
        'view': {'elev': 15, 'azim': -15, 'roll': 13},
        'zoom': 0.7,
        'trans': {'shift_x': -3, 'shift_y': 0, 'shift_z': 0},
        'annotations': [
            {'text': 't = 0', 'x': 0.9, 'y': 1.1, 'c': 'k', 'r': 0},
            {'text': 'Left stimulus', 'x': 0.5, 'y': 0.4, 'c': BLUE, 'r': -5},
            {'text': 'Right stimulus', 'x': 0.5, 'y': 0.65, 'c': RED, 'r': -5},
            {'text': 'Control', 'x': 0.9, 'y': 0.6, 'c': 'grey', 'r': 0}
        ]
    },
    'fback': {
        'view': {'elev': 79, 'azim': -78, 'roll': -94},
        'zoom': 0.8,
        'trans': {'shift_x': 0, 'shift_y': 0, 'shift_z': 0},
        'origin': {'x':0.25, 'y': 0.1, 'l': 0.15},
        'annotations': [
            {'text': 't = 0', 'x': 1, 'y': 0.1, 'c': 'k', 'r': 0},
            {'text': 'Incorrect', 'x': 0.75, 'y': 0.93, 'c': RED, 'r': 0},
            {'text': 'Correct', 'x': 0.25, 'y': 0.93, 'c': BLUE, 'r': 0},
            {'text': 'Control', 'x': 0.55, 'y': 0.68, 'c': 'grey', 'r': 0}]
    },
    'choice': {
        'view': {'elev': 0, 'azim': -90, 'roll': -180},
        'zoom': 0.7,
        'trans': {'shift_x': -3, 'shift_y': 1, 'shift_z': 0},
        'annotations': [
            {'text': 't = 0', 'x': 1.05, 'y': 0.65, 'c': 'k', 'r': 0},
            {'text': 'Right choice', 'x': 0.4, 'y': 0.5, 'c': RED, 'r': 0},
            {'text': 'Left choice', 'x': 0.4, 'y': 0.75, 'c': BLUE, 'r': 0},
            {'text': 'Control', 'x': 0.65, 'y': 0.9, 'c': 'grey', 'r': 0}]
    },
}




def zoom_axes(ax, scale=0.5):
    """
    Zooms the 3D plot by scaling the axis limits relative to the center.

    Parameters:
      ax: The 3D axes to zoom.
      scale: A factor less than 1 will zoom in (reduce limits),
             greater than 1 will zoom out (expand limits).
    """
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    # Compute the center of the current limits
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    zmean = np.mean(zlim)

    # Compute half-ranges scaled by the zoom factor
    x_range = (xlim[1] - xlim[0]) * scale / 2
    y_range = (ylim[1] - ylim[0]) * scale / 2
    z_range = (zlim[1] - zlim[0]) * scale / 2

    # Set the new limits
    ax.set_xlim3d([xmean - x_range, xmean + x_range])
    ax.set_ylim3d([ymean - y_range, ymean + y_range])
    ax.set_zlim3d([zmean - z_range, zmean + z_range])


def shift_axes(ax, shift_x=0, shift_y=0, shift_z=0):
    """
    Zooms the 3D plot by scaling the axis limits relative to the center.

    Parameters:
      ax: The 3D axes to zoom.
      scale: A factor less than 1 will zoom in (reduce limits),
             greater than 1 will zoom out (expand limits).
    """
    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()

    # Set the new limits
    ax.set_xlim3d(x0 + shift_x, x1 + shift_x)
    ax.set_ylim3d(y0 + shift_y, y1 + shift_y)
    ax.set_zlim3d(z0 + shift_z, z1 + shift_z)



def plot_traj3d(variable, ga_pcs=False, curve='euc', ax_3d=None, ax_2d=None, save=True, save_path=None):
                       
    """
    Using full window (not canonical!!) to see lick oscillation
    and auditory response in IC at 0.5 sec after fback
    """

    variable, data = load_trajectory_for_variable(variable, averages=ga_pcs)

    can = 'can' in curve

    if ax_3d is None:
        tight = True
        fig = plt.figure(figsize=(3, 3))
        ax_3d = fig.add_subplot(projection='3d')
    else:
        tight = False
        fig = ax_3d.get_figure()
                              
    region = TRAJECTORY_EXAMPLE_REGIONS[variable][0]

    if not ga_pcs:
        data = data[region]

    n_pcs, all_nobs = data['pcs'].shape
    n_obs = all_nobs // ntravis

    for j in range(ntravis):

        # 3d trajectory
        cs = data['pcs'][:, n_obs * j: n_obs * (j + 1)].T
        n_obs_ = n_obs
        if can:
            # Restrict to canonical window
            n_obs_ = data[f'd_{curve}'].shape[0]
            cs = cs[:n_obs_]

        if j == 0:
            col = grad('Blues_r', n_obs_)
        elif j == 1:
            col = grad('Reds_r', n_obs_)
        else:
            col = grad('Greys_r', n_obs_)

        ax_3d.plot(cs[:, 0], cs[:, 1], cs[:, 2], color=col[len(col) // 2],
                linewidth=5 if j in [0, 1] else 1, alpha=0.5)

        ax_3d.scatter(cs[:, 0], cs[:, 1], cs[:, 2], color=col, edgecolors=col,
                   s=20 if j in [0, 1] else 1, depthshade=False)
              
    ax_3d.grid(False)
    ax_3d.axis('off')

    if ax_2d is not None:

        ADJUST = ADJUST_AVERAGE_3D_TRAJECTORIES if ga_pcs else ADJUST_3D_TRAJECTORIES


        ax_3d.view_init(**ADJUST[variable]['view'])
        zoom_axes(ax_3d, ADJUST[variable]['zoom'])
        shift_axes(ax_3d, **ADJUST[variable]['trans'])

        ax_2d.set_zorder(10)
        ax_2d.set_axis_off()

        if 'origin' in ADJUST[variable].keys():
            x_origin, y_origin, arrow_len = ADJUST[variable]['origin'].values()

            # Add the labels and other stuff onto the 2d axis
            ax_2d.set_zorder(10)
            ax_2d.set_axis_off()

            ax_2d.annotate('', xy=(x_origin, y_origin + arrow_len + 0.05), xytext=(x_origin, y_origin),
                        arrowprops=dict(arrowstyle='->', color='k', lw=0.5))
            ax_2d.text(x_origin, y_origin + arrow_len + 0.04, 'PCA 1', ha='center', va='bottom',
                    fontsize=f_size_s)
            # Left-leaning axis (dim 2)
            ax_2d.annotate('', xy=(x_origin - arrow_len, y_origin - 0.1), xytext=(x_origin, y_origin),
                        arrowprops=dict(arrowstyle='->', color='k', lw=0.5))
            ax_2d.text(x_origin - arrow_len - 0.05, y_origin - 0.1, 'PCA 2', ha='center', va='top',
                    fontsize=f_size_s)
            # Right-leaning axis (dim 3)
            ax_2d.annotate('', xy=(x_origin + arrow_len, y_origin - 0.1), xytext=(x_origin, y_origin),
                        arrowprops=dict(arrowstyle='->', color='k', lw=0.5))
            ax_2d.text(x_origin + arrow_len + 0.05, y_origin - 0.1, 'PCA 3', ha='center', va='top',
                      fontsize=f_size_s)

        for text in ADJUST[variable]['annotations']:
            ax_2d.text(text['x'], text['y'], text['text'], c=text['c'], rotation=text['r'], ha='center', va='top',
                    fontsize=f_size_s)

        # Add title
        ax_2d.set_title(f'{region} all sessions')

    
    if tight:
        ax_3d.set_title(f"{variable}, {region} {data['nclus']}")
        fig.tight_layout()

    if save:
        save_path = save_path or imgs_variable_path(variable)
        fig.savefig(save_path.joinpath(f'{variable}_manifold_3d.svg'))
        fig.savefig(save_path.joinpath(f'{variable}_manifold_3d.pdf'))


def plot_trajectory_lines(variable, curve='euc', ax=None, adjust=False, save=True, save_path=None):

    variable, data = load_trajectory_for_variable(variable)

    if ax is None:
        tight = True
        fig, ax = plt.subplots(figsize=(80 * MM_TO_INCH, 120 * MM_TO_INCH))
    else:
        tight = False
        fig = ax.get_figure()

    can = 'can' in curve

    regions = TRAJECTORY_EXAMPLE_REGIONS[variable]
    region_colors = [REGION_COLS[reg] for reg in regions]
    interval = pre_post(variable, can=can)
    time = np.linspace(-interval[0], interval[1], data[regions[0]][f'd_{curve}'].size)

    for reg, col in zip(regions, region_colors):
        ax.plot(time, data[reg][f'd_{curve}'], color=col)
        ax.text(time[-1], data[reg][f'd_{curve}'][-1], f" {reg} {data[reg]['nclus']}", color=col,
                fontsize=f_size_s)

    ax.axvline(x=0, linestyle='--', c='k')
    ax.text(0, EVENT_LINE_LOC[variable], f"  {EVENT_LINES[variable]}", color='k', fontsize=f_size_s)
    ax.set_ylabel('Distance (Hz)', fontsize=f_size)
    ax.set_xlabel(f'{EVENT_LABELS[variable]}', fontsize=f_size)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))

    if adjust:
        adjust_trajectory_lines(variable, ax)

    if tight:
        fig.tight_layout()

    if save:
        save_path = save_path or imgs_variable_path(variable)
        fig.savefig(save_path.joinpath(f'{variable}_manifold_lines.svg'))
        fig.savefig(save_path.joinpath(f'{variable}_manifold_lines.pdf'))


def plot_trajectory_scatter(variable, curve='euc', ax=None, adjust=False, show_sig=True, save=True, save_path=None):

    variable, data = load_trajectory_for_variable(variable)

    if ax is None:
        tight = True
        fig, ax = plt.subplots(figsize=(80 * MM_TO_INCH, 120 * MM_TO_INCH))
    else:
        tight = False
        fig = ax.get_figure()

    sig_regions = np.zeros((len(data)), dtype=bool)
    amplitudes = np.empty((len(data)))
    latencies = np.empty((len(data)))
    region_colors = np.empty((len(data), 4))

    # Find significant regions with finite values for max amps
    for i, reg in enumerate(data.keys()):
        if data[reg][f'p_{curve}'] < sigl and np.isfinite(data[reg][f'amp_{curve}']):
            sig_regions[i] = True

        amplitudes[i] = data[reg][f'amp_{curve}']
        latencies[i] = data[reg][f'lat_{curve}']
        region_colors[i] = REGION_COLS[reg]

    # Plot significant regions
    ax.scatter(latencies[sig_regions], amplitudes[sig_regions], color=region_colors[sig_regions, :],
                  marker='D', s=6)

    # Plot insignificant regions
    ax.scatter(latencies[~sig_regions], amplitudes[~sig_regions], color=region_colors[~sig_regions, :],
                   marker='o', s=1)

    for reg in TRAJECTORY_EXAMPLE_REGIONS[variable]:
        ax.text(data[reg][f'lat_{curve}'], data[reg][f'amp_{curve}'], reg, color=REGION_COLS[reg],
                    fontsize=f_size_s, va='center', ha='center')

    ax.set_ylabel('Max distance (Hz)', fontsize=f_size)
    ax.set_xlabel(f'Latency {EVENT_LABELS[variable][5:]}', fontsize=f_size)
    ax.axvline(x=0, linestyle='--', c='k')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))

    if show_sig:
        n_sig = sig_regions.sum()
        n_reg = sig_regions.size
        per_sig = np.round(n_sig/n_reg, decimals=2)
        ax.set_title(f'{n_sig}/{n_reg} = {per_sig} sig', fontsize=f_size_s)
        #ax.text(0.5, 1, f'{n_sig}/{n_reg} = {per_sig} sig', fontsize=f_size_s)

    if adjust:
        adjust_trajectory_scatters(variable, ax)

    if tight:
        fig.tight_layout()

    if save:
        save_path = save_path or imgs_variable_path(variable)
        fig.savefig(save_path.joinpath(f'{variable}_manifold_scatter.svg'))
        fig.savefig(save_path.joinpath(f'{variable}_manifold_scatter.pdf'))


def plot_trajectory_control(variable, region=None, curve='euc', ax=None, save=True, save_path=None):

    variable, data = load_trajectory_for_variable(variable)

    if ax is None:
        tight = True
        fig, ax = plt.subplots(figsize=(80 * MM_TO_INCH, 120 * MM_TO_INCH))
    else:
        tight = False
        fig = ax.get_figure()

    can = 'can' in curve

    region = region or TRAJECTORY_EXAMPLE_REGIONS[variable][0]
    region_color = REGION_COLS[region]
    example_data = data[region][f'd_{curve}']
    control_data = data[region][f'd_{curve}_p']
    interval = pre_post(variable, can=can)
    time = np.linspace(-interval[0], interval[1], example_data.size)
    ax.plot(time, control_data.T, color='Gray')
    ax.plot(time, example_data, color=region_color)
    ax.text(time[-1], example_data[-1], f' {region}', color=region_color, fontsize=f_size_s)
    ax.text(time[-1], control_data[-1][-1], ' Control', color='Gray', fontsize=f_size_s)
    ax.axvline(x=0, linestyle='--', c='k')
    ax.set_ylabel('Distance (Hz)', fontsize=f_size)
    ax.set_xlabel(f'{EVENT_LABELS[variable]}', fontsize=f_size)
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))

    if tight:
        fig.tight_layout()

    if save:
        save_path = save_path or imgs_variable_path(variable)
        fig.savefig(save_path.joinpath(f'{variable}_manifold_controls.pdf'))
        fig.savefig(save_path.joinpath(f'{variable}_manifold_controls.svg'))


def adjust_trajectory_lines(variable, ax):

    positions = ADJUST_TRAJECTORIES[variable]
    text_objects = [obj for obj in ax.get_children() if isinstance(obj, Text)]
    x_div = 10 if variable == 'fback' else 100

    for object in text_objects:
        # this is really hackey as we added a space in the oirignal text that we set
        text = object.get_text()
        if 'onset' in text:
            continue
        text = text[1:].split(' ')[0]
        if text in br.acronym:
            x0, y0 = object.get_position()
            x = x0 + positions[text]['x_l'] / x_div
            y = y0 + positions[text]['y_l'] / 10
            object.set_position((x, y))


def adjust_trajectory_scatters(variable, ax):

    positions = ADJUST_TRAJECTORIES[variable]
    text_objects = [obj for obj in ax.get_children() if isinstance(obj, Text)]
    x_div = 10 if variable == 'fback' else 100

    for object in text_objects:
        text = object.get_text()
        if 'onset' in text:
            continue
        # this is really hacky as we added a space in the orignal text that we set
        if text in br.acronym:
            x0, y0 = object.get_position()
            x = x0 + positions[text]['x_s'] / x_div
            y = y0 + positions[text]['y_s'] / 10
            ha = positions[text].get('ha_s', None)
            if ha is None:
                ha = 'right' if x < x0 else 'left'
            va = 'bottom' if ha == 'center' else 'center'
            object.set_position((x, y))
            ax.annotate('', xy=(x0, y0), xytext=(x, y),arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
            object.set_horizontalalignment(ha)
            object.set_verticalalignment(va)


def plot_curves_scatter(variable, ga_pcs=False, region=None, curve='euc', axs=None, save=True):
    """"
    For a given region, plot example line with control,
    more example lines without control, scatter amps
    """

    if axs is None:
        tight = True
        fig, axs = plt.subplots(2, 2, figsize=(150 * MM_TO_INCH, 100 * MM_TO_INCH),
                                gridspec_kw={'height_ratios': [1, 2], 'wspace':0.5}, sharex=True)
        axs = axs.flatten()
        axs[1].axis('off')
    else:
        tight = False
        fig = axs[0].get_figure()

    tops = {}
    regsa = []

    variable, data = load_trajectory_for_variable(variable, averages=ga_pcs)

    maxs = np.array([data[x][f'amp_{curve}'] for x in data])
    acronyms = np.array(list(data.keys()))
    order = list(reversed(np.argsort(maxs)))
    maxs = maxs[order]
    acronyms = acronyms[order]

    tops[variable] = [acronyms,[data[reg][f'p_{curve}'] for reg in acronyms], maxs]

    maxs = np.array([data[reg][f'amp_{curve}'] for reg in acronyms if data[reg][f'p_{curve}'] < sigl])
    maxsf = [v for v in maxs if not (np.isinf(v) or np.isnan(v))]

    print(variable, curve)
    print(f'{len(maxsf)}/{len(data)} are significant')
    tops[variable + '_s'] = (f'{len(maxsf)}/{len(data)} = 'f'{np.round(len(maxsf)/len(data),2)}')

    regs_a = [tops[variable][0][j] for j in range(len(tops[variable][0])) if tops[variable][1][j] < sigl]

    regsa.append(list(data.keys()))
    print(regs_a)
    print(' ')
    print(variable, tops[variable + '_s'])

    reg = TRAJECTORY_EXAMPLE_REGIONS[variable][0] if not region else region

    if 'can' in curve:
        print('Using canonical time windows')
        can = True
    else:
        can = False          

    # Line plot for example region
    if reg != 'all':             
        if reg not in data:
            print(f'{reg} not in d: revise example regions for line plots')
            return

    if any(np.isinf(data[reg][f'd_{curve}'].flatten())):
        print(f'inf in {curve} of {reg}')
        return
        
    print(variable, reg, 'p_euc_can: ', data[reg]['p_euc_can'])

    plot_trajectory_control(variable, curve=curve, region=reg, ax=axs[0], save=False)

    # Line plot per 5 example regions per variable
    regs = TRAJECTORY_EXAMPLE_REGIONS[variable]

    for reg in regs:
        if reg not in data:
            print(f'{reg} not in d: revise example regions for line plots')
            return
    
        if any(np.isinf(data[reg][f'd_{curve}'])):
            print(f'inf in {curve} of {reg}')
            return

    plot_trajectory_lines(variable, curve=curve, ax=axs[2], save=False)

    # Scatter latency versus max amplitude for significant regions
    plot_trajectory_scatter(variable, curve=curve, ax=axs[3], save=False)

    if tight:
        fig.tight_layout()

    if save:
        fig.savefig(Path(imgs_pth, variable, 'manifold_lines_scatter.svg'))


def plot_trajectories_with_psd(variable='fback', curve='euc', axs=None):
    '''
    distance line plots for select regions;
    Welch psd as insets for each

    Supp figure about contrast dependence:
    plot_custom_lines(regs = ['APN', 'IRN','GRN','ACAv',
                              'PRNr','LP','AUDp','PO','ILA'],
                              split='choice',contr_=True)

    Supp figure on oscillations,
    plot_custom_lines(regs = ['MV', 'MRN', 'APN', 'SSp-m',
                              'SIM', 'PRM', 'PoT',
                              'MEA', 'ANcr2'],
                              split='fback',psd_=True, contr_=False)

    '''
    from scipy import signal
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    regions = ['SSp-n', 'VISal', 'GU', 'SSp-m', 'SIM', 'IRN', 'NTS', 'COAp', 'ENTl']

    if axs is None:
        fig, axs = plt.subplots(nrows=3, ncols=len(regions)//3, figsize=(110 * MM_TO_INCH, 120*MM_TO_INCH ),
                                sharex=True, constrained_layout=True)

    variable, data = load_trajectory_for_variable('fback')
    axsi = []
    axs = axs.flatten()

    for i, reg in enumerate(regions):
        col = np.mod(i, 3)
        ax = axs[i]
        set_max_ticks(ax)
        values = data[reg][f'd_{curve}']
        time = np.linspace(-pre_post(variable)[0], pre_post(variable)[1], len(data[reg][f'd_{curve}']))

        ax.plot(time, values, color=REGION_COLS[reg], label=f"{reg}")

        f, psd = signal.welch(values, fs=int(len(time) / (time[-1] - time[0])))

        ins = ax.inset_axes([1.1, 0, 0.4, 1], transform=ax.transAxes)
        ins.plot(f, psd, color='k')
        ins.set_xlim(3, 40)

        ins.set_yticklabels([])

        title = f" {reg} {data[reg]['nclus']}"
        if col == 0:
            ax.set_ylabel('Distance (Hz)')
            ins.set_ylabel('PSD (dB)', fontsize=f_size_s, labelpad=-2)
        if i >= 6:
            ax.set_xlabel(EVENT_LABELS[variable])
            ins.set_xlabel('Freq (Hz)', fontsize=f_size)
        else:
            ax.set_xticklabels([])
            ins.set_xticklabels([])

        ax.set_xlim(-0.1, ax.get_xlim()[1])

        if reg in ['COAp', 'GU']:
            ax.set_ylim(0, 2)
        if reg == 'NTS':
            ax.set_ylim(0, 2.5)

        ax.axvline(x=0, linestyle='--', c='k')

        if i == 7:
            ax.text(0, ax.get_ylim()[1] - 0.6, f"  {EVENT_LINES[variable]}", color='k', fontsize=f_size_s)

        ax.text(0.5, 1, title, color=REGION_COLS[reg], fontsize=f_size_s, ha='center', va='bottom', transform=ax.transAxes)


def plot_trajectories_with_licks(variable='fback_restr', curve='euc', axs=None):
    """
    Distance line plots for select regions;
    Welch psd as insets for each
    """

    from scipy import signal
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    regions = {'MRN': 'f8d5c8b0-b931-4151-b86c-c471e2e80e5d',
               'APN': 'f8d5c8b0-b931-4151-b86c-c471e2e80e5d',  # (multi session)
               'IRN': 'f8d5c8b0-b931-4151-b86c-c471e2e80e5d',
               'SUV': 'c4432264-e1ae-446f-8a07-6280abade813',
               'ANcr2': '83d85891-bd75-4557-91b4-1cbb5f8bfc9d'}


    if axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=len(regions), figsize=(110 * MM_TO_INCH, 40*MM_TO_INCH ),
                                constrained_layout=True)

    variable, data = load_trajectory_for_variable(variable)
    licks = np.load(Path(one.cache_dir,'bwm_res', 'bwm_figs_data', 'trajectory', 'fback_eids_only_lickograms.npy'),
                    allow_pickle=True).flat[0]

    for col, reg in enumerate(regions):
        ax = axs[col]
        set_max_ticks(ax)

        values = data[reg][f'd_{curve}']
        time = np.linspace(-pre_post(variable)[0], pre_post(variable)[1], len(data[reg][f'd_{curve}']))

        ax.plot(time, values, color=REGION_COLS[reg], label=f"{reg}")

        title = f" {reg} {data[reg]['nclus']}"
        if col == 0:
            ax.set_ylabel('Distance (Hz)')

        ax.set_xlabel(EVENT_LABELS[variable])
        ax.set_xlim(-0.1, ax.get_xlim()[1])

        ax.axvline(x=0, linestyle='--', c='k')
        ax.spines['right'].set_visible(True)

        ax.set_title(title, color=REGION_COLS[reg], fontsize=f_size_s, va='top')

        if regions[reg] in licks:
            values = licks[regions[reg]]
            time = np.linspace(-pre_post(variable)[0],pre_post(variable)[1], len(values))

            ax_x2 = ax.twinx()
            set_max_ticks(ax_x2)
            ax_x2.plot(time, values)
            if col == len(regions) - 1:
                ax_x2.set_ylabel('Trial-averaged licks (Hz)')

def plot_lick_raster_psth(axs=None):

    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(4, 6),
                                gridspec_kw={'height_ratios': [1, 3], 'hspace': 0}, sharex=True)
    from brainbox.task.trials import find_trial_ids
    eid = 'fc43390d-457e-463a-9fd4-b94a0a8b48f5'
    trials = one.load_object(eid, 'trials')
    trials = trials.to_df()
    licks = one.load_dataset(eid, 'licks.times.npy')
    trial_idx, dividers = find_trial_ids(trials, sort='choice')

    single_cluster_raster(licks, trials['feedback_times'], trial_idx, dividers, [BLUE, RED], ['Correct', 'Incorrect'],
                          axs=axs, fr=True, show_psth=True)

    axs[0].set_xlim(axs[1].get_xlim())
    axs[0].set_xticklabels([])
    axs[0].set_ylabel('Licks (Count)')
    axs[1].set_ylabel('Trials', labelpad=-2)
    axs[1].set_xlabel(EVENT_LABELS[variable])




'''
###########
combine all panels
###########
'''


def put_panel_label(ax, label):
    #string.ascii_lowercase[k]
    ax.annotate(label, (-0.05, 1.15),
                xycoords='axes fraction',
                fontsize=f_size, va='top',
                ha='right', weight='bold')    

def shift_plot(ax, x, y):
    pos = ax.get_position()
    pos.x0 = pos.x0 + x
    pos.x1 = pos.x1 + x
    pos.y0 = pos.y0 + y
    pos.y1 = pos.y1 + y
    ax.set_position(pos)


def main_fig_for_variable(variable, individual_pans=False, save=True, save_path=None):

    """
    Combine plots into main figures for variables
    :param variable:
    :param individual_pans:
    :return:
    """

    if not individual_pans:

        fig_name = {
            'choice': 'n5_main_fig5_choice',
            'stim': 'n5_main_fig4_stimulus',
            'fback': 'n5_main_fig6_feedback'
        }

        fig = plt.figure(figsize=(183 * MM_TO_INCH, 170 * MM_TO_INCH))
        width, height = fig.get_size_inches() / MM_TO_INCH
        xspans1 = get_coords(width, ratios=[1, 1, 1, 1, 1], space=0, pad=0, span=(0, 1))
        xspans2 = get_coords(width, ratios=[1], space=15, pad=15, span=(0, 1))
        xspans3 = get_coords(width, ratios=[1, 1, 1], space=[25, 25], pad=10, span=(0, 1))

        yspans = get_coords(height, ratios=[8, 4, 4, 4], space=[4, 12, 8], pad=2, span=(0, 1))
        yspans_sub1 = get_coords(height, ratios=[1, 1], space=6, pad=0, span=yspans[1])
        yspans_sub2 = get_coords(height, ratios=[1, 1], space=2, pad=0, span=yspans[2])
        yspans_sub3 = get_coords(height, ratios=[2, 3], space=12, pad=0, span=[yspans[2][0], yspans[3][1]])

        axs = {'a': fg.place_axes_on_grid(fig, xspan=xspans1[0], yspan=yspans[0]),
               'b': fg.place_axes_on_grid(fig, xspan=xspans1[1], yspan=yspans[0]),
               'c': fg.place_axes_on_grid(fig, xspan=xspans1[2], yspan=yspans[0]),
               'd': fg.place_axes_on_grid(fig, xspan=xspans1[3], yspan=yspans[0]),
               'e': fg.place_axes_on_grid(fig, xspan=xspans1[4], yspan=yspans[0]),
               'f_1': fg.place_axes_on_grid(fig, xspan=xspans2[0], yspan=yspans_sub1[0]),
               'f_2': fg.place_axes_on_grid(fig, xspan=xspans2[0], yspan=yspans_sub1[1]),
               'g': fg.place_axes_on_grid(fig, xspan=xspans3[0], yspan=yspans_sub3[0]),
               'h': fg.place_axes_on_grid(fig, xspan=xspans3[0], yspan=yspans_sub3[1], dim=(2, 1), hspace=0.75),
               'i': fg.place_axes_on_grid(fig, xspan=xspans3[1], yspan=yspans_sub2[0]),
               'j': fg.place_axes_on_grid(fig, xspan=[xspans3[2][0] - 0.08, xspans3[2][1]], yspan=yspans[2]),
               'k': fg.place_axes_on_grid(fig, xspan=xspans3[1], yspan=yspans_sub2[1] + 0.03),
               'l': fg.place_axes_on_grid(fig, xspan=xspans3[1], yspan=yspans[3]),
               'm': fg.place_axes_on_grid(fig, xspan=xspans3[2], yspan=yspans[3]),
               }

        gs_3d = fig.add_gridspec(4, 3, wspace=0, hspace=0)
        ax_3d = fig.add_subplot(gs_3d[2, 2], projection='3d')

        labels = []
        padx = 10
        pady = 5
        labels.append(add_label('a', fig, xspans1[0], yspans[0], 0, 8, fontsize=8))
        labels.append(add_label('b', fig, xspans1[1], yspans[0], 0, pady, fontsize=8))
        labels.append(add_label('c', fig, xspans1[2], yspans[0], 0, pady, fontsize=8))
        labels.append(add_label('d', fig, xspans1[3], yspans[0], 0, pady, fontsize=8))
        labels.append(add_label('e', fig, xspans1[4], yspans[0], 0, pady, fontsize=8))
        labels.append(add_label('f', fig, xspans2[0], yspans[1], 15, 1, fontsize=8))
        labels.append(add_label('g', fig, xspans3[0], yspans[2], padx, pady, fontsize=8))
        labels.append(add_label('h', fig, xspans3[0], yspans_sub3[1], padx, pady, fontsize=8))
        labels.append(add_label('i', fig, xspans3[1], yspans_sub2[0], padx, pady, fontsize=8))
        labels.append(add_label('j', fig, xspans3[2], yspans[2], padx, pady, fontsize=8))
        labels.append(add_label('k', fig, xspans3[1], yspans_sub2[1], padx, -2, fontsize=8))
        labels.append(add_label('l', fig, xspans3[1], yspans[3], padx, 0, fontsize=8))
        labels.append(add_label('m', fig, xspans3[2], yspans[3], padx, 0, fontsize=8))

        fg.add_labels(fig, labels)
        fig.subplots_adjust(top=0.99, bottom=0, left=0.02, right=0.98)

        plot_swansons_for_variable(variable, axs=[axs['a'], axs['b'], axs['c'], axs['d'], axs['e']], adjust=True,
                                   save=False)
        leg = axs['a'].legend_
        leg.set_bbox_to_anchor((0.7, 0.13))
        plot_table(variable, axs=[axs['f_1'], axs['f_2']], save=False)
        plot_trajectory_control(variable + '_restr', ax=axs['k'], save=False)
        axs['k'].set_xlabel('')
        plot_trajectory_lines(variable + '_restr', ax=axs['l'], adjust=True, save=False)
        plot_trajectory_scatter(variable + '_restr', ax=axs['m'], adjust=True, show_sig=False, save=False)
        axs['l'].sharex(axs['k'])

        if variable == 'fback':
            axs['l'].set_ylim(0, 4.2)
            axs['m'].set_ylim(0, 4.2)

        for text in axs['l'].texts + axs['m'].texts + axs['k'].texts:
            if 'onset' not in text.get_text():
                text.set_fontsize(f_size_xs)
            else:
                if variable == 'choice':
                    pos = text.get_position()
                    text.set_position((pos[0] - 0.078, pos[1]))

        plot_traj3d(variable + '_restr', ax_3d=ax_3d, ax_2d=axs['j'], save=False)
        if variable == 'fback':
            shift_plot(ax_3d, 0.03, -0.06)
        elif variable == 'choice':
            shift_plot(ax_3d, 0.02, -0.04)
        else:
            shift_plot(ax_3d, 0, -0.04)

        axs['f_2'].set_zorder(50)
        if variable == 'stim':
            plot_decoding_line(ax=axs['i'], save=False)
            axs['i'].set_xlabel('Stimulus contrast', fontsize=f_size, va='bottom')
        else:
            plot_decoding_scatter(variable, ax=axs['i'], save=False)
            axs['i'].set_zorder(50)


        plot_encoding_for_variable(variable, axs=[axs['g'], axs['h'][0], axs['h'][1]], save=False)
        axs['h'][0].set_xticklabels([])
        axs['h'][0].set_ylabel('')
        ylabel = axs['h'][1].yaxis.get_label()
        if variable != 'fback':
            axs['g'].set_ylabel('Trials', labelpad=-1)
            ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1] + 0.9))
        else:
            axs['h'][1].set_ylabel(ylabel.get_text(), labelpad=-0.25)
            ylabel = axs['h'][1].yaxis.get_label()
            ylabel.set_position((ylabel.get_position()[0], ylabel.get_position()[1] + 0.9))

        for coll in axs['g'].collections:
            coll.set_linewidth(5)
        leg = axs['h'][1].legend_
        if variable == 'choice':
            leg.set_bbox_to_anchor((1.36, 1.9))
        else:
            leg.set_bbox_to_anchor((1.42, 1.85))

        if save:
            save_path = save_path or imgs_pth
            save_path = Path(f'/Users/admin/bwm/main')
            save_path.mkdir(exist_ok=True, parents=True)
            save_name = fig_name[variable]
            fig.savefig(save_path.joinpath(f'{save_name}.pdf'))
            fig.savefig(save_path.joinpath(f'{save_name}.eps'), dpi=150)

    else:
        save_path = Path(f'/Users/admin/int-brain-lab/test_panels/{variable}')
        save_path.mkdir(exist_ok=True, parents=True)
        plot_swansons_for_variable(variable, adjust=True, save=save, save_path=save_path)
        plot_table(variable, save=save, save_path=save_path)
        plot_trajectory_control(variable, save=save, save_path=save_path)
        plot_trajectory_lines(variable, adjust=True, save=save, save_path=save_path)
        plot_trajectory_scatter(variable, adjust=True, save=save, save_path=save_path)
        plot_traj3d(variable, save=save, save_path=save_path)
        if variable == 'stim':
            plot_decoding_line(save=save, save_path=save_path)
        else:
            plot_decoding_scatter(variable, save=save, save_path=save_path)
        plot_encoding_for_variable(variable, save=save, save_path=save_path)

def main_fig_for_wheel(individual_pans=False, save=True, save_path=None):

    if not individual_pans:

        fig = plt.figure(figsize=(183 * MM_TO_INCH, 170 * MM_TO_INCH))
        width, height = fig.get_size_inches() / MM_TO_INCH
        xspans1 = get_coords(width, ratios=[1, 1, 1, 1], space=0, pad=0, span=(0, 1))
        xspans2 = get_coords(width, ratios=[1], space=15, pad=16, span=(0, 1))
        xspans3 = get_coords(width, ratios=[1, 1], space=[25], pad=5, span=(0, 1))
        xspans_sub1 = get_coords(width, ratios=[1, 1], space=10, pad=10, span=xspans3[0])

        yspans = get_coords(height, ratios=[5, 3, 4], space=[5, 13], pad=2, span=(0, 1))
        yspans_sub1 = get_coords(height, ratios=[1, 1, 1, 1], space=[1, 6, 1], pad=0, span=yspans[1])
        yspans_sub2 = get_coords(height, ratios=[1, 1], space=8, pad=0, span=yspans[2])

        axs = {'a': fg.place_axes_on_grid(fig, xspan=xspans1[0], yspan=yspans[0]),
               'b': fg.place_axes_on_grid(fig, xspan=xspans1[1], yspan=yspans[0]),
               'c': fg.place_axes_on_grid(fig, xspan=xspans1[2], yspan=yspans[0]),
               'd': fg.place_axes_on_grid(fig, xspan=xspans1[3], yspan=yspans[0]),
               'e_1': fg.place_axes_on_grid(fig, xspan=xspans2[0], yspan=yspans_sub1[0]),
               'e_2': fg.place_axes_on_grid(fig, xspan=xspans2[0], yspan=yspans_sub1[1]),
               'e_3': fg.place_axes_on_grid(fig, xspan=xspans2[0], yspan=yspans_sub1[2]),
               'e_4': fg.place_axes_on_grid(fig, xspan=xspans2[0], yspan=yspans_sub1[3]),
               'f_1': fg.place_axes_on_grid(fig, xspan=xspans_sub1[0], yspan=yspans_sub2[0]),
               'f_2': fg.place_axes_on_grid(fig, xspan=xspans_sub1[1], yspan=yspans_sub2[0]),
               'g_1': fg.place_axes_on_grid(fig, xspan=xspans_sub1[0], yspan=yspans_sub2[1]),
               'g_2': fg.place_axes_on_grid(fig, xspan=xspans_sub1[1], yspan=yspans_sub2[1]),
               'h': fg.place_axes_on_grid(fig, xspan=xspans3[1], yspan=yspans[2]),
               }

        labels = []
        padx = 10
        pady = 3
        labels.append(add_label('a', fig, xspans1[0], yspans[0], 0, 10, fontsize=8))
        labels.append(add_label('b', fig, xspans1[1], yspans[0], 0, 10, fontsize=8))
        labels.append(add_label('c', fig, xspans1[2], yspans[0], 0, 10, fontsize=8))
        labels.append(add_label('d', fig, xspans1[3], yspans[0], 0, 10, fontsize=8))
        labels.append(add_label('e', fig, xspans1[0], yspans[1], 0, 1, fontsize=8))
        labels.append(add_label('f', fig, xspans3[0], yspans[2], padx, pady, fontsize=8))
        labels.append(add_label('g', fig, xspans3[0], yspans_sub2[1], padx, pady, fontsize=8))
        labels.append(add_label('h', fig, xspans3[1], yspans[2], 15, pady, fontsize=8))
        fg.add_labels(fig, labels)
        plot_swansons_for_wheel(axs=[axs['a'], axs['b'], axs['c'], axs['d']], adjust=True, save=False)
        axs['a'].text(1.4, -0.25, 'Wheel-speed', fontsize=f_size_l, transform=axs['a'].transAxes, ha='center',
                      va='center')
        leg = axs['a'].legend_
        leg.set_bbox_to_anchor((0.7, 0.13))
        axs['c'].text(1.4, -0.25, 'Wheel-velocity', fontsize=f_size_l, transform=axs['c'].transAxes, ha='center',
                      va='center')
        plot_table('speed', axs=[axs['e_1'], axs['e_3']], save=False)
        axs['e_1'].set_xticklabels([])
        axs['e_3'].set_xticklabels([])
        plot_table('velocity', axs=[axs['e_2'], axs['e_4']], save=False)
        x = -0.08
        axs['e_1'].text(x, 0.6, 'Speed', fontsize=f_size_s, transform=axs['e_1'].transAxes, rotation=90, ha='center',
                        va='center')
        axs['e_2'].text(x, 0.4, 'Velocity', fontsize=f_size_s, transform=axs['e_2'].transAxes, rotation=90, ha='center',
                        va='center')
        axs['e_3'].text(x, 0.6, 'Speed', fontsize=f_size_s, transform=axs['e_3'].transAxes, rotation=90, ha='center',
                        va='center')
        axs['e_4'].text(x, 0.4, 'Velocity', fontsize=f_size_s, transform=axs['e_4'].transAxes, rotation=90, ha='center',
                        va='center')

        # plot_table('wheel', save=False)
        wheel_decoding_ex('speed', axs=[axs['f_1'], axs['f_2']], save=False)
        axs['f_1'].set_xlabel('')
        axs['f_2'].set_xlabel('')
        wheel_decoding_ex('velocity', axs=[axs['g_1'], axs['g_2']], save=False)
        axs['g_1'].set_title('')
        axs['g_2'].set_title('')
        leg = axs['g_1'].legend_
        leg.set_visible(False)
        for text in axs['g_2'].texts:
            text.remove()
        encoding_wheel_boxen(ax=axs['h'], save=False)
        fig.subplots_adjust(top=0.99, bottom=0.01, left=0.02, right=0.98)

        if save:
            save_path = save_path or imgs_pth
            save_path = Path(f'/Users/admin/bwm/main')
            save_path.mkdir(exist_ok=True, parents=True)
            fig.savefig(save_path.joinpath(f'n5_main_fig7_wheel.pdf'))
            fig.savefig(save_path.joinpath(f'n5_main_fig7_wheel.eps'), dpi=200)


def plot_supp_figures(save_path=None):
    """
    Supplementary figures for BWM paper
    """
    supp_save_path = Path('/Users/admin/bwm/supp')
    supp_save_path.mkdir(exist_ok=True, parents=True)

    # Figure 3: n6_supp_fig3_sampling_manifold
    fig = plt.figure(figsize=(183 * MM_TO_INCH, 100 * MM_TO_INCH))
    width, height = fig.get_size_inches() / MM_TO_INCH
    xspans = get_coords(width, ratios=[1], space=15, pad=15, span=(0, 1))
    yspans = get_coords(height, ratios=[1, 1], space=10, pad=10, span=(0, 1))

    axs = {'a': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0], dim=(1, 3), wspace=0.3),
           'b': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1], dim=(1, 3), wspace=0.3),}

    plot_strip_sampling(sampletype='neurons', norm='double', axs=axs['a'])
    plot_strip_sampling(sampletype='eids', norm='double', axs=axs['b'])
    for ax in [axs['a'][-1], axs['b'][-1]]:
        leg = ax.legend_
        leg.set_frame_on(False)
        leg.get_title().set_fontsize(f_size_s)
        for t in leg.get_texts():
            t.set_fontsize(f_size_s)

        leg.set_bbox_to_anchor((1.50, 0.25))

    for ax in axs['b']:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title('')

    for ax in axs['a']:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        title = ax.get_title()
        ax.set_title(title, fontsize=f_size_l, va='bottom')

    adjust_subplots(fig, adjust=[0, 10, 2, 20], extra=0)
    save_name = 'n6_supp_fig3_sampling_manifold'
    fig.savefig(Path(supp_save_path, f'{save_name}.pdf'))
    fig.savefig(Path(supp_save_path, f'{save_name}.eps'), dpi=150)

    # Figure 4: n6_supp_fig4_dec_repro_male_female
    fig, axs = plt.subplots(3, 1, figsize=([183 * MM_TO_INCH, 90 * MM_TO_INCH]))
    plot_female_male_repro(axs=axs, save=False)
    adjust_subplots(fig, adjust=[3, 10, 17, 5], extra=0)
    save_name = 'n6_supp_fig4_dec_repro_male_female'
    fig.savefig(Path(supp_save_path, f'{save_name}.pdf'))
    fig.savefig(Path(supp_save_path, f'{save_name}.eps'), dpi=150)

    # Figure 6: n6_supp_fig6_feedback_baseline
    fig = plt.figure(figsize=(183 * MM_TO_INCH, 60 * MM_TO_INCH))
    width, height = fig.get_size_inches() / MM_TO_INCH
    xspans = get_coords(width, ratios=[1, 1], space=5, pad=0, span=(0, 1))
    yspans = get_coords(height, ratios=[1], space=0, pad=5, span=(0, 1))
    axs = {'a': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0]),
           'b': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[0]), }
    _, cax1 = plot_swanson_for_single_cell_variable('correct_vs_baseline', ax=axs['a'])
    axs['a'].set_title('')
    axs['a'].text(0.1, 1.05, 'Feedback (correct) vs baseline', transform=axs['a'].transAxes)
    _, cax2 = plot_swanson_for_single_cell_variable('incorrect_vs_baseline', ax=axs['b'],
                                                    cbar=True, legend=False)
    axs['b'].set_title('')
    axs['b'].text(0.1, 1.05, 'Feedback (incorrect) vs baseline', transform=axs['b'].transAxes)
    cax2.ax.set_visible(False)
    orig_bbox = cax1.ax.get_position()

    axs['b'].text(0.5, -0.15, 'Single-cell statistics', transform=axs['b'].transAxes, ha='center', fontsize=f_size_l)
    axs['b'].text(0.5, -0.25, 'Condition combined Mann-Whitney test', transform=axs['b'].transAxes, ha='center', fontsize=f_size_s)
    labels = []
    padx = 0
    pady = 10
    labels.append(add_label('a', fig, xspans[0], yspans[0], padx, pady, fontsize=8))
    labels.append(add_label('b', fig, xspans[1], yspans[0], padx, pady, fontsize=8))
    fg.add_labels(fig, labels)

    adjust_subplots(fig, adjust=[3, 7, 3, 3], extra=0)

    cax1.ax.set_position([orig_bbox.x0 + 0.33, orig_bbox.y0 + 0.01 , orig_bbox.width, orig_bbox.height])

    leg = axs['a'].legend_
    leg.set_bbox_to_anchor((0.7, -0.05))

    save_name = 'n6_supp_fig6_feedback_baseline'
    fig.savefig(Path(supp_save_path, f'{save_name}.pdf'))
    fig.savefig(Path(supp_save_path, f'{save_name}.eps'), dpi=150)

    # Figure 7: n6_supp_fig7_decoding_wheelspeedvsvel
    fig = plt.figure(figsize=(183 * MM_TO_INCH, 60 * MM_TO_INCH))
    width, height = fig.get_size_inches() / MM_TO_INCH
    xspans = get_coords(width, ratios=[1, 1, 1, 1], space=20, pad=15, span=(0, 1))
    yspans = get_coords(height, ratios=[1], space=10, pad=5, span=(0, 1))

    axs = {'a': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0]),
           'b': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[0]),
           'c': fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[0]),
           'd': fg.place_axes_on_grid(fig, xspan=xspans[3], yspan=yspans[0]), }

    labels = []
    padx = 10
    pady = 0
    labels.append(add_label('a', fig, xspans[0], yspans[0], padx, pady, fontsize=8))
    labels.append(add_label('b', fig, xspans[1], yspans[0], padx, pady, fontsize=8))
    labels.append(add_label('c', fig, xspans[2], yspans[0], padx + 2, pady, fontsize=8))
    labels.append(add_label('d', fig, xspans[3], yspans[0], padx, pady, fontsize=8))
    fg.add_labels(fig, labels)

    plot_decoding_speed_velocity(axs=[axs['a'], axs['b'], axs['c'], axs['d']], save=False)
    adjust_subplots(fig, adjust=[0, 11, 0, 8], extra=0)
    leg = axs['d'].legend_
    leg.set_bbox_to_anchor((0.5, 0.2))
    save_name = 'n6_supp_fig7_decoding_wheelspeedvsvel'
    fig.savefig(Path(supp_save_path, f'{save_name}.pdf'))
    fig.savefig(Path(supp_save_path, f'{save_name}.eps'), dpi=150)

    # Figure 8: n6_supp_fig8_learning_stats
    fig, axs = plt.subplots(1, 2, figsize=([183 * MM_TO_INCH, 80 * MM_TO_INCH]))
    perf_scatter(axs=axs, save=False)
    adjust_subplots(fig, adjust=[3, 10, 12, 25], extra=0)
    save_name = 'n6_supp_fig8_learning_stats'
    fig.savefig(Path(supp_save_path, f'{save_name}.pdf'))
    fig.savefig(Path(supp_save_path, f'{save_name}.eps'), dpi=150)



def plot_eds_figures():
    ed_save_path = Path('/Users/admin/bwm/ed')
    ed_save_path.mkdir(exist_ok=True, parents=True)

    # Figure 2: n6_ed_fig2_all_variables
    fig, axs = plt.subplots(3, 4, figsize=([130 * MM_TO_INCH, 170 * MM_TO_INCH]),
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    plot_swansons_across_analysis(axs, save=False)
    adjust_subplots(fig, adjust=[8, 5, 1, 1], extra=0)
    save_name = 'n6_ed_fig2_all_variables'
    fig.savefig(Path(ed_save_path, f'{save_name}.pdf'))
    fig.savefig(Path(ed_save_path, f'{save_name}.eps'), dpi=150)

    # Figure 3: n6_ed_fig3_analyses_amp_pairs_grid
    fig, axs = plt.subplots(6, 3, figsize=(130 * MM_TO_INCH, 170 * MM_TO_INCH),
                            gridspec_kw={'hspace': 0.9, 'wspace': 0.5})
    plot_scatter_analysis_pairs(axs=axs, save=False)
    adjust_subplots(fig, adjust=[6, 10, 13, 2], extra=0)
    save_name = 'n6_ed_fig3_analyses_amp_pairs_grid'
    fig.savefig(Path(ed_save_path, f'{save_name}.pdf'))
    fig.savefig(Path(ed_save_path, f'{save_name}.eps'), dpi=150)

    # Figure 6, 11, 14
    annotate_list = {
        'choice': ['VISal', 'SAG', 'LDT', 'IF', 'RL', 'ISN', 'LIN', 'GR'],
        'fback': ['BST', 'PVH', 'CS', 'V', 'LDT', 'NI'],
        'stim': ['NB', 'TRN', 'PRNc']
    }
    names = ['n6_ed_fig11_choice', 'n6_ed_fig14_feedback', 'n6_ed_fig6_stimulus']
    for variable, save_name in zip(annotate_list.keys(), names):
        fig = plt.figure(figsize=(183 * MM_TO_INCH, 170 * MM_TO_INCH))
        width, height = fig.get_size_inches() / MM_TO_INCH

        xspans1 = get_coords(width, ratios=[5, 1], space=15, pad=5, span=(0, 1))
        xspans2 = get_coords(width, ratios=[1], space=15, pad=5, span=(0, 1))
        yspans = get_coords(height, ratios=[6, 6, 4], space=[5, 0], pad=5, span=(0, 1))

        axs = {'a': fg.place_axes_on_grid(fig, xspan=xspans1[0], yspan=yspans[0]),
               'b': fg.place_axes_on_grid(fig, xspan=xspans2[0], yspan=yspans[1], dim=(3, 5), hspace=0) +
                    [fg.place_axes_on_grid(fig, xspan=xspans2[0], yspan=yspans[2], dim=(1, 5))]}

        labels = []
        padx = 1
        labels.append(add_label('a', fig, xspans2[0], yspans[0], padx, 5, fontsize=8))
        labels.append(add_label('b', fig, xspans2[0], yspans[1], padx, 10, fontsize=8))
        fg.add_labels(fig, labels)
        _, cax = plot_swansons_decoding(variable, ax=axs['a'], adjust=True, save=False, annotate_list=annotate_list[variable])
        plot_slices_for_variable(variable, axs=axs['b'], save=False)
        orig_bbox = cax.ax.get_position()
        cax.ax.set_position((orig_bbox.x0 + 0.5, orig_bbox.y0 + 0.14, orig_bbox.width, orig_bbox.height))
        orig_bbox = cax.ax.get_position()
        leg = axs['a'].legend_
        leg.set_bbox_to_anchor((1.16, 0.35))
        adjust_subplots(fig, adjust=[3, 8, 0, 5], extra=0)
        cax.ax.set_position([orig_bbox.x0 + 0.05, orig_bbox.y0 + 0.1, orig_bbox.width, orig_bbox.height])
        axs['a'].set_title('')
        axs['a'].text(-0.07, 1, 'Decoding', transform=axs['a'].transAxes, fontsize=f_size_l)
        fig.savefig(Path(ed_save_path, f'{save_name}.pdf'))
        fig.savefig(Path(ed_save_path, f'{save_name}.eps'), dpi=150)

    # Fig 9:n6_ed_fig9_glm_rt_split
    # N.B This is fake data, need to add in the actual data using inkscape
    fig, axs = plt.subplots(2, 3, figsize=(130 * MM_TO_INCH, 170 * MM_TO_INCH),
                            gridspec_kw={'wspace': 0, 'hspace': 0.1})
    plot_glm_swansons(axs.ravel())
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0.02)
    axs[0, 0].text(0.5, 1.1, 'Early responses', transform=axs[0, 0].transAxes, fontsize=f_size_l, ha='center')
    axs[0, 1].text(0.5, 1.1, 'Late responses', transform=axs[0, 1].transAxes, fontsize=f_size_l, ha='center')
    axs[0, 2].text(0.5, 1.1, 'All responses', transform=axs[0, 2].transAxes, fontsize=f_size_l, ha='center')
    save_name = 'n6_ed_fig9_glm_rt_split'
    fig.savefig(Path(ed_save_path, f'{save_name}.pdf'))
    fig.savefig(Path(ed_save_path, f'{save_name}.eps'), dpi=150)

    # Fig 10 n6_ed_fig10_manifold_full
    fig = plt.figure(figsize=(183 * MM_TO_INCH, 150 * MM_TO_INCH))
    width, height = fig.get_size_inches() / MM_TO_INCH
    xspans = get_coords(width, ratios=[2, 2, 2], space=[10, 30], pad=0, span=(0, 1))
    yspans = get_coords(height, ratios=[1, 1, 1], space=20, pad=5, span=(0, 1))
    axs = {
        'a': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0]),
        'b': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1]),
        'c': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[2]),
        'd': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[0]),
        'e': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[1]),
        'f': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[2]),
        'g': fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[0]),
        'h': fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[1]),
        'i': fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[2]),
    }

    gs_3d = fig.add_gridspec(3, 3, wspace=0, hspace=0.2)
    axs['a_3d'] = fig.add_subplot(gs_3d[0, 0], projection='3d')
    axs['b_3d'] = fig.add_subplot(gs_3d[1, 0], projection='3d')
    axs['c_3d'] = fig.add_subplot(gs_3d[2, 0], projection='3d')

    labels = []
    padx = 10
    pady = 5
    labels.append(add_label('a', fig, xspans[0], yspans[0], padx, pady, fontsize=8))
    labels.append(add_label('b', fig, xspans[0], yspans[1], padx, pady, fontsize=8))
    labels.append(add_label('c', fig, xspans[0], yspans[2], padx, pady, fontsize=8))
    labels.append(add_label('d', fig, xspans[1], yspans[0], padx, pady, fontsize=8))
    labels.append(add_label('e', fig, xspans[1], yspans[1], padx, pady, fontsize=8))
    labels.append(add_label('f', fig, xspans[1], yspans[2], padx, pady, fontsize=8))
    labels.append(add_label('g', fig, xspans[2], yspans[0], padx + 2, pady, fontsize=8))
    labels.append(add_label('h', fig, xspans[2], yspans[1], padx + 2, pady, fontsize=8))
    labels.append(add_label('i', fig, xspans[2], yspans[2], padx + 2, pady, fontsize=8))
    fg.add_labels(fig, labels)

    plot_trajectory_lines('stim_restr', ax=axs['d'], adjust=True)
    plot_trajectory_scatter('stim_restr', ax=axs['g'], adjust=True)
    plot_trajectory_lines('choice_restr', ax=axs['e'], adjust=True)
    plot_trajectory_scatter('choice_restr', ax=axs['h'], adjust=True)
    plot_trajectory_lines('fback_restr', ax=axs['f'], adjust=True)
    plot_trajectory_scatter('fback_restr', ax=axs['i'], adjust=True)
    axs['i'].set_ylim(0, 4.2)
    axs['f'].set_ylim(0, 4.2)
    plot_traj3d('stim_restr', ax_3d=axs['a_3d'], ax_2d=axs['a'], ga_pcs=True)
    axs['a'].set_title('')
    plot_traj3d('choice_restr', ax_3d=axs['b_3d'], ax_2d=axs['b'], ga_pcs=True)
    axs['b'].set_title('')
    plot_traj3d('fback_restr', ax_3d=axs['c_3d'], ax_2d=axs['c'], ga_pcs=True)
    axs['c'].set_title('')
    adjust_subplots(fig, adjust=[5, 10, 5, 6], extra=0)

    def shift_plot(ax, x, y):
        pos = ax.get_position()
        pos.x0 = pos.x0 + x
        pos.x1 = pos.x1 + x
        pos.y0 = pos.y0 + y
        pos.y1 = pos.y1 + y
        ax.set_position(pos)

    axs['a'].text(0.2 + 0.16, 1.05, 'Stimulus', ha='center', fontsize=f_size)
    axs['b'].text(0.2 + 0.1, 1.1, 'Choice', ha='center', fontsize=f_size)
    axs['c'].text(0.2 + 0.04, 1.1, 'Feedback', ha='center', fontsize=f_size)

    shift_plot(axs['a_3d'], -0.05, -0.01)
    shift_plot(axs['a'], -0.05, -0.01)
    shift_plot(axs['b_3d'], -0.04, -0.01)
    shift_plot(axs['b'], -0.04, -0.01)
    shift_plot(axs['c_3d'], -0.02, -0.01)
    shift_plot(axs['c'], -0.02, -0.01)

    save_name = 'n6_ed_fig10_manifold_full'
    fig.savefig(Path(ed_save_path, f'{save_name}.pdf'))
    fig.savefig(Path(ed_save_path, f'{save_name}.eps'), dpi=150)

    # Fig 12: n6_ed_fig12_licking
    fig = plt.figure(figsize=(183 * MM_TO_INCH, 170 * MM_TO_INCH))
    width, height = fig.get_size_inches() / MM_TO_INCH

    xspans1 = get_coords(width, ratios=[1, 5], space=[18], pad=10, span=(0, 1))
    xspans2 = get_coords(width, ratios=[1], space=15, pad=20, span=(0, 1))
    xspans3 = get_coords(width, ratios=[1], space=15, pad=15, span=(0, 1))
    yspans = get_coords(height, ratios=[3, 1, 1], space=[20, 15], pad=5, span=(0, 1))
    yspans1 = get_coords(height, ratios=[1, 3], space=[2], pad=0, span=yspans[0])


    axs = {'a_1': fg.place_axes_on_grid(fig, xspan=xspans1[0], yspan=yspans1[0]),
           'a_2': fg.place_axes_on_grid(fig, xspan=xspans1[0], yspan=yspans1[1]),
           'b': fg.place_axes_on_grid(fig, xspan=xspans1[1], yspan=yspans[0], dim=(3, 3), wspace=0.8, hspace=0.25),
           'c': fg.place_axes_on_grid(fig, xspan=xspans2[0], yspan=yspans[1], dim=(1, 3), wspace=0.6),
           'd': fg.place_axes_on_grid(fig, xspan=xspans3[0], yspan=yspans[2], dim=(1, 5), wspace=0.7)}

    labels = []
    padx = 15
    pady = 10
    labels.append(add_label('a', fig, xspans1[0], yspans[0], padx, 20, fontsize=8))
    labels.append(add_label('b', fig, xspans1[1], yspans[0], 10, 20, fontsize=8))
    labels.append(add_label('c', fig, xspans1[0], yspans[1], padx, 5, fontsize=8))
    labels.append(add_label('d', fig, xspans1[0], yspans[2], padx, 5, fontsize=8))
    fg.add_labels(fig, labels)

    plot_lick_raster_psth(axs=np.array([axs['a_1'], axs['a_2']]))
    ylabel = axs['a_2'].get_ylabel()
    axs['a_2'].set_ylabel(ylabel, fontsize=f_size, labelpad=-2)
    #ylabel.set_position((ylabel.get_position()[0] + 0.5, ylabel.get_position()[1]))
    plot_trajectories_with_psd(axs=np.array(axs['b']))
    plot_trajectories_with_licks(axs=np.array(axs['d']))
    single_cell_psd(93, 'c4432264-e1ae-446f-8a07-6280abade813', 'probe01', plot_=True, axs=np.array(axs['c']),
                    save=False)
    for coll in axs['c'][0].get_children():
        if isinstance(coll, plt.Line2D):
            coll.remove()
    leg = axs['c'][0].legend_
    leg.remove()

    axs['c'][0].axvline(x=0.5 / T_BIN, linestyle='--', c='k', zorder=100)
    axs['c'][0].text(0.5/ T_BIN, axs['c'][0].get_ylim()[0] - 400, f"  {EVENT_LINES['fback']}", color='k', fontsize=f_size_s)
    axs['c'][1].set_xlim(0, 30)

    adjust_subplots(fig, adjust=[3, 10, 2, 15], extra=0)
    save_name = 'n6_ed_fig12_licking'
    fig.savefig(Path(ed_save_path, f'{save_name}.pdf'))
    fig.savefig(Path(ed_save_path, f'{save_name}.eps'), dpi=150)

    # Fig 13: n6_ed_fig13_glm_task
    fig = plt.figure(figsize=(183 * MM_TO_INCH, 140 * MM_TO_INCH))
    width, height = fig.get_size_inches() / MM_TO_INCH
    xspans = get_coords(width, ratios=[1, 1], space=0, pad=5, span=(0, 1))
    yspans = get_coords(height, ratios=[2, 1], space=0, pad=0, span=(0, 1))
    axs = {
        'd': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[1])
    }
    labels = []
    padx = 5
    pady = 0
    labels.append(add_label('a', fig, xspans[0], yspans[0], padx, pady, fontsize=8))
    labels.append(add_label('b', fig, xspans[1], yspans[0], padx, pady, fontsize=8))
    labels.append(add_label('c', fig, xspans[0], yspans[1], padx, 2, fontsize=8))
    labels.append(add_label('d', fig, xspans[1], yspans[1], padx, 2, fontsize=8))
    _, cax = plot_swanson_for_single_cell_variable('task', ax=axs['d'], cbar=True, legend=True)
    axs['d'].set_title('')
    fg.add_labels(fig, labels)
    orig_bbox = cax.ax.get_position()
    adjust_subplots(fig, adjust=[3, 10, 5, 5], extra=0)
    cax.ax.set_position([orig_bbox.x0 + 0.11, orig_bbox.y0 -0.04, orig_bbox.width, orig_bbox.height])
    leg = axs['d'].legend_
    leg.set_bbox_to_anchor((0.95, -0.05))
    cax.ax.xaxis.set_ticks([60, 80, 100])
    save_name = 'n6_ed_fig13_glm_task'
    fig.savefig(Path(ed_save_path, f'{save_name}.pdf'))
    fig.savefig(Path(ed_save_path, f'{save_name}.eps'), dpi=150)

    # Fig 15: n6_ed_fig15_movement
    fig = plt.figure(figsize=(183 * MM_TO_INCH, 150 * MM_TO_INCH))
    width, height = fig.get_size_inches() / MM_TO_INCH
    xspans = get_coords(width, ratios=[1, 1], space=10, pad=0, span=(0, 1))
    yspans = get_coords(height, ratios=[1, 1, 1], space=10, pad=0, span=(0, 1))
    axs = {
        'b': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[0]),
        'c': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1]),
        'd': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[1]),
        'e': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[2]),
        'f': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[2]),
    }

    labels = []
    padx = 5
    pady = 10
    labels.append(add_label('a', fig, xspans[0], yspans[0], padx, pady, fontsize=8))
    labels.append(add_label('b', fig, xspans[1], yspans[0], padx, pady, fontsize=8))
    labels.append(add_label('c', fig, xspans[0], yspans[1], padx, 3, fontsize=8))
    fg.add_labels(fig, labels)

    _, cax = plot_swanson_for_single_cell_variable('overall', ax=axs['b'], cbar=True, legend=True)
    _ = plot_swanson_for_single_cell_variable('paw', ax=axs['c'], cbar=False, legend=False)
    axs['b'].set_title('')
    _ = plot_swanson_for_single_cell_variable('tongue', ax=axs['d'], cbar=False, legend=False)
    _ = plot_swanson_for_single_cell_variable('nose', ax=axs['e'], cbar=False, legend=False)
    _ = plot_swanson_for_single_cell_variable('pupil', ax=axs['f'], cbar=False, legend=False)

    fg.add_labels(fig, labels)
    orig_bbox = cax.ax.get_position()
    adjust_subplots(fig, adjust=[3, 0, 5, 5], extra=0)
    cax.ax.set_position([orig_bbox.x0 + 0.11, orig_bbox.y0 + 0.04, orig_bbox.width, orig_bbox.height])
    leg = axs['b'].legend_
    leg.set_bbox_to_anchor((0.95, -0.05))

    save_name = 'n6_ed_fig15_movement'
    fig.savefig(Path(ed_save_path, f'{save_name}.pdf'))
    fig.savefig(Path(ed_save_path, f'{save_name}.eps'), dpi=150)
        

def ghostscript_compress_pdf(variable, level='/printer'):

    '''
    Compress main figs (inkscape pdfs) or whole manuscript  
    
    levels in [/screen, /ebook,  /printer]
    '''


    if variable in VARIABLES:
        input_path = Path(imgs_pth, variable, 
                         f'n5_main_figure_{variverb[variable]}_revised_raw.pdf')
        output_path = Path(imgs_pth, variable, 
                         f'n5_main_figure_{variverb[variable]}_revised.pdf')
                         
    elif variable == 'wheel':
        input_path = Path(imgs_pth, 'speed', 
                         f'n5_main_figure_wheel_revised_raw.pdf')
        output_path = Path(imgs_pth, 'speed', 
                         f'n5_main_figure_wheel_revised.pdf')
                         
    elif variable == 'manuscript':
        input_path = Path('/home/mic/Brainwide_Map_Paper.pdf')
        output_path = Path('/home/mic/Brainwide_Map_Paper2.pdf')    


    else:
       input_path = input("Please enter pdf input_path: ")
       print(f"Received input path: {input_path}")

       output_path = input("Please enter pdf output_path: ")
       print(f"Received output path: {output_path}")
       
       input_path = Path(input_path.strip("'\""))
       output_path = Path(output_path.strip("'\""))
       
       print('input_path', input_path)
       print('output_path', output_path)                 

    # Ghostscript command to compress PDF
    command = [
        'gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4',
        '-dAutoRotatePages=/None',
        f'-dPDFSETTINGS={level}', '-dNOPAUSE', '-dQUIET', '-dBATCH',
        f'-sOutputFile={output_path}', input_path
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print(f"PDF successfully compressed and saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")



def excel_to_latex(file_path):
    """
    Converts an Excel sheet into a LaTeX table with compact formatting, no frame, and tiny font.
    - Replaces 'Low', 'Medium', 'High' with LaTeX symbols.
    - Escapes '&' symbols in column names for LaTeX compatibility.
    - Uses ultra-compact columns.
    
    Parameters:
    - file_path: str, path to the Excel file.

    Returns:
    - LaTeX table as a string.
    """
    columns = ['Firstname', 'Middle name', 'Surname', 'Conceptualization', 'Data Curation',
               'Formal Analysis', 'Funding Acquisition', 'Experimental investigation',
               'Methodology', 'Project Administration', 'Resources',
               'Software (pipeline development)', 'Supervision', 'Validation',
               'Visualization', 'Writing - Original Draft Preparation',
               'Writing - Review & Editing']

    # Load Excel file
    df = pd.read_excel(file_path, sheet_name='Participation levels')

    # Select specific columns
    df = df[columns]

    # Replace NaN values with empty strings
    df = df.fillna("")

    # Escape '&' in column names for LaTeX
    df.columns = [col.replace("&", r"\&") for col in df.columns]

    # Define LaTeX-compatible mapping for Low, Medium, High
    symbol_map = {
    "Low": "+",  # ●
    "Medium": "++",  # ●● (closely spaced)
    "High": "+++"}  # ●●● (closely spaced)
        # Full circle ●}  # Uses LaTeX symbols
    df = df.replace(symbol_map)

    # Define column widths: "Firstname" & "Surname" wider, all others ultra-narrow
    col_format = "p{1cm} p{0.35cm} p{1cm} " + " ".join(["p{0.1cm}"] * (len(df.columns) - 3))

    # LaTeX table header
    latex_table = "\\begin{table}[h]\n\\centering\n"
    latex_table += "\\tiny\n"  # Tiny font size for the whole table
    latex_table += "\\begin{tabular}{" + col_format + "}\n"

    # Column headers with rotation (except Name/Surname)
    header_row = " & ".join(
        [df.columns[0], df.columns[1], df.columns[2]] + 
        [f"\\rotatebox{{90}}{{{col}}}" for col in df.columns[3:]]
    ) + " \\\\\n\\hline\n"
    latex_table += header_row

    # Data rows
    for _, row in df.iterrows():
        row_values = " & ".join(str(val) for val in row)
        latex_table += row_values + " \\\\\n"

    # Footer
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Low (+), medium (++) and high (+++) author contributions.}\n\\label{tab:author}\n\\end{table}"

    return latex_table


