import pandas as pd
import numpy as np
from pathlib import Path
import math, string
from collections import Counter, OrderedDict
from functools import reduce
import os
import itertools
from scipy import stats
from statsmodels.stats.multitest import multipletests
import subprocess

from one.api import ONE
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
from iblatlas.plots import plot_swanson_vector, plot_scalar_on_slice
import iblatlas
import ibllib

from matplotlib.colors import LinearSegmentedColormap
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

from brainwidemap import download_aggregate_tables, bwm_units
from brainwidemap.encoding.design import generate_design
from brainwidemap.encoding.glm_predict import GLMPredictor, predict
from brainwidemap.encoding.utils import load_regressors, single_cluster_raster, find_trial_ids
from brainbox.plot import peri_event_time_histogram

import neurencoding.linear as lm
from neurencoding.utils import remove_regressors

import warnings
#warnings.filterwarnings("ignore")



'''
This script is used to plot the main result figures of the BWM paper.
The raw results from each analysis can be found in bwm_figs_res.
There are 4 analyses: manifold, decoding, glm, single-cell
See first function in code block of this script for each analysis type
for data format conversion.
'''


ba = AllenAtlas()
br = BrainRegions()
one = ONE()
#one = ONE(base_url='https://openalyx.internationalbrainlab.org',
#          password='international', silent=True)


variables = ['stim', 'choice', 'fback']
variverb = dict(zip(variables,['stimulus', 'choice', 'feedback']))
 
# pooled results
meta_pth = Path(one.cache_dir, 'bwm_res', 'bwm_figs_data', 'meta')
meta_pth.mkdir(parents=True, exist_ok=True)

# main fig panels
imgs_pth = Path(one.cache_dir, 'bwm_res', 'bwm_figs_imgs')
for variable in variables:
    Path(imgs_pth, variable).mkdir(parents=True, exist_ok=True)         

# decoding results
dec_pth = Path(one.cache_dir, 'bwm_res', 'bwm_figs_data','decoding')
dec_pth.mkdir(parents=True, exist_ok=True)  

# manifold results
man_pth = Path(one.cache_dir, 'bwm_res', 'bwm_figs_data','manifold')
man_pth.mkdir(parents=True, exist_ok=True)

# encoding results
enc_pth = Path(one.cache_dir, 'bwm_res', 'bwm_figs_data', 'encoding')
enc_pth.mkdir(parents=True, exist_ok=True)


# single_cell (MannWhitney) results
sc_pth = Path(one.cache_dir, 'bwm_res', 'bwm_figs_data', 'single_cell')
sc_pth.mkdir(parents=True, exist_ok=True)


sigl = 0.05  # significance level (for stacking, plotting, fdr)

plt.ion()  # interactive plotting on

f_size = 8  # font size large
f_size_s = 0.7 * f_size # font size small

#mpl.rcParams['figure.autolayout']  = True
mpl.rcParams.update({'font.size': f_size})


'''
#####
meta (Data pooling, Swansons, intro bar and tables)
#####
'''

def pool_results_across_analyses(return_raw=False):

    '''
    input are various csv files from 
    4 different analysis types ['glm','euc', 'mw', 'dec']
    variables ['stim', ' choice', 'fback']

    some files need conversion to csv (manifold, glm); 
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

    d0 = dict(zip(variables,res))
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
    
    for vari in variables:
        d[vari] = pd.read_csv(Path(man_pth / f'{vari}_restr.csv'))[[
                    'region','amp_euc_can', 'lat_euc_can','p_euc_can']]
    
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
    print('intergated manifold results')    
    
    '''
    # Mann Whitney (single_cell)
    '''
    
    d = {}   
    mw = pd.read_csv(Path(sc_pth,  
            'Updated_Single_cell_analysis_July_10_2024 - Sheet1.csv'))
    #pd.set_option('future.no_silent_downcasting', True)
    for vari in variables:
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
        #d[vari].reset_index(inplace=True)    

    
    D['mannwhitney'] = d
    print('intergated MannWhitney results')
    
    '''   
    # decoding (dec)
    '''
   
    d = {}     
              
    for vari in variables:
    
        d[vari] = pd.read_parquet(Path(dec_pth,
                    f'{vari}_stage3.pqt'))[[
                    'region','valuesminusnull_median',
                    'sig_combined_corrected']].rename(columns = {
                    'valuesminusnull_median': 'decoding_effect',
                    'sig_combined_corrected': 'decoding_significant'})
                
        d[vari].dropna(axis=0,how='any',subset=['decoding_effect'], inplace=True)
        #d[vari].reset_index(inplace=True)
        
    D['decoding'] = d   
    print('intergated decoding results')
    if return_raw:
        return D
       
    # merge frames across analyses    
    for vari in variables:
    
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
                
        # ## Apply logarithm to manifold results              
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
          'velocity': 'GLMs_wheel_velocity.pkl'} 

    # odd cluster indices???    
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
    


def get_cmap_(variable):
    '''
    for each variable, get a colormap defined by Yanliang,
    updated by Chris
    '''
    dc = {'stim': ["#EAF4B3","#D5E1A0", "#A3C968",
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
                    "#42328E", "#262054"]}

    if '_' in variable:
        variable = variable.split('_')[0]

    return LinearSegmentedColormap.from_list("mycmap", dc[variable])


def beryl_to_cosmos(beryl_acronym,br):
    beryl_id = br.id[br.acronym==beryl_acronym]
    return br.get(ids=br.remap(beryl_id, source_map='Beryl', 
                  target_map='Cosmos'))['acronym'][0]


def swanson_to_beryl_hex(beryl_acronym,br):
    beryl_id = br.id[br.acronym==beryl_acronym]
    rgb = br.get(ids=beryl_id)['rgb'][0].astype(int)
    return '#' + rgb_to_hex((rgb[0],rgb[1],rgb[2]))


def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb


def plot_swansons(variable, fig=None, axs=None):

    '''
    for a single variable, plot 5 results swansons,
    4 effects for the 4 analyses and latencies for manifold
    '''


    res = pd.read_pickle(meta_pth / f"{variable}.pkl")

    lw = 0.1  # .01

    # results to plot in Swansons with labels for colorbars
    res_types = {'decoding_effect': ['Decoding. $R^2$ over null',[], 
                    ['Decoding', 'Regularized logistic regression']],
                 'mannwhitney_effect': ['Frac. sig. cells',[],
                    ['Single cell statistics', 'C.C Mann-Whitney test']],
                 'euclidean_effect': ['Nrml. Eucl. dist. (log)',[],
                    ['Manifold', 'Distance between trajectories']],
                 'euclidean_latency': ['Latency of dist. (sec)',[],
                    ['Manifold', 'Time near peak']],      
                 'glm_effect': ['Abs. diff. $\\Delta R^2$ (log)',[],
                    ['Encoding', 'General linear model']]}

     
    cmap = get_cmap_(variable)
    
    alone = False
    if not fig:
        fig = plt.figure(figsize=(8,3.34), layout='constrained')  
        gs = gridspec.GridSpec(1, len(res_types), 
                               figure=fig,hspace=.75)
        axs = []
        alone = True
                 
    k = 0
    for res_type in res_types:
        if alone:
            axs.append(fig.add_subplot(gs[0,k]))
            
            
        ana = res_type.split('_')[0]
        lat = True if 'latency' in res_type else False
        dt = 'effect' if not lat else 'latency'

        if ana != 'glm':
        
            # check if there are p-values and mask
            acronyms = res[res[f'{ana}_significant'] == True][
                        'region'].values
            scores = res[res[
                        f'{ana}_significant'] == True][
                        f'{ana}_{dt}'].values

            if lat:            
                mask = res[np.bitwise_or(
                            res[f'{ana}_significant'] == False,
                            np.isnan(res[f'{ana}_{dt}']))][
                            'region'].values               
            else:            
  
                # remove regs from mask with nan amps (not analyzed)            
                mask = res[np.bitwise_and(
                            res[f'{ana}_significant'] == False,
                            ~np.isnan(res[f'{ana}_{dt}']))][
                            'region'].values
        
        else:
            acronyms = res['region'].values
            scores = res[f'{ana}_effect'].values
            mask = [] 

        
        plot_swanson_vector(acronyms,
                            scores,
                            hemisphere=None, 
                            orientation='portrait',
                            cmap=cmap.reversed() if lat else cmap,
                            br=br,
                            ax=axs[k],
                            empty_color="white",
                            linewidth=lw,
                            mask=mask,
                            mask_color='silver',
                            annotate= True,
                            annotate_n=5,
                            annotate_order='bottom' if lat else 'top',
                            fontsize=f_size_s)

        clevels = (min(scores), max(scores))    
        norm = mpl.colors.Normalize(vmin=clevels[0], vmax=clevels[1])
        cbar = fig.colorbar(
                   mpl.cm.ScalarMappable(norm=norm,cmap=cmap.reversed() 
                   if lat else cmap),
                   ax=axs[k],shrink=0.4,aspect=12,pad=.025,
                   orientation="horizontal")
                   
        ticks = np.round(np.linspace(cbar.vmin, cbar.vmax, num=3), 2)             
        cbar.set_ticks(ticks)
        cbar.outline.set_visible(False)
        cbar.ax.xaxis.set_tick_params(pad=5)
        cbar.set_label(res_types[res_type][0], fontsize=f_size_s)
        cbar.ax.tick_params(labelsize=f_size_s)
        
        axs[k].text(-0.25, 0.5, res_types[res_type][2][0],
                fontsize=f_size, ha='center',va = 'center', 
                rotation='vertical', 
                transform=axs[k].transAxes)
        axs[k].text(-0.1, .5, res_types[res_type][2][1],
                fontsize=f_size_s, ha='center',va = 'center',
                rotation='vertical', 
                transform=axs[k].transAxes)
        axs[k].text(0.85, 0.95, f' {len(scores)}/{len(scores) + len(mask)}',
                fontsize=f_size_s, ha='center', 
                transform=axs[k].transAxes)                

        # print regions with largest (smallest) amp (lat) scores 
        if lat:
            exxregs = acronyms[np.argsort(scores)][:7]
        else:
            exxregs = acronyms[np.argsort(scores)][-7:]
        
        print(res_type)       
        print('highlight regs')
        print(exxregs)
#        for i, text in enumerate(exxregs[:3]):
#            axs[k].text(-0.2, 0.2 - i * 0.05,  
#            text, fontsize=f_size_s, 
#            ha='left', va='top',
#            transform=axs[k].transAxes)
#                   
#        for i, text in enumerate(exxregs[3:]):
#            axs[k].text(0.85, 0.1 - i * 0.05,  
#            text, fontsize=f_size_s, 
#            ha='left', va='top',
#            transform=axs[k].transAxes)                                

        axs[k].axis("off")
        
        axs[k].axes.invert_xaxis()
                        
        k += 1  

    if alone:
        fig.savefig(Path(imgs_pth, variable, 'swansons.svg'))          


def plot_slices(variable):

    '''
    For a single variable, plot effects for the 4 analyses and
    latencies of manifolds onto brain slices
    '''

    res = pd.read_pickle(meta_pth / f"{variable}.pkl")


    # results to plot in Swansons with labels for colorbars
    res_types = {'decoding_effect': ['Decoding. $R^2$ over null',[], 
                    ['Decoding \n', 'Regularized logistic regression \n']],
                 'mannwhitney_effect': ['Frac. sig. cells',[],
                    ['Single cell statistics \n', 'C.C Mann-Whitney test \n']],
                 'euclidean_effect': ['Nrml. Eucl. dist. (log)',[],
                    ['Manifold \n', 'Distance between trajectories \n']],
                 'euclidean_latency': ['Latency of dist. (sec)',[],
                    ['Manifold \n', 'Time near peak \n']],      
                 'glm_effect': ['Abs. diff. $\\Delta R^2$ (log)',[],
                    ['Encoding \n', 'General linear model \n']]}
    
    cmap = get_cmap_(variable)
    cmap_mask = LinearSegmentedColormap.from_list('custom_colormap', 
                                                 ['silver', 'silver'])
                                                   
    # Setup GridSpec with additional space for colorbars
    nrows = 4  # Number of rows for plots
    cb_height = 0.05  # Height of colorbar
    cb_space = 0.02  # Space between colorbar and plot
    fig = plt.figure(figsize=(8.16, 6.38))  
    gs = gridspec.GridSpec(nrows + 1, len(res_types), 
                           figure=fig,hspace=.75,
                           height_ratios=[1]*nrows + [0.05])
    axs = []
                 
    
    colm = 0  # column index
    k = 0  # panel index
    for res_type in res_types:
                       
        ana = res_type.split('_')[0]
        lat = True if 'latency' in res_type else False
        dt = 'effect' if not lat else 'latency'
        
        if ana != 'glm':
            # check if there are p-values
            
            acronyms = res[res[f'{ana}_significant'] == True]['region'].values
            scores = res[res[
                     f'{ana}_significant'] == True][f'{ana}_{dt}'].values
                        
            if lat:
                mask = res[np.bitwise_or(
                            res[f'{ana}_significant'] == False,
                            np.isnan(res[f'{ana}_{dt}']))][
                            'region'].values            
            
            else:            
                # remove regs from mask with nan amps (not analyzed)            
                mask = res[np.bitwise_and(
                            res[f'{ana}_significant'] == False,
                            ~np.isnan(res[f'{ana}_{dt}']))][
                            'region'].values
        
        else:
            acronyms = res['region'].values
            scores = res[f'{ana}_effect'].values
            mask = []

        vmin, vmax = (min(scores), max(scores))
        res_types[res_type][1] = [vmin,vmax]
        
        row = 0  #row index
      
        for st in ['sagittal', 'top']:
            if st == 'sagittal':
                coords = [-1800, -800, -200]  # coordinate in [um]
            else:
                coords = [-1800]  # ignored for top view

        
            for coord in coords:
                label = f'{res_type} {st} {coord}'
                axs.append(fig.add_subplot(gs[row,colm], 
                                           label=label))  
        
                # plot significant scores
                plot_scalar_on_slice(acronyms, scores, coord=coord, 
                                     slice=st, mapping='Beryl', 
                                     hemisphere='left', background='boundary', 
                                     cmap=cmap.reversed() if lat else cmap,
                                     brain_atlas=ba, ax=axs[k],
                                     empty_color='white')
                
                if len(mask) != 0:                               
                    # plot insignificant scores      
                    plot_scalar_on_slice(mask, np.random.rand(len(mask)), 
                                        coord=coord, slice=st, 
                                        mapping='Beryl', 
                                        hemisphere='left',
                                        background='boundary', 
                                        cmap=cmap_mask, brain_atlas=ba,
                                        ax=axs[k], empty_color='white')

                axs[k].axis("off")
                if row == 0:
                    axs[k].text(0.5, 1.2, res_types[res_type][2][0],
                            fontsize=f_size, ha='center', 
                            transform=axs[k].transAxes)                 
                    axs[k].text(0.5, 1, res_types[res_type][2][1],
                            fontsize=f_size_s, ha='center', 
                            transform=axs[k].transAxes)
                row += 1
                k += 1
        colm += 1                                                   

    # tweak layout
    fig.subplots_adjust(    
                top=0.92,
                bottom=0.1,
                left=0.0,
                right=1.0,
                hspace=0.0,
                wspace=0.0)
                
    # manually reduce size as tight_layout fails
    shrink = 1.7  # size reduction factor 
        
    for ax in axs:
        if 'top' in ax.get_label():
            bbox = ax.get_position()
            left, bottom, width, height = (bbox.x0, bbox.y0, 
                                           bbox.width, bbox.height)
            ax.set_position([left + (width - width*shrink) / 2 + 0.02, 
                             bottom + (height - height*shrink) / 2, 
                             width*shrink, height*shrink])
                             
    colorbar_width_proportion = 0.5                         
    # Create colorbar axes at the bottom row of the GridSpec
    for i, res_type in enumerate(res_types):
        cax = fig.add_subplot(gs[-1, i])  # New axis for colorbar
        vmin, vmax = res_types[res_type][1]
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                         norm=norm, orientation='horizontal')
        cbar.set_label(res_types[res_type][0], fontsize=f_size)
        cbar.ax.tick_params(labelsize=f_size_s)
        cbar.outline.set_visible(False)
        
        ticks = np.round(np.linspace(cbar.vmin, cbar.vmax, num=3), 2)             
        cbar.set_ticks(ticks)
        bbox = cax.get_position()
        center_offset = (bbox.width * (1 - colorbar_width_proportion)) / 1.5
        new_x_position = bbox.x0 + center_offset

        # Set the new position with the updated x position and adjusted width
        cax.set_position([new_x_position, bbox.y0, 
            bbox.width * colorbar_width_proportion, bbox.height])
        
        
        
                       
    fig.savefig(Path(imgs_pth, 'si', 
                     f'n6_supp_figure_{variverb[variable]}.svg'),  
                     bbox_inches='tight')
    fig.savefig(Path(imgs_pth, 'si', 
                     f'n6_supp_figure_{variverb[variable]}.pdf'),
                     dpi=150,
                     bbox_inches='tight')          

    
def plot_all_swansons():

    '''
    SI figure swansons for all three main variables and analyses
    scores are normalized across variables per analysis
    ''' 


    lw = 0.1  

    # results to plot in Swansons with labels for colorbars
    # and vmin, vmax for each analysis
    res_types = {'decoding_effect': ['Decoding. $R^2$ over null',[], 
                    ['Decoding', 'Regularized logistic regression']],
                 'mannwhitney_effect': ['Frac. sig. cells',[],
                    ['Single cell statistics', 'C.C Mann-Whitney test']],
                 'euclidean_effect': ['Nrml. Eucl. dist. (log)',[],
                    ['Manifold', 'Distance between trajectories']],
                 'glm_effect': ['Abs. diff. $\\Delta R^2$ (log)',[],
                    ['Encoding', 'General linear model']]}
     
    cmap = 'viridis'
    
    num_columns = len(res_types)

    fig = plt.figure(figsize=(8, 9))  
    gs = gridspec.GridSpec(len(variables) + 1, len(res_types),
                           height_ratios=[2]*len(variables) + [0.1],
                           figure=fig, hspace=.75)
    axs = []
    
    k = 0  # axis counter             
    row = 0
    for variable in variables:
        res = pd.read_pickle(meta_pth / f"{variable}.pkl")
        col = 0
        for res_type in res_types:
        
            ana = res_type.split('_')[0]
            # get vmin vmax across variable
            all_scores = []
            for vari in variables:
                res_ = pd.read_pickle(meta_pth / f"{vari}.pkl")
                if ana != 'glm':
                    scores = res_[res_[
                             f'{ana}_significant'] == True][
                             f'{ana}_effect'].values            
                else:    
                    scores = res_[f'{ana}_effect'].values
                all_scores.append(scores)
            
            all_scores = list(itertools.chain(*all_scores))        
        
                    
            axs.append(fig.add_subplot(gs[row, col]))
            
            if col == 0:
                axs[-1].text(-0.1, 0.5, variverb[variable], fontsize=f_size,
                 rotation='vertical', va='center', ha='right', 
                 transform=axs[-1].transAxes)
                  
            

            if ana != 'glm':
                # check if there are p-values
                
                acronyms = res[res[f'{ana}_significant'] == True]['region'].values
                scores = res[res[
                         f'{ana}_significant'] == True][f'{ana}_effect'].values
                            
                # remove regs from mask with nan amps (not analyzed)
                mask = res[np.bitwise_and(
                            res[f'{ana}_significant'] == False,
                            ~np.isnan(res[f'{ana}_effect']))][
                            'region'].values
            
            else:
                acronyms = res['region'].values
                scores = res[f'{ana}_effect'].values
                mask = []

#            if ana == 'euclidean':
#                scores = np.log(scores)
#                all_scores = np.log(all_scores)
                
            vmin, vmax = (np.min(all_scores), np.max(all_scores))
            res_types[res_type][1] = vmin, vmax  
            
            plot_swanson_vector(acronyms,
                                scores,
                                hemisphere=None, 
                                orientation='portrait',
                                cmap=cmap,
                                br=br,
                                ax=axs[k],
                                empty_color="white",
                                linewidth=lw,
                                mask=mask,
                                mask_color='silver',
                                annotate= False,
                                vmin=vmin,
                                vmax=vmax)

            axs[k].text(1.21, 0.9, f'{len(scores)}/{len(scores) + len(mask)}',
                    fontsize=f_size, ha='right', 
                    transform=axs[k].transAxes)                 

            if row == 0:
                axs[k].text(0.5, 1.1, res_types[res_type][2][0],
                        fontsize=f_size, ha='center', 
                        transform=axs[k].transAxes)                 
                axs[k].text(0.5, 1.05, res_types[res_type][2][1],
                        fontsize=f_size_s, ha='center', 
                        transform=axs[k].transAxes)                             
                               
                               
            axs[k].set_xticks([])
            axs[k].set_yticks([])
            axs[k].axis("off")
            
            axs[k].axes.invert_xaxis()
                            
            k += 1  
            col +=1
        row += 1

    # Create colorbars
    for i, res_type in enumerate(res_types):
        axs.append(fig.add_subplot(gs[-1, i]))

        vmin, vmax = res_types[res_type][1]
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = mpl.colorbar.ColorbarBase(axs[-1], cmap=cmap, norm=norm,
         orientation='horizontal')
        cbar.set_label(res_types[res_type][0], fontsize=f_size)
        cbar.ax.tick_params(labelsize=f_size_s)
        cbar.outline.set_visible(False)
        ticks = np.round(np.linspace(cbar.vmin, cbar.vmax, num=3), 2)             
        cbar.set_ticks(ticks)

    # manually adjust layout as tight_layout is not working
    fig.subplots_adjust(left=0.01, right=0.95, top=0.87, 
                        bottom=0) 
                                                                
    shrink = 1.5                        
    for ax in axs[:-4]:

        bbox = ax.get_position()
        left, bottom, width, height = (bbox.x0, bbox.y0, 
                                       bbox.width, bbox.height)
        ax.set_position([left + (width - width*shrink) / 2 + 0.02, 
                         bottom + (height - height*shrink) / 2, 
                         width*shrink, height*shrink])
                              
    for ax in axs[-4:]:
        # Adjust width of colorbars
        bbox = ax.get_position()
        new_width = bbox.width * 0.5
        new_center_x = bbox.x0 + bbox.width / 2
        new_y_position = bbox.y0 + 0.05   
        ax.set_position([new_center_x - new_width / 2, 
                         new_y_position, new_width, bbox.height])


    fig.text(0.01, 0.5, 'Task variable', fontsize=f_size, 
        rotation='vertical',  va='center')

    fig.suptitle('Analysis', fontsize=f_size, ha='center')        
    fig.savefig(Path(imgs_pth, 'si', 'n6_supp_all_variables_revised.svg'))
    fig.savefig(Path(imgs_pth, 'si', 'n6_supp_all_variables_revised.pdf'))    


def plot_wheel_swansons(fig=None, axs=None):

    '''
    For decoding and encoding, plot Swansons for speed and velocity
    '''

    lw = 0.1  # .01

    # results to plot in Swansons with labels for colorbars
    res_types = {'decoding_effect': ['Decoding. $R^2$ over null',[], 
                    ['Decoding', 'Regularized logistic regression']],
                 'glm_effect': ['Abs. diff. $\\Delta R^2$',[],
                    ['Encoding', 'General linear model']]}
    
    varis = ['speed', 'velocity']
     
    cmap = get_cmap_('speed')
    
    alone = False
    if not fig:
        fig = plt.figure(figsize=(8.25,3.75), layout='constrained')  
        gs = gridspec.GridSpec(1, len(res_types)*len(varis), 
                               figure=fig,hspace=.75)
        axs = []
        alone = True
                 
    k = 0
    for vari in varis:
        res = pd.read_pickle(meta_pth / f"{vari}.pkl")
        for res_type in res_types:
            if alone:
                axs.append(fig.add_subplot(gs[0,k]))
                
                
            ana = res_type.split('_')[0]

            if ana != 'glm':
                # check if there are p-values
                
                acronyms = res[res[f'{ana}_significant'] == True]['region'].values
                scores = res[res[
                         f'{ana}_significant'] == True][f'{ana}_effect'].values
                            
                # remove regs from mask with nan amps (not analyzed)
                mask = res[np.bitwise_and(
                            res[f'{ana}_significant'] == False,
                            ~np.isnan(res[f'{ana}_effect']))][
                            'region'].values
            
            else:
                acronyms = res['region'].values
                scores = res[f'{ana}_effect'].values
                mask = []
                
            vmin, vmax = (np.min(scores), np.max(scores))
            res_types[res_type][1] = vmin, vmax
            plot_swanson_vector(acronyms,
                                scores,
                                hemisphere=None, 
                                orientation='portrait',
                                cmap=cmap,
                                br=br,
                                ax=axs[k],
                                empty_color="white",
                                linewidth=lw,
                                mask=mask,
                                mask_color='silver',
                                annotate= True,
                                annotate_n=5,
                                annotate_order='top',
                                vmin=vmin,
                                vmax=vmax,
                                fontsize=f_size_s)

            clevels = (min(scores), max(scores))
        
            norm = mpl.colors.Normalize(vmin=clevels[0], vmax=clevels[1])
            cbar = fig.colorbar(
                       mpl.cm.ScalarMappable(norm=norm,cmap=cmap),
                       ax=axs[k],shrink=0.4,aspect=12,pad=.025,
                       orientation="horizontal")
                       
            cbar.ax.tick_params(axis='both', which='major',
                                labelsize=f_size_s)
            ticks = np.round(np.linspace(cbar.vmin, cbar.vmax, num=3),
                             2)
            cbar.set_ticks(ticks)                    
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(size=2)
            cbar.ax.xaxis.set_tick_params(pad=5)
            cbar.set_label(res_types[res_type][0], fontsize=f_size)
                     
            axs[k].text(-0.25, 0.5, res_types[res_type][2][0],
                    fontsize=f_size, ha='center',va = 'center', 
                    rotation='vertical', 
                    transform=axs[k].transAxes)
            axs[k].text(-0.1, .5, res_types[res_type][2][1],
                    fontsize=f_size_s, ha='center',va = 'center',
                    rotation='vertical', 
                    transform=axs[k].transAxes)
            axs[k].text(0.85, 0.95, f' {len(scores)}/{len(scores) + len(mask)}',
                    fontsize=f_size_s, ha='center', 
                    transform=axs[k].transAxes) 
                  
            axs[k].set_xticks([])
            axs[k].set_yticks([])
            axs[k].axis("off")
            
            axs[k].axes.invert_xaxis()
                            
            k += 1  

    if alone:
        fig.savefig(Path(imgs_pth, 'speed', 'wheel_swansons.svg')) 


def plot_table(variable):


    # # Plot comparison table
    res = pd.read_pickle(meta_pth / f"{variable}.pkl")
    cmap = get_cmap_(variable)
    
    # Normalize values in each amplitude column to interval [0,1]
    # assuming columns 
    if variable in ['speed', 'velocity']:
        anas = ['decoding', 'glm']
        si, se = 1,3
        column_labels = {
        'region': 'region',
        'glm_effect': 'encoding',
        'decoding_effect': 'decoding'}
        
           
    else:
        anas = ['decoding', 'mannwhitney', 'glm', 'euclidean']
        si, se = 2,6
        column_labels = {
        'region': 'region',
        'glm_effect': 'encoding',
        'euclidean_effect': 'manifold',
        'mannwhitney_effect': 'single-cell',
        'decoding_effect': 'decoding'}        
        
        

    effs = ['_'.join([a,'effect']) for a in anas]

    assert  (set(list(res.iloc[:,si:se].keys())) == 
             set(effs))       
            
    res.iloc[:,si:se] = (res.iloc[:,si:se] - res.iloc[:,si:se].min())/(
                      res.iloc[:,si:se].max()-res.iloc[:,si:se].min()) + 1e-4

    # order rows (regions) canonically, omitting those without data
    regs0 = list(res['region']) 
    p = (Path(iblatlas.__file__).parent / 'beryl.npy')
    regs_can = br.id2acronym(np.load(p), mapping='Beryl')
    regs1 = []
    for reg in regs_can:
        if reg in regs0:
            regs1.append(reg)        

    res['region'] = pd.Categorical(res['region'], 
                                   categories=regs1, ordered=True)
    res = res.sort_values('region')
    res = res.reset_index(drop=True)

    for ana in anas:
        if ana == 'glm':
            continue
        res[f'{ana}_effect'] = res[f'{ana}_effect'
                              ] * res[f'{ana}_significant']

    res = res[['region'] + effs]


    ## format table
    _, pal = get_allen_info()
    def region_formatting(x):
        '''
        Formatting for acronym strings
        '''
        color = mpl.colors.to_hex(pal[x])
        
        return 'background-color: w; color: ' + color


    def effect_formatting(x):
        '''
        Formatting for effect columns
        '''
        if x==0:  # not significant (values were set to zero)
            color = 'silver'
        elif pd.isna(x):  # not analysed
            color = 'w'    
        else:
            rgb = cmap(x)
            color =  ('#' + rgb_to_hex((int(255*rgb[0]),
                                        int(255*rgb[1]),
                                        int(255*rgb[2]))))
                         
        return 'background-color: ' + color



    res = res.rename(columns=column_labels)
    effs = list(column_labels.values())[1:]
    columns_order = ['region'] + list(reversed(effs[::-1]))
    res = res[columns_order]
    
    # split data in two to have two tables
    midpoint = len(res) // 2
    df1 = res.iloc[:midpoint].reset_index(drop=True)
    df2 = res.iloc[midpoint:].reset_index(drop=True)
      
    # Format table  
    def make_pretty(styler):
        
        styler.applymap(effect_formatting, subset=effs)
        styler.applymap(region_formatting, subset=['region'])
        styler.set_properties(subset=effs, **{'width': '8px'})
        styler.set_properties(subset=effs, **{'font-size': '0pt'})
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
 
    k = 0
    for daf in [df1, df2]:
        ## Plot table
        res0 = daf.style.pipe(make_pretty)
          
        pf = Path(imgs_pth, variable)
        pf.mkdir(parents=True, exist_ok=True)
        dfi.export(res0, Path(pf,f'table_{k}.png'), max_rows=-1, dpi = 200)
        k +=1
        
        

def scatter_analysis_effects(variable, analysis_pair,sig_only=False,
                             ax=None):
    '''
    Scatter plot: comparison of two analysis amplitudes for a given variable
    
    Parameters:
        variable (str): One of 'stim', 'choice', 'fback'
        analysis_pair (tuple): Pair of analyses, e.g., 
        ('glm_effect', 'decoding_effect')
        meta_pth (Path): Path to the directory containing data files
    
    '''

    ana_labs = {
        'glm': 'encoding',
        'euclidean': 'manifold',
        'mannwhitney': 'single-cell',
        'decoding': 'decoding'}

    
    # Load the data for the specified variable
    df = pd.read_pickle(Path(meta_pth) / f"{variable}.pkl")
    df['glm_significant'] = 1
    
    alone = False
    ss = 5 # scatter plot dot size
    fig = plt.gcf()
    if not ax:
        alone = True
        ss = 20
        # Prepare the figure
        fig, ax = plt.subplots(figsize=(6,6))
    
    if sig_only:
        # Fetch data for both analyses, sig regions only
        df2 = df.query(f'{analysis_pair[0].split("_")[0]}_significant == 1 &'
                       f'{analysis_pair[1].split("_")[0]}_significant == 1')
        val1 = df2[analysis_pair[0]].values
        val2 = df2[analysis_pair[1]].values
        regs = df2['region'].values
    else:
        # Fetch data for both analyses in the pair
        val1 = df[analysis_pair[0]].values
        val2 = df[analysis_pair[1]].values
        regs = df['region'].values
    
    # Load a color palette
    _, pal = get_allen_info()
    cols = np.array([pal[reg] for reg in regs])
    
    # Remove NaN values
    valid_indices = ~np.isnan(val1) & ~np.isnan(val2)
    val1 = val1[valid_indices]
    val2 = val2[valid_indices]
    cols = cols[valid_indices]
    regss = regs[valid_indices]
    
    # Scatter plot
    scatter = ax.scatter(val1, val2, color=cols, s=ss)
    
    if alone:
        # Annotating each point with the region code
        for i, reg in enumerate(regss):
            ax.annotate(' ' + reg, (val1[i], val2[i]), 
                        fontsize=f_size, color=cols[i])
    
    # Set labels
    ax.set_xlabel(ana_labs[f'{analysis_pair[0].split("_")[0]}'])
    ax.set_ylabel(ana_labs[f'{analysis_pair[1].split("_")[0]}'])
        
#    # Calculate and display correlation coefficients
#    cors, ps = spearmanr(val1, val2)
#    corp, pp = pearsonr(val1, val2)
    

    
    fig.tight_layout()   

#    fig.savefig(Path(fgs_path,
#        f'{variable}_{analysis_pair[0]}_{analysis_pair[1]}.png'))


def scatter_analysis_effects_grid(sig_only=True):

    '''
    for the three main variables and 4 analyses,
    plot grid of scatters of amplitudes
    '''
    
    ylabs = ['euclidean_effect',
             'glm_effect',
             'mannwhitney_effect',
             'decoding_effect'] 

    variables = ['stim', 'choice', 'fback']

    fig, axs = plt.subplots(nrows =len(list(itertools.combinations(ylabs,2))),
                           ncols=len(variables), 
                           figsize=(8.73,9.7))
    
    axs = axs.flatten('F')
    
    k = 0
    for variable in variables:
        for p in itertools.combinations(ylabs,2):
            scatter_analysis_effects(variable, p,sig_only = sig_only,
                                     ax =axs[k])
            axs[k].spines['top'].set_visible(False)
            axs[k].spines['right'].set_visible(False)
            if k%len(list(itertools.combinations(ylabs,2))) == 0:
                axs[k].set_title(f'{variverb[variable]}')
            else:
                axs[k].set_title(None)    
            k +=1 

    fig.tight_layout()

    fig.savefig(Path(imgs_pth, 'si',
        f'n6_supp_analyses_amp_pairs_grid.pdf'), dpi=150)
        

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
                
                    key = [x for x in res[v].keys()
                           if ((a in x) and ('effect' in x))]
                    if key == []:
                        res3[('_').join([v,a])] = None
                        continue
                    else:   
                        key = key[0]
                                  
                    if reg in res[v]['region'].values:
                        score = np.round(res[v][
                                    res[v]['region'] == reg][key].item(),2)      
                        res3[('_').join([v,a])] = score
                    else:
                        res3[('_').join([v,a])] = None
                       
            res2[reg] = res3   
    
                    
        # get all regions in the canonical set
        units_df = bwm_units(one)
        gregs = Counter(units_df['Beryl'])
        
        print('canonical bwm dataset')
        print(f"{len(np.unique(units_df['Beryl']))} regions")
        print(len(units_df['Beryl']), 'good neurons')
    
        columns = (['Beryl', 'Beryl', 'Cosmos', 'Cosmos', 
                   '# recordings', '# neurons', 
                   '# good neurons', 'canonical'] + effects)
                                      
        r = []                       
        for k in range(len(regs1)):
            cano = True if regs1[k] in gregs else False

            a = ([regs1[k], get_name(regs1[k]),
                 cosregs[regs1[k]],
                 get_name(cosregs[regs1[k]]), nclus_nins[k,1],
                 nclus_nins[k,0], nclus_nins[k,2], cano] + 
                 list(res2[regs1[k]].values()))
                      
            r.append(a)          
                      
        df  = pd.DataFrame(data=r,columns=columns)
        df  = df.reindex(index=df.index[::-1])       
        df.to_csv(meta_pth / 'region_info.csv')
                 
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


        
'''
#####
SI figures for single-cell
#####
'''


def load_single_cell():
    '''
    # Mann Whitney (single_cell)
    '''
    
    d = {}   
    mw = pd.read_csv(Path(sc_pth,  
            'Updated_Single_cell_analysis_July_10_2024 - Sheet1.csv'))
            
    vas = ['correct_vs_baseline', 'incorrect_vs_baseline', 'task']        

    for vari in vas:
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

    return d



def swansons_SI(vari):

    '''
    SI figure swanson for single-cell results
    vari in ['correct_vs_baseline', 'incorrect_vs_baseline', 'task']
    ''' 

#    cmaps = {'correct_vs_baseline': get_cmap_('fback'),
#             'incorrect_vs_baseline':get_cmap_('fback'),
#             'task': get_cmap_('block')}
             
    cmap = get_cmap_('fback')

    lw = 0.1  

    fig, ax = plt.subplots(figsize=(4.73, 2.38))  

    res = load_single_cell()[vari]
    ana = 'mannwhitney'
          
    acronyms = res[res[f'{ana}_significant'] == True]['region'].values
    scores = res[res[
             f'{ana}_significant'] == True][f'{ana}_effect'].values
    
    # turn fraction into percentage
    scores = scores * 100
                
    # remove regs from mask with nan amps (not analyzed)
    mask = res[np.bitwise_and(
                res[f'{ana}_significant'] == False,
                ~np.isnan(res[f'{ana}_{dt}']))][
                'region'].values
        
    if vari == 'task':
        vmin, vmax = (np.min(scores), np.max(scores))
    else:
        vmin, vmax = (0, 100)
    
    plot_swanson_vector(acronyms,
                        scores,
                        hemisphere=None, 
                        orientation='landscape',
                        cmap=cmap,
                        br=br,
                        ax=ax,
                        empty_color="white",
                        linewidth=lw,
                        mask=mask,
                        mask_color='silver',
                        annotate= False,
                        vmin=vmin,
                        vmax=vmax)
                        
    ax.set_axis_off() 
    
                      
    clevels = (vmin, vmax)
               
    norm = mpl.colors.Normalize(vmin=clevels[0], vmax=clevels[1])
    cbar = fig.colorbar(
               mpl.cm.ScalarMappable(norm=norm,cmap=cmap),
               ax=ax,shrink=0.4,aspect=12,pad=.025,
               orientation="horizontal")
               
    ticks = np.round(np.linspace(cbar.vmin, cbar.vmax, num=3), 2)             
    cbar.set_ticks(ticks)           
               
    cbar.ax.tick_params(axis='both', which='major',
                        labelsize=f_size_s)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(size=2)
    cbar.ax.xaxis.set_tick_params(pad=5)
    #cbar.set_label(res_types[res_type][0], fontsize=f_size_s)
    cbar.ax.tick_params(labelsize=f_size_s)                        

    fig.tight_layout()    
    fig.savefig(Path(imgs_pth, 'si', f'mannwhitney_SI_{vari}.svg'))    


'''
#####
decoding 
#####
'''

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

    dec_d = {'stim': 'stimside', 'choice': 'choice',
             'fback': 'feedback'} 

    # indicate in file name constraint
    exx = '' if MIN_TRIALS == 0 else ('_' + str(MIN_TRIALS))
    
    for vari in variables:
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


def stim_dec_line(fig=None, ax=None):

    '''
    plot decoding extra panels for bwm main figure
    '''

    alone = False
    if not fig:
        alone = True
        fig, ax = plt.subplots(figsize=(2.232,1.488))

    session_file = ('stimside_e0928e11-2b86-4387'
                    '-a203-80c77fab5d52_VISp_merged_'
                    'probes_pseudo_ids_-1_100.pkl')

    d = pd.read_pickle(open(Path(dec_pth, session_file), 'rb'))

    # base fit
    fit = d['fit'][0]

    # get average predictions from all runs
    run_idxs = np.unique([i for i, fit in enumerate(d['fit']) 
        if fit['pseudo_id'] == -1])
    preds_all = np.concatenate(
        [np.squeeze(d['fit'][i]['predictions_test'])[:, None] 
        for i in run_idxs], axis=1)
    preds = np.mean(preds_all, axis=1)
    # get decoding target (binary)
    targs = np.squeeze(fit['target'])
    # get signed stimulus contrast (categorical)
    targ_conts = (np.nan_to_num(fit['trials_df'].contrastLeft) 
        -np.nan_to_num(fit['trials_df'].contrastRight))
    # get good trials
    mask = fit['mask'][0]

    # compute neurometric curve
    targ_conts = targ_conts[mask]
    u_conts = np.unique(targ_conts)
    neurometric_curve = 1 - np.array([np.mean(preds[targ_conts==c]) 
                        for c in u_conts])
    neurometric_curve_err = np.array(
        [2 * np.std(preds[targ_conts==c]) / 
        np.sqrt(np.sum(targ_conts==c)) for c in u_conts])

    ax.set_title(f"{d['region'][0]}, single session")
    ax.plot(-u_conts, neurometric_curve, lw=2, c='k')
    ax.plot(-u_conts, neurometric_curve, 'ko', ms=4)
    ax.errorbar(-u_conts, neurometric_curve, neurometric_curve_err, color='k')
    ax.set_ylim(0,1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim(-1.03,1.03)
    ax.set_xticks([-1.0, -0.25, -0.125, -0.0625, 
                   0, 0.0625, 0.125, 0.25, 1.0])

    ax.set_xticklabels([-1] + ['']*7 + [1])             
    ax.set_xlabel('Stimulus contrast')
    ax.set_ylabel('predicted \n P(stim = right)')

    ax.spines[['top','right']].set_visible(False)
    
    if alone:
        fig.tight_layout()  
        fig.savefig(Path(imgs_pth, 'stim', 
                         'stim_dec_line.svg')) 


def dec_scatter(variable,fig=None, ax=None):

    '''
    plot decoding scatter for
    variable in [choice, fback]
    '''   
               
    red = (255/255, 48/255, 23/255)
    blue = (34/255,77/255,169/255)
    
    alone = False
    if not fig:
        alone = True
        fig, ax = plt.subplots(figsize=(2.232,1.488))        

    if variable == 'choice':
        session_file = (f'choice_671c7ea7-6726-4fbe-adeb'
                        '-f89c2c8e489b_GRN_merged_probes_'
                        'pseudo_ids_-1_100.pkl')
    elif variable == 'fback':
        session_file = (f'feedback_e012d3e3'
                        '-fdbc-4661-9ffa-5fa284e4e706_IRN_'
                        'merged_probes_pseudo_ids_-1_100.pkl')
    else:
        print('wrong variable')
        return
        
    d = pd.read_pickle(open(Path(dec_pth, session_file), 'rb'))
    # base fit
    fit = d['fit'][0]

    # get average predictions from all runs
    run_idxs = np.unique([i for i, fit in enumerate(d['fit']) 
                          if fit['pseudo_id'] == -1])
    preds_all = np.concatenate(
        [np.squeeze(d['fit'][i]['predictions_test'])[:, None] 
        for i in run_idxs], axis=1)

    preds = np.mean(preds_all, axis=1)
    # get decoding target (binary)
    targs = np.squeeze(fit['target'])
    # get good trials
    mask = fit['mask'][0]
    trials = np.arange(len(mask))[[m==1 for m in mask]]

    ax.set_title(f"{d['region'][0]}, single session", fontsize=f_size)

    ax.plot(
        trials[targs==-1], 
        preds[targs==-1] if variable=='feedback' else 1-preds[targs==-1],
        'o', c=red, lw=2, ms=2,
    )

    ax.plot(
        trials[targs==1], 
        preds[targs==1] if variable=='feedback' else 1-preds[targs==1],
        'o', c=blue, lw=2, ms=2,
    )

    if variable == 'fback':
        l = ['Incorrect', 'Correct']
    elif variable == 'choice':
        l = ['Right choice', 'Left choice']

    ax.legend(l,frameon=False, fontsize=f_size_s)  
    ax.set_yticks([0, 0.5, 1])

    ax.set_xlim(100,400)

    ax.set_xlabel('Trials')
    target = l[0] if variable != 'feedback' else l[1]
    ax.set_ylabel(f'Predicted \n {target}')
    ax.spines[['top','right']].set_visible(False)

    if alone:        
        fig.tight_layout()  
        fig.savefig(Path(imgs_pth, variable, 
                         'dec_scatter.svg')) 


def wheel_decoding_ex(vari, fig=None, axs=None):

    '''
    for vari in speed, velocity
    show example trials for decoding
    '''
    
    mpl.rcParams.update({'font.size': f_size_s})
    
    variable = f'wheel-{vari}'
    n_pseudo = 2 if variable == 'wheel-velocity' else 4
    session_file = (f'{variable}_671c7ea7-6726-4fbe-adeb'
                    f'-f89c2c8e489b_GRN_merged_probes_pseudo_ids_-1_'
                    f'{n_pseudo}.pkl')
    d = pd.read_pickle(open(Path(dec_pth, session_file), 'rb'))
    trials = [113, 216]

    if variable == 'wheel-speed':
        ymin, ymax = -0.9, 8.75
    else:
        ymin, ymax = -8.5, 8.5    
        
    # base fit
    fit = d['fit'][0]

    # get average predictions from all runs
    run_idxs = np.unique([i for i, fit in enumerate(d['fit']) 
        if fit['pseudo_id'] == -1])
    preds = []
    n_preds = len(fit['predictions_test'])
    for n in range(n_preds):
        preds_tmp = np.concatenate(
            [np.squeeze(d['fit'][i]['predictions_test'][n])[:, None] 
                for i in run_idxs], 
            axis=1,)
        preds.append(np.mean(preds_tmp, axis=1))
    # get decoding target
    targs = fit['target']
    # get good trials
    mask = fit['mask'][0]
    trial_idxs = np.where(mask)[0]
    # get some other metadata
    eid = d['eid']
    region = d['region']
    r2 = np.mean([d['fit'][i]['scores_test_full'] for i in range(2)])

    # build plot    
    alone = False
    if not fig:
        alone = True
        fig, axs = plt.subplots(1, len(trials), figsize=(4.32, 1.63))
        print(f"session: {eid} \n region: {region}")
        print(f" \n $R^2$ = {r2:.3f} (average across 2 models)")

    movetime = 0.2

    for i, (t, ax) in enumerate(zip(trials, axs)):
        targs_curr, preds_curr, trial_curr = targs[t], preds[t], trial_idxs[t]
        ax.plot((np.arange(len(targs_curr))-10)*0.02 + movetime, 
            targs_curr, 'k', lw=2)
        ax.plot((np.arange(len(targs_curr))-10)*0.02 + movetime, 
            preds_curr, 'r', lw=2)
        ax.plot(np.zeros(50) + movetime, np.linspace(ymin, ymax), 'k--', lw=1)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(f'{variable} (rad./s)')
        ax.set_xticks([0, .20, 0.5, 1.0], ['-0.2', '0.0', '0.5', '1.0'])
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Trial {trial_curr}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i == 1:
            ax.text(0.22, 0.8, 'Movement onset', transform=ax.transAxes)

    #fig.text(.5, 0.04, 'Time (s)', ha='center')
    
    if alone:
        fig.legend([f'Actual', 
            f'Predicted'], frameon=False)
    plt.tight_layout()
    fig.savefig(Path(imgs_pth, vari, 
                     f'wheel_decoding_ex.svg'),
                     bbox_inches='tight')
                     
                     

def plot_SI_speed_velocity():

    '''
    SI figure comparing wheel velocity/speed decoding
    '''
    
    MIN_UNITS = 5
    MIN_TRIALS = 250
    MIN_SESSIONS_PER_REGION = 2    
    
    targets = ['wheel-velocity', 'wheel-speed']
    results = {target: {} for target in targets}
    for target in targets: #, 'wheel-speed']:
        
        # load results combined across sessions for each region
        file_stage3_results = os.path.join(dec_pth, f'{target}_stage3.pqt')
        res_table_final = pd.read_parquet(file_stage3_results)

        # only get regions from final results table
        regions = res_table_final[
            res_table_final.n_sessions >= MIN_SESSIONS_PER_REGION
        ]['region'].values
        
        results[target]['vals'] = res_table_final
        results[target]['regions'] = regions
        
    # load raw targets
    dfs = {}
    for target in targets:
        imposter_file = os.path.join(dec_pth, 
            f'imposterSessions_{target}.pqt')
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

    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3))

    _, pal = get_allen_info()

    
    
    # -----------------------
    # metrics
    # -----------------------
    for ax, metric in zip(axes[:len(metrics_to_plot)], metrics_to_plot.keys()):
        # plot values
        
        cols = [pal[reg] for reg in results['wheel-velocity']['vals'].region]
        ax.scatter(
            results['wheel-velocity']['vals'][metric], 
            results['wheel-speed']['vals'][metric], s=1, color=cols,
        )
        # plot diagonal line
        xs = np.linspace(0, 0.55)
        if metric != 'null_median_of_medians':
            ax.plot(xs, xs, 'k--')
        # plot text
        for x, y, s in zip(
            results['wheel-velocity']['vals'][metric],
            results['wheel-speed']['vals'][metric],
            results['wheel-velocity']['vals'].region,
        ):
            if np.isnan(x) or np.isnan(y):
                continue
            ax.text(x, y, s, fontsize=f_size_s, color=pal[s])

        ax.set_xlabel(f'Wheel-velocity ({metrics_to_plot[metric]})')
        ax.set_ylabel(f'Wheel-speed ({metrics_to_plot[metric]})')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


    # -----------------------
    # target shapes
    # -----------------------
    ts = np.arange(60) * 0.020 - 0.2
    axes[-1].plot(ts, np.median(dfs['wheel-speed'], axis=0), lw=2)
    axes[-1].plot(ts, np.median(dfs['wheel-velocity'], axis=0), lw=2)
    axes[-1].fill_between(
        ts, 
        np.percentile(dfs['wheel-speed'], 5, axis=0),
        np.percentile(dfs['wheel-speed'], 95, axis=0), 
        alpha=0.2, color='C0',
    )
    axes[-1].fill_between(
        ts, 
        np.percentile(dfs['wheel-velocity'], 5, axis=0),
        np.percentile(dfs['wheel-velocity'], 95, axis=0), 
        alpha=0.2, color='C1',
    )
    axes[-1].legend(['wheel-speed', 'wheel-velocity'], fontsize=f_size)
    axes[-1].set_xlabel('Time from movement onset (s)')
    axes[-1].set_ylabel('Within trial speed/velocity')
    axes[-1].spines['top'].set_visible(False)
    axes[-1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

    fig.savefig(Path(imgs_pth, 'si', 
                     f'n6_supp_figure_decoding_wheelspeedvsvel.pdf'),
                     dpi=250,
                     bbox_inches='tight')

'''
##########
encoding (glm)
##########
'''

# Please use the saved parameters dict from 02_fit_sessions.py as params
glm_params = pd.read_pickle(enc_pth /"glm_params.pkl")


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
    ax = None):

    # Load in data and fit model to particular cluster
    (stdf, sspkt, sspkclu, 
        design, spkmask, nglm) = load_unit_fit_model(eid, pid, clu_id)
    # Construct GLM prediction object that does our model predictions

    pred = GLMPredictor(stdf, nglm, sspkt, sspkclu)
  
    # Construct design matrix without regressors of interest
    noreg_dm = remove_regressors(design, regressors)
    # Fit model without regressors of interest
    nrnglm = lm.LinearGLM(
        noreg_dm, sspkt[spkmask], sspkclu[spkmask],
         estimator=glm_params["estimator"],
         mintrials=0)
         
    nrnglm.fit()
    # Construct GLM prediction object that does model predictions without regressors of interest
    nrpred = GLMPredictor(stdf, nrnglm, sspkt, sspkclu)

    # Compute model predictions for each condition
    keyset1 = pred.compute_model_psth(
        align_time,
        t_before,
        t_after,
        trials=stdf[aligncond1(stdf[aligncol])].index,
    )
    cond1pred = pred.full_psths[keyset1][clu_id][0]
    keyset2 = pred.compute_model_psth(
        align_time,
        t_before,
        t_after,
        trials=stdf[aligncond2(stdf[aligncol])].index,
    )
    cond2pred = pred.full_psths[keyset2][clu_id][0]
    nrkeyset1 = nrpred.compute_model_psth(
        align_time,
        t_before,
        t_after,
        trials=stdf[aligncond1(stdf[aligncol])].index,
    )
    nrcond1pred = nrpred.full_psths[nrkeyset1][clu_id][0]
    nrkeyset2 = nrpred.compute_model_psth(
        align_time,
        t_before,
        t_after,
        trials=stdf[aligncond2(stdf[aligncol])].index,
    )
    nrcond2pred = nrpred.full_psths[nrkeyset2][clu_id][0]

    # Plot PSTH of original units and model predictions in both cases
    if not ax:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey="row")
        
    else:
        fig = plt.gcf()    
        
    x = np.arange(-t_before, t_after, nglm.binwidth)
    for rem_regressor in [False, True]:
        i = int(rem_regressor)
        oldticks = []
        peri_event_time_histogram(
            sspkt,
            sspkclu,
            stdf[aligncond1(stdf[aligncol])][align_time],
            clu_id,
            t_before,
            t_after,
            bin_size=nglm.binwidth,
            error_bars="sem",
            ax=ax[i],
            smoothing=0.01,
            pethline_kwargs={"color": "blue", "linewidth": 2},
            errbar_kwargs={"color": "blue", "alpha": 0.5},
        )
        oldticks.extend(ax[i].get_yticks())
        peri_event_time_histogram(
            sspkt,
            sspkclu,
            stdf[aligncond2(stdf[aligncol])][align_time],
            clu_id,
            t_before,
            t_after,
            bin_size=nglm.binwidth,
            error_bars="sem",
            ax=ax[i],
            smoothing=0.01,
            pethline_kwargs={"color": "red", "linewidth": 2},
            errbar_kwargs={"color": "red", "alpha": 0.5},
        )
        oldticks.extend(ax[i].get_yticks())
        pred1 = cond1pred if not rem_regressor else nrcond1pred
        pred2 = cond2pred if not rem_regressor else nrcond2pred
        ax[i].step(x, pred1, color="darkblue", linewidth=2)
        oldticks.extend(ax[i].get_yticks())
        ax[i].step(x, pred2, color="darkred", linewidth=2)
        oldticks.extend(ax[i].get_yticks())
        ax[i].set_ylim([0, np.max(oldticks) * 1.1])
    return fig, ax, sspkt, sspkclu, stdf


def load_unit_fit_model(eid, pid, clu_id):
    stdf, sspkt, sspkclu, _, __ = load_regressors(
        eid,
        pid,
        one,
        t_before=0.6,
        t_after=0.6,
        binwidth=glm_params["binwidth"],
        abswheel=True)

    design = generate_design(stdf, 
                             stdf["probabilityLeft"], 
                             t_before=0.6, **glm_params)
                             
    spkmask = sspkclu == clu_id
    nglm = lm.LinearGLM(design, 
                        sspkt[spkmask], 
                        sspkclu[spkmask],   
                        estimator=glm_params["estimator"], 
                        mintrials=0)
    nglm.fit()
    return stdf, sspkt, sspkclu, design, spkmask, nglm


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
            "fmoveL", #fmoveL
            "fmoveR", #fmoveR
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
            235,  # clu_id, was 235
            "VISp",  # region
            0.04540706,  # drsq (from 02_fit_sessions.py)
            "stimOn_times",  # Alignset key
        ),
        "choice": (
            "671c7ea7-6726-4fbe-adeb-f89c2c8e489b",
            "04c9890f-2276-4c20-854f-305ff5c9b6cf",
            143,
            "GRN",
            0.000992895,  # drsq
            "firstMovement_times",
        ),
        "fback": (
            "a7763417-e0d6-4f2a-aa55-e382fd9b5fb8",
            "57c5856a-c7bd-4d0f-87c6-37005b1484aa",
            83,
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
    
    
def ecoding_raster_lines(variable, clu_id0=None, axs=None,
                         frac_tr=3):    

    '''
    plot raster and two line plots
    ax = [ax_raster, ax_line0, ax_line1]
    '''
    if variable == 'fback':
        frac_tr=10
    
    
    alone = False
    if not axs:
        alone = True
        fig, axs = plt.subplots(nrows=3,ncols=1, 
                               figsize=(3.1,5.5), sharex=True,
                               gridspec_kw={'height_ratios': [2, 1, 1]})

    for k in range(len(axs)):
        axs[k].spines['top'].set_visible(False)
        axs[k].spines['right'].set_visible(False)

    print('alone:', alone)

    targetunits, alignsets, sortlookup = get_example_results()
    eid, pid, clu_id, region, drsq, aligntime = targetunits[variable]
  
    if clu_id0:
        clu_id = clu_id0

    (aligncol, aligncond1, aligncond2, 
        t_before, t_after, reg1, reg2) = alignsets[aligntime]
        
    _, _, sspkt, sspkclu, stdf = plot_twocond(
        eid,
        pid,
        clu_id,
        aligntime,
        aligncol,
        aligncond1,
        aligncond2,
        t_before,
        t_after,
        [reg1, reg2] if variable != "wheel" else ["wheel"],
        ax = [axs[1],axs[2]])
    
    axs[1].set_ylabel('Firing rate (Hz)')
    axs[2].set_ylabel('Firing rate (Hz)')
    #ax[2].set_ylabel('Time (s)')
    
    axs[1].yaxis.set_major_locator(plt.MaxNLocator(4))
    axs[2].yaxis.set_major_locator(plt.MaxNLocator(4))
        
    # Wheel only has one regressor, unlike all the others. 
    if variable != "wheel":
        remstr = f"\n[{reg1}, {reg2}] regressors rem."
    else:
        remstr = "\nwheel regressor rem."
    
        
    #names = [reg1, reg2, reg1 + remstr, reg2 + remstr]
    names = ['model with regressors', 'model without regressors']
    for subax, title in zip([axs[1],axs[2]], names):
        subax.set_title(title, fontsize=f_size)

    # custom legend
    all_lines = axs[1].get_lines()
    legend_labels = [reg2, reg1, 'model', 'model']
    axs[1].legend(all_lines, legend_labels, loc='upper right',
                 bbox_to_anchor=(1.2, 1.3), fontsize=f_size_s,
                 frameon=False)


    stdf["response_times"] = stdf["stimOn_times"]
    trial_idx, dividers = find_trial_ids(stdf, sort=sortlookup[variable])

    _, _ = single_cluster_raster(
        sspkt[sspkclu == clu_id],
        stdf[aligntime],
        trial_idx,
        dividers,
        ["b", "r"],
        [reg1, reg2],
        pre_time=t_before,
        post_time=t_after,
        raster_cbar=False,
        raster_bin=0.002,
        frac_tr = frac_tr,
        axs=axs[0])
        
    mpl.rcParams.update({'font.size': f_size})    
    ymin, ymax = axs[0].get_ylim()

        
    axs[0].axhline(y=dividers[0],c='k', linewidth=0.5)    
    axs[0].set_ylabel('Trials')
    
    #ax[0].set_xlabel('Time from event (s)')   
    axs[0].set_title("{} unit {} \n $\log \Delta R^2$ = {:.2f}".format(
                 region, clu_id, np.log10(drsq)), fontsize=f_size)    
    
    axs[0].sharex(axs[1])
    axs[1].set_xlabel(None)
                  
    if alone:
        fig.tight_layout()  
        fig.savefig(Path(imgs_pth, variable, 
                         'encoding_raster_lines.svg'))              
                 

def encoding_wheel_boxen(ax=None, fig=None):

    mpl.rcParams.update({'font.size': f_size_s})
    
    
    d = {}
    fs = {'speed': 'GLMs_wheel_speed.pkl',
          'velocity': 'GLMs_wheel_velocity.pkl'} 

    for v in fs:
        d[v] = pd.read_pickle(
            Path(enc_pth, fs[v]))['mean_fit_results']["wheel"].to_frame()

    joinwheel = d['speed'].join(d['velocity'], how="inner", 
                                rsuffix="_velocity", lsuffix="_speed")
    meltwheel = joinwheel.melt()
    
    alone = False
    if not ax:
        alone = True
        fig, ax = plt.subplots(constrained_layout=True, figsize=[3.42, 2.7 ])    
    
    ax = sns.boxenplot(data=meltwheel, y="value", x="variable",
                        hue="variable", dodge = False,
                        palette={'wheel_speed': sns.color_palette()[0], 
                                 'wheel_velocity': sns.color_palette()[1]})
                                 
    ax.set_ylim([-0.015, 0.04])    
    ax.set_ylabel(r'Distribution of population $\Delta R^2$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_legend().remove()
    
    if alone:
        fig.tight_layout()  
        fig.savefig(Path(imgs_pth, 'speed', 
                         'glm_boxen.svg')) 
                             

'''
##########
manifold
##########
'''

# canonical colors for left and right trial types
blue_left = [0.13850039, 0.41331206, 0.74052025]
red_right = [0.66080672, 0.21526712, 0.23069468]
ntravis = 30  # #trajectories for vis, first 2 real, rest pseudo

align = {'stim': 'stim on',
         'choice': 'motion on',
         'fback': 'feedback'}
         
ex_regs = {'stim_restr': 'VISp', 
       'choice_restr': 'GRN', 
       'fback_restr': 'IRN'}

def get_name(brainregion):
    regid = br.id[np.argwhere(br.acronym == brainregion)][0, 0]
    return br.name[np.argwhere(br.id == regid)[0, 0]]
    
def manifold_to_csv():

    '''
    reformat results for table
    '''
    
    mapping = 'Beryl'

    columns = ['region','name','nclus', 
               'p_euc_can', 'amp_euc_can',
               'lat_euc_can']

    for variable in variables:        
        r = []
        variable = variable + '_restr'
        d = np.load(Path(man_pth, f'{variable}.npy'),
                    allow_pickle=True).flat[0] 
        
        for reg in d:

            r.append([reg, get_name(reg), d[reg]['nclus'],
                      d[reg]['p_euc_can'],
                      d[reg]['amp_euc_can'],
                      d[reg]['lat_euc_can']])
                      
        df  = pd.DataFrame(data=r,columns=columns)        
        df.to_csv(Path(man_pth,f'{variable}.csv'))        

     
         
def pre_post(variable, can=False):
    '''
    [pre_time, post_time] relative to alignment event
    variable could be contr or restr variant, then
    use base window
    
    ca: If true, use canonical time windows
    '''

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
    '''
    color gradient for plotting trajectories
    c: color map type
    nobs: number of observations
    '''

    cmap = mpl.cm.get_cmap(c)

    return [cmap(fr * (nobs - p) / nobs) for p in range(nobs)]
    
    
def get_allen_info():
    r = np.load(Path(one.cache_dir, 'dmn', 'alleninfo.npy'),
                allow_pickle=True).flat[0]
                
   
#    cosmos_indices = np.unique(br.mappings['Cosmos'])
#    acronyms = br.acronym[cosmos_indices]
#    colors = br.rgb[cosmos_indices]           
                
                
    return r['dfa'], r['palette']

       
def plot_traj3d(variable, ga_pcs=False, curve='euc',
                       fig=None, ax=None):
                       
    '''
    using full window (not canonical!!) to see lick oscillation
    and auditory response in IC at 0.5 sec after fback
    '''

                       
    if 'can' in curve:
        print('using canonical time windows')
        can = True
    else:
        can = False   

    df, palette = get_allen_info()
    palette['all'] = (0.32156863, 0.74901961, 0.01568627,1)
     
    if not fig:
        alone = True     
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(projection='3d')
   
    else:
        alone = False
                              
    reg = ex_regs[variable]
    
    # 3d trajectory plot
    if ga_pcs:
        dd = np.load(Path(man_pth, f'{variable}_grand_averages.npy'),
                    allow_pickle=True).flat[0]            
    else:
        d = np.load(Path(man_pth, f'{variable}.npy'),
                    allow_pickle=True).flat[0]

        # pick example region
        dd = d[reg]

    npcs, allnobs = dd['pcs'].shape
    nobs = allnobs // ntravis


    for j in range(ntravis):

        # 3d trajectory
        cs = dd['pcs'][:, nobs * j: nobs * (j + 1)].T
        nobs_ = nobs
        if can:
            # restric to canonical window
            nobs_ = d[reg][f'd_{curve}'].shape[0]
            cs = cs[:nobs_]
             

        if j == 0:
            col = grad('Blues_r', nobs_)
        elif j == 1:
            col = grad('Reds_r', nobs_)
        else:
            col = grad('Greys_r', nobs_)


        ax.plot(cs[:, 0], cs[:, 1], cs[:, 2],
                    color=col[len(col) // 2],
                    linewidth=5 if j in [0, 1] else 1, alpha=0.5)

        ax.scatter(cs[:, 0], cs[:, 1], cs[:, 2],
                       color=col,
                       edgecolors=col,
                       s=20 if j in [0, 1] else 1,
                       depthshade=False)
              
    ax.grid(False)
    ax.axis('off')
    
    if alone:
        ax.set_title(f"{variable}, {reg} {dd['nclus']}")    
        if '_' in variable:
            variable = variable.split('_')[0]
        fig.tight_layout()    
        fig.savefig(Path(imgs_pth, variable, 
                         'manifold_3d.svg'))     
    


def plot_curves_scatter(variable, ga_pcs=False, curve='euc',
                       fig=None, axs=None):

    '''
    for a given region, plot example line with control,
    more example lines without control, scatter amps
    '''

    lw = 1  # linewidth  
    df, palette = get_allen_info()
    palette['all'] = (0.32156863, 0.74901961, 0.01568627,1)
     
    if not fig:
        alone = True     
        fig, axs = plt.subplots(2, 2, figsize=(5.089, 4), 
                       gridspec_kw={'height_ratios': [1, 2]},
                       sharex=True)
        axs = axs.flatten()
        axs[1].axis('off')
    else:
        alone = False
        

    tops = {}
    regsa = []


    d = np.load(Path(man_pth, f'{variable}.npy'),
                allow_pickle=True).flat[0]

    maxs = np.array([d[x][f'amp_{curve}'] for x in d])
    acronyms = np.array(list(d.keys()))
    order = list(reversed(np.argsort(maxs)))
    maxs = maxs[order]
    acronyms = acronyms[order]

    tops[variable] = [acronyms,
                   [d[reg][f'p_{curve}'] for reg in acronyms], maxs]

    maxs = np.array([d[reg][f'amp_{curve}'] for reg in acronyms
                     if d[reg][f'p_{curve}'] < sigl])

    maxsf = [v for v in maxs if not (math.isinf(v) or math.isnan(v))]

    print(variable, curve)
    print(f'{len(maxsf)}/{len(d)} are significant')
    tops[variable + '_s'] = (f'{len(maxsf)}/{len(d)} = '
                         f'{np.round(len(maxsf)/len(d),2)}')
                         
                         
    regs_a = [tops[variable][0][j] for j in range(len(tops[variable][0]))
              if tops[variable][1][j] < sigl]

    regsa.append(list(d.keys()))
    print(regs_a)
    print(' ')


    print(variable, tops[variable + '_s'])


    reg = ex_regs[variable]
    
    
    if 'can' in curve:
        print('using canonical time windows')
        can = True
    else:
        can = False          
                       
    k = 0
    
    '''
    line plot for example region
    '''
    if reg != 'all':             
        if reg not in d:
            print(f'{reg} not in d:'
                   'revise example regions for line plots')
            return

    if any(np.isinf(d[reg][f'd_{curve}'].flatten())):
        print(f'inf in {curve} of {reg}')
        return
        
    print(variable, reg, 'p_euc_can: ', d[reg]['p_euc_can'])    

    xx = np.linspace(-pre_post(variable,can=can)[0],
                     pre_post(variable,can=can)[1],
                     len(d[reg][f'd_{curve}']))

    # plot pseudo curves
    yy_p = d[reg][f'd_{curve}_p']

    for c in yy_p:
        axs[k].plot(xx, c, linewidth=1,
                    color='Gray')

    # get curve
    yy = d[reg][f'd_{curve}']

    axs[k].plot(xx, yy, linewidth=lw,
                color=palette[reg],
                label=f"{reg} {d[reg]['nclus']}")

    # put region labels
    y = yy[-1]
    x = xx[-1]
    ss = ' ' + reg

    axs[k].text(x, y, ss, color=palette[reg], fontsize=f_size)
    axs[k].text(x, c[-1], ' control', color='Gray', fontsize=f_size)

    axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')

    if variable == 'choice':
        ha = 'left'
    else:
        ha = 'right'

    axs[k].spines['top'].set_visible(False)
    axs[k].spines['right'].set_visible(False)

    axs[k].set_ylabel('Distance (Hz)')
    axs[k].set_xlabel('Time (s)')
    
    '''
    line plot per 5 example regions per variable
    '''
    
    
    k+=2
    exs0 = {'stim': ['LGd','VISp', 'PRNc','VISam','IRN', 'VISl',
                     'VISpm', 'VM', 'MS','VISli'],


            'choice': ['PRNc', 'VISal','PRNr', 'LSr', 'SIM', 'APN',
                       'MRN', 'RT', 'LGd', 'GRN','MV','ORBm'],

            'fback': ['IRN', 'SSp-n', 'PRNr', 'IC', 'MV', 'AUDp',
                      'CENT3', 'SSp-ul', 'GPe']}

    # use same example regions for variant variables
    exs = exs0.copy()
    for variable0 in  exs0:
        if variable0 in variable:
            exs[variable] = exs0[variable0]


    d = np.load(Path(man_pth, f'{variable}.npy'),
                allow_pickle=True).flat[0]

    # example regions to illustrate line plots
    regs = exs[variable]

    texts = []
    for reg in regs:
        if reg not in d:
            print(f'{reg} not in d:'
                   'revise example regions for line plots')
            continue
    
        if any(np.isinf(d[reg][f'd_{curve}'])):
            print(f'inf in {curve} of {reg}')
            continue

        xx = np.linspace(-pre_post(variable,can=can)[0],
                         pre_post(variable,can=can)[1],
                         len(d[reg][f'd_{curve}']))

        # get units in Hz
        yy = d[reg][f'd_{curve}']

        axs[k].plot(xx, yy, linewidth=lw,
                    color=palette[reg],
                    label=f"{reg} {d[reg]['nclus']}")

        # put region labels
        y = yy[-1]
        x = xx[-1]
        ss = f"{reg} {d[reg]['nclus']}"

        texts.append(axs[k].text(x, y, ss,
                                 color=palette[reg],
                                 fontsize=f_size))


    axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')

    if 'choice' in variable:
        ha = 'left'
    else:
        ha = 'right'

    axs[k].spines['top'].set_visible(False)
    axs[k].spines['right'].set_visible(False)

    axs[k].set_ylabel('Distance (Hz)')
    axs[k].set_xlabel('Time (s)')
    
    k += 1

    '''
    scatter latency versus max amplitude for significant regions
    '''

    dsize = 13  # diamond marker size


    d = np.load(Path(man_pth, f'{variable}.npy'),
                allow_pickle=True).flat[0]

    acronyms = [tops[variable][0][j] for j 
                in range(len(tops[variable][0]))]
                
    ac_sig = np.array([True if tops[variable][1][j] < sigl
                       else False for
                       j in range(len(tops[variable][0]))])

    maxes = np.array([d[x][f'amp_{curve}'] for x in acronyms])
    lats = np.array([d[x][f'lat_{curve}'] for x in acronyms])
    cols = [palette[reg] for reg in acronyms]

    axs[k].errorbar(lats, maxes, yerr=None, fmt='None',
                    ecolor=cols, ls='None', elinewidth=0.5)

    # plot significant regions
    axs[k].scatter(np.array(lats)[ac_sig],
                   np.array(maxes)[ac_sig],
                   color=np.array(cols)[ac_sig],
                   marker='D', s=dsize)

    # plot insignificant regions
    axs[k].scatter(np.array(lats)[~ac_sig],
                   np.array(maxes)[~ac_sig],
                   color=np.array(cols)[~ac_sig],
                   marker='o', s=dsize / 10)

    texts = []
    for i in range(len(acronyms)):

        if ac_sig[i]:  # only decorate marker with label if sig
        
            reg = acronyms[i]
            
            if reg not in exs[variable]:
                continue
            

            texts.append(
                axs[k].annotate(
                    '  ' + reg,
                    (lats[i], maxes[i]),
                    fontsize=f_size,
                    color=palette[acronyms[i]],
                    arrowprops=None))
                    

    axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')

    ha = 'left'


    axs[k].text(0, 0.95, align[variable.split('_')[0]
                               if '_' in variable else variable],
                transform=axs[k].get_xaxis_transform(),
                horizontalalignment=ha, rotation=90,
                fontsize=f_size_s)

    axs[k].spines['top'].set_visible(False)
    axs[k].spines['right'].set_visible(False)

    axs[k].set_ylabel('Max dist. (Hz)')
    axs[k].set_xlabel('Latency (s)')
    axs[k].sharey(axs[k-1])

    if alone:    
        if '_' in variable:
            variable = variable.split('_')[0]
        fig.tight_layout()    
        fig.savefig(Path(imgs_pth, variable, 
                         'manifold_lines_scatter.svg')) 


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



def main_fig(variable, clu_id0=None, save_pans=False):

    '''
    combine panels into main figure;
    variable in ['stim', 'choice', 'fback'] 
    using mosaic grid of 8 rows and 12 columns
    
    save_pans: save individual panels as svg
    '''
    
    if not save_pans:
    
        plt.ion()
 
        nrows = 16
        ncols = 15
        
        fig = plt.figure(figsize=(9, 9.77), facecolor='w', 
                         clear=True)
                                             
        gs = GridSpec(nrows=nrows, ncols=ncols, figure=fig)
        
        # for each panel type, get 
        # [row_start, row_end, col_start, col_end]
        gsd = {'dec_eff': [0, 4, 0, 3],
               'man_eff': [0, 4, 3, 6],
               'euc_eff': [0, 4, 6,9],
               'euc_lat': [0, 4, 9, 12],
               'glm_eff': [0, 4, 12, 15],
               'tab0': [4, 6, 0, 13],
               'tab1': [6, 8, 0, 13],
               'ras': [8, 12, 0, 5],
               'dec': [8, 10, 5, 10],
               'tra_3d': [8, 12, 10, 15],
               'ex_d': [10, 12, 5, 10],
               'enc0': [12, 14, 0, 5],
               'ex_ds': [12, 16, 5, 10],
               'scat': [12, 16, 10, 15],
               'enc1': [14, 16, 0, 5]}

        def ax_str(x):
            if '3d' in x:
                return fig.add_subplot(gs[gsd[x][0]: gsd[x][1],
                                          gsd[x][2]: gsd[x][3]], 
                                          projection='3d', label=x)
            
            elif x == 'dummy':
                fig0 ,ax0 = plt.subplots()
                plt.close(fig0)
                return ax0
                                                      
            else:
                return fig.add_subplot(gs[gsd[x][0]: gsd[x][1],
                                          gsd[x][2]: gsd[x][3]], label=x)    
            
    
    else:
        plt.ioff() 
                        

    '''
    meta 
    '''
    
    # 4 Swansons
    if not save_pans:
        s = ['dec_eff', 'man_eff', 'euc_eff', 'euc_lat', 'glm_eff']
        plot_swansons(variable, fig=fig, 
            axs=[ax_str(x) for x in s])
                  
    else:
        plot_swansons(variable)
        
    
    # plot table, reading from png
    plot_table(variable)
      
    if not save_pans:
        plot_table(variable)
        for tt in [0,1]:
            pf = Path(imgs_pth, variable, f'table_{tt}.png')   
            im = Image.open(pf)                  
            
            ax_tab = ax_str(f'tab{tt}')
                                        
            ax_tab.imshow(im.rotate(90, expand=True),
                aspect='equal')                  
            ax_tab.axis('off')                                             
        
         

    '''
    manifold
    '''

    # manifold panels, line plot with more regions and scatter
    if not save_pans:
        plot_curves_scatter(variable+'_restr',fig=fig, 
                 axs=[ax_str(x) for x in 
                      ['ex_d', 'dummy', 'ex_ds', 'scat']]) 

        plot_traj3d(variable+'_restr',fig=fig, ax = ax_str('tra_3d'))          
    else:
        plot_curves_scatter(variable+'_restr')
        plot_traj3d(variable+'_restr')

    '''
    decoding
    '''
    
    # decoding panel
    if variable == 'stim':
        if not save_pans:
            stim_dec_line(fig=fig, ax=ax_str('dec'))
        else:
            stim_dec_line()    
    else:
        if not save_pans:   
            dec_scatter(variable,fig=fig, ax=ax_str('dec'))
        else:
            dec_scatter(variable)

    '''
    encoding
    '''
    

    # encoding panels
    if not save_pans:   
        ecoding_raster_lines(variable,clu_id0= clu_id0, 
                             axs=[ax_str(x) for x in 
                             ['ras', 'enc0', 'enc1']])
                         
    else:
        ecoding_raster_lines(variable, clu_id0=clu_id0)
    
    
    '''
    adjust layout
    '''
                                      
    if not save_pans: 
        # Manually set the layout parameters for a tight layout
        left, right, top, bottom = 0.05, 0.98, 0.97, 0.05  
        wspace, hspace = 0.9, 0.9  
        fig.subplots_adjust(left=left, right=right, 
                            top=top, bottom=bottom, 
                            wspace=wspace, hspace=hspace)
                            
        # manually reduce size as tight_layout fails
        axs = fig.get_axes()
        shrink = 0.7  # size reduction factor 
        pans = ['ex_d','ex_ds','scat','dec','ras','enc0','enc1']
        s = ['glm_eff', 'euc_lat', 'euc_eff', 'man_eff', 'dec_eff']
        
        lettered = dict([('dec_eff', 'a'),
                     ('man_eff', 'b'),
                     ('euc_eff', 'c'),
                     ('euc_lat', 'd'),
                     ('glm_eff', 'e'),
                     ('tab0', 'f'),
                     ('ex_d', 'k'),
                     ('ex_ds', 'l'),
                     ('scat', 'm'),
                     ('dec', 'i'),
                     ('ras', 'g'),
                     ('enc0', 'h'),
                     ('tra_3d','j')])        
        
        for ax in axs:
            if ax.get_label() in pans:
                bbox = ax.get_position()
                left, bottom, width, height = (bbox.x0, bbox.y0, 
                                               bbox.width, bbox.height)
                ax.set_position([left + (width - width*shrink) / 2 + 0.02, 
                                 bottom + (height - height*shrink) / 2, 
                                 width*shrink, height*shrink])
                
                
                title_text = ax.get_title()
                ax.set_title(title_text, fontsize=f_size)
                                 
            if ax.get_label() in s:
                title_text = ax.get_title()
                ax.set_title(title_text, fontsize=f_size)                 
        
            if ax.get_label() in lettered:
                put_panel_label(ax, lettered[ax.get_label()])    
                

#        fig.savefig(Path(imgs_pth, variable, 
#                         f'n5_main_figure_{variverb[variable]}_revised_raw.svg'),  
#                         bbox_inches='tight')
#        fig.savefig(Path(imgs_pth, variable, 
#                         f'n5_main_figure_{variverb[variable]}_revised_raw.pdf'),
#                         dpi=300,
#                         bbox_inches='tight')                         
#        fig.savefig(Path(imgs_pth, variable, 
#                         f'n5_main_figure_{variverb[variable]}_revised.png'),
#                         dpi=250,
#                         bbox_inches='tight')    
    
        #plt.close(fig)
        
  
        
def main_wheel(save_pans=False):


    '''
    combine panels into wheel figure;
    using grid of 8 rows and 12 columns
    
    save_pans: save individual panels as svg
    '''


    # for each panel type, get 
    # [row_start, row_end, col_start, col_end]
    gsd = {'dec_speed': [0, 5, 0, 4,'a'],
           'glm_speed': [0, 5, 4, 8,'b'],
           'dec_velocity': [0, 5, 8, 12,'c'],
           'glm_velocity': [0, 5, 12, 16,'d'],
           'tab_speed0': [5, 7, 0, 16,'e'],
           'tab_speed1': [7, 9, 0, 16,''],
           'tab_velocity0': [9, 11, 0, 16,'f'],
           'tab_velocity1': [11, 13, 0, 16,''],
           'dec_ex_speed0': [13, 16, 0, 4,'g'],
           'dec_ex_speed1': [13, 16, 4, 8,''],
           'dec_ex_velocity0': [16, 18, 0, 4,'h'],
           'dec_ex_velocity1': [16, 18, 4, 8,''],
           'glm_boxen': [12, 18, 8, 16,'i']}
    
    if not save_pans:
    
        plt.ion()
 
        nrows = 19
        ncols = 16
        
        fig = plt.figure(figsize=(9, 9.77), facecolor='w', 
                         clear=True)
                                             
        gs = GridSpec(nrows=nrows, ncols=ncols, figure=fig)

        def ax_str(x):
            if x == 'dummy':
                fig0 ,ax0 = plt.subplots()
                plt.close(fig0)
                return ax0
                                                      
            else:
                return fig.add_subplot(gs[gsd[x][0]: gsd[x][1],
                                          gsd[x][2]: gsd[x][3]], label=x)    
            
    
    else:
        plt.ioff()     
    # use  plot_wheel_swansons to have 4 swansons on top
    
    plot_wheel_swansons
    '''
    meta 
    '''
    
    # 4 Swansons
    if not save_pans:
        s = ['dec_speed', 'glm_speed', 'dec_velocity', 'glm_velocity']
        plot_wheel_swansons(fig=fig, axs=[ax_str(x) for x in s])
                  
    else:
        plot_wheel_swansons()
        
        
    for variable in ['speed', 'velocity']:    
        # plot table, reading from png
        plot_table(variable)
          
        if not save_pans:
            plot_table(variable)
            for tt in [0,1]:
                pf = Path(imgs_pth, variable, f'table_{tt}.png')   
                im = Image.open(pf)                  
                
                ax_tab = ax_str(f'tab_{variable}{tt}')
                                            
                ax_tab.imshow(im.rotate(90, expand=True),
                    aspect='equal')                  
                ax_tab.axis('off')
                                                      
    
    # decoding example trials 
    for vari in ['speed', 'velocity']: 
        if not save_pans: 
            axsw = [ax_str(s) for s in 
                    [f'dec_ex_{vari}0', f'dec_ex_{vari}1']]  
            wheel_decoding_ex(vari, fig=fig, 
                axs=axsw)    
                
        else:
            wheel_decoding_ex(vari)

    # glm boxen plot
    if not save_pans: 
        encoding_wheel_boxen(ax=ax_str('glm_boxen'), fig=fig)               
    else:       
        encoding_wheel_boxen()
                    
    if not save_pans:        
    
        # Manually set the layout parameters for a tight layout
        left, right, top, bottom = 0.08, 0.98, 0.95, 0.05  
        wspace, hspace = 0.9, 0.9  
        fig.subplots_adjust(left=left, right=right, 
                            top=top, bottom=bottom, 
                            wspace=wspace, hspace=hspace)
                            
        # manually reduce size as tight_layout fails
        axs = fig.get_axes()
        shrink = 0.7  # size reduction factor 
        pans = ['dec_ex_speed0', 'dec_ex_speed1', 'dec_ex_velocity0',
                'dec_ex_velocity1', 'glm_boxen']   
        #['dec_speed', 'glm_speed', 'dec_velocity', 'glm_velocity']
        s = ['glm_eff', 'euc_lat', 'euc_eff', 'man_eff', 'dec_eff']       

        
        for ax in axs:
            if ax.get_label() in pans:
                bbox = ax.get_position()
                left, bottom, width, height = (bbox.x0, bbox.y0, 
                                               bbox.width, bbox.height)
                ax.set_position([left + (width - width*shrink) / 2 + 0.02, 
                                 bottom + (height - height*shrink) / 2, 
                                 width*shrink, height*shrink])
                
                
                title_text = ax.get_title()
                ax.set_title(title_text, fontsize=f_size)
                                 
            if ax.get_label() in s:
                title_text = ax.get_title()
                ax.set_title(title_text, fontsize=f_size)                 

            if (ax.get_label() in gsd) and (gsd[ax.get_label()][-1] != ''):
                put_panel_label(ax, gsd[ax.get_label()][-1])
                       
#        fig.savefig(Path(imgs_pth, 'speed', 
#                         f'n5_main_figure_wheel_revised_raw.svg'),  
#                         bbox_inches='tight')
#        fig.savefig(Path(imgs_pth, 'speed', 
#                         f'n5_main_figure_wheel_revised_raw.pdf'),
#                         dpi=300,
#                         bbox_inches='tight')          



def ghostscript_compress_pdf(variable):

    '''
    Compress main figs (inkscape pdfs) or whole manuscript    
    '''

    if variable in variables:
        input_path = Path(imgs_pth, variable, 
                         f'n5_main_figure_{variverb[variable]}_revised_raw.pdf')
        output_path = Path(imgs_pth, variable, 
                         f'n5_main_figure_{variverb[variable]}_revised.pdf')
                         
    if variable == 'wheel':
        input_path = Path(imgs_pth, 'speed', 
                         f'n5_main_figure_wheel_revised_raw.pdf')
        output_path = Path(imgs_pth, 'speed', 
                         f'n5_main_figure_wheel_revised.pdf')
                         
    if variable == 'manuscript':
        input_path = Path('/home/mic/Brainwide_Map_Paper.pdf')
        output_path = Path('/home/mic/Brainwide_Map_Paper2.pdf')    
                         

    # Ghostscript command to compress PDF
    command = [
        'gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4',
        '-dPDFSETTINGS=/ebook', '-dNOPAUSE', '-dQUIET', '-dBATCH',
        f'-sOutputFile={output_path}', input_path
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print(f"PDF successfully compressed and saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")




