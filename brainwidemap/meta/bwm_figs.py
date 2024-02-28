import pandas as pd
import numpy as np
from pathlib import Path
import math, string
from collections import Counter
from functools import reduce

from one.api import ONE
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
from iblatlas.plots import plot_swanson_vector 
import ibllib

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from PIL import Image
import dataframe_image as dfi

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
See first function in code block of this script for each analysis typoe
for data format conversion.
'''


ba = AllenAtlas()
br = BrainRegions()
one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)


variables = ['stim', 'choice', 'fback']
 
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

f_size = 10  # font size
#mpl.rcParams['figure.autolayout']  = True
mpl.rcParams.update({'font.size': f_size})


def put_panel_label(ax, k):
    ax.annotate(string.ascii_lowercase[k], (-0.05, 1.15),
                xycoords='axes fraction',
                fontsize=f_size * 1.5, va='top',
                ha='right', weight='bold')



'''
#####
meta (Swansons and table)
#####
'''

def pool_results_across_analyses(return_raw=False):

    '''
    input are various csv files from 
    4 different analysis types ['glm','euc', 'mw', 'dec']
    variables ['stim', ' choice', 'fback']
    '''

    D = {}
    
    # encoding (glm)   
    d = {}
    for vari in variables:
        df = pd.read_csv(Path(enc_pth, f'{vari}.csv'))
        df.rename(columns = {'0': 'glm_effect'}, inplace=True)
        d[vari] = df

    D['glm'] = d
    print('intergated glm results')  
     
    # euclidean (euc) 
    d = {}
    
    for vari in variables:
        d[vari] = pd.read_csv(Path(man_pth / f'{vari}_restr.csv'))[[
                    'region','amp_euc_can', 'lat_euc','p_euc']]
    
        d[vari]['euclidean_significant'] = d[vari].p_euc.apply(
                                               lambda x: x<sigl)
                                               

        d[vari].rename(columns = {'amp_euc_can': 'euclidean_effect',
                                  'lat_euc': 'euclidean_latency'},
                                  inplace=True)
                                  
        d[vari] = d[vari][['region',
                           'euclidean_effect',
                           'euclidean_latency',
                           'euclidean_significant']]

    D['euclidean'] = d
    print('intergated manifold results')    
    
    # Mann Whitney (single_cell)
    d = {}   
    mw = pd.read_csv(Path(sc_pth,  
            'Single_cell_analysis_Feb_25_2024 - Sheet1.csv'))
    pd.set_option('future.no_silent_downcasting', True)
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
    
       
    # decoding (dec)
    d = {} 
    
    dec_d = {'stim': 'stimside_stage3', 'choice': 'choice_stage3',
             'fback': 'feedback_stage3'}
              
    for vari in variables:
    
        d[vari] = pd.read_parquet(Path(dec_pth,
                    f'{dec_d[vari]}.pqt'))[[
                    'region','valuesminusnull_median',
                    'sig_combined_corrected']].rename(columns = {
                    'valuesminusnull_median': 'decoding_effect',
                    'sig_combined_corrected': 'decoding_significant'})
                
        d[vari].dropna(axis=0,how='any',subset=['decoding_effect'])
        #d[vari].reset_index(inplace=True)
        
    D['decoding'] = d   
    print('intergated decoding results')
    if return_raw:
        return D
       
    # merge frames across analyses    
    for vari in variables:
        df_ = reduce(lambda left,right: 
                     pd.merge(left,right,on='region'), 
                     [D[ana][vari] for ana in D])    
        df_.to_pickle(meta_pth / f"{vari}.pkl")
        #df_.reset_index(inplace=True)
        
    print('pooled and saved results at')
    print(meta_pth)   



def load_meta_results(variable):
    ''' 
    Load meta results for Swanson and table
    
    variable: ['stim', ' choice', 'fback']
    '''
    
    pd.set_option('future.no_silent_downcasting', True)
    res = pd.read_pickle(meta_pth / f"{variable}.pkl")
    res.replace(to_replace=[True, False], 
                      value=[1, 0], inplace=True)

    # ## Apply logarithm to GLM results
    res.glm_effect = np.log10(
                res.glm_effect.clip(lower=1e-5))

    # Reorder columns to match ordering in Figure
    res = res[['region',
               'euclidean_latency',
               'euclidean_effect',
               'glm_effect',
               'mannwhitney_effect',
               'decoding_effect',
               'decoding_significant',
               'mannwhitney_significant',
               'euclidean_significant']]
    return res


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

    res = load_meta_results(variable) 

    lw = 0.1  # .01
    
    # Labels for colorbars
    labels = ['Abs. diff. ' + r'$\Delta R^2$',
              'Latency (s)',
              'Nrml. Eucl. dist.',
              'Frac. sig. cells',
              'Decoding. ' + r'$R^2$' ' over null']

    # results to plot in Swansons
    res_types = ['glm_effect', 
                 'euclidean_latency',
                 'euclidean_effect',
                 'mannwhitney_effect',
                 'decoding_effect']

    # reverse list (until we can flip Swansons
    res_types.reverse()
    labels.reverse()
     
    cmap = get_cmap_(variable)
    
    alone = False
    if not fig:
        fig = plt.figure(figsize=(8,3), layout='constrained')  
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

        if ana != 'glm':
            # check if there are p-values
            
            acronyms = res[res[f'{ana}_significant'] == True]['region'].values
            scores = res[res[
                     f'{ana}_significant'] == True][f'{ana}_effect'].values
                        
            mask = res[res[f'{ana}_significant'] == False]['region'].values
        
        else:
            acronyms = res['region'].values
            scores = res[f'{ana}_effect'].values
            mask = []
        
        eucb =False    
        
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
                            annotate= True if not eucb else False,
                            annotate_n=5,
                            annotate_order='bottom' if lat else 'top')

        clevels = (min(scores), max(scores))
                   
        norm = mpl.colors.Normalize(vmin=clevels[0], vmax=clevels[1])
        cbar = fig.colorbar(
                   mpl.cm.ScalarMappable(norm=norm,cmap=cmap.reversed() 
                   if lat else cmap),
                   ax=axs[k],shrink=0.75,aspect=12,pad=.025,
                   orientation="horizontal")
                   
        cbar.ax.tick_params(labelsize=5,rotation=90)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(size=1)
        cbar.ax.xaxis.set_tick_params(pad=5)
        cbar.set_label(labels[k], fontsize=f_size)
        
        axs[k].set_title(f'{len(scores)}/{len(scores) + len(mask)}')  
        axs[k].set_xticks([])
        axs[k].set_yticks([])
        axs[k].axis("off")
        
        axs[k].axes.invert_xaxis()
                        
        k += 1  

    if alone:
        fig.savefig(Path(imgs_pth, variable, 'swansons.svg'))          


def plot_table(variable):


    # # Plot comparison table
    res = load_meta_results(variable)
    cmap = get_cmap_(variable)
    
    # Normalize values in each amplitude column to interval [0,1]
    # assuming columns 1:5 are 
    #'euclidean_latency', 'euclidean_effect', 'glm_effect',
    #'mannwhitney_effect', 'decoding_effect'
    
    res.iloc[:,1:5] = (res.iloc[:,1:5] - res.iloc[:,1:5].min())/(
                      res.iloc[:,1:5].max()-res.iloc[:,1:5].min()) + 1e-4

    # ## Sum values in each row to use for sorting
    # The rows of the table are sorted by the sum of all effects 
    # across the row(excluding latency). 
    # Here we create a new column with this sum.

    res['sum']  = res[['decoding_effect',
                       'mannwhitney_effect',
                       'euclidean_effect',
                       'glm_effect']].apply(np.sum,axis=1)
                       
    res = res.reset_index()

    # ## Sort rows by 'sum' within each Cosmos region
    # The sorting by sum of effects is done within each Cosmos region. 
    # So here I add the cosmos acronym as a column, group and then sort 'sum'.

    res['cosmos'] = res.region.apply(lambda x : beryl_to_cosmos(x,br))
    res = res.groupby('cosmos').apply(
              lambda x: x.sort_values(['sum'], ascending=False))

    # ## Add hex values for Beryl regions
    # Here I add a column of hex values corresponding to beryl acronyms. 
    # This is used to color each row by region.

    res['beryl_hex'] = res.region.apply(swanson_to_beryl_hex,args=[br])    
    beryl_palette = dict(zip(res.region, res.beryl_hex))
    
    # Add dummy column to be colored according beryl rgb
    res['region_color'] = res.region 

    # ## Order columns according to panels in Figure

    res = res[['region','region_color', 
               'decoding_effect','mannwhitney_effect',
               'glm_effect','euclidean_effect',
               'decoding_significant',
               'mannwhitney_significant','euclidean_significant']]
            
    for rt in ['decoding', 'mannwhitney','euclidean']:
        res[f'{rt}_effect'] = res[f'{rt}_effect'
                              ] * res[f'{rt}_significant']
    
    res = res[['region',#'region_color',
               'glm_effect','euclidean_effect',
               'mannwhitney_effect','decoding_effect']]


    #return beryl_palette

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
        if x==0:
                color = 'silver'
        else:
            rgb = cmap(x)
            color =  ('#' + rgb_to_hex((int(255*rgb[0]),
                                        int(255*rgb[1]),
                                        int(255*rgb[2]))))
                         
        return 'background-color: ' + color


    def significance_formatting(x):
        if x==True:
            color = colors[-1]
            
        if x==False:
            color =  colors[0]
            
        if pd.isna(x):
            color='silver'
        return 'background-color: ' + color
        
        
    # Format table  
    def make_pretty(styler):
        
        styler.applymap(effect_formatting,
            subset=['decoding_effect','mannwhitney_effect',
                    'euclidean_effect','glm_effect'])
        styler.applymap(region_formatting,subset=['region'])
        styler.set_properties(subset=['decoding_effect',
                                      'euclidean_effect',
                                      'mannwhitney_effect',
                                      'glm_effect'], 
                                      **{'width': '16px'})
        styler.set_properties(subset=['region'], **{'width': '65px'})
        styler.set_properties(subset=['decoding_effect',
                                      'euclidean_effect',
                                      'mannwhitney_effect',
                                      'glm_effect'] , 
                                      **{'font-size': '0pt'})

        styler.set_properties(subset=['region'] , 
                              **{'font-size': '9pt'})
        styler.hide(axis="index")

       
        styler.set_table_styles([         
            {"selector": "tr", "props": "line-height: 11px"},
            {"selector": "td, th", 
                "props": "line-height: inherit; padding: 0 "},
                
            {"selector": "tbody td", 
                "props": [("border", "1px solid white")]},
            {'selector': 'thead', 
                'props': [('display', 'table-header-group')]},
            {'selector': 'th.col_heading', 
                        'props': [('writing-mode', 'vertical-rl')]}])
     
        return styler
 
    ## Plot table
    res0 = res.style.pipe(make_pretty)
      
    pf = Path(imgs_pth, variable)
    pf.mkdir(parents=True, exist_ok=True)
    dfi.export(res0, Path(pf,'table.png'), max_rows=-1, dpi = 200)



'''
#####
decoding 
#####
'''

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
        fig, ax = plt.subplots(figsize=(3,2))

    session_file = ('stim_e0928e11-2b86-4387'
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

    ax.set_title(f"{d['region']}, single session")
    ax.plot(-u_conts, neurometric_curve, lw=2, c='k')
    ax.plot(-u_conts, neurometric_curve, 'ko', ms=4)
    ax.errorbar(-u_conts, neurometric_curve, neurometric_curve_err, color='k')
    ax.set_ylim(0,1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlim(-1.03,1.03)
    ax.set_xticks([-1.0, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1.0])

    ax.set_xticklabels([-1] + ['']*7 + [1])             
    ax.set_xlabel('Stimulus contrast')
    ax.set_ylabel('predicted \n P(stim = right)')

    ax.spines[['top','right']].set_visible(False)
    
    if alone:
        fig.tight_layout()  
        fig.savefig(Path(imgs_pth, variable, 
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
        fig, ax = plt.subplots(figsize=(3,2))        

    if variable == 'choice':
        session_file = (f'{variable}_671c7ea7-6726-4fbe-adeb'
                        '-f89c2c8e489b_GRN_merged_probes_'
                        'pseudo_ids_-1_100.pkl')
    elif variable == 'fback':
        session_file = (f'{variable}_e012d3e3'
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

    ax.set_title(f"{d['region']}, single session")

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

    ax.legend(l,frameon=True)  
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



'''
##########
encoding (glm)
##########
'''

# Please use the saved parameters dict from 02_fit_sessions.py as params
glm_params = pd.read_pickle(enc_pth /"glm_params.pkl")


def glm_to_csv():

    '''
    encoding pkl to csv
    '''

    t = pd.read_pickle(
            Path(enc_pth,'2024-02-26_glm_fit.pkl'))['mean_fit_results']
                        
    res = [abs(t['stimonL'] - t['stimonR']),
           abs(t['fmoveR']  - t['fmoveL']),
           abs(t['correct']  - t['incorrect'])]

    d0 = dict(zip(variables,res))
    d = {i: d0[i].to_frame().reset_index() for i in d0}
    
    rr = t['region'].reset_index()
    acs = rr['region'].values
    
    for variable in d:
        df = pd.DataFrame(d[variable])
        df.drop(['eid', 'pid', 'clu_id'], axis=1, inplace=True)
        df['region'] = acs
        df = df.groupby(['region']).mean()               
        df.to_csv(Path(enc_pth,f'{variable}.csv'))



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
            "fmoveL",
            "fmoveR",
        ),
        "feedback_times": (
            "feedbackType",
            lambda f: f == 1,
            lambda f: f == -1,
            0.1,
            0.4,
            "correct",
            "incorrect",
        ),
    }

    # Which units we're going to use for plotting
    targetunits = {  # eid, pid, clu_id, region, drsq, alignset key
        "stim": (
            "e0928e11-2b86-4387-a203-80c77fab5d52",  # EID
            "799d899d-c398-4e81-abaf-1ef4b02d5475",  # PID
            209,  # clu_id
            "VISp",  # region
            0.04540706,  # drsq (from 02_fit_sessions.py)
            "stimOn_times",  # Alignset key
        ),
        "choice": (
            "671c7ea7-6726-4fbe-adeb-f89c2c8e489b",
            "04c9890f-2276-4c20-854f-305ff5c9b6cf",
            123,
            "GRN",
            0.000992895,  # drsq
            "firstMovement_times",
        ),
        "fback": (
            "a7763417-e0d6-4f2a-aa55-e382fd9b5fb8",
            "57c5856a-c7bd-4d0f-87c6-37005b1484aa",
            98,
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


def ecoding_raster_lines(variable, ax=None):    

    '''
    plot raster and two line plots
    ax = [ax_raster, ax_line0, ax_line1]
    '''
    
    alone = False
    if not ax:
        alone = True
        fig, ax = plt.subplots(nrows=3,ncols=1)

    targetunits, alignsets, sortlookup = get_example_results()
    eid, pid, clu_id, region, drsq, aligntime = targetunits[variable]

    (aligncol, aligncond1, aligncond2, 
        t_before, t_after, reg1, reg2) = alignsets[aligntime]
        
    _, axs, sspkt, sspkclu, stdf = plot_twocond(
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
        ax = [ax[1],ax[2]])
        
    # Wheel only has one regressor, unlike all the others. 
    if variable != "wheel":
        remstr = f"\n[{reg1}, {reg2}] regressors rem."
    else:
        remstr = "\nwheel regressor rem."
        
    names = [reg1, reg2, reg1 + remstr, reg2 + remstr]
    for subax, title in zip(axs, names):
        subax.set_title(title)

    stdf["response_times"] = stdf["stimOn_times"]
    trial_idx, dividers = find_trial_ids(stdf, sort=sortlookup[variable])
    
    _, ax = single_cluster_raster(
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
        axs=ax[0])
        
    ax.set_ylabel('Resorted trial index')
    ax.set_xlabel('Time from event (s)')   
    ax.set_title("{} unit {} : $\log \Delta R^2$ = {:.2f}".format(
                 region, clu_id, np.log(drsq)))
    
               
    if alone:
        fig.tight_layout()  
        fig.savefig(Path(imgs_pth, variable, 
                         'encoding_raster_lines.svg'))              
                 


'''
##########
manifold
##########
'''

def get_name(brainregion):
    regid = br.id[np.argwhere(br.acronym == brainregion)][0, 0]
    return br.name[np.argwhere(br.id == regid)[0, 0]]
    
def manifold_to_csv():

    '''
    reformat results for table
    '''
    
    mapping = 'Beryl'

    columns = ['region','name','nclus', 
               'p_var', 'amp_var', 'lat_var',
               'p_euc', 'amp_euc', 'lat_euc', 'amp_euc_can',
               'lat_euc_can', 'amp_eucn_can', 'lat_eucn_can']

    for variable in variables:        
        r = []
        variable = variable + '_restr'
        d = np.load(Path(one.cache_dir, 'bwm_res', 
                    'manifold', 'res', f'{variable}.npy'),
                    allow_pickle=True).flat[0] 
        
        for reg in d:

            r.append([reg, get_name(reg), d[reg]['nclus'],
                      d[reg]['p_var'], 
                      d[reg]['amp_var'],
                      d[reg]['lat_var'],                          
                      d[reg]['p_euc'],
                      d[reg]['amp_euc'], 
                      d[reg]['lat_euc'],
                      d[reg]['amp_euc_can'],
                      d[reg]['lat_euc_can'],
                      d[reg]['amp_eucn_can'],
                      d[reg]['lat_eucn_can']])
                      
        df  = pd.DataFrame(data=r,columns=columns)        
        df.to_csv(Path(man_pth,f'{variable}.csv'))        


# canonical colors for left and right trial types
blue_left = [0.13850039, 0.41331206, 0.74052025]
red_right = [0.66080672, 0.21526712, 0.23069468]
ntravis = 30  # #trajectories for vis, first 2 real, rest pseudo

align = {'stim': 'stim on',
         'choice': 'motion on',
         'fback': 'feedback'}
         
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
    return r['dfa'], r['palette']


def plot_all(variables=None, curve='euc', show_tra=False, axs=None,
             all_labs=False,ga_pcs=True, extra_3d=False, fig=None):
    '''
    main manifold figure:
    1. plot example 3D trajectories,
    2. plot lines for distance(t) (curve 'var' or 'euc')
       for select regions
    3. plot 2d scatter [amplitude, latency] of all regions

    sigl: significance level, default 0.01, p_min = 1/(nrand+1)
    ga_pcs: If true, plot 3d trajectories of all cells,
            else plot for a single region (first in exs list)

    all_labs: show all labels in scatters, else just examples

    '''
    if variables is None:
        variables = align

    # specify grid; scatter longer than other panels
    ncols = 12
    
    if not fig:
        alone = True  # stand alone figure
        axs = []
        if show_tra:
            fig = plt.figure(figsize=(20, 2.5*len(variables)))
            gs = fig.add_gridspec(len(variables), ncols)
        else:   
            fig = plt.figure(figsize=(4, 6), layout='constrained')
            gs = fig.add_gridspec(2, ncols)
    else:
        alone = False
        
        
    if 'can' in curve:
        print('using canonical time windows')
        can = True
    else:
        can = False    

    k = 0  # panel counter


    dsize = 13  # diamond marker size
    lw = 1  # linewidth        
    

    dfa, palette = get_allen_info()

    '''
    get significant regions
    '''
    tops = {}
    regsa = []

    for variable in variables:

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

    for variable in variables:
        print(variable, tops[variable + '_s'])

    #  get Cosmos parent region for yellow color adjustment
    regsa = np.unique(np.concatenate(regsa))
    cosregs_ = [
        dfa[dfa['id'] == int(dfa[dfa['acronym'] == reg][
            'structure_id_path'].values[0].split('/')[4])][
            'acronym'].values[0] for reg in regsa]

    cosregs = dict(zip(regsa, cosregs_))

    '''
    example regions per variable for embedded space and line plots
    
    first in list is used for pca illustration
    '''

    exs0 = {'stim': ['LGd','VISp', 'PRNc','VISam','IRN', 'VISl',
                     'VISpm', 'VM', 'MS','VISli'],


            'choice': ['PRNc', 'VISal','PRNr', 'LSr', 'SIM', 'APN',
                       'MRN', 'RT', 'LGd', 'GRN','MV','ORBm'],

            'fback': ['IRN', 'SSp-n', 'PRNr', 'IC', 'MV', 'AUDp',
                      'CENT3', 'SSp-ul', 'GPe']}

    # use same example regions for variant variables
    exs = exs0.copy()
    for variable in variables:
        for variable0 in  exs0:
            if variable0 in variable:
                exs[variable] = exs0[variable0]


    if show_tra:

        '''
        Trajectories for example regions in PCA embedded 3d space
        ''' 
            
        row = 0
        for variable in variables:

            if ga_pcs:
                dd = np.load(Path(man_pth, f'{variable}_grand_averages.npy'),
                            allow_pickle=True).flat[0]
            else:
                d = np.load(Path(man_pth, f'{variable}.npy'),
                            allow_pickle=True).flat[0]

                # pick example region
                reg = exs[variable][0]
                dd = d[reg]

            if extra_3d:
                axs.append(fig.add_subplot(gs[:,row*3: (row+1)*3],
                                           projection='3d'))        
            else:
                if alone:
                    axs.append(fig.add_subplot(gs[row, :3],
                               projection='3d'))            

            npcs, allnobs = dd['pcs'].shape
            nobs = allnobs // ntravis

            for j in range(ntravis):

                # 3d trajectory
                cs = dd['pcs'][:, nobs * j: nobs * (j + 1)].T

                if j == 0:
                    col = grad('Blues_r', nobs)
                elif j == 1:
                    col = grad('Reds_r', nobs)
                else:
                    col = grad('Greys_r', nobs)

                axs[k].plot(cs[:, 0], cs[:, 1], cs[:, 2],
                            color=col[len(col) // 2],
                            linewidth=5 if j in [0, 1] else 1, alpha=0.5)

                axs[k].scatter(cs[:, 0], cs[:, 1], cs[:, 2],
                               color=col,
                               edgecolors=col,
                               s=20 if j in [0, 1] else 1,
                               depthshade=False)

            if extra_3d:
                axs[k].set_title(variable.split('_')[0])    

            else:
                axs[k].set_title(f"{variable}, {reg} {d[reg]['nclus']}"
                                 if not ga_pcs else variable)
            axs[k].grid(False)
            axs[k].axis('off')

            if not extra_3d:
                put_panel_label(axs[k], k)

            k += 1
            row += 1
            
        if extra_3d:
            return
        
    '''
    line plot per 5 example regions per variable
    '''
    row = 0  # index

    for variable in variables:

        if show_tra:
            axs.append(fig.add_subplot(gs[row, 3:6]))
        else:
            if alone:
                axs.append(fig.add_subplot(gs[0, :]))


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

        if variable == 'choice':
            ha = 'left'
        else:
            ha = 'right'

        axs[k].spines['top'].set_visible(False)
        axs[k].spines['right'].set_visible(False)

        axs[k].set_ylabel('distance [Hz]')
        axs[k].set_xlabel('time [sec]')
        
        if show_tra:
            put_panel_label(axs[k], k)

        row += 1
        k += 1

    '''
    scatter latency versus max amplitude for significant regions
    '''

    row = 0  # row idx

    for variable in variables:

        if show_tra:
            axs.append(fig.add_subplot(gs[row, 6:]))
        else:
            if alone:
                axs.append(fig.add_subplot(gs[1,:]))    


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
                    if not all_labs: # restrict to example regions   
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
                    fontsize=f_size * 0.8)

        axs[k].spines['top'].set_visible(False)
        axs[k].spines['right'].set_visible(False)

        axs[k].set_ylabel('max dist. [Hz]')
        axs[k].set_xlabel('latency [sec]')
        
        
        
        if show_tra:
            put_panel_label(axs[k], k)
            axs[k].set_title(f"{tops[variable+'_s']} sig")


        row += 1
        k += 1

    if not show_tra:
        axs[-1].sharex(axs[-2])
        
    if alone:        
        if '_' in variable:
            variable = variable.split('_')[0]
        fig.tight_layout()    
        fig.savefig(Path(imgs_pth, variable, 'manifold_lines_scatter.svg'))          


def plot_traj_and_dist(variable, ga_pcs=False, curve='euc',
                       fig=None, axs=None):

    '''
    for a given region, plot 3d trajectory and 
    line plot below
    '''
    
    ex_regs = {'stim_restr': 'VISp', 
           'choice_restr': 'GRN', 
           'fback_restr': 'IRN'}
    
    reg = ex_regs[variable]
    
    
    if 'can' in curve:
        print('using canonical time windows')
        can = True
    else:
        can = False   

    df, palette = get_allen_info()
    palette['all'] = (0.32156863, 0.74901961, 0.01568627,1)
     
    if not fig:
        alone = True     
        fig = plt.figure(figsize=(3,3.79))
        gs = fig.add_gridspec(5, 1)
        axs = [] 
        axs.append(fig.add_subplot(gs[:4, 0],
                               projection='3d'))
        axs.append(fig.add_subplot(gs[4:,0]))
   
    else:
        alone = False
       
    k = 0 
      
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

        if j == 0:
            col = grad('Blues_r', nobs)
        elif j == 1:
            col = grad('Reds_r', nobs)
        else:
            col = grad('Greys_r', nobs)

        axs[k].plot(cs[:, 0], cs[:, 1], cs[:, 2],
                    color=col[len(col) // 2],
                    linewidth=5 if j in [0, 1] else 1, alpha=0.5)

        axs[k].scatter(cs[:, 0], cs[:, 1], cs[:, 2],
                       color=col,
                       edgecolors=col,
                       s=20 if j in [0, 1] else 1,
                       depthshade=False)

    if alone:
        axs[k].set_title(f"{variable}, {reg} {dd['nclus']}")
                         
    axs[k].grid(False)
    axs[k].axis('off')

    #put_panel_label(axs[k], k)

    k += 1

    # line plot
    if reg != 'all':             
        if reg not in d:
            print(f'{reg} not in d:'
                   'revise example regions for line plots')
            return

    if any(np.isinf(dd[f'd_{curve}'].flatten())):
        print(f'inf in {curve} of {reg}')
        return
        
    print(variable, reg, 'p_euc: ', dd['p_euc'])    

    xx = np.linspace(-pre_post(variable,can=can)[0],
                     pre_post(variable,can=can)[1],
                     len(dd[f'd_{curve}']))

    # plot pseudo curves
    yy_p = dd[f'd_{curve}_p']

    for c in yy_p:
        axs[k].plot(xx, c, linewidth=1,
                    color='Gray')

    # get curve
    yy = dd[f'd_{curve}']

    axs[k].plot(xx, yy, linewidth=2,
                color=palette[reg],
                label=f"{reg} {dd['nclus']}")

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

    axs[k].set_ylabel('distance [Hz]')
    axs[k].set_xlabel('time [sec]')

    if alone:    
        if '_' in variable:
            variable = variable.split('_')[0]
        fig.tight_layout()    
        fig.savefig(Path(imgs_pth, variable, 
                         'manifold_single_reg_3d_line.svg')) 


'''
###########
combine all panels
###########
'''
    

def main_fig(variable, save_pans=False):

    '''
    combine panels into main figure;
    variable in ['stim', 'choice', 'fback'] 
    using mosaic grid of 8 rows and 12 columns
    
    save_pans: save individual panels as svg
    '''
    
    if not save_pans:
    
        plt.ion()
        # create mosaic plot
    
        fig = plt.figure(figsize=(20, 13), facecolor='w', 
                         clear=True)
        
        # Swansons ('p0' placeholder for tight_layout to work)
        s = ['glm_eff', 'euc_lat', 'euc_eff', 'man_eff', 'dec_eff']
        s2 = [[x]*2 for x in s] + [['p0']*2]
        s3 = [[item for sublist in s2 for item in sublist]]*3
        
        # panels under swansons and table
        pe = [['ras']*4 + ['dec']*4 + ['tra_3d']*4,
              ['ras']*4 + ['ex_d']*4 + ['tra_3d']*4,
              ['enc0']*4 + ['ex_ds']*4 + ['scat']*4,
              ['enc1']*4 + ['ex_ds']*4 +['scat']*4]

         
        mosaic = s3 + [['tab']*12]*2 + pe
            
        axs = fig.subplot_mosaic(mosaic,
                                 per_subplot_kw={
                                 "tra_3d": {"projection": "3d"}})
                                 
        # put panel labels                         
        pans = Counter([item for sublist in 
                        mosaic for item in sublist])

                         
        del pans['p0']
        axs['p0'].axis('off')
        
        for k,pa in enumerate(pans):
            put_panel_label(axs[pa], k)
    
    else:
        plt.ioff() 
                        

    '''
    meta 
    '''
    
    # 4 Swansons
    if not save_pans:
        plot_swansons(variable, fig=fig, axs=[axs[x] for x in s])
    else:
        plot_swansons(variable)
        
    pf = Path(imgs_pth, variable, 'table.png')  
    # plot table, reading from png
    if not pf.is_file():
        plot_table(variable)

    if not save_pans:    
        im = Image.open(pf)
        axs['tab'].imshow(im.rotate(90, expand=True), aspect='auto')                  
        axs['tab'].axis('off')                     
         

    '''
    manifold
    '''

    # trajectory extra panels
    # manifold example region line plot 
    if not save_pans:            
        plot_traj_and_dist(variable+'_restr', 
                           fig = fig,
                           axs=[axs['tra_3d'],axs['ex_d']])
        axs['tra_3d'].axis('off')
    else:
        plot_traj_and_dist(variable+'_restr')

    # manifold panels, line plot with more regions and scatter
    if not save_pans:  
        plot_all(variables=[variable+'_restr'], fig=fig, 
                 axs=[axs['ex_ds'], axs['scat']]) 
    else:
        plot_all(variables=[variable+'_restr'])         
    

    '''
    decoding
    '''
    
    # decoding panel
    if variable == 'stim':
        if not save_pans:
            stim_dec_line(fig=fig, ax=axs['dec'])
        else:
            stim_dec_line()    
    else:
        if not save_pans:   
            dec_scatter(variable,fig=fig, ax=axs['dec'])
        else:
            dec_scatter(variable)

    '''
    encoding
    '''
    

    # encoding panels
    if not save_pans:   
        ecoding_raster_lines(variable, ax=[axs['ras'], 
                                        axs['enc0'],
                                        axs['enc1']])
        fig.tight_layout()
        fig.tight_layout()
        fig.savefig(Path(imgs_pth, variable, 
                         'main_fig.svg'))
        fig.savefig(Path(imgs_pth, variable, 
                         'main_fig.png'), dpi=250)                          
    else:
        ecoding_raster_lines(variable)
                                      

    
    
    
