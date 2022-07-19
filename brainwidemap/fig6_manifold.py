from one.api import ONE
from reproducible_ephys_processing import bin_spikes2D
from brainwidemap import bwm_query
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
from brainbox.io.one import SpikeSortingLoader

from ibllib.atlas.plots import prepare_lr_data
from ibllib.atlas.plots import plot_scalar_on_flatmap, plot_scalar_on_slice
from ibllib.atlas import FlatMap
from ibllib.atlas.flatmaps import plot_swanson

from scipy import optimize, signal, stats
import pandas as pd
import numpy as np
from collections import Counter, ChainMap
from sklearn.decomposition import PCA
import gc
from scipy.stats import percentileofscore, zscore
import umap
import os
from PIL import Image
from pathlib import Path
import glob
from dateutil import parser

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from adjustText import adjust_text
import random
import time
import matplotlib.image as mpimg
import math
import string

import cProfile
import pstats

import warnings
warnings.filterwarnings("ignore")


blue_left = [0.13850039, 0.41331206, 0.74052025]
red_right = [0.66080672, 0.21526712, 0.23069468]

c = 0.005  # 0.005 for a static bin size, or None for single bin

def T_BIN(split, c=c):

    # c = 0.005 # time bin size in seconds (5 ms)
    if c is None:
        return pre_post[split][0] + pre_post[split][1]
    else:
        return c    


align = {'block':'stim on',
         'stim':'stim on',
         'choice':'motion on',
         'action':'motion on',
         'fback':'feedback'}
         
align = {'stim':'stim on',
         'choice':'motion on',
         'fback':'feedback'}         
         

pre_post = {'choice':[0.1,0],'stim':[0,0.1],
            'fback':[0,0.1],'block':[0.4,0],
            'action':[0.025,0.3]}  #[pre_time, post_time], 
            #if single bin, not possible to have pre/post
            
trial_split = {'choice':['choice left', 'choice right'],
               'stim':[1.0,1.0],
               'fback':['correct','false'],
               'block':['pleft 0.8','pleft 0.2'],
               'action':['choice left', 'choice right']}            


nrand = 500  # number of random trial splits for control
min_reg = 100  # minimum number of neurons in pooled region
one = ONE() 
ba = AllenAtlas()
br = BrainRegions()


def xs(split):
    return np.arange((pre_post[split][0] + pre_post[split][1])/
                      T_BIN(split))*T_BIN(split)


def generate_pseudo_blocks(n_trials, factor=60, min_=20, max_=100, first5050=90):
    """
    Generate a pseudo block structure
    Parameters
    ----------
    n_trials : int
        how many trials to generate
    factor : int
        factor of the exponential
    min_ : int
        minimum number of trials per block
    max_ : int
        maximum number of trials per block
    first5050 : int
        amount of trials with 50/50 left right probability at the beginning
    Returns
    ---------
    probabilityLeft : 1D array
        array with probability left per trial
    """

    block_ids = []
    while len(block_ids) < n_trials:
        x = np.random.exponential(factor)
        while (x <= min_) | (x >= max_):
            x = np.random.exponential(factor)
        if (len(block_ids) == 0) & (np.random.randint(2) == 0):
            block_ids += [0.2] * int(x)
        elif (len(block_ids) == 0):
            block_ids += [0.8] * int(x)
        elif block_ids[-1] == 0.2:
            block_ids += [0.8] * int(x)
        elif block_ids[-1] == 0.8:
            block_ids += [0.2] * int(x)
    return np.array([0.5] * first5050 + block_ids[:n_trials - first5050])


def compute_impostor_behavior():

    '''
    for a given split, get the behavior of 
    5 random concatenated sessions to match length
    for block there is the pseudo session method
    
    eid_no is the one eid to exclude
    '''
    
    df = bwm_query(one)
    
    eids_plus = list(set(df['eid'].values))
    
    R = {}
    for split in align:
        d = {}
        for eid in eids_plus: 
            try:            
                trials = one.load_object(eid, 'trials', collection='alf')
                
                # discard trials were feedback - stim is above that [sec]
                toolong = 2  

                # Load in trials data
                trials = one.load_object(eid, 'trials', collection='alf')
                   
                # remove certain trials    
                stim_diff = trials['feedback_times'] - trials['stimOn_times']     
       
                rm_trials = np.bitwise_or.reduce([np.isnan(trials['stimOn_times']),
                                           np.isnan(trials['choice']),
                                           np.isnan(trials['feedback_times']),
                                           np.isnan(trials['probabilityLeft']),
                                           np.isnan(trials['firstMovement_times']),
                                           np.isnan(trials['feedbackType']),
                                           stim_diff > toolong])
                       
                trn = []

                if split in ['choice', 'action']:
                    for choice in [1,-1]:
                        trn.append(np.arange(len(trials['choice']))
                            [np.bitwise_and.reduce([
                            ~rm_trials,trials['choice'] == choice])])             

                elif split == 'stim':    
                    for side in ['Left', 'Right']:
                        trn.append(np.arange(len(trials['stimOn_times']))
                            [np.bitwise_and.reduce([ ~rm_trials,
                            trials[f'contrast{side}'] == 1.0])])
                   
                elif split == 'fback':    
                    for fb in [1,-1]:
                        trn.append(np.arange(len(trials['choice']))
                            [np.bitwise_and.reduce([
                            ~rm_trials,trials['feedbackType'] == fb])])
              
                elif split == 'block':
                    for pleft in [0.8, 0.2]:
                        trn.append(np.arange(len(trials['choice']))
                            [np.bitwise_and.reduce([
                             ~rm_trials,trials['probabilityLeft'] == pleft])])    

                d[eid] = trn
                print(eid, 'done')
            except:
                print(eid, 'faulty')
        R[split] = d

    np.save('/home/mic/paper-brain-wide-map/manifold_analysis/'
            'bwm_behave.npy',R,allow_pickle=True)    


def get_name(brainregion):
    regid = br.id[np.argwhere(br.acronym == brainregion)][0, 0]
    return br.name[np.argwhere(br.id == regid)[0, 0]]


def get_d_vars(split,eid, probe, 
               mapping='Swanson', control=False):

    '''
    for a given session, probe, bin neural activity
    cut into trials, compute d_var per region
    '''    
       
    toolong = 2  # discard trials were feedback - stim is above that [sec]

    # Load in spikesorting
    sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    # Find spikes that are from the clusterIDs
    spike_idx = np.isin(spikes['clusters'], clusters['cluster_id'])

    # Load in trials data
    trials = one.load_object(eid, 'trials', collection='alf')
       
    # remove certain trials    
    stim_diff = trials['feedback_times'] - trials['stimOn_times']     
    rm_trials = np.bitwise_or.reduce([np.isnan(trials['stimOn_times']),
                               np.isnan(trials['choice']),
                               np.isnan(trials['feedback_times']),
                               np.isnan(trials['probabilityLeft']),
                               np.isnan(trials['firstMovement_times']),
                               np.isnan(trials['feedbackType']),
                               stim_diff > toolong])
    events = []
    trn = []


    if split in ['choice', 'action']:
        for choice in [1,-1]:
            events.append(trials['firstMovement_times'][np.bitwise_and.reduce([
                       ~rm_trials,trials['choice'] == choice])])
            trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                       ~rm_trials,trials['choice'] == choice])])             

    elif split == 'stim':    
        for side in ['Left', 'Right']:
            events.append(trials['stimOn_times'][np.bitwise_and.reduce([
                ~rm_trials,trials[f'contrast{side}'] == 1.0])])
            trn.append(np.arange(len(trials['stimOn_times']))[np.bitwise_and.reduce([
                       ~rm_trials,trials[f'contrast{side}'] == 1.0])])
       
    elif split == 'fback':    
        for fb in [1,-1]:
            events.append(trials['feedback_times'][np.bitwise_and.reduce([
                ~rm_trials,trials['feedbackType'] == fb])])
            trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                       ~rm_trials,trials['feedbackType'] == fb])])
                       
    elif split == 'block':
        for pleft in [0.8, 0.2]:
            events.append(trials['stimOn_times'][np.bitwise_and.reduce([
                ~rm_trials,trials['probabilityLeft'] == pleft])])
            trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                       ~rm_trials,trials['probabilityLeft'] == pleft])])      

    else:
        print('what is the split?', split)
        return

    # bin and cut into trials    
    bins = []

    for event in events:
    
        #  overlapping time bins (assuming T_BIN = 0.005)
        #  that results in 3 ms off
        bis = []
        st = 4
        for ts in range(st):
    
            bi, _ = bin_spikes2D(spikes['times'][spike_idx], 
                               spikes['clusters'][spike_idx],
                               clusters['cluster_id'],
                               np.array(event) + ts*0.001, 
                               pre_post[split][0], pre_post[split][1], 
                               T_BIN(split))
            bis.append(bi)
            
        ntr, nn, nbin = bi.shape
        ar = np.zeros((ntr, nn, st*nbin))
        
        for ts in range(st):
            ar[:,:,ts::st] = bis[ts]
                           
        bins.append(ar)                   
                                              
    b = np.concatenate(bins)
    
    #  recreate temporal trial order              
    dx = np.concatenate([list(zip([True]*len(trn[0]),trn[0])),
                    list(zip([False]*len(trn[1]),trn[1]))])

    b = b[np.argsort(dx[:, 1])] 
    
           
    ntr, nclus, nbins = b.shape
    

    if control:
        # get mean and var across trials
        w0 = [bi.mean(axis=0) for bi in bins]  
        s0 = [bi.var(axis=0) for bi in bins]
        
        #  Load impostor behavior
        spl = np.load('/home/mic/paper-brain-wide-map/'
                      'manifold_analysis/bwm_behave.npy',
                      allow_pickle=True).flat[0][split]
        
        #  exclude current session
        del spl[eid]    
        
        # nrand times random impostor/pseudo split of trials 
        for i in range(nrand):
            if split == 'block':  # pseudo sessions
                ys = generate_pseudo_blocks(ntr, first5050=0) == 0.8
            else:  # impostor sessions
                eids = random.choices([*spl],k=30)
                bs = []
                for eid in eids:
                    t = spl[eid]
                    
                    # some sessions have empty behavior
                    if (len(t[0]) < 2) or (len(t[1]) < 2):
                        continue
                           
                    x = np.concatenate([list(zip([True]*len(t[0]),t[0])),
                                    list(zip([False]*len(t[1]),t[1]))])
                
                    bs.append(np.array(x[np.argsort(x[:, 1])][:,0],
                              dtype=bool))
                              
                ys = np.concatenate(bs)[:ntr]              

            w0.append(b[ys].mean(axis=0))
            s0.append(b[ys].var(axis=0))
            
            w0.append(b[~ys].mean(axis=0))
            s0.append(b[~ys].var(axis=0))                      

    else: # average all trials
        print('all trials')
        w0 = [bi.mean(axis=0) for bi in bins] 
        s0 = [bi.var(axis=0) for bi in bins]


    ws = np.array(w0)
    ss = np.array(s0)
    acs = br.id2acronym(clusters['atlas_id'],mapping=mapping)            
    acs = np.array(acs) 
    
    regs = Counter(acs)

    D = {}
    
    for reg in regs:
        if reg in ['void','root']:
            continue

        res = {}

        ws_ = [y[acs == reg] for y in ws]
        ss_ = [y[acs == reg] for y in ss]
     
        wsc = np.concatenate(ws_,axis=1)

        #Discard cells that have nan entries in the PETH or PETH = 0 for all t
        respcells = [k for k in range(wsc.shape[0]) if 
                     (not np.isnan(wsc[k]).any()
                     and wsc[k].any())] 
                        
        ws_ = [x[respcells] for x in ws_]
        ss_ = [x[respcells] for x in ss_]        

        res['nclus'] = len(respcells)
        
        d_vars = []
                
        for j in range(len(ws_)//2):


            # strictly standardized mean difference
            d_var = (((ws_[2*j] - ws_[2*j + 1])/
                      ((ss_[2*j] + ss_[2*j + 1])**0.5))**2)

            # sum over cells, divide by #neu later
            d_var_m = np.nansum(d_var,axis=0)
            d_vars.append(d_var_m)

        res['d_vars'] = d_vars

        D[reg] = res
        
    return D    

      
'''    
###
### bulk processing 
###    
'''  


def download_data_only(eids_plus = None):
    if eids_plus is None:
        df = bwm_query(one)
        eids_plus = df[['eid','probe_name']].values
    Fs = []
   
    k=0
    for pid in eids_plus:
        eid, probe = pid

        try:        
            sl = SpikeSortingLoader(eid=eid, pname=probe, 
                                    one=one, atlas=ba)
            spikes, clusters, channels = sl.load_spike_sorting()    
            gc.collect() 
            print(k, 'of', len(eids_plus), 'ok') 
        except:
            Fs.append(pid)
            gc.collect()
            print(k, 'of', len(eids_plus), 'fail')
        k+=1          
    return Fs        
        


def get_all_d_vars(split,eids_plus = None, control = True, 
                   mapping='Swanson'):

    '''
    for all BWM insertions, get the PSTHs and acronyms,
    '''
    
    print('split', split)
    
    if eids_plus == None:
        df = bwm_query(one)
        eids_plus = df[['eid','probe_name']].values
        #eids_plus = get_bwm_sessions()

    Fs = []
    eid_probe = []
    Ds = []   
    k=0
    for pid in eids_plus:
        eid, probe = pid
        try:
        
            D = get_d_vars(split,eid, probe, 
                           control=control, mapping=mapping)
            Ds.append(D)                               
            eid_probe.append(eid+'_'+probe)
            gc.collect() 
            print(k+1, 'of', len(eids_plus), 'ok') 
        except:
            Fs.append(pid)
            gc.collect()
            print(k+1, 'of', len(eids_plus), 'fail')
        k+=1            
        
    R = {'Ds':Ds, 'eid_probe':eid_probe} 
    
    np.save('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/manifold/'
           f'd_vars_{split}_{mapping}.npy', R, allow_pickle=True) 

    print(f'{len(Fs)}, load failures:')
    return Fs

    
def d_var_stacked(split, mapping='Swanson'):
                  

    time0 = time.perf_counter()

    '''
    average d_var_m via nanmean across insertions,
    remove regions with too few neurons (min_reg)
    compute maxes, latencies and p-values
    '''
  
    print(split)
    
    R = np.load('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/manifold/'
                f'd_vars_{split}_{mapping}.npy', allow_pickle=True).flat[0]

    # pooling of insertions per region, discard low-neuron-number regs
    regs = np.concatenate([list(x.keys()) for x in R['Ds']])  
    regd = {reg:[] for reg in Counter(regs)}
    
    for D in R['Ds']:
        for reg in D:
            regd[reg].append(D[reg]['nclus'])
            
    regs = [x for x in regd if sum(regd[x]) > min_reg]
    nclus = {reg:sum(regd[reg]) for reg in regs}
    
    print(f'pre min_reg filter: {len(regd)}; post: {len(regs)}')

    regd = {reg:[] for reg in regs}
    for D in R['Ds']:
        for reg in D:
            if reg in regs:
                regd[reg].append(D[reg]['d_vars'])

    # nanmean across insertions and take sqrt
    for reg in regd:
        regd[reg] = (np.nansum(np.array(regd[reg]),axis=0)**0.5/
                     nclus[reg])


    r = {}
    for reg in regd:
        res = {}

        # nclus
        res['nclus'] = nclus[reg]
        
        # full curve
        d_var_m = regd[reg][0]
        res['d_var_m'] = d_var_m

        # maximum
        maxes = [np.max(x) for x in regd[reg]]
        d_var_max = maxes[0]
        res['max-min/max+min'] = ((np.max(d_var_m) - np.min(d_var_m))/
                                  (np.max(d_var_m) + np.min(d_var_m)))
        
        # p value
        null_d = maxes[1:]
        p = 1 - (0.01 * percentileofscore(null_d,d_var_max,kind='weak'))
        res['p'] = p
        
        # latency  
        if d_var_max == np.inf:
            loc = np.where(d_var_m == np.inf)[0]  
        else:
            loc = np.where(d_var_m > 0.95 * d_var_max)[0]
                            
        res['lat'] = (xs(split)-pre_post[split][0])[loc[0]]

        r[reg] = res


    np.save('/home/mic/paper-brain-wide-map/manifold_analysis/'
           f'curves_{split}_{mapping}.npy', 
           r, allow_pickle=True)
           
    time1 = time.perf_counter()
    
    print(time1 - time0, 'sec')


def curves_params_all(split):

    get_all_d_vars(split)
    d_var_stacked(split)        

      
'''    
###
### plotting 
###    
'''

def get_allen_info():
    dfa = pd.read_csv('/home/mic/paper-brain-wide-map/'
                       'allen_structure_tree.csv')
    
    # get colors per acronym and transfomr into RGB
    dfa['color_hex_triplet'] = dfa['color_hex_triplet'].fillna('FFFFFF')
    dfa['color_hex_triplet']  = dfa['color_hex_triplet'
                                    ].replace('19399','19399a')
    dfa['color_hex_triplet']  = dfa['color_hex_triplet'] .replace('0','FFFFFF')
    dfa['color_hex_triplet'] = '#' + dfa['color_hex_triplet'].astype(str)
    dfa['color_hex_triplet'] = dfa['color_hex_triplet'
                                ].apply(lambda x: 
                                mpl.colors.to_rgba(x))
                                
    palette = dict(zip(dfa.acronym, dfa.color_hex_triplet))
    
    return dfa, palette           


def put_panel_label(ax, n):                    
    ax.text(-0.1, 1.30,  string.ascii_lowercase[n], 
                  transform=ax.transAxes, 
                  fontsize=16, va='top', ha='right', weight='bold')


def plot_all(curve = 'd_var_m', curve_only=False):

    '''

    ''' 
    
    if curve_only:
        nrows = 1
    else:
        nrows = 5
        
    mapping='Swanson'
    
    fig = plt.figure(figsize=(15, 10))
    _, palette = get_allen_info()
    
    font = {'size'   : 8}
    mpl.rc('font', **font)                

    axs = []
    axsi = []

   
    k = 0

    if not curve_only:
        '''
        load schematic intro
        '''
        axs.append(fig.add_subplot(nrows, 1, 1))
        
        img = mpimg.imread('/home/mic/paper-brain-wide-map/'
                           'overleaf_figs/manifold/intro.png')
        imgplot = axs[k].imshow(img)
        axs[k].axis('off')
        put_panel_label(axs[k], k)
        k += 1
    

    tops = {}
    lower = {}
    for split in align:
    
        d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'         
                    f'curves_{split}_{mapping}.npy',
                    allow_pickle=True).flat[0]              
             
        maxs = np.array([d[x]['max-min/max+min'] for x in d])
        acronyms = np.array(list(d.keys()))
        order = list(reversed(np.argsort(maxs)))
        maxs = maxs[order]
        acronyms = acronyms[order] 
             
        tops[split] = [acronyms, 
                      [d[reg]['p'] for reg in acronyms]]
        
        maxs = np.array([d[reg]['max-min/max+min'] for reg in acronyms
                         if d[reg]['p'] < 0.01])
                         
        maxsf = [v for v in maxs if not (math.isinf(v) or math.isnan(v))]         
        lower[split] = np.percentile(maxsf, 25)          
        print(split, curve)
        print('25 percentile: ',np.percentile(maxsf, 25))
        print(f'{len(maxsf)} of {len(d)} are significant')
        tops[split+'_s'] = f'{len(maxsf)} of {len(d)}'
            
    '''
    plot average curve per region
    curve types: dist_0, dist_split
    regk is what chunk of Beryl regions to plot
    in groups of consecutive 10
    '''

    for split in align:
        if curve_only:
            axs.append(fig.add_subplot(nrows, len(align), k + 1))
        else:
            axs.append(fig.add_subplot(nrows, len(align), k + len(align)))
            
        f = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'
                    f'curves_{split}_{mapping}.npy',
                    allow_pickle=True).flat[0]
                    
        if curve_only:                        
            regs = tops[split][0][np.array(tops[split][1]) < 0.01]         
        else:
            regs = tops[split][0][np.array(tops[split][1]) < 0.01][:15]     
        
        for reg in regs:
            xx = xs(split)-pre_post[split][0]
            
            axs[k].plot(xx,f[reg][curve], linewidth = 2,
                          color=palette[reg], 
                          label=f"{reg} {f[reg]['nclus']}")

        axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')
        
        if split == 'block':
            ha = 'right'
        else:
            ha = 'left'    
        
        axs[k].text(0, 0.01, align[split],
                      transform=axs[k].get_xaxis_transform(),
                      horizontalalignment = ha)           
        texts = []
        for reg in regs:
            y = np.max(f[reg][curve])
            x = xx[np.argmax(f[reg][curve])]
            ss = f"{reg} {f[reg]['nclus']}" # {np.round(f[reg]['max-min/max+min'],2)}"
            texts.append(axs[k].text(x, y, ss, 
                                     color = palette[reg],
                                     fontsize=9))
                                     
        #adjust_text(texts)             

        
        axs[k].set_ylabel(curve)
        axs[k].set_xlabel('time [sec]')
        axs[k].set_title(f'{split}')
        put_panel_label(axs[k], k)
                
        k +=1

    if curve_only:
        return
    '''
    plot latency versus max for all significant regions
    '''   
    
    for split in align:
    
        axs.append(fig.add_subplot(nrows, len(align), k + len(align)))
        
        d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'         
                    f'curves_{split}_{mapping}.npy',
                    allow_pickle=True).flat[0]

        acronyms = [tops[split][0][j] for j in range(len(tops[split][0]))
                if tops[split][1][j] < 0.01]

        maxes = np.array([d[x][f'max-min/max+min'] for x in acronyms])
        lats = np.array([d[x]['lat'] for x in acronyms])
        cols = [palette[reg] for reg in acronyms]
        
        axs[k].scatter(lats, maxes, color=cols, marker='o',s=1)        
        axs[k].axhline(y=lower[split],linestyle='--',color='r')
        
        for i in range(len(acronyms)):
            if d[acronyms[i]]['max-min/max+min'] > lower[split]:
                axs[k].annotate('  ' + acronyms[i],
                                (lats[i], maxes[i]),
                    fontsize=5,color=palette[acronyms[i]])            

        axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')
        
        if split == 'block':
            ha = 'right'
        else:
            ha = 'left'    
        
        axs[k].text(0, 0.01, align[split],
                      transform=axs[k].get_xaxis_transform(),
                      horizontalalignment = ha)           
  
        axs[k].set_ylabel(f'(max-min)/(max+min)')
        axs[k].set_xlabel('latency (0.95 * max) [sec]')
        axs[k].set_title(f"{split}, {tops[split+'_s']}")
        put_panel_label(axs[k], k)     
        k +=1

   
    '''
    max dist_split onto swanson flat maps
    (only regs with p < 0.01)
    '''

    for split in align:
    
        axs.append(fig.add_subplot(nrows, len(align),k + len(align)))   
 
        d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'         
                    f'curves_{split}_{mapping}.npy',
                    allow_pickle=True).flat[0]

        acronyms = [tops[split][0][j] for j in range(len(tops[split][0]))
                if tops[split][1][j] < 0.01]

        values = np.array([d[x][f'max-min/max+min'] for x in acronyms])
             
        plot_swanson(list(acronyms), list(values), cmap='Blues', 
                     ax=axs[k], br=br)#, orientation='portrait')
        axs[k].axis('off')
        axs[k].set_title(f'(max-min)/(max+min), {split}')
        put_panel_label(axs[k], k)
        k += 1

    '''
    lat onto swanson flat maps
    (only regs with p < 0.01)
    '''

    for split in align:
    
        axs.append(fig.add_subplot(nrows, len(align),k + len(align)))   
 
        d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'         
                    f'curves_{split}_{mapping}.npy',
                    allow_pickle=True).flat[0]
                    
        acronyms = [tops[split][0][j] for j in range(len(tops[split][0]))
                if tops[split][1][j] < 0.01]

        #  compute latencies (inverted, shorter latency is darker)
        for x in acronyms:
            
            if np.max(d[x]['d_var_m']) == np.inf:
                loc = np.where(d[x]['d_var_m'] == np.inf)[0]  
            else:
                loc = np.where(d[x]['d_var_m'] > 
                                0.95 * np.max(d[x]['d_var_m']))[0]
                                
            d[x]['lat'] = xs(split)[-1] - xs(split)[loc[0]]

        values = np.array([d[x]['lat'] for x in acronyms])
                         
        plot_swanson(list(acronyms), list(values), cmap='Blues', 
                     ax=axs[k], br=br)#, orientation='portrait')
        axs[k].axis('off')
        axs[k].set_title(f'lat (dark = early), {split}')
        put_panel_label(axs[k], k)
        k += 1


    '''
    general subplots settings
    '''


    fig.subplots_adjust(top=0.999,
bottom=0.01,
left=0.057,
right=0.979,
hspace=0.4,
wspace=0.15)
                   
    #fig.suptitle(f'pcadim {pcadim}')
    #fig.canvas.manager.set_window_title(f'pcadim {pcadim}')     
#    fig.savefig('/home/mic/paper-brain-wide-map/'
#               f'overleaf_figs/manifold/plots/'
#               f'all_panels_pcadim{pcadim}.png',dpi=200)
#    plt.close()

    font = {'size'   : 10}
    mpl.rc('font', **font)    



def plot_session_numbers():


    split = 'choice'
    mapping = 'Swanson'
    R = np.load('/home/mic/paper-brain-wide-map/'
                f'manifold_analysis/full_res/bwm_psths_{split}.npy',
                allow_pickle=True).flat[0]

    nt, nclus, nobs = R['ws'][0].shape

    # possible that certain psths are nan if there were not enough trials
    # remove those insertions
    
    nanins = [k for k in range(len(R['ws'])) if np.isnan(R['ws'][k]).any()]   
       
    ids = [R['ids'][k] for k in range(len(R['ws'])) if k not in nanins]  

    eid_probe = [R['eid_probe'][k] for k in range(len(R['ws'])) 
                          if k not in nanins]
    
    assert len(ids) == len(eid_probe), 'check dims!'
    
    r_ins = []
    for ins in range(len(eid_probe)):
        ep = eid_probe[ins]
        for c in ids[ins]:
            r_ins.append(ep)   
    
    acs = br.id2acronym(np.concatenate(ids),mapping=mapping)               
    acs = np.array(acs,dtype=object)
    ins = np.array(r_ins,dtype=object)
    
    regs = Counter(acs)
    print(len(regs))

    d2 = {}
    c = []
    for reg in regs:
        if reg in ['root','void']:
            continue
        cc = list(Counter(ins[acs == reg]).values())    
        d2[reg] = cc    
        c.append(len(cc))
    
    regs = sorted(d2, key = lambda key: len(d2[key]))
    
    d3 = {}
    for reg in regs:
        d3[reg] = d2[reg]
        
    #return d3
    # plot histogram, how many recordings per session in a region
    fig, ax = plt.subplots(figsize=(6,2))
    counts = np.concatenate(list(d3.values()))
    
    
    _, bins = np.histogram(np.log10(counts + 1), bins='auto')
    
    binwidth = 50
#    axs[0].hist(counts, bins=np.arange(min(counts), 
#            max(counts) + binwidth, binwidth), histtype='step')
    ax.hist(counts, bins=10**bins, histtype='step',
                label = 'number of neurons \n per regional recording')            
    q = [sum(d3[x]) for x in d3]
    ax.hist(q, bins=10**bins, histtype='step',
                label = 'number of pooled neurons \n  per region')

    ax.set_xscale("log")        
            
#    axs[0].set_xlabel('number of neurons \n per regional recording')
    ax.set_ylabel('frequency')    
#    fig.tight_layout()
#    fig.savefig('number_of_neurons_per_regional_recording.pdf')
    ax.legend(ncol=1).set_draggable(True)
 

    
    #plot histogram of number of cells per area
    #fig, ax = plt.subplots(figsize=(3,2))
#    axs[1].hist(q,bins=np.arange(min(q), max(q) + binwidth, binwidth), 
#            histtype='step')
#    axs[1].set_xlabel('number of pooled neurons \n  per region')
#    axs[1].set_ylabel('frequency')    
    fig.tight_layout()
#    #fig.savefig('number_of_pooled_neurons_per_region.png', dpi=200)    

    #return d3
    #df = pd.DataFrame.from_dict({'acs':acs, 'ins':r_ins})
    

def swanson_gif(split, curve='d_var_m', recompute=True):

    '''
    use dist_split(t) to make swanson plots for each t
    then combine to GIF
    
    split in stim, action
    '''

    mapping = 'Swanson'      
    
    f = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'
                f'curves_{split}_mapping{mapping}.npy',
                allow_pickle=True).flat[0]                

    T = []
    for t in range(int((pre_post[split][0] + pre_post[split][1])//T_BIN(split))):
        acronyms = []
        values = []
        for reg in f:
            values.append(f[reg]['d_var_m'][t])
            acronyms.append(reg)
        T.append([acronyms, values])
        
    all_vals = np.concatenate([x[1] for x in T])
    vmin = min(all_vals)
    vmax = max(all_vals)
    
    plt.ioff()
    
    s0 = f'/home/mic/paper-brain-wide-map/manifold_analysis/gif/{curve}'
    Path(s0+f'/{split}').mkdir(parents=True, exist_ok=True)
    
    if recompute:
        for t in range(int((pre_post[split][0] + pre_post[split][1])//T_BIN(split))):
            acronyms = T[t][0]
            values = T[t][1]    
            fig, ax = plt.subplots(figsize=(15,7))
            plot_swanson(acronyms, values, cmap='Blues', 
                         ax=ax, br=br, vmin=vmin, vmax=vmax, annotate=True)
                         
            ax.set_title(f'split {split}; t = {t*T_BIN(split)} sec')             
            ax.axis('off') 
            fig.tight_layout()
            fig.savefig(s0+f'/{split}/{t:03}.png',dpi=200)
            plt.close()


    images = sorted([image for image in glob.glob(s0+f'/{split}/*.png')])
    frames = [Image.open(image) for image in images]
    frame_one = frames[0]
    frame_one.save(s0+f'/{split}.gif', format="GIF", append_images=frames,
               save_all=True, duration=300, loop=1)

    plt.ion()


        
def plot_average_curves(split='stim', mapping='Swanson', curve = 'd_var_m',
                        single_reg='VISl'):
    
    '''
    plot average curve per region
    curve types: dist_0, dist_split

    single_reg=region to illustrate null distribution;
    if single_reg=False, the first 55 regions are shown
    '''

    fig, ax = plt.subplots()
    _, palette = get_allen_info()
    
    g = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'
                f'curves_{split}_mapping{mapping}.npy',
                allow_pickle=True).flat[0]
    
    f = g[0]            
                   
    maxs = np.array([f[x][f'max_{curve}'] for x in f])
    acronyms = np.array([x for x in f])
    order = list(reversed(np.argsort(maxs)))
    maxs = maxs[order]
    acronyms = acronyms[order]               
    
    print(len(acronyms),'regs in total')
    print('first 20:')
    print(acronyms[:20])
    print('last 5:')
    print(acronyms[-5:])               
    
    if single_reg:
        regs = [single_reg]
    else:                             
        regs = np.concatenate([acronyms[:55], acronyms[-10:]])           

    for reg in regs:
        xx = xs(split)-pre_post[split][0]
        
        ax.plot(xx,f[reg][curve], linewidth = 2,
                      color=palette[reg], 
                      label=f"{reg}")# [{f[reg]['nclus'][1]}]
                      
        if single_reg:
            for i in range(len(g))[1:]:
                ax.plot(xx,g[i][reg][curve], linewidth = 1,
                              color='gray')            

    ax.axvline(x=0, lw=0.5, linestyle='--', c='k')
    
    if split == 'block':
        ha = 'right'
    else:
        ha = 'left'    
    
    ax.text(0, 0.01, align[split],
                  transform=ax.get_xaxis_transform(),
                  horizontalalignment = ha)           
    texts = []
    for reg in regs:
        y = np.max(f[reg][curve])
        x = xx[np.argmax(f[reg][curve])]
        texts.append(ax.text(x, y, f"{reg}",#['nclus'][1] 
                             color = palette[reg],
                             fontsize=9))
                                 
    #adjust_text(texts)             
    ax.set_ylabel(curve)
    ax.set_xlabel('time [sec]')
    ax.set_title(f'{split}')




