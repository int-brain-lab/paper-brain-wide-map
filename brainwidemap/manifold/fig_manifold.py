from one.api import ONE
from reproducible_ephys_processing import bin_spikes2D
from brainwidemap import bwm_query, load_good_units
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
from brainbox.io.one import SpikeSortingLoader, SessionLoader

from ibllib.atlas import FlatMap
from ibllib.atlas.flatmaps import plot_swanson

from scipy import optimize, signal, stats
import pandas as pd
import numpy as np
from collections import Counter, ChainMap
from sklearn.decomposition import PCA
import gc
import umap
import os
from pathlib import Path
import glob
from dateutil import parser

from matplotlib.axis import Axis
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from adjustText import adjust_text
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from pdf2image import convert_from_path
import fitz
from PIL import Image
import io
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

import random
from random import shuffle
from copy import deepcopy
import time
import sys

import math
import string

import cProfile
import pstats

import warnings
#warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)

plt.rcParams.update(plt.rcParamsDefault)
plt.ion()


f_size = 15
def set_sizes(f_s = f_size):
    plt.rc('font', size=f_s)
    plt.rc('axes', titlesize=f_s)
    plt.rc('axes', labelsize=f_s)
    plt.rc('xtick', labelsize=f_s)
    plt.rc('ytick', labelsize=f_s)
    plt.rc('legend', fontsize=f_s)
    plt.rc('figure', titlesize=f_s)

#set_sizes()    

#plt.rcParams['figure.constrained_layout.use'] = True

blue_left = [0.13850039, 0.41331206, 0.74052025]
red_right = [0.66080672, 0.21526712, 0.23069468]

b_size = 0.0125  # 0.005 sec for a static bin size, or None for single bin
sts = 0.002  # stride size in s for overlapping bins 
ntravis = 30  # number of trajectories for visualisation, first 2 real
#nrand = 2000  # number of random trial splits for null_d
#min_reg = 100  # 100, minimum number of neurons in pooled region


# trial split types, see get_d_vars for details       
align = {'stim':'stim on',
         'choice':'motion on',
         'fback':'feedback',
         'block':'stim on'}
                  

# [pre_time, post_time]
pre_post = {'choice':[0.15,0],'stim':[0,0.15],
            'fback':[0,0.7],'block':[0.4,-0.1]}


# labels for illustrative trajectory legend 
trial_split = {'choice':['choice left', 'choice right','pseudo'],
               'stim':['stim left','stim right','pseudo'],
               'fback':['correct','false','pseudo'],
               'block':['pleft 0.8','pleft 0.2','pseudo']}

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)  # (mode='local')
ba = AllenAtlas()
br = BrainRegions()

# save results for plotting here
pth_res = Path(one.cache_dir, 'manifold', 'res') 
pth_res.mkdir(parents=True, exist_ok=True)


def T_BIN(split, b_size=b_size):

    # c = 0.005 # time bin size in seconds (5 ms)
    if b_size is None:
        return pre_post[split][0] + pre_post[split][1]
    else:
        return b_size  


def grad(c,nobs):
    cmap = mpl.cm.get_cmap(c)
    
    return [cmap(0.5*(nobs - p)/nobs) for p in range(nobs)]


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


def get_name(brainregion):
    regid = br.id[np.argwhere(br.acronym == brainregion)][0, 0]
    return br.name[np.argwhere(br.id == regid)[0, 0]]


def get_d_vars(split, pid, mapping='Beryl', control=True,
               nrand = 100):

    '''
    for a given session, probe, bin neural activity
    cut into trials, compute d_var per region
    nrand: nuber of random trial splits
    '''
    
    
    eid,probe = one.pid2eid(pid)
    
    # load in spikes
    spikes, clusters = load_good_units(one, pid)    

    # Load in trials data
    sess_loader = SessionLoader(one, eid)
    sess_loader.load_trials()
    trials = sess_loader.trials
       
    # remove certain trials     
    rs_range = [0.08, 2]  # discard [long/short] reaction time trials
    stim_diff = trials['firstMovement_times'] - trials['stimOn_times']     
    rm_trials = np.bitwise_or.reduce([np.isnan(trials['stimOn_times']),
                               np.isnan(trials['choice']),
                               np.isnan(trials['feedback_times']),
                               np.isnan(trials['probabilityLeft']),
                               np.isnan(trials['firstMovement_times']),
                               np.isnan(trials['feedbackType']),
                               stim_diff > rs_range[-1],
                               stim_diff < rs_range[0]])
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
                ~rm_trials,~np.isnan(trials[f'contrast{side}'])])])
            trn.append(np.arange(len(trials['stimOn_times']))
            [np.bitwise_and.reduce([ 
            ~rm_trials,~np.isnan(trials[f'contrast{side}'])])])
       
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

    print('#trials per condition: ',len(trn[0]), len(trn[1]))
    assert (len(trn[0]) != 0) and (len(trn[1]) != 0), 'zero trials to average'
           
    assert len(spikes['times']) == len(spikes['clusters']), 'spikes != clusters'   
            
    # bin and cut into trials    
    bins = []

    for event in events:
    
        #  overlapping time bins, bin size = T_BIN, stride = sts 
        bis = []
        st = int(T_BIN(split)//sts) 
        
        for ts in range(st):
    
            bi, _ = bin_spikes2D(spikes['times'],
                               clusters['cluster_id'][spikes['clusters']],
                               clusters['cluster_id'],
                               np.array(event) + ts*sts, 
                               pre_post[split][0], pre_post[split][1], 
                               T_BIN(split))
            bis.append(bi)
            
        ntr, nn, nbin = bi.shape
        ar = np.zeros((ntr, nn, st*nbin))
        
        for ts in range(st):
            ar[:,:,ts::st] = bis[ts]
                           
        bins.append(ar)                   
                                              
    b = np.concatenate(bins)
    
    # recreate temporal trial order              
    dx = np.concatenate([list(zip([True]*len(trn[0]),trn[0])),
                    list(zip([False]*len(trn[1]),trn[1]))])

    b = b[np.argsort(dx[:, 1])]    
           
    ntr, nclus, nbins = b.shape    
    
    acs = br.id2acronym(clusters['atlas_id'],mapping=mapping)
               
    acs = np.array(acs)
    wsc = np.concatenate(b,axis=1)

    # Discard cells with any nan or 0 for all bins    
    goodcells = [k for k in range(wsc.shape[0]) if 
                 (not np.isnan(wsc[k]).any()
                 and wsc[k].any())] 

    acs = acs[goodcells]
    b = b[:,goodcells,:]
    bins2 = [x[:,goodcells,:] for x in bins]
    bins = bins2    

    goodcells = ~np.bitwise_or.reduce([acs == reg for 
                     reg in ['void','root']])
    
    acs = acs[goodcells]
    b = b[:,goodcells,:]
    bins2 = [x[:,goodcells,:] for x in bins]
    bins = bins2

    if control:
        # get mean and var across trials
        w0 = [bi.mean(axis=0) for bi in bins]  
        s0 = [bi.var(axis=0) for bi in bins]
        
        perms = []  # keep track of random trial splits to test sig
        
        # nrand times random impostor/pseudo split of trials 
        for i in range(nrand):
            if split == 'block':  #'block' pseudo sessions
                ys = generate_pseudo_blocks(ntr, first5050=0) == 0.8             
            
            elif split == 'stim':
                # shuffle stim sides within block and choice classes
                
                # get real block labels
                y_ = trials['probabilityLeft'][
                            sorted(dx[:,1])].values
                            
                # get real choices
                stis = trials['choice'][
                            sorted(dx[:,1])].values
                 
                # block/choice classes            
                c0 = np.bitwise_and(y_ == 0.8, stis == 1)            
                c1 = np.bitwise_and(y_ != 0.8, stis == 1)
                c2 = np.bitwise_and(y_ == 0.8, stis != 1)
                c3 = np.bitwise_and(y_ != 0.8, stis != 1)
          
                tr_c = dx[np.argsort(dx[:,1])][:,0]  # true stim sides
                tr_c2 = deepcopy(tr_c)
                
                # shuffle stim sides within each class
                for cc in [c0,c1,c2,c3]:
                    r = tr_c[cc]
                    tr_c2[cc] = np.array(random.sample(list(r), len(r)))

                ys = tr_c2 == 1  # boolean shuffled stim sides            
                                      
                
            elif split in ['choice', 'fback']:  
                # shuffle choice sides within block and stim classes
                
                # get real block labels
                y_ = trials['probabilityLeft'][
                            sorted(dx[:,1])].values
                # get real stim sides
                stis = trials['contrastLeft'][
                            sorted(dx[:,1])].values
                 
                # block/stim classes            
                c0 = np.bitwise_and(y_ == 0.8, np.isnan(stis))            
                c1 = np.bitwise_and(y_ != 0.8, np.isnan(stis))
                c2 = np.bitwise_and(y_ == 0.8, ~np.isnan(stis))
                c3 = np.bitwise_and(y_ != 0.8, ~np.isnan(stis))
          
                tr_c = dx[np.argsort(dx[:,1])][:,0]  # true choices
                tr_c2 = deepcopy(tr_c)
                
                # shuffle choices within each class
                for cc in [c0,c1,c2,c3]:
                    r = tr_c[cc]
                    tr_c2[cc] = np.array(random.sample(list(r), len(r)))

                ys = tr_c2 == 1  # boolean shuffled choices

                if split == 'fback':
                    # get feedback types from shuffled choices
                    cl = np.bitwise_and(tr_c == 0, np.isnan(stis))
                    cr = np.bitwise_and(tr_c == 1, ~np.isnan(stis))  
                    ys = np.bitwise_or(cl,cr)

                
            w0.append(b[ys].mean(axis=0))
            s0.append(b[ys].var(axis=0))
            
            w0.append(b[~ys].mean(axis=0))
            s0.append(b[~ys].var(axis=0))                      

            perms.append(ys)

    else: # average trials per condition
        print('all trials')
        w0 = [bi.mean(axis=0) for bi in bins] 
        s0 = [bi.var(axis=0) for bi in bins]

    ws = np.array(w0)
    ss = np.array(s0)
    
    regs = Counter(acs)

    # Keep single cell d_var in extra file for computation of mean
    # Can't be done with control data as files become too large 
    # strictly standardized mean difference
    d_var = (((ws[0] - ws[1])/
              ((ss[0] + ss[1])**0.5))**2)
              
    d_euc = (ws[0] - ws[1])**2          
              
    D_ = {}
    D_['acs'] = acs
    D_['d_vars'] = d_var
    D_['d_eucs'] = d_euc
    D_['ws'] = ws[:ntravis]

    if not control:
        return D_

    #  Sum together cells in same region to save memory
    D = {}
    
    for reg in regs:
    
        res = {}

        ws_ = [y[acs == reg] for y in ws]
        ss_ = [y[acs == reg] for y in ss]
     
        res['nclus'] = sum(acs == reg)
        d_vars = []
        d_eucs = []
                
        for j in range(len(ws_)//2):

            # strictly standardized mean difference
            d_var = (((ws_[2*j] - ws_[2*j + 1])/
                      ((ss_[2*j] + ss_[2*j + 1])**0.5))**2)
            
            # Euclidean distance          
            d_euc = (ws_[2*j] - ws_[2*j + 1])**2          

            # sum over cells, divide by #neu later
            d_var_m = np.nansum(d_var,axis=0)
            d_euc_m = np.sum(d_euc,axis=0)
            
            d_vars.append(d_var_m)
            d_eucs.append(d_euc_m)
            
        res['d_vars'] = d_vars
        res['d_eucs'] = d_eucs
        
        D[reg] = res
        
    D_['uperms'] = len(np.unique([str(x.astype(int)) for x in perms]))
    D_['D'] = D    
    return D_    


'''    
###
### bulk processing 
###    
''' 

def get_all_d_vars(split, eids_plus = None, control = True, 
                   mapping='Beryl'):

    '''
    for all BWM insertions, get the PSTHs and acronyms,
    '''
    
    time00 = time.perf_counter()
    
    print('split', split, 'control', control)
    
    if eids_plus is None:
        df = bwm_query(one)
        eids_plus = df[['eid','probe_name', 'pid']].values

    # save results per insertion (eid_probe) in FlatIron folder
    pth = Path(one.cache_dir, 'manifold', split) 
    pth.mkdir(parents=True, exist_ok=True)
 
    Fs = []   
    k=0
    print(f'Processing {len(eids_plus)} insertions')
    for i in eids_plus:
        eid, probe, pid = i  
          
        time0 = time.perf_counter()
        try:
            D_ = get_d_vars(split, pid, control=control, mapping=mapping)
            eid_probe = eid+'_'+probe
            
            np.save(Path(pth,f'{eid_probe}.npy'), D_, allow_pickle=True)
                                         
            gc.collect() 
            print(k+1, 'of', len(eids_plus), 'ok') 
        except:
            Fs.append(pid)
            gc.collect()
            print(k+1, 'of', len(eids_plus), 'fail', pid)
            
        time1 = time.perf_counter()
        print(time1 - time0, 'sec')
                    
        k+=1            
               
    time11 = time.perf_counter()
    print((time11 - time00)/60, f'min for the complete bwm set, {split}')
    print(f'{len(Fs)}, load failures:')
    print(Fs)


def d_var_stacked(split, min_reg = 100, uperms_ = False):
                  
    time0 = time.perf_counter()

    '''
    average d_var_m via nanmean across insertions,
    remove regions with too few neurons (min_reg)
    compute maxes, latencies and p-values
    '''
    
    print(split)
    pth = Path(one.cache_dir, 'manifold', split) 
    ss = os.listdir(pth)  # get insertions
    

    # pool data for illustrative PCA
    acs = []
    ws = [] 
    regdv0 = {}
    regde0 = {}
    uperms = {}
    
    # group results across insertions
    for s in ss:
    
        D_ = np.load(Path(pth,s), 
                    allow_pickle=True).flat[0]
                       
        uperms[s.split('.')[0]] = D_['uperms']
        if uperms_:
            continue
        acs.append(D_['acs'])
        ws.append(D_['ws'])        
        
        for reg in D_['D']:
            if reg not in regdv0:
                regdv0[reg] = []
            regdv0[reg].append(np.array(D_['D'][reg]['d_vars'])/b_size)
            if reg not in regde0:
                regde0[reg] = []            
            regde0[reg].append(np.array(D_['D'][reg]['d_eucs'])/b_size)
    
    if uperms_:
        return uperms

    acs = np.concatenate(acs)
    ws = np.concatenate(ws, axis=1)
    regs0 = Counter(acs)
    regs = {reg: regs0[reg] for reg in regs0 if regs0[reg] > min_reg}
        
    # nansum across insertions and take sqrt
    regdv = {reg: (np.nansum(regdv0[reg],axis=0)/regs[reg])**0.5 
                  for reg in regs }         
    regde = {reg: (np.nansum(regde0[reg],axis=0)/regs[reg])**0.5
                  for reg in regs}
        
    r = {}
    for reg in regs:         
        res = {}        
    
        # get PCA for 3d trajectories
        dat = ws[:,acs == reg,:]
        pca = PCA(n_components = 3)
        wsc = pca.fit_transform(np.concatenate(dat,axis=1).T).T

        res['pcs'] = wsc
        res['nclus'] = regs[reg]

        '''
        var
        '''        
        # amplitudes
        ampsv = [np.max(x) - np.min(x) for x in regdv[reg]]                

        # p value
        res['p_var'] = np.mean(np.array(ampsv) >= ampsv[0])      

        # full curve, subtract null-d mean
        d_var = regdv[reg][0] - np.mean(regdv[reg][1:], axis=0)
        res['d_var'] = d_var - min(d_var)
        res['amp_var'] = max(res['d_var'])
    
        # latency  
        if np.max(res['d_var']) == np.inf:
            loc = np.where(res['d_var'] == np.inf)[0]  
        else:
            loc = np.where(res['d_var'] > 0.7*(np.max(res['d_var'])))[0]
        
        res['lat_var'] = np.linspace(-pre_post[split][0], 
                        pre_post[split][1], len(res['d_var']))[loc[0]]
                      
        '''
        euc
        '''          
        # amplitudes
        ampse = [np.max(x) - np.min(x) for x in regde[reg]]                

        # p value
        res['p_euc'] = np.mean(np.array(ampse) >= ampse[0])      

        # full curve, subtract null-d mean
        d_euc = regde[reg][0] - np.mean(regde[reg][1:], axis=0)
        res['d_euc'] = d_euc - min(d_euc)
        res['amp_euc'] = max(res['d_euc'])
    
        # latency  
        loc = np.where(res['d_euc'] > 0.7*(np.max(res['d_euc'])))[0]
        
        res['lat_euc'] = np.linspace(-pre_post[split][0], 
                        pre_post[split][1], len(res['d_euc']))[loc[0]]   

        r[reg] = res        
    
    np.save(Path(pth_res,f'{split}.npy'), r, allow_pickle=True)
           
    time1 = time.perf_counter()    
    print('total time:', time1 - time0, 'sec')


def curves_params_all(split):

    get_all_d_vars(split)
    d_var_stacked(split)        

      
'''    
#####################################################
### plotting 
#####################################################    
'''

def get_allen_info():
    dfa = pd.read_csv(Path(pth_res,'allen_structure_tree.csv'))
    
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


def put_panel_label(ax, k):                    
        ax.annotate(string.ascii_lowercase[k],(-0.05, 1.15),
                        xycoords='axes fraction',
                        fontsize=f_size*1.5, va='top', 
                        ha='right', weight='bold')


def plot_all(curve='euc', amp_number=False, intro = True, 
             sigl=0.01, only3d = False, onlyScat = False, 
             single_scat=False, auto_label_adjust = False):

    '''
    main figure: show example trajectories,
    d_var_m for select regions and scatter all sigificant regions
    curve in [var, euc]
    sigl: significance level, default 0.05
    intro: if False, don't plot schematics of method and contrast
    only3d: only 3d plots 
    '''

    
    nrows = 12
        
    fig = plt.figure(figsize=(10, 10*(2**.5)))  #10, 15
    gs = fig.add_gridspec(nrows, len(align))
    
    dfa, palette = get_allen_info()
                   

    axs = []
    k = 0  # panel counter
    row = 0  # row counter
    

    '''
    get significant regions
    '''           
    tops = {}
    regsa = []

    for split in align: 
    
        d = np.load(Path(pth_res,f'{split}.npy'),
                    allow_pickle=True).flat[0]              
             
        maxs = np.array([d[x][f'amp_{curve}'] for x in d])
        acronyms = np.array(list(d.keys()))
        order = list(reversed(np.argsort(maxs)))
        maxs = maxs[order]
        acronyms = acronyms[order] 
             
        tops[split] = [acronyms, 
                      [d[reg][f'p_{curve}'] for reg in acronyms], maxs]

            
        maxs = np.array([d[reg][f'amp_{curve}'] for reg in acronyms
                         if d[reg][f'p_{curve}'] < sigl])
                         
        maxsf = [v for v in maxs if not (math.isinf(v) or math.isnan(v))] 

        print(split, curve)
        print(f'{len(maxsf)}/{len(d)} are significant')
        tops[split+'_s'] = f'{len(maxsf)}/{len(d)}'
        regs_a = [tops[split][0][j] for j in range(len(tops[split][0]))
                if tops[split][1][j] < sigl]
        
        
        regsa.append(regs_a)
        print(regs_a)
        print(' ')


    #  get Cosmos parent region for yellow color adjustment
    regsa = np.unique(np.concatenate(regsa))
    cosregs_ = [dfa[dfa['id'] == int(dfa[dfa['acronym']==reg]['structure_id_path']
           .values[0].split('/')[4])]['acronym']
           .values[0] for reg in regsa]
    
    cosregs = dict(zip(regsa,cosregs_)) 

    v = 0 if intro else 3  # indexing rows
    vv = 1 if v == 0 else 0  # indexing columns

    '''
    example regions per split for embedded space and line plots
    '''

    tops_p = {split: [tops[split][0][j] for j in range(len(tops[split][0]))
                if tops[split][1][j] < sigl] for split in align}
          
#    exs = {split: list(set(exs0[split]
#                  ).intersection(set(tops_p[split]))) 
#                  for split in align}           
               
    exs = {'stim': ['VISp', 'LP', 'LGd', 'VISpm', 'VISam', 
                    'SCm', 'CP', 'MRN','GRN', 'GPe'],
         'choice': ['GRN','ACAv', 'APN', 'IRN', 'PRM', 
                    'PRNr', 'RT', 'SIM', 'SSp-ul', 'ZI'],
          'fback': ['IRN', 'SSp-m', 'PRNr', 'IC', 'MV', 'AUDp', 
                    'CENT3', 'SSp-ul', 'GPe'],
          'block': ['Eth', 'IC']}


    if not onlyScat:

        
        '''
        load schematic intro and contrast trajectory plot (convert from svg)
        '''

        if intro:
    
            axs.append(fig.add_subplot(gs[:3,:]))
      
            pdf = fitz.open(Path(pth_res,'intro2.pdf'))
            rgb = pdf[0].get_pixmap(dpi=600)
            pil_image = Image.open(io.BytesIO(rgb.tobytes()))
                    
            imgplot = axs[k].imshow(pil_image.convert('RGB'))
            axs[k].axis('off')
            put_panel_label(axs[k], k)
            k += 1
            row += 1

           
        '''
        Trajectories for example regions in PCA embedded 3d space
        '''
                         
        c = 0
        for split in align:
            d = np.load(Path(pth_res,f'{split}.npy'),
                        allow_pickle=True).flat[0] 

            # pick example region
            reg = exs[split][0]
            
            if only3d:
                axs.append(fig.add_subplot(gs[:,c],
                                           projection='3d'))             
            else:
                axs.append(fig.add_subplot(gs[3-v:6-v,c],
                                           projection='3d'))           

            npcs, allnobs = d[reg]['pcs'].shape
            nobs = allnobs // ntravis

            for j in range(ntravis):
            
                # 3d trajectory
                cs = d[reg]['pcs'][:,nobs * j: nobs * (j+1)].T
        

                if j == 0:
                    col = grad('Blues_r', nobs)
                elif j == 1:
                    col = grad('Reds_r', nobs)
                else:
                    col = grad('Greys_r', nobs)
                        
                axs[k].plot(cs[:,0],cs[:,1],cs[:,2], color=col[len(col)//2],
                        linewidth = 5 if j in [0,1] else 1,
                        label=trial_split[split][j] if j in range(3)
                                  else '_nolegend_', alpha = 0.5)
                              
                p = axs[k].scatter(cs[:,0],cs[:,1],cs[:,2], color=col,
                               edgecolors = col, 
                               s=20 if j in [0,1] else 1,
                               depthshade=False)


            axs[k].set_title(f"{split}, {reg} {d[reg]['nclus']}")   
            axs[k].grid(False)
            axs[k].axis('off')

            if c == 0:
                put_panel_label(axs[k], row)

            c += 1
            k += 1
            
        row += 1
        
                   
        if only3d:
            return
                 
        '''
        line plot per 5 example regions per split
        '''
        c = 0  # column index

        for split in align:
            if c == 0:
                axs.append(fig.add_subplot(gs[6-v:8-v,c]))
                #axs[-1].set_ylim(0, 4.5/b_size)
            else:  # to share y axis
                axs.append(fig.add_subplot(gs[6-v:8-v,c]))
                           #,sharey=axs[len(axs)-1]))
                               
            d = np.load(Path(pth_res,f'{split}.npy'),
                        allow_pickle=True).flat[0] 
                       
            # example regions to illustrate line plots              
            regs = exs[split]
            
            texts = []          
            for reg in regs:
                if any(np.isinf(d[reg][f'd_{curve}'])):
                    print(f'inf in {curve} of {reg}')
                    continue


                xx = np.linspace(-pre_post[split][0], 
                                  pre_post[split][1], 
                                  len(d[reg][f'd_{curve}']))       
      
                # get units in Hz          
                yy = d[reg][f'd_{curve}']
                
                
                axs[k].plot(xx,yy, linewidth = 2,
                              color=palette[reg], 
                              label=f"{reg} {d[reg]['nclus']}")
                

#                # plot stderror bars on lines
#                axs[k].fill_between(xx,    
#                                 yy + d[reg][f'stde_{curve}'],
#                                 yy - d[reg][f'stde_{curve}'],
#                                 color=palette[reg],
#                                 alpha = 0.2)
                                 
                # put region labels                 
                y = yy[-1]
                x = xx[-1]
                ss = f"{reg} {d[reg]['nclus']}" 

                if cosregs[reg] in ['CBX', 'CBN']:  # darken yellow
                    texts.append(axs[k].text(x, y, ss, 
                                             color = 'k',
                                             fontsize=8))             
                                               
                texts.append(axs[k].text(x, y, ss, 
                                         color = palette[reg],
                                         fontsize=8))                 




            axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')
            
            if split in ['block', 'choice']:
                ha = 'left'
            else:
                ha = 'right'              

            axs[k].spines['top'].set_visible(False)
            axs[k].spines['right'].set_visible(False)
            if c == 0:        
                axs[k].set_ylabel(f'distance [Hz]')
            axs[k].set_xlabel('time [sec]')

            if c == 0:
                put_panel_label(axs[k], row)
            
            c += 1        
            k += 1
            
        row += 1
                   
    if onlyScat:
        v = 8

    '''
    scatter latency versus max amplitude for significant regions
    '''   
 
    fsize = 20 if single_scat else 9 if onlyScat else 7
    dsize = 120 if single_scat else 9 if onlyScat else 4 # was 1
    

    if amp_number: 
        fig2 = plt.figure()
        axss = []
        fig2.suptitle(f'distance metric: {curve}')
    
    if single_scat:
        figs = [plt.subplots(figsize=(10,10)) 
                for split in align]
           
            
    c = 0  
    for split in align:

        if not single_scat:
            
            if c == 0:
                axs.append(fig.add_subplot(gs[8-v:,c]))
            else:
                axs.append(fig.add_subplot(gs[8-v:,c]))
                           #,sharey=axs[len(axs)-1]
                                  
            axs[-1].set_ylim(0, 4.5)
        
        else:   
            axs.append(figs[c][1])
            axs[-1].set_ylim(0, 7.5/b_size if split == 'fback' else 4.5/b_size)
            axs[-1].xaxis.set_major_locator(ticker.MultipleLocator(0.05))


        d = np.load(Path(pth_res,f'{split}.npy'),
                    allow_pickle=True).flat[0] 

        acronyms = [tops[split][0][j] for j in range(len(tops[split][0]))] 
        ac_sig = np.array([True if tops[split][1][j] < sigl 
                        else False for j in range(len(tops[split][0]))])

        maxes = np.array([d[x][f'amp_{curve}'] for x in acronyms])
        lats = np.array([d[x][f'lat_{curve}'] for x in acronyms])
        #stdes = np.array([d[x][f'stde_{curve}'] for x in acronyms])
        cols = [palette[reg] for reg in acronyms]


        if amp_number:  # supp figure for correlation of nclus and maxes
            axss.append(fig2.add_subplot(int(f'1{len(align)}{c+1}')))
            nums = [1/d[reg]['nclus'] for reg in np.array(acronyms)[ac_sig]]

            l = list(zip(nums, np.array(maxes)[ac_sig], 
                               np.array(cols)[ac_sig]))
            df = pd.DataFrame(l, columns=['1/nclus', 'maxes', 'cols'])
                  
            sns.regplot(ax=axss[c], x="1/nclus", y="maxes", data=df)
            axss[c].set_title(split)
        
        # yerr = 100*maxes/d[reg]['nclus']
        axs[k].errorbar(lats, maxes, yerr=None, fmt='None', 
                        ecolor=cols, ls = 'None', elinewidth = 0.5)
        
        # plot significant regions               
        axs[k].scatter(np.array(lats)[ac_sig], 
                       np.array(maxes)[ac_sig], 
                       color = np.array(cols)[ac_sig], 
                       marker='D',s=dsize)
        
        # plot insignificant regions               
        axs[k].scatter(np.array(lats)[~ac_sig], 
                       np.array(maxes)[~ac_sig], 
                       color = np.array(cols)[~ac_sig], 
                       marker='o',s=dsize/10)                       
                
        # put extra marker for highlighted regions
        exs_i = [i for i in range(len(acronyms)) 
                 if acronyms[i] in exs[split]]
                 
#        axs[k].scatter(np.array(lats)[exs_i], np.array(maxes)[exs_i], 
#                       color=np.array(cols)[exs_i], marker='x',s=10*dsize)
         
        texts = []
        alw = 0.1
        for i in range(len(acronyms)):
            if ac_sig[i]:  # only decorate marker with label if sig
                reg = acronyms[i]
                
                if cosregs[reg] in ['CBX', 'CBN']:
                        texts.append(axs[k].annotate('  ' + reg, 
                                    (lats[i], maxes[i]),
                                    fontsize=fsize,color='k',
                                    arrowprops=dict(arrowstyle="-", 
                                    color='k', 
                                    lw=alw)))

#                        axs[k].annotate('  ' + reg, 
#                                    (lats[i], maxes[i]),
#                        fontsize=fsize,color=palette[acronyms[i]])
                else:
                    texts.append(axs[k].annotate('  ' + reg, 
                    # f"{reg} {f[reg]['nclus']}" ,
                        (lats[i], maxes[i]),
                        fontsize=fsize,color=palette[acronyms[i]],
                        arrowprops=dict(arrowstyle="-", 
                                    color=palette[acronyms[i]], 
                                    lw=alw)))
                                    
        if auto_label_adjust:        
            adjust_text(texts,force_text=(3, 6),ax=axs[k])

        axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')
        
        ha = 'left'   
        
        axs[k].text(0, 0.95, align[split],
                      transform=axs[k].get_xaxis_transform(),
                      horizontalalignment = ha, rotation=90,
                      fontsize = f_size*0.8)
                                
        axs[k].spines['top'].set_visible(False)
        axs[k].spines['right'].set_visible(False)        
        if c == 0:   
            axs[k].set_ylabel(f'max dist. [Hz]')
        axs[k].set_xlabel('latency [sec]')
        axs[k].set_title(f"{tops[split+'_s']} sig")
        
        if c == 0:
            put_panel_label(axs[k], row)        

        
        if single_scat:
            figs[c][0].tight_layout()
            figs[c][0].savefig(Path(pth_res, 
                               f'scat_{split}.pdf'), dpi=300)         
            plt.close()
        
        c += 1     
        k += 1


#    fig = plt.gcf()
#    fig.tight_layout()
#    
#    fig.subplots_adjust(top=0.98,
#                        bottom=0.051,
#                        left=0.1,
#                        right=0.971,
#                        hspace=1.3,
#                        wspace=0.52)


    #fig.suptitle(f'd_{curve}')

#    fig.savefig(Path(pth_res,f'manifold_{curve}.pdf'), dpi=400)              


def plot_swanson_supp(curve = 'd_var_m', mapping = 'Beryl'):
    
    figs = plt.figure(figsize=(10, 9), constrained_layout=True)
    nrows = 3
    ncols = len(align) 
    gs = GridSpec(nrows, ncols, figure=figs)


    
    axs = []
    k = 0
    '''
    plot Swanson flatmap with labels and colors
    '''
    axs.append(figs.add_subplot(gs[0,:]))
    plot_swanson(annotate=True, ax=axs[k])
    axs[k].axis('off')
    put_panel_label(axs[k], k)
    k += 1
   
    '''
    max dist_split onto swanson flat maps
    (only regs with p < 0.01)
    '''
    
    c = 0
    for split in align:
    
        axs.append(figs.add_subplot(gs[1,c]))   
        c += 1
        d = np.load(Path(pth_res,f'{split}.npy'),
                    allow_pickle=True).flat[0] 

        # get significant regions only
        acronyms = [reg for reg in d
                if d[reg][f'p_{curve}'] < 0.05]

        values = np.array([d[x][amp_curve] for x in acronyms])
        
        print(split, acronyms, values)
             
        plot_swanson(list(acronyms), list(values), cmap='Blues', 
                     ax=axs[k], br=br)#, orientation='portrait')
        axs[k].axis('off')
        axs[k].set_title(f'{split} \n amplitude')
        put_panel_label(axs[k], k)
        k += 1

    '''
    lat onto swanson flat maps
    (only regs with p < 0.01)
    '''

    c = 0
    for split in align:
    
        axs.append(figs.add_subplot(gs[2,c]))   
        c += 1
        d = np.load(Path(pth_res,f'{split}.npy'),
                    allow_pickle=True).flat[0] 
                    
        # get significant regions only
        acronyms = [reg for reg in d
                if d[reg][f'p_{curve}'] < 0.05]

        #  compute latencies (inverted, shorter latency is darker)
        for x in acronyms:
           
            if np.max(d[x][curve]) == np.inf:
                loc = np.where(d[x][curve] == np.inf)[0]  
            else:
                loc = np.where(d[x][curve] > 
                               np.min(d[x][curve]) + 
                               0.7*(np.max(d[x][curve]) - 
                               np.min(d[x][curve])))[0]                 
                                

            xs = np.linspace(0, 
                             pre_post[split][0] + pre_post[split][1],
                             len(d[x][curve]))            

            d[x]['lat'] = xs[-1] - xs[loc[0]]

        values = np.array([d[x]['lat'] for x in acronyms])
                         
        plot_swanson(list(acronyms), list(values), cmap='Blues', 
                     ax=axs[k], br=br)#, orientation='portrait')
        axs[k].axis('off')
        axs[k].set_title(f'{split} \n latency (dark = early)')
        put_panel_label(axs[k], k)
        k += 1


        '''
        general subplots settings
        '''


#    figs.subplots_adjust(top=0.89,
#bottom=0.018,
#left=0.058,
#right=0.985,
#hspace=0.3,
#wspace=0.214)


def plot_cosmos_lines(curve = 'euc', single=True, 
                      ssplit='fback', sigl=0.01):

    if single:
        splits = [ssplit]
    else:
        splits = align

    mapping = 'Beryl'
    df, palette = get_allen_info()
    fig = plt.figure(figsize=(20, 15), constrained_layout=True)
     
    gs = GridSpec(len(splits), 7, figure=fig)

    sc = 0
    for split in splits:
    
        d = np.load(Path(pth_res,f'{split}.npy'),
                    allow_pickle=True).flat[0] 
                    
        # get significant regions only
        regsa = [reg for reg in d
                if d[reg][f'p_{curve}'] < sigl]
                                
        # get cosmos parent regions for Swanson acronyms 
        cosregs = [df[df['id'] == int(df[df['acronym']==reg
                    ]['structure_id_path']
                   .values[0].split('/')[4])]['acronym']
                   .values[0] for reg in regsa]
                   
        cosregsC = list(Counter(cosregs))
        cosregsC = sorted(cosregsC)         
        
        
        k = 0
        axs = []                 
        for cos in cosregsC:
            
            axs.append(fig.add_subplot(gs[sc,k]))
            regs = np.array(regsa)[np.array(cosregs)==cos]
            ma_x = max([d[reg][f'amp_{curve}'] for reg in regs])
            
            print(split, cos, regs)

            cc = 0
            texts = []
            for reg in regs:
                if any(np.isinf(d[reg][f'd_{curve}'])):
                    print(f'{curve} inf in {reg}')
                    continue
                    
                if single:
                    yy = d[reg][f'd_{curve}'] + cc * ma_x
                else:
                    yy = d[reg][f'd_{curve}']
                         
                xx = np.linspace(-pre_post[split][0], 
                                  pre_post[split][1], 
                                  len(d[reg][f'd_{curve}']))        

                axs[k].plot(xx,yy, linewidth = 2,
                              color=palette[reg], 
                              label=f"{reg}")
                              
                # put region labels                 
                y = yy[-1]
                x = xx[-1]
                ss = f"{reg} {d[reg]['nclus']}" 
                
                if cos in ['CBX', 'CBN']:  # darken yellow
                    texts.append(axs[k].text(x, y, ss, 
                                             color = 'k',
                                             fontsize=10))                
                
                texts.append(axs[k].text(
                             x, y, ss, 
                             color = palette[reg], fontsize=9))
                  
                cc +=1  
                                 
            #adjust_text(texts)
            
    
#            axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')
#            axs[k].text(0, 0.01, align[split],
#                          transform=axs[k].get_xaxis_transform(),
#                          horizontalalignment = 'right' if split == 'block'
#                          else 'left')           
            axs[k].spines['top'].set_visible(False)
            axs[k].spines['right'].set_visible(False)
            axs[k].set_ylabel('distance')
            axs[k].set_xlabel('time [sec]')
            axs[k].set_title(f'{split}, {cos}')
            if not single: put_panel_label(axs[k], k)

            k +=1
        sc +=1
    plt.tight_layout()

    if single:
        fig.savefig(Path(pth_res,f'lines_{curve}_{ssplit}.png'))
    else:
        fig.savefig(Path(pth_res,f'lines_{curve}_.png')) 


def plot_session_numbers():


    split = 'choice'
    mapping = 'Beryl'
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

    mapping = 'Beryl'      
    
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



def plot_all_3d(reg, split, mapping = 'Beryl'):

    '''
    for a given region and split
    plot 3d trajectory for all insertions
    '''


    d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'
                f'curves_{split}_{mapping}.npy',
                allow_pickle=True).flat[0]['pcs']
                
    print(d.keys())
    
    # loop through insertions for that region
    print(d[reg].keys())
    for ins in d[reg]:
        fig = plt.figure(figsize=(3,3))                   
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        npcs, nobs = d[reg][ins].shape
        nobs = nobs // nrand


        for j in range(ntravis):
        
            # 3d trajectory
            cs = d[reg][ins][:,nobs * j: nobs * (j+1)].T
    
            if split == 'fback':
                if j == 0:
                    col = ['green'] * nobs
                elif j == 1:
                    col = ['purple'] * nobs
                else:
                    col = ['gray'] * nobs
 
            else:
                if j == 0:
                    col = [blue_left] * nobs
                elif j == 1:
                    col = [red_right] * nobs
                else:
                    col = ['gray'] * nobs            
               
      
            p = ax.scatter(cs[:,0],cs[:,1],cs[:,2], color=col, 
                           s=10 if j in [0,1] else 0.1,
                           label=trial_split[split][j] if j in range(3)
                              else '_nolegend_', depthshade=False)
                              
            ax.plot(cs[:,0],cs[:,1],cs[:,2], color=col[0],
                    linewidth = 0.2)        

                              
        ax.set_xlabel('pc1')    
        ax.set_ylabel('pc2')
        ax.set_zlabel('pc3')
        eid, probe = ins.split('_')
        pa = one.eid2path(eid)
        n = ' '.join([str(pa).split('/')[i] for i in [6,7]])
        ax.set_title(f'{split}, {reg} \n {n} {probe}')   
        ax.grid(False)
        ax.axis('off') 
              
        ax.legend(frameon=False).set_draggable(True)
        
        
def inspect_regional_PETH(reg, split):
    '''
    for a given split and region, display PETHs as image,
    lines separating insertions
    '''
    mapping = 'Beryl'       
    print(split)
    R = np.load('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/manifold/'
                f'single_d_vars_{split}_{mapping}.npy', 
                allow_pickle=True).flat[0]

    # get stderror across cells for max-min/max+min
    # pool data for illustrative PCA
    acs = []
    ws = []
    sess = []

    k = 0
    for D_ in R['D_s']:

        acs.append(D_['acs'])
        ws.append(D_['ws'][:2])
        sess.append([R['eid_probe'][k]]*len(D_['acs']))
        k += 1
        
    sess = np.concatenate(sess)    
    acs = np.concatenate(acs)
    ws = np.concatenate(ws, axis=1)           
    
    # for the region under consideration plot temp plot
    dat = ws[:,acs == reg,:]
    sess0 = sess[acs == reg]

    #return dat    
    fig, axs = plt.subplots(ncols =2, nrows=1)    
    
    vmax = np.amax(dat)
    vmin = np.amin(dat)
                         
    for i in range(2): 
        axs[i].imshow(dat[i], cmap='Greys',aspect="auto",
            interpolation='none', vmax = vmax, vmin = vmin)
        
        for s in Counter(sess0):
            axs[i].axhline(y = np.where(sess0==s)[0][0], c = 'r', 
                       linestyle = '--') 
            #axs[i].annotate(s, (0, np.where(sess0==s)[0][0]), c = 'r')
        
        axs[i].set_ylabel('cells')
        axs[i].set_xlabel('time [sec]')
        axs[i].set_title(f'PETH, {trial_split[split][i]}')
        xs = np.linspace(0,pre_post[split][0] + pre_post[split][1],5)    
        axs[i].set_xticks(np.linspace(0,dat[i].shape[-1],5))
        axs[i].set_xticklabels(["{0:,.3f}".format(x) for x in xs])            
        
    fig.suptitle(f'{reg}, {split}')        
    fig.tight_layout()
#    fig.savefig('/home/mic/paper-brain-wide-map/'
#                f'manifold_analysis/figs/visps_lr/{reg}.png')
#    plt.close()               
        


def check_lr():
#    df = pd.read_csv('/home/mic/paper-brain-wide-map/'
#                    f'manifold_analysis/bwm_sess_regions.csv')
#                    
#    df2 = df[df['Beryl'].str.contains('VIS')].sort_values(
#                                'nclus',ascending=False)
#                                
#    pids = df2[df2['nclus']>10]['pid'].values                           


#    eids_plus = [np.concatenate([one.pid2eid(pid),[pid]]) for pid in pids]
    
    cell_off = [
           0.10091743, 0.1146789 , 0.10091743, 0.1146789 , 0.10091743,
           0.10091743, 0.10550459, 0.08715596, 0.0733945 , 0.06422018,
           0.05963303, 0.06422018, 0.0733945 , 0.07798165, 0.08715596,
           0.08715596, 0.09633028, 0.09633028, 0.08256881, 0.10091743,
           0.10091743, 0.09633028, 0.10550459, 0.10091743, 0.1146789 ,
           0.09174312, 0.09633028, 0.08715596, 0.08715596, 0.09174312,
           0.08715596, 0.09174312, 0.10091743, 0.1146789 , 0.10550459,
           0.1146789 , 0.10550459, 0.11009174, 0.09174312, 0.08715596,
           0.07798165, 0.06422018, 0.07798165, 0.06422018, 0.06422018,
           0.05963303, 0.05963303, 0.06422018, 0.05963303, 0.05963303,
           0.06422018, 0.05504587, 0.05963303, 0.04587156, 0.04587156,
           0.05045872, 0.03669725, 0.04587156, 0.0412844 , 0.04587156,
           0.03211009, 0.04587156, 0.06880734, 0.0733945 , 0.07798165,
           0.0733945 , 0.09633028, 0.07798165, 0.0733945 , 0.06880734,
           0.0733945 , 0.08256881]    

    cell_on = [    
           0.12727273, 0.12272727, 0.12272727, 0.12727273, 0.10454545,
           0.11818182, 0.13181818, 0.14090909, 0.15      , 0.14545455,
           0.15      , 0.13636364, 0.10909091, 0.10909091, 0.11363636,
           0.13181818, 0.14090909, 0.15909091, 0.19090909, 0.20454545,
           0.24545455, 0.27727273, 0.31818182, 0.32727273, 0.35909091,
           0.37727273, 0.4       , 0.43636364, 0.41363636, 0.42272727,
           0.44090909, 0.45454545, 0.42272727, 0.38181818, 0.42727273,
           0.42727273, 0.40909091, 0.41818182, 0.43636364, 0.42272727,
           0.42272727, 0.42727273, 0.37727273, 0.37272727, 0.35909091,
           0.35454545, 0.33181818, 0.29090909, 0.31818182, 0.26818182,
           0.24545455, 0.23636364, 0.2       , 0.20454545, 0.2       ,
           0.21363636, 0.20454545, 0.20454545, 0.21363636, 0.2       ,
           0.19090909, 0.19545455, 0.19545455, 0.18181818, 0.17272727,
           0.19545455, 0.17727273, 0.16363636, 0.15      , 0.16363636,
           0.15      , 0.14090909]


    # 100 one orientation (2, 100, 72)
    ws = np.array([np.array([cell_on for i in range(100)]),
                         np.array([cell_off for i in range(100)])])


    # 200 both orientations
    ws_r = np.array([np.vstack([np.array([cell_on for i in range(10)]),
                                np.array([cell_off for i in range(90)])]),
                     np.vstack([np.array([cell_off for i in range(90)]), 
                                np.array([cell_on for i in range(10)])])])


    split = 'stim'

    labs = ['one hemisphere', 'mixed hemispheres']
    k = 0 
    
    fig0, ax0 = plt.subplots(figsize=(3,3))
    for dat in [ws, ws_r]:
    
        fig, axs = plt.subplots(ncols =2, nrows=1,figsize=(3,3))
        for i in range(2): 
            axs[i].imshow(dat[i], cmap='Greys',aspect="auto",
                interpolation='none', vmin=min(cell_off), vmax=max(cell_on))
            axs[i].set_ylabel('cells')
            axs[i].set_xlabel('time [sec]')
            axs[i].set_title(f'PETH, {trial_split[split][i]}')
        fig.suptitle(labs[k])   
        fig.tight_layout()
        pca = PCA(n_components = 30)
        wsc = pca.fit_transform(np.concatenate(dat,axis=1).T).T
        #pcs[reg] = wsc[:3]
        
        ntr, nclus, nobs = dat.shape
        
        dists = []
        for tr in range(ntr // 2):
            t0 = wsc[:,nobs * tr*2 :nobs * (tr*2 + 1)] 
            t1 = wsc[:,nobs * (tr*2 + 1):nobs * (tr*2 + 2)] # actual trajectories
            
            dists.append(sum((t0 - t1)**2)**0.5)

        ax0.plot(dists[0],label=labs[k])
        k += 1
    fig0.legend()
    fig0.tight_layout()
    ax0.set_xlabel('time')
    ax0.set_ylabel('trajectory distance')


def plot_custom_lines(curve = 'euc', split='fback'):


    '''
    distance line plots for select regions;
    Welch psd as insets for each
    
    for supp figure on oscillations
    '''
    
   
    regs = ['MV', 'MRN', 'APN', 'SSp-m', 
            'SIM', 'PRM', 'PoT', 'MEA', 'ANcr2']
            
    mapping = 'Beryl'
    df, palette = get_allen_info()
    nr = 3
    fig, axs = plt.subplots(nrows =nr, ncols = int(np.ceil(len(regs)/nr)), 
                            figsize=(7, 7),
                            sharex=True, constrained_layout=True)
    axs = axs.flatten()
    axsi = []  # inset axes
    
    # get results
    d = np.load(Path(pth_res,f'{split}.npy'),
                allow_pickle=True).flat[0] 
                
    # get cosmos parent regions for Swanson acronyms 
    cosregs = {reg: df[df['id'] == int(df[df['acronym']==reg
                ]['structure_id_path']
               .values[0].split('/')[4])]['acronym']
               .values[0] for reg in regs}
               
    
    k = 0                 
    for reg in regs:
     
        yy = d[reg][f'd_{curve}']
                 
        xx = np.linspace(-pre_post[split][0], 
                          pre_post[split][1], 
                          len(d[reg][f'd_{curve}']))        

        axs[k].plot(xx,yy, linewidth = 2,
                      color=palette[reg], 
                      label=f"{reg}")
                      
        # put region labels                 
        y = yy[-1]
        x = xx[-1]
        ss = f"{reg} {d[reg]['nclus']}" 
        
        cos =  cosregs[reg]
#        if cos in ['CBX', 'CBN']:  # darken yellow
#            axs[k].text(x, y, ss,color = 'k', fontsize=10)               
#        
#        axs[k].text(x, y, ss, color = palette[reg], fontsize=9)
      
        axs[k].spines['top'].set_visible(False)
        axs[k].spines['right'].set_visible(False)
        axs[k].set_ylabel('distance')
        axs[k].set_xlabel('time [sec]')
        axs[k].set_title(ss,color = palette[reg] if cos not in ['CBX', 'CBN']
                         else 'k')

                               
        f, psd = signal.welch(yy,
                              fs=int(len(xx)/(xx[-1] - xx[0])))
                              
        with plt.rc_context({'font.size': 0.8*plt.rcParams['font.size']}):
            # plot psd as inset 
            axsi.append(inset_axes(axs[k], width="30%", height="35%", 
                                   loc=4 if reg != 'PoT' else 1,
                                   borderpad=1,
                                   bbox_to_anchor=(-0.02,0.1,1,1), 
                                   bbox_transform=axs[k].transAxes))
                                   #

            axsi[-1].plot(f,psd)
            axsi[-1].set_xlim(3, 40)
            axsi[-1].set_xlabel('f [Hz]')
            Axis.set_label_coords(axsi[-1].xaxis,-0.3, -0.05)
            
            axsi[-1].set_ylabel('psd ')        
            axsi[-1].spines['top'].set_visible(False)
            axsi[-1].spines['right'].set_visible(False)        
            axsi[-1].set_yticks([])
            axsi[-1].set_yticklabels([])
            if k > 0:
                axsi[-1].sharex(axsi[-2])
        
        k +=1

    #plt.tight_layout()
    
    
    
def get_average_peth():

    split = 'stim'
    res = {}
    
    for split in align:
        r = {}
    
        pth = Path(one.cache_dir, 'manifold', split) 
        ss = os.listdir(pth)  # get insertions
        
        ws = [] 
        
        # group results across insertions
        for s in ss:
        
            D_ = np.load(Path(pth,s), 
                        allow_pickle=True).flat[0]
                           
            ws.append(D_['ws'])        
            

        ws = np.concatenate(ws, axis=1)
        ntr, ncells, nt = ws.shape

        r['m0'] = np.mean(ws[0], axis=0)
        r['m1'] = np.mean(ws[1], axis=0)
        r['ms'] = np.mean(ws[2:], axis=(0,1))

        r['v0'] = np.std(ws[0], axis=0)/(ncells**0.5)
        r['v1'] = np.std(ws[1], axis=0)/(ncells**0.5)
        r['vs'] = np.std(ws[2:], axis=(0,1))/(ncells**0.5)
        
        r['euc'] = np.mean((((ws[0] - ws[1])**2)/ncells)**0.5, axis=0) 
        

        res[split] = r

    np.save(Path(pth_res,'grand_averages.npy'), res,
                allow_pickle=True)

def plot_grand_average():

    fig, axs = plt.subplots(nrows=2,
                            ncols=len(align))
    
    res = np.load(Path(pth_res,'grand_averages.npy'),
                  allow_pickle=True).flat[0]
    
    cols = {'0':blue_left,
            '1':red_right,
            's': 'gray'}
    
    k = 0
    for split in res:
        r = 0
        for t in ['0', '1', 's']:
            yy = res[split][f'm{t}']
            xx = np.linspace(-pre_post[split][0], 
                      pre_post[split][1], 
                      len(yy))
                                          
            axs[r,k].plot(xx,yy, c = cols[t])   
            axs[r,k].fill_between(xx,    
                         yy + res[split][f'v{t}']/2,
                         yy - res[split][f'v{t}']/2,
                         color=cols[t],
                         alpha = 0.2)
                                
        axs[r,k].axvline(x=0, lw=0.5, linestyle='--', c='k')
        axs[r,k].spines['top'].set_visible(False)
        axs[r,k].spines['right'].set_visible(False)
        if k == 0:        
            axs[r,k].set_ylabel(f'average firing rate [Hz]')
        axs[r,k].set_xlabel('time [sec]')
        axs[r,k].set_title(split)    
        
        r = 1
        
        #yy = np.abs(res[split][f'm0'] - res[split][f'm1'])
        yy = res[split]['euc']
        xx = np.linspace(-pre_post[split][0], 
                  pre_post[split][1], 
                  len(yy))
                                      
        axs[r,k].plot(xx,yy, c = 'k')   

                                
        axs[r,k].axvline(x=0, lw=0.5, linestyle='--', c='k')
        axs[r,k].spines['top'].set_visible(False)
        axs[r,k].spines['right'].set_visible(False)
        if k == 0:        
            axs[r,k].set_ylabel(f'distance [Hz]')
        axs[r,k].set_xlabel('time [sec]')
        axs[r,k].set_title(split) 
        
        axs[0,k].sharex(axs[1,k])
        
        if k > 0:
            axs[r,k].sharey(axs[r,k-1])
            
          
        k +=1           
    
    fig.tight_layout()
    
    
    
    
