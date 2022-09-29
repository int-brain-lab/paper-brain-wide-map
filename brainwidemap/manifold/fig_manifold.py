from one.api import ONE
from reproducible_ephys_processing import bin_spikes2D
from brainwidemap import bwm_query, load_good_units
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
from brainbox.io.one import SpikeSortingLoader, SessionLoader

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
from pathlib import Path
import glob
from dateutil import parser

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

import random
from random import shuffle
import time

import math
import string

import cProfile
import pstats

import warnings
warnings.filterwarnings("ignore")


blue_left = [0.13850039, 0.41331206, 0.74052025]
red_right = [0.66080672, 0.21526712, 0.23069468]

c = 0.0125  # 0.005 sec for a static bin size, or None for single bin
sts = 0.002  # stride size in s for overlapping bins 
ntravis = 30  # number of trajectories for visualisation, first 2 real
nrand = 200  # 100, takes 10 h number of random trial splits for control
min_reg = 100  # 100, minimum number of neurons in pooled region


def T_BIN(split, c=c):

    # c = 0.005 # time bin size in seconds (5 ms)
    if c is None:
        return pre_post[split][0] + pre_post[split][1]
    else:
        return c    

# trial split types, see get_d_vars for details       
align = {
         'block':'stim on',
         'stim':'stim on',
         'choice':'motion on',
         'fback':'feedback'}
                  
# 'action':'motion on',         
#'block_stim_r':'stim on',
#'block_stim_l':'stim on',

# [pre_time, post_time]
pre_post = {'choice':[0.15,0],'stim':[0,0.15],
            'fback':[0,0.15],'block':[0.4,-0.1],
            'block_stim_r':[0,0.15], 'block_stim_l':[0,0.15]}

            #'action':[0.025,0.3]}  

# labels for illustrative trajectory legend 
trial_split = {'choice':['choice left', 'choice right','pseudo'],
               'stim':['stim left','stim right','pseudo'],
               'fback':['correct','false','pseudo'],
               'block':['pleft 0.8','pleft 0.2','pseudo'],
               'action':['choice left', 'choice right','pseudo'],
               'block_stim_r':['pleft 0.8','pleft 0.2','pseudo'],            
               'block_stim_l':['pleft 0.8','pleft 0.2','pseudo']}

one = ONE()  # (mode='local')
ba = AllenAtlas()
br = BrainRegions()


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


def compute_impostor_behavior():

    '''
    for a given split, get the behavior of 
    5 random concatenated sessions to match length
    for block there is the pseudo session method
    
    eid_no is the one eid to exclude
    
    SOON TO BE REPLACED BY CHARLES' MODEL
    '''
    
    df = bwm_query(one)
    
    eids_plus = list(set(df['eid'].values))
    
    R = {}
    for split in align:
        d = {}
        for eid in eids_plus: 
            try:  
                


                # Load in trials data
                trials = one.load_object(eid, 'trials', collection='alf')
                   
                # remove certain trials    
                # discard trials were feedback - stim is outside that range [sec]
                rs_range = [0.08, 2]
                stim_diff = trials['feedback_times'] - trials['stimOn_times']     
       
                rm_trials = np.bitwise_or.reduce([np.isnan(trials['stimOn_times']),
                                           np.isnan(trials['choice']),
                                           np.isnan(trials['feedback_times']),
                                           np.isnan(trials['probabilityLeft']),
                                           np.isnan(trials['firstMovement_times']),
                                           np.isnan(trials['feedbackType']),
                                           stim_diff > rs_range[-1],
                                           stim_diff < rs_range[0]])
                       
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


def get_d_vars(split, pid, 
               mapping='Beryl', control=True):

    '''
    for a given session, probe, bin neural activity
    cut into trials, compute d_var per region
    '''    

    
    eid,probe = one.pid2eid(pid)
#    # Load in spikesorting
#    sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
#    spikes, clusters, channels = sl.load_spike_sorting()
#    clusters = sl.merge_clusters(spikes, clusters, channels)
#    
#    #only good units
#    clusters_labeled = clusters.to_df()
#    good_clusters = clusters_labeled[clusters_labeled['label']==1]
#    good_clusters.reset_index(drop=True, inplace=True)
#    clusters = good_clusters
#    # Find spikes that are from the clusterIDs
#    spike_idx = np.isin(spikes['clusters'], clusters['cluster_id'])

    
    # load in spikes
    spikes, clusters = load_good_units(one, pid)    
    


    # Load in trials data

    #trials = one.load_object(eid, 'trials', collection='alf')
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
            trn.append(np.arange(len(trials['stimOn_times']))[np.bitwise_and.reduce([ 
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
                       
    elif split == 'block_stim_l':
        for pleft in [0.8, 0.2]:
            events.append(trials['stimOn_times'][np.bitwise_and.reduce([
                ~rm_trials,trials['probabilityLeft'] == pleft,
                np.isnan(trials[f'contrastRight'])])])
            trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                       ~rm_trials,trials['probabilityLeft'] == pleft,
                       np.isnan(trials[f'contrastRight'])])])
                       
    elif split == 'block_stim_r':
        for pleft in [0.8, 0.2]:
            events.append(trials['stimOn_times'][np.bitwise_and.reduce([
                ~rm_trials,trials['probabilityLeft'] == pleft,
                np.isnan(trials[f'contrastLeft'])])])
            trn.append(np.arange(len(trials['choice']))[np.bitwise_and.reduce([
                       ~rm_trials,trials['probabilityLeft'] == pleft,
                       np.isnan(trials[f'contrastLeft'])])])
                       
    else:
        print('what is the split?', split)
        return

    print('#trials per condition: ',len(trn[0]), len(trn[1]))
    assert (len(trn[0]) != 0) and (len(trn[0]) != 0), 'zero trials to average'
           
    assert len(spikes['times']) == len(spikes['clusters']), 'spikes != clusters'    
            
    # bin and cut into trials    
    bins = []

    for event in events:
    
        #  overlapping time bins, bin size = T_BIN, stride = sts 
        bis = []
        st = int(T_BIN(split)//sts) 
        
        for ts in range(st):
    
            bi, _ = bin_spikes2D(spikes['times'],#[spike_idx], 
                               clusters['cluster_id'][spikes['clusters']],
                                #spikes['clusters'][spike_idx],
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
    
    #  recreate temporal trial order              
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

    # discard cells if not in csv list (at least 10 clus & at least 2 recs)
    gregs = set(np.unique(pd.read_csv('/home/mic/paper-brain-wide-map/'
                                      'manifold_analysis/'
                                      'bwm_sess_regions.csv')['Beryl']))
    acs_ = set(np.unique(acs)).intersection(gregs)
    goodcells = np.bitwise_or.reduce([acs == reg for reg in acs_])
    
    acs = acs[goodcells]
    b = b[:,goodcells,:]
    bins2 = [x[:,goodcells,:] for x in bins]
    bins = bins2
    
    # return trials, trn, ntr

    if control:
        # get mean and var across trials
        w0 = [bi.mean(axis=0) for bi in bins]  
        s0 = [bi.var(axis=0) for bi in bins]
        
        
        if not 'block' in split:
            #  Load impostor behavior
            spl = np.load('/home/mic/paper-brain-wide-map/'
                          'manifold_analysis/bwm_behave.npy',
                          allow_pickle=True).flat[0][split]
            
            #  exclude current session if present
            if eid in list(spl.keys()):
                del spl[eid]    
        
        # nrand times random impostor/pseudo split of trials 
        for i in range(nrand):
            if split == 'block':  #'block' pseudo sessions
                ys = generate_pseudo_blocks(ntr, first5050=0) == 0.8
                
            elif 'block_stim' in split:  # shuffle uniformly 
                assert ntr == (len(trn[0]) + len(trn[1])), 'ntr??'    
                ys = np.concatenate([np.full(len(trn[0]), True),
                                     np.full(len(trn[1]), False)])               
                shuffle(ys)                
            
            elif split == 'stim':
                # get real block labels; use to generate stim side
                y_ = trials['probabilityLeft'][sorted(dx[:, 1])].values
                o = np.random.uniform(low=0, high=1, size=(ntr,))
                ys = o < y_                
                
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
    D_ = {}
    D_['acs'] = acs
    D_['d_vars'] = d_var
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
                
        for j in range(len(ws_)//2):


            # strictly standardized mean difference
            d_var = (((ws_[2*j] - ws_[2*j + 1])/
                      ((ss_[2*j] + ss_[2*j + 1])**0.5))**2)

            # sum over cells, divide by #neu later
            d_var_m = np.nansum(d_var,axis=0)
            d_vars.append(d_var_m)

        res['d_vars'] = d_vars

        D[reg] = res
        
    return D, D_    



'''    
###
### bulk processing 
###    
''' 

def get_BWM_region_pid_pairs():

    '''
    get all BWM insertion/region pairs, then filter for those with
    at least two insertions and at least 10 neurons;
    takes 1 h min to run (loading clusters is the slow bit)
    '''

    columns = ['pid', 'Beryl', 'nclus']        
    data = []    
    
    df = bwm_query(one)
    pids = df['pid'].values
    k = 0
    for pid in pids:
        spikes, clusters = load_good_units(one, pid)
        acs = br.id2acronym(clusters['atlas_id'],
                        mapping='Beryl')
                        
        # discard cells in regions root or void
        goodcells = np.bitwise_and(acs != 'void', acs != 'root')

        acs = Counter(acs[goodcells])                        
                        
        for reg in acs:
            data.append([pid, reg, acs[reg]])

        print(pid, f'{k} of {len(pids)} done')

    df = pd.DataFrame(data, columns=columns)

    # count number of insertions for each region
    nsC = Counter(df['Beryl'].values)
    ns = [nsC[reg] for reg in df['Beryl'].values]
    df['#pids'] = ns
    
    df1 = df[df['nclus'] > 9]  # at least 10 neurons
    df2 = df1[df1['#pids'] > 1]  # at least 2 recordings
    
    # df2.to_csv('bwm_sess_regions.csv')
    return df, df2    
    


def get_all_d_vars(split, eids_plus = None, control = True, 
                   mapping='Beryl'):

    '''
    for all BWM insertions, get the PSTHs and acronyms,
    '''
    
    time00 = time.perf_counter()
    
    print('split', split, 'control', control)
    
    if eids_plus == None:
        df = bwm_query(one)
        eids_plus = df[['eid','probe_name', 'pid']].values

    # exclude pids that are not in csv list (min 10 and double ins)
    gpids = np.unique(pd.read_csv('/home/mic/paper-brain-wide-map/'
                                  'manifold_analysis/'
                                  'bwm_sess_regions.csv')['pid'])    

    Fs = []
    eid_probe = []
    Ds = []
    D_s = []   
    k=0
    print(f'Processing {len(gpids)} of {len(eids_plus)} insertions')
    for i in eids_plus:
        eid, probe, pid = i  
                
        if pid not in gpids:
            continue    
          
        time0 = time.perf_counter()
        try:
            if not control:
                D_ = get_d_vars(split, pid, control=control, mapping=mapping)
                D_s.append(D_)            
            else:
                D, D_ = get_d_vars(split, pid, control=control, mapping=mapping)
                Ds.append(D)             
                D_s.append(D_)
                                         
            eid_probe.append(eid+'_'+probe)
            gc.collect() 
            print(k+1, 'of', len(eids_plus), 'ok') 
        except:
            Fs.append(pid)
            gc.collect()
            print(k+1, 'of', len(eids_plus), 'fail', pid)
            
        time1 = time.perf_counter()
        print(time1 - time0, 'sec')
                    
        k+=1            
    
    
    if not control:
        R_ = {'D_s':D_s, 'eid_probe':eid_probe} 
        
        np.save('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/manifold/'
               f'single_d_vars_{split}_{mapping}.npy', 
               R_, allow_pickle=True)    
    
    else:     
        R = {'Ds':Ds, 'eid_probe':eid_probe} 
        
        np.save('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/manifold/'
               f'd_vars_{split}_{mapping}.npy', R, allow_pickle=True)
               
        R_ = {'D_s':D_s, 'eid_probe':eid_probe} 
        
        np.save('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/manifold/'
               f'single_d_vars_{split}_{mapping}.npy', 
               R_, allow_pickle=True)
               
    time11 = time.perf_counter()
    print((time11 - time00)/60, 'min for the complete bwm set')
    print(f'{len(Fs)}, load failures:')
    print(Fs)

    
def d_var_stacked(split, mapping='Beryl'):
                  
    time0 = time.perf_counter()

    '''
    average d_var_m via nanmean across insertions,
    remove regions with too few neurons (min_reg)
    compute maxes, latencies and p-values
    
    load in single_d_vars for PCA and std of d_var maxes across cells 
    '''
  
    print(split)
    R = np.load('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/manifold/'
                f'single_d_vars_{split}_{mapping}.npy', 
                allow_pickle=True).flat[0]

    # get stderror across cells for max-min/max+min
    # pool data for illustrative PCA
    maxes = []
    acs = []
    ws = []
    for D_ in R['D_s']:
        ma = np.nanmax(D_['d_vars'],axis=1)    
        mi = np.nanmin(D_['d_vars'],axis=1)  
        maxes.append((ma-mi)/(ma+mi))
        acs.append(D_['acs'])
        ws.append(D_['ws'])
        
    maxes = np.concatenate(maxes)
    acs = np.concatenate(acs)
    ws = np.concatenate(ws, axis=1)           

    # remove all cells that have a nan max
    keep = ~np.isnan(maxes)
    acs = acs[keep]
    maxes = maxes[keep]
    ws = ws[:,keep,:]
       
    # for each region get standard error
    regs = Counter(acs)
    stde = {}
    pcs = {}
    euc = {}  # also compute Euclidean distance on PCA-reduced trajectories
    p_euc = {}
    for reg in regs:         
                
        if (reg in ['root', 'void']) or (regs[reg] < min_reg):
            continue
            
        stde[reg] = np.std(maxes[acs == reg])/np.sqrt(regs[reg])

        dat = ws[:,acs == reg,:]
        pca = PCA(n_components = 30)
        wsc = pca.fit_transform(np.concatenate(dat,axis=1).T).T
        pcs[reg] = wsc[:3]
        
        nclus = sum(acs == reg)
        
        nobs = wsc.shape[1] // ntravis
        
        dists = []
        for tr in range(ntravis // 2):
            t0 = wsc[:,nobs * tr*2 :nobs * (tr*2 + 1)] 
            t1 = wsc[:,nobs * (tr*2 + 1):nobs * (tr*2 + 2)] # actual trajectories
            
            dists.append(sum((t0 - t1)**2)**0.5)
            
        euc[reg] = dists[0]
        
        # get p-value for pca based Euclidean distance
        amps = [((np.max(x) - np.min(x))/
                (np.max(x) + np.min(x))) for x in dists]
                          
        p_euc[reg] = 1 - (0.01 * 
                     percentileofscore(amps[1:],amps[0],kind='weak'))
        
        
    # getting d_vars params for controls (extra file for storage)    
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
        regd[reg] = (np.nansum(np.array(regd[reg]),axis=0)/
                     nclus[reg])**0.5

    r = {}
    for reg in regd:
        res = {}

        # nclus
        res['nclus'] = nclus[reg]
        res['stde'] = stde[reg]
        res['pcs'] = pcs[reg]
        res['euc'] = euc[reg]
        res['p_euc'] = p_euc[reg]
        
        # full curve
        d_var_m = regd[reg][0]
        res['d_var_m'] = d_var_m

        # amplitudes
        amps = [((np.max(x) - np.min(x))/
                 (np.max(x) + np.min(x)))  for x in regd[reg]]
        res['max-min/max+min'] = amps[0]
        
        # p value
        null_d = amps[1:]
        p = 1 - (0.01 * percentileofscore(null_d,amps[0],kind='weak'))
        res['p'] = p
        
        # latency  
        if np.max(d_var_m) == np.inf:
            loc = np.where(d_var_m == np.inf)[0]  
        else:
            loc = np.where(d_var_m > 
                           np.min(d_var_m) + 0.7*(np.max(d_var_m) - 
                           np.min(d_var_m)))[0]
        
        res['lat'] = np.linspace(-pre_post[split][0], 
                        pre_post[split][1], len(d_var_m))[loc[0]]
                      
                      
        # amplitude and latitude for Euclidean distance also; no p                
        res['amp_euc'] = ((np.max(euc[reg]) - np.min(euc[reg]))/
                          (np.max(euc[reg]) + np.min(euc[reg])))
        
        loc = np.where(euc[reg] > 
               np.min(euc[reg]) + 0.7*(np.max(euc[reg]) - 
               np.min(euc[reg])))[0]
                                 
        res['lat_euc'] = np.linspace(-pre_post[split][0], 
                        pre_post[split][1], len(euc[reg]))[loc[0]]
                        
        r[reg] = res
    
    np.save('/home/mic/paper-brain-wide-map/manifold_analysis/'
           f'curves_{split}_{mapping}.npy', 
           r, allow_pickle=True)
           
    time1 = time.perf_counter()    
    print('total time:', time1 - time0, 'sec')



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


def put_panel_label(ax, k):                    
        ax.annotate(string.ascii_lowercase[k],(-0.05, 1.15),
                        xycoords='axes fraction',
                        fontsize=16, va='top', ha='right', weight='bold')


def plot_all(curve='euc', amp_number=False, 
             mapping='Beryl', top_amps=False):

    '''
    main figure: show example trajectories,
    d_var_m for select regions and scatter all sigificant regions
    Instead of d_var_m there is also euc
    '''

    plt.rcParams['font.size'] = '11'
    plt.rcParams['figure.constrained_layout.use'] = True
    
    nrows = 12
        
    fig = plt.figure(figsize=(10, 15))
    gs = fig.add_gridspec(nrows, len(align))
    
    _, palette = get_allen_info()
                   

    axs = []
    k = 0
    
    amp_curve = ('max-min/max+min' if curve == 'd_var_m' 
                         else 'amp_euc')
                         
    p_type = 'p' if curve == 'd_var_m' else 'p_euc'         
    
    
    '''
    get significant regions and high pass threshold for amp
    '''           
    tops = {}
    lower = {}
    for split in align: 
    
        d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'         
                    f'curves_{split}_{mapping}.npy',
                    allow_pickle=True).flat[0]              
             
        maxs = np.array([d[x][amp_curve] for x in d])
        acronyms = np.array(list(d.keys()))
        order = list(reversed(np.argsort(maxs)))
        maxs = maxs[order]
        acronyms = acronyms[order] 
             
        tops[split] = [acronyms, 
                      [d[reg][p_type] for reg in acronyms]]

            
        maxs = np.array([d[reg][amp_curve] for reg in acronyms
                         if d[reg][p_type] < 0.05])
                         
        maxsf = [v for v in maxs if not (math.isinf(v) or math.isnan(v))] 
               
               
        if maxsf == []:
            lower[split] = 0
        else:        
            lower[split] = np.percentile(maxsf, 25)          
        print(split, curve)
        print('25 percentile: ',np.round(lower[split],3))
        print(f'{len(maxsf)}/{len(d)} are significant')
        tops[split+'_s'] = f'{len(maxsf)} of {len(d)}'
        print([tops[split][0][j] for j in range(len(tops[split][0]))
                if tops[split][1][j] < 0.05])
        
    
    '''
    load schematic intro and contrast trajectory plot (convert from svg)
    '''
    
    for imn in ['intro', 'contrast_HB']:
        
        if imn == 'intro':
            axs.append(fig.add_subplot(gs[:3,0:2]))
        else:
            axs.append(fig.add_subplot(gs[:3,-2:]))            

        data_path = '/home/mic/paper-brain-wide-map/manifold_analysis/'
        pdf = fitz.open(data_path + f'{imn}.pdf')
        rgb = pdf[0].get_pixmap(dpi=600)
        pil_image = Image.open(io.BytesIO(rgb.tobytes()))
                
        imgplot = axs[k].imshow(pil_image.convert('RGB'))
        axs[k].axis('off')
        put_panel_label(axs[k], k)
        k += 1


    '''
    example regions per split for embedded space and line plots
    '''
    
    exs = {'stim': ['VISp', 'VISpm', 'VISl', 'GRN', 'MOs'],
           'choice': ['GRN', 'VISl', 'SSp-ul', 'SSs', 'IRN'],
           'fback': ['AUDp', 'CA1', 'EPd', 'SPVI', 'SSp-ul'],
           'block': ['ACAv', 'SPVI', 'IC', 'Eth', 'RSPagl']} 
#           'block_stim_l': 'CA1',
#           'block_stim_r': 'VPL'} #VPL 
                  
    if top_amps:  # pick max amp region as example
        exs = {split: [tops[split][0][j] for j in 
               range(len(tops[split][0])) if 
               tops[split][1][j] < 0.05][0] for split in align}   

    '''
    Trajectories for example regions in PCA embedded 3d space
    '''
                     
    c = 0
    for split in align:
        d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'
                f'curves_{split}_{mapping}.npy',
                allow_pickle=True).flat[0]

        # pick example region
        reg = exs[split][0]
        
        axs.append(fig.add_subplot(gs[3:6,c],
                                   projection='3d'))           

        npcs, nobs = d[reg]['pcs'].shape
        nobs = nobs // ntravis

        for j in range(ntravis):
        
            # 3d trajectory
            cs = d[reg]['pcs'][:,nobs * j: nobs * (j+1)].T
    
            if split == 'fback':
                if j == 0:
                    col = grad('Greens_r', nobs)# ['green'] * nobs
                elif j == 1:
                    col = grad('Purples_r', nobs)
                else:
                    col =  grad('Greys_r', nobs)
 
            else:
                if j == 0:
                    col = grad('Blues_r', nobs)
                elif j == 1:
                    col = grad('Reds_r', nobs)
                else:
                    col = grad('Greys_r', nobs)
                    
                            

            axs[k].plot(cs[:,0],cs[:,1],cs[:,2], color=col[len(col)//2],
                    linewidth = 2 if j in [0,1] else 1,
                    label=trial_split[split][j] if j in range(3)
                              else '_nolegend_')
                          
            p = axs[k].scatter(cs[:,0],cs[:,1],cs[:,2], color=col,
                           edgecolors = col, 
                           s=20 if j in [0,1] else 1,
                           depthshade=False)


        # add triplet of arrows as coordinate system?

        axs[k].set_title(f"{split}, {reg} {d[reg]['nclus']}")   
        axs[k].grid(False)
        #axs[k].axis('off')
        
        axs[k].xaxis.set_ticklabels([])
        axs[k].yaxis.set_ticklabels([])
        axs[k].zaxis.set_ticklabels([])
            
        axs[k].xaxis.labelpad=-12
        axs[k].yaxis.labelpad=-12
        axs[k].zaxis.labelpad=-12            
            
        axs[k].set_xlabel('pc1')
        axs[k].set_ylabel('pc2')
        axs[k].set_zlabel('pc3')
        
        # remove box panes
        axs[k].xaxis.pane.fill = False
        axs[k].yaxis.pane.fill = False
        axs[k].zaxis.pane.fill = False
        axs[k].xaxis.pane.set_edgecolor('w')
        axs[k].yaxis.pane.set_edgecolor('w')
        axs[k].zaxis.pane.set_edgecolor('w')
   
               
        '''
        draw coordinate system with pc1, pc2, pc3 labels
        '''
  
        
        handles, labels =  axs[k].get_legend_handles_labels()
        updated_handles = []
        for handle in handles:
            updated_handles.append(mpatches.Patch(
                                   color=handle.get_markerfacecolor(),
                                   label=handle.get_label()))
                                   
        by_label = dict(sorted(dict(zip(labels,
                        updated_handles)).items()))
                        
        axs[k].legend(by_label.values(), by_label.keys(),
                      frameon=False, fontsize=6).set_draggable(True)


        put_panel_label(axs[k], k)

        c += 1
        k += 1
      
    '''
    line plot per 5 example regions per split
    '''
    c = 0  # column index


    for split in align:
        if c == 0:
            axs.append(fig.add_subplot(gs[6:8,c]))
        else:  # to share y axis
            axs.append(fig.add_subplot(gs[6:8,c],sharey=axs[len(align)+2]))    
            
            
        f = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'
                    f'curves_{split}_{mapping}.npy',
                    allow_pickle=True).flat[0]
                   
        # example regions to illustrate line plots              
        regs = exs[split]
        
        texts = []          
        for reg in regs:
            if any(np.isinf(f[reg][curve])):
                print(f'inf in {curve} of {reg}')
                continue


            xx = np.linspace(-pre_post[split][0], 
                              pre_post[split][1], len(f[reg][curve]))
                                    
            # normalize curve
            yy = ((f[reg][curve] - min(f[reg][curve]))/
                  (max(f[reg][curve]) + min(f[reg][curve])))
            
            #yy = f[reg][curve]
            
            axs[k].plot(xx,yy, linewidth = 2,
                          color=palette[reg], 
                          label=f"{reg} {f[reg]['nclus']}")
            
            if curve == 'd_var_m':
                # plot stderror bars on lines
                axs[k].fill_between(xx,    
                                 yy + f[reg]['stde'],
                                 yy - f[reg]['stde'],
                                 color=palette[reg],
                                 alpha = 0.2)
                             
            # put region labels                 
            y = np.max(yy)
            x = xx[np.argmax(yy)]
            ss = f"{reg} {f[reg]['nclus']}" 
            
            texts.append(axs[k].text(x, y, ss, 
                                     color = palette[reg],
                                     fontsize=9))                 
                              
        adjust_text(texts)                      

        axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')
        
        if split == 'block':
            ha = 'right'
        else:
            ha = 'left'    
        
        axs[k].text(0, 0.01, align[split],
                      transform=axs[k].get_xaxis_transform(),
                      horizontalalignment = ha)           
        

        axs[k].spines['top'].set_visible(False)
        axs[k].spines['right'].set_visible(False)
        if c == 0:        
            axs[k].set_ylabel('amplitude \n [a.u.]')
        axs[k].set_xlabel('time [sec]')
        axs[k].set_title(f'{split}')
        put_panel_label(axs[k], k)
        
        c += 1        
        k += 1

    '''
    scatter latency versus max amplitude for significant regions
    '''   

    if amp_number: 
        fig2 = plt.figure()
        axss = []
        if curve == 'euc':
            fig2.suptitle(f'distance metric: {curve}')
            
    c = 0          
    for split in align:
        
        if c == 0:
            axs.append(fig.add_subplot(gs[8:,c]))
        else:
            axs.append(fig.add_subplot(gs[8:,c],sharey=axs[len(align)*2+2]))
        
        d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'         
                    f'curves_{split}_{mapping}.npy',
                    allow_pickle=True).flat[0]

        acronyms = [tops[split][0][j] for j in range(len(tops[split][0]))
                if tops[split][1][j] < 0.05]
                

        if curve == 'euc':
            maxes = np.array([d[x][f'amp_euc'] for x in acronyms])
            lats = np.array([d[x]['lat_euc'] for x in acronyms])        
        else:
            maxes = np.array([d[x][f'max-min/max+min'] for x in acronyms])
            lats = np.array([d[x]['lat'] for x in acronyms])

        cols = [palette[reg] for reg in acronyms]
        stdes = np.array([d[x]['stde'] for x in acronyms])
        

        if amp_number:
            axss.append(fig2.add_subplot(int(f'1{len(align)}{c+1}')))
            nums = [1/d[reg]['nclus'] for reg in acronyms]

            l = list(zip(nums, maxes, cols))
            df = pd.DataFrame(l, columns=['1/nclus', 'maxes', 'cols'])
                  
            sns.regplot(ax=axss[c], x="1/nclus", y="maxes", data=df)
            axss[c].set_title(split)
        
        if curve == 'd_var_m':
            axs[k].errorbar(lats, maxes, yerr=stdes, fmt='None', 
                            ecolor=cols, ls = 'None', elinewidth = 0.5)
                        
        axs[k].scatter(lats, maxes, color=cols, marker='o',s=1)
        
        # put extra marker for highlighted regions
        exs_i = [i for i in range(len(acronyms)) 
                 if acronyms[i] in exs[split]]
                 
        axs[k].scatter(np.array(lats)[exs_i], np.array(maxes)[exs_i], 
                       color=np.array(cols)[exs_i], marker='x',s=25)                 
               
        axs[k].axhline(y=lower[split],linestyle='--',color='r')
        
        for i in range(len(acronyms)):
            reg = acronyms[i]
            if d[acronyms[i]][amp_curve] > lower[split]:
                axs[k].annotate('  ' + reg, # f"{reg} {f[reg]['nclus']}" ,
                                (lats[i], maxes[i]),
                    fontsize=6,color=palette[acronyms[i]])            

        axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')
        
        if split == 'block':
            ha = 'right'
        else:
            ha = 'left'    
        
        axs[k].text(0, 0.01, align[split],
                      transform=axs[k].get_xaxis_transform(),
                      horizontalalignment = ha)
                                
        axs[k].spines['top'].set_visible(False)
        axs[k].spines['right'].set_visible(False)        
        if c == 0:   
            axs[k].set_ylabel('max amplitude [a.u.]')
        axs[k].set_xlabel('latency [sec]')
        axs[k].set_title(f"{split}, {tops[split+'_s']} sig")
        put_panel_label(axs[k], k)
        c += 1     
        k += 1

    fig = plt.gcf()
    fig.tight_layout()
    
    fig.subplots_adjust(top=0.98,
                        bottom=0.051,
                        left=0.1,
                        right=0.971,
                        hspace=1.3,
                        wspace=0.52)

    fig.tight_layout()

    fig.savefig('/home/mic/paper-brain-wide-map/'
           f'overleaf_figs/manifold/'
           f'manifold.pdf', dpi=400)              

    print(f'distance metric: {curve}') 
 


def plot_swanson_supp(curve = 'd_var_m', mapping = 'Beryl'):
    
    figs = plt.figure(figsize=(10, 9), constrained_layout=True)
    nrows = 3
    ncols = len(align) 
    gs = GridSpec(nrows, ncols, figure=figs)


    amp_curve = ('max-min/max+min' if curve == 'd_var_m' 
                         else 'amp_euc')
                         
    p_type = 'p' if curve == 'd_var_m' else 'p_euc' 


    
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
        d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'         
                    f'curves_{split}_{mapping}.npy',
                    allow_pickle=True).flat[0]

        # get significant regions only
        acronyms = [reg for reg in d
                if d[reg][p_type] < 0.05]

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
        d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'         
                    f'curves_{split}_{mapping}.npy',
                    allow_pickle=True).flat[0]
                    
        # get significant regions only
        acronyms = [reg for reg in d
                if d[reg][p_type] < 0.05]

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


    figs.subplots_adjust(top=0.89,
bottom=0.018,
left=0.058,
right=0.985,
hspace=0.3,
wspace=0.214)
                       

    font = {'size'   : 10}
    mpl.rc('font', **font)    


def plot_cosmos_lines(curve = 'd_var_m'):

    amp_curve = ('max-min/max+min' if curve == 'd_var_m' 
                         else 'amp_euc')
                         
    p_type = 'p' if curve == 'd_var_m' else 'p_euc' 

    mapping = 'Beryl'
    df, palette = get_allen_info()
    fig = plt.figure(figsize=(20, 15), constrained_layout=True)
     
    gs = GridSpec(len(align), 7, figure=fig)

    sc = 0
    for split in list(align):
    
        d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'
                    f'curves_{split}_{mapping}.npy',
                    allow_pickle=True).flat[0]
                    
        # get significant regions only
        regsa = [reg for reg in d
                if d[reg][p_type] < 0.05]
                                
        # get cosmos parent regions for Swanson acronyms 
        cosregs = [df[df['id'] == int(df[df['acronym']==reg]['structure_id_path']
                   .values[0].split('/')[4])]['acronym']
                   .values[0] for reg in regsa]
                   
        cosregsC = list(Counter(cosregs))
        
        cosregsC = sorted(cosregsC)         
        
        k = 0
        axs = []                 
        for cos in cosregsC:
            
            axs.append(fig.add_subplot(gs[sc,k]))
            regs = np.array(regsa)[np.array(cosregs)==cos]
 
            print(split, cos, regs)

            texts = []
            for reg in regs:
                if any(np.isinf(d[reg][curve])):
                    print(f'inf in d_var_m of {reg}')
                    continue

                xx = np.linspace(-pre_post[split][0], 
                                  pre_post[split][1], len(d[reg][curve]))        

                axs[k].plot(xx,d[reg][curve], linewidth = 2,
                              color=palette[reg], 
                              label=f"{reg}")# {d[reg]['nclus']}
                              
                y = np.max(d[reg][curve])
                x = xx[np.argmax(d[reg][curve])]
                ss = f"{reg}"#  {d[reg]['nclus']}"
                texts.append(axs[k].text(x, y, ss, 
                                         color = palette[reg],
                                         fontsize=9))                             

            #adjust_text(texts)
            
    
            axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')
            axs[k].text(0, 0.01, align[split],
                          transform=axs[k].get_xaxis_transform(),
                          horizontalalignment = 'right' if split == 'block'
                          else 'left')           
            axs[k].spines['top'].set_visible(False)
            axs[k].spines['right'].set_visible(False)
            axs[k].set_ylabel('distance')
            axs[k].set_xlabel('time [sec]')
            axs[k].set_title(f'{split}, {cos}')
            put_panel_label(axs[k], k)
                    
            k +=1
        sc +=1
    plt.tight_layout()


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
        nobs = nobs // ntravis


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
           
    print(split)
    R = np.load('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/manifold/'
                f'single_d_vars_{split}_Swanson.npy', 
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
    
    fig, axs = plt.subplots(ncols =2, nrows=1)    
                         
    for i in range(2): 
        axs[i].imshow(dat[i], cmap='Greys',aspect="auto",
            interpolation='none')
        
        for s in Counter(sess0):
            axs[i].axhline(y = np.where(sess0==s)[0][0], c = 'r', 
                       linestyle = '--') 
    
        
        axs[i].set_ylabel('cells')
        axs[i].set_xlabel('time [sec]')
        axs[i].set_title(f'PETH type {i}')
        xs = np.linspace(0,pre_post[split][0] + pre_post[split][1],5)    
        axs[i].set_xticks(np.linspace(0,dat[i].shape[-1],5))
        axs[i].set_xticklabels(["{0:,.3f}".format(x) for x in xs])            
        
    fig.suptitle(f'{reg}, {split}')        
    fig.tight_layout()   
        
        
def save_df_for_table(): 
    '''
    reformat results for table
    '''
    
    mapping = 'Beryl'
    
    columns = ['acronym', 'name', 'nclus'] 
    
    l = ['stde', 'max-min/max+min', 'p', 'lat', 'amp_euc', 'lat_euc']
    
    
    r = []
    
    regs = []   
    for split in align: 
    
        for x in l:
            columns.append('_'.join([x, split]))

        d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'         
                    f'curves_{split}_{mapping}.npy',
                    allow_pickle=True).flat[0] 
        regs.append(list(d.keys()))
        
    regs = np.unique(np.concatenate(regs))
    
    for reg in regs:
        rr = []
        
        for split in align:
            d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'         
                        f'curves_{split}_{mapping}.npy',
                        allow_pickle=True).flat[0]        
            if reg in d:
            
                nclus = d[reg]['nclus']
                break
        
        
        rr.append([reg, get_name(reg), nclus])
        
        
        for split in align:
            d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'         
                        f'curves_{split}_{mapping}.npy',
                        allow_pickle=True).flat[0]

            if reg in d:    
                rr.append([d[reg][x] for x in l])
            else:
                rr.append([np.nan for x in l])        
                
        r.append(np.concatenate(rr))    
            
      
                          
    df  = pd.DataFrame(data=r,columns=columns)        
    df.to_pickle('/home/mic/paper-brain-wide-map/'
                 'manifold_analysis/manifold_results.pkl')
    df.to_excel('/home/mic/paper-brain-wide-map/'
                 'manifold_analysis/manifold_results.xlsx')   


def load_brandon():

    # identify sessions/regions with highest block decoding

    df = pd.read_csv('/home/mic/paper-brain-wide-map/manifold_analysis'
                     '/10-09-2022_BWMmetaanalysis_decodeblock.csv')

