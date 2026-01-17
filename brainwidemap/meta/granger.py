#import os
#os.environ["SPECTRAL_CONNECTIVITY_ENABLE_GPU"] = "true"

import math
import numpy as np
import random
from collections import Counter
from pathlib import Path
from spectral_connectivity import Multitaper, Connectivity
import time
import timeit
from itertools import combinations
from scipy.stats import norm
import pandas as pd
import umap, os
from copy import copy
#from sklearn.manifold import MDS
from scipy.stats import pearsonr, spearmanr, norm, chi2, zscore
from statsmodels.stats.multitest import multipletests
import networkx as nx
from scipy.sparse.csgraph import shortest_path, csgraph_from_dense
from scipy.sparse import csr_matrix 
 
from brainwidemap import (load_good_units, bwm_query, 
    download_aggregate_tables, load_trials_and_mask)
from iblutil.numerical import bincount2D
from one.api import ONE
from iblatlas.regions import BrainRegions
from iblatlas.atlas import AllenAtlas
import iblatlas
import sys
sys.path.append('Dropbox/scripts/IBL/')

import logging
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gc as garbage
import matplotlib
#matplotlib.use('QtAgg')
#matplotlib.use('tkagg')
sns.reset_defaults()
plt.ion()

one = ONE()
#base_url='https://openalyx.internationalbrainlab.org',
#          password='international', silent=True

bad_eids = ['4e560423-5caf-4cda-8511-d1ab4cd2bf7d',
            '3a3ea015-b5f4-4e8b-b189-9364d1fc7435',
            'd85c454e-8737-4cba-b6ad-b2339429d99b',
            'de905562-31c6-4c31-9ece-3ee87b97eab4',
            '2d9bfc10-59fb-424a-b699-7c42f86c7871',
            '7cc74598-9c1b-436b-84fa-0bf89f31adf6',
            '642c97ea-fe89-4ec9-8629-5e492ea4019d',
            'a2ec6341-c55f-48a0-a23b-0ef2f5b1d71e', # clear saturation
            '195443eb-08e9-4a18-a7e1-d105b2ce1429',
            '549caacc-3bd7-40f1-913d-e94141816547',
            '90c61c38-b9fd-4cc3-9795-29160d2f8e55',
            'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d',
            'a9138924-4395-4981-83d1-530f6ff7c8fc',
            '8c025071-c4f3-426c-9aed-f149e8f75b7b',
            '29a6def1-fc5c-4eea-ac48-47e9b053dcb5',
            '0cc486c3-8c7b-494d-aa04-b70e2690bcba']

         
# window names: [alignment times, segment length, gap, side]           
wins = {'whole_session': ['no_events', 10, 0, 'plus'],
        'feedback_plus1': ['feedback_times',1, 0, 'plus'],
        'stim_plus01': ['stimOn_times', 0.1, 0, 'plus'],
        'stim_minus06_minus02': ['stimOn_times', 0.4, 0.2, 'minus'],
        'move_minus01': ['firstMovement_times', 0.1, 0, 'minus']}

          
ba = AllenAtlas()          
br = BrainRegions()

T_BIN = 0.0125  # 0.005
sigl=0.05  # alpha throughout

#df = bwm_query(one)

# save results here
pth_res = Path(one.cache_dir, 'granger') #, 'res_feedback_times')
pth_res.mkdir(parents=True, exist_ok=True)


def get_allen_info():
    r = np.load(Path(one.cache_dir, 'dmn', 'alleninfo.npy'),
                allow_pickle=True).flat[0]
    return r['dfa'], r['palette']


def trans_(d):
    '''
    turn adjacency matrix into A --> B format dictionary
    '''
    d0 = {}
    res = d['res']
    regs = d['regs']
    for i in range(len(regs)):
        for j in range(len(regs)):
            if i == j:
                continue
            d0[f'{regs[i]} --> {regs[j]}'] = res[i,j]
                
    return d0

def p_fisher(p_values):
    # combine p-values via Fisher's method
    if len(p_values) == 0:
        raise ValueError("Input list of p-values is empty.")
    if len(p_values) == 1:
        return p_values[0]
    
    z_scores = norm.ppf(1 - np.array(p_values) / 2)
    X_squared = np.sum(z_scores**2)
    p_combined = chi2.sf(X_squared, 2 * len(p_values))
    return p_combined


def get_nregs(win='whole_session'):

    '''
    get a dict with regs per eid
    '''

    pthh = pth_res / win
    p = pthh.glob('**/*')
    files = [x for x in p if x.is_file()]
    
    d = {}

    for sess in files:    
        D = np.load(sess, allow_pickle=True).flat[0]
        eid = str(sess).split('/')[-1].split('.')[0]
        d[eid] = D['regsd']

    return d


def get_structural(rerun=False, shortestp=False, fign=4):

    '''
    load structural connectivity matrix
    https://www.nature.com/articles/nature13186
    fig3
    https://static-content.springer.com
    /esm/art%3A10.1038%2Fnature13186/MediaObjects/
    41586_2014_BFnature13186_MOESM70_ESM.xlsx
    
    fig4 '/home/mic/41586_2014_BFnature13186_MOESM72_ESM.xlsx'
    '''
    
    pth_ = Path(one.cache_dir, 'granger', f'structural{fign}.npy')
    if (not pth_.is_file() or rerun):
    
        if fign == 3:
            s=pd.read_excel('/home/mic'
                '/41586_2014_BFnature13186_MOESM70_ESM.xlsx')
            cols = list(s.keys())[1:297]
            rows = s['Unnamed: 0'].array
            
            M = np.zeros((len(cols), len(rows)))
            for i in range(len(cols)):
                M[i] = s[cols[i]].array
                
            M = M.T

            cols1 = np.array([reg.strip().replace(",", "") for reg in cols])
            rows1 = np.array([reg.strip().replace(",", "") for reg in rows])
            
            # average across injections
            regsr = list(Counter(rows1))
            M2 = []
            for reg in regsr:
                M2.append(np.mean(M[rows1 == reg], axis=0))       

#            # thresholding as in the paper
#            M[M > 10**(-0.5)] = 1
#            M[M < 10**(-3.5)] = 0

            M2 = np.array(M2)
            regs_source = regsr
            regs_target = cols1

            # turn into dict
            d = {}
            for i in range(len(regs_source)):
                for j in range(len(regs_target)):
                    if M2[i,j] < 0:
                        continue
                    d[' --> '.join([regs_source[i], 
                                     regs_target[j]])] = M2[i,j]

            np.save(pth_, d,
                    allow_pickle=True)
                    
        elif fign == 4:
        
            s=pd.read_excel(
                '/home/mic/41586_2014_BFnature13186_MOESM71_ESM.xlsx',
                sheet_name='W_ipsi')
            cols = list(s.keys())[1:]
            rows = s['Unnamed: 0'].array
            
            M = np.zeros((len(cols), len(rows)))
            for i in range(len(cols)):
                M[i] = s[cols[i]].array
            M = M.T    
                
            # load p-values
            s=pd.read_excel(
                '/home/mic/41586_2014_BFnature13186_MOESM71_ESM.xlsx',
                sheet_name='PValue_ipsi')
            colsp = list(s.keys())[1:]
            rowsp = s['Unnamed: 0'].array                
            P = np.zeros((len(colsp), len(rowsp)))
            for i in range(len(colsp)):
                P[i] = s[colsp[i]].array                
            P = P.T         
                     
            
            # turn into dict
            d = {}
            for i in range(len(rows)):
                for j in range(len(cols)): 
                    if np.isnan(P[i,j]):
                        continue
                    if P[i,j] > 0.05:
                        continue
#                        d[' --> '.join([rows[i], 
#                                         cols[j]])] = 0                
                    else:      
                        d[' --> '.join([rows[i], 
                                         cols[j]])] = M[i,j]

            np.save(pth_, d,
                    allow_pickle=True)        
        
        
                
    else:
        d = np.load(pth_, allow_pickle=True).flat[0]        
        
        
    if shortestp:
    
        # create adjacency matrix, setting absent connections to zero
        # Unnamed: 1 is injection volume; discard
        regs = Counter(np.array([x.split(' --> ') 
                    for x in list(d.keys()) 
                    if 'Unnamed: 1' not in x]).flatten())    

        regs = list(regs)
    
        A = np.zeros((len(regs), len(regs)))
        for i in range(len(regs)):
            for j in range(len(regs)):
                if f'{regs[i]} --> {regs[j]}' in d:
                    A[i,j] = d[f'{regs[i]} --> {regs[j]}']
                                              
        adjacency_matrix = A
        # Convert the adjacency matrix to a sparse matrix (CSR format)
        sparse_matrix = csr_matrix(adjacency_matrix)

        # Find the shortest path lengths using the Floyd-Warshall algorithm
        distances, predecessors = shortest_path(
            csgraph=sparse_matrix, directed = False,
            method='FW', return_predecessors=True,unweighted=False)
            
        d0 = {}
        res = distances
        for i in range(len(regs)):
            for j in range(len(regs)):
                if i == j:
                    continue
                d0[f'{regs[i]} --> {regs[j]}'] = res[i,j]
                
        return d0
               
    return d                                



def get_centroids(rerun=False, dist_=False):

    '''
    Beryl region centroids xyz
    '''
    
    pth_ = Path(one.cache_dir, 'granger', 'beryl_centroid.npy')
    if (not pth_.is_file() or rerun):
        beryl_vol = ba.regions.mappings['Beryl-lr'][ba.label]
        beryl_idx = np.unique(ba.regions.mappings['Beryl-lr'])

        d = {}
        k = 0
        for ridx in beryl_idx:
            idx = np.where(beryl_vol == ridx)
            ixiyiz = np.c_[idx[1], idx[0], idx[2]]
            xyz = ba.bc.i2xyz(ixiyiz)   
            d[br.index2acronym(ridx,
                mapping='Beryl')] = np.mean(xyz, axis=0)
            print(br.index2acronym(ridx,mapping='Beryl'),k, 'of',
                  len(beryl_idx))  
            k+=1

        np.save(pth_, d,
                allow_pickle=True)
                
    else:
        d = np.load(pth_, allow_pickle=True).flat[0]        

    if dist_:
        print('note this is the inverse of the centroid euc dist')
        regs = list(d.keys())
        res = np.zeros((len(regs), len(regs)))
        for i in range(len(regs)):
            for j in range(len(regs)):
                res[i,j] = np.sum((d[regs[i]] - d[regs[j]])**2)**0.5
        
        # invert and normalize, so that 1 is maximally similar
        # and 0 is maximally distant 
        max_val = np.max(res)
        min_val = np.min(res)
            
        # Perform linear transformation
        res = 1 - ((res - min_val) / (max_val - min_val))           
                    
        D = {'res': res, 'regs': regs}
        return D
        
    return d
            

def get_volume(rerun=False):

    '''
    Beryl region volumina in mm^3
    '''

    pth_ = Path(one.cache_dir, 'granger', 'beryl_volumina.npy')
    if (not pth_.is_file() or rerun):
        ba.compute_regions_volume()  
        acs = np.unique(br.id2acronym(
                ba.regions.id, mapping='Beryl'))  
         
        d2 = {}  
        for ac in acs:
            d2[ac] = ba.regions.volume[
                        ba.regions.acronym2index(
                            ac, mapping='Beryl')[1]].sum()
                            
        np.save(pth_, d2,
                allow_pickle=True)                        
                          
    else:
        d2 = np.load(pth_, allow_pickle=True).flat[0]        
        
    return d2



def make_data(T=300000, vers='oscil', peak_freq_factor0=0.55,
              peak_freq_factor1=0.2, phase_lag_factor=0.2):
    
    '''
    auto-regressive data creation
    x2 dependend on x1, not vice versa
    '''
    
    x1 = np.random.normal(0, 1,T+3)
    x2 = np.random.normal(0, 1,T+3)

    if vers == 'dc': 
        x1 = x2[30:]
        x2 = x2[0:-30]
        regsd = {'x2[30:]':1, 'x2[0:-30]':1} 
        return np.array([x1,x2]), regsd  
    
    elif vers == 'oscil':
        for t in range(2,T+2):
            x2[t] = (peak_freq_factor1*x2[t-1] - 0.8*x2[t-2] + 
                     x2[t+1])
            x1[t] = (peak_freq_factor0*x1[t-1] - 0.8*x1[t-2] + 
                     phase_lag_factor * x2[t-1] + x1[t+1])
            
    elif vers == 'loopy':
        for t in range(2,T+2):
            x2[t] = (peak_freq_factor1 * x2[t - 1] - 0.8 * x2[t - 2] 
                     + phase_lag_factor * x1[t - 1] + x2[t + 1])
            x1[t] = (peak_freq_factor0 * x1[t - 1] - 0.8 * x1[t - 2] 
                     + phase_lag_factor * x2[t - 1] + x1[t + 1])        
    
    regsd = {'dep':1, 'indep':1}
    return np.array([x1[2:-1],x2[2:-1]]), regsd
    

    
def bin_average_neural(eid, mapping='Beryl', nmin=1):
    '''
    bin neural activity; bin, then average firing rates per region
    from both probes if available
    
    used to get session-wide time series, not cut into trials
    
    nmin: 
        minumum number of neurons per brain region to consider it
    returns: 
        R2: binned firing rate per region per time bin
        times: time stamps for all time bins
        redg: dict of neurons per region acronym
    '''
    
    pids0, probes = one.eid2pid(eid)
    df = bwm_query(one)
    pids = []
    
    for pid in pids0:
        if str(pid) in df['pid'].values:
            pids.append(pid)

    if len(pids) == 1:
        spikes, clusters = load_good_units(one, pids[0])
        R, times, _ = bincount2D(spikes['times'], 
                        spikes['clusters'], T_BIN)
        acs = br.id2acronym(clusters['atlas_id'], mapping=mapping)
        regs = Counter(acs)
        regs2 = {x: regs[x] for x in regs if 
                 ((regs[x] >= nmin) and (x not in ['root','void']))}
                 
        R2 = np.array([np.mean(R[acs == reg],axis=0) for reg in regs2])
          
        return R2, times, regs2  
    
    else:
        sks = []
        clus = []
        for pid in pids:
            spikes, clusters = load_good_units(one, pid)    
            sks.append(spikes)
            clus.append(clusters)
    
        # add max cluster of p0 to p1, then concat, sort 
        max_cl0 = max(sks[0]['clusters'])
        sks[1]['clusters'] = sks[1]['clusters'] + max_cl0 + 1
         
        times_both = np.concatenate([sks[0]['times'],
                                     sks[1]['times']])
        clusters_both = np.concatenate([sks[0]['clusters'],
                                        sks[1]['clusters']])
                                        
        acs_both = np.concatenate([
                       br.id2acronym(clus[0]['atlas_id'],
                       mapping=mapping), 
                       br.id2acronym(clus[1]['atlas_id'],
                       mapping=mapping)])
                                                        
        t_sorted = np.sort(times_both)
        c_ordered = clusters_both[np.argsort(times_both)] 
        
        R, times, clus = bincount2D(t_sorted, c_ordered, T_BIN)  

        regs = Counter(acs_both)
        regs2 = {x: regs[x] for x in regs if 
                 ((regs[x] >= nmin) and (x not in ['root','void']))}
                 
        R2 = np.array([np.mean(R[acs_both == reg],axis=0) 
                       for reg in regs2])
          
        return R2, times, regs2    


def gc(r, segl=10, shuf=False, shuf_type = 'reg_shuffle'):    

    '''
    chop up times series into segments of length segl [sec]
    Independent of trial-structure, then compute metrics
    '''
    
    nchans, nobs = r.shape
    segment_length = int(segl / T_BIN)
    num_segments = nobs // segment_length
    
    # reshape into: n_signals x n_segments x n_time_samples
    r_segments = r[:, :num_segments * segment_length
                   ].reshape((nchans, num_segments, 
                   segment_length))

    if shuf:
        

        if shuf_type == 'reg_shuffle':
            # shuffle region order per trial        
            indices = np.arange(r_segments.shape[0])
            
            rs = np.zeros(r_segments.shape)
            for trial in range(r_segments.shape[1]):
                np.random.shuffle(indices)    
                rs[:,trial,:] = r_segments[indices, trial, :]
                
            r_segments = np.array(rs)    

        else:
            # shuffle segment order
            indices = np.arange(r_segments.shape[1])
            
            rs = np.zeros(r_segments.shape)
            for chan in range(r_segments.shape[0]):
                np.random.shuffle(indices)    
                rs[chan] = r_segments[chan, indices]
                
            r_segments = np.array(rs)    
            #print('segments channel-independently shuffled')
            
                   
    # reshape into:  n_time_samples x n_segments x n_signals               
    r_segments_reshaped = r_segments.transpose((2, 1, 0))

    m = Multitaper(
        r_segments_reshaped,
        sampling_frequency=1/T_BIN,
        time_halfbandwidth_product=2,
        start_time=0)
    
    c = Connectivity(
        fourier_coefficients=m.fft(), 
        frequencies=m.frequencies, 
        time=m.time)
 
    return c    

  
def fr_performance(eid, nmin=1, nts = 20):

    '''
    for a given session get fr per region in inter trial interval
    (-0.4 t0 -0.1 relative to stim onset)
    and return with performance average
    
    return:
        ftp: performance per trial (0 incorrect, 1 correct)
        frs: firing rate per inter trial interval per region
    '''
    
    # combine probes, bin fr per region     
    r, ts, regd = bin_average_neural(eid, nmin=nmin)
    
    # cut out inter-trial fr
    trials, mask = load_trials_and_mask(one, eid)

    iti = np.array([trials['stimOn_times'][mask] - 0.4, 
                    trials['stimOn_times'][mask] - 0.1])
    ftp = trials['feedbackType'][mask]
    ftp.replace(-1, 0, inplace=True)
    fp = np.array(ftp)
    
    cis = []
    for i in range(2):
        indices = np.searchsorted(ts, iti[i])
        adjusted_indices = np.clip(indices - 1, 0, len(ts) - 1)
        closest_indices = np.where(np.abs(iti[i] - ts[adjusted_indices]) < 
                                   np.abs(iti[i] - ts[adjusted_indices + 1]), 
                                   adjusted_indices, adjusted_indices + 1)
        cis.append(closest_indices) 
 
    cis = np.array(cis).T 

    frs = []
    for tr in cis:
        frs.append(r[:,tr[0] : tr[1] +1])
    
    fr = np.mean(np.array(frs),axis=-1).T      

    fp_m = []
    fr_m = []
    
    for chunk in range(len(fp)//nts):
        fp_m.append(np.mean(fp[chunk * nts: (chunk+1) * nts]))
        fr_m.append(np.mean(fr[:, chunk * nts: (chunk+1) * nts], axis=-1))   
    
    fp_m = np.array(fp_m)
    fr_m = np.array(fr_m).T
        
    corrd = {}
    for i in range(len(fr_m)):
        reg = list(regd)[i] 
        corrd[reg] = [list(pearsonr(fr_m[i], fp_m)),
                      list(spearmanr(fr_m[i], fp_m))]         
    
    D = {}    
    D['fp_m'] = fp_m        
    D['fr_m'] = fr_m         
    D['regd'] = regd
    D['corrd'] = corrd

    return D
    
    
def cut_segments(r, ts, te, segment_length=100, side='plus', gap_length=0):

    '''
    r:
        binned activity time series
    ts:
        time stamps per bin
    te:
        event times where segments start
    segment_length:
        seg length in bins
    side: ['plus', 'minus']
        if segments start or end at alignement time
    gap_length:
        gap between segment and alignement event in bins
        
    Returns:
        A 3D array of segments with shape (n_regions, n_events, segment_length)

    ''' 

    r = np.array(r)
    ts = np.array(ts)
    te = np.array(te)
    
    # Ensure r is 2D, even if it's a single region
    if r.ndim == 1:
        r = r[np.newaxis, :]
        
    # Find indices of the nearest time stamps to event times
    event_indices = np.searchsorted(ts, te)  
      
    # Adjust start indices based on 'side' and gap_length
    if side == 'plus':
        # Start segment after the event time plus the gap
        start_indices = event_indices + gap_length
    elif side == 'minus':
        # End segment at event time minus the gap, so start earlier
        start_indices = event_indices - segment_length - gap_length
    else:
        raise ValueError("Invalid value for 'side'. Choose 'plus' or 'minus'.")
    
    # Create an array of indices for each segment
    indices = start_indices[:, np.newaxis] + np.arange(segment_length)
    
    # Clip indices to ensure they're within bounds
    indices = np.clip(indices, 0, r.shape[1] - 1)
    
    # Extract segments
    segments = r[:, indices]

    # Rearrange dimensions to (n_regions, n_events, segment_length)
    segments = np.transpose(segments, (0, 1, 2))
    
    # If original input was 1D, remove the singleton dimension
    if r.shape[0] == 1:
        segments = segments.squeeze(axis=1)
    
    return segments
    
       
'''  
####################
bulk processing
####################
'''


def get_all_granger(eids='all', nshufs = 100, nmin=10, wins=wins):

    '''
    get spectral directed granger for all bwm sessions
    segl: 
        segment length in seconds (unless wins are given)
    wins: 
        Window of interest and seg length, if None, the whole session 
        is binned and cut into segments; else segments
        of length segl are cut after win times                
        eid, probe = one.pid2eid(pid)
   
    '''

    if isinstance(eids, str):
        df = bwm_query(one)
        eids = np.unique(df[['eid']].values)
                
    Fs = []
    k = 0
    print(f'Processing {len(eids)} sessions')

    time0 = time.perf_counter()
    for eid in eids:
        print('eid:', eid)
        
        # remove lick artefact eid and late fire only
        if eid in bad_eids:
            print('exclude', eid)
            continue
     
        try:
            time00 = time.perf_counter()
            
            r, ts, regsd = bin_average_neural(eid, nmin=nmin)
            if not bool(regsd):
                print(f'no data for {eid}') 
                continue
                
            nchans, nobs = r.shape

            for win in wins:
            
                if win == 'whole_session':
                    print('chop up whole session into segments')
                    print(win, 'align|segl|gap|side', wins[win])
                    segl = wins[win][1]  # in sec
                    segment_length = int(segl / T_BIN)  # in bins
                    
                    # chop up whole session into segments
                    num_segments = nobs // segment_length
                    
                    # reshape into: n_signals x n_segments x n_time_samples
                    r_segments = r[:, :num_segments * segment_length
                                   ].reshape((nchans, num_segments, 
                                   segment_length))
                                               
                    # reshape to n_time_samples x n_segments x n_signals
                    r_segments_reshaped = r_segments.transpose((2, 1, 0))
                    
                else:
                    print(win, 'align|segl|gap|side', wins[win])
                    
                    segl = wins[win][1]  # in sec
                    segment_length = int(segl / T_BIN)  # in bins
                    gap = wins[win][2]  # in sec
                    gap_length = int(gap / T_BIN)  # in bins
                    side = wins[win][3]
                    
                    # only pick segments starting at "win" times
                    # Load in trials data and mask bad trials (False if bad)
                    trials, mask = load_trials_and_mask(one, eid,
                        saturation_intervals=['saturation_stim_plus04',
                                              'saturation_feedback_plus04',
                                              'saturation_move_minus02',
                                              'saturation_stim_minus04_minus01',
                                              'saturation_stim_plus06',
                                              'saturation_stim_minus06_plus06'])
                                              
                    te = trials[mask][wins[win][0]].values
                     
                    # n_regions x n_segments x n_time_samples
                    r_segments = cut_segments(r, ts, te, gap_length=gap_length,
                                    side=side, segment_length=segment_length)
                                              
                    # reshape to n_time_samples x n_segments x n_signals
                    r_segments_reshaped = r_segments.transpose((2, 1, 0))
                    
                    
                m = Multitaper(
                    r_segments_reshaped,
                    sampling_frequency=1/T_BIN,
                    time_halfbandwidth_product=2,
                    start_time=0)

                c = Connectivity(
                    fourier_coefficients=m.fft(), 
                    frequencies=m.frequencies, 
                    time=m.time)
                

                psg = c.pairwise_spectral_granger_prediction()[0]
                coh = c.coherence_magnitude()[0]
                score_g = np.mean(psg,axis=0)
                score_c = np.mean(coh,axis=0)
                
                # get scores after shuffling segments
                shuf_g = []
                shuf_c = []
                
                # shuffle pairs of regions separately
                pairs = np.array(list(combinations(range(nchans),2)))
                
                print('data binned', regsd, f'{len(pairs)} pairs')
                            
                for i in range(nshufs):
                    if i%10 == 0:
                        print('shuf', i, f'({nshufs})')
                    
                    mg = np.zeros([nchans,nchans])
                    mc = np.zeros([nchans,nchans])
                    
                    for pair in pairs:                                      
                        rs = np.zeros([2, r_segments.shape[1],
                                          r_segments.shape[2]])
                                          
                        for trial in range(r_segments.shape[1]):
                            np.random.shuffle(pair)    
                            rs[:,trial,:] = r_segments[pair, trial, :]
                            
                        r_segments0 = np.array(rs)
                        
                        #into n_time_samples x n_segments x n_signals               
                        r_segments_reshaped0 = r_segments0.transpose((2, 1, 0))

                        m = Multitaper(
                            r_segments_reshaped0,
                            sampling_frequency=1/T_BIN,
                            time_halfbandwidth_product=2,
                            start_time=0)
                        
                        c0 = Connectivity(
                            fourier_coefficients=m.fft(), 
                            frequencies=m.frequencies, 
                            time=m.time)
                    
                        mmg = np.mean(
                            c0.pairwise_spectral_granger_prediction()[0],
                                axis=0)
                        mmc = np.mean(
                            c0.coherence_magnitude()[0],
                                axis=0)        
                                
                        mg[pair[0], pair[1]] = mmg[0,1]
                        mg[pair[1], pair[0]] = mmg[1,0]
                        mc[pair[0], pair[1]] = mmc[0,1]
                        mc[pair[1], pair[0]] = mmc[1,0]
                 
                    shuf_g.append(mg)
                    shuf_c.append(mc)
                    
                shuf_g.append(score_g)
                shuf_c.append(score_c)          
                    
                shuf_g = np.array(shuf_g)
                shuf_c = np.array(shuf_c)

                p_g = np.mean(shuf_g >= score_g, axis=0)
                p_c = np.mean(shuf_c >= score_c, axis=0)    
                
                D = {'regsd': regsd,
                     'freqs': c.frequencies,
                     'p_granger': p_g,
                     'p_coherence': p_c,
                     'coherence': score_c,
                     'granger': score_g,
                     'coherence_pks': c.frequencies[np.argmax(coh,axis=0)],  
                     'granger_pks': c.frequencies[np.argmax(psg,axis=0)]}

                pthh = Path(pth_res, win)
                pthh.mkdir(parents=True, exist_ok=True)     
                np.save(pthh / f'{eid}.npy', D, allow_pickle=True)

                garbage.collect()
                print(k + 1, 'of', len(eids), 'ok')
                time11 = time.perf_counter()
                print('runtime [sec]: ', time11 - time00)
            
        except BaseException:
            Fs.append(eid)
            garbage.collect()
            print(k + 1, 'of', len(eids), 'fail', eid)

        k += 1    

    time1 = time.perf_counter()
    print(time1 - time0, f'sec for {len(eids)} sessions')
    print(len(Fs), 'failures')
    return Fs



def get_res(nmin=10, metric='granger', combine_=False, c_mc =False,
            rerun=False, sig_only=False, sessmin=2, win='whole_session'):

    '''
    Group results
    
    nmin: minimum number of neurons per region to be included
    sessmin: min number of sessions with region combi
    
    metric in ['coherence', 'granger']  
    
    c_ms: correction for multiple comparisons (fdr_bh)  
    '''
       
    pth_ = Path(pth_res, f'{metric}_{win}.npy')
    if (not pth_.is_file() or rerun):

        pthh = pth_res / win
        p = pthh.glob('**/*')
        files = [x for x in p if x.is_file()]
        
        d = {}
        ps = []

        
        for sess in files:
        
            # remove lick artefact eid and late fire only

                        
            eid = str(sess).split('/')[-1].split('.')[0]
            
            if eid in bad_eids:
                print('exclude', sess)
                continue
                    
            D = np.load(sess, allow_pickle=True).flat[0]
            m = D[metric]
            regs = list(D['regsd'])
            p_c = D[f'p_{metric}']
            if not isinstance(D['regsd'], dict):
                nd.append(sess)
                continue        
            
            for i in range(len(regs)):
                for j in range(len(regs)):
                    if i == j:
                        continue
                    
                    if ((D['regsd'][regs[i]] <= nmin) or
                        (D['regsd'][regs[j]] <= nmin)):
                        continue
                        
                    if f'{regs[i]} --> {regs[j]}' in d:
                        d[f'{regs[i]} --> {regs[j]}'].append(
                            [m[j, i], p_c[i,j], 
                            D['regsd'][regs[i]], D['regsd'][regs[j]], eid])

                    else:
                        d[f'{regs[i]} --> {regs[j]}'] = []
                        d[f'{regs[i]} --> {regs[j]}'].append(
                            [m[j, i], p_c[i,j],
                            D['regsd'][regs[i]], D['regsd'][regs[j]], eid])

                    
                    ps.append(p_c[i,j])
        
        if c_mc:                   
            _, corrected_ps, _, _ = multipletests(ps, sigl, 
                                               method='fdr_bh')
        else:
            corrected_ps = np.array(ps)                                       
                                           
        kp = 0                                           
        d2 = {}
        for pair in d:
            scores = [] 
            for score in d[pair]:
                scores.append([score[0],corrected_ps[kp], 
                               score[2], score[3], score[4]])
                kp+=1
            if scores == []:
                continue
            else:                                           
                d2[pair] = scores
                                
        print(f'{metric} measurements in total: {len(ps)} ')
        print(f'Uncorrected significant: {np.sum(np.array(ps)<sigl)}')
        print(f'Corrected significant: {np.sum(corrected_ps<sigl)}')

        np.save(Path(one.cache_dir, 'granger', f'{metric}_{win}.npy'), 
                d2, allow_pickle=True)
                
        d = d2        
                   
    else:
        d = np.load(pth_, allow_pickle=True).flat[0]         


    if combine_:
        # take mean score across measurements
        dd = {
            k: np.array([
                np.mean(np.array(d[k])[:, 0].astype(float)), 
                p_fisher(np.array(d[k])[:, 1].astype(float))
            ], dtype=float)  # Ensure dtype is float for the array
            for k in d if len(d[k]) >= sessmin
        }
        
        if sig_only:
            ddd = {}
            for pair in dd:
                if dd[pair][1] < sigl:
                    ddd[pair] = float(dd[pair][0])

            dd = ddd

    else:        
        if sig_only:
            dd = {}
            for pair in d:
                l = [float(x[0]) for x in d[pair] if x[1] < sigl]    
                if l == []:
                    continue
                else:
                    dd[pair] = l    
    
        else:
            dd = d
        
    return dd    
    
    
def get_meta_info(rerun=False, win='whole_session'):

    '''
    get neuron number and peak freq_s per region???
    '''
    

    pth_ = Path(pth_res, f'all_regs.npy')
    if (not pth_.is_file() or rerun):
    
        pthh = pth_res / win
        p = pthh.glob('**/*')
        files = [x for x in p if x.is_file()]
        
        d = {}
        for sess in files:
            
            D = np.load(sess, allow_pickle=True).flat[0]

            if not isinstance(D['regsd'], dict):
                continue

            dd = {key: D[key] for key in ['regsd', 'granger_pks',
                                          'coherence_pks', 
                                          'p_granger', 'p_coherence',
                                          'granger', 'coherence']}

            d[str(sess).split('/')[-1].split('.')[0]] = dd

        np.save(pth_, d, allow_pickle=True)
        
    else:
        d = np.load(pth_, allow_pickle=True).flat[0]        
        
    return d


def get_all_fr_performance(eids='all', nmin=10, nts=20):
    '''
    For each session, bin firing rate per region,
    compute also performance for chunks of consecutive trials.

    Parameters:
    - eids: Session IDs or 'all' to process all sessions
    - nmin: Minimum number of neurons to include a region
    - nts: Chunk length
    '''
    if isinstance(eids, str) and eids == 'all':
        df = bwm_query(one)
        eids = np.unique(df[['eid']].values)

    logging.info(f'Processing {len(eids)} sessions')
    for k, eid in enumerate(eids):
        logging.info(f'Processing eid: {eid} ({k + 1}/{len(eids)})')
        
        # Remove lick artifact eid and late fire only
        if eid in bad_eids:
            logging.info(f'Skipping {eid} due to known issues')
            continue
            
        try:    
            D = fr_performance(eid, nmin=nmin, nts=nts)
        except Exception as e:
            logging.error(f'Error processing {eid}: {e}')
            continue    
        
        logging.info(f'{eid} done')
        
        # Use timeit module for accurate timing
        np.save(Path(one.cache_dir, 'fr_performance', f'{eid}.npy'),
            D, allow_pickle=True)
            
        print(f'{eid}, {k} of {len(eids)} done')


'''
#####################
plotting
#####################    
'''

def plot_fr_performance(eid, nmin=10):

    '''
    scatter time chunks of size nts trials
    x axis is performance for chunk
    y axis is firing rate in inter trial interval for chunk
    '''
    
    pthr = Path(one.cache_dir, 'fr_performance')
    D = np.load(Path(pthr, f'{eid}.npy'), allow_pickle=True).flat[0]

    fig, axs = plt.subplots(nrows =1, ncols = len(D['regd']), 
                   figsize=(4,4))
                   
    if not isinstance(axs, np.ndarray):
        axs = [axs]               
                   
    _, pal = get_allen_info()
    
    for i in range(len(D['fr_m'])):
        reg = list(D['regd'])[i] 
        axs[i].scatter(D['fr_m'][i], D['fp_m'], 
                       color = pal[reg], label = reg)

                      
        #axs[i].legend()
        
        axs[i].set_title(f'{reg} '
                        f'\n Pear. r,p: {D["corrd"][reg][0][0]:.2f},'
                        f' {D["corrd"][reg][0][1]:.2f} \n'
                        f' Spear. r,p: {D["corrd"][reg][1][0]:.2f},'
                        f' {D["corrd"][reg][1][1]:.2f}')   
        axs[i].set_xlabel('firing rate [Hz]')  
        axs[i].set_ylabel('performance')

    fig.suptitle(f'{eid} \n trial chunk size = 20')
    fig.tight_layout()
        

def plot_all_fr_performance(per_reg=True, kind='scatter'):

    '''
    big scatter, one point per trial chunk, colored by region
    '''
    
    pthr = Path(one.cache_dir, 'fr_performance')
    eids = [x.split('.')[0] for x in os.listdir(pthr)]
    
    frs = []  # firing rate per trial chunk
    pers = []  # performance per trial chunk
    cols = []  # region colors
    regs = []

    
    _, pal = get_allen_info()
    
    for eid in eids:
        
        D = np.load(Path(pthr, f'{eid}.npy'), allow_pickle=True).flat[0]
        
        if np.array_equal(D.get('fp_m', np.array([])), np.array([])):
            print(eid, 'empty data??')
            continue 
        
        
        k = 0     
        for reg in D['regd']:
            frs.append(D['fr_m'][k])   
            pers.append(D['fp_m'])
            cols.append([pal[reg]]*len(D['fp_m']))
            regs.append([reg]*len(D['fp_m']))
            k+=1

    frs_concat = [item for sublist in frs for item in sublist]    
    pers_concat = [item for sublist in pers for item in sublist] 
    cols_concat = [item for sublist in cols for item in sublist]
    regs_concat = [item for sublist in regs for item in sublist]

    df = pd.DataFrame({'firing_rate': frs_concat,
                       'performance': pers_concat,
                       'color': cols_concat,
                       'region': regs_concat})
                       
    if not per_reg:                    
        sns.pairplot(data=df, x_vars='firing_rate', y_vars='performance', 
                      hue='color', height=5, 
                      plot_kws={'alpha': 0.3, 'legend': False},       
                      #plot_kws={'scatter_kws': {'label': None}},
                      kind=kind)

    else:
        # plot s panel per region
        fig, axs = plt.subplots(ncols=17, nrows=10, 
                                sharex=True, sharey=True,
                                figsize=(15,10))

        axs = axs.flatten()
        i = 0
        for reg in np.unique(df['region']):
            
            df_ = df[df['region'] == reg]
            r,p = pearsonr(df_['firing_rate'], df_['performance'])
            
            
            axs[i].scatter(df_['firing_rate'], df_['performance'], 
                           color = pal[reg], label = reg, s=0.1)
            
            axs[i].set_title(f'{reg} {r:.2f}' if p<0.05 else reg,
                             fontsize=10)   
            #axs[i].set_xlabel('firing rate [Hz]')  
            #axs[i].set_ylabel('performance')
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)    

            
            i+=1
            
            
        fig.text(0.5, 0.02, 'firing_rate', ha='center')
    
        fig.text(0.04, 0.5, 'performance', 
                 va='center', rotation='vertical')
        fig.tight_layout()                
                 


def plot_strip_fr_perf(corrtype='pears'):

    '''
    stripplot of firing rate/performance correlation
    per area, dot being a measurement
    '''
    
    pthr = Path(one.cache_dir, 'fr_performance')
    eids = [x.split('.')[0] for x in os.listdir(pthr)]
    
    pears_r = []
    pears_p = []
    spear_r = []
    spear_p = []
    regs = []
    
    _, pal = get_allen_info()
    
    for eid in eids:
        
        D = np.load(Path(pthr, f'{eid}.npy'), allow_pickle=True).flat[0]
        
        if np.array_equal(D.get('fp_m', np.array([])), np.array([])):
            print(eid, 'empty data??')
            continue 
        
        for reg in D['regd']:
            regs.append(reg)
            pears_r.append(D['corrd'][reg][0][0])
            pears_p.append(D['corrd'][reg][0][1])
            spear_r.append(D['corrd'][reg][1][0])
            spear_p.append(D['corrd'][reg][1][1])            
                
    
    df = pd.DataFrame({'pears_r': pears_r,
                       'pears_p': pears_p,
                       'spear_r': spear_r,
                       'spear_p': spear_p,
                       'reg': regs}) 
    
    for key in ['pears_p', 'spear_p']:
        df[key] = df[key] < sigl
    
    
    # order by canonical order
    p = (Path(iblatlas.__file__).parent / 'beryl.npy')
    regs_ = br.id2acronym(np.load(p), mapping='Beryl')
    
    
    regs = np.unique(df['reg'])
    regsC = [reg for reg in regs_ if reg in regs]
    
    df_ordered = df[df['reg'].isin(regsC)].sort_values(by=['reg'], 
                    key=lambda x: x.map({region: i for i, region in 
                    enumerate(regsC)}))
                    
    colors = {False: '#DDDDDD', True: 'black'}
    
    # make ncols columns
    ncols = 5                 
    fig, axs = plt.subplots(ncols=ncols, figsize=(10,15))                  
    _, pal = get_allen_info()
    for k in range(ncols):
        print(k)
        colsreg = regsC[ k * len(regsC)//ncols : (k+1) * len(regsC)//ncols]
                                                              
        df_fil1 = df[df['reg'].isin(colsreg)]
        df_fil1.sort_values(by='reg', 
            key=lambda x: x.map(colsreg.index), inplace=True)

        sns.stripplot(x=f'{corrtype}_r', y='reg', hue = f'{corrtype}_p', 
                      marker='o', size=3, palette=colors,  
                      data=df_fil1, ax=axs[k])

        for label in axs[k].get_yticklabels():
            label.set_color(pal[label.get_text()])
            
        axs[k].spines['top'].set_visible(False)
        axs[k].spines['right'].set_visible(False)
        if k != 0:
            axs[k].legend().remove()
    fig.tight_layout()

  
def plot_gc(eid, segl=10, shuf=False,
            metric0='granger', vers='oscil', 
            peak_freq_factor0=0.55, peak_freq_factor1=0.2,
            phase_lag_factor=0.2, single_pair=False, T=300000):

    '''
    For all regions, plot example segment time series, psg,
    matrix for mean gc across freqs
    
    metric = 'pairwise_spectral_granger_prediction'
    or 'coherence_magnitude'

    For SI:
    plot_gc('af55d16f-0e31-4073-bdb5-26da54914aa2', single_pair=True)
    '''
    time00 = time.perf_counter()
    
    if eid == 'sim':
        r, regsd = make_data(vers = vers, peak_freq_factor0=peak_freq_factor0,
                                   peak_freq_factor1=peak_freq_factor1,
                      phase_lag_factor=0.2, T=T)
        ts = np.linspace(0, (r.shape[1] - 1) * T_BIN, r.shape[1])
    else:
        r, ts, regsd = bin_average_neural(eid)   

    print(regsd)
    
    # single channel pair for shuffle test
    if single_pair:
        regA, regB = -1, -2
        r = r[[regA, regB]]
        regsd = dict(np.array(list(regsd.items()))[[regA, regB]])
    
    if metric0 == 'granger':
        metric = 'pairwise_spectral_granger_prediction'
        
    if metric0 == 'coherence':
        metric = 'coherence_magnitude'


    if metric0 == 'cross_corr':
        # compute mean cross correlation
        nchans, nobs = r.shape
        segment_length = int(segl / T_BIN)
        num_segments = nobs // segment_length
        
        # reshape into: n_signals x n_segments x n_time_samples
        r_segments = r[:, :num_segments * segment_length
                       ].reshape((nchans, num_segments, 
                       segment_length))
                       
        m = np.zeros((nchans,nchans))               
        for i in range(nchans):
            for j in range(i, nchans):
                for seg in range(r_segments.shape[1]):
                    ss = []
                    ss.append(np.mean(np.abs(np.correlate(
                                                r_segments[i,seg], 
                                                r_segments[j,seg], mode='full'))))
                                           
                m[i, j] = np.mean(ss)               
                m[j, i] = m[i, j]              
        
    else:
        c = gc(r, segl=segl, shuf=shuf)
        # freqs x chans x chans
        psg = getattr(c, metric)()[0]
        
        # mean score across frequencies
        m = np.mean(psg, axis=0)
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6.19,3),
                            label = eid)
    
    # plot example time series, first segment
    exdat = r[:,:int(segl/T_BIN)]/T_BIN
    extime = ts[:int(segl/T_BIN)]    
   
    _, pal = get_allen_info()  
    if eid == 'sim':
        pal['dep'] = 'b'
        pal['indep'] = 'r'
        pal['x2[30:]'] = 'b'
        pal['x2[0:-30]'] = 'r'
        
        
    regs = list(regsd)
     
    i = 0
    s = 0
    for y in exdat:       
        axs[0].plot(extime, y + s,c=pal[regs[i]])
        axs[0].text(extime[-1], s, regs[i], 
                    c=pal[regs[i]])
        s += np.max(y)
        i +=1
              
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Firing rate (Hz)')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].set_title('example segment')

    if metric0 != 'cross_corr':
        # plot top five granger line plots with text next to it
        si = np.unravel_index(np.argsort(np.abs(m),axis=None), m.shape)
        # top j connections
        j = 10  
        exes = [tup for tup in reversed(list(zip(*si))) 
                if (~np.isnan(m[tup]) and tup[0] != tup[1])][:j]
        
        for tup in exes: 
            yy = psg[:,tup[0],tup[1]]
            # Order is inversed! Result is:   
            axs[1].plot(c.frequencies, yy, 
                        label =f'{regs[tup[1]]} --> {regs[tup[0]]}') 

        axs[1].legend()
        axs[1].set_xlabel('frequency [Hz]')
        if metric == 'pairwise_spectral_granger_prediction':
            axs[1].set_ylabel('Pairwise spectral \n granger prediction')
        else:
            axs[1].set_ylabel(metric)
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)  
        # axs[1].set_title(f'top {j} tuples') 
    
#     # plot directed granger matrix
#     ims = axs[2].imshow(m, interpolation=None, cmap='gray_r')#, 
#                   #origin='lower')
                  
# #    # highlight max connections              
# #    for i, j in exes:
# #        rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
# #                                  linewidth=2, 
# #                                  edgecolor='red', 
# #                                  facecolor='none')
# #        axs[2].add_patch(rect)              
                  
#     axs[2].set_xticks(np.arange(m.shape[0]))
#     axs[2].set_xticklabels(regs, rotation=90)
#     axs[2].set_xlabel('source')
#     axs[2].set_yticks(np.arange(m.shape[1]))
#     axs[2].set_yticklabels(regs)
#     axs[2].set_ylabel('target')     
#     axs[2].set_title('mean abs corr across lags' 
#                      if metric0 == 'cross_corr' 
#                      else 'mean across freqs')
                  
#     cb = plt.colorbar(ims,fraction=0.046, pad=0.04)              


    # fig.suptitle(f'eid = {eid} {"shuffled" if shuf else ""} ' 
    #              f'vers={vers}, peak_freq_factor0 = {peak_freq_factor0}, '
    #              f'peak_freq_factor1 = {peak_freq_factor1}, ' 
    #              f'phase_lag_factor={phase_lag_factor},'
    #              if eid == 'sim' else 
    #              f'eid = {eid} {"shuffled" if shuf else ""} ')   
    fig.tight_layout()
    time11 = time.perf_counter()
    print('runtime [sec]: ', time11 - time00)     
    
    
    
def plot_strip_pairs(metric='granger', sessmin = 3, 
             ptype='strip', shuf=False, expo=1, sig_only=True):

    '''
    for spectral Granger, metric in ['granger', coherence']
    '''
    d0 = get_res(metric=metric, sig_only=sig_only, combine_=False)

                        
    regs = list(Counter(np.array([s.split(' --> ') for s in
                             d0]).flatten()))
                             
    _, palette = get_allen_info()
                                  
    if ptype == 'strip':                                             
        if metric == 'coherence':
            # remove directionality
            sep = ','
            
            d = {}
            for s in d0:
                a,b = s.split(' --> ')
                if ((sep.join([a,b]) in d) or (sep.join([b,a]) in d)):
                    continue
                else:
                    d[sep.join([a,b])] = d0[s]
            
  
        else:
            d = d0
            sep = ' --> '
                
                  
        dm = {x: np.mean(d[x]) for x in d if (len(d[x]) >= sessmin)}
        
        dm_sorted = dict(sorted(dm.items(), key=lambda item: item[1]))
                      
        exs = list(dm_sorted.keys())
        nrows = 5
        fs = 5
        per_row = math.ceil(len(exs)/nrows)
           
        d_exs = {x:d[x] for x in exs}
    
        fig, axs = plt.subplots(nrows=nrows, 
                                ncols=1, figsize=(9,6.76), sharey=True)

        for row in range(nrows):
            
            pairs = exs[per_row*row: per_row*(row+1)]
            
            # plot also bar with means
            ms = [np.mean(d[x]) for x in pairs]            
            axs[row].bar(np.arange(len(pairs)), ms, 
                         color='grey', alpha=0.5)
            
            
            data = pd.DataFrame.from_dict({x:d[x] for x in pairs},
                orient='index').transpose()             
            sns.stripplot(data=data, ax=axs[row], color='k', size=2)

            
            axs[row].set_xticklabels([x.split(sep)[0] for x in pairs], 
                                      rotation=90, 
                                      fontsize=fs if 
                                      (metric == 'granger') else 8)
            
            for label in axs[row].get_xticklabels():
                label.set_color(palette[label.get_text()])
                
            low_regs = [x.split(sep)[-1] for x in pairs]   
            for i, tick in enumerate(axs[row].get_xticklabels()):
                axs[row].text(tick.get_position()[0], -0.7, low_regs[i],
                    ha='center', va='center', rotation=90,
                    color=palette[low_regs[i]],
                    transform=axs[row].get_xaxis_transform(), 
                    fontsize=fs if (metric == 'granger') else 8)    
            
            axs[row].set_ylabel(metric)
                   

        fig.tight_layout()
        print(f'sessmin={sessmin}; sig_only={sig_only};'
              f'#reg pairs {len(dm)}')        
        
    else:
        # plot matrix of all regions 
        
        # canonical region order
        # order regions by canonical list 
        p = (Path(iblatlas.__file__).parent / 'beryl.npy')
        regs_c = br.id2acronym(np.load(p), mapping='Beryl')


        regs_ = []
        for reg in regs_c:
            if reg in regs:
                regs_.append(reg)
                
        regs = regs_        
              
        M = np.zeros((len(regs),len(regs)))
        for i in range(len(regs)):
            for j in range(len(regs)):
                if (i == j) or (' --> '.join([regs[i], regs[j]]) not in d0):
                    M[i,j] = 0
                else:               
                    M[i,j] = np.mean(d0[' --> '.join([regs[i], regs[j]])])
       
        if shuf:
            # shuffle region list         
            np.random.shuffle(regs)
            print('region list shuffled')
        
        
        if ptype=='emb':        
        
        
            # incomplete embedding; distance matrix entries with 0        
            emb = MDS(n_components=2,metric=False, 
                      dissimilarity='precomputed').fit_transform(1 - M**expo)
                      
            cols = np.array([palette[reg] for reg in regs])
            
            
            fig, ax = plt.subplots(figsize=(10,10))
            
            ax.scatter(emb[:,0], emb[:,1],c=cols, s = 20)

                     
            for i in range(len(regs)):
                ax.annotate('  ' + regs[i], 
                    (emb[i][0], emb[i][1]),
                    fontsize=10,color=palette[regs[i]])   
                    
            ax.set_title(f'Dissimilarity: 1 - M^{expo}')                   

        else:            
       
            fig, ax = plt.subplots(figsize=(10,10))
            
            
            # plot directed granger matrix
            ims = ax.imshow(M, interpolation=None, cmap='gray_r', 
                          origin='lower')
                          
            ax.set_xticks(np.arange(M.shape[0]))
            ax.set_xticklabels(regs, rotation=90)
            ax.set_xlabel('source')
            ax.set_yticks(np.arange(M.shape[1]))
            ax.set_yticklabels(regs)
            ax.set_ylabel('target')     
            ax.set_title(f'mean across sessions')
                          
            cb = plt.colorbar(ims,fraction=0.046, pad=0.04)              
        
        
            #fig.suptitle(f'eid = {eid} {"shuffled" if shuf else ""}')   
            fig.tight_layout()        

    
def scatter_two(winns=['whole_session', 'charles_ephys_Granger'],
                sig_only=True):

    # winns=['whole_session', 'feedback_plus1']  
    
    if winns[1] == 'charles_ephys_Granger':
    
        # group scores across recordings    
        dg = get_res(win=winns[0],combine_=True)
    
        dc0 = pd.read_parquet(Path(one.cache_dir, 'granger', 
                                f'{winns[1]}.pqt'))
        # change format

        dc = {' --> '.join(x.split(' -> ')): 
                [dc0[dc0['reg'] == x]['corrected_score'].item(),
                 dc0[dc0['reg'] == x]['pvalue'].item()]
                for x in dc0['reg'].values}
                               
                                
                                  
    else:
        dg = get_res(win=winns[0],combine=False)     
        dc = np.load(Path(one.cache_dir, 'granger', 
                            f'granger_{winns[1]}.npy'), 
                            allow_pickle=True).flat[0]
                                               
    pairs = list(set(dg.keys()).intersection(set(dc.keys())))
    
    pts = []
    scores = []

    if winns[1] == 'charles_ephys_Granger':
        for p in pairs:
                scores.append([dg[p][0],dg[p][1], 
                               dc[p][0],dc[p][1]])
                pts.append(p)    
    else:
        for p in pairs:
            for i in range(len(dg[p])):
                scores.append([dg[p][i][0],dg[p][i][1], 
                               dc[p][i][0],dc[p][i][1]])
                pts.append(p)
                  
    scores = np.array(scores) 
    scores[:,0] = np.log(scores[:,0])
    scores[:,2] = np.log(scores[:,2])
            
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # all pairs in black
    ax.scatter(scores[:,0], scores[:,2], color='k', s=1, label='neither sig')
    
    # sig ones x in red crosses
    ax.scatter(scores[:,0][scores[:,1]<sigl], scores[:,2][scores[:,1]<sigl], 
               color='r', s=15, marker='o', label='x sig')
    
    # sig ones y in blue circles
    ax.scatter(scores[:,0][scores[:,3]<sigl], scores[:,2][scores[:,3]<sigl], 
               color='b', s=15, marker='x', label='y sig')
    
    plt.legend()      
    ax.set_xlabel(f'{winns[0]} (log(granger))')       
    ax.set_ylabel(f'{winns[1]} (log(granger))')
    both_sig = np.bitwise_and(scores[:,1]<sigl, scores[:,3]<sigl) 
    
    cors,ps = spearmanr(scores[:,0][both_sig], scores[:,2][both_sig])
    corp,pp = pearsonr(scores[:,0][both_sig], scores[:,2][both_sig])
    ax.set_title(f'pearson: (r,p)=({np.round(corp,2)},{np.round(pp,2)}),'
                 f'spearman: (r,p)=({np.round(cors,2)},{np.round(ps,2)});'
                 ' only for pairs where both score types were significant')
    
    
def plot_dist_scat(dist_='centroids'):

    '''
    correlate granger and coherence with distance of pair
    '''
    
    
    if dist_ == 'centroids':
        dcent = get_centroids()
        
    elif dist_ == 'structural':
        dstru = get_structural()
    

    dg = np.load(Path(one.cache_dir, 'granger', 
                        f'granger_all.npy'), 
                        allow_pickle=True).flat[0]    
    
    dc = np.load(Path(one.cache_dir, 'granger', 
                        f'coherence_all.npy'), 
                        allow_pickle=True).flat[0]    
        
    pts = []
    gs = []
    cs = []
    dists = []

    
    for p in dg:
        a,b = p.split(' --> ')
        
        if dist_ == 'centroids':
            dist = sum((dcent[a] - dcent[b])**2)**0.5
            
        elif dist_=='structural':
            if p in dstru:
                dist = dstru[p]
            else:
                continue
            
        for i in range(len(dg[p])):
            gs.append(dg[p][i])
            cs.append(dc[p][i])
            pts.append(p)    
            dists.append(dist)
             
    
    fig, axs = plt.subplots(ncols=2, sharex=True)
    
    
    ylabs = ['granger', 'coherence']
    vals = [gs, cs]
    
    for k in range(len(vals)):
        
        axs[k].scatter(dists, vals[k], color='k', s=0.5)
        
        for i in range(len(pts)):
            axs[k].annotate('  ' + pts[i], 
                (dists[i], vals[k][i]),
                fontsize=5,color='k')
                
        axs[k].set_xlabel('structural connectivity' if 
                          dist_ == 'structural' else 'centroid distance')       
        axs[k].set_ylabel(ylabs[k])

        cors,ps = spearmanr(dists, vals[k])
        corp,pp = pearsonr(dists, vals[k])

        axs[k].set_title(f'({np.round(corp,2)},{np.round(pp,2)}), '
                     f'({np.round(cors,2)},{np.round(ps,2)})')    


def replot_struc():

    '''
    replot structural connectivity matrix from Allen data
    '''

    # get region order as in paper
    s=pd.read_excel('/home/mic/fig3.xlsx')
    cols = list(s.keys())[1:296]
    rows = s['Unnamed: 0'].array
    
    M = np.zeros((len(cols), len(rows)))
    for i in range(len(cols)):
        M[i] = s[cols[i]].array
        
    M = M.T

    # thresholding as in the paper
    M[M > 10**(-0.5)] = 1
    M[M < 10**(-3.5)] = 0

    cols1 = [reg.strip().replace(",", "") for reg in cols]
    rows1 = [reg.strip().replace(",", "") for reg in rows]
    
    fig, ax = plt.subplots()            
    ax.imshow(M)        
                                             
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols1, 
                       rotation=90, fontsize=5)
                                
    _, pal = get_allen_info()
    for label in ax.get_xticklabels():
        label.set_color(pal[label.get_text()]) 

    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows1, fontsize=5)
                                
    for label in ax.get_yticklabels():
        label.set_color(pal[label.get_text()]) 
 
    ax.set_title('structural connectivity, fig3 in Allen paper')
      
 
def freq_maxs_hists(perc = 95, freqlow=10):

    '''
    histograms of peak frequencies
    perc: threshold of percentile of scores to highlight 
        high freq interaction 
    '''

    d = get_meta_info()

    thr = {metric: np.nanpercentile(np.concatenate([d[eid][metric].flatten() 
                              for eid in d]), perc) for metric in 
                              ['granger', 'coherence']}

    fig, axs = plt.subplots(ncols=2, sharex=True)
    k = 0
    for metric in ['granger', 'coherence']:

        pks = []
        sess_high = []
        for eid in d:
            ks = d[eid][f'{metric}_pks'][d[eid][f'p_{metric}'] == 0.0]
            pks.append(ks)
            ks2 = d[eid][f'{metric}_pks'][np.bitwise_and.reduce([
                d[eid][f'p_{metric}'] == 0.0,
                d[eid][f'{metric}'] > thr[metric],
                d[eid][f'{metric}_pks'] > freqlow])]
            
            if np.all(ks2.size == 0):
                continue
            else:
                sess_high.append(eid)        
            
     
        pks = np.concatenate(pks) 
        axs[k].hist(pks,bins=600)
        axs[k].set_xlabel('peak frequency [Hz]')
        axs[k].set_ylabel('# region pairs')
        axs[k].set_title(metric)
        print(metric)
        print(sess_high) 
        k+=1

    fig.suptitle('only region-shuffle significant')
    fig.tight_layout()
    

def scatter_direction(sig_only=True, annotate=True):

    '''
    scatter plot for region pairs 
    Granger A --> B on x, B --> A on y
    source region colored
    '''
    
    _, pa = get_allen_info()    
    dg = get_res(metric='granger',combine_=False, sig_only = sig_only)

    sep = ' --> '
    
    pairs = []
    for s in dg:
        a,b = s.split(sep)
        # check if both directions were significant
        if ((sep.join([a,b]) in dg) and (sep.join([b,a]) in dg)):
            if (([a,b] in pairs) or ([b,a] in pairs)):
                continue
            else:
                pairs.append([a,b])    
            
    # for each region pair get mean granger 
    dir0 = []
    dir1 = []
    pairs0 = []
    for pair in pairs:
        a,b = pair
        dir0.append(np.mean(dg[sep.join([a,b])]))
        dir1.append(np.mean(dg[sep.join([b,a])]))
        pairs0.append(', '.join([a,b]))
          
            
    fig, ax = plt.subplots(figsize=(3,3))
    ax.scatter(dir0, dir1, color='k', s=0.5)
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
    
    if annotate:
        for i in random.sample(range(len(pairs)),3):
            ax.annotate('  ' + pairs0[i], 
                (dir0[i], dir1[i]),
                fontsize=10,color='k', fontweight='bold')  

            
    ax.set_xlabel('A --> B')       
    ax.set_ylabel('B --> A')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
         
    cors,ps = spearmanr(dir0, dir1)
    corp,pp = pearsonr(dir0, dir1)

    ax.set_title(f'pearson: (r,p)=({np.round(corp,2)},{np.round(pp,2)}) \n'
                 f'spearman: (r,p)=({np.round(cors,2)},{np.round(ps,2)})')
                 
    ax.set_xscale('log')
    ax.set_yscale('log') 
       
    fig.tight_layout()    


def get_ari():

    visual_sensory = ["VISp", "SSs", "BST", "PRM", "ANcr1", "LSr", "CA1", "SUB", "VISam", "VISpm", "AON"]
   
    stim_integrator = set([
    "CENT3", "GPe", "PRNr", "IRN", "PL", "MOs", "ORBvl", "MRN",
    "IP", "PRM", "BST", "CP", "PRNr", "GRN", "MV", "DG", "CA3", "ProS", "SUB", "ZI", "SSp-l", "MOp", "MOs", "ACAd",
    "RSPagl", "SSs", "ORBl", "VISp", "SCm", "APN", "Eth", "RT", "VAL", "PO", "LP", "LGd", "SSp-bfd", "SSp-m", "VPM", "VPL"])
    
    choice_action = set([
    "CUL45", "SIM", "IP", "CENT2", "CENT3", "PRNr", "GRN", "MV", "ILA", "ORBm", "AId", "AIv", "MOp", "VISa", "SSp-ul",
    "RSPagl", "PL", "ACAd", "PAG", "SCm", "RT", "VPL",
    "ANcr1", "BST", "MEA", "SI", "LSv", "CA1", "DG", "CA3", "SSp-l", "SSp-tr", "SSp-ul", "PL", "ILA", "ACAv", "RSPv",
    "IC", "MG", "PoT", "MD", "LD", "Eth"])
    
    block_prior = set([
    "SIM", "CENT3", "IP", "ANcr1", "CENT2", "CP", "GRN", "IRN", "ENTl", "SUB", "ZI", "MOp", "ACAd", "AUDp", "RSPagl",
    "PL", "AIv", "MOs", "PAG", "APN", "MRN", "PoT", "VAL", "LD", "LGd",
    "NOD", "ANcr1", "LSr", "ACB", "CP", "PB", "MPO", "SSp-n", "AIp", "ECT", "VISli", "AId", "RSPd", "PL", "SSs", "TEa",
    "VISa", "MOs", "AUDp", "RN", "PIR", "DP", "SMT", "Eth"])

    unique_visual_sensory = set(visual_sensory)

    # Remove duplicates from the subsequent lists
    unique_stim_integrator = [x for x in stim_integrator 
        if x not in unique_visual_sensory]
    unique_choice_action = [x for x in choice_action 
        if x not in unique_visual_sensory and x not in unique_stim_integrator]
    unique_block_prior = [x for x in block_prior 
        if x not in unique_visual_sensory and x not in unique_stim_integrator
        and x not in unique_choice_action]

    print('unique_visual_sensory')
    print(list(unique_visual_sensory))
    print('unique_stim_integrator')    
    print(list(unique_stim_integrator))
    print('unique_choice_action')        
    print(list(unique_choice_action))
    print('unique_block_prior')     
    print(list(unique_block_prior))     

    return np.concatenate([list(unique_visual_sensory),    
                           unique_stim_integrator,    
                           unique_choice_action,     
                           unique_block_prior])     


def plot_graph(metric='granger', restrict='', ax=None, win='whole_session',
               direction='both', sa = 1.5, sessmin=2, acronym_ring=True,
               ari=False, sig_only=False, ews = 50, rrad=0.6):

    '''
    circular graph
    
    highlight in cosmos regions, to only show those 
    
    if restrict in a certain cosmos region, only show these edges
    ['CB', 'TH', 'HPF', 'Isocortex', 'OLF', 'CTXsp', 'CNU', 'HY', 'HB', 'MB']
    
    If ari: order regions by Ari's lists
    
    '''
    from dmn_bwm import trans_, get_umap_dist, get_pw_dist
    alone = False
    if ax == None:
        alone = True
        fig, ax = plt.subplots(figsize=(4,4), label=win)

    if metric == 'cartesian':
        d = trans_(get_centroids(dist_=True))
    elif metric == 'granger':
        d = get_res(metric=metric, sessmin=sessmin, win=win, combine_=True)   
    elif metric == 'pw':
        d = trans_(get_pw_dist(vers='concat'))   
    elif metric == 'umap_z':     
        d = trans_(get_umap_dist(algo='umap_z', vers='concat'))
    else:
        print('what metric?')
        return
   
    # ews was 80
    ews = ews if metric == 'granger' else ews/10  # edge width
    fontsize = 11 if alone else 1
    
    # scale symbols for multi-panel graphs
    node_size = 30 if alone else 3
    
    # non-significant edge width
    if sig_only:
        nsw = 0
    else:
        nsw = 0.02 if metric == 'coherence' else 0.005
    ews = ews/sa  # edge width
    node_size = node_size/sa
    fontsize = fontsize/sa
       
    # get a dict to translate Beryl acronym into Cosmos
    file_ = download_aggregate_tables(one)
    df = pd.read_parquet(file_)
    dfa, palette = get_allen_info()
    df['Beryl'] = br.id2acronym(df['atlas_id'], mapping='Beryl')
    df['Cosmos'] = br.id2acronym(df['atlas_id'], mapping='Cosmos')  
    cosregs = dict(list(Counter(zip(df['Beryl'],df['Cosmos']))))
                     
    _, pa = get_allen_info()
            

    G = nx.DiGraph()
    for edge, weight in d.items():
        source, target = edge.split(' --> ')
        if (source in ['void', 'root']) or (target in ['void', 'root']):
            continue
        
        if metric in ['granger', 'coherence']:
            w = weight[0] if (weight[1] < sigl) else nsw
            w = w*ews
            G.add_edge(source, target, 
                       weight=w, 
                       color='k' if weight[1] < sigl else 'cyan')
        else:
            w = weight[0]
            w = w*ews
            G.add_edge(source, target, 
                       weight=w, 
                       color='k')

    if ari:
        rs = get_ari()
    
        ints = []
        for reg in rs:
            if reg in G.nodes:
                ints.append(reg)
        
        rems = [reg for reg in G.nodes if reg not in ints] 
        print(list(ints)[0], rems[0])
        node_order = list(ints) + rems    
    
    else:        
        # order regions by canonical list 
        p = (Path(iblatlas.__file__).parent / 'beryl.npy')
        regs = br.id2acronym(np.load(p), mapping='Beryl')


        node_order = []
        for reg in regs:
            if reg in G.nodes:
                node_order.append(reg)
                  
    pos0 = nx.circular_layout(G)
    pos = dict(zip(node_order,pos0.values()))

    nx.draw_networkx_nodes(G, pos, node_size=node_size, ax=ax,
                           node_color=[pa[node] for node in G.nodes])

    for edge in G.edges():
    
        # only plot edges for certain Cosmos region    
        if restrict != '':
            if direction == 'both':
                if (cosregs[edge[0]] != restrict and 
                   cosregs[edge[1]] != restrict):
                    continue 
            if direction == 'source':
                if (cosregs[edge[0]] == restrict and 
                   cosregs[edge[1]] != restrict):
                    pass    
                else:
                    continue
                    
            if direction == 'target':
                if (cosregs[edge[0]] != restrict and 
                   cosregs[edge[1]] == restrict):
                    pass    
                else:
                    continue            
                
        w = G.edges[edge]['weight']
        nx.draw_networkx_edges(G, pos, edgelist=[(edge[0],edge[1])], 
        arrowsize=w*10, width=w, 
        edge_color=G[edge[0]][edge[1]]['color'],
        connectionstyle=f'arc3,rad={rrad}', ax=ax)

    if acronym_ring:    
        for node, (x, y) in pos.items():
            angle = np.arctan2(y, x)
            angle = np.degrees(angle)
            # Radial shift factor (adjust as needed)
            r_shift = 1.3 #1.12
                
            # Calculate new positions
            x_new = r_shift * np.cos(np.radians(angle))
            y_new = r_shift * np.sin(np.radians(angle))

            q = (' ---- ' if cosregs[node] == restrict 
                and restrict != '' else '')              
            ax.text(x_new, y_new, node + q if x < 0 else q + node,
                    fontsize=fontsize, ha='center', 
                    va='center', rotation=angle if x > 0 else angle + 180,
                    color=pa[node])
    
        
    ax.set_aspect('equal')
    ax.set_axis_off()
        
    if alone:
        #ax.set_title(f'{metric}, black edges significant; restrict {restrict}')
        fig.tight_layout()
        
        pth_ = Path(pth_res, 'plots', f'graph_{win}.png')
        fig.savefig(pth_, dpi=200)         
#        fig.savefig(Path(one.cache_dir,
#                        'bwm_res/bwm_figs_imgs/si/granger/',
#                        'granger_single_graph.svg'))


def plot_multi_graph(sessmin=2, win='whole_session', sig_only=False, sa=2, rrad=0.2,
                     acronym_ring=False):

    cregs = ['CB', 'TH', 'HPF', 'Isocortex', 
             'OLF', 'CTXsp', 'CNU', 'HY', 'HB', 'MB']
 
    directions = ['source', 'target']
    fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(8.5,14), label=win)
    axs = axs.flatten()
    
    k = 0
    for creg in cregs: 
        for direction in directions:
            plot_graph(metric='granger', restrict=creg, sessmin = sessmin, 
                       ax=axs[k], sa = sa, direction=direction, win=win, 
                       sig_only=sig_only, rrad=rrad, acronym_ring=acronym_ring)     
            axs[k].set_title(f'{creg} {direction}')
            k += 1
  
    fig.tight_layout()
#    fig.savefig(Path(one.cache_dir,
#                    'bwm_res/bwm_figs_imgs/si/granger/',
#                    f'granger_multi_graph_{win}.svg'))

    pth_ = Path(pth_res, 'plots', f'multi_graph_{win}.png')
    fig.savefig(pth_, dpi=200)                    
                    
                    

def plot_series(eid):

    '''
    plot full time series
    '''
    #plt.ioff()
    r, ts, regsd = bin_average_neural(eid)
    _, pa = get_allen_info()    
    fig, axs = plt.subplots(nrows=len(regsd), 
                            figsize=(15,10),
                            sharex=True)
    axs = np.array(axs).flatten()
    regs = list(regsd)
    for i in range(len(regsd)):
        axs[i].plot(ts, r[i],c=pa[regs[i]])
        axs[i].set_ylabel(f'{regs[i]} \n {regsd[regs[i]]}',
                          c=pa[regs[i]])   

    axs[i].set_xlabel('time [sec]')
    fig.suptitle(f'eid = {eid} \n combining both probes,'
                 f' firing rate per region')
    fig.tight_layout()
    fig.savefig(f'top_granger/{eid}.png')
    #fig.close()


def make_table():

    '''
    table to accompany granger results
    '''
    
    cols = ['source name', 'source #neur', 'target name', 'target #neur', 
           'Granger score', 'corrected p', 'eid']
    
    d = get_res(metric='granger', combine_=False, rerun=True)

    r = []
    for pair in d:
        for m in d[pair]:
            r.append([pair.split(' --> ')[0], m[2],
                     pair.split(' --> ')[1], m[3], m[0], m[1], m[-1]])
                 
    df = pd.DataFrame(r, columns=cols)
    df_sorted = df.sort_values(by='Granger score', ascending=False)
    df_sorted.to_csv(Path(pth_res.parent, 'granger.csv'), index=False)



def heatmap_adjacency():

    data = get_res(c_mc=True, sig_only=True, combine_=True, sessmin=1)

    # Step 1: Extract unique brain regions
    regions = set()
    for key in data.keys():
        source, target = key.split(' --> ')
        regions.add(source)
        regions.add(target)

    # Sort regions for consistency

    # order by canonical order
    p = (Path(iblatlas.__file__).parent / 'beryl.npy')
    regs_ = br.id2acronym(np.load(p), mapping='Beryl')

    regsC = [reg for reg in regs_ if reg in regions]
    regions = regsC

    # Step 2: Create a mapping of regions to matrix indices
    region_index = {region: idx for idx, region in enumerate(regions)}

    # Step 3: Initialize an adjacency matrix (square, with zeroes)
    n = len(regions)
    adj_matrix = np.zeros((n, n))

    # Step 4: Populate the adjacency matrix using the dictionary data
    for key, values in data.items():
        source, target = key.split(' --> ')
        source_idx = region_index[source]
        target_idx = region_index[target]
        
        # Use the first Granger score (or you can decide how to handle multiple scores)
        adj_matrix[source_idx][target_idx] = np.mean(values) # [0]

    # Step 5: Convert the adjacency matrix to a pandas DataFrame for readability
    adj_matrix_df = pd.DataFrame(adj_matrix, index=regions, columns=regions)


    _, palette = get_allen_info()
    # Step 7: Create a heatmap with colored region labels
    fig, ax = plt.subplots(figsize=(3, 3))

    # Plot the heatmap
    #sns.heatmap(adj_matrix_df, cmap='viridis', ax=ax, cbar=True, square=True)
    sparse_matrix = csr_matrix(adj_matrix_df)
    ax.spy(sparse_matrix, markersize=1, color='k')


    # Set tick label colors and ensure all ticks are shown
    ax.set_xticks(np.arange(len(regions)) + 0.5)  # Ensure all ticks are placed correctly
    ax.set_yticks(np.arange(len(regions)) + 0.5)

    ax.set_xticklabels(regions, fontsize=6, rotation=90)  # Small fontsize and rotate labels for x-axis
    ax.set_yticklabels(regions, fontsize=6)

    # Set tick label colors based on the palette
    for label in ax.get_yticklabels():
        region = label.get_text()
        label.set_color(palette.get(region, 'black'))  # Default to black if not in palette
    
    for label in ax.get_xticklabels():
        region = label.get_text()
        label.set_color(palette.get(region, 'black'))  

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()

    # Save or show the figure
    plt.show()

  
  
def scatter_similarity(ranks=False, hexbin_=False, anno=False):

    '''
    for pairs of 3 similarity metrics, 
    scatter and correlate region pairs;
    '''

    D = {}


    D['cartesian']= trans_(get_centroids(dist_=True))
    D['granger'] = get_res(metric='granger', sig_only=True, combine_=True, sessmin=1)
    D['axonal'] = get_structural(fign=3)


#         '30ephys': trans_(get_umap_dist(algo='umap_e')),
#         #'coherence': get_res(metric='coherence', 
                              #sig_only=True, combine_=True),
         #'granger': get_res(metric='granger', 
                           # sig_only=True, combine_=True),
         #'structural3_sp': get_structural(fign=3, shortestp=True),
         #'axonal': get_structural(fign=3)
         
     
    # tt = len(list(combinations(list(D.keys()),2)))
    # ncols = math.ceil(math.sqrt(tt))
    # nrows = math.ceil(tt / ncols)

       
    nrows = 3
    ncols = 1    
        
     
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows,
                           figsize=[2.57, 5.51])     
    ax = np.array(ax).flatten()
    
    metrics = list(D.keys())   
    nf = list(combinations(range(len(D)),2))
     
    for k in range(len(nf)):
       
        dg,dc = D[metrics[nf[k][0]]], D[metrics[nf[k][1]]]
                        
        pairs = list(set(dg.keys()).intersection(set(dc.keys())))
        
        pts = []
        gs = []
        cs = []
        
        for p in pairs:
            gs.append(np.mean(dg[p]))
            cs.append(np.mean(dc[p]))
            pts.append(p)        

        gs = gs
        cs = cs
        pts = pts
        
        corp,pp = pearsonr(gs, cs)
        cors,ps = spearmanr(gs, cs)
        

        if ranks:
            gs = np.argsort(np.argsort(gs))
            cs = np.argsort(np.argsort(cs))

            
        if hexbin_:
            ax[k].hexbin(gs, cs, cmap='Greys', gridsize=150)
        else:                
            ax[k].scatter(gs, cs, color='b' if ranks else 'k', 
                          s=0.1, alpha=0.1, rasterized=True)

        if anno:
     
            for i in range(len(pts)):
                ax[k].annotate('  ' + pts[i], 
                    (gs[i], cs[i]),
                    fontsize=5,color='b' if ranks else 'k')
                       
        # cc = ('r' if 'cartesian' in (metrics[nf[k][0]], metrics[nf[k][1]]) 
        #       else 'k')  
        cc = 'k'

        a = 'ranks' if ranks else ''
        ax[k].set_xlabel(f'{metrics[nf[k][0]]} '.capitalize() + a, color=cc)       
        ax[k].set_ylabel(f'{metrics[nf[k][1]]} '.capitalize() + a, color=cc)
        ax[k].spines['right'].set_visible(False)
        ax[k].spines['top'].set_visible(False)
        ss = (f"{np.round(corp,2) if pp<0.05 else '_'}, "
              f"{np.round(cors,2) if ps<0.05 else '_'}")
        ax[k].set_title(ss + f'\n {len(pts)}')
    
        print(metrics[nf[k][0]], metrics[nf[k][1]], len(pts))
        print(f'pe: (r,p)=({np.round(corp,2)},{np.round(pp,2)})')
        print(f'sp: (r,p)=({np.round(cors,2)},{np.round(ps,2)})')
        
    # Check if axes is taken, if not, switch axes off
    [a.axis('off') for a in ax if not a.title.get_text()]


    fig.tight_layout()