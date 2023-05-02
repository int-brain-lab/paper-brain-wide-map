from one.api import ONE
from brainbox.singlecell import bin_spikes2D
from brainwidemap import (bwm_query, load_good_units, 
                          load_trials_and_mask, bwm_units)
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
import ibllib
from ibllib.atlas.flatmaps import plot_swanson_vector 
from brainbox.io.one import SessionLoader

from scipy import signal
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
import gc
from pathlib import Path
import random
from copy import deepcopy
import time
import sys
import math
import string
import os
from scipy.stats import spearmanr

from matplotlib.axis import Axis
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, LinearSegmentedColormap   
from matplotlib.gridspec import GridSpec   
from statsmodels.stats.multitest import multipletests



'''
script to process the BWM dataset for manifold analysis,
saving intermediate results (PETHs), computing
metrics and plotting them (plot_all, also supp figures)

A split is a variable, such as stim, where the trials are
disected by it - e.g. left stim side and right stim side

To compute all from scratch, including data download, run:

##################
for split in ['choice', 'stim','fback','block']:
    get_all_d_vars(split)  # computes PETHs, distance sums
    d_var_stacked(split)  # combine results across insertions
    
plot_all()  # plot main figure    
##################    
'''


np.set_printoptions(threshold=sys.maxsize)

plt.rcParams.update(plt.rcParamsDefault)
plt.ion()

f_size = 15  # font size

# canonical colors for left and right trial types
blue_left = [0.13850039, 0.41331206, 0.74052025]
red_right = [0.66080672, 0.21526712, 0.23069468]

b_size = 0.0125  # bin size [sec] for neural binning
sts = 0.002  # stride size in [sec] for overlapping bins
ntravis = 30  # #trajectories for vis, first 2 real, rest pseudo

# trial split types, with string to define alignment
align = {'stim': 'stim on',
         'choice': 'motion on',
         'fback': 'feedback',
         'block': 'stim on'}

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)
ba = AllenAtlas()
br = BrainRegions()
units_df = bwm_units(one)  # canonical set of cells


# save results here
pth_res = Path(one.cache_dir, 'manifold', 'res')
pth_res.mkdir(parents=True, exist_ok=True)

sigl = 0.01  # significance level (for stacking and plotting)

def pre_post(split, can=False):
    '''
    [pre_time, post_time] relative to alignment event
    split could be contr or restr variant, then
    use base window
    
    ca: If true, use canonical time windows
    '''

    pre_post0 = {'stim': [0, 0.15],
                 'choice': [0.15, 0],
                 'fback': [0, 0.7],
                 'block': [0.4, -0.1]}

    # canonical windows
    pre_post_can =  {'stim': [0, 0.1],
                     'choice': [0.1, 0],
                     'fback': [0, 0.2],
                     'block': [0.4, -0.1]}

    pp = pre_post_can if can else pre_post0

    if '_' in split:
        return pp[split.split('_')[0]]
    else:
        return pp[split]


def grad(c, nobs, fr=1):
    '''
    color gradient for plotting trajectories
    c: color map type
    nobs: number of observations
    '''

    cmap = mpl.cm.get_cmap(c)

    return [cmap(fr * (nobs - p) / nobs) for p in range(nobs)]


def eid_probe2pid(eid, probe_name):

    df = bwm_query(one)    
    return df[np.bitwise_and(df['eid'] == eid, 
                             df['probe_name'] == probe_name)]['pid']
                         

def generate_pseudo_blocks(
        n_trials,
        factor=60,
        min_=20,
        max_=100,
        first5050=90):
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
    '''
    get verbose name for brain region acronym
    '''
    regid = br.id[np.argwhere(br.acronym == brainregion)][0, 0]
    return br.name[np.argwhere(br.id == regid)[0, 0]]


def get_region_info(reg):

    '''
    return sessions for a given region, ordered by cell count
    '''
    
    units_df = bwm_units(one)
    return Counter(units_df[units_df['Beryl']==reg]['eid'])
    
    
def get_eid_info(eid):

    '''
    return counter of regions for a given session
    '''
    units_df = bwm_units(one)

    return Counter(units_df[units_df['eid']==eid]['Beryl'])    


def get_d_vars(split, pid, mapping='Beryl', control=True, get_fr=False,
               nrand=1000, contr=None, restr=False):
    '''
    for a given variable and insertion,
    cut neural data into trials, bin the activity,
    compute distances of trajectories per cell
    to be aggregated across insertions later
    Also save PETHs and cell numbers per region

    input
    split: trial split variable, such as choice side
    pid: insertion id
    control: if nrand randomized trials are averaged
    contr: contrast as a float, only for split = choice
    restr: restrict cells to those from other analysis (decoding)
    get_fr: get firing rates only

    returns:
    Dictionary D_ with entries
    acs: region acronyms per cell
    ws: PETHs for both trial types
    d_eucs: Euclidean distance between PETHs,
            summed across same reg
    d_vars: cell wise variance normalized distance,
            summed across same reg
    '''

    eid, probe = one.pid2eid(pid)

    # load in spikes
    spikes, clusters = load_good_units(one, pid)

    # Load in trials data and mask bad trials (False if bad)
    trials, mask = load_trials_and_mask(one, eid) 
                                     
    events = []
    trn = []

    if split == 'choice':
        for choice in [1, -1]:

            if contr is None:  # include any contrast
                events.append(
                    trials['firstMovement_times'][
                        np.bitwise_and.reduce(
                            [mask, trials['choice'] == choice])])
                trn.append(
                    np.arange(len(trials['choice']))[np.bitwise_and.reduce(
                        [mask, trials['choice'] == choice])])

            else:  # include only trials with given contrast
                events.append(
                    trials['firstMovement_times'][
                        np.bitwise_and.reduce([
                            mask, trials['choice'] == choice,
                            np.bitwise_or(
                                trials['contrastLeft'] == contr,
                                trials['contrastRight'] == contr)])])
                trn.append(
                    np.arange(len(trials['choice']))[
                        np.bitwise_and.reduce([
                            mask, trials['choice'] == choice,
                            np.bitwise_or(
                                trials['contrastLeft'] == contr,
                                trials['contrastRight'] == contr)])])

    elif split == 'stim':
        for side in ['Left', 'Right']:
            events.append(trials['stimOn_times'][np.bitwise_and.reduce(
                [mask, ~np.isnan(trials[f'contrast{side}'])])])
            trn.append(
                np.arange(len(trials['stimOn_times']))[
                    np.bitwise_and.reduce([
                        mask, ~np.isnan(trials[f'contrast{side}'])])])
                        
    elif split == 'stim_cl':  # restrict to left choice only
        for side in ['Left', 'Right']:
            events.append(trials['stimOn_times'][np.bitwise_and.reduce(
                [mask, ~np.isnan(trials[f'contrast{side}']),
                trials['choice'] == 1])])
                
            trn.append(
                np.arange(len(trials['stimOn_times']))[
                    np.bitwise_and.reduce([
                        mask, ~np.isnan(trials[f'contrast{side}']),
                        trials['choice'] == 1])])
                
    elif split == 'stim_cr':  # restrict to right choice only
        for side in ['Left', 'Right']:
            events.append(trials['stimOn_times'][np.bitwise_and.reduce(
                [mask, ~np.isnan(trials[f'contrast{side}']),
                trials['choice'] == -1])])
                
            trn.append(
                np.arange(len(trials['stimOn_times']))[
                    np.bitwise_and.reduce([
                        mask, ~np.isnan(trials[f'contrast{side}']),
                        trials['choice'] == -1])])

    elif split == 'choice_sl':  
        for choice in [1, -1]:            
            events.append(trials['firstMovement_times'][
                    np.bitwise_and.reduce(
                        [mask, trials['choice'] == choice,
                        ~np.isnan(trials[f'contrastRight'])])])
                        
            trn.append(
                    np.arange(len(trials['choice']))
                    [np.bitwise_and.reduce(
                    [mask, trials['choice'] == choice,
                    ~np.isnan(trials[f'contrastRight'])])])
                
    elif split == 'choice_sr':  
        for choice in [1, -1]:            
            events.append(trials['firstMovement_times'][
                    np.bitwise_and.reduce(
                        [mask, trials['choice'] == choice,
                        ~np.isnan(trials[f'contrastLeft'])])])
                        
            trn.append(
                    np.arange(len(trials['choice']))
                    [np.bitwise_and.reduce(
                    [mask, trials['choice'] == choice,
                    ~np.isnan(trials[f'contrastLeft'])])])


    elif split == 'fback':
        for fb in [1, -1]:
            events.append(
                trials['feedback_times'][np.bitwise_and.reduce([
                    mask, trials['feedbackType'] == fb])])
            trn.append(
                np.arange(len(trials['choice']))[
                    np.bitwise_and.reduce([
                        mask, trials['feedbackType'] == fb])])

    elif split == 'block':
        for pleft in [0.8, 0.2]:
            events.append(
                trials['stimOn_times'][
                    np.bitwise_and.reduce([
                        mask,
                        trials['probabilityLeft'] == pleft])])
            trn.append(np.arange(len(trials['choice']))[
                np.bitwise_and.reduce([
                    mask,
                    trials['probabilityLeft'] == pleft])])

    else:
        print('what is the split?', split)
        return

    print('#trials per condition: ', len(trn[0]), len(trn[1]))
    assert (len(trn[0]) != 0) and (len(trn[1]) != 0), 'zero trials to average'

    assert len(
        spikes['times']) == len(
        spikes['clusters']), 'spikes != clusters'

    # bin and cut into trials
    bins = []
    for event in events:

        #  overlapping time bins, bin size = b_size, stride = sts
        bis = []
        st = int(b_size // sts)

        for ts in range(st):

            bi, _ = bin_spikes2D(
                spikes['times'],
                clusters['cluster_id'][spikes['clusters']],
                clusters['cluster_id'],
                np.array(event) + ts * sts,
                pre_post(split)[0], pre_post(split)[1],
                b_size)
            bis.append(bi)

        ntr, nn, nbin = bi.shape
        ar = np.zeros((ntr, nn, st * nbin))

        for ts in range(st):
            ar[:, :, ts::st] = bis[ts]

        bins.append(ar)

    b = np.concatenate(bins)

    # recreate temporal trial order
    dx = np.concatenate([list(zip([True] * len(trn[0]), trn[0])),
                         list(zip([False] * len(trn[1]), trn[1]))])

    b = b[np.argsort(dx[:, 1])]

    ntr, nclus, nbins = b.shape

    wsc = np.concatenate(b, axis=1)  # all trials, all bins

    acs = br.id2acronym(clusters['atlas_id'], mapping=mapping)
    acs = np.array(acs)

    if get_fr:
        # return firing rates per cell
        
        dr = {'pid':pid,
              'eid':eid,
              'probe':probe,
              'cluster_ids':clusters['cluster_id'].values, 
              'f_rates': np.mean(wsc,axis=1)}
        return dr


    if restr:
        # restrict to canonical list of cells
        uu = units_df[units_df['pid'] == pid]
        css = uu['cluster_id'].values

        inv_map = {v: k for k, v in
                   clusters['cluster_id'].to_dict().items()}

        # map cluster id to index
        goodcells = [inv_map[cell_id] for cell_id in css]

        # restrict data
        acs = acs[goodcells]
        b = b[:, goodcells, :]
        bins2 = [x[:, goodcells, :] for x in bins]
        bins = bins2

    # discard ill-defined regions
    goodcells = ~np.bitwise_or.reduce([acs == reg for
                                       reg in ['void', 'root']])

    acs = acs[goodcells]

    bins2 = [x[:, goodcells, :] for x in bins]
    bins = bins2
    b = np.concatenate(bins)
    wsc = np.concatenate(b, axis=1)

    # Discard cells with any nan or 0 for all bins
    goodcells = [k for k in range(wsc.shape[0]) if
                 (not np.isnan(wsc[k]).any()
                 and wsc[k].any())]

    acs = acs[goodcells]
    b = b[:, goodcells, :]

    bins2 = [x[:, goodcells, :] for x in bins]
    bins = bins2

    if control:
        # get mean and var across trials
        w0 = [bi.mean(axis=0) for bi in bins]
        s0 = [bi.var(axis=0) for bi in bins]

        perms = []  # keep track of random trial splits to test sig

        # nrand times random impostor/pseudo split of trials
        for i in range(nrand):
            if split == 'block':  # 'block' pseudo sessions
                ys = generate_pseudo_blocks(ntr, first5050=0) == 0.8

            elif 'stim' in split:
                # shuffle stim sides within block and choice classes

                # get real block labels
                y_ = trials['probabilityLeft'][
                    sorted(dx[:, 1])].values

                # get real choices
                stis = trials['choice'][
                    sorted(dx[:, 1])].values

                # block/choice classes
                c0 = np.bitwise_and(y_ == 0.8, stis == 1)
                c1 = np.bitwise_and(y_ != 0.8, stis == 1)
                c2 = np.bitwise_and(y_ == 0.8, stis != 1)
                c3 = np.bitwise_and(y_ != 0.8, stis != 1)

                tr_c = dx[np.argsort(dx[:, 1])][:, 0]  # true stim sides
                tr_c2 = deepcopy(tr_c)

                # shuffle stim sides within each class
                for cc in [c0, c1, c2, c3]:
                    r = tr_c[cc]
                    tr_c2[cc] = np.array(random.sample(list(r), len(r)))

                ys = tr_c2 == 1  # boolean shuffled stim sides

            elif (('choice' in split) or ('fback' in split)):
                # shuffle choice sides within block and stim classes

                # get real block labels
                y_ = trials['probabilityLeft'][
                    sorted(dx[:, 1])].values
                # get real stim sides
                stis = trials['contrastLeft'][
                    sorted(dx[:, 1])].values

                # block/stim classes
                c0 = np.bitwise_and(y_ == 0.8, np.isnan(stis))
                c1 = np.bitwise_and(y_ != 0.8, np.isnan(stis))
                c2 = np.bitwise_and(y_ == 0.8, ~np.isnan(stis))
                c3 = np.bitwise_and(y_ != 0.8, ~np.isnan(stis))

                tr_c = dx[np.argsort(dx[:, 1])][:, 0]  # true choices
                tr_c2 = deepcopy(tr_c)

                # shuffle choices within each class
                for cc in [c0, c1, c2, c3]:
                    r = tr_c[cc]
                    tr_c2[cc] = np.array(random.sample(list(r), len(r)))

                ys = tr_c2 == 1  # boolean shuffled choices

                if split == 'fback':
                    # get feedback types from shuffled choices
                    cl = np.bitwise_and(tr_c == 0, np.isnan(stis))
                    cr = np.bitwise_and(tr_c == 1, ~np.isnan(stis))
                    ys = np.bitwise_or(cl, cr)

            w0.append(b[ys].mean(axis=0))
            s0.append(b[ys].var(axis=0))

            w0.append(b[~ys].mean(axis=0))
            s0.append(b[~ys].var(axis=0))

            perms.append(ys)

    else:  # average trials per condition
        print('all trials')
        w0 = [bi.mean(axis=0) for bi in bins]
        s0 = [bi.var(axis=0) for bi in bins]

    ws = np.array(w0)
    ss = np.array(s0)

    regs = Counter(acs)

    D_ = {}
    D_['acs'] = acs
    D_['ws'] = ws[:ntravis]

    if not control:
        return D_

    #  Sum together cells in same region to save memory
    D = {}

    for reg in regs:

        res = {}

        ws_ = [y[acs == reg] for y in ws]
        ss_ = [y[acs == reg] for y in ss]

        # keep track of neuron numbers for averaging later
        res['nclus'] = sum(acs == reg)

        d_vars = []
        d_eucs = []

        # first two trajectories are real, other pseudo
        for j in range(len(ws_) // 2):

            # strictly standardized mean difference
            d_var = (((ws_[2 * j] - ws_[2 * j + 1]) /
                      ((ss_[2 * j] + ss_[2 * j + 1])**0.5))**2)

            # Euclidean distance
            d_euc = (ws_[2 * j] - ws_[2 * j + 1])**2

            # sum over cells, divide by #neu later
            d_var_m = np.nansum(d_var, axis=0)
            d_euc_m = np.sum(d_euc, axis=0)

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


def get_all_d_vars(split, eids_plus=None, control=True, restr=False,
                   mapping='Beryl', contr=None, get_fr=False, nrand=1000):
    '''
    for all BWM insertions, get the PSTHs and acronyms,
    i.e. run get_d_vars
    '''

    time00 = time.perf_counter()

    if get_fr:
        print('only computing firing rates')
    else:    
        print('split: ', split)
        print('control: ', control)
        print('contr: ', contr)
        print('restr: ', restr)

    if eids_plus is None:
        df = bwm_query(one)
        eids_plus = df[['eid', 'probe_name', 'pid']].values

    # save results per insertion (eid_probe) in FlatIron folder
    if (split == 'fback' and contr is not None):
        ps = f'{split}_{contr}'
    elif restr:
        ps = f'{split}_restr'
    else:
        ps = split

    pth = Path(one.cache_dir, 'manifold', f'{ps}_fr' if get_fr else ps)

    pth.mkdir(parents=True, exist_ok=True)

    Fs = []
    k = 0
    print(f'Processing {len(eids_plus)} insertions')
    for i in eids_plus:
        eid, probe, pid = i

        time0 = time.perf_counter()
        try:
            D_ = get_d_vars(split, pid, control=control, restr=restr,
                            mapping=mapping, contr=contr,
                            get_fr=get_fr, nrand=nrand)
                            
            if get_fr:
                eid_probe = eid + '_' + probe
                np.save(Path(pth, f'{eid_probe}_fr.npy'), D_, 
                        allow_pickle=True)
                                
            else:
                eid_probe = eid + '_' + probe
                np.save(Path(pth, f'{eid_probe}.npy'), D_, 
                        allow_pickle=True)

            gc.collect()
            print(k + 1, 'of', len(eids_plus), 'ok')
        except BaseException:
            Fs.append(pid)
            gc.collect()
            print(k + 1, 'of', len(eids_plus), 'fail', pid)

        time1 = time.perf_counter()
        print(time1 - time0, 'sec')

        k += 1

    time11 = time.perf_counter()
    print((time11 - time00) / 60, f'min for the complete bwm set, {split}')
    print(f'{len(Fs)}, load failures:')
    print(Fs)


def d_var_stacked(split, min_reg=20, uperms_=False, 
                  eids_only=None, get_data=False,
                  fdr=True):

    time0 = time.perf_counter()

    '''
    average d_var_m (d_euc_m) via nanmean across insertions,
    remove regions with too few neurons (min_reg)
    compute maxes, latencies and p-values
    
    eids_only: list of eids to restrict to (for licking)
    '''

    
    def can_restr(split):
        # canonical time bins to restrict amplitudes 
        # (for meta anlysis), assumes original windows
    
        restr_can0 =  {'stim': 48, #[0, 0.1]
                     'choice': 48, #[0.1, 0],
                     'fback': 96, #[0, 0.2],
                     'block': 144} #[0.4, -0.1]}

        if '_' in split:
            return restr_can0[split.split('_')[0]]
        else:
            return restr_can0[split]

    pth = Path(one.cache_dir, 'manifold', split)
    ss = os.listdir(pth)  # get insertions
    print(f'combining {len(ss)} insertions for split {split}')
    
    # pool data for illustrative PCA
    acs = []
    ws = []
    regdv0 = {}
    regde0 = {}
    uperms = {}

    # group results across insertions
    for s in ss:

        if eids_only:
            # skip insertion that are not eids list
            if s.split('_')[0] not in eids_only:
                continue
            
        
        D_ = np.load(Path(pth, s),
                     allow_pickle=True).flat[0]

        acs.append(D_['acs'])
        ws.append(D_['ws'])

        if uperms_:
            uperms[s.split('.')[0]] = D_['uperms']
            continue

        for reg in D_['D']:
            if reg not in regdv0:
                regdv0[reg] = []
            regdv0[reg].append(np.array(D_['D'][reg]['d_vars']) / b_size)
            if reg not in regde0:
                regde0[reg] = []
            regde0[reg].append(np.array(D_['D'][reg]['d_eucs']) / b_size)

    if uperms_:
        return uperms

    acs = np.concatenate(acs)
    ws = np.concatenate(ws, axis=1)

    print('computing grand average metrics ...')
    ntr, ncells, nt = ws.shape

    ga = {}
    ga['m0'] = np.mean(ws[0], axis=0)
    ga['m1'] = np.mean(ws[1], axis=0)
    ga['ms'] = np.mean(ws[2:], axis=(0, 1))

    ga['v0'] = np.std(ws[0], axis=0) / (ncells**0.5)
    ga['v1'] = np.std(ws[1], axis=0) / (ncells**0.5)
    ga['vs'] = np.std(ws[2:], axis=(0, 1)) / (ncells**0.5)

    ga['euc'] = np.mean((ws[0] - ws[1])**2, axis=0)**0.5
    ga['nclus'] = ncells

    pca = PCA(n_components=3)
    wsc = pca.fit_transform(np.concatenate(ws, axis=1).T).T
    ga['pcs'] = wsc

    np.save(Path(pth_res, f'{split}_grand_averages.npy'), ga,
            allow_pickle=True)

    print('computing regional metrics ...')
    
    
    regs0 = Counter(acs)
    regs = {reg: regs0[reg] for reg in regs0 if regs0[reg] > min_reg}

    # nansum across insertions and take sqrt
    regdv = {reg: (np.nansum(regdv0[reg], axis=0) / regs[reg])**0.5
             for reg in regs}
    regde = {reg: (np.nansum(regde0[reg], axis=0) / regs[reg])**0.5
             for reg in regs}

    if get_data:
        return regde

    r = {}
    for reg in regs:
        res = {}

        # get PCA for 3d trajectories
        dat = ws[:, acs == reg, :]
        res['ws'] = dat[:2, :, :]

        pca = PCA(n_components=3)
        wsc = pca.fit_transform(np.concatenate(dat, axis=1).T).T

        res['pcs'] = wsc
        res['nclus'] = regs[reg]

        '''
        var
        '''
        # amplitudes
        ampsv = [np.max(x) - np.min(x) for x in regdv[reg]]

        # p value
        res['p_var'] = np.mean(np.array(ampsv) >= ampsv[0])

        # full curve
        res['d_var'] = regdv[reg][0] - min(regdv[reg][0])
        res['amp_var'] = max(res['d_var'])

        # latency
        if np.max(res['d_var']) == np.inf:
            loc = np.where(res['d_var'] == np.inf)[0]
        else:
            loc = np.where(res['d_var'] > 0.7 * (np.max(res['d_var'])))[0]

        res['lat_var'] = np.linspace(-pre_post(split)[0],
                                     pre_post(split)[1],
                                     len(res['d_var']))[loc[0]]
                                     
        # same for null_d normalised curve
        res['p_varn'] = res['p_var']                              
        res['d_varn'] = regdv[reg][0] - np.mean(regdv[reg][1:], axis=0)
        res['d_varn'] = res['d_varn'] - min(res['d_varn'])
        res['amp_varn'] = max(res['d_varn'])
        
        # latency - exceptions for inf/nan     
        if np.max(res['d_varn']) == np.inf:
            loc = np.where(res['d_varn'] == np.inf)[0]
            print('inf d_varn', reg)
        elif np.isnan(np.max(res['d_varn'])):
            print('nan d_varn', reg)
            loc = [-1]
        else:            
            loc = np.where(res['d_varn'] > 0.7 * (np.max(res['d_varn'])))[0]
                           
        res['lat_varn'] = np.linspace(-pre_post(split)[0],
                                     pre_post(split)[1],
                                     len(res['d_varn']))[loc[0]]

        '''
        euc
        '''
        
        # baseline subtract from each curve
        v = np.min(regde[reg],axis=1)
        M = regde[reg] - np.tile(v,(regde[reg].shape[1],1)).T
        
        # p value per time point
        ps = np.mean((M >=  M[0]), axis=0)
        
        # amplitudes (max modulations)
        ampse = np.max(M,axis=1) - np.min(M,axis=1)

        # p value per max modulation
        res['p_euc'] = np.mean(np.array(ampse) >= ampse[0])

        # full curve
        res['d_euc'] = M[0]

        # pseudo curves
        res['d_euc_p'] = M[1:ntravis]

        # amplitude
        res['amp_euc'] = max(M[0])

        # latency, must be significant
        ups = np.where(M[0] > 0.7 * (np.max(M[0])))[0]
        if not any(ps[ups]<sigl):
            # none are significant
            res['lat_euc'] = np.nan
            
        else:
            loc = ups[ps[ups]<sigl]
            res['lat_euc'] = np.linspace(-pre_post(split)[0],
                                         pre_post(split)[1],
                                         len(res['d_euc']))[loc[0]]
                                    
        # same for null_d normalised curve
        res['p_eucn'] = res['p_euc']  # NOTE THIS                             
        res['d_eucn'] = regde[reg][0] - np.mean(regde[reg][1:], axis=0)
        res['d_eucn'] = res['d_eucn'] - min(res['d_eucn'])
        res['amp_eucn'] = max(res['d_eucn'])
        
        # latency
        loc = np.where(res['d_eucn'] > 0.7 * (np.max(res['d_eucn'])))[0]
        res['lat_eucn'] = np.linspace(-pre_post(split)[0],
                                     pre_post(split)[1],
                                     len(res['d_eucn']))[loc[0]]
        
        
        # canonical time window restricted versions
        res['d_euc_can'] = res['d_euc'][:can_restr(split)]
        res['amp_euc_can'] = max(res['d_euc_can'])
        
        # p value per point for latency
        M = M[:,:can_restr(split)]
        ps = np.mean((M >=  M[0]), axis=0) 

        # p value for max modulation across time
        ampse = np.max(M,axis=1) - np.min(M,axis=1)

        # p value per max modulation
        res['p_euc_can'] = np.mean(np.array(ampse) >= ampse[0])
        
        # latency, must be significant
        ups = np.where(M[0] > 0.7 * (np.max(M[0])))[0]
        if not any(ps[ups]<sigl):
            # none are significant
            res['lat_euc_can'] = np.nan
            
        else:
            loc = ups[ps[ups]<sigl]
            res['lat_euc_can'] = np.linspace(-pre_post(split)[0],
                                         pre_post(split)[1],
                                         len(res['d_euc_can']))[loc[0]]        

               
        # subtractively normalized metrics
        res['d_eucn_can'] = res['d_eucn'][:can_restr(split)]
        res['amp_eucn_can'] = max(res['d_eucn_can'])
        
        # latency
        loc = np.where(res['d_eucn_can'] > 0.7 * res['amp_eucn_can'])[0]
        res['lat_eucn_can'] = np.linspace(-pre_post(split,can=True)[0],
                                     pre_post(split,can=True)[1],
                                     len(res['d_eucn_can']))[loc[0]]         
        
        
        r[reg] = res

    if fdr:
        print(f'correcting for multiple comparisons at {sigl}')
        
        ptypes = [x for x in r[list(r.keys())[0]] if x[:2] == 'p_']
        for ptype in ptypes:
            regs = [x for x in r]
            pvals = [r[x][ptype] for x in r]
            _, pvals_c, _, _ = multipletests(pvals, sigl, 
                                             method='fdr_bh')
            for i in range(len(regs)):
                r[regs[i]][ptype] = pvals_c[i]

    
    if eids_only:
        split = split+'_eids_only' 

    np.save(Path(pth_res, f'{split}.npy'),
            r, allow_pickle=True)

    time1 = time.perf_counter()
    print('total time:', np.round(time1 - time0, 0), 'sec')


def curves_params_all(split):

    get_all_d_vars(split)
    d_var_stacked(split)


'''
#####################################################
### plotting
#####################################################
'''


def get_allen_info():
    '''
    Function to load Allen atlas info, like region colors
    '''

    p = (Path(ibllib.__file__).parent /
         'atlas/allen_structure_tree.csv')

    dfa = pd.read_csv(p)

    # replace yellow by brown #767a3a    
    cosmos = []
    cht = []
    
    for i in range(len(dfa)):
        try:
            ind = dfa.iloc[i]['structure_id_path'].split('/')[4]
            cr = br.id2acronym(ind, mapping='Cosmos')[0]
            cosmos.append(cr)
            if cr == 'CB':
                cht.append('767A3A')
            else:
                cht.append(dfa.iloc[i]['color_hex_triplet'])    
                    
        except:
            cosmos.append('void')
            cht.append('FFFFFF')
            

    dfa['Cosmos'] = cosmos
    dfa['color_hex_triplet2'] = cht
    
    # get colors per acronym and transfomr into RGB
    dfa['color_hex_triplet2'] = dfa['color_hex_triplet2'].fillna('FFFFFF')
    dfa['color_hex_triplet2'] = dfa['color_hex_triplet2'
                                   ].replace('19399', '19399a')
    dfa['color_hex_triplet2'] = dfa['color_hex_triplet2'].replace(
                                                     '0', 'FFFFFF')
    dfa['color_hex_triplet2'] = '#' + dfa['color_hex_triplet2'].astype(str)
    dfa['color_hex_triplet2'] = dfa['color_hex_triplet2'
                                   ].apply(lambda x:
                                           mpl.colors.to_rgba(x))

    palette = dict(zip(dfa.acronym, dfa.color_hex_triplet2))

    return dfa, palette


def put_panel_label(ax, k):
    ax.annotate(string.ascii_lowercase[k], (-0.05, 1.15),
                xycoords='axes fraction',
                fontsize=f_size * 1.5, va='top',
                ha='right', weight='bold')


def plot_all(splits=None, curve='euc', show_tra=True,
             all_labs=False,ga_pcs=True, extra_3d=False):
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
    if splits is None:
        splits = align

    # specify grid; scatter longer than other panels
    ncols = 12
    
    if show_tra:
        fig = plt.figure(figsize=(20, 2.5*len(splits)))
        gs = fig.add_gridspec(len(splits), ncols)
    else:   
        fig = plt.figure(figsize=(4, 6), layout='constrained')
        gs = fig.add_gridspec(2, ncols)
        
    
    axs = []
    k = 0  # panel counter


    if not show_tra:
        fsize = 12 # font size
        dsize = 13  # diamond marker size
        lw = 1  # linewidth        

    else:
        fsize = 12  # font size
        dsize = 13  # diamond marker size
        lw = 1  # linewidth       

    dfa, palette = get_allen_info()

    '''
    get significant regions
    '''
    tops = {}
    regsa = []

    for split in splits:

        d = np.load(Path(pth_res, f'{split}.npy'),
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
        tops[split + '_s'] = (f'{len(maxsf)}/{len(d)} = '
                             f'{np.round(len(maxsf)/len(d),2)}')
                             
                             
        regs_a = [tops[split][0][j] for j in range(len(tops[split][0]))
                  if tops[split][1][j] < sigl]

        regsa.append(list(d.keys()))
        print(regs_a)
        print(' ')

    for split in splits:
        print(split, tops[split + '_s'])

    #  get Cosmos parent region for yellow color adjustment
    regsa = np.unique(np.concatenate(regsa))
    cosregs_ = [
        dfa[dfa['id'] == int(dfa[dfa['acronym'] == reg][
            'structure_id_path'].values[0].split('/')[4])][
            'acronym'].values[0] for reg in regsa]

    cosregs = dict(zip(regsa, cosregs_))

    '''
    example regions per split for embedded space and line plots
    
    first in list is used for pca illustration
    '''
#                              'LP',  'PRNr',
#                     , 'MRN', 'SCm', 'IP', 'IRN'],

                    # 

    exs0 = {'stim': ['LGd','VISp', 'PRNc','VISam','IRN', 'VISl',
                     'VISpm', 'VM', 'MS','VISli'],


            'choice': ['PRNc', 'VISal','PRNr', 'LSr', 'SIM', 'APN',
                       'MRN', 'RT', 'LGd', 'GRN','MV','ORBm'],

            'fback': ['IRN', 'SSp-n', 'PRNr', 'IC', 'MV', 'AUDp',
                      'CENT3', 'SSp-ul', 'GPe'],
            'block': ['Eth', 'IC']}

    # use same example regions for variant splits
    exs = exs0.copy()
    for split in splits:
        for split0 in  exs0:
            if split0 in split:
                exs[split] = exs0[split0]


    if show_tra:

        '''
        Trajectories for example regions in PCA embedded 3d space
        ''' 
            
        row = 0
        for split in splits:

            if ga_pcs:
                dd = np.load(Path(pth_res, f'{split}_grand_averages.npy'),
                            allow_pickle=True).flat[0]
            else:
                d = np.load(Path(pth_res, f'{split}.npy'),
                            allow_pickle=True).flat[0]

                # pick example region
                reg = exs[split][0]
                dd = d[reg]

            if extra_3d:
                axs.append(fig.add_subplot(gs[:,row*3: (row+1)*3],
                                           projection='3d'))        
            else:
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
                axs[k].set_title(split.split('_')[0])    

            else:
                axs[k].set_title(f"{split}, {reg} {d[reg]['nclus']}"
                                 if not ga_pcs else split)
            axs[k].grid(False)
            axs[k].axis('off')

            if not extra_3d:
                put_panel_label(axs[k], k)

            k += 1
            row += 1
            
        if extra_3d:
            return
        
    '''
    line plot per 5 example regions per split
    '''
    row = 0  # index

    for split in splits:

        if show_tra:
            axs.append(fig.add_subplot(gs[row, 3:6]))
        else:
            axs.append(fig.add_subplot(gs[0, :]))


        d = np.load(Path(pth_res, f'{split}.npy'),
                    allow_pickle=True).flat[0]

        # example regions to illustrate line plots
        regs = exs[split]

        texts = []
        for reg in regs:
            if reg not in d:
                print(f'{reg} not in d:'
                       'revise example regions for line plots')
                continue
        
            if any(np.isinf(d[reg][f'd_{curve}'])):
                print(f'inf in {curve} of {reg}')
                continue

            xx = np.linspace(-pre_post(split)[0],
                             pre_post(split)[1],
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
                                     fontsize=fsize))


        axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')

        if split in ['block', 'choice']:
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

    for split in splits:

        if show_tra:
            axs.append(fig.add_subplot(gs[row, 6:]))
        else:
            axs.append(fig.add_subplot(gs[1,:]))    


        d = np.load(Path(pth_res, f'{split}.npy'),
                    allow_pickle=True).flat[0]

        acronyms = [tops[split][0][j] for j in range(len(tops[split][0]))]
        ac_sig = np.array([True if tops[split][1][j] < sigl
                           else False for j in range(len(tops[split][0]))])

        maxes = np.array([d[x][f'amp_{curve}'] for x in acronyms])
        lats = np.array([d[x][f'lat_{curve}'] for x in acronyms])
        # stdes = np.array([d[x][f'stde_{curve}'] for x in acronyms])
        cols = [palette[reg] for reg in acronyms]

#        if split == 'stim':
#            fig0, ax0 = plt.subplots()
#            axs[k] = ax0

        # yerr = 100*maxes/d[reg]['nclus']
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
                
                if reg not in exs[split]:
                    if not all_labs: # restrict to example regions   
                        continue
                

                texts.append(
                    axs[k].annotate(
                        '  ' + reg,
                        (lats[i], maxes[i]),
                        fontsize=fsize,
                        color=palette[acronyms[i]],
                        arrowprops=None))
                        

        axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')

        ha = 'left'


        axs[k].text(0, 0.95, align[split.split('_')[0]
                                   if '_' in split else split],
                    transform=axs[k].get_xaxis_transform(),
                    horizontalalignment=ha, rotation=90,
                    fontsize=f_size * 0.8)

        axs[k].spines['top'].set_visible(False)
        axs[k].spines['right'].set_visible(False)

        axs[k].set_ylabel('max dist. [Hz]')
        axs[k].set_xlabel('latency [sec]')
        
        
        
        if show_tra:
            put_panel_label(axs[k], k)
            axs[k].set_title(f"{tops[split+'_s']} sig")


        row += 1
        k += 1

    if not show_tra:
        axs[-1].sharex(axs[-2])
        

#    gs.update(left=0.1,right=0.9,
#              top=0.965,bottom=0.03,
#              wspace=0.3,hspace=0.09)
#              
#    fig = plt.gcf()
#    fig.tight_layout()


def plot_custom_lines(regs=None, curve='euc', split='choice',
                      psd_=False, contr_=False, lickogram=False):
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

    ds = {'MRN': 'f8d5c8b0-b931-4151-b86c-c471e2e80e5d',
         'APN': 'f8d5c8b0-b931-4151-b86c-c471e2e80e5d', #(multi session)
         'SSp-m': 'ffef0311-8ffa-49e3-a857-b3adf6d86e12',
         'SIM': 'c4432264-e1ae-446f-8a07-6280abade813',
         'PRM': '83d85891-bd75-4557-91b4-1cbb5f8bfc9d',
         'PoT': 'db4df448-e449-4a6f-a0e7-288711e7a75a',
         'MEA': 'e535fb62-e245-4a48-b119-88ce62a6fe67',
         'ANcr2': '83d85891-bd75-4557-91b4-1cbb5f8bfc9d'}
         
    licks = np.load('/home/mic/bwm/manifold_analysis/'
                    'fback_eids_only_lickograms.npy', 
                    allow_pickle=True).flat[0]

    if regs is None:

        if 'choice' in split:
            regs = ['APN', 'IRN','GRN','ACAv',
                    'PRNr','LP','AUDp','PO','ILA']
        if 'fback' in split:        
            regs = ['MV', 'MRN', 'APN', 'SSp-m',
             'SIM', 'PRM', 'PoT','MEA', 'ANcr2']
            if split == 'fback_eids_only':
                regs = ['MRN', 'APN', 'SSp-m',         
                        'MEA', 'ANcr2']

    df, palette = get_allen_info()
    

    nr = 3
    
    if split == 'fback_eids_only':
        fig, axs = plt.subplots(nrows=1, 
                                ncols=len(regs),
                                figsize=(12, 2.5),
                                sharex=True, constrained_layout=True)    
    else:
        fig, axs = plt.subplots(nrows=nr, 
                                ncols=int(np.ceil(len(regs) / nr)),
                                figsize=(7, 7),
                                sharex=True, constrained_layout=True)
    axs = axs.flatten()
    axsi = []  # inset axes
    axts = [] # twin axis 


    contrs = ([0., 0.0625, 0.125, 0.25, 1.]
              if contr_ else [0])

    end_pts = {}  # keep end points to quantify contrast variance
    jj = 0
    for contr in contrs:

        print(jj, contr)
        # get results
        d = np.load(Path(pth_res, f'{split}_{contr}.npy'
                    if contr_ else f'{split}.npy'),
                    allow_pickle=True).flat[0]

        # get cosmos parent regions for Swanson acronyms
        cosregs = {reg: df[df['id'] == int(df[df['acronym'] == reg
                                              ]['structure_id_path']
                   .values[0].split('/')[4])]['acronym']
                   .values[0] for reg in regs}

        k = 0
        for reg in regs:

            yy = d[reg][f'd_{curve}']

            xx = np.linspace(-pre_post(split)[0],
                             pre_post(split)[1],
                             len(d[reg][f'd_{curve}']))

            axs[k].plot(xx, yy, linewidth=2, alpha=1
                        if contr == 1. or split != 'choice'
                        else contr + 0.4,
                        color=palette[reg],
                        label=f"{reg}")

            if jj == 0:
                end_pts[reg] = [yy]
            else:
                end_pts[reg].append(yy)

            cos = cosregs[reg]

            if jj == 0:
                # quanitfy contr stratification
                if contr:
                    m = sum(sum(np.diff(np.array(end_pts[reg]), axis=0)))
                    end_pts[reg] = m
                ss = f"{reg} {d[reg]['nclus']}"  # , {np.round(m,4)}"
                axs[k].spines['top'].set_visible(False)
                axs[k].spines['right'].set_visible(False)
                axs[k].set_ylabel('distance')
                axs[k].set_xlabel('time [sec]')
                axs[k].set_title(ss, color=palette[reg])

            if psd_:
                f, psd = signal.welch(yy,
                                      fs=int(len(xx) / (xx[-1] - xx[0])))

                with plt.rc_context({'font.size':
                                     0.8 * plt.rcParams['font.size']}):
                    # plot psd as inset
                    axsi.append(inset_axes(axs[k], width="30%", height="35%",
                                           loc=4 if reg != 'PoT' else 1,
                                           borderpad=1,
                                           bbox_to_anchor=(-0.02, 0.1, 1, 1),
                                           bbox_transform=axs[k].transAxes))

                    axsi[-1].plot(f, psd)
                    axsi[-1].set_xlim(3, 40)
                    axsi[-1].set_xlabel('f [Hz]')
                    Axis.set_label_coords(axsi[-1].xaxis, -0.3, -0.05)

                    axsi[-1].set_ylabel('psd ')
                    axsi[-1].spines['top'].set_visible(False)
                    axsi[-1].spines['right'].set_visible(False)
                    axsi[-1].set_yticks([])
                    axsi[-1].set_yticklabels([])
                    if k > 0:
                        axsi[-1].sharex(axsi[-2])
                        
                        
            if lickogram:
                if ds[reg] in licks:

                    yy = licks[ds[reg]]
                    xx = np.linspace(-pre_post(split)[0],
                                     pre_post(split)[1],
                                     len(yy))
                                     
                    axts.append(axs[k].twinx())                
                    axts[-1].plot(xx,yy)
                    axts[-1].set_ylabel('trial-averaged licks',
                                        color='C0')
                                                

            k += 1

        jj += 1
    #fig.savefig(f'{"_".join(regs)}.png')


def get_cmap(split):
    '''
    for each split, get a colormap defined by Yanliang
    '''
    dc = {'stim': ["#ffffff","#D5E1A0","#A3C968",
                   "#86AF40","#517146"],
          'choice': ["#ffffff","#F8E4AA","#F9D766",
                     "#E8AC22","#DA4727"],
          'fback': ["#ffffff","#F1D3D0","#F5968A",
                    "#E34335","#A23535"],
          'block': ["#ffffff","#D0CDE4","#998DC3",
                    "#6159A6","#42328E"]}

    if '_' in split:
        split = split.split('_')[0]

    return LinearSegmentedColormap.from_list("mycmap", dc[split])
   
    
def plot_swanson_supp(splits = None, curve = 'euc',
                      show_legend = False, bina=False):
 
    '''
    swanson maps for maxes
    '''
    
    if splits is None:
        splits0 = ['stim', 'choice', 'fback','block']
        splits = [x+'_restr' for x in splits0]
    
    nrows = 2  # one for amplitudes, one for latencies
    ncols = len(splits)  # one per variable


    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 11)) 
    
    if show_legend:
        '''
        plot Swanson flatmap with labels and colors
        '''
        fig0, ax0 = plt.subplots()
        plot_swanson_vector(annotate=True, ax=ax0)
        ax0.axis('off')
   
    '''
    max dist_split onto swanson flat maps
    (only regs with p < sigl)
    '''
    
    k = 0  # panel counter
    c = 0  # column counter
    
    sws = []
    for split in splits:

        d = np.load(Path(pth_res, f'{split}.npy'),
                    allow_pickle=True).flat[0]
                    
        # get significant regions only
        acronyms = [reg for reg in d
                if d[reg][f'p_{curve}'] < sigl]
                
        print(split, len(acronyms), 'sig out of', len(d))        

        # plot amplitudes
        if bina:
            # set all significant regions to 1, others zero
            amps = np.array([1 for x in acronyms])
        
        else:
            amps = np.array([d[x][f'amp_{curve}_can'] for x in acronyms])
            
        plot_swanson_vector(np.array(acronyms), np.array(amps), 
                            cmap=get_cmap(split), 
                            ax=axs[0,c], br=br, 
                            orientation='portrait',
                            linewidth=0.1)
                            
        # add colorbar
        clevels = (np.nanmin(amps), np.nanmax(amps))
        norm = mpl.colors.Normalize(vmin=clevels[0], 
                                    vmax=clevels[1])
                                    
        cbar = fig.colorbar(mpl.cm.ScalarMappable(
                                norm=norm, 
                                cmap=get_cmap(split)), 
                                ax=axs[0,c])
    
        cbar.set_label('effect size [spikes/second]')

                            
        axs[0,c].axis('off')
        axs[0,c].set_title(f'{split} \n {len(acronyms)}/{len(d)} sig.')
        put_panel_label(axs[0,c], k)
        k += 1

        # plot latencies (cmap reversed, dark is early)    
        lats = np.array([d[x][f'lat_{curve}'] for x in acronyms]) 
        plot_swanson_vector(np.array(acronyms),np.array(lats), 
                     cmap=get_cmap(split).reversed(), 
                     ax=axs[1,c], br=br, orientation='portrait')

        clevels = (np.nanmin(lats), np.nanmax(lats))
        norm = mpl.colors.Normalize(vmin=clevels[0], 
                                    vmax=clevels[1])
                                    
        cbar = fig.colorbar(mpl.cm.ScalarMappable(
                                norm=norm, 
                                cmap=get_cmap(split).reversed()), 
                                ax=axs[1,c])
        cbar.set_label('latency [second]')


        axs[1,c].axis('off')
        axs[1,c].set_title(f'{split}')
        put_panel_label(axs[1,c], k)
        
        
        #print(split, acronyms, amps, lats)        
        
        k += 1
        c += 1

    fig.tight_layout()
    



def plot_corr(splits=None, curve='euc',
              x='nclus', y='f-rate'):

    # supp figure for correlation of nclus and maxes
    if splits is None:
        splits = align    
    dfa, palette = get_allen_info()    
   
      # to make yellow labels black  

    
    fig, axs = plt.subplots(ncols=len(splits), nrows=1)
    fig.suptitle(f'distance metric: {curve}')    

    row = 0
    for split in splits:
    
        d = np.load(Path(pth_res, f'{split}.npy'),
                         allow_pickle=True).flat[0]

        cosregs_ = [dfa[dfa['id'] == 
                    int(dfa[dfa['acronym']==reg]['structure_id_path']
                    .values[0].split('/')[4])]['acronym']
                    .values[0] for reg in d]
        cosregs = dict(zip(list(d.keys()),cosregs_))
    
        ll = list(zip([d[reg]['nclus'] for reg in d], 
                      np.array([d[x][f'amp_{curve}'] for x in d]),
                      np.array([palette[reg] for reg in d]),
                      [d[reg][f'p_{curve}'] for reg in d],
                      [np.mean(d[reg]['ws']) for reg in d],
                      list(d.keys())))
        

        df = pd.DataFrame(ll, columns=['nclus', 
                                       'max dist', 
                                       'col',
                                       'p-value',
                                       'f-rate',
                                       'reg'])

        # correlate results
        co0,p0 = spearmanr(df[x],df[y])
        co_sig0,p_sig0 = spearmanr(df[df['p-value'] < sigl][x],
                                   df[df['p-value'] < sigl][y])
        
        co, p, co_sig, p_sig = [np.round(x,2) for x 
                                in [co0, p0, co_sig0, p_sig0]]


        axs[row].scatter(df[x],
                         df[y],
                         c=df['col'], s = 5, 
                        marker = '.', label = f'p > {sigl}')
    
    
        axs[row].scatter(df[df['p-value'] < sigl][x],
                         df[df['p-value'] < sigl][y],
                         c=df[df['p-value'] < sigl]['col'], s = 10, 
                        marker = 'o', label = f'p < {sigl}')
               

        axs[row].set_title(f'{split} \n'
             f'n_regs_all/sig: {len(df)}/{sum(df["p-value"] < sigl)} \n'
             f' corr_all [p] = {co} [{p}], \n'
             f'corr_sig [p_sig]= {co_sig} [{p_sig}]')
                     
        axs[row].set_xlabel(x)
        axs[row].set_ylabel(y) 
        axs[row].legend()
      
      
        for i in range(len(df)):

            axs[row].annotate('  ' + df.iloc[i]['reg'], 
                (df.iloc[i][x], df.iloc[i][y]),
                fontsize=5,color=palette[df.iloc[i]['reg']])   

        row+=1

    fig.tight_layout()


def plot_traj_and_dist(split, reg, ga_pcs=False, curve='euc'):

    '''
    for a given region, plot 3d trajectory and 
    line plot below
    '''
    df, palette = get_allen_info()
    
    # get cosmos parent regions for Swanson acronyms
    cosregs = {reg: df[df['id'] == int(df[df['acronym'] == reg
               ]['structure_id_path']
               .values[0].split('/')[4])]['acronym']
               .values[0] for reg in [reg]}
           
    fig = plt.figure(figsize=(3,3.79))
    axs = []
    gs = fig.add_gridspec(5, 1) 
    k = 0 
      
    # 3d trajectory plot
    if ga_pcs:
        dd = np.load(Path(pth_res, f'{split}_grand_averages.npy'),
                    allow_pickle=True).flat[0]
    else:
        d = np.load(Path(pth_res, f'{split}.npy'),
                    allow_pickle=True).flat[0]

        # pick example region
        dd = d[reg] 

    axs.append(fig.add_subplot(gs[:4, 0],
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

    axs[k].set_title(f"{split}, {reg} {d[reg]['nclus']}"
                     if not ga_pcs else split)
    axs[k].grid(False)
    axs[k].axis('off')

    #put_panel_label(axs[k], k)

    k += 1
    d = np.load(Path(pth_res, f'{split}.npy'),
                allow_pickle=True).flat[0]

    # line plot 
    axs.append(fig.add_subplot(gs[4:,0]))    
    
    if reg not in d:
        print(f'{reg} not in d:'
               'revise example regions for line plots')
        return

    if any(np.isinf(d[reg][f'd_{curve}'])):
        print(f'inf in {curve} of {reg}')
        return
        
    print(split, reg, 'p_euc: ', d[reg]['p_euc'])    

    xx = np.linspace(-pre_post(split)[0],
                     pre_post(split)[1],
                     len(d[reg][f'd_{curve}']))

    # plot pseudo curves
    yy_p = d[reg][f'd_{curve}_p']

    for c in yy_p:
        axs[k].plot(xx, c, linewidth=1,
                    color='Gray')

    # get curve
    yy = d[reg][f'd_{curve}']

    axs[k].plot(xx, yy, linewidth=2,
                color=palette[reg],
                label=f"{reg} {d[reg]['nclus']}")

    # put region labels
    y = yy[-1]
    x = xx[-1]
    ss = reg

    axs[k].text(x, y, ss, color=palette[reg], fontsize=8)
    axs[k].text(x, c[-1], 'control', color='Gray', fontsize=8)

    axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')

    if split in ['block', 'choice']:
        ha = 'left'
    else:
        ha = 'right'

    axs[k].spines['top'].set_visible(False)
    axs[k].spines['right'].set_visible(False)

    axs[k].set_ylabel('distance [Hz]')
    axs[k].set_xlabel('time [sec]')

    #put_panel_label(axs[k], k)

    fig.tight_layout() 
    fig.tight_layout()
    

def combine_left_right():

    '''
    Combine stim_lc and stim_rc by averaging;
    (same for choice_sl, choice_sr)
    '''
    
    for split in ['stim_c','choice_s']:

        dd_ = []
        ds = []
  
        for side in ['l','r']:           
    
            dd_.append(np.load(Path(pth_res, 
                       f'{split + side +"_restr"}_grand_averages.npy'),
                       allow_pickle=True).flat[0]['pcs'])
                    
            ds.append(np.load(Path(pth_res, 
                      f'{split  + side +"_restr"}.npy'),
                      allow_pickle=True).flat[0])                     
                    
                    
        np.save(Path(pth_res, f'{split[:-1]}mean_restr_grand_averages.npy'),
                {'pcs':np.mean(np.array(dd_),axis=0)}, allow_pickle=True)
                            
        # get instersection of regions
        regs = list(set(ds[0].keys()).intersection(set(ds[1].keys())))   

        # average keys (amplitudes, p-values, etc)
        d = {}
        for reg in regs:
            sd = {}
            for key in ds[0][reg]:
                v = []
                for i in range(2):
                    v.append(ds[i][reg][key])
                try:   
                    sd[key] = np.mean(np.array(v),axis=0)       
                except:
                    print(reg, key)  
            d[reg] = sd
                
        np.save(Path(pth_res, 
                f'{split[:-1]}mean_restr.npy'),
                d, allow_pickle=True)        
        
 

    
'''
fback licking example regions:     
    
regs = ['MV', 'MRN', 'APN', 'SSp-m',
 'SIM', 'PRM', 'PoT','MEA', 'ANcr2']     
    
    
eids_only = ['8db36de1-8f17-4446-b527-b5d91909b45a', 
        'f8d5c8b0-b931-4151-b86c-c471e2e80e5d', 
        'f8d5c8b0-b931-4151-b86c-c471e2e80e5d', 
        'ffef0311-8ffa-49e3-a857-b3adf6d86e12', 
        'c4432264-e1ae-446f-8a07-6280abade813', 
        '83d85891-bd75-4557-91b4-1cbb5f8bfc9d', 
        'db4df448-e449-4a6f-a0e7-288711e7a75a', 
        'e535fb62-e245-4a48-b119-88ce62a6fe67', 
        '83d85891-bd75-4557-91b4-1cbb5f8bfc9d']    
    
    
{'MV': '8db36de1-8f17-4446-b527-b5d91909b45a',
 'MRN': 'f8d5c8b0-b931-4151-b86c-c471e2e80e5d',
 'APN': 'f8d5c8b0-b931-4151-b86c-c471e2e80e5d', #(multi session)
 'SSp-m': 'ffef0311-8ffa-49e3-a857-b3adf6d86e12',
 'SIM': 'c4432264-e1ae-446f-8a07-6280abade813',
 'PRM': '83d85891-bd75-4557-91b4-1cbb5f8bfc9d',
 'PoT': 'db4df448-e449-4a6f-a0e7-288711e7a75a',
 'MEA': 'e535fb62-e245-4a48-b119-88ce62a6fe67',
 'ANcr2': '83d85891-bd75-4557-91b4-1cbb5f8bfc9d'}
    
{'8db36de1-8f17-4446-b527-b5d91909b45a': Counter({'CUL4 5': 23, 'MV': 22}),
 'f8d5c8b0-b931-4151-b86c-c471e2e80e5d': Counter({'APN': 56,
          'MD': 94,
          'MRN': 13,
          'PeF': 13,
          'VM': 11}),
 'ffef0311-8ffa-49e3-a857-b3adf6d86e12': Counter({'CP': 36, 'SSp-m': 15}),
 'c4432264-e1ae-446f-8a07-6280abade813': Counter({'APN': 20, 'SIM': 20}),
 '83d85891-bd75-4557-91b4-1cbb5f8bfc9d': Counter({'ANcr1': 41,
          'ANcr2': 88,
          'PRM': 43}),
 'db4df448-e449-4a6f-a0e7-288711e7a75a': Counter({'CA1': 13,
          'LP': 17,
          'PoT': 14,
          'SGN': 11}),
 'e535fb62-e245-4a48-b119-88ce62a6fe67': Counter({'APN': 21, 'MEA': 10})}

    
'APN' (multi)
        
'''
    
    
    
