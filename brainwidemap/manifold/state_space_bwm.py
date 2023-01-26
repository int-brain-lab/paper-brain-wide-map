from one.api import ONE
from reproducible_ephys_processing import bin_spikes2D
from brainwidemap import bwm_query, load_good_units
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
from brainbox.io.one import SessionLoader
import ibllib

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

from matplotlib.axis import Axis
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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


def pre_post(split):
    '''
    [pre_time, post_time] relative to alignment event
    split could be contr or restr variant, then
    use base window
    '''

    pre_post0 = {'stim': [0, 0.15],
                 'choice': [0.15, 0],
                 'fback': [0, 0.7],
                 'block': [0.4, -0.1]}

    if '_' in split:
        return pre_post0[split.split('_')[0]]
    else:
        return pre_post0[split]


one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)
ba = AllenAtlas()
br = BrainRegions()

# save results here
pth_res = Path(one.cache_dir, 'manifold', 'res')
pth_res.mkdir(parents=True, exist_ok=True)


def grad(c, nobs, fr=1):
    '''
    color gradient for plotting trajectories
    c: color map type
    nobs: number of observations
    '''

    cmap = mpl.cm.get_cmap(c)

    return [cmap(fr * (nobs - p) / nobs) for p in range(nobs)]


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(
            value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


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


def get_restricted_cells(split, pid, sig_lev=0.05, alys='decoding'):
    '''
    for a given insertion (pid) and a given split
    get cell ids that are significant for single-cell
    analysis (alys = 'single-cell'),
    or weights per cluster from Brandon (alys = 'decoding')

    Files converted via meta_bwm.py
    '''

    if alys == 'single-cell':
        s1 = pd.read_csv('/home/mic/paper-brain-wide-map/'
                         f'meta/per_eid/single-cell/{split}.csv')

        # load sig cells, then feed as restr
        u = s1[np.bitwise_and(s1['pid'] == pid,
                              s1[f'p_value_{split}'] < sig_lev)]

        return u['cluster_id'].values

    elif alys == 'decoding':

        s1 = pd.read_csv('/home/mic/paper-brain-wide-map/meta/'
                         f'per_cell/decoding/{split}_weights.csv')

        thresh = np.percentile(s1['abs_weight'], 70)

        # load sig cells, then feed as restr
        u = s1[np.bitwise_and(s1['pid'] == pid,
                              s1['abs_weight'] < thresh)]

        return u['cluster_id'].values


def get_d_vars(split, pid, mapping='Beryl', control=True,
               nrand=1000, contr=None, restr=False, shuf=False):
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
    shuf: if True, load random sample of cells

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

    if split == 'choice':
        for choice in [1, -1]:

            if contr is None:  # include any contrast
                events.append(
                    trials['firstMovement_times'][
                        np.bitwise_and.reduce(
                            [~rm_trials, trials['choice'] == choice])])
                trn.append(
                    np.arange(len(trials['choice']))[np.bitwise_and.reduce(
                        [~rm_trials, trials['choice'] == choice])])

            else:  # include only trials with given contrast
                events.append(
                    trials['firstMovement_times'][
                        np.bitwise_and.reduce([
                            ~rm_trials, trials['choice'] == choice,
                            np.bitwise_or(
                                trials['contrastLeft'] == contr,
                                trials['contrastRight'] == contr)])])
                trn.append(
                    np.arange(len(trials['choice']))[
                        np.bitwise_and.reduce([
                            ~rm_trials, trials['choice'] == choice,
                            np.bitwise_or(
                                trials['contrastLeft'] == contr,
                                trials['contrastRight'] == contr)])])

    elif split == 'stim':
        for side in ['Left', 'Right']:
            events.append(trials['stimOn_times'][np.bitwise_and.reduce(
                [~rm_trials, ~np.isnan(trials[f'contrast{side}'])])])
            trn.append(
                np.arange(len(trials['stimOn_times']))[
                    np.bitwise_and.reduce([
                        ~rm_trials, ~np.isnan(trials[f'contrast{side}'])])])

    elif split == 'fback':
        for fb in [1, -1]:
            events.append(
                trials['feedback_times'][np.bitwise_and.reduce([
                    ~rm_trials, trials['feedbackType'] == fb])])
            trn.append(
                np.arange(len(trials['choice']))[
                    np.bitwise_and.reduce([
                        ~rm_trials, trials['feedbackType'] == fb])])

    elif split == 'block':
        for pleft in [0.8, 0.2]:
            events.append(
                trials['stimOn_times'][
                    np.bitwise_and.reduce([
                        ~rm_trials,
                        trials['probabilityLeft'] == pleft])])
            trn.append(np.arange(len(trials['choice']))[
                np.bitwise_and.reduce([
                    ~rm_trials,
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

    wsc = np.concatenate(b, axis=1)

    acs = br.id2acronym(clusters['atlas_id'], mapping=mapping)
    acs = np.array(acs)

    if restr:
        # restrict to cells given in restr

        css = get_restricted_cells(split, pid)

        inv_map = {v: k for k, v in
                   clusters['cluster_id'].to_dict().items()}

        # shuf control
        if shuf:
            print('restr: RANDOM CONTROL')
            goodcells = random.sample(range(len(clusters['cluster_id'])),
                                      len(css))

        else:
            # map to index
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

            elif split == 'stim':
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

            elif split in ['choice', 'fback']:
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
                   mapping='Beryl', contr=None, shuf=False):
    '''
    for all BWM insertions, get the PSTHs and acronyms,
    i.e. run get_d_vars
    '''

    time00 = time.perf_counter()

    print('split', split, 'control', control,
          'contr', contr, 'restr', restr)

    if eids_plus is None:
        df = bwm_query(one)
        eids_plus = df[['eid', 'probe_name', 'pid']].values

    # save results per insertion (eid_probe) in FlatIron folder
    if (split == 'fback' and contr is not None):
        ps = f'{split}_{contr}'
    elif restr:
        ps = f'{split}_restr_shuf' if shuf else f'{split}_restr'
    else:
        ps = split

    pth = Path(one.cache_dir, 'manifold', ps)

    pth.mkdir(parents=True, exist_ok=True)

    Fs = []
    k = 0
    print(f'Processing {len(eids_plus)} insertions')
    for i in eids_plus:
        eid, probe, pid = i

        time0 = time.perf_counter()
        try:
            D_ = get_d_vars(split, pid, control=control, restr=restr,
                            mapping=mapping, contr=contr, shuf=shuf)

            eid_probe = eid + '_' + probe

            np.save(Path(pth, f'{eid_probe}.npy'), D_, allow_pickle=True)

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


def d_var_stacked(split, min_reg=100, uperms_=False):

    time0 = time.perf_counter()

    '''
    average d_var_m (d_euc_m) via nanmean across insertions,
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
        '''
        euc
        '''
        # amplitudes
        ampse = [np.max(x) - np.min(x) for x in regde[reg]]

        # p value
        res['p_euc'] = np.mean(np.array(ampse) >= ampse[0])

        # full curve
        res['d_euc'] = regde[reg][0] - min(regde[reg][0])
        res['mean_rand'] = np.mean(regde[reg][1:], axis=0)
        res['amp_euc'] = max(res['d_euc'])

        # latency
        loc = np.where(res['d_euc'] > 0.7 * (np.max(res['d_euc'])))[0]

        res['lat_euc'] = np.linspace(-pre_post(split)[0],
                                     pre_post(split)[1],
                                     len(res['d_euc']))[loc[0]]

        r[reg] = res

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

    # get colors per acronym and transfomr into RGB
    dfa['color_hex_triplet'] = dfa['color_hex_triplet'].fillna('FFFFFF')
    dfa['color_hex_triplet'] = dfa['color_hex_triplet'
                                   ].replace('19399', '19399a')
    dfa['color_hex_triplet'] = dfa['color_hex_triplet'] .replace('0', 'FFFFFF')
    dfa['color_hex_triplet'] = '#' + dfa['color_hex_triplet'].astype(str)
    dfa['color_hex_triplet'] = dfa['color_hex_triplet'
                                   ].apply(lambda x:
                                           mpl.colors.to_rgba(x))

    palette = dict(zip(dfa.acronym, dfa.color_hex_triplet))

    return dfa, palette


def put_panel_label(ax, k):
    ax.annotate(string.ascii_lowercase[k], (-0.05, 1.15),
                xycoords='axes fraction',
                fontsize=f_size * 1.5, va='top',
                ha='right', weight='bold')


def plot_all(splits=None, curve='euc', amp_number=False,
             sigl=0.01, ga_pcs=True):
    '''
    main manifold figure:
    1. plot example 3D trajectories,
    2. plot lines for distance(t) (curve 'var' or 'euc')
       for select regions
    3. plot 2d scatter [amplitude, latency] of all regions

    sigl: significance level, default 0.01, p_min = 1/(nrand+1)
    ga_pcs: If true, plot 3d trajectories of all cells,
            else plot for a single region (first in exs list)

    '''
    if splits is None:
        splits = align

    # specify grid; scatter longer than other panels
    ncols = 12
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(len(splits), ncols)
    axs = []
    k = 0  # panel counter

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
        tops[split + '_s'] = (f'{len(maxsf)}/{len(d)}='
                             f'{np.round(len(maxsf)/len(d),2)}')
        regs_a = [tops[split][0][j] for j in range(len(tops[split][0]))
                  if tops[split][1][j] < sigl]

        regsa.append(regs_a)
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
    '''

    exs0 = {'stim': ['PRNr', 'LGd', 'LP', 'VISpm', 'VISp',
                     'VISam', 'MRN', 'SCm', 'IP', 'IRN'],

            'choice': ['ACAv', 'PRNr', 'LP', 'SIM', 'APN',
                       'MRN', 'RT', 'IRN', 'IP', 'GRN'],

            'fback': ['IRN', 'SSp-m', 'PRNr', 'IC', 'MV', 'AUDp',
                      'CENT3', 'SSp-ul', 'GPe'],
            'block': ['Eth', 'IC']}

    exs = exs0.copy()
    for split in exs0:
        exs[f'{split}_restr'] = exs0[split]
        exs[f'{split}_restr_shuf'] = exs0[split]

    '''
    Trajectories for example regions in PCA embedded 3d space
    '''

    row = 0
    for split in splits:

        if ga_pcs:
            d = np.load(Path(pth_res, f'{split}_grand_averages.npy'),
                        allow_pickle=True).flat[0]
        else:
            d = np.load(Path(pth_res, f'{split}.npy'),
                        allow_pickle=True).flat[0][split]

            # pick example region
            reg = exs[split][0]

        axs.append(fig.add_subplot(gs[row, :3],
                                   projection='3d'))

        npcs, allnobs = d['pcs'].shape
        nobs = allnobs // ntravis

        for j in range(ntravis):

            # 3d trajectory
            cs = d['pcs'][:, nobs * j: nobs * (j + 1)].T

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

        put_panel_label(axs[k], k)

        k += 1
        row += 1

    '''
    line plot per 5 example regions per split
    '''
    row = 0  # index

    for split in splits:

        axs.append(fig.add_subplot(gs[row, 3:6]))

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

            axs[k].plot(xx, yy, linewidth=2,
                        color=palette[reg],
                        label=f"{reg} {d[reg]['nclus']}")

            # put region labels
            y = yy[-1]
            x = xx[-1]
            ss = f"{reg} {d[reg]['nclus']}"

            if cosregs[reg] in ['CBX', 'CBN']:  # darken yellow
                texts.append(axs[k].text(x, y, ss,
                                         color='k',
                                         fontsize=8))

            texts.append(axs[k].text(x, y, ss,
                                     color=palette[reg],
                                     fontsize=8))

        axs[k].axvline(x=0, lw=0.5, linestyle='--', c='k')

        if split in ['block', 'choice']:
            ha = 'left'
        else:
            ha = 'right'

        axs[k].spines['top'].set_visible(False)
        axs[k].spines['right'].set_visible(False)

        axs[k].set_ylabel('distance [Hz]')
        axs[k].set_xlabel('time [sec]')

        put_panel_label(axs[k], k)

        row += 1
        k += 1

    '''
    scatter latency versus max amplitude for significant regions
    '''

    fsize = 7
    dsize = 4

    if amp_number:
        fig2 = plt.figure()
        axss = []
        fig2.suptitle(f'distance metric: {curve}')

    row = 0  # row idx

    for split in splits:

        axs.append(fig.add_subplot(gs[row, 6:]))

        d = np.load(Path(pth_res, f'{split}.npy'),
                    allow_pickle=True).flat[0]

        acronyms = [tops[split][0][j] for j in range(len(tops[split][0]))]
        ac_sig = np.array([True if tops[split][1][j] < sigl
                           else False for j in range(len(tops[split][0]))])

        maxes = np.array([d[x][f'amp_{curve}'] for x in acronyms])
        lats = np.array([d[x][f'lat_{curve}'] for x in acronyms])
        # stdes = np.array([d[x][f'stde_{curve}'] for x in acronyms])
        cols = [palette[reg] for reg in acronyms]

        if amp_number:  # supp figure for correlation of nclus and maxes
            axss.append(fig2.add_subplot(int(f'1{len(splits)}{row+1}')))
            nums = [1 / d[reg]['nclus'] for reg in np.array(acronyms)[ac_sig]]

            ll = list(zip(nums, np.array(maxes)[ac_sig],
                          np.array(cols)[ac_sig]))
            df = pd.DataFrame(ll, columns=['1/nclus', 'maxes', 'cols'])

            sns.regplot(ax=axss[row], x="1/nclus", y="maxes", data=df)
            axss[row].set_title(split)

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

                if cosregs[reg] in ['CBX', 'CBN']:
                    texts.append(
                        axs[k].annotate(
                            '  ' + reg,
                            (lats[i],
                             maxes[i]),
                            fontsize=fsize,
                            color='k',
                            arrowprops=None))

                else:
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
        axs[k].set_title(f"{tops[split+'_s']} sig")

        put_panel_label(axs[k], k)

        row += 1
        k += 1

    fig = plt.gcf()
    fig.tight_layout()


def plot_custom_lines(regs=None, curve='euc', split='choice',
                      psd_=False, contr_=True):
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
                              split='fback',psd_=True)
    '''

    if regs is None:

        regs = ['SUB', 'BST', 'VPM', 'ANcr2', 'SI', 'LSv', 'CEA', 'MG', 'CA3']

    df, palette = get_allen_info()
    nr = 3
    fig, axs = plt.subplots(nrows=nr, ncols=int(np.ceil(len(regs) / nr)),
                            figsize=(7, 7),
                            sharex=True, constrained_layout=True)
    axs = axs.flatten()
    axsi = []  # inset axes

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
                        if contr == 1. else contr + 0.4,
                        color=palette[reg],
                        label=f"{reg}")

            if jj == 0:
                end_pts[reg] = [yy]
            else:
                end_pts[reg].append(yy)

            cos = cosregs[reg]

            if jj == 4:
                # quanitfy contr stratification
                m = sum(sum(np.diff(np.array(end_pts[reg]), axis=0)))
                end_pts[reg] = m
                ss = f"{reg} {d[reg]['nclus']}"  # , {np.round(m,4)}"
                axs[k].spines['top'].set_visible(False)
                axs[k].spines['right'].set_visible(False)
                axs[k].set_ylabel('distance')
                axs[k].set_xlabel('time [sec]')
                axs[k].set_title(ss, color=palette[reg]
                                 if cos not in ['CBX', 'CBN']
                                 else 'k')

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
            k += 1

        jj += 1
    fig.savefig(f'{"_".join(regs)}.png')
