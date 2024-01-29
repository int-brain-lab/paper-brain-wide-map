import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import (percentileofscore, spearmanr,
combine_pvalues, fisher_exact, barnard_exact, boschloo_exact, ttest_ind)
from pathlib import Path
from one.api import ONE
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
from itertools import combinations
from one.remote import aws
from one.webclient import AlyxClient
from brainwidemap import download_aggregate_tables, bwm_units
import os
import ibllib
from ibllib.atlas.plots import plot_swanson, plot_swanson_vector
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import string
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.ticker as tck


pd.options.mode.chained_assignment = None
#plt.rcParams.update(plt.rcParamsDefault)
one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)
pth_res = Path(one.cache_dir, 'manifold', 'res')
ba = AllenAtlas()
br = BrainRegions()

sig_level = 0.01  # significance level
align = {'stim':'stim on',
         'choice':'motion on',
         'fback':'feedback',
         'block':'stim on'}

f_size = 15  # font size


def load_results(variable):
    ''' Load results

    variable: ['stim', ' choice', 'fback',
               'block', 'speed', 'veloc']
    '''

    res = pd.read_pickle(pth_res.parent.parent / 'meta' / f"{variable}.pkl")
    res = res.replace(True, 1)
    res = res.replace(False, 0)

    # ## Apply logarithm to GLM results
    res.glm_effect = np.log10(
                res.glm_effect.clip(lower=1e-5))

    cols = ['euclidean_latency',
            'euclidean_effect',
            'glm_effect',
            'mannwhitney_effect',
            'decoding_effect',
            'decoding_significant',
            'decoding_frac_significant',
            'mannwhitney_significant',
            'euclidean_significant']

    if variable not in ['speed', 'veloc']:
        # Reorder columns to match ordering in Figure
        res = res[cols]

    return res


def put_panel_label(ax, k):
    ax.annotate(string.ascii_lowercase[k], (-0.05, 1.15),
                xycoords='axes fraction',
                fontsize=f_size * 1.5, va='top',
                ha='right', weight='bold')

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

    splits = ['stim', 'choice', 'fback','block']
    for split in align:
        r = []
        split = split+'_restr'
        d = np.load(Path(pth_res,f'{split}.npy'),
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
        df.to_csv('/home/mic/bwm/'
                 f'meta/per_reg/manifold/{split}.csv')


def glm_to_csv():

    t = pd.read_parquet('/home/mic/paper-brain-wide-map'
                        '/meta/per_reg/glm/2023-01-16'
                        '_glm_mean_drsq.parquet')

    splits = ['stim', 'choice','fback','block']
    res = [abs(t['stimonL'] - t['stimonR']),
           abs(t['fmoveR']  - t['fmoveL']),
           abs(t['correct']  - t['incorrect']),
           abs(t['pLeft'])]

    d0 = dict(zip(splits,res))
    d = {i: d0[i].to_frame().reset_index() for i in d0}

    rr = t['region'].reset_index()
    acs = rr['region'].values

    for split in d:
        d[split]['region'] = acs
        d[split] = d[split].groupby(['region']).mean()
        d[split] = d[split].reset_index()

        if 'pLeft' in d[split].keys():
            d[split] = d[split].rename(columns={'pLeft': 'abs_diff'})
        else:
            d[split] = d[split].rename(columns={0: 'abs_diff'})

        d[split] = d[split].drop('clu_id', axis=1)

        d[split].to_csv('/home/mic/paper-brain-wide-map/'
                 f'meta/per_reg/glm/{split}.csv')


def brandon_weights(split):

    '''
    transform Brandons csv to get cluster id, pid
    '''

    file_ = download_aggregate_tables(one)
    df = pd.read_parquet(file_)

    # decoding weights
    s0 = pd.read_csv('/home/mic/paper-brain-wide-map/meta/'
                    f'per_cell/decoding/18-01-2023_{split}_'
                    'clusteruuids_weights.csv')

    # get cluster_id and pid per cell
    uuids_b = s0['cluster_uuids'].values
    uuids_t = df['uuids'].values
    uuids = pd.Series(list(set(uuids_b).intersection(set(uuids_t))))
    y = df.set_index('uuids').loc[uuids,['cluster_id', 'pid']]
    y = y.reset_index()

    print(split, 'len b', len(uuids_b), 'len intersect', len(uuids))

    ws = []
    for i in range(len(s0)):
        ws.append(abs(s0.loc[i,'ws_fold0_runid0':].mean()))

    dd = pd.DataFrame()
    dd['uuids'] = s0['cluster_uuids']
    dd['abs_weight'] = ws

    ddd = pd.merge(y,dd, on =['uuids'])

    # load in firing rates
    pth = Path(one.cache_dir, 'manifold', f'{split}_fr')
    ss = os.listdir(pth)  # get insertions
    us = []
    for s in ss:
        u = np.load(Path(pth,s), allow_pickle=True).flat[0]
        us.append(pd.DataFrame.from_dict(u))

    us = pd.concat(us, axis=0)
    us.rename(columns = {'cluster_ids':'cluster_id'}, inplace = True)
    d3 = pd.merge(us,ddd, on =['pid','cluster_id'])
    d3.reset_index()
    d3.to_csv('/home/mic/paper-brain-wide-map/'
             f'meta/per_cell/decoding/{split}_weights.csv')


def histograms_of_decoding_weights():

    '''
    across all bwm cells included in decoding analysis (20 k)
    plot histograms of weights, firing rates, weights * firing rates
    for each split
    '''

    splits = ['choice', 'stim','fback','block']

    fig, axs = plt.subplots(nrows=len(splits), ncols=3)


    r = 0
    for split in splits:
        s1 = pd.read_csv('/home/mic/paper-brain-wide-map/meta/'
                         f'per_cell/decoding/{split}_weights.csv')

        # multiply firing rates with decoding weights
        s1['fr_x_weight'] = (s1['abs_weight'].values *
                             s1['f_rates'].values)

        c = 0
        for htype in ['f_rates', 'abs_weight', 'fr_x_weight']:
            s1.hist(htype, ax = axs[r,c],bins = 1000)
            axs[r,c].set_title(' '.join([split, htype]))
            axs[r,c].set_xlabel(htype)

            c += 1
        r += 1



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


def cor_per_reg(split, a1 = 'manifold', a2 = 'decoding', curve = 'euc'):

    '''
    c1/2 amplitudes to correlate per region
    p1/2 p-values for significance filter
    '''

    ss = [pd.read_csv('/home/mic/paper-brain-wide-map/'
                      f'meta/per_reg/{a}/{split}.csv')
                      for a in [a1, a2]]

    #  analysis: [amplitude, pvalue]
    A = {'manifold': [f'amp_{curve}',f'p_{curve}'],
         'decoding': ['values_median','combined_p-value'],
         'single-cell': ['fra', 0],
         'glm': ['abs_diff', 0]}

    c1, p1 = A[a1]
    c2, p2 = A[a2]

    s = pd.merge(ss[0],ss[1], how='inner', on=['region'])

    dfa, palette = get_allen_info()  # for colors

    regsa = s['region'].values
    cosregs_ = [dfa[dfa['id'] ==
                int(dfa[dfa['acronym']==reg]['structure_id_path']
                .values[0].split('/')[4])]['acronym']
                .values[0] for reg in regsa]

    cosregs = dict(zip(regsa,cosregs_))  # to make yellow labels black

    if p1 == 0 and p2 != 0:
        sigs2 = s[p2] < sig_level
        sigs = sigs2

    if p2 == 0 and p1 != 0:
        sigs = s[p1] < sig_level
        sigs2 = sigs

    if p2 == 0 and p1 == 0:
        sigs = s['region'] != None
        sigs2 = sigs

    if p1 != 0 and p2 != 0:

        sigs = np.bitwise_and(s[p1] < sig_level,s[p2] < sig_level)
        sigs2 = np.bitwise_or(s[p1] < sig_level,s[p2] < sig_level)

    m_both = ~np.bitwise_or(np.isnan(s[c1]),np.isnan(s[c2]))
    n_regs = sum(m_both)
    n_sig_regs = sum(sigs)

    # correlate results
    co0,p0 = spearmanr(s[c1][m_both],s[c2][m_both])
    co_sig0,p_sig0 = spearmanr(s[c1][sigs],s[c2][sigs])

    co, p, co_sig, p_sig = [np.round(x,2) for x in [co0, p0, co_sig0, p_sig0]]

    cols = np.array([palette[reg] for reg in regsa])

    fig,ax = plt.subplots(figsize=(10,10))
    ax.scatter(s[c1],s[c2],c=cols, s = 1,
               label = 'neither sig')

    if p2 != 0 and p1 != 0:

        ax.scatter(s[c1][s[p1] < sig_level],s[c2][s[p1] < sig_level],
                   c=cols[s[p1] < sig_level], s = 10,
                   marker = 'v', label = 'x only sig')

        ax.scatter(s[c1][s[p2] < sig_level],s[c2][s[p2] < sig_level],
                   c=cols[s[p2] < sig_level], s = 10,
                   marker = '^', label = 'y only sig')

    ax.scatter(s[c1][sigs],s[c2][sigs],c=cols[sigs], s = 20,
               label = 'both sig')

    ax.set_title(f'{split}; #regs with amps [both p < {sig_level}]:'
                 f' {n_regs} [{n_sig_regs}] \n'
                 f' corr_all [p] = {co} [{p}], '
                 f'corr_sig [p_sig]= {co_sig} [{p_sig}]')
    ax.set_xlabel(f'{c1} ({a1})')
    ax.set_ylabel(f'{c2} ({a2})')
    ax.legend()


    for i in s[sigs2].index:
        reg = s.iloc[i]['region']

        if cosregs[reg] in ['CBX', 'CBN']:
            ax.annotate('  ' + reg, # f"{reg} {f[reg]['nclus']}" ,
                        (s.iloc[i][c1], s.iloc[i][c2]),
                fontsize=10,color='k')

        ax.annotate('  ' + reg, # f"{reg} {f[reg]['nclus']}" ,
            (s.iloc[i][c1], s.iloc[i][c2]),
            fontsize=10,color=palette[reg])

    fig.tight_layout()

    fig.savefig('/home/mic/paper-brain-wide-map/meta/figs/'
               f'per_reg/{split}_{a1}_{a2}.png')

    return s


def bulk_per_reg():

    als = ['manifold','decoding','single-cell','glm']

    plt.ioff()
    for a1, a2 in combinations(als,2):
        for split in align:
            try:
                cor_per_reg(split, a1 = a1, a2 = a2)
            except:
                print(split, a1, a2, 'no data')
        plt.close()


def cor_per_eid(split, get_merged = False, ptype=0):

    '''
    Per eid/region analysis; single-cell versus decoding
    one point per eid/reg pair in scatter,
    different markers for significance
    '''

    s0 = pd.read_csv('/home/mic/paper-brain-wide-map/'
                      f'meta/per_eid/decoding/{split}.csv')

    s1 = pd.read_csv('/home/mic/paper-brain-wide-map/'
                      f'meta/per_eid/single-cell/{split}.csv')

    # for block and stim there's two analyses, pick first
    psc = [y for y in s1.keys() if 'p_' in y][ptype]

    # group single-cell results into region/eid pairs, get
    # frac_of_sig_cells for each and list of p-vaues for hists

    eids = Counter(s1['eid'].values)

    cols = ['eid', 'region', 'frac_cells', 'ps']
    r = []
    for eid in eids:

        st = s1[s1['eid'] == eid]
        atids = st['atlas_id'].values
        acs = br.id2acronym(atids,mapping='Beryl')
        st['region'] = acs
        regs = Counter(st['region'].values)
        for reg in regs:
            ps = st[st['region'] == reg][psc].values
            r.append([eid, reg, np.mean(ps < sig_level), ps])

    s2 = pd.DataFrame(columns = cols, data = r)
    s = pd.merge(s0,s2, how='inner', on=['eid', 'region'])

    # drop region in root, void
    s = s[~np.bitwise_or.reduce([s['region'] == x for x in ['root', 'void']])]

    if get_merged:
        return s

    dfa, palette = get_allen_info()  # for colors

    regsa = s['region'].values
    cosregs_ = [dfa[dfa['id'] == int(dfa[dfa['acronym']==reg]['structure_id_path']
           .values[0].split('/')[4])]['acronym']
           .values[0] for reg in regsa]
    cosregs = dict(zip(regsa,cosregs_))  # to make yellow labels black

    sigs = s['p-value'] < sig_level

    c1, a1 = 'score', 'decoding'
    c2, a2 = 'frac_cells', 'single-cell'

    m_both = ~np.bitwise_or(np.isnan(s[c1]),np.isnan(s[c2]))
    n_regs = sum(m_both)
    n_sig_regs = sum(sigs)

    # correlate results
    co0,p0 = spearmanr(s[c1][m_both],s[c2][m_both])
    co_sig0,p_sig0 = spearmanr(s[c1][sigs],s[c2][sigs])

    co, p, co_sig, p_sig = [np.round(x,2) for x in [co0, p0, co_sig0, p_sig0]]

    cols = np.array([palette[reg] for reg in regsa])

    fig,ax = plt.subplots(figsize=(10,10))
    ax.scatter(s[c1],s[c2],c=cols, s = 1)
    ax.scatter(s[c1][sigs],s[c2][sigs],c=cols[sigs], s = 20)
    ax.set_title(f'{split}; #regs with amps [p_decoding < {sig_level}]:'
                 f' {n_regs} [{n_sig_regs}] \n'
                 f' corr_all [p] = {co} [{p}], '
                 f'corr_sig [p_sig]= {co_sig} [{p_sig}] \n'
                 f' single-cell p-value type: {psc}')
    ax.set_xlabel(f'{c1} ({a1})')
    ax.set_ylabel(f'{c2} ({a2})')

    s = s[sigs].reset_index()
    for i in s.index:
        reg = s.iloc[i]['region']

        if cosregs[reg] in ['CBX', 'CBN']:
            ax.annotate('  ' + reg, # f"{reg} {f[reg]['nclus']}" ,
                        (s.iloc[i][c1], s.iloc[i][c2]),
                fontsize=10,color='k')

        ax.annotate('  ' + reg, # f"{reg} {f[reg]['nclus']}" ,
            (s.iloc[i][c1], s.iloc[i][c2]),
            fontsize=10,color=palette[reg])

        if reg == 'MRN':
            if np.bitwise_and.reduce([s.iloc[i]['score'] > 0.85,
                               s.iloc[i]['p-value'] < 0.01,
                               s.iloc[i]['frac_cells'] == 0]):
                print(s.iloc[i])

    fig.tight_layout()

    fig.savefig('/home/mic/paper-brain-wide-map/meta/figs/'
               f'per_eid/{split}_{a1}_{a2}.png')

    fig.tight_layout()


def inspect_hists(s):

    '''
    recording cases with low single-cell modulation
    but high decoding
    '''

    split = 'choice'
    reg = 'GRN'
    eids = ['571d3ffe-54a5-473d-a265-5dc373eb7efc',
            '671c7ea7-6726-4fbe-adeb-f89c2c8e489b',
            'aec5d3cc-4bb2-4349-80a9-0395b76f04e2']

    fig, axs = plt.subplots(nrows=1, ncols=len(eids),
                            sharey = True, sharex=True)

    k = 0
    for eid in eids:
        x = s[np.bitwise_and(s['region']=='GRN',s['eid'] == eid)]
        axs[k].hist(x['ps'].values, bins=25)

        t = ' '.join([f'decoding score: {np.round(x["score"].values[0],3)}',
                    ' \n', f'p_dec = {np.round(x["p-value"].values[0], 3)}', ' \n',
                    f"{x['eid'].values[0][:6]}...", reg,' \n',
                    f"{x['subject'].values[0]}",
                    f'n_units: {x["n_units"].values[0]}', '\n',
                    f"frac sig cells: {np.round(x['frac_cells'].values[0],5)}"])
        axs[k].set_title(t)
        axs[k].axvline(x=sig_level,c='r',linestyle='--')
        axs[k].set_xlabel('p-value [single-cell]')
        axs[k].set_ylabel('#cells')

        k += 1

    fig.suptitle('Example recordings eid/reg with high decoding of CHOICE but low fraction of significant cells (single-cell)')
    fig.tight_layout()



def motor_block_eid(sig_lev = 0.01):
    '''
    comparing motor correlates and block decoding
    '''

    dm = pd.read_csv('/home/mic/bwm/meta/'
                     'motor_corr_0.6_0.2.csv')
    db = pd.read_csv('/home/mic/bwm/meta/'
                     'per_eid/decoding/block.csv')

    eids = list(set(db['eid'].values
                ).intersection(set(dm['eid'].values)))

    print(len(eids), 'eids in common')

    # for each eid count number of significant regions and significant
    # behaviors

    cols = ['eid','sig_beh', 'frac_beh', 'sig_regs', 'frac_regs',
            'min_p_beh', 'min_p_dec', 'max_acc', 'scores', 'regs']
    r = []
    for eid in eids:

        # count number of sig decodable regs and sig behaviors
        x = db[db['eid'] == eid]
        y = dm[dm['eid'] == eid]

        ps = [k for k in dm.keys() if k[-2:] == '_p']

        sig_beh = [ps[i] for i in range(len(ps)) if
                y[ps[i]].values[0] < sig_lev]

        sig_regs = list(x[x['p-value'] < sig_lev]['region'].values)

        # get minimal p-value across behaviors
        min_p_beh = min([y[ps[i]].values[0] for i in range(len(ps))])
        min_p_dec = min(x['p-value'].values)
        max_acc = max(x['score'].values)
        scores = x['score'].values
        regs = x['region'].values

        r.append([eid, sig_beh, len(sig_beh)/len(ps),
                       sig_regs, len(sig_regs)/len(x),
                       min_p_beh, min_p_dec, max_acc, scores, regs])

    df = pd.DataFrame(columns = cols, data = r)

    # print contingency table

    df['at_least1_reg'] = df['sig_regs'].map(lambda d: len(d)) > 0
    df['at_least1_beh'] = df['sig_beh'].map(lambda d: len(d)) > 0
    ct = pd.crosstab(index=df['at_least1_reg'], columns=df['at_least1_beh'])
    print(ct)
    print('fisher_exact:', fisher_exact(ct,alternative="two-sided"))
    print('barnard_exact:', barnard_exact(ct,alternative="two-sided"))
    print('boschloo_exact:', boschloo_exact(ct,alternative="two-sided"))
    #mosaic(myDataframe, ['size', 'length'])

    #return df


    # Unpaired t-test between max decoding across regions;
    # one of those that have at_least1_beh
    a = np.concatenate(df[~df['at_least1_beh']]['scores'].values)
    b = np.concatenate(df[df['at_least1_beh']]['scores'].values)
    t, p = ttest_ind(a,b)

    fig, ax = plt.subplots()
    ax.hist(a, label='dec. scores without 1 beh', color='r')
    ax.hist(b, label='dec. scores with at least 1 beh', color='b')
    ax.set_xlabel('decoding scores')
    ax.set_ylabel('frequency')
    ax.set_title(f't-test statistic, p = ({np.round(t,2)}, {np.round(p,2)})')
    ax.legend()

    '''
    plot scatter per session
    '''

    # column keys to scatter
    a = ['min_p_beh','min_p_dec']#'max_acc'

    fig, ax = plt.subplots(ncols=2)
    ax[0].scatter(np.log(df[a[0]]), np.log(df[a[1]]))
    ax[0].set_xlabel(f'log({a[0]})')
    ax[0].set_ylabel(f'log({a[1]})')

    c,p = spearmanr(np.log(df[a[0]]),np.log(df[a[1]]))

    ax[0].set_title(f'Point = session; Spearman: {np.round(c,2)}, {np.round(p,2)}')

    ax[1].scatter(df[a[0]], df[a[1]])
    ax[1].set_xlabel(f'{a[0]}')
    ax[1].set_ylabel(f'{a[1]}')
    c,p = spearmanr(df[a[0]],df[a[1]])

    ax[1].set_title(f'Point = session; Spearman: {np.round(c,2)}, {np.round(p,2)}')

    fig = plt.gcf()
    fig.tight_layout()

#
#    ax = sns.lmplot(x=a[0],y=a[1],data=df,fit_reg=True)
#
#    ax = plt.gca()

    return df


def motor_res_to_df():


    # save results for plotting here
    pth_res = Path(one.cache_dir, 'brain_wide_map', 'motor_correlates')
    pth_res.mkdir(parents=True, exist_ok=True)
    sr = {'licking': 'T_BIN', 'whisking_l': 60, 'whisking_r': 150,
          'wheeling': 'T_BIN', 'nose_pos': 60, 'paw_pos_r': 150,
          'paw_pos_l': 60}

    t = '0.6'  # lag = -0.6 sec
    s = (pth_res / f'behave7_{t}.npy')
    R = np.load(s,allow_pickle=True).flat[0]

    columns = ['eid'] + list(np.concatenate([[x+'_p', x+'_amp'] for x in sr]))
    r = []
    for eid in R:
        #print(eid)
        try:  # there's one weird session, trials object nan
            r.append([eid] +
                     list(np.concatenate([[R[eid][b][1],R[eid][b][0]]
                     for b in R[eid]])))
        except:
            continue

    df = pd.DataFrame(columns = columns, data=r)
    df.to_csv('/home/mic/bwm/meta/'
              'motor_corr_0.6_0.2.csv')



def neuron_count(split = 'stim'):

    '''
    reformat results for table
    '''

    columns = ['reg Beryl', 'reg Cosmos', 'nclus']

    r = []
    d = np.load(Path(pth_res,f'{split}.npy'),
                allow_pickle=True).flat[0]

    dfa, palette = get_allen_info()
    cosregs_ = {reg: dfa[dfa['id'] == int(dfa[dfa['acronym'] == reg][
                'structure_id_path'].values[0].split('/')[4])][
                'acronym'].values[0] for reg in d}

    for reg in d:

        r.append([reg, cosregs_[reg],  d[reg]['nclus']])

    df  = pd.DataFrame(data=r,columns=columns)
    print(split, 'total neuron count', sum(df['nclus']))
    #print(df.groupby('reg Cosmos').sum(numeric_only=True))


def plot_bar_neuron_count(table_only=False):

    '''
    bar plot for neuron count per region
    second bar for recording count per region

    for BWM intro figure;

    Adding additional info in the table, including effect sizes
    '''

    file_ = download_aggregate_tables(one)
    df = pd.read_parquet(file_)
    dfa, palette = get_allen_info()
    df['Beryl'] = br.id2acronym(df['atlas_id'], mapping='Beryl')
    df['Cosmos'] = br.id2acronym(df['atlas_id'], mapping='Cosmos')

    cosregs = dict(list(Counter(zip(df['Beryl'],df['Cosmos']))))


    # number of clusters per region, c0
    # remove regions with < 10 or root, void
    c0 = dict(Counter(df['Beryl']))
    #c0 = {reg : c[reg] for reg in c if c[reg] > 10}
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
    p = (Path(ibllib.__file__).parent / 'atlas/beryl.npy')
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
    print(f'{len(regs1)} regions, {sum(nclus_nins[:,0])} neurons,'
          f'{sum(nclus_nins[:,1])} recordings')



    if table_only:
        vs = ['stim', 'choice', 'fback', 'block','speed', 'veloc']
        ans = ['euc', 'glm', 'dec', 'man']
        effects = [('_').join([x,y]) for x in vs for y in ans]
        res = {}

        for v in vs:
            res[v] = load_results(v).to_dict()

        res2 = {}
        for reg in regs1:
            res3 = {}
            for v in vs:
                for a in ans:
                    key = [x for x in res[v].keys()
                           if ((a in x) and ('effect' in x))]
                    if key == []:
                        res3[('_').join([v,a])] = None
                        continue
                    else:
                        key = key[0]

                    if reg in res[v][key].keys():
                        res3[('_').join([v,a])] = np.round(
                                                   res[v][key][reg],2)
                    else:
                        res3[('_').join([v,a])] = None

            res2[reg] = res3

        # get all regions in the canonical set
        units_df = bwm_units(one)
        gregs = Counter(units_df['Beryl'])

        columns = ['Beryl', 'name', 'Cosmos',
                   '# recordings', '# neurons',
                   '# good neurons', 'canonical'] + effects

        r = []
        for k in range(len(regs1)):
            cano = True if regs1[k] in gregs else False

            r.append([regs1[k], get_name(regs1[k]),
                      cosregs[regs1[k]], nclus_nins[k,1],
                      nclus_nins[k,0], nclus_nins[k,2], cano] +
                      list(res2[regs1[k]].values()))

        df  = pd.DataFrame(data=r,columns=columns)
        df  = df.reindex(index=df.index[::-1])
        df.to_csv('/home/mic/bwm/'
                 f'meta/region_info.csv')
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


def neuron_number_swansons():

    '''
    3 Swansons: all neurons, only good neurons,
               recordings
    '''
    df = pd.read_csv('/home/mic/bwm/'
             f'meta/table.csv')

    fig, axs = plt.subplots(nrows=1, ncols=3)

    dd = ['# neurons', '# good neurons', '# recordings']

    vmin0 = np.min([df[dd[0]].min(), df[dd[1]].min()])
    vmax0 = np.max([df[dd[0]].max(), df[dd[1]].max()])

    for c in range(3):

        amps = df[dd[c]]
        plot_swanson_vector(df['Beryl'], amps,
                            cmap=get_cmap('block'),
                            ax=axs[c], br=br,
                            orientation='portrait',
                            linewidth=0.1,
                            vmin=vmin0 if c!= 2 else min(amps),
                            vmax=vmax0 if c!= 2 else max(amps),
                            annotate=True,
                            annotate_n=5,
                            annotate_order='bottom' if c == 0 else 'top')

        # add colorbar
        norm = mpl.colors.Normalize(vmin=vmin0 if c!= 2 else min(amps),
                                    vmax=vmax0 if c!= 2 else max(amps))

        cbar = fig.colorbar(mpl.cm.ScalarMappable(
                                norm=norm,
                                cmap=get_cmap('block')),
                                ax=axs[c])

        cbar.set_label(dd[c])


        axs[c].axis('off')
        #axs[c].set_title(dd[c])
        put_panel_label(axs[c], c)


