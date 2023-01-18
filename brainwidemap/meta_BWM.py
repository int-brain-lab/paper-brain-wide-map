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

pd.options.mode.chained_assignment = None

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


def get_name(brainregion):
    regid = br.id[np.argwhere(br.acronym == brainregion)][0, 0]
    return br.name[np.argwhere(br.id == regid)[0, 0]]


def manifold_to_csv():

    '''
    reformat results for table
    '''
    
    mapping = 'Beryl'

    columns = ['region','name','nclus', 
               f'p_var', f'amp_var', f'lat_var',
               f'p_euc', f'amp_euc', f'lat_euc']

    b = np.load(Path(pth_res,f'beryl.npy'))
    for split in align:        
        r = []
        
        regs =br.id2acronym(b,mapping='Beryl')

        d = np.load(Path(pth_res,f'{split}.npy'),
                    allow_pickle=True).flat[0] 
        
        for reg in regs:

            if reg in d:
                r.append([reg, get_name(reg), d[reg]['nclus'],
                          d[reg][f'p_var'], 
                          np.max(d[reg]['d_var'][:48]) 
                          if split == 'stim' else d[reg][f'amp_var'],
                          d[reg][f'lat_var'],                          
                          d[reg][f'p_euc'],
                          np.max(d[reg]['d_euc'][:48]) 
                          if split == 'stim' else d[reg][f'amp_euc'],
                          d[reg][f'lat_euc']]) 
            else:
                r.append([reg, get_name(reg), '','', '','', '', '','']) 
                      
        df  = pd.DataFrame(data=r,columns=columns)        
        df.to_csv('/home/mic/paper-brain-wide-map/'
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
                 
                 

def get_allen_info():
    dfa = pd.read_csv('/home/mic/paper-brain-wide-map/'
                       'allen_structure_tree.csv')
    
    # get colors per acronym and transfomr into RGB
    dfa['color_hex_triplet'] = dfa['color_hex_triplet'].fillna('FFFFFF')
    dfa['color_hex_triplet']  = dfa['color_hex_triplet'
                                    ].replace('19399','19399a')
    dfa['color_hex_triplet']  = dfa['color_hex_triplet'].replace('0','FFFFFF')
    dfa['color_hex_triplet'] = '#' + dfa['color_hex_triplet'].astype(str)
    dfa['color_hex_triplet'] = dfa['color_hex_triplet'
                                ].apply(lambda x: 
                                mpl.colors.to_rgba(x))
                                
    palette = dict(zip(dfa.acronym, dfa.color_hex_triplet))
    
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

    dm = pd.read_csv('/home/mic/paper-brain-wide-map/meta/'
                     'motor_corr_0.6_0.2.csv')
    db = pd.read_csv('/home/mic/paper-brain-wide-map/meta/'
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


    t = '0.6'  # lag = -0.6 sec
    s = (pth_res / 'behave7{t}.npy')
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
    df.to_csv(path_res / 'motor_corr_0.6_0.2.csv')





#    else:    
#        # combine eid results from decoding and single-cell per region
#        dss = cor_res_eid(split, get_merged = True)
#        dm =  np.load(Path(pth_res,f'{split}.npy'),
#                      allow_pickle=True).flat[0]
#                                        
#        if a1 == 'manifold' and a2 == 'decoding':              
#            regs = list(set(dss['region'].values).
#                       intersection(set(dm.keys())))
#                          
#        elif a1 == 'manifold' and a2 == 'single-cell':
#            regs = list(set(dss['region'].values).
#                       intersection(set(dm.keys())))
#            
#        elif a1 == 'decoding' and a2 == 'single-cell':
#            regs = Counter(dss['region'].values)
#         
#        else:
#            print('wring analysis order')
#            return              
#                      
#        cols = ['region'] + list(np.concatenate([A[x] for x in [a1,a2]]))
#        r = []
#        for reg in regs:
#            if a1 == 'manifold' and a2 == 'decoding':

#                median_vals = dss[dss['region']==reg]['score'].median() 
#                combined_p = combine_pvalues(dss[dss['region']==reg]
#                                             ['p-value'].values)[1]        
#            
#                r.append([reg, dm[reg][A[a1][0]], dm[reg][A[a1][1]],
#                          median_vals, combined_p])

#            elif a1 == 'manifold' and a2 == 'single-cell':

#                frac_of_sig = (dss[dss['region']==reg]
#                                 ['frac_cells'].median())           
#            
#                r.append([reg, dm[reg][A[a1][0]], dm[reg][A[a1][1]],
#                          frac_of_sig, 0])
#                          
#                          
#            elif a1 == 'decoding' and a2 == 'single-cell':
#            
#                median_vals = dss[dss['region']==reg]['scores'].median() 
#                combined_p = combine_pvalues(dss[dss['region']==reg]
#                                             ['p-value'].values)[1]
#                frac_of_sig = (dss[dss['region']==reg]
#                                 ['frac_cells'].median())           
#            
#                r.append([reg, median_vals, combined_p,
#                          frac_of_sig, 0])        
#                                    
#            else:
#                print('wrong analysis order')
#                return             
#            
#        s = pd.DataFrame(data = r,columns = cols)
