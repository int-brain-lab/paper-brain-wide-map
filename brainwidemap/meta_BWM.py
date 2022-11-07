import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import percentileofscore, spearmanr

from one.api import ONE
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions

pd.options.mode.chained_assignment = None

one = ONE()  # (mode='local')
ba = AllenAtlas()
br = BrainRegions()

sig_level = 0.01  # significance level


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
      
 
def cor_res(split, a1 = 'manifold', a2 = 'decoding'):

    '''
    17.10, using newest excel sheets from Brandon
    c1/2 amplitudes to correlate per region
    p1/2 p-values for significance filter
    
    '''
    
    #  analysis: [amplitude, pvalue]
    A = {'manifold': ['amp_euc2','p_euc2'], 
         'decoding': ['median_vals',"combined_p-values (Fisher's method)"], 
         'single-cell': ['fraction_of_sig_cells', 0]}
    
    assert a1 != 'single-cell', 'swap order as single-cell has no p'
    
    c1, p1 = A[a1]
    c2, p2 = A[a2] 
    
    
    s = pd.read_excel('/home/mic/paper-brain-wide-map/meta/'
                      f'per_reg/{split}.xlsx', sheet_name='Sheet1')

    dfa, palette = get_allen_info()  # for colors
    
    regsa = s['regions'].values
    cosregs_ = [dfa[dfa['id'] == 
                int(dfa[dfa['acronym']==reg]['structure_id_path']
                .values[0].split('/')[4])]['acronym']
                .values[0] for reg in regsa]
           
    cosregs = dict(zip(regsa,cosregs_))  # to make yellow labels black        
    
    if p2 == 0:
        sigs = s[p1] < sig_level
        sigs2 = sigs
    else:
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
     
    if p2 != 0:
    
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
        reg = s.iloc[i]['regions']
        
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
               
               
               
def motor_block_eid(sig_lev = 0.05):
    '''
    comparing motor correlates and block decoding
    '''

    dm = pd.read_csv('/home/mic/paper-brain-wide-map/meta/'
                     'motor_corr_0.6_0.2.csv')
    db = pd.read_csv('/home/mic/paper-brain-wide-map/meta/'
                     'block_eid_brandon.csv')
    
    eids = list(set(db['eid'].values
                ).intersection(set(dm['eid'].values)))
    
    # for each eid count number of significant regions and significant 
    # behaviors
    
    cols = ['eid','sig_beh', 'frac_beh', 'sig_regs', 'frac_regs'] 
    r = []
    for eid in eids:
    
        # count number of sig decodable regs and sig behaviors
        x = db[db['eid'] == eid]  
        y = dm[dm['eid'] == eid]
  
        ps = [k for k in dm.keys() if k[-2:] == '_p']
        
        sig_beh = [ps[i] for i in range(len(ps)) if 
                y[ps[i]].values[0] < sig_lev]
                 
        sig_regs = list(x[x['p-value'] < sig_lev]['region'].values)
        
        r.append([eid, sig_beh, len(sig_beh)/len(ps), 
                       sig_regs, len(sig_regs)/len(x)])
    
    df = pd.DataFrame(columns = cols, data = r)
    
    
 
def cor_res_eid(split, get_merged = False, ptype=0):

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
    
    
