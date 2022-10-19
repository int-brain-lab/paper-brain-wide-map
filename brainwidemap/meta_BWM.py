import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import percentileofscore, spearmanr


def reformat_results():

    align = {'block':'stim on',
             'stim':'stim on',
             'choice':'motion on',
             'action':'motion on',
             'fback':'feedback'}

    pre_post = {'choice':[0.1,0],'stim':[0,0.1],
                'fback':[0,0.1],'block':[0.4,0],
                'action':[0.025,0.3]}  #[pre_time, post_time]

    mapping = 'Swanson'
    curve = 'd_var_m'    

    D = {} 
    D['date'] = '2022-06-09' 
     
    D['description'] = ('We pool independent neural recordings from a given area'
                        'by stacking peri-event time histograms (PETHs) of all cells'
                        ' in a region across recordings, and thus creating the'
                        ' trial-averaged activity map of a super session. '
                        'For each region we consider the temporal evolution of'
                        ' differences in  trial-averaged firing rate for different'
                        ' trial types,such as activity during the first 100 ms after'
                        ' stimulus onset for the stimulus being on the right versus'
                        ' on the left side of the screen. The results here consist'
                        ' only of super-session regions and maxima of the differences'
                        ' of the trajectories, to be compared with decoding'
                        'accuracies. Differnces for the following 5 trial conditions'
                        ' are reported: d_block, d_choice, d_fback, d_stim, d_action.'
                        'see header of this script for full details:'
                        'https://github.com/int-brain-lab/'
                        'paper-brain-wide-map/blob/develop/'
                        'brainwidemap/manifold_figure.py')
             
    D['inclusion_crit'] = ('BWM standard query; min number of cells per region'
                          ' after pooling: 200.')  
    
    r = []
    cols = ['Swanson_acronym', 'n_clus', 'd_block', 'p_block',
            'd_stim', 'p_stim', 'd_choice', 'p_choice',
            'd_action', 'p_action', 'd_fback', 'p_fback']
                 
    regs = []
    nclus = {}            
    for split in align:
        f = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'
                    f'curves_{split}_mapping{mapping}.npy',
                    allow_pickle=True).flat[0][0]
        regs.append(list(f.keys()))
        for x in f:
            try:
                nclus[x].append(f[x]['nclus'][1])      
            except:
                nclus[x] = []
        
    regsc = Counter(np.concatenate(regs))
    regs2 = [reg for reg in regsc if regsc[reg] == len(align)]
    
    for reg in regs2:
        l = [reg, np.rint(np.mean(nclus[reg]))]  
        for split in align:
            d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'
                        f'curves_{split}_mapping{mapping}.npy',
                        allow_pickle=True).flat[0]
                        
            max_ = d[0][reg][f'max_{curve}']
            null_ = [d[i][reg][f'max_{curve}'] for i in range(1,len(d))]
            p = 1 - (0.01 * percentileofscore(null_,max_,kind='weak'))           
                        
            l.append(max_)
            l.append(p)           
                        
        r.append(l)
       
    D['data'] = pd.DataFrame(data=r,columns=cols)
    np.save('/home/mic/paper-brain-wide-map/fig5_res.npy',D, allow_pickle=True)   


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
      
        
def correlate_decoding_manifold():

    # split in ['block','stim','choice','fback']
    
    D = np.load('/home/mic/paper-brain-wide-map/fig5_res.npy',
                allow_pickle=True).flat[0]['data']

    _,palette = get_allen_info()
    fig, axs = plt.subplots(nrows=1, ncols=4)
    
    k = 0
    for split in ['block','stim','choice','fback']:
        # load Brandon's results            
        b = np.load('/home/mic/paper-brain-wide-map/'
                    'Figure 4 - Decoding-20220609T102805Z-001/'
                    f'Figure 4 - Decoding/{split}_BWM_2022-06-01.p', 
                    allow_pickle=True)   
                    
        # keep only significant accuracies            
        b = b['data'][b['data']['pvalue']<0.05]
        
        # mean sessions to get one value per region 
        try:           
            b = b[['acronym','score_accuracy']].groupby('acronym').mean()    
        except:
            b = b[['acronym','score_r2']].groupby('acronym').mean()
                
        cols = ['acronym', 'accuracy', 'distance']
        r = []
        
        for i in range(len(b)):        
                
            reg = b.iloc[i].name
            if reg in ['void','root']:
                continue

            if reg in D['Swanson_acronym'].values:
                idx = list(D['Swanson_acronym'].values).index(reg)
                try:        
                    r.append([reg, b.iloc[i].score_accuracy,
                              D[f'd_{split}'].values[idx]])
                except:
                    r.append([reg, b.iloc[i].score_r2, D[f'd_{split}'].values[idx]])
    
        df = pd.DataFrame(data=r,columns=cols)       
        sns.regplot(x="accuracy", y="distance", data=df, ax = axs[k])
        
        for i in range(len(r)):
        
            axs[k].annotate('  ' + r[i][0],(r[i][1], r[i][2]),
                fontsize=12,color=palette[r[i][0]])
        
        axs[k].set_title(split)
        k+=1

    plt.tight_layout()
    
    
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
    
    c1, p1 = A[a1]
    c2, p2 = A[a2] 
    
    sig_level = 0.01
    s = pd.read_excel('/home/mic/paper-brain-wide-map/'
                      f'meta/{split}.xlsx', sheet_name='Sheet1')

    dfa, palette = get_allen_info()  # for colors
    
    regsa = s['regions'].values
    cosregs_ = [dfa[dfa['id'] == int(dfa[dfa['acronym']==reg]['structure_id_path']
           .values[0].split('/')[4])]['acronym']
           .values[0] for reg in regsa]    
    cosregs = dict(zip(regsa,cosregs_))  # to make yellow labels black        
    
    if p2 == 0:
        sigs = s[p1] < sig_level
    else:
        sigs = np.bitwise_and(s[p1] < sig_level,s[p2] < sig_level)
        
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
    ax.set_title(f'{split}; #regs with amps [both p < {sig_level}]:'
                 f' {n_regs} [{n_sig_regs}] \n'
                 f' corr_all [p] = {co} [{p}], '
                 f'corr_sig [p_sig]= {co_sig} [{p_sig}]')
    ax.set_xlabel(f'{c1} ({a1})')
    ax.set_ylabel(f'{c2} ({a2})') 
    
      
    for i in s[sigs].index:
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
               f'{split}_{a1}_{a2}.png')
               
               
               
               
               
               
    
    
 
#  Motor corelate and decoding per eid    
#    
#    df.to_csv('/home/mic/paper-brain-wide-map/'
#              'behavioral_block_correlates/motor_corr_0.6_0.2.csv')

#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
           

        
