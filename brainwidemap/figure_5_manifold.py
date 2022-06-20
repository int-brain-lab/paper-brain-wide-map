from one.api import ONE
from reproducible_ephys_processing import bin_spikes2D
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
from brainbox.io.one import SpikeSortingLoader

from ibllib.atlas.plots import prepare_lr_data
from ibllib.atlas.plots import plot_scalar_on_flatmap, plot_scalar_on_slice
from ibllib.atlas import FlatMap
from ibllib.atlas.flatmaps import plot_swanson

from scipy import optimize,signal
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

pre_post = {'choice':[0.1,0],'stim':[0,0.1],
            'fback':[0,0.1],'block':[0.4,0],
            'action':[0.025,0.3]}  #[pre_time, post_time], 
            #if single bin, not possible to have pre/post
            
trial_split = {'choice':['choice left', 'choice right'],
               'stim':[1.0,1.0],
               #[0.0, 0.0625, 0.125, 0.25,1.0, 
               #       0.0, 0.0625, 0.125, 0.25,1.0],
               'fback':['correct','false'],
               'block':['pleft 0.8','pleft 0.2'],
               'action':['choice left', 'choice right']}            
  
nrand = 20  # number of random trial splits for choice control
min_reg = 200  # minimum number of neurons in pooled region
one = ONE() #ONE(mode='local')
ba = AllenAtlas()
br = BrainRegions()


def xs(split):
    return np.arange((pre_post[split][0] + pre_post[split][1])/
                      T_BIN(split))*T_BIN(split)


def norm_curve(r,split):

    '''
    normalize curve by pre-stim baseline
    '''
    # normalise via pre-stim baseline
    bsl_m = np.mean(r[0:int(abs(pre_post[split][0])/T_BIN(split))])
    bsl_st = np.std(r[0:int(abs(pre_post[split][0])/T_BIN(split))])
    postStim = np.mean(r[int(abs(pre_post[split][0])/T_BIN(split)):])         
    r = (r-bsl_m)/(bsl_m + 0.1)
    # check if mean baseline is above mean after stim
    stim_responsive = True
    if bsl_m > postStim:
       stim_responsive = False
        
    return r, stim_responsive     


def get_name(brainregion):
    regid = br.id[np.argwhere(br.acronym == brainregion)][0, 0]
    return br.name[np.argwhere(br.id == regid)[0, 0]]


def get_psths_atlasids(split,eid, probe, ind_trials=False, control=False):

    '''
    for a given session, probe, bin neural activity
    cut into trials, reduced to PSTHs, reduced via PCA per region,
    reduce to 2 D curves describing the PCA trajectories
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

    if split == 'choice':
        for choice in [1,-1]:
            events.append(trials['firstMovement_times'][np.bitwise_and.reduce([
                       ~rm_trials,trials['choice'] == choice])])   

    elif split == 'action':
        for choice in [1,-1]:
            events.append(trials['firstMovement_times'][np.bitwise_and.reduce([
                       ~rm_trials,trials['choice'] == choice])])  
                       
    elif split == 'stim':    
        for side in ['Left', 'Right']:
            for contrast in [1.0]:#[0.0, 0.0625, 0.125, 0.25,
                events.append(trials['stimOn_times'][np.bitwise_and.reduce([
                    ~rm_trials,trials[f'contrast{side}'] == contrast])])
       
    elif split == 'fback':    
        for fb in [1,-1]:
            events.append(trials['feedback_times'][np.bitwise_and.reduce([
                ~rm_trials,trials[f'feedbackType'] == fb])])       
              
    elif split == 'block':
        for pleft in [0.8, 0.2]:
            events.append(trials['stimOn_times'][np.bitwise_and.reduce([
                ~rm_trials,trials['probabilityLeft'] == pleft])])

    else:
        print('what is the split?', split)
        return

    # bin and cut into trials    
    bins = []
    for event in events:
        bi, _ = bin_spikes2D(spikes['times'][spike_idx], 
                           spikes['clusters'][spike_idx],
                           clusters['cluster_id'],event, 
                           pre_post[split][0], pre_post[split][1], T_BIN(split))
        bins.append(bi)                   
                                                     
    b = np.concatenate(bins)        
    ntr, nclus, nbins = b.shape
    
    if ind_trials:
        return bins, clusters['atlas_id']
    
    else:
        if control:
            # only average a random half of the trials
            print('only half the trials')
            ntrs = [bi.shape[0] for bi in bins]
            trs = [random.sample(range(ntr), ntr//2) for ntr in ntrs]
            
            w0 = []
            s0 = []
            k = 0
            for bi in bins:
                w0.append(bi[trs[k]].mean(axis=0))
                s0.append(bi[trs[k]].var(axis=0))
                k +=1    
            
        else: # average all trials
            print('all trials')     
            w0 = [bi.mean(axis=0) for bi in bins]  # average trials to get PSTH
            s0 = [bi.var(axis=0) for bi in bins]
            
        # zscore each PETH
        #w = [zscore(x,nan_policy='omit',axis=1) for x in w0]
        
    #    if control:
    #        # nrand times random split of trials as control    
    #        for i in range(nrand):
    #            if split == 'block':
    #                spl = generate_pseudo_blocks(ntr, first5050=0) == 0.8
    #            else:
    #                spl = np.random.randint(0, high=2, size=ntr).astype(bool)
    #            b_r = (b[spl]).mean(axis=0)  # compute PSTH
    #            b_l = (b[~spl]).mean(axis=0)  # compute PSTH
    #            w.append(b_r)
    #            w.append(b_l)
        
        w0 = np.array(w0)  # PSTHs, first two are left. right choice, rest rand

        return w0, s0, clusters['atlas_id']

      
'''    
###
### bulk processing 
###    
'''  


def bwm_query(one, alignment_resolved=True, return_details=False):
    """
    Function to query for brainwide map sessions that pass the most important quality controls. Returns a dataframe
    with one row per insertions and columns ['pid', 'eid', 'probe_name', 'session_number', 'date', 'subject', 'lab']
    Parameters
    ----------
    one: ONE instance (can be remote or local)
    alignment_resolved: bool, default is True. If True, only returns sessions with resolved alignment,
                        if False returns all sessions with at least one alignment
    return_details: bool, default is False. If True returns a second output a list containing the full insertion
                    dictionary for all insertions returned by the query. Only needed if you need information that is
                    not contained in the output dataframe
    Returns
    -------
    bwm_df: pd.DataFrame of BWM sessions to be included in analyses with columns
            ['pid', 'eid', 'probe_name', 'session_number', 'date', 'subject', 'lab']
    """

    base_query = (
        'session__project__name__icontains,ibl_neuropixel_brainwide_01,'
        'session__json__IS_MOCK,False,'
        'session__qc__lt,50,'
        '~json__qc,CRITICAL,'
        'session__extended_qc__behavior,1,'
        'json__extended_qc__tracing_exists,True,'
    )

    if alignment_resolved:
        base_query += 'json__extended_qc__alignment_resolved,True,'
    else:
        base_query += 'json__extended_qc__alignment_count__gt,0,'

    qc_pass = (
        '~session__extended_qc___task_stimOn_goCue_delays__lt,0.9,'
        '~session__extended_qc___task_response_feedback_delays__lt,0.9,'
        '~session__extended_qc___task_wheel_move_before_feedback__lt,0.9,'
        '~session__extended_qc___task_wheel_freeze_during_quiescence__lt,0.9,'
        '~session__extended_qc___task_error_trial_event_sequence__lt,0.9,'
        '~session__extended_qc___task_correct_trial_event_sequence__lt,0.9,'
        '~session__extended_qc___task_reward_volumes__lt,0.9,'
        '~session__extended_qc___task_reward_volume_set__lt,0.9,'
        '~session__extended_qc___task_stimulus_move_before_goCue__lt,0.9,'
        '~session__extended_qc___task_audio_pre_trial__lt,0.9')

    marked_pass = (
        'session__extended_qc___experimenter_task,PASS')

    insertions = list(one.alyx.rest('insertions', 'list', django=base_query + qc_pass))
    insertions.extend(list(one.alyx.rest('insertions', 'list', django=base_query + marked_pass)))

    bwm_df = pd.DataFrame({
        'pid': np.array([i['id'] for i in insertions]),
        'eid': np.array([i['session'] for i in insertions]),
        'probe_name': np.array([i['name'] for i in insertions]),
        'session_number': np.array([i['session_info']['number'] 
                                    for i in insertions]),
        'date': np.array([parser.parse(i['session_info']['start_time']).date() 
                          for i in insertions]),
        'subject': np.array([i['session_info']['subject'] for i in insertions]),
        'lab': np.array([i['session_info']['lab'] for i in insertions]),
    }).sort_values(by=['lab', 'subject', 'date', 'eid'])
    bwm_df.drop_duplicates(inplace=True)
    bwm_df.reset_index(inplace=True, drop=True)

    if return_details:
        return bwm_df, insertions
    else:
        return bwm_df
   

def download_data_only():
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
        


def get_all_psths_ids(split,eids_plus = None, control = False,
                      ind_trials=False):

    '''
    for all BWM insertions, get the PSTHs and acronyms,
    '''
    
    print('split', split)
    
    if eids_plus == None:
        df = bwm_query(one)
        eids_plus = df[['eid','probe_name']].values
        #eids_plus = get_bwm_sessions()

    ws = []
    ss = []
    ids = []
    Fs = []
    eid_probe = []
        
    k=0
    for pid in eids_plus:
        eid, probe = pid
        try:
            if ind_trials:
                w, ac = get_psths_atlasids(split,eid, probe, control=control,
                                              ind_trials=ind_trials)            
            else:        
                w, s, ac = get_psths_atlasids(split,eid, probe, control=control,
                                              ind_trials=ind_trials)
                ss.append(s)
                                              
            ws.append(w)
            ids.append(ac)
            eid_probe.append(eid+'_'+probe)
            gc.collect() 
            print(k, 'of', len(eids_plus), 'ok') 
        except:
            Fs.append(pid)
            gc.collect()
            print(k, 'of', len(eids_plus), 'fail')
        k+=1            
        
    R = {'ws':ws, 'ss': ss, 'ids':ids, 'eid_probe':eid_probe} 
    
    np.save('/home/mic/paper-brain-wide-map/manifold_analysis/'
           f'bwm_psths_{split}.npy', R, allow_pickle=True) 

    print(f'{len(Fs)}, load failures:')
    return Fs

    
def raw_d_stacked(split, mapping='Swanson', control=False,
                  ind_trials=False):
                  

    time0 = time.perf_counter()

    '''
    stack psths of all sessions
    for each region, 
    get Euclidean dist curves (dist_0, dist_choice)
    and get 3d PCA trajectory
    mapping in Cosmos, Beryl, Swanson
    
    '''
    # minimum variance explained ratio when picking PC dim 
    minvar = 0.95  

    
    print(split)
    
    R = np.load('/home/mic/paper-brain-wide-map/'
                f'manifold_analysis/bwm_psths_{split}.npy',
                allow_pickle=True).flat[0]

    
    # concatenate sessions   
    if not ind_trials:
        nt, nclus, nobs = R['ws'][0].shape
        ws = [np.concatenate([R['ws'][k][i] for k in range(len(R['ws']))])
              for i in range(nt)]   

        ss = [np.concatenate([R['ss'][k][i] for k in range(len(R['ws']))])
              for i in range(nt)] 
    
    ids = np.concatenate([R['ids'][k] for k in range(len(R['ws']))])
    
    # mapp atlas ids to acronyms                       
    acs = br.id2acronym(ids,mapping=mapping)            
    acs = np.array(acs)

    regs = Counter(acs)

    print(len(regs), 'regions pre filter')
    print(regs)
    
    regs1 = {}
    for reg in regs:
        if reg in ['void','root']:
            continue
        if regs[reg] < min_reg:
            continue
        regs1[reg] = regs[reg] 
    
    regs = regs1   
    
    print(len(regs), 'regions post filter')
    
    if control:
        nshufs = 20 + 1
    else:
        nshufs = 1            
        
    M = {}
    for j in range(nshufs):
    
        if j != 0:
            # shuffle region label list
            random.shuffle(acs)    

        D = {}
        
        for reg in regs:

            
            print(reg)
            res = {}

            if ind_trials:

                # get all trials of all cells in region
                ws0 = []  # trials for trial condition 0
                ws1 = []  # trials for trial condition 1
                
                discard = []
                for k in range(len(R['ws'])):
                    acs0 = br.id2acronym(R['ids'][k],mapping=mapping)
                    if sum(acs0 == reg) == 0:
                        continue
                    
                    a = R['ws'][k][0][:,acs0 == reg]
                    b = R['ws'][k][1][:,acs0 == reg]
                    
                    # match trial numbers per session
                    n_t = np.min([a.shape[0], b.shape[0]])
                    
                    # keep track of # discarded trials
                    discard.append(abs(a.shape[0] - b.shape[0])/
                                  np.max([a.shape[0],b.shape[0]]))
                    
                    a = a[:n_t]
                    b = b[:n_t] 
                    
                    a = np.reshape(a,(a.shape[0]*a.shape[1], a.shape[2]))
                    b = np.reshape(b,(b.shape[0]*b.shape[1], b.shape[2]))
                    
                    ws0.append(a)
                    ws1.append(b)
                
                # reshape to have trials x obs
                ws_ = [np.array(np.concatenate(ws0)),
                       np.array(np.concatenate(ws1))]
                wsc = np.concatenate(ws_,axis=1)
                
                res['nclus'] = [discard, wsc.shape[0]]
      
            else:
           
                ws_ = [y[acs == reg] for y in ws]
                ss_ = [y[acs == reg] for y in ss]
             
                wsc = np.concatenate(ws_,axis=1)
                ssc = np.concatenate(ss_,axis=1)

                #Discard cells that have nan entries in the PETH or PETH = 0 for all t
                respcells = [k for k in range(wsc.shape[0]) if 
                             (not np.isnan(wsc[k]).any()
                             and wsc[k].any())] 
                        
                wsc = wsc[respcells]
                ws_ = [x[respcells] for x in ws_]
                
                ssc = ssc[respcells]
                ss_ = [x[respcells] for x in ss_]        

                res['nclus'] = [regs[reg], len(respcells)]


#            if split == 'stim':
#                a,b = 4,9
#            else:
            a,b = 0,1
            
            # save average PETH
            res['avg_PETH_a'] = np.nanmean(ws_[a],axis=0)
            res['avg_PETH_b'] = np.nanmean(ws_[b],axis=0)
            
            ncells, nobs = ws_[a].shape
            nt = len(ws_)
            
            if not ind_trials:           
                # get dist_split, i.e. dist of corresponding psth points  

#                # vanilla Euclidean distance
#                res['d_euc'] = np.sum((ws_[0] - ws_[1])**2,axis=0)**0.5
#                res['max_d_euc'] = np.max(res['d_euc'])
                
                # strictly standardized mean difference
                d_var = ((ws_[0] - ws_[1])/((ss_[0] + ss_[1])**0.5))**2

                # square the take root to make a metric
                res['d_var_m'] = np.nanmean(d_var,axis=0)**0.5  #average over cells
                
                # d_var can be negative or positive, finding largest magnitude
                res['max_d_var_m'] = np.max(res['d_var_m'])     

            if not control:
                if c != None:
                    # save first 6 PCs for illustration
                    pca = PCA()
                    wsc  = pca.fit_transform(wsc.T)
                    res['pca_tra'] = wsc[:,:6] 

                    pcadim = np.where(np.cumsum(pca.explained_variance_ratio_)
                                                > minvar)[0][0]
                                                
                    # reduce prior to distance computation
                    wsc = wsc[:,:pcadim].T  # nPCs x nobs, trial means          
                    
                    # split it up again into trajectories after PCA        
                    ws_ = [wsc[:,u * nobs: (u + 1) * nobs] for u in range(nt)]

                    # PCA Euclidean distance
                    res['d_euc_pca'] = np.sum((ws_[0] - ws_[1])**2,axis=0)**0.5
                    res['max_d_euc_pca'] = np.max(res['d_euc_pca'])
                    
            D[reg] = res
        M[j] = D 
    np.save('/home/mic/paper-brain-wide-map/manifold_analysis/'
           f'curves_{split}_mapping{mapping}.npy', 
           M, allow_pickle=True)
           
    time1 = time.perf_counter()
    
    print(time1 - time0, 'sec')


def curves_params_all():

    for split in align:
        get_all_psths_ids(split,eids_plus = None, control = False,
                      ind_trials=True)
        raw_d_stacked(split, mapping='Swanson',control=True)        

      
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


def reformat_res(mapping='Swanson'):


    for split in align:
    
        d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'
                    f'curves_{split}_mapping{mapping}.npy',
                    allow_pickle=True).flat[0]

#        if split == 'stim':
#            a,b = 4,9
#        else:
        a,b = 0,1

        for reg in d:
            # save average PETH
            d[reg]['avg_PETH_a'] = d[reg]['avg_PETH'][a]
            d[reg]['avg_PETH_b'] = d[reg]['avg_PETH'][b]
            d[reg]['max_avg_PETH_a'] = np.max(d[reg]['avg_PETH'][a])
            d[reg]['max_avg_PETH_b'] = np.max(d[reg]['avg_PETH'][b])
            
        np.save('/home/mic/paper-brain-wide-map/manifold_analysis/'
               f'curves_{split}_mapping{mapping}.npy', 
               d, allow_pickle=True)            
                    


def plot_all(curve = 'd_var_m'):

    '''
    First row all example region PCs
    Second row PCs for top region, by top curve max
    Third row curves top 5 last 5 for max curve metric 
    Fourth row being max curve mapped on Swanson
    
    curve in ['d_euc', 'd_var_m', 'd_var_s', 'avg_PETH_a', 'avg_PETH_b']

    multi 3d
    ''' 
        
    mapping='Swanson'
    reg = 'CP'
    
    fig = plt.figure(figsize=(12, 10))
    dfa, palette = get_allen_info()
    
    font = {'size'   : 8}
    mpl.rc('font', **font)

    
    cbar_types = {0:'Blues', 1:'Reds'} | {j:'Greys' 
                  for j in range(2,nrand*2+2)}                  

    axs = []
    axsi = []
    
    k = 0
#    for split in align:

#        axs.append(fig.add_subplot(4, len(align), k + 1, projection='3d'))

#         
#        f = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'
#                    f'curves_{split}_mapping{mapping}.npy',
#                    allow_pickle=True).flat[0]

#        # get Allen colors and info
#        
#        regs = list(Counter(f.keys()).keys())
#        nobs, npcs = f[regs[0]]['pca_tra'].shape
#        ntra = nobs//len(xs(split))
#           

#        cs = [f[reg]['pca_tra'][len(xs(split))*tra:len(xs(split))*(tra+1)] 
#              for tra in range(ntra)]

#        pp = range(ntra)
#        
#        for j in range(ntra):
#        
#        
#            if split == 'stim':
#                col = []

#                if j < 5:
#                    cmap = (plt.get_cmap('Blues')
#                            (np.linspace(0,1,ntra//2 + 5)))
#                else:
#                    cmap = (plt.get_cmap('Reds')
#                            (np.linspace(0,1,ntra//2 + 5)))                     

#                
#                for i in range(len(xs(split))):
#                    col.append(cmap[j%5 + 5])
#          
#            elif split == 'fback':
#                col = [['g', 'purple'][j]] * len(xs(split))
#                   
#            else:

#                col = [[blue_left, red_right][j]] * len(xs(split))               
#                   
#            p = axs[k].scatter(cs[pp[j]][:,0],cs[pp[j]][:,1],cs[pp[j]][:,3],
#                               color=col, s=20,
#                               label=trial_split[split][pp[j]], 
#                               depthshade=False) 
#            
#        axs[k].set_xlabel('pc1')    
#        axs[k].set_ylabel('pc2')
#        axs[k].set_zlabel('pc3')
#        axs[k].set_title(split)   
#        axs[k].grid(False)
#        axs[k].axis('off') 
#              
#        axs[k].legend(frameon=False).set_draggable(True)
#        
#        # put dist_split as inset        
#        axsi.append(inset_axes(axs[k], width="20%", height="20%", 
#                               loc=4, borderpad=1,
#                               bbox_to_anchor=(-0.1,0.1,1,1), 
#                               bbox_transform=axs[k].transAxes))
#                            
#        xx = xs(split)-pre_post[split][0]

#        axsi[k].plot(xx,f[reg][curve], linewidth = 1,
#                      color=palette[reg], label=f'{curve} {reg}')
#                   
#        axsi[k].spines['right'].set_visible(False)
#        axsi[k].spines['top'].set_visible(False)
#        axsi[k].spines['left'].set_visible(False)
#        axsi[k].set_yticks([])
#        axsi[k].set_xlabel('time [sec]')                         
#        #axsi[k].set_title(reg) 
#        
#                   
#        axsi[k].axvline(x=0, lw=0.5, linestyle='--', c='k',
#                        label = align[split])
#        axsi[k].legend(frameon=False,
#                       loc='upper right', 
#                       bbox_to_anchor=(-0.7, 0.5)).set_draggable(True)
#        k +=1


    ## get top 5 and last 2 max regions for each split
    mapping='Swanson'
    tops = {}
    for split in align:
    
        d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'         
                    f'curves_{split}_mapping{mapping}.npy',
                    allow_pickle=True).flat[0]
             
        maxs = np.array([d[0][x][f'max_{curve}'] for x in d[0]])
        acronyms = np.array([x for x in d[0]])
        order = list(reversed(np.argsort(maxs)))
        maxs = maxs[order]
        acronyms = acronyms[order] 
        
        # get p-values
        ps = {}
        for reg in d[0]:
            max_ = d[0][reg][f'max_{curve}']
            null_ = [d[i][reg][f'max_{curve}'] for i in range(1,len(d))]
            p = 1 - (0.01 * percentileofscore(null_,max_,kind='weak'))
            ps[reg] = p
                        
        tops[split] = [acronyms[:15], [ps[reg] for reg in acronyms[:15]]]
        print(split, curve)
        print(acronyms[:15])
        print(sum([ps[reg] < 0.01 for reg in ps]), len(ps), 
              np.round(sum([ps[reg] < 0.01 for reg in ps])/len(ps),2) )              
        
    return
                      
#    '''
#    multi 3d -- this time with top region only
#    '''     


#    for split in align:

#        axs.append(fig.add_subplot(4, len(align), k + 1, projection='3d'))

#         
#        f = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'
#                    f'curves_{split}_mapping{mapping}.npy',
#                    allow_pickle=True).flat[0]

#        # get Allen colors and info
#        
#        regs = list(Counter(f.keys()).keys())
#        nobs, npcs = f[regs[0]]['pca_tra'].shape
#        ntra = nobs//len(xs(split))
#           
#        reg = tops[split][0]
#        cs = [f[reg]['pca_tra'][len(xs(split))*tra:len(xs(split))*(tra+1)] 
#              for tra in range(ntra)]

#        
#        pp = range(ntra)
#        
#        for j in range(ntra):
#        
#        
#            if split == 'stim':
#                col = []

#                if j < 5:
#                    cmap = (plt.get_cmap('Blues')
#                            (np.linspace(0,1,ntra//2 + 5)))
#                else:
#                    cmap = (plt.get_cmap('Reds')
#                            (np.linspace(0,1,ntra//2 + 5)))                     

#                
#                for i in range(len(xs(split))):
#                    col.append(cmap[j%5 + 5])
#          
#            elif split == 'fback':
#                col = [['g', 'purple'][j]] * len(xs(split))
#                   
#            else:

#                col = [[blue_left, red_right][j]] * len(xs(split))               

#            p = axs[k].scatter(cs[pp[j]][:,0],cs[pp[j]][:,1],cs[pp[j]][:,2],
#                               color=col, s=20,
#                               label=trial_split[split][pp[j]], 
#                               depthshade=False) 
#            
#        axs[k].set_xlabel('pc1')    
#        axs[k].set_ylabel('pc2')
#        axs[k].set_zlabel('pc3')
#        axs[k].set_title(split)   
#        axs[k].grid(False)
#        axs[k].axis('off') 
#              
#        axs[k].legend(frameon=False).set_draggable(True)
#        
#        # put dist_split as inset
#        
#        axsi.append(inset_axes(axs[k], width="20%", height="20%", 
#                               loc=4, borderpad=1,
#                               bbox_to_anchor=(-0.1,0.1,1,1), 
#                               bbox_transform=axs[k].transAxes))
#                            

#        xx = xs(split)-pre_post[split][0]

#        axsi[k].plot(xx,f[reg][curve], linewidth = 1,
#                      color=palette[reg], label=f'{curve} {reg}')
#                   
#        axsi[k].spines['right'].set_visible(False)
#        axsi[k].spines['top'].set_visible(False)
#        axsi[k].spines['left'].set_visible(False)
#        axsi[k].set_yticks([])
#        axsi[k].set_xlabel('time [sec]')                         
#        #axsi[k].set_title(reg) 
#        
#                   
#        axsi[k].axvline(x=0, lw=0.5, linestyle='--', c='k',
#                        label = align[split])
#        axsi[k].legend(frameon=False,
#                       loc='upper right', 
#                       bbox_to_anchor=(-0.7, 0.5)).set_draggable(True)
#        k +=1

    
    '''
    plot average curve per region
    curve types: dist_0, dist_split
    regk is what chunk of Beryl regions to plot
    in groups of consecutive 10
    '''

    regk=0

    for split in align:
        axs.append(fig.add_subplot(4, len(align), k + 1))
        f = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'
                    f'curves_{split}_mapping{mapping}.npy',
                    allow_pickle=True).flat[0][0]
                                 
        regs = tops[split][0]            

        for reg in regs:
            xx = xs(split)-pre_post[split][0]
            
            axs[k].plot(xx,f[reg][curve], linewidth = 2,
                          color=palette[reg], 
                          label=f"{reg}")# [{f[reg]['nclus'][1]}]

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
            texts.append(axs[k].text(x, y, 
                                     f"{reg}",#['nclus'][1] 
                                     color = palette[reg],
                                     fontsize=9))
                                     
        adjust_text(texts)             

        
        axs[k].set_ylabel(curve)
        axs[k].set_xlabel('time [sec]')
        axs[k].set_title(f'{split}')

                
        k +=1
   
    '''
    max dist_split onto swanson flat maps
    '''

    for split in align:
    
        axs.append(fig.add_subplot(4, len(align),k + 1))   
 
        d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'         
                    f'curves_{split}_mapping{mapping}.npy',
                    allow_pickle=True).flat[0][0]


        values = np.array([d[x][f'max_{curve}'] for x in d])
        acronyms = np.array([x for x in d])                  
        plot_swanson(list(acronyms), list(values), cmap='Blues', 
                     ax=axs[k], br=br)#, orientation='portrait')
        axs[k].axis('off')
        axs[k].set_title(f'max_{curve} {split}')
        k += 1


    fig.subplots_adjust(top=0.999,
bottom=0.01,
left=0.057,
right=0.979,
hspace=0.1,
wspace=0.3)
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


def plot_average_PETHs(region, split):

    '''
    plot average PETH per region
    '''
    
    _, palette = get_allen_info()
    
    f = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'
                f'avPETH_{mapping}.npy',
                allow_pickle=True).flat[0][split]
                             
    regs = f.keys()

              
    xx = list(xs(split)-pre_post[split][0])*len(trial_split[split])
    
    
    fig, ax = plt.subplots()
    
    ax.plot(f[reg], linewidth = 2,
                  color=palette[reg], 
                  label=f"{reg}")# [{f[reg]['nclus']}]


    for i in range(len(trial_split[split])):
        ax.axvline(x=i*len(xs(split)) + pre_post[split][0]/T_BIN(split), 
                   lw=0.5, linestyle='--', c='k')
        
                                 
    adjust_text(texts)
    axs[k].set_ylabel(curve)
    axs[k].set_xlabel('time [sec]')
    axs[k].set_title(f'{split}')


def get_tops(split, curve = 'd_var_m'):

    '''
    also get fraction of significant regions
    '''
    
    mapping='Swanson'

    d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/one_bin/'         
                f'curves_{split}_mapping{mapping}.npy',
                allow_pickle=True).flat[0]
    
    maxs = np.array([d[x][f'max_{curve}'] for x in d])
    acronyms = np.array([x for x in d])
    order = list(reversed(np.argsort(maxs)))
    maxs = maxs[order]
    acronyms = acronyms[order] 
    return acronyms[:35]

        
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










