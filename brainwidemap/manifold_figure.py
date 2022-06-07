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

blue_left = [0.13850039, 0.41331206, 0.74052025]
red_right = [0.66080672, 0.21526712, 0.23069468]

T_BIN = 0.005  # time bin size in seconds (5 ms)

align = {'block':'stim on',
         'stim':'stim on',
         'choice':'motion on',
         'action':'motion on',
         'fback':'feedback'}

pre_post = {'choice':[0.1,0],'stim':[0,0.1],
            'fback':[0,0.1],'block':[0.4,0],
            'action':[0.025,0.3]}  #[pre_time, post_time]
            
trial_split = {'choice':['choice left', 'choice right'],
               'stim':[0.0, 0.0625, 0.125, 0.25, 1.0,
                       0.0, 0.0625, 0.125, 0.25, 1.0],
               'fback':['correct','false'],
               'block':['pleft 0.8','pleft 0.2'],
               'action':['choice left', 'choice right']}            
  
nrand = 20  # number of random trial splits for choice control
min_reg = 200  # minimum number of neurons in region
one = ONE() #ONE(mode='local')
ba = AllenAtlas()
br = BrainRegions()

def xs(split):
    return np.arange((pre_post[split][0] + pre_post[split][1])/T_BIN)*T_BIN
      

def distE(x,y):

    '''
    Euclidean distance of two points x, y
    '''    
    return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))


def norm_curve(r,split):

    '''
    normalize curve by pre-stim baseline
    '''
    # normalise via pre-stim baseline
    bsl_m = np.mean(r[0:int(abs(pre_post[split][0])/T_BIN)])
    bsl_st = np.std(r[0:int(abs(pre_post[split][0])/T_BIN)])
    postStim = np.mean(r[int(abs(pre_post[split][0])/T_BIN):])         
    r = (r-bsl_m)/(bsl_m + 0.1)
    # check if mean baseline is above mean after stim
    stim_responsive = True
    if bsl_m > postStim:
       stim_responsive = False
        
    return r, stim_responsive     



def get_psths_atlasids(split,eid, probe, control=False):

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
            for contrast in [0.0, 0.0625, 0.125, 0.25, 1.0]:
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
                           pre_post[split][0], pre_post[split][1], T_BIN)
        bins.append(bi)                   
                                                     
    b = np.concatenate(bins)        
    ntr, nclus, nbins = b.shape

    
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
        


def get_all_psths_ids(split,eids_plus = None, control = False):

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
            w, s, ac = get_psths_atlasids(split,eid, probe, control=control)
            ws.append(w)
            ss.append(s)
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

    
def raw_d_stacked(split, mapping='Swanson', control=False):

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

    nt, nclus, nobs = R['ws'][0].shape
    
    # concatenate sessions
    ws = [np.concatenate([R['ws'][k][i] for k in range(len(R['ws']))])
          for i in range(nt)]   
    ids = np.concatenate([R['ids'][k] for k in range(len(R['ws']))])
    
    ss = [np.concatenate([R['ss'][k][i] for k in range(len(R['ws']))])
          for i in range(nt)] 
    
    # mapp atlas ids to acronyms                       
    acs = br.id2acronym(ids,mapping=mapping)            
    acs = np.array(acs)
    
    # get psths per region, then reduce to 3 dims via PCA
    regs = Counter(acs)
    print(regs)
    
    D = {}
    
    for reg in regs:
        if reg in ['void','root']:
            continue
        if sum(acs == reg) < min_reg:
            continue
        
        print(reg)
        res = {}
        
        ws_ = [y[acs == reg] for y in ws]
        ss_ = [y[acs == reg] for y in ss]
     
        wsc = np.concatenate(ws_,axis=1)
        ssc = np.concatenate(ss_,axis=1)

        # save average PETH
        res['avg_PETH'] = [np.nanmean(x,axis=0) for x in ws_]

        #Discard cells that have nan entries in the PETH or PETH = 0 for all t
        respcells = [k for k in range(wsc.shape[0]) if (not np.isnan(wsc[k]).any()
                                                        and wsc[k].any())] 
        
        wsc = wsc[respcells]
        ws_ = [x[respcells] for x in ws_]
        
        ssc = ssc[respcells]
        ss_ = [x[respcells] for x in ss_]        

        res['nclus'] = [regs[reg], len(respcells)]

        if control:
            n_neus, n_obs = wsc.shape 
            ns = random.sample(range(n_neus), 2*n_neus//3)        
            wsc = wsc[ns]
            
        # save first 6 PCs for illustration
        pca = PCA()
        res['pca_tra'] = pca.fit_transform(wsc.T)[:,:6] 

#        pcadim = np.where(np.cumsum(pca.explained_variance_ratio_)
#                                    > minvar)[0][0]
#        # reduce prior to distance computation
#        wsc = wsc[:,:pcadim].T  # nPCs x nobs, trial means          
#        ssc = ssc[:,:pcadim].T  # nPCs x nobs, trial vars
#        
#        # split it up again into trajectories after PCA
#        
#        ws_ = [wsc[:,u * nobs: (u + 1) * nobs] for u in range(nt)]
#        ss_ = [ssc[:,u * nobs: (u + 1) * nobs] for u in range(nt)]        
        
        if split == 'stim':
            a,b = 4,9
        else:
            a,b = 0,1
           
        # get dist_split, i.e. dist of corresponding psth points  
        ncells, nobs = ws_[a].shape

        # vanilla Euclidean distance
        res['d_euc'] = np.array([distE(ws_[a][:,i], ws_[b][:,i]) 
                                 for i in range(nobs)])
                                 
        res['max_d_euc'] = np.max(res['d_euc'])

        # strictly standardized mean difference
        d_var = np.array([[(ws_[a][j,i] - ws_[b][j,i])/
                          (ss_[a][j,i] + ss_[b][j,i])**0.5 
                          for i in range(nobs)] for j in range(ncells)])
        
        d_var = d_var**2
        # square the take root to make a metric
        res['d_var_m'] = np.nanmean(d_var,axis=0)**0.5  #average over cells
        res['d_var_s'] = np.nansum(d_var,axis=0)**0.5
        
        # d_var can be negative or positive, finding largest magnitude
        res['max_d_var_m'] = np.max(res['d_var_m'])
        res['max_d_var_s'] = np.max(res['d_var_s'])      

        D[reg] = res
 
    np.save('/home/mic/paper-brain-wide-map/manifold_analysis/'
           f'curves_{split}_mapping{mapping}.npy', 
           D, allow_pickle=True)


def curves_params_all():

    for split in align:
        raw_d_stacked(split, mapping='Swanson')        

      
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


def plot_all(curve = 'd_var_m'):

    '''
    First row all example region PCs
    Second row PCs for top region, by top curve max
    Third row curves top 5 last 5 for max curve metric 
    Fourth row being max curve mapped on Swanson
    
    curve in ['d_euc', 'd_var_m', 'd_var_s']

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
    for split in align:

        axs.append(fig.add_subplot(4, len(align), k + 1, projection='3d'))

         
        f = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'
                    f'curves_{split}_mapping{mapping}.npy',
                    allow_pickle=True).flat[0]

        # get Allen colors and info
        
        regs = list(Counter(f.keys()).keys())
        nobs, npcs = f[regs[0]]['pca_tra'].shape
        ntra = nobs//len(xs(split))
           

        cs = [f[reg]['pca_tra'][len(xs(split))*tra:len(xs(split))*(tra+1)] 
              for tra in range(ntra)]

        pp = range(ntra)
        
        for j in range(ntra):
        
        
            if split == 'stim':
                col = []

                if j < 5:
                    cmap = (plt.get_cmap('Blues')
                            (np.linspace(0,1,ntra//2 + 5)))
                else:
                    cmap = (plt.get_cmap('Reds')
                            (np.linspace(0,1,ntra//2 + 5)))                     

                
                for i in range(len(xs(split))):
                    col.append(cmap[j%5 + 5])
          
            elif split == 'fback':
                col = [['g', 'purple'][j]] * len(xs(split))
                   
            else:

                col = [[blue_left, red_right][j]] * len(xs(split))               
                   
            p = axs[k].scatter(cs[pp[j]][:,0],cs[pp[j]][:,1],cs[pp[j]][:,3],
                               color=col, s=20,
                               label=trial_split[split][pp[j]], 
                               depthshade=False) 
            
        axs[k].set_xlabel('pc1')    
        axs[k].set_ylabel('pc2')
        axs[k].set_zlabel('pc3')
        axs[k].set_title(split)   
        axs[k].grid(False)
        axs[k].axis('off') 
              
        axs[k].legend(frameon=False).set_draggable(True)
        
        # put dist_split as inset        
        axsi.append(inset_axes(axs[k], width="20%", height="20%", 
                               loc=4, borderpad=1,
                               bbox_to_anchor=(-0.1,0.1,1,1), 
                               bbox_transform=axs[k].transAxes))
                            
        xx = xs(split)-pre_post[split][0]

        axsi[k].plot(xx,f[reg][curve], linewidth = 1,
                      color=palette[reg], label=f'{curve} {reg}')
                   
        axsi[k].spines['right'].set_visible(False)
        axsi[k].spines['top'].set_visible(False)
        axsi[k].spines['left'].set_visible(False)
        axsi[k].set_yticks([])
        axsi[k].set_xlabel('time [sec]')                         
        #axsi[k].set_title(reg) 
        
                   
        axsi[k].axvline(x=0, lw=0.5, linestyle='--', c='k',
                        label = align[split])
        axsi[k].legend(frameon=False,
                       loc='upper right', 
                       bbox_to_anchor=(-0.7, 0.5)).set_draggable(True)
        k +=1


    ## get top 5 and last 2 max regions for each split
    mapping='Swanson'
    tops = {}
    for split in align:
    
        d = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'         
                    f'curves_{split}_mapping{mapping}.npy',
                    allow_pickle=True).flat[0]
        
        maxs = np.array([d[x][f'max_{curve}'] for x in d])
        acronyms = np.array([x for x in d])
        order = list(reversed(np.argsort(maxs)))
        maxs = maxs[order]
        acronyms = acronyms[order] 
                        
        tops[split] = np.concatenate([acronyms[:5],acronyms[-3:]])
        print(split, curve)
        print(tops[split])
                       
                      
    '''
    multi 3d -- this time with top region only
    '''     


    for split in align:

        axs.append(fig.add_subplot(4, len(align), k + 1, projection='3d'))

         
        f = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'
                    f'curves_{split}_mapping{mapping}.npy',
                    allow_pickle=True).flat[0]

        # get Allen colors and info
        
        regs = list(Counter(f.keys()).keys())
        nobs, npcs = f[regs[0]]['pca_tra'].shape
        ntra = nobs//len(xs(split))
           
        reg = tops[split][0]
        cs = [f[reg]['pca_tra'][len(xs(split))*tra:len(xs(split))*(tra+1)] 
              for tra in range(ntra)]

        
        pp = range(ntra)
        
        for j in range(ntra):
        
        
            if split == 'stim':
                col = []

                if j < 5:
                    cmap = (plt.get_cmap('Blues')
                            (np.linspace(0,1,ntra//2 + 5)))
                else:
                    cmap = (plt.get_cmap('Reds')
                            (np.linspace(0,1,ntra//2 + 5)))                     

                
                for i in range(len(xs(split))):
                    col.append(cmap[j%5 + 5])
          
            elif split == 'fback':
                col = [['g', 'purple'][j]] * len(xs(split))
                   
            else:

                col = [[blue_left, red_right][j]] * len(xs(split))               

            p = axs[k].scatter(cs[pp[j]][:,0],cs[pp[j]][:,1],cs[pp[j]][:,2],
                               color=col, s=20,
                               label=trial_split[split][pp[j]], 
                               depthshade=False) 
            
        axs[k].set_xlabel('pc1')    
        axs[k].set_ylabel('pc2')
        axs[k].set_zlabel('pc3')
        axs[k].set_title(split)   
        axs[k].grid(False)
        axs[k].axis('off') 
              
        axs[k].legend(frameon=False).set_draggable(True)
        
        # put dist_split as inset
        
        axsi.append(inset_axes(axs[k], width="20%", height="20%", 
                               loc=4, borderpad=1,
                               bbox_to_anchor=(-0.1,0.1,1,1), 
                               bbox_transform=axs[k].transAxes))
                            

        xx = xs(split)-pre_post[split][0]

        axsi[k].plot(xx,f[reg][curve], linewidth = 1,
                      color=palette[reg], label=f'{curve} {reg}')
                   
        axsi[k].spines['right'].set_visible(False)
        axsi[k].spines['top'].set_visible(False)
        axsi[k].spines['left'].set_visible(False)
        axsi[k].set_yticks([])
        axsi[k].set_xlabel('time [sec]')                         
        #axsi[k].set_title(reg) 
        
                   
        axsi[k].axvline(x=0, lw=0.5, linestyle='--', c='k',
                        label = align[split])
        axsi[k].legend(frameon=False,
                       loc='upper right', 
                       bbox_to_anchor=(-0.7, 0.5)).set_draggable(True)
        k +=1


    
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
                    allow_pickle=True).flat[0]
                                 
        regs = tops[split]            

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
                    allow_pickle=True).flat[0]


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
    

def swanson_gif(split, recompute=True):

    '''
    use dist_split(t) to make swanson plots for each t
    then combine to GIF
    
    split in stim, action
    '''

    mapping = 'Swanson'
    pcadim = 20      
    
    f = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'
                f'curves_{split}_mapping{mapping}.npy',
                allow_pickle=True).flat[0]                

    T = []
    for t in range(int((pre_post[split][0] + pre_post[split][1])//T_BIN)):
        acronyms = []
        values = []
        for reg in f:
            values.append(f[reg]['dist_split'][t])
            acronyms.append(reg)
        T.append([acronyms, values])
        
    all_vals = np.concatenate([x[1] for x in T])
    vmin = min(all_vals)
    vmax = max(all_vals)
    
    plt.ioff()
    
    s0 = '/home/mic/paper-brain-wide-map/manifold_analysis/gif'
    Path(s0+f'/{split}').mkdir(parents=True, exist_ok=True)
    
    if recompute:
        for t in range(int((pre_post[split][0] + pre_post[split][1])//T_BIN)):
            acronyms = T[t][0]
            values = T[t][1]    
            fig, ax = plt.subplots(figsize=(15,7))
            plot_swanson(acronyms, values, cmap='Blues', 
                         ax=ax, br=br, vmin=vmin, vmax=vmax, annotate=True)
                         
            ax.set_title(f'split {split}; t = {t*T_BIN} sec')             
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
        ax.axvline(x=i*len(xs(split)) + pre_post[split][0]/T_BIN, 
                   lw=0.5, linestyle='--', c='k')
        
                                 
    adjust_text(texts)
    axs[k].set_ylabel(curve)
    axs[k].set_xlabel('time [sec]')
    axs[k].set_title(f'{split}')



#def regional_plots(split, plot_type='bar', pcadim=20, ax = None, fig = None,
#                   mapping = 'Cosmos', umap_=False, reg_id=None, restrict=True,
#                   p_filter=False):

#             
#    df = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'         
#                f'curve_params_split{split}_mapping{mapping}_pcadim{pcadim}.pkl',    
#                allow_pickle=True)
#    
#    print(df.keys())                

#    df = df[['max','acronym']]
#    
#    if plot_type == 'flatmap':

#        values_lh = np.array(df['max'].values)
#        values_rh = np.array([np.nan]*len(values_lh))
#        acronyms_lh = np.array(df['acronym'].values)       
#        acronyms_rh = np.array([np.nan]*len(values_lh))
#                
#        acronyms_lr, values_lr = prepare_lr_data(acronyms_lh, values_lh, 
#                                                 acronyms_rh, values_rh)

#        flmap_py = FlatMap(flatmap='dorsal_cortex', res_um=25)

#        flmap_py.plot_flatmap(ax=ax)

#        fig, ax = plot_scalar_on_flatmap(acronyms_lh, values_lh,
#                                         hemisphere='left',
#                                         mapping=mapping, 
#                                         flmap_atlas=flmap_py, ax=ax,
#                                         background='boundary')
#        ax.axis('off')                                
#                                         

#    elif plot_type == 'slices':
#    
#        values_lh = np.array(df['max'].values,dtype='<U10')
#        values_rh = np.array([np.nan]*len(values_lh),dtype='<U10')
#        acronyms_lh = np.array(df['acronym'].values,dtype='<U10')       
#        acronyms_rh = np.array([np.nan]*len(values_lh),dtype='<U10')
#                
#        acronyms, values = prepare_lr_data(acronyms_lh, values_lh, 
#                                                 acronyms_rh, values_rh)
#        
#        vmin, vmax = np.nanmin(values), np.nanmax(values)
#        
#        fig, axs = plt.subplots(nrows=3, ncols=1, 
#                                figsize=(3.5, 7))        
#                                                         
#        
#        _, im0 = plot_scalar_on_slice(acronyms, values, 
#                             slice='top', mapping=mapping, 
#                             hemisphere='left',background='boundary', 
#                             cmap='Reds', brain_atlas=ba, ax=axs[0])
#                         
#        im0.axis('off')
#        _, im1 = plot_scalar_on_slice(acronyms, values, coord=-1000, 
#                                       slice='coronal', mapping=mapping, 
#                                       hemisphere='left', background='boundary', 
#                                       cmap='Reds', brain_atlas=ba, ax=axs[1])
#                                       
#                                       
#        im1.axis('off')         
#        _, im2 = plot_scalar_on_slice(acronyms, values, coord=-1000, 
#                             slice='sagittal', mapping=mapping, 
#                             hemisphere='left',background='boundary', 
#                             cmap='Reds', brain_atlas=ba, ax=axs[2])
#                             
#        im2.axis('off')                                   
#        plt.tight_layout()                     
#        plt.show()
#        
#        
#    elif plot_type == 'swanson':
#        assert mapping == 'Swanson', 'wrong mapping'
#        fig, ax = plt.subplots(figsize=(2,3))
#        values = np.array(df['max'].values)
#        acronyms = np.array(df['acronym'].values)                    
#        plot_swanson(list(acronyms), list(values), cmap='Blues', 
#                     ax=ax, br=br, orientation='portrait')
#        ax.set_title(f'max split {split}')             
#        ax.axis('off') 
#    
#    elif plot_type == 'bar':
#        dfa, palette = get_allen_info()

#        # ignore regions that don't match between Beryl and Allen atlas
#        a = set(df['acronym'].values)
#        b = set(dfa['acronym'].values)
#        overlap = a.intersection(b)        
#        dfo = df[df["acronym"].isin(overlap)]
#        
#        sorter = dfa['acronym'].values
#        sorterIndex = dict(zip(sorter, range(len(sorter))))
#        dfo['acronym_rank'] = dfo['acronym'].map(sorterIndex)
#        dfo.sort_values(['acronym_rank'],
#                ascending = True, inplace = True)
#        dfo.drop('acronym_rank', 1, inplace = True)
#        palette = {your_key: palette[your_key] for 
#                   your_key in dfo['acronym'].values }


#        # remove 'void', 'root'
#        dfo.drop(dfo[np.logical_or((dfo.acronym == 'void'),
#                                   (dfo.acronym == 'root'))].index, 
#                                   inplace = True)
#        print('acronyms void and root dropped') 
#         
#        if restrict:
#            #  sort data by 'max', only show 15 regions with
#            # highest (lowest) value
#            dfo.sort_values('max',inplace=True)
#            if ax is None:
#                fig, axs = plt.subplots(nrows=2, ncols=1, 
#                                        sharey=True, figsize=(3, 3))         
#        
#            print(len(dfo))
#            # split data in two for better visibility
#            dfs = [dfo.head(15),dfo.tail(15)]
#            
#            for k in range(2):            
#                sns.barplot(data = dfs[k],x='acronym',y='max',ax=axs[k],
#                           palette=palette,estimator=np.mean,errwidth=.5, ci=68) 
#               
#                for item in axs[k].get_xticklabels():
#                    item.set_rotation(90)
#                    item.set_fontsize(10)
#                    

#        elif mapping == 'Cosmos':
#            if ax is None:
#                fig, ax = plt.subplots(figsize=(8, 4))        

#            sns.swarmplot(data=dfo, x="acronym", y='max', 
#                          color='k', size=0.5,ax=ax)
#            bp = sns.barplot(data = dfo,x='acronym',y='max',ax=ax, capsize=0.2,
#                             palette=palette,estimator=np.mean,errwidth=.5) 

#            #return bp, dfo
##            # put number of regional measurements on top of bars
##            bp.bar_label(dfo.groupby('acronym').count()) 
#             
#            for item in ax.get_xticklabels():
#                item.set_rotation(90)
#                item.set_fontsize(10)            


#        elif reg_id is not None:
#            if ax is None:
#                fig, ax = plt.subplots(figsize=(8, 4))        

#            # Isocortex: 315, HPF: 1089, MB: 313, TH: 549
#            # dfa[dfa['acronym']=='HPF']['id'].values[0]

#            # just keep regions that have structure id of reg on left
#            acs_ids = []
#            k = 0 
#            for h in dfa.structure_id_path.values: 
#                try:
#                    if str(reg_id) in h: 
#                        hl = h.split('/')
#                        acs_ids.append(hl[-2])
#                except:
#                    pass
#                k += 1
#                
#            acs = [(dfa[dfa['id']==int(x)]['acronym'].values)[0] 
#                    for x in acs_ids]  
#              
#            a = set(dfo['acronym'].values)
#            b = set(acs)
#            overlap = a.intersection(b)        
#            dfo = dfo[dfo["acronym"].isin(overlap)]            
#            
#            
#            print('void and root dropped')    
#            sns.swarmplot(data=dfo, x="acronym", y='max', 
#                          color='k', size=0.5,ax=ax)        
#            bp = sns.barplot(data = dfo,x='acronym',y='max',ax=ax, capsize=0.2,
#                             palette=palette,estimator=np.mean,errwidth=.5) 

#            #return bp, dfo
##            # put number of regional measurements on top of bars
##            bp.bar_label(dfo.groupby('acronym').count()) 
#             
#            for item in ax.get_xticklabels():
#                item.set_rotation(90)
#                item.set_fontsize(10)
#                   
#            ax.set_title(dfa[dfa['id']==int(reg_id)]['acronym'].values[0])

#        else:
#            if ax is None:
#                fig, axs = plt.subplots(nrows=3, ncols=1, 
#                                        sharey=True, figsize=(15, 10))         
#        
#            print(len(dfo))
#            # split data in two for better visibility
#            dfs = [dfo.iloc[:len(dfo)//3], 
#                   dfo.iloc[(len(dfo)//3):(2*len(dfo)//3)],
#                   dfo.iloc[2*len(dfo)//3:]]
#            for k in range(3):            
#                sns.barplot(data = dfs[k],x='acronym',y='max',ax=axs[k],
#                           palette=palette,estimator=np.mean,errwidth=.5, ci=68) 
#               
#                for item in axs[k].get_xticklabels():
#                    item.set_rotation(90)
#                    item.set_fontsize(10)
#        
#    fig.suptitle(f'max of {split}')
#    fig.canvas.manager.set_window_title(f'{split}_max')  
#    fig.tight_layout()
#    fig.savefig('/home/mic/paper-brain-wide-map/'
#               f'overleaf_figs/manifold/plots/'
#               f'{split}_{plot_type}.png')
#            
#    plt.close()        
