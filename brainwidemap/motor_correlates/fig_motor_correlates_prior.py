#from oneibl.one import ONE
from one.api import ONE
from brainbox.io.one import load_channel_locations 
import brainbox.behavior.wheel as wh
from brainbox.processing import bincount2D
from ibllib.atlas import regions_from_allen_csv
import ibllib.atlas as atlas
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
from brainbox.io.one import SpikeSortingLoader
from brainbox.io.one import SessionLoader
from brainwidemap import bwm_query, load_good_units, \
    load_trials_and_mask, filter_regions, filter_sessions, \
    download_aggregate_tables
    
import numpy as np
from pathlib import Path
from collections import Counter
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr
from copy import deepcopy
import pandas as pd
import random
import seaborn as sns
import matplotlib as mpl

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.stats import zscore
import itertools
from mpl_toolkits.mplot3d import Axes3D
import os, sys
from scipy.interpolate import interp1d
import matplotlib
from scipy import stats
from scipy.stats import percentileofscore

matplotlib.rcParams.update({'font.size': 10})


one = ONE()
ba = AllenAtlas()
br = BrainRegions()


T_BIN = 0.02
Fs = {'left':60,'right':150, 'body':30}

# specify binning type, either bins or sampling rate; see cut_gahavior for defs
sr = {'licking':'T_BIN','whisking_l':60, 'whisking_r':150, 
      'wheeling':'T_BIN','nose_pos':60, 'paw_pos_r':150, 
      'paw_pos_l':60}

blue_left = [0.13850039, 0.41331206, 0.74052025]
red_right = [0.66080672, 0.21526712, 0.23069468]
cdi = {0.8:blue_left,0.2:red_right,0.5:'g',-1:'cyan',1:'orange'}



def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__



def get_all_sess_with_ME():
    one = ONE()
    # get all bwm sessions with dlc
    all_sess = one.alyx.rest('sessions', 'list', 
                              project='ibl_neuropixel_brainwide_01',
                              task_protocol="ephys", 
                              dataset_types='camera.ROIMotionEnergy')

    eids = [s['url'].split('/')[-1] for s in all_sess]
    
    return eids   


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



def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
 
 
def get_licks(dlc):

    '''
    define a frame as a lick frame if
    x or y for left or right tongue point
    change more than half the sdt of the diff
    '''  
    
    licks = []
    for point in ['tongue_end_l', 'tongue_end_r']:
        for co in ['x', 'y']:
           c = dlc[point+'_'+co]
           thr = np.nanstd(np.diff(c))/4
           licks.append(set(np.where(abs(np.diff(c))>thr)[0]))
    return sorted(list(set.union(*licks)))         


def get_ME(eid, video_type, query_type='remote'):

    #video_type = 'left'           
    Times = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.times.npy',
                             query_type=query_type) 
    ME = one.load_dataset(eid,f'alf/{video_type}Camera.ROIMotionEnergy.npy', 
                          query_type=query_type)

    return Times, ME  


def get_dlc_XYs(eid, video_type, query_type='remote'):

    #video_type = 'left'    
    Times = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.times.npy',
                             query_type=query_type) 
    cam = one.load_dataset(eid,f'alf/_ibl_{video_type}Camera.dlc.pqt', 
                           query_type=query_type)
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])

    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        x = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array(
            [x, y])    

    return Times, XYs      



def cut_behavior(eid, duration =0.4, lag = -0.6, 
                 align='stimOn_times', stim_to_stim=False, 
                 endTrial=False, query_type='remote',pawex=False):
                 
    '''
    cut segments of behavioral time series for PSTHs
    
    param: eid: session eid
    param: align: in stimOn_times, firstMovement_times, feedback_times    
    param: lag: time in sec wrt to align time to start segment
    param: duration: length of cut segment in sec 
    '''
    sess_loader = SessionLoader(one, eid)

    # get wheel speed
    sess_loader.load_wheel()    
    wheel = sess_loader.wheel 
    
    # load whisker motion energy, separate for both cams
    sess_loader.load_motion_energy(views=['left', 'right'])
    left_whisker = sess_loader.motion_energy['leftCamera']
    right_whisker = sess_loader.motion_energy['rightCamera']
    
    # load DLC
    sess_loader.load_pose(views=['left', 'right'])
    dlc_left = sess_loader.pose['leftCamera']
    dlc_right = sess_loader.pose['rightCamera']
    
    # get licks using both cameras    
    lick_times = []
    for dlc in [dlc_left, dlc_right]:
        r = get_licks(dlc)           
        lick_times.append(dlc['times'][r])
    
    # combine left/right video licks and bin
    lick_times = sorted(np.concatenate(lick_times))
    R, times_lick, _ = bincount2D(lick_times, np.ones(len(lick_times)), T_BIN)
    lcs = R[0]    

    # get paw position, for each cam separate
    if pawex:
        paw_pos_r0 = list(zip(dlc_right['paw_r_x'],dlc_right['paw_r_y']))
        paw_pos_l0 = list(zip(dlc_left['paw_r_x'],dlc_left['paw_r_y'])) 
    else:
        paw_pos_r0 = (dlc_right['paw_r_x']**2 + dlc_right['paw_r_y']**2)**0.5
        paw_pos_l0 = (dlc_left['paw_r_x']**2 + dlc_left['paw_r_y']**2)**0.5
 

    licking = []
    whisking_l = []
    whisking_r = []
    wheeling = [] 
    nose_pos = []
    paw_pos_r = []
    paw_pos_l = []
        
    DD = []
           
    pleft = []
    sides = []
    choices = []
    T = [] 


    d = (licking, whisking_l, whisking_r, wheeling,
         nose_pos, paw_pos_r, paw_pos_l,
         pleft, sides, choices, T)

    ds = ('licking','whisking_l', 'whisking_r', 'wheeling',
         'nose_pos', 'paw_pos_r', 'paw_pos_l',
         'pleft', 'sides', 'choices', 'T')
         
    D = dict(zip(ds,d))
    
    # continuous time series of behavior and stamps
    behaves = {'licking':[times_lick, lcs],
               'whisking_l':[left_whisker['times'],
                             left_whisker['whiskerMotionEnergy']],
               'whisking_r':[right_whisker['times'],
                             right_whisker['whiskerMotionEnergy']],
               'wheeling':[wheel['times'], abs(wheel['velocity'])],
               'nose_pos':[dlc_left['times'], dlc_left['nose_tip_x']],
               'paw_pos_r':[dlc_right['times'],paw_pos_r0], 
               'paw_pos_l':[dlc_left['times'],paw_pos_l0]}


    trials, mask = load_trials_and_mask(one, eid)
            
    kk = 0     

    for tr in range(1,len(trials) - 1):
        
        if not mask[tr]:
            continue
        # skip block boundary trials
        if trials['probabilityLeft'][tr] != trials['probabilityLeft'][tr+1]:
            continue
            
        start_t = trials[align][tr] + lag     
                
        if np.isnan(trials['contrastLeft'][tr]):
            cont = trials['contrastRight'][tr]            
            side = 0  # right side stimulus
        else:   
            cont = trials['contrastLeft'][tr]         
            side = 1  # left side stimulus                   
                  
        sides.append(side) 
        
        if endTrial:
            choices.append(trials['choice'][tr+1])
        else:                              
            choices.append(trials['choice'][tr])   
                 
        pleft.append(trials['probabilityLeft'][tr])

        for be in behaves: 
            times = behaves[be][0] 
            series = behaves[be][1] 
            start_idx = find_nearest(times,start_t)        
            if stim_to_stim:  
                end_idx = find_nearest(times, trials['stimOn_times'][tr + 1])
            else:
                if sr[be] == 'T_BIN':
                    end_idx = start_idx + int(duration/T_BIN)
                else:
                    fs = sr[be]
                    end_idx = start_idx + int(duration*fs)              

            
            if (pawex and ('paw' in be)): #for illustration on frame
                D[be].append([series[0][start_idx:end_idx],
                              series[1][start_idx:end_idx]])            
            else:              
                if start_idx > len(series):
                    print('start_idx > len(series)')
                    break            
                D[be].append(series[start_idx:end_idx])         

                  
        T.append(tr)
        kk+=1
    
    print(kk, 'trials used')
    return D




'''
####
batch processing
####
'''

def get_PSTHs_7behaviors(lag = -0.4):

    '''
    run once with lag = -0.6, -0.4
    '''
        
    lagd = {-0.6: '', -0.4: '0.4'}    
    df = bwm_query(one)
    eids = list(set(df['eid'].values))
    
    R = {}
    plt.ioff()
    for eid in eids:

        try:
            # only let sessions pass that have dlc passed for both side cams
            qc = one.get_details(eid, True)['extended_qc']
            if not (qc['dlcLeft'] == 'PASS' and qc['dlcRight'] == 'PASS'):
                continue        
                
            R[eid] = PSTH_pseudo(eid,lag = lag, duration = 0.4)    
        except:
            print(f'something off with {eid}')
            continue
    
    s = ('/home/mic/paper-brain-wide-map/'
         f'behavioral_block_correlates/behave7{lagd[lag]}.npy')
    np.save(s,R,allow_pickle=True)            
            


# eids with block modulated whisker ME
#eids = ['15f742e1-1043-45c9-9504-f1e8a53c1744',
#'38649476-84a9-4fc1-b9d9-e50a80eb3fbe',
#'7af49c00-63dd-4fed-b2e0-1b3bd945b20b',
#'35ed605c-1a1a-47b1-86ff-2b56144f55af',
#'3f76dfa7-db6f-4cbe-85d6-2834adffe708',
#'d6d829f9-a5b9-4ea5-a916-c7d2aadeccba',
#'e1931de1-cf7b-49af-af33-2ade15e8abe7',
#'00539228-f250-4b2f-8858-45421720d419',
#'6cf2a88a-515b-4f7f-89a2-7d53eab9b5f4',
#'8c552ddc-813e-4035-81cc-3971b57efe65',
#'36280321-555b-446d-9b7d-c2e17991e090',
#'c7b0e1a3-4d4d-4a76-9339-e73d0ed5425b',
#'594f962d-9bfc-41d9-8072-53c78cd976b3',
#'572a95d1-39ca-42e1-8424-5c9ffcb2df87',
#'99e5be7c-b4f8-46be-87d2-f31e662aea17',
#'e5c75b62-6871-4135-b3d0-f6464c2d90c0',
#'02181623-ca0d-4e6e-a6c9-730e9fd04e94',
#'d855576e-5b34-41bf-8e3b-2bea0cae1380']


def get_inter_trial_ints(eids):
    one = ONE()
    ds = []
    for eid in eids:
        trials = one.load_object(eid, 'trials')          
        for tr in range(len(trials['intervals']))[:-1]:  
            ds.append(trials['stimOn_times'][tr + 1] - trials['feedback_times'][tr])

    return ds


def save_ME(eids):

    for eid in eids:  
        velo, sides, pleft, MEs, licks, T, choices = cut_behavior(eid, -0.6, 0.4)
        np.save(f'/home/mic/behavior_models-master/MEs/{eid}.npy',MEs,
                allow_pickle=True)


def get_ps():

    eids = get_all_sess_with_ME()
    
    #columns = ['eid','video_type','lcs','p','dom']
    
    columns = ['eid','p','dist']
    r = []
    
    for eid in eids[:-2]:
        for video_type in ['left']:#,'right']:
            for lcs in [False]:
                try: 
                    blockPrint()                                    
                    p, dom, samp  = whisker_PSTH_pseudo(eid,duration=0.4, lag=-0.6, 
                                                        plotting=False, lcs=lcs, 
                                                        video_type=video_type)
                    enablePrint()
                    r.append([eid, p, samp])
                    print(f'{eids.index(eid)} of {len(eids)} done')
                except:
                    print('something off')
                    continue     
                            
    df  = pd.DataFrame(data=r,columns=columns)  
    df.to_pickle('/home/mic/paper-brain-wide-map/'
                 'behavioral_block_correlates/ME.pkl')


def get_bias(eid):
    '''
    compute bias; by Chris Langdon
    '''
    
    one = ONE()    
    trials = one.load_object(eid, 'trials')    
    
    choices = trials['choice'] 
    contrast_left = trials['contrastLeft'] 
    contrast_right = trials['contrastRight'] 
    probability_left = trials['probabilityLeft'] 
    
    if len(choices)==len(contrast_left)==len(contrast_right)==len(probability_left):
    
        zero_contrast_left_block = np.logical_and(
                                   np.logical_or(
                                   contrast_left==0,contrast_right==0),
                                   probability_left==0.8)
                                   
        zero_contrast_right_block = np.logical_and(
                                    np.logical_or(
                                    contrast_left==0,contrast_right==0),
                                    probability_left==0.2)
        
        a = np.sum(choices[zero_contrast_left_block]==1)
        b = np.sum(zero_contrast_left_block)    
        c = np.sum(choices[zero_contrast_right_block]==1)
        d = np.sum(zero_contrast_right_block)
                               
        delta = a/b - c/d 

        return delta   
        
    else:
        return None            
        
        
'''###################################
plotting
######################################'''


def example_block_structure():

    eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'
    trials = one.load_object(eid,'trials')
    
    fig, ax = plt.subplots(figsize=(2,1))
    plt.plot(trials['probabilityLeft'],color='k', 
             linestyle='',marker='o', markersize=1)        
    ax.set_xlabel('trials') 
    ax.set_yticks([0.2,0.5,0.8]) 
    ax.set_ylabel('p(stim left)')
    plt.tight_layout()



def Result_7behave(hists=False, save_df = False):

    '''
    bar plot
    Used in overleaf motor-correlates figure
    '''


    behave7 = ['licking', 'whisking_l', 'whisking_r', 
               'wheeling', 'nose_pos', 'paw_pos_r', 'paw_pos_l']

    infos = {'0.4': [[],'-0.4 s to stim'],
             '': [[],'-0.6 to -0.2 s to stim']} 

    
    if hists:
        fig = plt.figure(figsize=(5,4))
        ax = plt.subplot(1,2,1)
    else:
        fig = plt.figure(figsize=(3,4))
        ax = plt.subplot(1,1,1)    
    bwidth = 0.25
    
    cs = ['purple', 'orange']
    k = 0
    for t in ['0.4','']:
        s = ('/home/mic/paper-brain-wide-map/'
             f'behavioral_block_correlates/behave7{t}.npy')
             
        R = np.load(s,allow_pickle=True).flat[0]

        ms = ['dist','p','dom']
        
        c1 = [[x+'_'+y for y in ms] for x in behave7]
        c1flat = [item for sublist in c1 for item in sublist]
        c1flat.insert(0,'eid')      
        columns = c1flat

        r = []
        
        for eid in R:
            flag = False
            l = []
            if R[eid] == (None, None):
                continue
            else:
                for b in R[eid]:
                    if flag:
                        break    
                    for j in R[eid][b]:

                        if type(j)!= str and np.isnan(j):
                            flag = True
                            break   
                        l.append(j)
                    
                if flag:
                    continue
                else:            
                    l.insert(0,eid)
                    r.append(l)
        
        df  = pd.DataFrame(data=r,columns=columns)

        vals = [sum(df[x+'_p']<0.05)/len(df) for x in behave7]
        
        # check how many sessions have at least one modulated behavior
        ps  = [x+'_p' for x in behave7]
        rr = [[float(df[df['eid']==eid][y]) for y in ps] for eid in df['eid']]
        sigs = np.zeros(len(rr))
        sigs0 = np.zeros(len(rr))   
        for i in range(len(rr)):
            if any(np.array(rr[i])*len(behave7)*len(rr) < 0.05):
                sigs[i] = 1  # Bonferroni corrected
            if any(np.array(rr[i]) < 0.05):
                sigs0[i] = 1  # uncorrected
                      
        df['atLeastOne_p<0.05'] = sigs0
        df['atLeastOne_p<0.05_BF'] = sigs
 
                  
        sigs = sum(sigs) 
                
        print(t, (f'{sigs} out of {len(rr)}' 
              ' have at least one behavior modulated'))         
        
        vals.append(float(sigs)/len(rr))        
        infos[t][0].append(vals)        
        
        ax.barh(np.arange(len(vals))+k*0.25, list(reversed(vals)), 
                height = bwidth, label = infos[t][1], color=cs[k])

        if save_df:  
            df.to_pickle('/home/mic/paper-brain-wide-map/'
                         'behavioral_block_correlates/ME.pkl')
            df.to_excel('/home/mic/paper-brain-wide-map/'
                         'behavioral_block_correlates/ME.xlsx')
        k += 1
    
    ax.axvline(x=0.05, linestyle='--',c='k')
    ax.set_yticks(range(len(behave7)+1))    
    ax.set_yticklabels( list(reversed(behave7 +['at least one'])))
    ax.set_xlabel('fraction of sessions with p<0.05')
    print(f'{len(df)} sessions, {len(behave7)} behaviors \n'
                f'tested for block modulation \n')
    ax.set_ylabel('behavior')
    plt.tight_layout()
    plt.legend(ncol=1, frameon=False).set_draggable(True)                 

    if hists:
        ax = plt.subplot(1,2,2,sharey=ax)
        r2 = []
        cols2 = ['behavior','p']
        for eid in R:
                for b in R[eid]:
                    r2.append([b,R[eid][b][1]])

        df  = pd.DataFrame(data=r2,columns=cols2)  
       
        sns.violinplot(y="behavior", x="p", 
                       data=df, inner = None, 
                       color=".8", split=True,meanline=True, bw=.01,
                       orient='h')             

        plt.axvline(x=0.05, linestyle = '--', c='gray', 
                    label = f'p=0.05')
                    
        plt.title('smoothed historgam of p-values')
        plt.tight_layout()
       

def scatter_matrix_7behave():

    behave7 = ['licking', 'whisking_l', 'whisking_r', 
               'wheeling', 'nose_pos', 'paw_pos_r', 'paw_pos_l']


    s = ('/home/mic/paper-brain-wide-map/'
         f'behavioral_block_correlates/behave70.4.npy')
         
    R = np.load(s,allow_pickle=True).flat[0]
    
    #c1 = [x+'_dist' for x in behave7]     
    columns = behave7
    r = []
    
    for eid in R:
        l = []
        for b in R[eid]:
            l.append(R[eid][b][0])
        r.append(l)
 
    df  = pd.DataFrame(data=r,columns=columns)  
#    axes = pd.plotting.scatter_matrix(df)

#    for i in range(np.shape(axes)[0]):
#        for j in range(np.shape(axes)[1]):
#            if i < j:
#                axes[i,j].set_visible(False)


    def reg_coef(x,y,label=None,color=None,**kwargs):
        ax = plt.gca()        
        nas = np.logical_or(np.isnan(x), np.isnan(y)) 
        r,p = pearsonr(x[~nas], y[~nas])
        
        if p < 0.05:
            txt_c = 'k'
        else:
            txt_c = 'grey'
            
            
        ax.annotate(f'r = {np.round(r,2)} \n p = {np.round(p,3)}', 
                    xy=(0.5,0.5), xycoords='axes fraction', 
                    ha='center', color = txt_c)
        ax.set_axis_off()

    sns.set(font_scale=1)
    g = sns.PairGrid(df)
    g.map_diag(sns.distplot)
    g.map_lower(sns.regplot)
    g.map_upper(reg_coef)


def behave7_mouse_session():

    
    behave7 = ['licking', 'whisking_l', 'whisking_r', 
               'wheeling', 'nose_pos', 'paw_pos_r', 'paw_pos_l']


    s = ('/home/mic/paper-brain-wide-map/'
         f'behavioral_block_correlates/behave70.4.npy')
         
    R = np.load(s,allow_pickle=True).flat[0]
    
    #c1 = [x+'_dist' for x in behave7]
         
    columns = deepcopy(behave7)
    columns.insert(0,'mouse')
    columns.insert(0,'eid')
    
    r = []
    
    for eid in R:
        l = []
        l.append(eid)
        mouse = str(one.eid2path(eid)).split('/')[6]
        l.append(mouse)
        for b in R[eid]:
            l.append(R[eid][b][0])
        r.append(l)
 
    df0  = pd.DataFrame(data=r,columns=columns)  

    fig = plt.figure()
    
    k = 1 
    for behave in behave7:

        ax = plt.subplot(1,len(behave7),k)
        df = df0[['mouse',behave]]
        
        meds = df.groupby(['mouse']).median()
        meds.sort_values(ascending=False, inplace=True, by = behave)
        df['mouse'] = pd.Categorical(df['mouse'], meds.index)
        df.sort_values('mouse', inplace=True)
        
        
        sns.stripplot(x=behave, y="mouse", data=df, orient='h')
        plt.xlabel(f'{behave} dist, z-scored by null-d')

        #ax.set_yticks([])
        k += 1


def PSTH_pseudo(eid,duration =0.4, lag = -0.4, plotting=True, 
                control=False, query_type='remote',pawex=False):

    '''
    function to plot PSTH of whisking or licking in
    inter-trial interval, split by block, 
    with pseudo-session control

    TO PLOT OVERLEAF MOTOR CORRELATE EXAMPLE, pawexample = pawex
    eid = "15f742e1-1043-45c9-9504-f1e8a53c1744"
    param: control, boolean. If True, subdivide PSTH by choice
    '''

    if pawex:
        sr0 = {'paw_pos_l':60}
    else:
        sr0 = sr

    D = cut_behavior(eid, lag=lag, duration=duration,query_type=query_type)
    
    if len(Counter(D['pleft'])) != 3:
        print('no block structure')
        return None, None

    if plotting:
        plt.figure(figsize=(len(sr0)* 18/7,6))
        Ax = [plt.subplot(2,len(sr0),q) for q in range(1,2*len(sr0)+1)]

        
    Res = {}
    
    k = 0 
    for motor in sr0:
        if sr0[motor] == 'T_BIN':
            xs = np.arange(duration/T_BIN)*T_BIN       
        else:
            fs = sr0[motor] 
            xs = np.arange(duration * fs) / fs 
            
        xs = xs + lag
        
        D2 = {}

        null_d = []
        n_pseudo = 100
       
        for i in range(n_pseudo):    
            pb = generate_pseudo_blocks(len(D['pleft']))    
            
            left_block = np.nanmean((np.array(D[motor])
                           [np.where(np.array(pb)==0.8)[0]]),axis=0)
            right_block = np.nanmean((np.array(D[motor])
                           [np.where(np.array(pb)==0.2)[0]]),axis=0)  
               
            D2[f'p_{i}_{i + 1}'] = abs(left_block - right_block)
            null_d.append(np.nanmean(D2[f'p_{i}_{i + 1}']))       
                      
            if plotting:

                if i == 0:
                    l1 = f'{n_pseudo} x pseudo'
                else:
                    l1 = '_nolegend_'    

                Ax[k].plot(xs, left_block,label=l1,c='gray',linewidth=0.5)
                Ax[k].plot(xs, right_block,c='gray',linewidth=0.5)   
                   
                   
        left_block = np.nanmean((np.array(D[motor])
                      [np.where(np.array(D['pleft'])==0.8)[0]]),axis=0)
        right_block = np.nanmean((np.array(D[motor])
                       [np.where(np.array(D['pleft'])==0.2)[0]]),axis=0)
                          
        D2[f'left_right'] = abs(left_block - right_block)               
                
        if control:

            ts = []
            for pl in [0.8,0.2]:
                for ch in [1,-1]:

                    b1 = np.where(np.array(D['pleft'])==pl)[0]
                    b2 = np.where(np.array(D['choices'])==ch)[0]
                    s = np.array(list(set(b1).intersection(set(b2)))) 
                    ts.append(np.nanmean((np.array(D[motor])[s]),axis=0))          

        # for each observation, get p value between distance 
        # of real block psth curves and distances for the pseudo 
        # session PSTH curves

        obs = len(D2['left_right'])
        
        # save on which block is more whisking
        if (np.nanmean(left_block) - np.nanmean(right_block)) < 0:
            dom = 'pleft02'
        else:
            dom = 'pleft08'
        
        # average distances across observations
        samp = np.nanmean(D2['left_right'])
        
        # p value via percentile
        alpha = 1 - (0.01 * percentileofscore(null_d,samp,kind='weak'))        
        
        # z-scored distance
        samp = (samp - np.mean(null_d))/np.std(null_d)
        
        Res[motor] = [samp, alpha, dom]
                        
        if plotting:

            Ax[k].plot(xs, left_block,label='p(l) = 0.8',c= blue_left)
            Ax[k].plot(xs, right_block,label='p(l) = 0.2',c= red_right) 

            if control:
                
                cols = ['burlywood','coral','lime','teal']
                labs  = ['pleft0.8_choice1','pleft0.8_choice-1',
                         'pleft0.2_choice1','pleft0.2_choice-1']
                
                for i in range(len(ts)):
                    Ax[k].plot(xs,ts[i],label=labs[i],c=cols[i])               
       

            Ax[k].axvline(x=0,linestyle='--', 
                          label='stimOn',color='pink')    
            Ax[k].set_xlabel('time [sec]')     
            Ax[k].set_ylabel(motor)
            Ax[k].set_title(f'{motor} \n'
                            f'dist={np.round(samp,3)} \n'
                            f'p={np.round(alpha,2)}',fontsize=15)
            
            if k == 0:
                Ax[k].legend().set_draggable(True)
#            ax1.axvspan(abs(lag)-0.6, abs(lag)-0.2, facecolor='pink',
#                        alpha=0.3)     

            # Plot motor per trial
              
            cols = [cdi[x] for x in D['pleft']]    
            ttype = list(Counter(D['pleft']).keys())
            
            Ax[k + len(sr0)].scatter(D['T'], 
                        np.nanmean(D[motor],axis=1),
                        c=cols, s=1)        
        
            Ax[k + len(sr0)].set_ylabel(f'{motor}')
            Ax[k + len(sr0)].set_xlabel('trial number')
            legend_elements = [Line2D([0],[0], marker='o', color=cdi[y], 
                    label=y, markerfacecolor=cdi[y], 
                    markersize=5, linestyle='') for y in ttype]
            if k == 0:
                Ax[k + len(sr0)].legend(handles=legend_elements,
                            loc='best').set_draggable(True)    

        k += 1           
        plt.tight_layout()
        
    if plotting:
        pa = one.eid2path(eid)
        n = '_'.join([str(pa).split('/')[i] for i in [4,6,7,8]])
        plt.suptitle(f'{n}')    
        plt.tight_layout()           
#        plt.savefig('/home/mic/paper-brain-wide-map/'
#                    f'behavioral_block_correlates/figs/'
#                    f'7_behaviors0.4/{n}__{eid}.png')    
        #plt.close()    
    
    del D
    return Res


def whisker_inspection(eid, query_type='remote'):

    D = cut_behavior(eid, query_type=query_type) 


    fig = plt.figure(figsize=(10,10))

    # scatter: whisking versus neural firing
    ax = plt.subplot(2,2,1)
    ttype = list(Counter(D['pleft']).keys())
    cols = [cdi[x] for x in D['pleft']]    
    cmap = plt.get_cmap('Greys')(np.linspace(0,1,20)) 
    cm = mpl.colors.ListedColormap(cmap[10:,:-1])    

    plt.scatter(np.nanmean(D['MEs'],axis=1), 
                np.nanmean(np.nanmean(D['DD'],axis=1),axis=1),
                c=D['T'][0], s=20, alpha=0.5, cmap=cm, edgecolors=cols)
        
    plt.xlabel('MEs')
    plt.ylabel('pre stim neural firing \n per probe00')


    legend_elements = [Line2D([0],[0], marker='o', color=cdi[y], 
                        label=y, markerfacecolor=cdi[y], 
                        markersize=5, linestyle='') for y in ttype]
 
    ax.legend(handles=legend_elements, loc='best').set_draggable(True)    
    
    # time series, whisking per block
    ax = plt.subplot(2,2,2)

    plt.scatter(D['T'][0], 
                np.nanmean(D['MEs'],axis=1),
                c=D['T'][0], s=20, alpha=0.5, cmap=cm, edgecolors=cols)

    plt.ylabel('pre stim whisking (ME)')
    plt.xlabel('trial number')

    ax1 = plt.subplot(2,2,3)
    ax2 = plt.subplot(2,2,4,sharex=ax1)    
    whisker_PSTH_pseudo(eid,duration =4, lag = -2, 
                        ax1=ax1,ax2=ax2, query_type=query_type)


    one = ONE()
    p = one.eid2path(eid)
    s1 = '_'.join([str(p).split('/')[i] for i in [4,6,7,8]])
   
    plt.suptitle(f"each point is one of {len(D['pleft'])} trials \n"
                 f'{s1}_{eid}') 
    plt.tight_layout()    

#    pt = '/home/mic/paper-brain-wide-map/'\
#            f'behavioral_block_correlates/figs_scatters3/{s1}_{eid}.png'
#    plt.savefig(pt)    
#    plt.close()     


def per_trial_scatters(eid, frame=True):

    '''
    showing licks, nose and whisking as a function of 
    probe00 neural firing in scatters, colored by block/choice
    '''

    D = cut_behavior(eid)   
    
    fig = plt.figure(figsize=(10,10))
    k = 1
    for bs in ['MEs','licks','nose']:
        for tr in ['pleft','choices']:
            ax = plt.subplot(3,2,k)
            ttype = list(Counter(D[tr]).keys())
            cols = [cdi[x] for x in D[tr]]    
    
            plt.scatter(np.nanmean(D[bs],axis=1), 
                        np.nanmean(np.nanmean(D['DD'],axis=1),axis=1),
                        c=cols, s=5, alpha=0.5)
                
            plt.xlabel(bs)
            plt.ylabel('pre stim neural firing \n per probe00')


            legend_elements = [Line2D([0],[0], marker='o', color=cdi[y], 
                                label=y, markerfacecolor=cdi[y], 
                                markersize=5, linestyle='') for y in ttype]
         
            ax.legend(handles=legend_elements, loc='best').set_draggable(True)   
            k += 1
            
    plt.suptitle(f"each point is one of {len(D['pleft'])} trials \n"
                 f'{eid}') 
    plt.tight_layout()
    
    one = ONE()
    p = one.path_from_eid(eid)
    s1 = '_'.join([str(p).split('/')[i] for i in [4,6,7,8]])
    pt = '/home/mic/paper-brain-wide-map/'\
            f'behavioral_block_correlates/figs_scatters/{s1}_{eid}.png'
    plt.savefig(pt)    
    plt.close() 


def compare_neuro_whisk():

    '''
    scatter plot of p values of neural/whisker modulation by block
    '''
    
    R_whisk = np.load('/home/mic/paper-brain-wide-map/'
                     f'behavioral_block_correlates/ME.pkl', allow_pickle=True)
    R_whisk = dict(zip(R_whisk['eid'],R_whisk['p']))
    
    R_neuro = np.load('/home/mic/paper-brain-wide-map/manifold_analysis/'
                     f'manifold_ps/all_me.npy', allow_pickle=True).flat[0]

    D = {}
    for eid in R_neuro:
        if eid in R_whisk:
            D[eid] = [R_neuro[eid],R_whisk[eid]]

    plt.scatter(np.array(list(D.values()))[:,0],np.array(list(D.values()))[:,1])
    plt.xlabel('p neural')
    plt.ylabel('p whisking')
    plt.title(f'{len(D)} sessions')
    
    

def rasters10(eid,duration=4,lag=-2,video_type='left',
             align='stimOn_times', only_two=False):

    '''
    4 rasters and a summary plot, trial averages
    only_two: boolean: if only two blocks are compared
    '''
      
    delta = get_bias(eid)
    
    fs = Fs[video_type]
    
    one = ONE()
    #put stim_to_stim=True, for full range between trials
    [velo, sides, pleft, MEs, 
    licks, T, choices] = cut_behavior(eid, lag, duration,
                                      video_type=video_type,
                                      align=align)
                        
    if only_two:
        blocks = {}
        
        bs = np.where(np.diff(pleft)!=0)[0]
        ls = np.diff(bs)
        i = 0    
        for st in bs[:-1]:
            blocks[st] = [pleft[st+1], ls[i]]                   
            i += 1
        # get indices of two random blocks
        s08 = random.sample([x for x in blocks if blocks[x][0] == 0.8],1)[0]
        s02 = random.sample([x for x in blocks if blocks[x][0] == 0.2],1)[0]
            
        s08idx = range(s08 + 1, bs[list(bs).index(s08)+1]+1) 
        s02idx = range(s02 + 1, bs[list(bs).index(s02)+1]+1)
            
        s2 = set(s08idx).union(set(s02idx))                              
                                      
                                                 
    trials = one.load_object(eid, 'trials')
  
    data = {'whisk':MEs,'lick':licks,'wheel':velo}
    ii = 1        
    f = plt.figure(figsize=(18,10))
    axs = {}  
    for btype in ['wheel','whisk']:
        dat = np.array(data[btype],dtype=object)
        
        Ms = {}    
        Ds = {}
        for pl in [0.8,0.2]:
            for ch in [1,-1]:
            
                b1 = np.where(np.array(pleft)==pl)[0]
                b2 = np.where(np.array(choices)==ch)[0] 
                s = np.array(list(set(b1).intersection(set(b2))))  
                              
                if only_two:
                    s = np.array(list(set(s).intersection(s2)))

                # order by stimOn - last wheel motion                       
                aa = np.argsort(np.array(T[1])[s])        
                dat_s = dat[s][aa]
                Ds[f'{pl}_{ch}'] = np.array(T[1])[s][aa]
                M = dat_s
                
                Ms[f'{pl}_{ch}'] = np.array(M,dtype=float)
                

        Vmax = max([np.amax(Ms[x]) for x in Ms])
        Vmin = 0

        
        axs[5*ii] = plt.subplot(4,5,5*ii)
        axs[5*ii].set_xlabel('time [sec]')
        axs[5*ii].set_ylabel('mean across trials')   
        #ax0.set_xlim(-2,0)
        cols = ['orange','gold','lime','cyan']

        i = 1
        for pl in [0.8,0.2]:                
            for ch in [1,-1]:
                           
                M = Ms[f'{pl}_{ch}']
                
                y_dims, x_dims = M.shape
                if btype == 'whisk':
                    x_sec = x_dims / fs
                    xs = np.arange(x_dims) / fs
                else:
                    x_sec = x_dims * T_BIN
                    xs = np.arange(x_dims) * T_BIN

                axs[i+(5*(ii-1))] = plt.subplot(4,5,i+(5*(ii-1)),
                                                sharex=axs[5*ii])
                    
                axs[i+(5*(ii-1))].imshow(M,cmap='gray_r',aspect='auto',
                          extent=[lag,duration + lag,y_dims,0],
                          vmin=Vmin,vmax=Vmax)#
                          
             
                # plot stimOn
                pp = 0
                for q in Ds[f'{pl}_{ch}']:    
                    
                    if q > duration + lag:
                        pp += 1
                        continue
                    else:
                        if pp == 0:
                            lab = 'wheel_stop'
                        else:     
                            lab = '_nolegend_'            
                        plt.scatter(-q, pp, color='r',label=lab,
                        s=0.5)                 
                        pp += 1 
                                 
                axs[i+(5*(ii-1))].set_title(f'pleft{pl},choice{ch}',
                                     color=cols[i-1])
                axs[i+(5*(ii-1))].set_xlabel('time [sec]')
                axs[i+(5*(ii-1))].set_ylabel('trial number')
                axs[i+(5*(ii-1))].axvline(x=0,c='r',linestyle='--',label=align)
                if (i == 1) and btype == 'wheel':
                    axs[i+(5*(ii-1))].legend(loc='lower right') 
                
                axs[5*ii].plot(np.array(list(xs)) + lag,M.mean(axis=0),
                         label=f'{pl}_{ch}', color=cols[i-1])    

                i += 1    
        axs[5*ii].axvline(x=0,c='r',linestyle='--',label=align)       
        #axs[5*ii].legend()     
        axs[5*ii].set_title(f'btype={btype}')
        
        ii += 1

    for cond in range(2):
        # extra scatter plots of quiecent time/pre stim whisking
        
        # summary plot
        if cond == 0:
            axs[5*ii] = plt.subplot(4,5,5*ii) 
        else:
            axs[5*ii] = plt.subplot(4,5,5*ii,sharey=axs[15])    
        axs[5*ii].set_xlabel('stimOn - wheel_stop [sec]')
        axs[5*ii].set_ylabel('mean whisking 0.5 sec to stim') 
        axs[5*ii].set_title(f'superposition of row')
        axs[5*ii].grid()
        
        i = 1
        whisk_int = 1  # pre stim on whisking window to average [sec]
        for pl in [0.8,0.2]:
            for ch in [1,-1]:    

                axs[i+5*(ii-1)] = plt.subplot(4,5,i+5*(ii-1),
                                              sharex=axs[5*ii],
                                              sharey=axs[15])    
                    
                M = Ms[f'{pl}_{ch}']   
                D = Ds[f'{pl}_{ch}']    
                
                # filter by trials with stimon-wheel_stop threshold
                if cond == 0:
                    idx = np.where(D<1)[0]
                else:
                    idx = np.where(D>1)[0]
                     
                M = M[idx]
                D = D[idx]
                            
                trs, obs = M.shape
                
                xs = D
                ys = np.mean(M[:,int(obs/2-whisk_int*fs):int(obs/2)],axis=1)
                            
                axs[i+5*(ii-1)].scatter(xs,ys,color=cols[i-1],s=1)
                axs[5*ii].scatter(xs,ys,color=cols[i-1],s=1)
                
                # line fit
                m, b = np.polyfit(xs, ys, 1)
                axs[i+5*(ii-1)].plot(xs, m*xs + b, color=cols[i-1], 
                         linewidth=1)
                
                axs[5*ii].plot(xs, m*xs + b, color=cols[i-1], 
                         linewidth=1)
                
                axs[i+5*(ii-1)].set_title(f'pleft{pl},choice{ch}',
                                     color=cols[i-1])
                axs[i+5*(ii-1)].set_xlabel('stimOn - wheel_stop [sec]')
                axs[i+5*(ii-1)].set_ylabel('mean whisking 1 sec to stim')   
                axs[i+5*(ii-1)].grid()  
                                     
                i += 1                
    
        ii += 1
        
    if only_two:
        plt.suptitle(f'{eid}; align={align};'
                     f' only two blocks: '
                     f's08(start trial, length)={(s08,blocks[s08][1])}, '
                     f's02(start trial, length)={(s02,blocks[s02][1])}')
    else:
        plt.suptitle(f'{eid}; align={align}; rasters all on same scale')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])   

    if only_two:
        pt = f'/home/mic/paper-brain-wide-map/'\
             f'behavioral_block_correlates/figs_rasters10/'\
             f'{eid}_{s08}_{s02}.png'
    else:    
        pt = '/home/mic/paper-brain-wide-map/'\
            f'behavioral_block_correlates/figs_rasters10/{eid}.png'
    plt.savefig(pt)    
    plt.close()        
    


def PSTH_pseudo4(eid,duration=0.4,lag=-0.6,plotting=True, 
                   video_type='left'):
                    
    '''
    PSTH of whisking, licking, paw speed, wheel speed,
    aligned to stim on
    split by block and choice - pseudo session controlled

    ''' 
    delta = get_bias(eid)
    
    fs = Fs[video_type]
    
    one = ONE()
        
    if plotting:
        plt.figure(figsize=(16,10))
     
    aligns = ['stimOn_times']#, 'firstMovement_times', 'feedback_times']    
           
    Nds = []
    Samps = []       
           
    k = 1              
    for align in aligns:

        velo, sides, pleft, MEs, licks, T, choices = cut_behavior(eid, lag,
            duration, video_type=video_type, align=align)
        
        if len(Counter(pleft)) != 3:
            print('no block structure')
            return None, None
    
        data = {'whisk':MEs,'lick':licks,'wheel':velo}
    
        for btype in data:
        
            if plotting:
                if k % 3 == 1:
                    ax = plt.subplot(len(data),len(aligns),k) 
                else:    
                    ax = plt.subplot(len(data),len(aligns),k,sharex = ax)    
                         
            dat = np.array(data[btype])     
            if btype == 'whisk':
                xs = np.arange(duration * fs) / fs     
            else:
                xs = np.arange(duration / T_BIN) * T_BIN
                
            
            D = {}

            null_d = []
            for i in range(100):    
                pb = generate_pseudo_blocks(len(pleft))    
                
                left_block = dat[np.where(np.array(pb)==0.8)[0]].mean(axis=0)
                right_block = dat[np.where(np.array(pb)==0.2)[0]].mean(axis=0)  
                   
                D[f'p_{i}_{i + 1}'] = abs(left_block - right_block)
                null_d.append(np.mean(D[f'p_{i}_{i + 1}']))
                                
                if plotting:
                
                    if i == 0:
                        l1 = 'pseudo block'
                        l2 = '_nolegend_'
                    else:
                        l1 = l2 = '_nolegend_'    

                    plt.plot(xs, left_block,label=l1,c='k',linewidth=1)
                    plt.plot(xs, right_block,label=l2,c='k',linewidth=1)   
                       
             
                       
            left_block = dat[np.where(np.array(pleft)==0.8)[0]].mean(axis=0)
            right_block = dat[np.where(np.array(pleft)==0.2)[0]].mean(axis=0)     
            D[f'left_right'] = abs(left_block - right_block)               
                    

            ts = []
            for pl in [0.8,0.2]:
                for ch in [1,-1]:

                    b1 = np.where(np.array(pleft)==pl)[0]
                    b2 = np.where(np.array(choices)==ch)[0] 
                    s = np.array(list(set(b1).intersection(set(b2)))) 
                    ts.append(dat[s].mean(axis=0))          
                    
            # for each observation, get p value between distance 
            # of real block psth curves and distances for the pseudo 
            # session PSTH curves

            obs = len(D['left_right'])
            
            # save on which block is more whisking
            if (np.mean(left_block) - np.mean(right_block)) < 0:
                dom = 'pleft02'
            else:
                dom = 'pleft08'
            
            # average distances across observations
            samp = np.mean(D['left_right'])
                               
            # get p-values split by choice
            samps = [samp, np.mean(abs(ts[0] - ts[2])),
                     np.mean(abs(ts[1] - ts[3]))]   
                                            
            alphas = []
            for samp in samps:
                alphas.append(1 - (0.01 * percentileofscore(null_d,samp,kind='weak')))
                            
                               
            Nds.append(null_d)
            Samps.append(samps)                          
                               
            if plotting:

#                plt.plot(xs, left_block,label='pleft0.8',c='orange')
#                plt.plot(xs, right_block,label='pleft0.2',c='g') 


                cols = ['orange','gold','lime','cyan']
                labs  = ['pleft0.8_choice1','pleft0.8_choice-1',
                         'pleft0.2_choice1','pleft0.2_choice-1']
                
                for i in range(len(ts)):
                    plt.plot(xs,ts[i],label=labs[i],
                             c=cols[i],linewidth=1)               
           

                plt.axvline(x=-lag,linestyle='--',color='r')    
                plt.xlabel('time [sec]')     
                plt.ylabel(f'{btype} [a.u.]')   
                plt.title(f'align: {align}; p_s={np.round(alphas[0],3)},'
                                              f'{np.round(alphas[1],3)},'
                                              f'{np.round(alphas[2],3)}')
                
                if k == 2:       
                    plt.legend(fontsize=7) 
                
            k += 1

    
    p = one.path_from_eid(eid)
    s1 = ' '.join([str(p).split('/')[i] for i in [4,6,7,8]])
    if delta == None:
        plt.suptitle(f'{eid}, {s1}, bias= None')    
    else:    
        plt.suptitle(f'{eid}, {s1}, bias={np.round(delta,2)}')     
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])        
#    plt.savefig('/home/mic/paper-brain-wide-map/'
#                f'behavioral_block_correlates/figs_3x3/{eid}.png')    
#    plt.close()    
        #print(p,alpha)
        #return samp,null_d
    #return alpha, dom
    


def plt_hists(Nds, Samps):

    plt.figure()
    cs = ['b', 'g', 'r']
    labs = ['mean across choices', 'choice 1', 'choice 2']     
     
    for i in range(len(Samps)):
        plt.subplot(3, 1, i+1)
        plt.hist(Nds[i])
        k = 0
        for j in Samps[i]:
            plt.axvline(j, c=cs[k], label = labs[k])
            k += 1 
        plt.legend()


def PSTH_pseudo3x3(eid,duration=4,lag=-2,plotting=True, 
                   video_type='left'):
                    
    '''
    PSTH of whisking, licking and wheel speed,
    aligned to feedback type, stim on and motion onset,
    split by block and choice - pseudo session controled
    
    param: control, boolean. If True, subdivide PSTH by choice
    
    '''
    delta = get_bias(eid)
    
    fs = Fs[video_type]
    
    one = ONE()
        
    if plotting:
        plt.figure(figsize=(16,10))
     
    aligns = ['stimOn_times', 'firstMovement_times', 'feedback_times']    
           
    k = 1              
    for align in aligns:

        velo, sides, pleft, MEs, licks, T, choices = cut_behavior(eid, lag, duration,
                                                     video_type=video_type,
                                                     align=align)           
        
        if len(Counter(pleft)) != 3:
            print('no block structure')
            return None, None
    
        data = {'whisk':MEs,'lick':licks,'wheel':velo}
    
        for btype in data:
        
            if plotting:
                if k % 3 == 1:
                    ax = plt.subplot(len(data),len(aligns),k) 
                else:    
                    ax = plt.subplot(len(data),len(aligns),k,sharex = ax)    
                         
            dat = np.array(data[btype])     
            if btype == 'whisk':
                xs = np.arange(duration * fs) / fs     
            else:
                xs = np.arange(duration / T_BIN) * T_BIN
                
            
            D = {}

            null_d = []
            for i in range(100):    
                pb = generate_pseudo_blocks(len(pleft))    
                
                left_block = dat[np.where(np.array(pb)==0.8)[0]].mean(axis=0)
                right_block = dat[np.where(np.array(pb)==0.2)[0]].mean(axis=0)  
                   
                D[f'p_{i}_{i + 1}'] = abs(left_block - right_block)
                null_d.append(np.mean(D[f'p_{i}_{i + 1}']))
                                
                if plotting:
                
                    if i == 0:
                        l1 = 'pseudo block'
                        l2 = '_nolegend_'
                    else:
                        l1 = l2 = '_nolegend_'    

                    plt.plot(xs, left_block,label=l1,c='k',linewidth=1)
                    plt.plot(xs, right_block,label=l2,c='k',linewidth=1)   
                       
                       
            left_block = dat[np.where(np.array(pleft)==0.8)[0]].mean(axis=0)
            right_block = dat[np.where(np.array(pleft)==0.2)[0]].mean(axis=0)     
            D[f'left_right'] = abs(left_block - right_block)               
                    

            ts = []
            for pl in [0.8,0.2]:
                for ch in [1,-1]:

                    b1 = np.where(np.array(pleft)==pl)[0]
                    b2 = np.where(np.array(choices)==ch)[0] 
                    s = np.array(list(set(b1).intersection(set(b2)))) 
                    ts.append(dat[s].mean(axis=0))          
                    
            # for each observation, get p value between distance 
            # of real block psth curves and distances for the pseudo 
            # session PSTH curves

            obs = len(D['left_right'])
            
            # save on which block is more whisking
            if (np.mean(left_block) - np.mean(right_block)) < 0:
                dom = 'pleft02'
            else:
                dom = 'pleft08'
            
            # average distances across observations
            samp = np.mean(D['left_right'])

            alpha = 1 - (0.01 * percentileofscore(null_d,samp,kind='weak'))        
                               
            if plotting:

#                plt.plot(xs, left_block,label='pleft0.8',c='orange')
#                plt.plot(xs, right_block,label='pleft0.2',c='g') 


                cols = ['orange','gold','lime','cyan']
                labs  = ['pleft0.8_choice1','pleft0.8_choice-1',
                         'pleft0.2_choice1','pleft0.2_choice-1']
                
                for i in range(len(ts)):
                    plt.plot(xs,ts[i],label=labs[i],
                             c=cols[i],linewidth=1)               
           

                plt.axvline(x=duration+lag,linestyle='--',color='r')    
                plt.xlabel('time [sec]')     
                plt.ylabel(f'{btype} [a.u.]')   
                plt.title(f'align: {align}; p={np.round(alpha,3)}')
                
                if k == 2:       
                    plt.legend(fontsize=7) 
                
            k += 1

    
    p = one.path_from_eid(eid)
    s1 = ' '.join([str(p).split('/')[i] for i in [4,6,7,8]])
    if delta == None:
        plt.suptitle(f'{eid}, {s1}, bias= None')    
    else:    
        plt.suptitle(f'{eid}, {s1}, bias={np.round(delta,2)}')     
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])        
#    plt.savefig('/home/mic/paper-brain-wide-map/'
#                f'behavioral_block_correlates/figs_3x3/{eid}.png')    
#    plt.close()    
        #print(p,alpha)
        #return samp,null_d
    #return alpha, dom
    
    


def plot_action_prior_ME(signal, MEs, eid):

    #signal = np.load( # pre-compute via Charles' script
    #velo, sides, pleft, MEs, licks, T = cut_behavior(eid, lag, duration)
    assert len(signal) == len(MEs), 'not the same trials used'    
    mes = np.array([np.mean(x) for x in MEs])    
    plt.plot(mes, label='left whisker pad motion energy')
    plt.plot(signal * np.percentile(mes, 90) + np.percentile(mes, 5),
             label="Charles' model, action prior", linewidth=2)
    plt.xlabel('trial number')
    plt.title(f'{eid}; -0.6 t0 -.4 sec')
    plt.legend()
    

def plot_result(lcs = False):

    R  = np.load('/home/mic/paper-brain-wide-map/'
                 'behavioral_block_correlates/ps_lcs_me.npy',
                 allow_pickle=True).flat[0]
    v = 'ps_me'
    if lcs == True:
        v = 'ps_lc'
    
    u = np.argsort(R[v])
    ps = np.array(R[v])[u]
    eids = np.array(R['eids'])[u]

    plt.figure(figsize=(3,2))
    xs = np.arange(len(ps))
    plt.plot(xs,ps)
    ax = plt.gca()
    ax.axvline(x=xs[np.where(ps<0.001)[0][-1]],
               linestyle='--',c='r',label='p = 0.001')
    plt.ylabel('p')
    plt.title(v)    
      
#    ax.set_xticks(range(len(ps)))
#    ax.set_xticklabels(eids,rotation=90)
    plt.xlabel('sessions')
    plt.tight_layout() 
    return eids, ps


def plot_trials(neu, pleft, Clusters, MEs, licks,
                clus=235, duration =4, lag = -2, probe ='probe00',
                eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'):


#    neu, whe, times_, velo, sides, pleft, Clusters, MEs = 
#    pair_neural_wheel_vel(eid, -2, 4)


    one = ONE()
    # get true idx via Clusters

    idx = np.where(Clusters==clus)[0][0]

    ma = np.array(neu)[:,:,idx]
    left_block = ma[np.where(np.array(pleft)==0.8)[0]].mean(axis=0)
    right_block = ma[np.where(np.array(pleft)==0.2)[0]].mean(axis=0)    

    fig = plt.figure(figsize=(8,8))    
    ax0 = plt.subplot(2,3,1)
    plt.plot(np.arange(duration/T_BIN)*T_BIN, ma.mean(axis=0),label='both blocks')
    plt.plot(np.arange(duration/T_BIN)*T_BIN, left_block,label='prob left 0.8')
    plt.plot(np.arange(duration/T_BIN)*T_BIN, right_block,label='prob left 0.2')    

    plt.axvline(x=2,linestyle='--', label='stimOn',color='r')
    plt.xlabel('time [sec]')     
    plt.ylabel('mean activity \n across trials')   
    plt.title('all trials')
    plt.legend() 

    # plot the same for left whisker ME
    ax00 = plt.subplot(2,3,2) 

    MEs = np.array(MEs)
    left_block = MEs[np.where(np.array(pleft)==0.8)[0]].mean(axis=0)
    right_block = MEs[np.where(np.array(pleft)==0.2)[0]].mean(axis=0)     
    plt.axvline(x=2,linestyle='--', label='stimOn',color='r')
    plt.plot(np.arange(duration * 60)/60, MEs.mean(axis=0),label='both blocks')
    plt.plot(np.arange(duration * 60)/60, left_block,label='prob left 0.8')
    plt.plot(np.arange(duration * 60)/60, right_block,label='prob left 0.2')  
    plt.xlabel('time [sec]')     
    plt.ylabel('left whisker pad ME [a.u.]')   
    plt.title('PSTH: all trials')
    plt.legend() 
    
   
    ax1 = plt.subplot(2,3,4,sharex=ax0) 
    plt.imshow(ma,aspect='auto', extent=[0,duration,len(neu),0],
               cmap='gray_r')
    plt.xlabel('time [sec]')     
    plt.ylabel('trials') 
       
    ax2 = plt.subplot(2,3,5,sharey=ax1) 
    
    cols = {'0.8':'r','0.2':'g','0.5':'b'}    
    for i in range(len(neu)):
        plt.axhspan(i, i+1, facecolor=cols[str(pleft[i])],
                    alpha=0.5, label=f'prob left: {str(pleft[i])}')
                           
    plt.plot(ma.mean(axis=1),np.arange(len(neu)),color='k',linewidth=0.5)        
    plt.ylabel('trials')
    plt.xlabel('mean activity per trial')       
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),
               bbox_to_anchor=(0, 1.3), loc='upper left')

       
    ax3 = plt.subplot(2,3,3,sharex=ax00)    
    
    
    licks = np.array(licks)
    left_block = licks[np.where(np.array(pleft)==0.8)[0]].mean(axis=0)
    right_block = licks[np.where(np.array(pleft)==0.2)[0]].mean(axis=0)     
    plt.axvline(x=2,linestyle='--', label='stimOn',color='r')
    plt.plot(np.arange(duration/T_BIN)* T_BIN, licks.mean(axis=0),label='both blocks')
    plt.plot(np.arange(duration/T_BIN)* T_BIN, left_block,label='prob left 0.8')
    plt.plot(np.arange(duration/T_BIN)* T_BIN, right_block,label='prob left 0.2')  
    plt.xlabel('time [sec]')     
    plt.ylabel('licks [a.u.]')   
    plt.title('Lick PSTH: all trials')
    plt.legend()     
    

    p = one.path_from_eid(eid)
    plt.suptitle(' '.join([str(p).split('/')[i] for i in [4,6,7,8]]) 
                 + f' {probe}' + f' cluster {clus}',backgroundcolor= 'white') 

    plt.subplots_adjust(
    top=0.919,
    bottom=0.121,
    left=0.11,
    right=0.977,
    hspace=0.335,
    wspace=0.345)
    #plt.tight_layout()

'''
############
Overleaf BWM intro figure
############
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
    
    
def example_block_structure(eid):

    #eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'
    trials = one.load_object(eid,'trials')
    
    fig, ax = plt.subplots(figsize=(2,1))
    plt.plot(trials['probabilityLeft'],color='k', 
             linestyle='',marker='|', markersize=0.1)        
    ax.set_xlabel('trials') 
    ax.set_yticks([0.2,0.5,0.8]) 
    ax.set_ylabel('p(stim left)')
    plt.tight_layout()
    

def bwm_data_series_fig(cnew = False):
    '''
    plot a behavioral time series on block-colored background
    above neural data - intro figure
    '''

    
    eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'
    video_type='left'
    #reg = 'PRNr'#'SCiw''MRN'
    probe = 'probe00'
    trial_range = [242,244] 
    nsubplots = 4   
        

    trials = one.load_object(eid,'trials')
    tstart = trials['intervals'][trial_range[0]][0]
    tend = trials['intervals'][trial_range[-1]][-1]
    

    fig, axs = plt.subplots(nsubplots,1,figsize=(5,4),sharex=True, 
                gridspec_kw={'height_ratios': [1, 1,1,4]})
    # plot wheel speed trace
    #axs.append(plt.subplot(nsubplots,1,1))
    
    if cnew:
        Q = []
        wheel = one.load_object(eid, 'wheel')
        pos, times_w = wh.interpolate_position(wheel.timestamps, 
                                               wheel.position, freq=1/T_BIN)

        v = np.append(np.diff(pos),np.diff(pos)[-1])    
        
        s_idx = find_nearest(times_w, tstart)
        e_idx = find_nearest(times_w, tend)
        
        x = times_w[s_idx:e_idx]
        y = v[s_idx:e_idx]
        
        Q.append([times_w[s_idx:e_idx],v[s_idx:e_idx]])
        
    else:
        Q = np.load('Q.npy',allow_pickle=True)
        x,y = Q[0]
    
    
    axs[0].plot(x, y, c='k', label='wheel speed',linewidth=0.5)  
    axs[0].set_ylabel('wheel speed')# \n [rad/sec]
    axs[0].axes.yaxis.set_visible(False)
    # plot whisker motion           
    #axs.append(plt.subplot(nsubplots,1,2,sharex=axs[0]))
    
    if cnew:                                                   
        times_me, ME = get_ME(eid, video_type)  # when changing cam type, change fs in cut

        s_idx = find_nearest(times_me, tstart)
        e_idx = find_nearest(times_me, tend)
        
        x,y = times_me[s_idx:e_idx],ME[s_idx:e_idx]
        Q.append([x,y])
        
    else: 
        x,y = Q[1]    
       
    axs[1].plot(x, y, c='k', label='whisker ME',linewidth=0.5)
    axs[1].set_ylabel('whisking')#\n [a.u.]
    axs[1].axes.yaxis.set_visible(False)
    # plot licks
    #axs.append(plt.subplot(nsubplots,1,3,sharex=axs[0])) 
    
    if cnew:       
        times_, XYs = get_dlc_XYs(eid, video_type)    
        times_l = times_[get_licks(XYs)]
        R, times_l, _ = bincount2D(times_l, np.ones(len(times_l)), T_BIN)
        lcs = R[0]    

        s_idx = find_nearest(times_l, tstart)
        e_idx = find_nearest(times_l, tend)
        
        x,y = times_l[s_idx:e_idx],lcs[s_idx:e_idx]
        Q.append([x, y])

    else:
       x,y = Q[2]   

    axs[2].plot(x, y, c='k',linewidth=0.5)
    axs[2].set_ylabel('licking')# \n [a.u.]
    axs[2].axes.yaxis.set_visible(False)
    
    # neural activity
    #axs.append(plt.subplot(nsubplots,1,4,sharex=axs[0]))
    
    if cnew:

        # Load in spikesorting
        sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
   
        R, times_n, Clusters = bincount2D(
            spikes['times'], spikes['clusters'], T_BIN)   
        D = R.T 

        acs = br.id2acronym(clusters['atlas_id'],mapping='Beryl')
        _, palette = get_allen_info() 
        cols_ = [palette[x] for x in acs]

        s_idx = find_nearest(times_n, tstart)
        e_idx = find_nearest(times_n, tend)
        
        D = D[s_idx:e_idx]#,m_ask]
        D = D.T

        x,y = D,[times_n[s_idx],times_n[e_idx]]
        Q.append([x,y])
        
    else:
        x,y = Q[3]
    
    nneu,nobs = x.shape
    axs[3].imshow(x,aspect='auto',cmap='Greys',
            extent=[y[0],y[1],0,nneu],vmin=0,vmax=2)
    
            
    axs[3].set_ylabel('neuron')
    if cnew:
        for i in range(len(cols_)):            
            axs[3].plot([960.76,960.90],[i,i], color=cols_[i])        

    # correct, incorrect, stimulus contrast, choice, stim side

    cols = {'0.8':[0.13850039, 0.41331206, 0.74052025],
            '0.2':[0.66080672, 0.21526712, 0.23069468], '0.5':'g'}    
    for i in range(trial_range[0],trial_range[-1]+1):
        st = trials['intervals'][i][0]
        en = trials['intervals'][i][1]
        
        pl = trials['probabilityLeft'][i]
        
        if np.isnan(trials['contrastLeft'][i]):
            cont = trials['contrastRight'][i]            
            side = 'r'  # right side stimulus
        else:   
            cont = trials['contrastLeft'][i]         
            side = 'l'  # left side stimulus 
        
        choi = {1:'l',-1:'r'}[trials['choice'][i]]
        ftype = trials['feedbackType'][i]
        s = f'Trial: {i} \n stim: {side} \n choice: {choi} \n correct: {ftype}'
        axs[0].text(st,0.1,s)
        
        for k in range(nsubplots):
            axs[k].axvspan(st, en, facecolor=cols[str(pl)],
                    alpha=0.1, label=f'p(l)={str(pl)}')
            axs[k].axvline(x=trials['stimOn_times'][i],color='magenta',
                           linestyle='--', label='stimOn')
            axs[k].axvline(x=trials['feedback_times'][i],color='blue',
                           linestyle='--', label='feedback')
                           
    [axs[k].get_xaxis().set_visible(False) for k in range(len(axs[1:]))] 
    axs[-1].set_xlabel('time [sec]')
    
    fig.subplots_adjust(top=0.812,
                        bottom=0.176,
                        left=0.191,
                        right=0.985,
                        hspace=0.15,
                        wspace=0.2)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[-1].legend(by_label.values(), 
            by_label.keys(),ncol=4).set_draggable(True)
    
    if cnew:
        np.save('Q.npy',Q,allow_pickle=True)
#    return Q

'''
##########
overleaf motor_correlates figure
##########
'''

def paw_position_onframe(D):

    '''
    example video frame with average paw position locations on top
    '''
    
    video_type = 'left'
    behave = 'paw_pos_l'  

    eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'    
    r = np.load(f'/home/mic/VIDEO_QC/reproducible_dlc/example_images/'
              f'{eid}_{video_type}.npy')[0]
    fig,ax = plt.subplots(figsize=(5,4))
    ax.imshow(r, cmap='gray', origin="upper")
    if D == None:
        D = cut_behavior(eid,pawex=True)
    
 
    # get list of colors according to 

    cols = [cdi[x] for x in D['pleft']]
    
    xs = [np.mean(u[0]) for u in D[behave]] 
    ys = [np.mean(u[1]) for u in D[behave]]   
    
    # plot in random order for better visibility
    # plot only every nth trial
    n = 1
    ids = np.arange(len(xs))
    random.shuffle(ids)
    xs = np.array(xs)[ids][::n]
    ys = np.array(ys)[ids][::n]
    cols = np.array(cols)[ids][::n]
    print(xs[:3],ys[:3])
    ax.scatter(xs,ys,c=cols,s=200,alpha=1,marker='x')  
     
    legend_elements = [Line2D([0],[0], marker='x', color=cdi[y], 
            label=f'p(l)={y}', markerfacecolor=cdi[y], 
            markersize=8, linestyle='') for y in Counter(D['pleft']).keys()]
          
    plt.legend(handles=legend_elements, loc='lower right', 
               prop={'size': 8}).set_draggable(True)        
                
    plt.axis('off')
    plt.tight_layout()
    return D
    #continue here

#To illustrate paw position with pseudo sessions
#PSTH_pseudo(eid,pawex=True)


def res_to_df():

    t = ''  # lag = -0.6 sec
    s = ('/home/mic/paper-brain-wide-map/'
        f'behavioral_block_correlates/behave7{t}.npy')
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
    df.to_csv('/home/mic/paper-brain-wide-map/'
              'behavioral_block_correlates/motor_corr_0.6_0.2.csv')




