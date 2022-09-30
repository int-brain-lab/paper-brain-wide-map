from one.api import ONE
from brainbox.io.one import load_channel_locations 
import brainbox.behavior.wheel as wh
from brainbox.processing import bincount2D
from ibllib.atlas import regions_from_allen_csv
import ibllib.atlas as atlas
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
from brainbox.io.one import SpikeSortingLoader
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


import glob
from braindelphi.params import CACHE_PATH
from datetime import datetime
import pickle

matplotlib.rcParams.update({'font.size': 10})

one = ONE()
ba = AllenAtlas()
br = BrainRegions()
T_BIN = 0.02
Fs = {'left':60,'right':150, 'body':30}

# specify binning type, either bins or sampling rate; 

sr = {'licking':'T_BIN','whisking_l':'T_BIN', 'whisking_r':'T_BIN', 
      'wheeling':'T_BIN','nose_pos':'T_BIN', 'paw_pos_r':'T_BIN', 
      'paw_pos_l':'T_BIN'}

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


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def get_licks(XYs):
    '''
    define a frame as a lick frame if
    x or y for left or right tongue point
    change more than half the sdt of the diff
    '''  
    
    licks = []
    for point in ['tongue_end_l', 'tongue_end_r']:
        for c in XYs[point]:
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
    '''get_dlc_XYsdlc
    cut segments of behavioral time series for PSTHs
    
    param: eid: session eid
    param: align: in stimOn_times, firstMovement_times, feedback_times    
    param: lag: time in sec wrt to align time to start segment
    param: duration: length of cut segment in sec 
    '''
    # get wheel speed    
    wheel = one.load_object(eid, 'wheel', query_type=query_type)
    pos, times_w = wh.interpolate_position(wheel.timestamps,
                                           wheel.position, freq=1/T_BIN)
    v = np.append(np.diff(pos),np.diff(pos)[-1]) 
    v = abs(v) 
    v = v/max(v)  # else the units are very small
    
    # load whisker motion energy, separate for both cams
    times_me_l, whisking_l0 = get_ME(eid, 'left', query_type=query_type)
    times_me_r, whisking_r0 = get_ME(eid, 'right', query_type=query_type)    
    
    times_l, XYs_l = get_dlc_XYs(eid, 'left')
    times_r, XYs_r = get_dlc_XYs(eid, 'right')    
    
    DLC = {'left':[times_l, XYs_l], 'right':[times_r, XYs_r]}
    
    # get licks using both cameras    
    lick_times = []
    for video_type in ['right','left']:
        times, XYs = DLC[video_type]
        r = get_licks(XYs)
        try :
            idx = np.where(np.array(r)<len(times))[0][-1]  # ERROR HERE ...    
            lick_times.append(times[r[:idx]])
        except :
            print('ohoh')
    
    lick_times = sorted(np.concatenate(lick_times))
    R, times_lick, _ = bincount2D(lick_times, np.ones(len(lick_times)), T_BIN)
    lcs = R[0]    
    # get paw position, for each cam separate
    
    if pawex:
        paw_pos_r0 = XYs_r['paw_r']
        paw_pos_l0 = XYs_l['paw_r']    
    else:
        paw_pos_r0 = (XYs_r['paw_r'][0]**2 + XYs_r['paw_r'][1]**2)**0.5
        paw_pos_l0 = (XYs_l['paw_r'][0]**2 + XYs_l['paw_r'][1]**2)**0.5
    
    # get nose x position from left cam only
    nose_pos0 = XYs_l['nose_tip'][0]
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
    difs = [] # difference between stim on and last wheel movement
    d = (licking, whisking_l, whisking_r, wheeling,
         nose_pos, paw_pos_r, paw_pos_l,
         pleft, sides, choices, T, difs)
    ds = ('licking','whisking_l', 'whisking_r', 'wheeling',
         'nose_pos', 'paw_pos_r', 'paw_pos_l',
         'pleft', 'sides', 'choices', 'T', 'difs')
         
    D = dict(zip(ds,d))
    
    # continuous time series of behavior and stamps
    behaves = {'licking':[times_lick, lcs],
               'whisking_l':[times_me_l, whisking_l0], 
               'whisking_r':[times_me_r, whisking_r0], 
               'wheeling':[times_w, v],
               'nose_pos':[times_l, nose_pos0],
               'paw_pos_r':[times_r,paw_pos_r0], 
               'paw_pos_l':[times_l,paw_pos_l0]}
    trials = one.load_object(eid, 'trials', query_type=query_type)    
    wheelMoves = one.load_object(eid, 'wheelMoves', query_type=query_type)
    
    print('cutting data')
    trials = one.load_object(eid, 'trials', query_type=query_type)
    evts = ['stimOn_times', 'feedback_times', 'probabilityLeft',
            'choice', 'feedbackType','firstMovement_times']
            
    kk = 0     
    for tr in range(len(trials['intervals'])): 
        
        '''
        # skip trial if any key info is nan 
        if any(np.isnan([trials[k][tr] for k in evts])):
            continue
        
        # skip trial if any key info is nan for subsequ. trial    
        if any(np.isnan([trials[k][tr + 1] for k in evts])):
            continue            

        # skip trial if too long
        if trials['feedback_times'][tr] - trials['stimOn_times'][tr] > 10: 
            continue   
        
        # skip block boundary trials
        if trials['probabilityLeft'][tr] != trials['probabilityLeft'][tr+1]:
            continue
        '''
        
        a = wheelMoves['intervals'][:,1]
        '''
        b = trials['stimOn_times'][tr]
        c = trials['feedback_times'][tr - 1]     
        # making sure the motion onset time is in a coupled interval
        ind = np.where((a < b) & (a > c))[0]
        try:
            a = a[ind][-1]
        except:
            continue   
        
        
        difs.append(b-a)
        '''

        if stim_to_stim:
            start_t = trials['stimOn_times'][tr]    
                    
        elif align == 'wheel_stop':            
            start_t = a + lag    
            
        else:                                
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
                '''  # bug inducing 
                if start_idx > len(series):
                    print('start_idx > len(series)')
                    break      
                '''  
                D[be].append(series[start_idx:end_idx])         
                  
        T.append(tr)
        kk+=1

    print(kk, 'trials used')
    return(D)



def preprocess_motors_old(eid,metadata):

    motor_eids = get_all_sess_with_ME()

    assert eid in motor_eids, "no motor signals for this session" # test if the eid belong to sessions with recorded motor signals

    DURATION = metadata['time_window'][1]-metadata['time_window'][0]
    LAG = metadata['time_window'][0]
    
    D = cut_behavior(eid, duration = DURATION, lag = LAG, align=metadata['align_time'])

    motor_signals_of_interest = ['licking', 'whisking_l', 'whisking_r', 'wheeling', 'nose_pos', 'paw_pos_r', 'paw_pos_l']

    motor_signals = np.zeros((len(D['licking']),len(motor_signals_of_interest)))
    for i in range(len(D['licking'])):
        for j,motor in enumerate(motor_signals_of_interest) :
            # we add all bin values to get a unique regressor value for decoding interval
            motor_signals[i][j] = np.nansum(D[motor][i]) # we don't take the NaN into account in the sum

    motor_signals = np.expand_dims(motor_signals,1)

    return list(motor_signals)


def preprocess_motors(eid,kwargs):

    neural_dtype_paths = glob.glob(CACHE_PATH.joinpath('*_motor_metadata.pkl').as_posix())

    neural_dtype_dates = [datetime.strptime(p.split('/')[-1].split('_')[0], '%Y-%m-%d %H:%M:%S.%f')
                            for p in neural_dtype_paths]

    path_id = np.argmax(neural_dtype_dates)

    motor_metadata = pickle.load(open(neural_dtype_paths[path_id], 'rb'))

    try :
        regressor_path = motor_metadata['dataset_filenames'][ motor_metadata['dataset_filenames']['eid'] == eid]['reg_file'].values[0]
    except :
        print('not cached...')
        preprocess_motors_old(eid,kwargs)

    regressors = pickle.load(open(regressor_path, 'rb'))

    motor_signals_of_interest = ['licking', 'whisking_l', 'whisking_r', 'wheeling', 'nose_pos', 'paw_pos_r', 'paw_pos_l']

    # the cache contain 100 values for each regressors : t_bins=0.02s & t-min= -1s , t_max= +1s
    # we select the bins inside the decoding interval
    t_min = kwargs['time_window'][0]
    t_max = kwargs['time_window'][1]
    T_bin = 0.02
    i_min = int((t_min + 1)/T_bin)
    i_max =  int((t_max + 1)/T_bin)

    motor_signals = np.zeros((len(regressors['licking']),len(motor_signals_of_interest)))
    for i in range(len(regressors['licking'])):
        for j,motor in enumerate(motor_signals_of_interest) :
            # we add all bin values to get a unique regressor value for decoding interval
            try :
                motor_signals[i][j] = np.nansum(regressors[motor][i][i_min:i_max])
            except :
                print('time bounds reached')
                motor_signals[i][j] = np.nansum(regressors[motor][i]) # TO CORRECT

    # normalize the motor signals
    motor_signals = stats.zscore(motor_signals,axis=0)

    motor_signals = np.expand_dims(motor_signals,1)

    return list(motor_signals)






    


