from one.api import ONE
#from brainbox.plot import driftmap
from brainwidemap import load_good_units
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
from ibllib.plots import Traces
from reproducible_ephys_processing import bin_spikes2D
from brainbox.behavior.dlc import likelihood_threshold

from iblatlas.regions import BrainRegions

import os
import numpy as np
from scipy import stats, signal, interpolate
from scipy.io import savemat, loadmat
from pathlib import Path
from collections import Counter, ChainMap
import math
from scipy.stats import pearsonr, spearmanr, percentileofscore
from copy import deepcopy
import random
from itertools import combinations
import subprocess
import gc
import matplotlib.pyplot as plt
#import nitime.algorithms as tsa
from matplotlib.colors import LogNorm
import matplotlib as mpl
#import squarify 
from PIL import ImageColor
import pandas as pd
#from wordcloud import WordCloud
from scipy.fft import rfft, rfftfreq, fftshift
from matplotlib.lines import Line2D
from scipy.stats import zscore
from scipy.signal import find_peaks
from matplotlib.patches import Patch

one_online = ONE()
T_BIN = 0.005    
#pre_time = 0.5
#post_time = 1.5
bin_size = T_BIN  # time bin size in seconds
one = ONE() #ONE(mode='local')
ba = AllenAtlas()
br = BrainRegions()
# matlab function for spectral granger
#https://github.com/oliche/spectral-granger/blob/master/matlab/spectral_granger.m


def get_allen_info(rerun=False):
    '''
    Function to load Allen atlas info, like region colors
    '''
    
    pth_dmna = Path(one.cache_dir, 'dmn', 'alleninfo.npy')
    
    if (not pth_dmna.is_file() or rerun):
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

        #add layer colors
        bc = ['b', 'g', 'r', 'c', 'm', 'y', 'brown', 'pink']
        for i in range(7):
            palette[str(i)] = bc[i]
        
        palette['thal'] = 'k'    
        r = {}
        r['dfa'] = dfa
        r['palette'] = palette    
        np.save(pth_dmna, r, allow_pickle=True)   

    r = np.load(pth_dmna, allow_pickle=True).flat[0]
    return r['dfa'], r['palette']  


def get_dlc_XYs(eid, cam):

    vf = one.load_object(eid, f'{cam}Camera',collection='alf')

    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in vf['dlc'].keys()])

    for point in points:
        x = np.ma.masked_where(
            vf['dlc'][point + '_likelihood'] < 0.9, vf['dlc'][point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(
            vf['dlc'][point + '_likelihood'] < 0.9, vf['dlc'][point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array(
            [x, y])    

    return vf['times'], XYs 



def get_single_cells_cut(eid, probe):

    '''
    for a given region, session, probe, bin and trial cut
    all clusters
    '''
        
#    eid = 'fc43390d-457e-463a-9fd4-b94a0a8b48f5'
#    reg = 'VPL'
#    probe = 'probe00'
    pre_time = 0.5
    post_time = 1.5    
    data = {}

    # Load in spikesorting
    sl = SpikeSortingLoader(eid=eid, pname=probe, one=one_online, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting(revision="2024-05-06")
    clusters = sl.merge_clusters(spikes, clusters, channels)
                          
    data['cluster_ids'] = clusters['cluster_id']#[
                           #clusters['acronym'] == reg]
                           
    # Find spikes that are from the clusterIDs
    spike_idx = np.isin(spikes['clusters'], data['cluster_ids'])

    # Load in trials data
    trials = one_online.load_object(eid, 'trials', collection='alf')
    # For this computation we use correct, non zero contrast trials
    trial_idx = np.bitwise_and(trials['feedbackType'] == 1,
                               np.bitwise_or(trials['contrastLeft'] > 0,
                               trials['contrastRight'] > 0))
    # Find nan trials
    nan_trials = np.bitwise_or(np.isnan(trials['stimOn_times']),
                               np.isnan(trials['firstMovement_times']))

    # Find trials that are too long
    stim_diff = trials['feedback_times'] - trials['stimOn_times']
    rm_trials = stim_diff > 10
    # Remove these trials from trials object
    rm_trials = np.bitwise_or(rm_trials, nan_trials)

    eventMove = trials['firstMovement_times'][np.bitwise_and(trial_idx, ~rm_trials)]
    eventStim = trials['stimOn_times'][np.bitwise_and(trial_idx, ~rm_trials)]
    eventFback = trials['feedback_times'][np.bitwise_and(trial_idx, ~rm_trials)]


    # Movement firing rate
    bins, t = bin_spikes2D(spikes['times'][spike_idx], 
                           spikes['clusters'][spike_idx],
                           data['cluster_ids'],eventFback, 
                           pre_time, post_time, bin_size)
                           
    acs = br.id2acronym(clusters['atlas_id'],mapping='Beryl')
    return bins, data['cluster_ids'], acs


def get_psths_singlecell_lick_aligned(eid, probe):

    '''
    for a given session, probe, and cell, bin neural activity
    into PETHs aligned to licks
    '''    
    
    pre_time = 0.075
    post_time = 0.075
       
    # Load in spikesorting
    sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting(revision='2024-05-06')
    clusters = sl.merge_clusters(spikes, clusters, channels)

    # Find spikes that are from the clusterIDs
    spike_idx = np.isin(spikes['clusters'], clusters['cluster_id'])

    # get licks using both cameras  
    events = get_lick_times(eid)

    # bin and cut into trials    
    bi, _ = bin_spikes2D(spikes['times'][spike_idx], 
                       spikes['clusters'][spike_idx],
                       clusters['cluster_id'],np.array(events), 
                       pre_time, post_time, T_BIN)
                       
    w = np.array(bi.mean(axis=0))  # average trials to get PSTH
    acs = br.id2acronym(clusters['atlas_id'],mapping='Beryl')
    return w, clusters['cluster_id'], acs


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or 
    math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx        
        
        
def get_acronyms_per_eid(eid):

    T_BIN = 1
    
    As = {}

    dsets = one.list_datasets(eid)
    r = [x.split('/') for x in dsets if 'probe' in x]
    rr = [item for sublist in r for item in sublist
          if 'probe' in item and '.' not in item]
    probes = list(Counter(rr))         
    
    for probe in probes:
        sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting(revision='2024-05-06')
        clusters = sl.merge_clusters(spikes, clusters, channels)    
        As[probe] = clusters['acronym']

    return As

  
def get_lick_times(eid):
    times_l, XYs_l = get_dlc_XYs(eid, 'left')
    times_r, XYs_r = get_dlc_XYs(eid, 'right')    
    
    DLC = {'left':[times_l, XYs_l], 'right':[times_r, XYs_r]}
          
    lick_times = []
    for video_type in ['right','left']:
        times, XYs = DLC[video_type]
        r = get_licks(XYs)
        idx = np.where(np.array(r)<len(times))[0][-1]            
        lick_times.append(times[r[:idx]])
    
    lick_times = sorted(np.concatenate(lick_times))     
    return lick_times   


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
    
    
def get_eids_per_reg(reg):

    fp = '/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/granger/0.005/data'    
    eids = os.listdir(fp)
    D = {}
    eids2 = []
    for eid in eids:
        if reg in np.load(fp+'/'+eid+'/regs.npy'):
            eids2.append(eid)
    return eids2        
    
            
'''
####################
#################### batch processing
####################
'''

 
def batch_welch_whole_session():
    
    eids = get_bwm_sessions()

    #eids = ['15f742e1-1043-45c9-9504-f1e8a53c1744'] 
    plt.ioff()

    R = {}
    Fs = []
    
    ks = ['regs','neurons','psd']
    
    k=0
    for eid in eids:
        try: 
            regs, neurons, psd = full_session_Welch(eid,bulk=True)                
            D = dict(zip(ks,[regs, neurons, psd]))
            R[eid] = D         

            gc.collect()
        except:
            Fs.append(eid)
            gc.collect()
            k+=1
            print(k, 'of', len(eids), 'DONE')
            continue  

    
    np.save('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/granger/'
           f'whole_session_psd.npy', R, allow_pickle=True) 

    print(f'{len(Fs)}, load failures:')
    return Fs



def bulk_psd():
 
    eids = get_bwm_sessions()
    ks = ['regs','neurons','psd']
    
    plt.ioff()
    
    duration = 2
    lag = -0.5   
    
    fs = int(1/T_BIN)
    b = f'/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/granger'      

    R = {}
    for eid in eids[38:]:

        try:
            dd, regs,neurons = trial_cut(eid, duration=duration ,
                                      lag=lag, pseudo=False, 
                                      include_licks=False)            
            gc.collect()
                            

            PSDs = []
            for reg in regs:
                  
                reg_in = list(regs).index(reg)

                # get from freshly loaded data
                ds = [[],[]]
                for i in dd:
                    if i[1][1] == 1:
                        ds[0].append(i[0][:,reg_in])
                    else:
                        ds[1].append(i[0][:,reg_in])    

                ds[0] = np.array(ds[0]).T
                ds[1] = np.array(ds[1]).T
                
                ntime,ntr_ex = ds[0].shape    
                ntime,ntr_unex = ds[1].shape    

                d = np.hstack(ds).T #stack ex, unex     
                
                # normalisation
                d = d/np.mean(d)
                
                ntr, ntime = d.shape

                fig,axs = plt.subplots(nrows=1, ncols=3,figsize=(9,3),
                                       gridspec_kw={'width_ratios': [1,1,2]})
                     
                k = 0     

                axs[k].imshow(d,cmap='Greys',aspect="auto",
                    interpolation='none')#,vmin=minY,vmax=maxY)       


                axs[k].axvline(x=0.5/T_BIN,linestyle='--',color='g', 
                               label='stim on')
                axs[k].axhline(y=ntr_ex, linewidth=1, linestyle='-', c='b')
                axs[k].text(0,ntr_ex,'engaged trials above')    
                axs[k].set_xticks(np.linspace(0,ntime,5))
                axs[k].set_xticklabels(np.linspace(0,ntime,5)*T_BIN - 0.5)
                axs[k].set_title(f'mean neural firing rate')
                axs[k].set_xlabel('time [sec]')
                axs[k].set_ylabel('trials, ordered by time')
                axs[k].legend()
                plt.tight_layout()
                k+=1

                f, psd = signal.welch(d,fs=fs)

                axs[k].imshow(psd,cmap='Greys',aspect="auto",
                    interpolation='none')#,vmin=minY,vmax=maxY)  
                                
                axs[k].set_xticks(np.linspace(0,len(f),5))
                axs[k].set_xticklabels(np.linspace(0,f[-1],5))              
                axs[k].set_title(f'psd (Welch) of neural')
                axs[k].set_xlabel('frequency [Hz]')
                axs[k].set_ylabel('trials, ordered by time')         
                axs[k].axhline(y=ntr_ex, linewidth=1, linestyle='-', 
                                      c='b')       
                axs[k].text(0,ntr_ex,'engaged trials above')
                k+=1

                # doing fft directly per trial and then showing average
                # "roll your own Welch"
                ntr, nobs = d.shape
                yf = rfft(d)
                xf = rfftfreq(nobs, T_BIN)
                yf = np.mean(np.abs(yf)**2,axis=0)
                axs[k].plot(xf,np.log(yf))
                axs[k].set_xlabel('frequency [Hz]')
                axs[k].set_ylabel('log(<abs(fft(trial))^2>_trials)')
                axs[k].set_title(f'average spectrum across trials')    
                mouse_date = ' '.join(str(one.eid2path(eid)).split('/')[6:8])    
                plt.suptitle(f'{eid}, {mouse_date}, {reg}, #neu: {neurons[reg_in]}')
              
                plt.tight_layout() 

                s2 = b+'/trial_average_figs_norm'    
                Path(s2).mkdir(parents=True, exist_ok=True)   
                plt.savefig(s2+f'/{eid}_{mouse_date}_{reg}.png')
                plt.close()
                PSDs.append(np.log(yf))
                
                #save spectrum
            D = dict(zip(ks,[regs, neurons, PSDs]))
            R[eid] = D  
            
        except:
            gc.collect()
            continue             
 
    np.save('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/granger/'
           f'trial_average_psd_norm.npy', R, allow_pickle=True) 


def full_session_spectrogram(eids=None, reg_pairs=True):

    '''
    for a given session plot the spectrogram
    for the whole recording and save auto-correlogram
    for each region.
    
    If reg_pairs, also save the correlograms of pairs of regions.
    '''
    plt.ioff()
    fs = int(1/T_BIN)

    # make also results folder structure    
    s = (f'/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/'
         f'granger/')

    if eids is None:
        eids = get_bwm_sessions()    

    c = 0
    for eid in eids:
    
        try:
            D, times, regs, neurons = bin_average_neural(eid)

            if reg_pairs:
                for a,b in combinations(range(len(regs)),2):
                
                
                    if regs[a] in ['root','void']:
                        continue
                    if regs[b] in ['root','void']:
                        continue                        

                    f, t, Sxxa = signal.spectrogram(D[:,a], fs)   
                    f, t, Sxxb = signal.spectrogram(D[:,b], fs)
                            
                    Sxx = np.concatenate([Sxxa,Sxxb])        
                            
                    fig, ax = plt.subplots(figsize=(5,5))        
                    im = ax.imshow(np.corrcoef(Sxx), cmap='Greys') 
                    ax.set_title(f'{regs[a]} [{neurons[a]}], {regs[b]} [{neurons[b]}]; '
                                 f'len [min]:{int(times[-1]/60)}; fs: {fs} \n'
                                 f'{eid}')
                    ax.set_xticks(np.linspace(1,258,16))            
                    ax.set_xticklabels(np.round(np.concatenate(
                       [np.linspace(1,129,8),np.linspace(1,129,8)])),rotation = 90)
                    ax.set_yticks(np.linspace(1,258,16))            
                    ax.set_yticklabels(np.round(np.concatenate(
                       [np.linspace(1,129,8),np.linspace(1,129,8)])))                
                                 
                    ax.set_xlabel('freq [Hz]')
                    ax.set_ylabel('freq [Hz]')
                    
                    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)            
                    cbar.set_label('Pearson r between spectrogram')
                    plt.tight_layout()
                    plt.savefig(s+'full_session_spec_corr_pairs/'+
                                f'{eid}_{regs[a]}_{regs[b]}.png')                
                                
                    plt.close()
                    gc.collect()
                    
        
            else:  
                for k in range(len(regs)):  
                    if regs[k] in ['root','void']:
                        continue
                    
                    f, t, Sxx = signal.spectrogram(D[:,k], fs)
                    fig, ax = plt.subplots(figsize=(5,5))        
                    ax.imshow(np.corrcoef(Sxx), cmap='Greys') 
                    ax.set_title(f'reg: {regs[k]}, nclus:{neurons[k]}, '
                                 f'len [min]:{int(times[-1]/60)}, fs: {fs} \n'
                                 f'{eid}')
                    ax.set_xlabel('freq [Hz]')
                    ax.set_ylabel('freq [Hz]')
                    plt.savefig(s+'full_session_spectrogram_corr/'+
                                f'{eid}_{regs[k]}.png')                                
                    plt.close()
                    gc.collect()
            c += 1             
            print(f'{c} of {len(eids)} done')
        except:
            c += 1             
            print(f'{c} of {len(eids)} failed')                                     
                     
   
'''
############## 
############## plotting
##############
''' 

def single_psd(eid, dd=None, regs=None,neurons=None, reg=None):

    
    duration = 2
    lag = -0.5   

    fs = int(1/T_BIN)

    if dd is None:
        dd, regs,neurons = trial_cut(eid, duration=duration ,
                                  lag=lag, pseudo=False, 
                                  include_licks=False)

      
    reg_in = list(regs).index(reg)

    # get from freshly loaded data
    ds = [[],[]]
    for i in dd:
        if i[1][1] == 1:
            ds[0].append(i[0][:,reg_in])
        else:
            ds[1].append(i[0][:,reg_in])    

    ds[0] = np.array(ds[0]).T
    ds[1] = np.array(ds[1]).T
    
    ntime,ntr_ex = ds[0].shape    
    ntime,ntr_unex = ds[1].shape    

    d = np.hstack(ds).T #stack ex, unex trials     
    
    ntr, ntime = d.shape

    fig,axs = plt.subplots(nrows=1, ncols=3,figsize=(9,3),
                           gridspec_kw={'width_ratios': [1,1,2]})
               
    k = 0     

    axs[k].imshow(d,cmap='Greys',aspect="auto",
        interpolation='none')#,vmin=minY,vmax=maxY)       


    axs[k].axvline(x=0.5/T_BIN,linestyle='--',color='g', 
                   label='stim on')
    axs[k].axhline(y=ntr_ex, linewidth=1, linestyle='-', c='b')
    axs[k].text(0,ntr_ex,'engaged trials above')    
    axs[k].set_xticks(np.linspace(0,ntime,5))
    axs[k].set_xticklabels(np.linspace(0,ntime,5)*T_BIN - 0.5)
    axs[k].set_title(f'mean neural firing rate')
    axs[k].set_xlabel('time [sec]')
    axs[k].set_ylabel('trials, ordered by time')
    axs[k].legend()
    plt.tight_layout()
    k+=1

    f, psd = signal.welch(d,fs=fs)

    axs[k].imshow(psd,cmap='Greys',aspect="auto",
        interpolation='none')#,vmin=minY,vmax=maxY)  
                    
    axs[k].set_xticks(np.linspace(0,len(f),5))
    axs[k].set_xticklabels(np.linspace(0,f[-1],5))              
    axs[k].set_title(f'psd (Welch) of neural')
    axs[k].set_xlabel('frequency [Hz]')
    axs[k].set_ylabel('trials, ordered by time')         
    axs[k].axhline(y=ntr_ex, linewidth=1, linestyle='-', 
                          c='b')       
    axs[k].text(0,ntr_ex,'engaged trials above')
    k+=1

    # doing fft directly per trial and then showing average
    # "roll your own Welch"
    ntr, nobs = d.shape
    yf = rfft(d)
    xf = rfftfreq(nobs, T_BIN)
    yf = np.mean(np.abs(yf)**2,axis=0)
    axs[k].plot(xf,np.log(yf))
    axs[k].set_xlabel('frequency [Hz]')
    axs[k].set_ylabel('log(<abs(fft(trial))^2>_trials)')
    axs[k].set_title(f'average spectrum across trials')    
    mouse_date = ' '.join(str(one.eid2path(eid)).split('/')[6:8])    
    plt.suptitle(f'{eid}, {mouse_date}, {reg}, #neu: {neurons[reg_in]}')
  
    plt.tight_layout() 


def plot_rasters(eids, new_spike=False):
    plt.ioff()
    for eid in eids:
    
        dsets = one.list_datasets(eid)
        r = [x.split('/') for x in dsets if 'probe' in x]
        rr = [item for sublist in r for item in sublist
              if 'probe' in item and '.' not in item]
                     
        if len(list(Counter(rr))) != 2:
            print("not two probes present, using one only")    
                         
                                
        for probe in list(Counter(rr)): 
            try:
                if new_spike:
                    spikes, clusters, channels = load_spike_sorting_fast(eid=eid, 
                                one=one,  probe=probe, spike_sorter='pykilosort')
                                
                    spikes = spikes[probe]
                    clusters = clusters[probe]            
                    channels = channels[probe] 
                    if spikes == {}:
                        print(f'!!! No new spike sorting for {probe} !!!')
                        return           
                
                else:
                    spikes = one.load_object(eid, 'spikes', 
                        collection=f'alf/{probe}',
                        attribute=['times','clusters','depths'])
                    clusters = one.load_object(eid, 'clusters', 
                        collection=f'alf/{probe}',
                        attribute=['channels'])
            
                driftmap(spikes.times, spikes.depths, t_bin=0.1, d_bin=5)
                plt.title(f'{eid},{probe}')
                plt.savefig(f'{eid}_{probe}.png')
                plt.close()
            except:
                continue



def lineplots_per_region(reg_):

    '''
    plot spectra as superimposed line plots for the same region
    '''

    s = ('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/'
         'granger/trial_average_psd.npy')

    r = np.load(s,allow_pickle=True).flat[0] 

    min_neu = 10  # discard regions with less than this neurons
    min_recordings = 3  # discard regions with less than three recordings
    

    regs = []
    for eid in r:
        for i in range(len(r[eid]['neurons'])):
            if r[eid]['neurons'][i] > min_neu:
                regs.append(r[eid]['regs'][i])
                
    f = dict(Counter(regs))            
    f2 = {k: v for k, v in sorted(f.items(), key=lambda item: item[1])}    
#    plt.bar(range(len(f2)), list(f2.values()), align='center')
#    plt.xticks(range(len(f2)), list(f2.keys()),fontsize=5,rotation=90)    
#    plt.title('#recordings from brain regions with at least 20 neurons; BWM')

    
    xf = rfftfreq(400, T_BIN)

    for reg in [reg_]:# f2:
        if f2[reg] > min_recordings:
        
            fig, ax = plt.subplots()
            k = 0 
            for eid in r:
                if reg in r[eid]['regs']:
                    i = r[eid]['regs'].index(reg)
                    mouse_date = ' '.join(str(one.eid2path(eid)).split('/')[6:8])
                    ax.plot(xf,r[eid]['psd'][i]+0.5*k,label=mouse_date)
#                    ax.plot(xf,zscore(r[eid]['psd'][i]),label=mouse_date,
#                            linewidth=0.1, c='k')
                    
                    k +=1
            ax.set_xlabel('f [Hz]')
            ax.set_ylabel('psd')
            ax.set_title(reg)
            plt.tight_layout()
#            plt.savefig('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/'
#                        f'granger/per_region_trial_average_psd/{reg}.png')    
#    plt.close()
    
    
def single_cell_psd(clus,eid,probe, plot_=True, axs=None, save=True):

    '''
    used in BWM SI figure on lick correlates
    '''
# pid = '63a32e5c-f63a-450d-85cb-140947b67eaf'
# eid = 'c4432264-e1ae-446f-8a07-6280abade813'
# probe = 'probe00'
# reg = 'APN'
# clus = 72


#    eid = 'fc43390d-457e-463a-9fd4-b94a0a8b48f5'
#    clus = 15
#    probe = 'probe00'
#    reg = 'VPL'   

    cc = one.eid2pid(eid)
    pid = cc[0][cc[-1].index('probe00')]

    bins, clus_ids, acs = get_single_cells_cut(eid, probe)
    fs = int(1/T_BIN)

    ntr, nclus, ntime = bins.shape

    clus = np.where(clus_ids == clus)[0][0]
    print(clus)
    if clus == []:
        print('check indices')
    

    d = bins[:,clus,:]
    
    ntr, ntime = d.shape
    f, psd = signal.welch(d,fs=fs)
    
    # doing fft directly per trial and then showing average
    # "roll your own Welch"
    ntr, nobs = d.shape
    yf = rfft(d)
    xf = rfftfreq(nobs, T_BIN)
    yf = np.mean(np.abs(yf)**2,axis=0)    

    if plot_:
        if axs is None:
            tight = True
            fig,axs = plt.subplots(nrows=3, ncols=1,figsize=(3,8))
        else:
            tight = False
            fig = axs[0].get_figure()
        #,gridspec_kw={'width_ratios': [1,1,2]})
                               
        axs = axs.flatten()
                      
        k = 0     

        # show only every third trial [::6]
        axs[k].imshow(d,cmap='Greys',aspect="auto",
            interpolation='none',vmin=0,vmax=0.0001)       

        axs[k].axvline(x=0.5/T_BIN,linestyle='--',color='g', label='fback')
      
        axs[k].set_xticks(np.linspace(0,ntime,5))
        axs[k].set_xticklabels(np.linspace(0,ntime,5)*T_BIN - 0.5)
        axs[k].set_title(f'Activity of example neuron')
        axs[k].set_xlabel('Time (s)')
        axs[k].set_ylabel('Trials - ordered by time')
        axs[k].legend(frameon=False, loc='lower center').set_draggable(True)


        k+=1

#        # psd Welch
#        axs[k].imshow(psd,cmap='Greys',aspect="auto",
#            interpolation='none')#,vmin=minY,vmax=maxY)  
#                        
#        axs[k].set_xticks(np.linspace(0,len(f),5))
#        axs[k].set_xticklabels(np.linspace(0,f[-1],5))              
#        axs[k].set_title(f'psd (Welch) of neural')
#        axs[k].set_xlabel('frequency [Hz]')
#        axs[k].set_ylabel('trials, ordered by time')         

#        k+=1

        axs[k].plot(xf,np.log(yf), c='k')
        axs[k].set_xlabel('Frequency (Hz)')
        #axs[k].set_ylabel('log(<abs(fft(trial))^2>_trials)')
        axs[k].set_ylabel('PSD (dB)')
        axs[k].set_title(f'Trial-averaged spectrum')    
        mouse_date = ' '.join(str(one.eid2path(eid)).split('/')[6:8])
        # axs[k].set_xlim(0.5,40)
        # axs[k].set_ylim(3.6,6)
        #plt.suptitle(f'{eid}, {mouse_date}, cluster {clus_ids[clus]}')      
        
        k+=1        
        compare_waveforms(pid, clus_ids[clus],ax=axs[k])
        axs[k].set_ylabel('Time (ms)')
        axs[k].set_title('Waveforms')

    if tight:
        plt.tight_layout()
    st,en = list(xf).index(3),list(xf).index(18)
    yf = np.log(yf)
    p10 = abs(np.mean(yf[st:en])) > 1.01*abs(np.mean(yf[en:]))
    #print(st,en,abs(np.mean(yf[st:en])), 1.01*abs(np.mean(yf[en:]))) 
    
    if save:
        imgs_pth = Path(one.cache_dir, 'bwm_res', 'bwm_figs_imgs','si')
        fig = plt.gcf()
        fig.savefig(imgs_pth / 'lick_SI_part.svg')
    
    return p10, clus_ids[clus]
      
    
def trial_spectrogram(eid, probe,reg,bins=None,
                      t=None, plot_=False):

#    # example with strong 10 Hz
#    eid = 'fc43390d-457e-463a-9fd4-b94a0a8b48f5'
#    reg = 'VPL'
#    probe = 'probe00'    


#    # example with strong 30 Hz
#    eid = 'f304211a-81b1-446f-a435-25e589fe3a5a'
#    reg = 'CA1'
#    probe = 'probe01'    


    if bins is None:
        bins,t, clus_ids = get_single_cells_cut(eid, probe, reg)
    fs = int(1/T_BIN)

    ntr, nclus, ntime = bins.shape

    x = bins.mean(axis=1)
    S = []
    for tr in x:
        f, t, Sxx = signal.spectrogram(tr, fs)
        S.append(Sxx)
        
        
    fig, ax = plt.subplots()    
    plt.pcolormesh(t - pre_time, f, np.array(S).mean(axis=0),
                   snap=True, cmap='Greys')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(f'{eid}, {probe}, {reg}')
    ax.axvline(x=0,linestyle='--',color='g', 
                   label='stim on')    
    plt.legend()
    plt.show()


def get_10Hz_cells(eid, reg, probe):

#    eid = 'fc43390d-457e-463a-9fd4-b94a0a8b48f5'
#    reg = 'VPL'
#    probe = 'probe00' 
   

    bins,t, clus_ids = get_single_cells_cut(eid, probe, reg)
    ntrials, ncells, ntime = bins.shape

    res = []
    for clus in range(ncells):
        res.append(single_cell_psd(bins,t,clus_ids,clus,eid,
                   reg, plot_=False))
            
    return res


#np.ndenumerate for the square
#def get_trace_waveform(eid, probe, clustID):


#    pid = one.alyx.rest('insertions', 'list', 
#                        session=eid, name=probe)[0]['id']


#    # Load in the spikesorting
#    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
#    spikes, clusters, channels = sl.load_spike_sorting()
#    clusters = sl.merge_clusters(spikes, clusters, channels)

#    # Load the spike waveforms
#    spike_wfs = one.load_object(sl.eid, '_phy_spikes_subset',
#                                collection=sl.collection)

#    # Find the cluster id for each sample waveform
#    wf_clusterIDs = spikes['clusters'][spike_wfs['spikes']]
#    
#    # Find waveforms for this cluster
#    wf_idx = np.where(wf_clusterIDs == clustID)[0]
#    wfs = spike_wfs['waveforms'][wf_idx, :, :]
#    Traces(wfs[:, :, 0])
#    plt.title(f'{eid}, {probe}, clus {clustID}')    


def compare_waveforms(pid, ic,ax=None):


#    eid = 'fc43390d-457e-463a-9fd4-b94a0a8b48f5'
#    probe = 'probe00'
#    ic = 263
#    pid = '3b729602-20d5-4be8-a10e-24bde8fc3092'
   
    eid, probe = one.pid2eid(pid)
    sl = SpikeSortingLoader(pid=pid, one=one)
    waveforms = sl.load_spike_sorting_object('waveforms', revision='2024-05-06')
    #templates = waveforms['templates']
    
    templates = one.load_object(eid, 'templates',
                            collection=sl.collections[-1])
                                
    tr = templates['waveformsChannels'][ic]
    wav = templates['waveforms'][ic]
    trind = np.argsort(tr)
    
    if ax is None:
        tight = True
        fig, ax = plt.subplots(figsize=(3.5,3))
    else:
        tight = False
    
    Traces(wav[:, trind], ax=ax, fs=30000)
    #plt.imshow(wav[:, trind],aspect='auto')

    if tight:
        plt.title(f'{eid} \n {probe}, clus {ic}', fontsize=9)
        plt.tight_layout()
#    plt.savefig(f'traces/traces_{eid}_{probe}_{ic}.png')
#    plt.close()


def get_waveforms_metrics():

    # example cells

    cells = [
    ['02fbb6da-3034-47d6-a61b-7d06c796a830', 'probe00', 'CA1', 262],
    ['f9860a11-24d3-452e-ab95-39e199f20a93', 'probe01', 'CA2', 120],
    ['f9860a11-24d3-452e-ab95-39e199f20a93', 'probe01', 'CA2', 114],
    ['5339812f-8b91-40ba-9d8f-a559563cc46b', 'probe00', 'CA3', 270],
    ['5339812f-8b91-40ba-9d8f-a559563cc46b', 'probe00', 'CA3', 269],
    ['5339812f-8b91-40ba-9d8f-a559563cc46b', 'probe00', 'CA3', 272],
    ['fc43390d-457e-463a-9fd4-b94a0a8b48f5','probe00', 'VPL', 172],
    ['fc43390d-457e-463a-9fd4-b94a0a8b48f5','probe00', 'VPL', 56],
    ['fc43390d-457e-463a-9fd4-b94a0a8b48f5','probe00', 'VPL', 5],
    ['fc43390d-457e-463a-9fd4-b94a0a8b48f5','probe00', 'VPL', 7]]
    

    columns = ['eid', 'probe', 'ic', 'w', 'min_max', 'sRMS', 'reg']
    r = []
    for cell in cells:
        eid, probe, reg, ic = cell

        pid = one.alyx.rest('insertions', 'list', 
                            session=eid, name=probe)[0]['id']
                            
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)   
        templates = one.load_object(eid, 'templates',
                                    collection=sl.collections[-1], revision='2024-05-06')
                                    
        peth, clus, acs = get_psths_singlecell_lick_aligned(eid, probe)
        
        #check that cluster neighbors exist
        assert clus[-1] == len(clus)- 1, 'cluster inconsistent'
        
        for k in [-1, 0, +1]:
        
            reg = acs[list(clus).index(ic + k)]
                                  
            tr = templates['waveformsChannels'][ic + k]
            wav = templates['waveforms'][ic + k]
            trind = np.argsort(tr)
            w = wav[:, trind]
            
            # compute shifted RMS, sRMS
            
            w_ = w - np.roll(w, -4, axis=1)
            x = w.flatten()
            x_ = w_.flatten()
            
            sRMS = (np.mean(x_**2)**0.5)/(np.mean(x**2)**0.5)            
            
            # compute PETH max-min
            u = peth[list(clus).index(ic + k)]
            min_max = (max(u) - min(u))/(max(u) + min(u) + 0.01)
            r.append([eid, probe, ic + k, w, min_max, sRMS, reg])   
    
    
    df  = pd.DataFrame(data=r,columns=columns)
    return df

#    df.to_pickle('/home/mic/waveforms.pkl')


def lick_lock_sRMS(eid):

    # for all cells of this list of eids

    #eids = ['f1db6257-85ef-4385-b415-2d078ec75df2']

    #'f10efe41-0dc0-44d0-8f26-5ff68dca23e9', 'eebacd5a-7dcd-4ba6-9dff-ec2a4d2f19e0'
    columns = ['eid', 'probe', 'ic', 'w', 'min_max', 'sRMS', 'reg']
    
    
    #for eid in eids:    
    
    dsets = one.list_datasets(eid)
    r = [x.split('/') for x in dsets if 'probe' in x]
    rr = [item for sublist in r for item in sublist
          if 'probe' in item and '.' not in item]
    probes = list(Counter(rr))
    
    
    r = []
    for probe in probes:

        sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        templates = one.load_object(eid, 'templates',
                                    collection=sl.collections[-1], revision='2024-05-06')
                                    
        peth, clus, acs = get_psths_singlecell_lick_aligned(eid, probe)

        assert (len(peth) == len(clus) == 
               len(acs) == len(templates['waveformsChannels']))
        
        print(probe, eid, Counter(acs))
            
        for k in range(len(clus)):
        
            reg = acs[k]
                                  
            tr = templates['waveformsChannels'][k]
            wav = templates['waveforms'][k]
            trind = np.argsort(tr)
            w = wav[:, trind]
            
            # compute shifted RMS, sRMS
            w_ = w - np.roll(w, -4, axis=1)
            x = w.flatten()
            x_ = w_.flatten()
            sRMS = (np.mean(x_**2)**0.5)/(np.mean(x**2)**0.5)            
            
            # compute PETH max-min
            u = peth[k]
            min_max = (max(u) - min(u))/(max(u) + min(u) + 0.01)
            r.append([eid, probe, clus[k], w, min_max, sRMS, reg])   
    
    
    df  = pd.DataFrame(data=r,columns=columns)
    df.to_pickle(f'/home/mic/waveforms_{eid}.pkl')
    return df


def scatter_metrics(df=None, eid=None):

    if eid is not None:
        if df is None:
            df = np.load(f'/home/mic/waveforms_{eid}.pkl',allow_pickle=True)
    
    _, palette = get_allen_info()
    df.plot.scatter('min_max','sRMS', 
                    color = [palette[reg] for reg in df['reg'].values])
    ax = plt.gca()
    for i in range(len(df)):
        ax.annotate(str(df.iloc[i]['ic']),
                    (df.iloc[i]['min_max'],df.iloc[i]['sRMS']),
                    fontsize=10,color='k')
                    
                    
    le_labs = [Patch(facecolor = palette[reg], 
           edgecolor = palette[reg],
           label = reg) for reg in Counter(df['reg'].values)]
           
    plt.tight_layout()               
    ax.legend(handles=le_labs,loc='best',
                  ncol=1, frameon=False,
                  prop={'size': 5}).set_draggable(True)
    r, p = pearsonr(df['min_max'].values, df['sRMS'].values)
    ax.set_title(f'{eid}, pearson={np.round(r,3)}')              
    plt.tight_layout()              
                  

def grid_plot(df):


    fig, axs = plt.subplots(nrows=3, ncols= len(df)//3,figsize=(13,8), 
                            sharex=True, sharey=True)  
    axs = axs.flatten()
    
    for k in range(len(df)):
    
        w = df.iloc[k]['w']
        nech, ntr = w.shape
        tscale = np.arange(nech)
        sf = 0.71 / np.mean((w.flatten())**2)**0.5
        axs[k].plot(w * sf + np.arange(ntr), tscale, 
                                   color = 'k',linewidth=0.5)
        axs[k].set_xlim(-1, ntr + 1)
        axs[k].set_ylim(tscale[0], tscale[-1])
        
        s = (df.iloc[k]['eid'][:3] +'...'+', '+df.iloc[k]['probe']+
             '\n  id:'+str(df.iloc[k]['ic']) +', '+ str(df.iloc[k]['reg']) +
             '\n licklock:'+ str(np.round(df.iloc[k]['min_max'],2))
             +', sRMS:'+str(np.round(df.iloc[k]['sRMS'],2)))
         
        axs[k].set_title(s, fontsize=7)
        

    plt.tight_layout()
    
   


def plot_lick_peth(eid,probe):

    pre_time = 0.075
    post_time = 0.075
    xs = np.arange((pre_time + post_time)/T_BIN)*T_BIN   

    w, clus, acs = get_psths_singlecell_lick_aligned(eid, probe)
    
    dfa, palette = get_allen_info()
    
    fig, axs = plt.subplots(nrows=1,ncols=3, figsize=(8,3))
    
    im = axs[1].imshow(w,aspect='auto', cmap='Greys',interpolation='none',
               extent = [- pre_time, post_time, clus[0] , clus[-1]])
    for k in range(2):           
        axs[k].axvline(x=0,linestyle='--',color='r',label='lick')
        axs[k].set_xlabel('time [sec]')

    axs[1].set_ylabel('neuron number')
    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)            
    cbar.set_label('PETH firing rate')    
    
    # line plots with threshold
    
    minmax = []
    for i in range(len(w)):
        minmax.append([i, max(w[i]) - min(w[i])])
        
        if np.mean(w[i]) > 0.1:
            axs[0].plot(xs - pre_time, w[i], label = clus[i],
                        c = palette[acs[i]], linewidth=0.5)
        

    fig.suptitle(f'Lick-locked PETHs \n {eid}, {probe}')     

    # bar plot of max-min per cell, ordered, colored by allen
    ids = np.array([x[0] for x in minmax])
    scs = np.array([x[1] for x in minmax])
    
    ids = ids[np.argsort(scs)]
    scs = scs[np.argsort(scs)]
    cols = [palette[acs[i]] for i in ids]
    
    axs[2].bar(range(len(scs)), scs,color = cols)
    axs[2].set_xticks(range(len(scs)))
    axs[2].set_xticklabels(ids, rotation=90)
    axs[2].set_xlabel('clsuter ids')
    axs[2].set_ylabel('max(PETH) - min(PETH)')   
    axs[2].axhline(y=0.15, linestyle ='--', color='k')
    
    xlim = np.where(scs > 0.15)[0][0]
    axs[2].set_xlim(left = xlim)

    le_labs = [Patch(facecolor = palette[reg], 
           edgecolor = palette[reg],
           label = reg) for reg in Counter(acs)]
           
    plt.tight_layout()               
    axs[2].legend(handles=le_labs,loc='best',
                  ncol=1, frameon=False,
                  prop={'size': 5}).set_draggable(True)
          
    return minmax, clus    












