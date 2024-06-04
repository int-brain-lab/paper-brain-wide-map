from one.api import ONE
import brainbox.behavior.wheel as wh
from brainbox.processing import bincount2D
from iblatlas.atlas import AllenAtlas, BrainRegions
from brainbox.io.one import SpikeSortingLoader
from brainbox.io.one import SessionLoader
import ibllib

import numpy as np
from pathlib import Path
import math
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

import sys
sys.path.append('Dropbox/scripts/IBL/')
from dmn_bwm import get_allen_info


mpl.rcParams.update({'font.size': 10})

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)

# save results for plotting here
pth_res = Path(one.cache_dir, 'bwm_res', 'si')
pth_res.mkdir(parents=True, exist_ok=True)

ba = AllenAtlas()
br = BrainRegions()

T_BIN = 0.02
Fs = {'left': 60, 'right': 150, 'body': 30}


blue_left = [0.13850039, 0.41331206, 0.74052025]
red_right = [0.66080672, 0.21526712, 0.23069468]
cdi = {0.8: blue_left, 0.2: red_right, 0.5: 'g', -1: 'cyan', 1: 'orange'}



def find_nearest(array, value):
    '''
    Find the index of the array, such that the
    value is closest to that
    '''

    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(
            value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
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
            c = dlc[point + '_' + co]
            thr = np.nanstd(np.diff(c)) / 4
            licks.append(set(np.where(abs(np.diff(c)) > thr)[0]))
    return sorted(list(set.union(*licks)))


def get_ME(eid, video_type, query_type='remote'):
    '''
    load motion energy for a given session (eid)
    and video type (e.g. video_type = 'left')

    returns:

    Times: time stamps of motion energy
    ME: motion energy time series
    '''

    Times = one.load_dataset(eid, f'alf/_ibl_{video_type}Camera.times.npy',
                             query_type=query_type)
    ME = one.load_dataset(eid, f'alf/{video_type}Camera.ROIMotionEnergy.npy',
                          query_type=query_type)

    return Times, ME


'''
############
Overleaf BWM intro figure
############
'''


def example_block_structure(eid):
    '''
    illustrate block structure in a line plot

    eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'
    '''

    trials = one.load_object(eid, 'trials')

    fig, ax = plt.subplots(figsize=(2, 1))
    plt.plot(trials['probabilityLeft'], color='k',
             linestyle='', marker='|', markersize=0.1)
    ax.set_xlabel('trials')
    ax.set_yticks([0.2, 0.5, 0.8])
    ax.set_ylabel('p(stim left)')
    plt.tight_layout()


def bwm_data_series_fig(cnew=True, alter=True):
    '''
    plot a behavioral time series on block-colored background
    above neural data - intro figure
    '''

    eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'
    video_type = 'left'
    # reg = 'PRNr'#'SCiw''MRN'
    probe = 'probe00'
    trial_range = [242, 244]
    nsubplots = 4

    trials = one.load_object(eid, 'trials')
    tstart = trials['intervals'][trial_range[0]][0]
    tend = trials['intervals'][trial_range[-1]][-1]

    fig, axs = plt.subplots(nsubplots, 1, figsize=(5.98, 6.91), sharex=True,
                            gridspec_kw={'height_ratios': [1, 1, 1, 4]})
    # plot wheel speed trace
    # axs.append(plt.subplot(nsubplots,1,1))

    if cnew:
        Q = []
        wheel = one.load_object(eid, 'wheel')
        pos, times_w = wh.interpolate_position(wheel.timestamps,
                                               wheel.position, freq=1 / T_BIN)

        v = np.append(np.diff(pos), np.diff(pos)[-1])

        s_idx = find_nearest(times_w, tstart)
        e_idx = find_nearest(times_w, tend)

        x = times_w[s_idx:e_idx]
        y = v[s_idx:e_idx]

        Q.append([times_w[s_idx:e_idx], v[s_idx:e_idx]])

    else:
        Q = np.load(pth_res / 'Q.npy', allow_pickle=True)
        x, y = Q[0]

    axs[0].plot(x, y, c='k', label='wheel speed', linewidth=0.5)
    axs[0].set_ylabel('wheel speed')  # \n [rad/sec]
    axs[0].axes.yaxis.set_visible(False)
    # plot whisker motion
    # axs.append(plt.subplot(nsubplots,1,2,sharex=axs[0]))

    if cnew:
        # when changing cam type, change fs in cut
        times_me, ME = get_ME(eid, video_type)

        s_idx = find_nearest(times_me, tstart)
        e_idx = find_nearest(times_me, tend)

        x, y = times_me[s_idx:e_idx], ME[s_idx:e_idx]
        Q.append([x, y])

    else:
        x, y = Q[1]

    axs[1].plot(x, y, c='k', label='whisker ME', linewidth=0.5)
    axs[1].set_ylabel('whisking')  # \n [a.u.]
    axs[1].axes.yaxis.set_visible(False)
    # plot licks
    # axs.append(plt.subplot(nsubplots,1,3,sharex=axs[0]))

    if cnew:

        sess_loader = SessionLoader(one, eid)

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
        R, times_l, _ = bincount2D(lick_times,
                                   np.ones(len(lick_times)), T_BIN)
        lcs = R[0]

        s_idx = find_nearest(times_l, tstart)
        e_idx = find_nearest(times_l, tend)

        x, y = times_l[s_idx:e_idx], lcs[s_idx:e_idx]
        Q.append([x, y])

    else:
        x, y = Q[2]

    axs[2].plot(x, y, c='k', linewidth=0.5)
    axs[2].set_ylabel('licking')  # \n [a.u.]
    axs[2].axes.yaxis.set_visible(False)

    # neural activity
    # axs.append(plt.subplot(nsubplots,1,4,sharex=axs[0]))

    if cnew:

  

        R, times_n, Clusters = bincount2D(
            spikes['times'], spikes['clusters'], T_BIN)
        D = R.T

        acs = br.id2acronym(clusters['atlas_id'], mapping='Beryl')
        _, palette = get_allen_info()
        cols_ = [palette[x] for x in acs]

        s_idx = find_nearest(times_n, tstart)
        e_idx = find_nearest(times_n, tend)

        D = D[s_idx:e_idx]  # ,m_ask]
        D = D.T

        x, y = D, [times_n[s_idx], times_n[e_idx]]
        Q.append([x, y])
        
        

         
        

    else:
        x, y = Q[3]


    if alter:
    
        # Load in spikesorting
        sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
    
        # Alternative plotting using single spikes
        
        s_idx = find_nearest(spikes['times'], tstart)
        e_idx = find_nearest(spikes['times'], tend)
        times_s = spikes['times'][s_idx: e_idx]
        clus_s = spikes['clusters'][s_idx: e_idx]

        adict = {}
        uclus = np.unique(clus_s)
        
        # order by depth
        depthd = dict(zip(clusters['cluster_id'],
                          clusters['depths']))
                          
        depthl = [depthd[c] for c in uclus]                  
        uclus = uclus[np.argsort(depthl)]
        
        
        # assume cluster are in depth order
        # re-index for plotting
        k = 0
        for c in uclus:
            adict[k] = times_s[clus_s == c]     
            k += 1
        
        x_values = []
        y_values = []

        for key, timestamps in adict.items():
            x_values.extend(timestamps)
            y_values.extend([key] * len(timestamps))   
        
        axs[3].scatter(x_values, y_values, marker='|',s=1, color='k')

                
                
    else:

        nneu, nobs = x.shape
        axs[3].imshow(x, aspect='auto', cmap='Greys',
                      extent=[y[0], y[1], 0, nneu], vmin=0, vmax=2)

        axs[3].set_ylabel('neuron')
        
    if cnew:
        for i in range(len(cols_)):
            axs[3].plot([960.76, 960.90], [i, i], color=cols_[i])

    # correct, incorrect, stimulus contrast, choice, stim side

    cols = {'0.8': [0.13850039, 0.41331206, 0.74052025],
            '0.2': [0.66080672, 0.21526712, 0.23069468], '0.5': 'g'}
    for i in range(trial_range[0], trial_range[-1] + 1):
        st = trials['intervals'][i][0]
        en = trials['intervals'][i][1]

        pl = trials['probabilityLeft'][i]

        if np.isnan(trials['contrastLeft'][i]):
            side = 'r'  # right side stimulus
        else:
            side = 'l'  # left side stimulus

        choi = {1: 'l', -1: 'r'}[trials['choice'][i]]
        ftype = trials['feedbackType'][i]
        s = f'Trial: {i} \n stim: {side} \n choice: {choi} \n correct: {ftype}'
        axs[0].text(st, 0.1, s)

        for k in range(nsubplots):
#            axs[k].axvspan(st, en, facecolor=cols[str(pl)],
#                           alpha=0.1, label=f'p(l)={str(pl)}')
            axs[k].axvline(x=trials['stimOn_times'][i], color='k',
                           linestyle='--', label='stimOn')
            axs[k].axvline(x=trials['feedback_times'][i], color='k',
                           linestyle='--', label='feedback')

    [axs[k].get_xaxis().set_visible(False) for k in range(len(axs[1:]))]
    axs[-1].set_xlabel('time [sec]')
    [ax.axis("off") for ax in axs]
#    fig.subplots_adjust(top=0.812,
#                        bottom=0.176,
#                        left=0.191,
#                        right=0.985,
#                        hspace=0.15,
#                        wspace=0.2)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[-1].legend(by_label.values(),
                   by_label.keys(), ncol=4).set_draggable(True)

    if cnew:
        np.save(pth_res / 'Q.npy', Q, allow_pickle=True)
        
        
if __name__ == "__main__":        

    '''
    Create panel of intro figure of the brain-wide-map paper
    illustrating time series and neural data for
    3 example trials
    '''
    sl = SessionLoader(one=one, eid=eid)
    sl.load_session_data()
