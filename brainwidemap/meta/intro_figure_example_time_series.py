from one.api import ONE
from brainbox.processing import bincount2D
from ibllib.atlas import AllenAtlas
from ibllib.atlas.regions import BrainRegions
from brainbox.io.one import SpikeSortingLoader
from brainbox.io.one import SessionLoader
import ibllib

import numpy as np
from pathlib import Path
import math
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import pickle


mpl.rcParams.update({'font.size': 10})

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international')
eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'

# save results for plotting here
pth_res = Path(one.cache_dir, 'brain_wide_map', 'motor_correlates')
pth_res.mkdir(parents=True, exist_ok=True)

ba = AllenAtlas()
br = BrainRegions()

T_BIN = 0.02
Fs = {'left': 60, 'right': 150, 'body': 30}

blue_left = [0.13850039, 0.41331206, 0.74052025]
red_right = [0.66080672, 0.21526712, 0.23069468]
cdi = {0.8: blue_left, 0.2: red_right, 0.5: 'g', -1: 'cyan', 1: 'orange'}


def get_allen_info():
    '''
    Function to load Allen region info, e.g. region colors palette
    '''

    p = (Path(ibllib.__file__).parent /
         'atlas/allen_structure_tree.csv')

    dfa = pd.read_csv(p)

    # get colors per acronym and transfomr into RGB
    dfa['color_hex_triplet'] = dfa['color_hex_triplet'].fillna('FFFFFF')
    dfa['color_hex_triplet'] = dfa['color_hex_triplet'
                                   ].replace('19399', '19399a')
    dfa['color_hex_triplet'] = dfa['color_hex_triplet'] .replace('0', 'FFFFFF')
    dfa['color_hex_triplet'] = '#' + dfa['color_hex_triplet'].astype(str)
    dfa['color_hex_triplet'] = dfa['color_hex_triplet'
                                   ].apply(lambda x:
                                           mpl.colors.to_rgba(x))

    palette = dict(zip(dfa.acronym, dfa.color_hex_triplet))

    return dfa, palette


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


'''
############
Overleaf BWM intro figure
############
'''


def example_block_structure(sl):
    '''
    illustrate block structure in a line plot
    :param sl: SessionLoader instance.
    '''
    fig, ax = plt.subplots(figsize=(2, 1))
    plt.plot(sl.trials['probabilityLeft'], color='k',
             linestyle='', marker='|', markersize=0.1)
    ax.set_xlabel('trials')
    ax.set_yticks([0.2, 0.5, 0.8])
    ax.set_ylabel('p(stim left)')
    plt.tight_layout()
    plt.show()


def bwm_data_series_fig(sl, cnew=True):
    '''
    plot a behavioral time series on block-colored background
    above neural data - intro figure
    :param sl: SessionLoader instance.
    '''

    video_type = 'left'
    # reg = 'PRNr'#'SCiw''MRN'
    probe = 'probe00'
    trial_start = 242
    trial_end = 244
    nsubplots = 4

    trials = sl.trials
    # start time of first trial
    tstart = trials['intervals_0'][trial_start]
    # end time of last trial
    tend = trials['intervals_1'][trial_end]

    fig, axs = plt.subplots(nsubplots, 1, figsize=(5, 4), sharex=True,
                            gridspec_kw={'height_ratios': [1, 1, 1, 4]})
    # plot wheel speed trace
    # axs.append(plt.subplot(nsubplots,1,1))

    if cnew:
        Q = []
        wheel = sl.wheel
        pos, times_w = wheel['times'].to_numpy(),wheel['position'].to_numpy()

        v = np.append(np.diff(pos), np.diff(pos)[-1])

        s_idx = find_nearest(times_w, tstart)
        e_idx = find_nearest(times_w, tend)

        x = times_w[s_idx:e_idx]
        y = v[s_idx:e_idx]

        Q.append([times_w[s_idx:e_idx], v[s_idx:e_idx]])

    else:
        Q = pickle.load(open(pth_res / 'precomputed.p', "rb"))
        x, y = Q[0]

    axs[0].plot(x, y, c='k', label='wheel velocity', linewidth=0.5)
    axs[0].set_ylabel('wheel velocity')  # \n [rad/sec]
    axs[0].axes.yaxis.set_visible(False)
    # plot whisker motion
    # axs.append(plt.subplot(nsubplots,1,2,sharex=axs[0]))

    if cnew:
        # when changing cam type, change fs in cut
        me_data = sl.motion_energy[f'{video_type}Camera'][['times','whiskerMotionEnergy']].to_numpy()
        times_me, ME = me_data[:, 0], me_data[:, 1]

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
        # load DLC
        sl.load_pose(views=['left', 'right'])
        dlc_left = sl.pose['leftCamera']
        dlc_right = sl.pose['rightCamera']

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

        # Load in spikesorting
        sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)

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
    for i in range(trial_start, trial_end + 1):
        st = trials['intervals_0'][i]
        en = trials['intervals_1'][i]

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
            axs[k].axvspan(st, en, facecolor=cols[str(pl)],
                           alpha=0.1, label=f'p(l)={str(pl)}')
            axs[k].axvline(x=trials['stimOn_times'][i], color='magenta',
                           linestyle='--', label='stimOn')
            axs[k].axvline(x=trials['feedback_times'][i], color='blue',
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
                   by_label.keys(), ncol=4).set_draggable(True)

    plt.show()
    
    if cnew:
        pickle.dump(Q, open(pth_res / 'precomputed.p', "wb"))


if __name__ == "__main__":        

    '''
    Create panel of intro figure of the brain-wide-map paper
    illustrating time series and neural data for
    3 example trials
    '''
    sl = SessionLoader(one=one, eid=eid)
    sl.load_session_data()

    example_block_structure(sl)
    bwm_data_series_fig(sl)
