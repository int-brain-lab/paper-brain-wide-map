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

from matplotlib.lines import Line2D
import sys
sys.path.append('Dropbox/scripts/IBL/')
from dmn_bwm import get_allen_info


mpl.rcParams.update({'font.size': 10})

one = ONE()
    # base_url='https://openalyx.internationalbrainlab.org',
    #       password='international', silent=True)

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


def bwm_data_series_fig(cnew=True, alter=True, sort_reg=True):
    """
    Plot a behavioral time series on block-colored background
    above neural data - intro figure.

    Parameters
    ----------
    cnew : bool
    alter : bool
        If True, plot neural activity as a true binary raster image.
        If False, plot binned population activity as a grayscale image.
    sort_reg : bool
        True  -> depth sorting (GitHub default)
        False -> no sorting (np.unique ordering, Dropbox-like)
    """

    eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'
    video_type = 'left'
    probe = 'probe00'
    trial_range = [242, 244]
    nsubplots = 4

    trials = one.load_object(eid, 'trials')
    tstart = trials['intervals'][trial_range[0]][0]
    tend = trials['intervals'][trial_range[-1]][-1]

    fig, axs = plt.subplots(
        nsubplots, 1,
        figsize=(5.98, 6.91),
        sharex=True,
        gridspec_kw={'height_ratios': [1, 1, 1, 4], 'hspace': 0.06}
    )

    # -------------------------
    # COMPUTE / LOAD CACHED TIME SERIES
    # -------------------------
    if cnew:
        Q = []

        sl_spk = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        spikes, clusters, channels = sl_spk.load_spike_sorting()
        clusters = sl_spk.merge_clusters(spikes, clusters, channels)

        # --- wheel ---
        wheel = one.load_object(eid, 'wheel')
        pos, times_w = wh.interpolate_position(
            wheel.timestamps, wheel.position, freq=1 / T_BIN
        )
        v = np.append(np.diff(pos), np.diff(pos)[-1])

        s_idx = find_nearest(times_w, tstart)
        e_idx = find_nearest(times_w, tend)

        x0 = times_w[s_idx:e_idx]
        y0 = v[s_idx:e_idx]
        Q.append([x0, y0])

        # --- whisker ---
        times_me, ME = get_ME(eid, video_type)
        s_idx = find_nearest(times_me, tstart)
        e_idx = find_nearest(times_me, tend)

        x1 = times_me[s_idx:e_idx]
        y1 = ME[s_idx:e_idx]
        Q.append([x1, y1])

        # --- licking ---
        sess_loader = SessionLoader(one=one, eid=eid)
        sess_loader.load_pose(views=['left', 'right'])

        dlc_left = sess_loader.pose['leftCamera']
        dlc_right = sess_loader.pose['rightCamera']

        lick_times = []
        for dlc in [dlc_left, dlc_right]:
            r = get_licks(dlc)
            lick_times.append(dlc['times'][r])

        lick_times = np.sort(np.concatenate(lick_times))
        R, times_l, _ = bincount2D(
            lick_times, np.ones(len(lick_times)), T_BIN
        )
        lcs = R[0]

        s_idx = find_nearest(times_l, tstart)
        e_idx = find_nearest(times_l, tend)

        x2 = times_l[s_idx:e_idx]
        y2 = lcs[s_idx:e_idx]
        Q.append([x2, y2])

        # --- neural (binned activity image for alter=False) ---
        Rn, times_n, _ = bincount2D(
            spikes['times'], spikes['clusters'], T_BIN
        )

        s_idx = find_nearest(times_n, tstart)
        e_idx = find_nearest(times_n, tend)

        x3 = Rn[:, s_idx:e_idx]
        y3 = [times_n[s_idx], times_n[e_idx]]
        Q.append([x3, y3])

    else:
        Q = np.load(pth_res / 'Q.npy', allow_pickle=True)
        x0, y0 = Q[0]
        x1, y1 = Q[1]
        x2, y2 = Q[2]
        x3, y3 = Q[3]

    # -------------------------
    # PLOT TIME SERIES
    # -------------------------
    axs[0].plot(x0, y0, c='k', linewidth=0.5, label='wheel velocity')
    axs[0].set_ylabel('wheel velocity')

    axs[1].plot(x1, y1, c='k', linewidth=0.5, label='whisking')
    axs[1].set_ylabel('whisking')

    axs[2].plot(x2, y2, c='k', linewidth=0.5, label='licking')
    axs[2].set_ylabel('licking')

    # -------------------------
    # NEURAL PANEL
    # -------------------------
    if alter:
        sl_spk = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        spikes, clusters, channels = sl_spk.load_spike_sorting()
        clusters = sl_spk.merge_clusters(spikes, clusters, channels)

        # restrict time using boolean mask for exact windowing
        mask = (spikes['times'] >= tstart) & (spikes['times'] <= tend)
        times_s = spikes['times'][mask]
        clus_s = spikes['clusters'][mask]

        # Dropbox-like baseline: np.unique ordering
        uclus = np.unique(clus_s)

        if sort_reg:
            depthd = dict(zip(clusters['cluster_id'], clusters['depths']))
            uclus = np.array([c for c in uclus if c in depthd])
            if len(uclus) > 0:
                depthl = np.array([depthd[c] for c in uclus])
                uclus = uclus[np.argsort(depthl)]

        # map cluster -> row
        clus2row = {c: i for i, c in enumerate(uclus)}

        # binary time-neuron raster
        dt = T_BIN
        t_bins = np.arange(tstart, tend + dt, dt)
        n_t = len(t_bins) - 1
        n_n = len(uclus)

        M = np.zeros((n_n, n_t), dtype=np.uint8)

        if n_n > 0 and len(times_s) > 0:
            t_idx = np.searchsorted(t_bins, times_s, side='right') - 1
            valid = (t_idx >= 0) & (t_idx < n_t)

            for tbin, clu in zip(t_idx[valid], clus_s[valid]):
                row = clus2row.get(clu, None)
                if row is not None:
                    M[row, tbin] = 1

        axs[3].imshow(
            M,
            aspect='auto',
            cmap='binary',              # 0 white, 1 black
            interpolation='nearest',   # no smoothing
            extent=[tstart, tend, 0, n_n],
            origin='lower',
            vmin=0,
            vmax=1
        )
        axs[3].set_ylabel('neuron')

    else:
        nneu, _ = x3.shape
        axs[3].imshow(
            x3, aspect='auto', cmap='Greys',
            extent=[y3[0], y3[1], 0, nneu],
            vmin=0, vmax=2, origin='lower'
        )
        axs[3].set_ylabel('neuron')

    # -------------------------
    # TRIAL STRUCTURE
    # -------------------------
    y0_top = np.nanmax(y0) if len(y0) else 0.1
    y0_rng = (np.nanmax(y0) - np.nanmin(y0)) if len(y0) else 1.0
    y0_text = y0_top + 0.08 * (y0_rng if y0_rng > 0 else 1.0)

    for i in range(trial_range[0], trial_range[-1] + 1):
        st = trials['intervals'][i][0]

        if np.isnan(trials['contrastLeft'][i]):
            side = 'r'
        else:
            side = 'l'

        choi = {1: 'l', -1: 'r'}.get(trials['choice'][i], 'n/a')
        ftype = trials['feedbackType'][i]

        txt = f'Trial: {i}\nstim: {side}\nchoice: {choi}\ncorrect: {ftype}'
        axs[0].text(st, y0_text, txt, va='bottom', fontsize=8)

        for k in range(nsubplots):
            if not np.isnan(trials['stimOn_times'][i]):
                axs[k].axvline(
                    trials['stimOn_times'][i],
                    color='k', linestyle='--', linewidth=0.8,
                    label='stimOn' if (i == trial_range[0] and k == 0) else None
                )
            if not np.isnan(trials['feedback_times'][i]):
                axs[k].axvline(
                    trials['feedback_times'][i],
                    color='k', linestyle=':', linewidth=0.8,
                    label='feedback' if (i == trial_range[0] and k == 0) else None
                )

    # -------------------------
    # COSMETICS
    # -------------------------
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for ax in axs[:-1]:
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    axs[-1].set_xlabel('time [sec]')

    legend_handles = [
        Line2D([0], [0], color='k', linestyle='--', linewidth=0.8, label='stimOn'),
        Line2D([0], [0], color='k', linestyle=':',  linewidth=0.8, label='feedback'),
    ]

    axs[-1].legend(handles=legend_handles, frameon=False, loc='upper right')

    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.08, top=0.96, hspace=0.08)

    if cnew:
        np.save(pth_res / 'Q.npy', np.array(Q, dtype=object), allow_pickle=True)

    return fig, axs
        
        
# if __name__ == "__main__":        

#     '''
#     Create panel of intro figure of the brain-wide-map paper
#     illustrating time series and neural data for
#     3 example trials
#     '''
#     eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'
#     bwm_data_series_fig()
