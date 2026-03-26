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


def bwm_data_series_fig(cnew=True, alter=True):
    """
    Plot a behavioral time series on block-colored background
    above neural data - intro figure.

    Parameters
    ----------
    cnew : bool
        If True, recompute and cache the behavioral/neural traces.
        If False, load them from pth_res / 'Q.npy'.
    alter : bool
        If True, plot neural activity as a spike raster.
        If False, plot binned population activity as an image.
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
        nsubplots, 1, figsize=(5.98, 6.91), sharex=True,
        gridspec_kw={'height_ratios': [1, 1, 1, 4]}
    )

    if cnew:
        Q = []

        # -------------------------
        # Load spike sorting first
        # -------------------------
        sl_spk = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        spikes, clusters, channels = sl_spk.load_spike_sorting()
        clusters = sl_spk.merge_clusters(spikes, clusters, channels)

        # -------------------------
        # 1) Wheel speed
        # -------------------------
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

        # -------------------------
        # 2) Whisker motion energy
        # -------------------------
        times_me, ME = get_ME(eid, video_type)
        s_idx = find_nearest(times_me, tstart)
        e_idx = find_nearest(times_me, tend)

        x1 = times_me[s_idx:e_idx]
        y1 = ME[s_idx:e_idx]
        Q.append([x1, y1])

        # -------------------------
        # 3) Licking
        # -------------------------
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

        # -------------------------
        # 4) Neural activity (binned)
        # -------------------------
        Rn, times_n, cluster_ids = bincount2D(
            spikes['times'], spikes['clusters'], T_BIN
        )

        # Rn shape is (n_clusters_present, n_timebins)
        D = Rn

        s_idx = find_nearest(times_n, tstart)
        e_idx = find_nearest(times_n, tend)

        D_cut = D[:, s_idx:e_idx]
        x3 = D_cut
        y3 = [times_n[s_idx], times_n[e_idx]]

        # colors for present cluster ids only
        _, palette = get_allen_info()
        acs = br.id2acronym(clusters['atlas_id'], mapping='Beryl')
        cid2acronym = dict(zip(clusters['cluster_id'], acs))
        cols_ = [palette[cid2acronym[cid]] for cid in cluster_ids if cid in cid2acronym]

        Q.append([x3, y3])

    else:
        Q = np.load(pth_res / 'Q.npy', allow_pickle=True)

        x0, y0 = Q[0]
        x1, y1 = Q[1]
        x2, y2 = Q[2]
        x3, y3 = Q[3]
        cols_ = None

    # -------------------------
    # Plot wheel
    # -------------------------
    axs[0].plot(x0, y0, c='k', label='wheel speed', linewidth=0.5)
    axs[0].set_ylabel('wheel speed')
    axs[0].axes.yaxis.set_visible(False)

    # -------------------------
    # Plot whisker ME
    # -------------------------
    axs[1].plot(x1, y1, c='k', label='whisker ME', linewidth=0.5)
    axs[1].set_ylabel('whisking')
    axs[1].axes.yaxis.set_visible(False)

    # -------------------------
    # Plot licks
    # -------------------------
    axs[2].plot(x2, y2, c='k', linewidth=0.5)
    axs[2].set_ylabel('licking')
    axs[2].axes.yaxis.set_visible(False)

    # -------------------------
    # Neural panel
    # -------------------------
    if alter:
        # Need spike sorting loaded even if cnew=False
        sl_spk = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        spikes, clusters, channels = sl_spk.load_spike_sorting()
        clusters = sl_spk.merge_clusters(spikes, clusters, channels)

        s_idx = find_nearest(spikes['times'], tstart)
        e_idx = find_nearest(spikes['times'], tend)

        times_s = spikes['times'][s_idx:e_idx]
        clus_s = spikes['clusters'][s_idx:e_idx]

        uclus = np.unique(clus_s)

        depthd = dict(zip(clusters['cluster_id'], clusters['depths']))
        uclus = np.array([c for c in uclus if c in depthd])

        if len(uclus) > 0:
            depthl = np.array([depthd[c] for c in uclus])
            uclus = uclus[np.argsort(depthl)]

        clus2row = {c: i for i, c in enumerate(uclus)}

        x_values = []
        y_values = []
        for c in uclus:
            ts = times_s[clus_s == c]
            x_values.extend(ts)
            y_values.extend([clus2row[c]] * len(ts))

        axs[3].scatter(x_values, y_values, marker='|', s=1, color='k')
        axs[3].set_ylabel('neuron')

    else:
        nneu, nobs = x3.shape
        axs[3].imshow(
            x3, aspect='auto', cmap='Greys',
            extent=[y3[0], y3[1], 0, nneu],
            vmin=0, vmax=2, origin='lower'
        )
        axs[3].set_ylabel('neuron')

        # optional colored side bars for cluster identity
        if cnew and cols_ is not None:
            xbar0 = y3[0] + 0.01 * (y3[1] - y3[0])
            xbar1 = y3[0] + 0.03 * (y3[1] - y3[0])
            for i, col in enumerate(cols_[:nneu]):
                axs[3].plot([xbar0, xbar1], [i, i], color=col, linewidth=1)

    # -------------------------
    # Trial annotations
    # -------------------------
    cols = {
        '0.8': [0.13850039, 0.41331206, 0.74052025],
        '0.2': [0.66080672, 0.21526712, 0.23069468],
        '0.5': 'g'
    }

    y0_text = np.nanpercentile(y0, 90) if len(y0) else 0.1

    for i in range(trial_range[0], trial_range[-1] + 1):
        st = trials['intervals'][i][0]
        en = trials['intervals'][i][1]
        pl = trials['probabilityLeft'][i]

        if np.isnan(trials['contrastLeft'][i]):
            side = 'r'
        else:
            side = 'l'

        choice_val = trials['choice'][i]
        choi = {1: 'l', -1: 'r'}.get(choice_val, 'n/a')
        ftype = trials['feedbackType'][i]

        s = f'Trial: {i}\nstim: {side}\nchoice: {choi}\ncorrect: {ftype}'
        axs[0].text(st, y0_text, s, va='top', fontsize=8)

        for k in range(nsubplots):
            # Uncomment if you want shaded block backgrounds
            # axs[k].axvspan(st, en, facecolor=cols[str(pl)], alpha=0.1)

            if not np.isnan(trials['stimOn_times'][i]):
                axs[k].axvline(
                    x=trials['stimOn_times'][i],
                    color='k', linestyle='--',
                    label='stimOn' if (i == trial_range[0] and k == 0) else None
                )

            if not np.isnan(trials['feedback_times'][i]):
                axs[k].axvline(
                    x=trials['feedback_times'][i],
                    color='k', linestyle=':',
                    label='feedback' if (i == trial_range[0] and k == 0) else None
                )

    # -------------------------
    # Cosmetics
    # -------------------------
    for ax in axs[:-1]:
        ax.get_xaxis().set_visible(False)

    axs[-1].set_xlabel('time [sec]')

    # If you really want everything visually off, this also hides labels.
    # Keeping axes on is usually more useful for debugging/inspection.
    # for ax in axs:
    #     ax.axis("off")

    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if len(by_label) > 0:
        axs[-1].legend(by_label.values(), by_label.keys(), ncol=4).set_draggable(True)

    plt.tight_layout()

    if cnew:
        np.save(pth_res / 'Q.npy', np.array(Q, dtype=object), allow_pickle=True)

    return fig, axs
        
        
if __name__ == "__main__":        

    '''
    Create panel of intro figure of the brain-wide-map paper
    illustrating time series and neural data for
    3 example trials
    '''
    eid = '15f742e1-1043-45c9-9504-f1e8a53c1744'
    bwm_data_series_fig()
