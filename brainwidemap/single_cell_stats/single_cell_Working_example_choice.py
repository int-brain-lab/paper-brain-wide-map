#!/usr/bin/env python
# coding: utf-8
import numpy as np
from one.api import ONE
from scipy.stats import rankdata

from brainbox.population.decode import get_spike_counts_in_bins

from brainwidemap.bwm_loading import load_good_units, load_trials_and_mask
from brainwidemap.single_cell_stats.single_cell_util import Time_TwoNmannWhitneyUshuf

########### p-value for choice side############

def get_choice_time_shuffle(rate, c_L, c_R, block_label, choice_label, nShuf=3000):
    # nShuf=10000;
    # nShuf=5000;

    num_neuron = len(rate[:, 0])

    p = np.zeros(num_neuron)

    ######## get label of unique blocks #############

    s_block = np.zeros(len(block_label))

    for i in range(1, len(block_label)):
        s_block[i] = s_block[i - 1]
        if abs(block_label[i] - block_label[i - 1]) > 0:
            s_block[i] = s_block[i] + 1

    ##################################################

    for i_neuron in range(num_neuron):
        spike_count = rate[i_neuron, :]

        ############ block=0.8, stim=L ############
        con1 = np.logical_and(block_label == 0.8, c_L > 0)
        con2 = np.logical_and(choice_label == 1, con1)
        con3 = np.logical_and(choice_label == -1, con1)

        index1 = np.argwhere(con2)
        index2 = np.argwhere(con3)

        x2 = spike_count[index1[:, 0]]
        y2 = spike_count[index2[:, 0]]

        bx2 = s_block[index1[:, 0]]
        by2 = s_block[index2[:, 0]]

        nA2 = len(x2)
        nB2 = len(y2)

        numer2 = Time_TwoNmannWhitneyUshuf(x2, y2, bx2, by2, nShuf)

        ############ block=0.8, stim=R ############
        con1 = np.logical_and(block_label == 0.8, c_R > 0)
        con2 = np.logical_and(choice_label == 1, con1)
        con3 = np.logical_and(choice_label == -1, con1)
        index1 = np.argwhere(con2)
        index2 = np.argwhere(con3)

        x5 = spike_count[index1[:, 0]]
        y5 = spike_count[index2[:, 0]]

        bx5 = s_block[index1[:, 0]]
        by5 = s_block[index2[:, 0]]

        nA5 = len(x5)
        nB5 = len(y5)

        numer5 = Time_TwoNmannWhitneyUshuf(x5, y5, bx5, by5, nShuf)

        ############ block=0.2, stim=L ############
        con1 = np.logical_and(block_label == 0.2, c_L > 0)
        con2 = np.logical_and(choice_label == 1, con1)
        con3 = np.logical_and(choice_label == -1, con1)
        index1 = np.argwhere(con2)
        index2 = np.argwhere(con3)

        x3 = spike_count[index1[:, 0]]
        y3 = spike_count[index2[:, 0]]

        bx3 = s_block[index1[:, 0]]
        by3 = s_block[index2[:, 0]]

        nA3 = len(x3)
        nB3 = len(y3)

        numer3 = Time_TwoNmannWhitneyUshuf(x3, y3, bx3, by3, nShuf)

        ############ block=0.2, stim=R ############
        con1 = np.logical_and(block_label == 0.2, c_R > 0)
        con2 = np.logical_and(choice_label == 1, con1)
        con3 = np.logical_and(choice_label == -1, con1)
        index1 = np.argwhere(con2)
        index2 = np.argwhere(con3)

        x6 = spike_count[index1[:, 0]]
        y6 = spike_count[index2[:, 0]]

        bx6 = s_block[index1[:, 0]]
        by6 = s_block[index2[:, 0]]

        nA6 = len(x6)
        nB6 = len(y6)

        numer6 = Time_TwoNmannWhitneyUshuf(x6, y6, bx6, by6, nShuf)

        nTotal = numer2 + numer3 + numer5 + numer6

        dTotal = nA2 * nB2 + nA3 * nB3 + nA5 * nB5 + nA6 * nB6

        cp = nTotal / dTotal

        t = rankdata(cp)
        p[i_neuron] = t[0] / (1 + nShuf)

    return p


def BWM_choice_test(pid, eid, TimeWindow=np.array([-0.1, 0.0]), one=None):
    one = one or ONE()
    spikes, clusters = load_good_units(one, pid, compute_metrics=True)

    # load trial data
    trials, mask = load_trials_and_mask(
        one, eid, min_rt=0.08, max_rt=2.0, nan_exclude="default"
    )
    # select good trials
    trials = trials.loc[mask == True]

    stim_on = trials.stimOn_times.to_numpy()
    contrast_R = trials.contrastRight.to_numpy()
    contrast_L = trials.contrastLeft.to_numpy()
    choice = trials.choice.to_numpy()
    block = trials.probabilityLeft.to_numpy()

    num_neuron = len(np.unique(spikes["clusters"]))
    num_trial = len(stim_on)

    ############ compute firing rate ###################

    T_1 = TimeWindow[0]
    T_2 = TimeWindow[1]

    raw_events = np.array([stim_on + T_1, stim_on + T_2]).T
    events = raw_events

    spike_count, cluster_id = get_spike_counts_in_bins(spikes["times"], spikes["clusters"], events)
    spike_rate = spike_count / (T_2 - T_1)
    area_label = clusters["atlas_id"][cluster_id].to_numpy()

    ############ return cluster id ########################
    QC_cluster_id = clusters["cluster_id"][cluster_id].to_numpy()

    ############ compute p-value for block ###################

    ########## Pre-move, time_shuffle_test #############

    p_1 = get_choice_time_shuffle(spike_rate, contrast_L, contrast_R, block, choice, 3000)

    return p_1, area_label, QC_cluster_id


if __name__ == "__main__":
    # Example session
    pid = "3675290c-8134-4598-b924-83edb7940269"
    eid = "15f742e1-1043-45c9-9504-f1e8a53c1744"  # probe00

    p_1, area_label, QC_cluster_id = BWM_choice_test(pid, eid)
