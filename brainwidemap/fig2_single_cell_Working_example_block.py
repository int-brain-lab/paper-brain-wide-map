#!/usr/bin/env python
# coding: utf-8
from pathlib import Path

import numpy as np
from one.api import ONE
from scipy.stats import rankdata

from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.task.closed_loop import generate_pseudo_blocks

from brainwidemap import load_good_units, load_trials_and_mask
from .fig2_util import TwoNmannWhitneyUshuf

# Specify a path to download the cluster and trials tables
local_path = Path.home().joinpath("bwm_examples")
local_path.mkdir(exist_ok=True)


def get_block(rate, c_L, c_R, block_label, choice_label):
    # nShuf=10000;
    nShuf = 5000

    num_neuron = len(rate[:, 0])

    p = np.zeros(num_neuron)

    for i_neuron in range(num_neuron):
        spike_count = rate[i_neuron, :]

        ############ left choice=1 ############
        con1 = np.logical_and(c_L > 0, choice_label == 1)
        con2 = np.logical_and(block_label == 0.8, con1)
        con3 = np.logical_and(block_label == 0.2, con1)
        index1 = np.argwhere(con2)
        index2 = np.argwhere(con3)

        x2 = spike_count[index1[:, 0]]
        y2 = spike_count[index2[:, 0]]

        nA2 = len(x2)
        nB2 = len(y2)

        numer2 = TwoNmannWhitneyUshuf(x2, y2, nShuf)

        ############ left choice=-1 ############
        con1 = np.logical_and(c_L > 0, choice_label == -1)
        con2 = np.logical_and(block_label == 0.8, con1)
        con3 = np.logical_and(block_label == 0.2, con1)
        index1 = np.argwhere(con2)
        index2 = np.argwhere(con3)

        x5 = spike_count[index1[:, 0]]
        y5 = spike_count[index2[:, 0]]

        nA5 = len(x5)
        nB5 = len(y5)

        numer5 = TwoNmannWhitneyUshuf(x5, y5, nShuf)

        ############ left, block=0.2, choice=1 ############
        con1 = np.logical_and(c_R > 0, choice_label == 1)
        con2 = np.logical_and(block_label == 0.8, con1)
        con3 = np.logical_and(block_label == 0.2, con1)
        index1 = np.argwhere(con2)
        index2 = np.argwhere(con3)

        x3 = spike_count[index1[:, 0]]
        y3 = spike_count[index2[:, 0]]

        nA3 = len(x3)
        nB3 = len(y3)

        numer3 = TwoNmannWhitneyUshuf(x3, y3, nShuf)

        ############ left, block=0.2, choice=-1 ############
        con1 = np.logical_and(c_R > 0, choice_label == -1)
        con2 = np.logical_and(block_label == 0.8, con1)
        con3 = np.logical_and(block_label == 0.2, con1)
        index1 = np.argwhere(con2)
        index2 = np.argwhere(con3)

        x6 = spike_count[index1[:, 0]]
        y6 = spike_count[index2[:, 0]]

        nA6 = len(x6)
        nB6 = len(y6)

        numer6 = TwoNmannWhitneyUshuf(x6, y6, nShuf)

        nTotal = numer2 + numer3 + numer5 + numer6

        dTotal = nA2 * nB2 + nA3 * nB3 + nA5 * nB5 + nA6 * nB6

        cp = nTotal / dTotal

        t = rankdata(cp)
        p[i_neuron] = t[0] / (1 + nShuf)

    return p


####################### pseudo session methods: ######################


def MW_test(x, y):
    nA1 = len(x)
    nB1 = len(y)
    ################### x>y #####################
    t2 = np.append(x, y, axis=0)
    t = rankdata(t2)

    t1 = t[range(nA1)]
    if nA1 == 1:
        numer1 = t1[0]
    else:
        numer1 = np.sum(t1)

    numer = numer1 - nA1 * (nA1 + 1) / 2

    ############### y>x ###########################
    t4 = np.append(y, x, axis=0)
    t5 = rankdata(t4)

    t3 = t5[range(nB1)]

    if nB1 == 1:
        numer3 = t3[0]
    else:
        numer3 = np.sum(t3)

    numer2 = numer3 - nB1 * (nB1 + 1) / 2

    ######################################################
    numer_final = np.minimum(numer, numer2)

    nTotal = numer_final
    dTotal = nA1 * nB1

    cp = nTotal / dTotal

    return cp


def Pseudo_block_test(spike_count, block, n_trial):
    nShuf = 5000
    num_unit = len(spike_count[:, 0])
    cp = np.zeros((num_unit, nShuf + 1))

    prob = np.zeros(num_unit)

    for i in range(nShuf):
        p_L = generate_pseudo_blocks(n_trial)

        for i_neuron in range(num_unit):
            x = spike_count[i_neuron, np.argwhere(p_L == 0.8)]
            y = spike_count[i_neuron, np.argwhere(p_L == 0.2)]
            cp[i_neuron, i + 1] = MW_test(x, y)

    for i_neuron in range(num_unit):
        dx = spike_count[i_neuron, np.argwhere(block == 0.8)]
        dy = spike_count[i_neuron, np.argwhere(block == 0.2)]
        cp[i_neuron, 0] = MW_test(dx, dy)

        t = rankdata(cp[i_neuron, :])
        prob[i_neuron] = t[0] / (1 + nShuf)

    return prob


def BWM_block_test(pid, eid, TimeWindow=np.array([-0.4, -0.1]), one=None):
    one = one or ONE()
    # load spike data
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

    ########## Method 1: pseudo-session #############

    p_1 = Pseudo_block_test(spike_rate, block, num_trial)

    ########## Method 2: condition-combined MW test #############

    p_2 = get_block(spike_rate, contrast_L, contrast_R, block, choice)

    return p_1, p_2, area_label, QC_cluster_id


if __name__ == "__main__":
    ### example session ###
    pid = "3675290c-8134-4598-b924-83edb7940269"
    eid = "15f742e1-1043-45c9-9504-f1e8a53c1744"  # probe00

    p_1, p_2, area_label, QC_cluster_id = BWM_block_test(pid, eid)
