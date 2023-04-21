import numpy as np
from scipy.stats import rankdata
from one.api import ONE

from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.task.closed_loop import generate_pseudo_blocks
from brainbox.io.one import SpikeSortingLoader
from ibllib.atlas import AllenAtlas

from .fig2_util import get_block

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


def Pseudo_block_test(spike_count, block, n_trial, nShuf=5000):
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


def BWM_block_test(pid, eid, TimeWindow=np.array([-0.4, -0.1]), one=None, atlas=None):
    one = one or ONE()

    # Load spike data
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas or AllenAtlas())
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    # Load trial data
    collections = one.list_collections(eid, filename="_ibl_trials*")
    trials = one.load_object(eid, collection=collections[0], obj="trials")

    stim_on = trials.stimOn_times
    response = trials.response_times
    contrast_R = trials.contrastRight
    contrast_L = trials.contrastLeft
    choice = trials.choice
    block = trials.probabilityLeft
    first_move = trials.firstMovement_times

    nan_index = np.argwhere(np.isnan(first_move))
    first_move[nan_index[:, 0]] = response[nan_index[:, 0]]

    num_trial = len(stim_on)

    ########## Time bins %%%%%%%%%%%%%%%%%%%%

    # TimeWindow[0] = -0.2
    # TimeWindow[1] =  1
    # BinSize = 0.02

    # num_bin = np.floor((TimeWindow[1]-TimeWindow[0])/BinSize).astype(int)

    # 3 conditions R:[0.25 1], [0], L:[0.25 1]
    # firing_rate = np.zeros((num_neuron,num_bin,num_cond))

    # number of trials included
    # spike_rate = np.zeros((num_neuron, num_trial))

    ############ compute firing rate ###################

    T_1 = TimeWindow[0]
    T_2 = TimeWindow[1]

    raw_events = np.array([stim_on + T_1, stim_on + T_2]).T
    events = raw_events

    spike_count, cluster_id = get_spike_counts_in_bins(
        spikes["times"], spikes["clusters"], events
    )
    # firing_rate.shape=(num_neuron,num_bin,num_condition)
    # count_number[i_bin]=   np.nansum(spike_count,axis=1)

    spike_rate = spike_count / (T_2 - T_1)
    # num_trial_cond=len(events[:,0])
    # spike_count.shape(num_neuron,num_trial_cond)
    #  cluster_id.shape(num_neuron,1)

    # rate_1=(num_neuron,)
    # rate_1=np.nanmean(np.nanmean(firing_rate,axis=1),axis=1)
    # rate_1=(np.nanmean(firing_rate[:,:,0],axis=1)
    area_label_1 = clusters["atlas_id"][cluster_id]

    # only include units with firing rates > 1 Hz
    # rate_threshold = 1

    # rate_good = np.argwhere(rate_1>rate_threshold)[:,0]

    # only include units pass single unit QC criterion
    incl = clusters["amp_median"][cluster_id] > 50 / 1000000
    incl &= clusters["slidingRP_viol"][cluster_id] == 1
    incl &= clusters["noise_cutoff"][cluster_id] < 20

    included_units = cluster_id[incl]
    rate = spike_rate[included_units, :]
    area_label = area_label_1[included_units]

    ############ return cluster id ########################
    QC_cluster_id = cluster_id[included_units]

    ############ compute p-value for block coding ###################

    ########## Method 1: pseudo-session #############
    p_1 = Pseudo_block_test(rate, block, num_trial)

    ########## Method 2: condition-combined MW test #############
    p_2 = get_block(rate, contrast_L, contrast_R, block, choice)

    return p_1, p_2, area_label, QC_cluster_id


if __name__ == "__main__":
    # The example session
    pid = "3675290c-8134-4598-b924-83edb7940269"  # probe = "probe00"
    eid = "15f742e1-1043-45c9-9504-f1e8a53c1744"

    p_1, p_2, area_label, QC_cluster_id = BWM_block_test(pid, eid)
