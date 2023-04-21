import numpy as np
from scipy.stats import rankdata


def TwoNmannWhitneyUshuf(x, y, nShuf=10000):
    nA1 = len(x)
    nB1 = len(y)

    ################### x>y #####################
    t1 = np.zeros((nShuf + 1, nA1))

    t2 = np.append(x, y, axis=0)
    t = rankdata(t2)

    t1[0,:]=t[range(nA1)]
    for i in range(nShuf):
        z1=np.random.choice(nA1+nB1, size=nA1, replace=False)
        t1[i+1,:]=t[z1]

    if nA1==1:
        numer1=t1[:,0]
    else:
        numer1=np.sum(t1,axis=1)

    numer=numer1-nA1*(nA1+1)/2


    ############### y>x ###########################
    t3=np.zeros((nShuf+1,nB1))

    t4=np.append(y,x,axis=0)
    t5=rankdata(t4)

    t3[0,:]=t5[range(nB1)]
    for i in range(nShuf):
        z1=np.random.choice(nA1+nB1, size=nB1, replace=False)
        t3[i+1,:]=t5[z1]

    if nB1==1:
        numer3=t3[:,0]
    else:
        numer3=np.sum(t3,axis=1)

    numer2=numer3-nB1*(nB1+1)/2

    ######################################################
    numer_final=np.minimum(numer,numer2)

    return numer_final


def get_block(rate, c_L, c_R, block_label, choice_label, nShuf=5000):
    # p-value for block side
    num_neuron = rate.shape[0]
    p = np.zeros(num_neuron)

    for i_neuron in range(num_neuron):
        spike_count = rate[i_neuron, :]

        ############ left choice = 1 ############
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

        ############ left choice = -1 ############
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

        ############ left, block = 0.2, choice = 1 ############
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

        ############ left, block = 0.2, choice = -1 ############
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
