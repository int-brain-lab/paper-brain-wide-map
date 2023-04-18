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
