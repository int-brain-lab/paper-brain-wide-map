import numpy as np
from scipy.stats import rankdata


def Time_TwoNmannWhitneyUshuf(x,y,bx,by,nShuf):
    
    nx=len(x)
    ny=len(y)


################### x>y #####################
    t1=np.zeros((nShuf+1,nx))
    
    t2=np.append(x,y,axis=0)
    t=rankdata(t2)
 
    
    t1[0,:]=t[range(nx)]
    


    block_list=np.intersect1d(bx, by)


    for i_Shuf in range(nShuf):
        Final_index=np.zeros(nx+ny)
        Final_index[:]=range(nx+ny)
    #### generate random permutation sequence for individual block ####        
        for i_block in range(len(block_list)):
    
            bx_index=np.argwhere(bx==block_list[i_block])
            by_index=np.argwhere(by==block_list[i_block])
            temp_index=np.append(bx_index,by_index+len(bx))
    
            z1=np.random.choice(len(bx_index)+len(by_index), size=(len(bx_index)+len(by_index)), replace=False)
            z=temp_index[z1]
            Final_index[temp_index]=z
        
        Final_index=Final_index.astype(int)        
        t1[i_Shuf+1,:]=t[Final_index[range(nx)]]    

    if nx==1:
        numer1=t1[:,0]
    else: 
        numer1=np.sum(t1,axis=1)
        
    numer=numer1-nx*(nx+1)/2   
    
    
################### y>x #####################
    t3=np.zeros((nShuf+1,ny))
    
    t4=np.append(y,x,axis=0)
    t5=rankdata(t4)
 
    
    t3[0,:]=t5[range(ny)]
    


    block_list=np.intersect1d(by, bx)


    for i_Shuf in range(nShuf):
        Final_index=np.zeros(nx+ny)
        Final_index[:]=range(nx+ny)
    #### generate random permutation sequence for individual block ####        
        for i_block in range(len(block_list)):
    
            bx_index=np.argwhere(bx==block_list[i_block])
            by_index=np.argwhere(by==block_list[i_block])
            temp_index=np.append(by_index,bx_index+len(by))
    
            z1=np.random.choice(len(bx_index)+len(by_index), size=(len(bx_index)+len(by_index)), replace=False)
            z=temp_index[z1]
            Final_index[temp_index]=z
        
        Final_index=Final_index.astype(int)       
        t3[i_Shuf+1,:]=t5[Final_index[range(ny)]]    

    if ny==1:
        numer2=t3[:,0]
    else: 
        numer2=np.sum(t3,axis=1)
        
    numer3=numer2-ny*(ny+1)/2  
    


    ######################################################
    numer_final=np.minimum(numer,numer3)
    
    return numer_final


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

